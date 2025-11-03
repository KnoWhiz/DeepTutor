"""Local PDF → RAG chatbot backed by Lemonade-hosted models.

This experimental helper covers the full flow requested in features_lab:

1. Discover PDF files under one or more directories supplied by the caller.
2. Convert every page into a clean text chunk using the existing tutor pipeline
   helpers so we stay aligned with the production chunking rules.
3. Generate page-level embeddings with the local embedding model exposed by
   Lemonade (`/api/v1`) and persist them to a FAISS index on disk.
4. On every user question, run similarity search over the cached embeddings,
   pack the top N page chunks into the prompt, and stream the local chat model's
   response back to the caller.

The public entry-point is :func:`stream_local_rag_answer`, which returns a
Python generator yielding the assistant's answer tokens as they arrive.

Example (single run) ::

    from pathlib import Path
    from pipeline.science.features_lab.Local_models_test.Local_models_test import (
        stream_local_rag_answer,
    )

    pdf_root = Path(\"./my-papers\")
    question = \"How does the paper describe superconducting qubits?\"

    for token in stream_local_rag_answer([pdf_root], question):
        print(token, end=\"\", flush=True)
    print()
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator, List, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# Ensure we can import from the main tutor pipeline package.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.science.pipeline.embeddings import extract_document_from_file  # noqa: E402
from pipeline.science.pipeline.utils import generate_file_id  # noqa: E402

logger = logging.getLogger("tutorpipeline.science.features_lab.local_models")


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = os.getenv("LEMONADE_BASE_URL", "http://localhost:8000/api/v1")
DEFAULT_API_KEY = os.getenv("LEMONADE_API_KEY", "lemonade")
DEFAULT_EMBED_MODEL = os.getenv("LEMONADE_EMBED_MODEL", "nomic-embed-text-v1-GGUF")
DEFAULT_CHAT_MODEL = os.getenv("LEMONADE_CHAT_MODEL", "DeepSeek-Qwen3-8B-GGUF")
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "tmp" / "local_rag_cache"
MAX_EMBED_CHARS = int(os.getenv("LEMONADE_EMBED_MAX_CHARS", "1800"))
DEFAULT_DEMO_PATHS = [
    PROJECT_ROOT
    / "pipeline"
    / "science"
    / "features_lab"
    / "(Benchmarks and evals, safety vs. capabilities, machine ethics) DecodingTrust A Comprehensive Assessment of Trustworthiness in GPT Models.pdf",
    PROJECT_ROOT
    / "pipeline"
    / "science"
    / "features_lab"
    / "(Half-wave Resonator - 2017 Gorman) Noise sensing and quantum simulation with trapped atomic ions.pdf",
]
DEFAULT_DEMO_QUESTION = "what is a resonator?"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _iter_batches(items: Sequence[str], batch_size: int) -> Iterator[List[str]]:
    """Yield sequential batches without copying the whole collection."""
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _truncate(text: str, max_chars: int = 1800) -> str:
    """Clamp long context chunks so prompts stay within the local model limits."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _discover_pdfs(paths: Sequence[str | Path]) -> List[Path]:
    """Expand directories and files into a unique, sorted list of PDFs."""
    resolved: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if path.is_dir():
            found = sorted(p for p in path.rglob("*.pdf") if p.is_file())
            if not found:
                logger.warning("No PDFs found under directory %s", path)
            resolved.extend(found)
        elif path.suffix.lower() == ".pdf":
            resolved.append(path)
        else:
            logger.warning("Skipping non-PDF file: %s", path)

    # De-duplicate while preserving sort order.
    seen: set[Path] = set()
    unique: List[Path] = []
    for pdf in sorted(resolved):
        if pdf not in seen:
            seen.add(pdf)
            unique.append(pdf)

    if not unique:
        raise ValueError("No PDF files discovered from the provided paths.")
    return unique


def _page_documents_from_pdf(pdf_path: Path) -> List[Document]:
    """Produce one LangChain document per PDF page using pipeline helpers."""
    raw_pages = extract_document_from_file(str(pdf_path))
    documents: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_EMBED_CHARS,
        chunk_overlap=min(200, MAX_EMBED_CHARS // 5),
        separators=["\n\n", "\n", " ", ""],
    )

    for page_index, page_doc in enumerate(raw_pages, start=1):
        page_text = getattr(page_doc, "page_content", "") or ""
        clean = page_text.strip()
        if not clean:
            continue
        clean = clean.replace("-\n", " ")
        clean = " ".join(clean.split())
        if not clean:
            continue
        if len(clean) <= MAX_EMBED_CHARS:
            chunk_texts = [clean]
        else:
            chunk_texts = splitter.split_text(clean)

        multi_chunk = len(chunk_texts) > 1
        for chunk_index, chunk_text in enumerate(chunk_texts):
            metadata = {
                "file_path": str(pdf_path),
                "file_name": pdf_path.name,
                "page": page_index,
                "source": f"{pdf_path.name}#page_{page_index}"
                + (f"_chunk_{chunk_index}" if multi_chunk else ""),
            }
            if multi_chunk:
                metadata["chunk"] = chunk_index
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata,
                )
            )

    if not documents:
        logger.warning("PDF produced no extractable pages: %s", pdf_path)
    else:
        logger.info("Loaded %s pages from %s", len(documents), pdf_path.name)
    return documents


def _collection_id(file_ids: Sequence[str]) -> str:
    """Stable hash for a set of PDFs to name the cache directory."""
    digest = hashlib.md5()
    for file_id in sorted(file_ids):
        digest.update(file_id.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class CollectionMetadata:
    """Persisted metadata describing a cached embedding collection."""

    pdfs: list[dict]
    embedding_model: str
    chunking: str
    created_at: float

    @staticmethod
    def load(path: Path) -> "CollectionMetadata | None":
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return CollectionMetadata(
            pdfs=data.get("pdfs", []),
            embedding_model=data.get("embedding_model", ""),
            chunking=data.get("chunking", "page"),
            created_at=data.get("created_at", 0.0),
        )

    def dump(self, path: Path) -> None:
        payload = {
            "pdfs": self.pdfs,
            "embedding_model": self.embedding_model,
            "chunking": self.chunking,
            "created_at": self.created_at,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class LocalOpenAIEmbeddings(Embeddings):
    """Minimal embeddings wrapper that talks to Lemonade's OpenAI-compatible API."""

    def __init__(self, client: OpenAI, model: str, batch_size: int = 8) -> None:
        self._client = client
        self.model = model
        self.batch_size = max(1, batch_size)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings: List[List[float]] = []
        for batch in _iter_batches(texts, self.batch_size):
            response = self._client.embeddings.create(model=self.model, input=batch)
            # Lemonade preserves request ordering, so we can extend directly.
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self._client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


class LocalRAGPipeline:
    """End-to-end PDF ingestion, retrieval, and response streaming."""

    def __init__(
        self,
        pdf_paths: Sequence[str | Path],
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        embedding_model: str = DEFAULT_EMBED_MODEL,
        chat_model: str = DEFAULT_CHAT_MODEL,
        cache_root: Path | str = DEFAULT_CACHE_ROOT,
        embeddings_batch_size: int = 8,
    ) -> None:
        self.pdf_paths = _discover_pdfs(pdf_paths)
        self.base_url = base_url
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.embeddings = LocalOpenAIEmbeddings(
            self.client, self.embedding_model, batch_size=embeddings_batch_size
        )
        self._vectorstore = self._ensure_vectorstore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream_answer(
        self,
        question: str,
        *,
        top_k: int = 10,
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """Yield the streaming response for *question* using cached PDFs."""
        if not question.strip():
            raise ValueError("Question must be a non-empty string.")

        ranked_docs = self._vectorstore.similarity_search_with_score(question, k=top_k)
        if not ranked_docs:
            raise RuntimeError("No relevant context found for the supplied PDFs.")

        context_sections: List[str] = []
        citations: List[str] = []
        for rank, (doc, score) in enumerate(ranked_docs, start=1):
            meta = doc.metadata or {}
            file_name = meta.get("file_name") or Path(meta.get("file_path", "")).name
            page_num = meta.get("page", "N/A")
            citations.append(f"[{rank}] {file_name} (page {page_num}, score={score:.3f})")
            context_sections.append(
                f"[{rank}] {file_name} — page {page_num}\n{_truncate(doc.page_content)}"
            )

        prompt_context = "\n\n".join(context_sections)
        context_hint = "\n".join(citations)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are DeepTutor running fully offline on local models. "
                    "Answer with clear, structured explanations that rely only on the "
                    "provided PDF context. Cite sources as [index] where index matches "
                    "the provided context list. If the answer is unknown, say so."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PDF context:\n{prompt_context}\n\n"
                    f"Context summary:\n{context_hint}\n\n"
                    f"Question: {question.strip()}\n\n"
                    "Respond with a helpful answer grounded in the context above."
                ),
            },
        ]

        stream = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            stream=True,
            temperature=temperature,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_vectorstore(self) -> FAISS:
        """Build or load the FAISS index for the current PDF set."""
        pdf_records = [
            {
                "path": str(pdf),
                "file_id": generate_file_id(pdf),
                "modified": int(pdf.stat().st_mtime),
            }
            for pdf in self.pdf_paths
        ]
        collection_id = _collection_id([record["file_id"] for record in pdf_records])
        collection_dir = self.cache_root / collection_id
        index_dir = collection_dir / "faiss_index"
        metadata_path = collection_dir / "metadata.json"

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        existing = CollectionMetadata.load(metadata_path)

        needs_rebuild = True
        if (
            existing
            and existing.embedding_model == self.embedding_model
            and existing.chunking == "page"
        ):
            existing_fids = sorted(item.get("file_id") for item in existing.pdfs)
            current_fids = sorted(record["file_id"] for record in pdf_records)
            if existing_fids == current_fids and index_dir.exists():
                needs_rebuild = False

        if needs_rebuild:
            logger.info("Building FAISS index for %s PDFs", len(pdf_records))
            documents: List[Document] = []
            for pdf in self.pdf_paths:
                documents.extend(_page_documents_from_pdf(pdf))

            if not documents:
                raise RuntimeError("No chunks generated from PDFs; cannot build index.")

            vectorstore = FAISS.from_documents(documents, self.embeddings)
            if index_dir.exists():
                shutil.rmtree(index_dir)
            index_dir.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(index_dir))

            CollectionMetadata(
                pdfs=pdf_records,
                embedding_model=self.embedding_model,
                chunking="page",
                created_at=time.time(),
            ).dump(metadata_path)

            logger.info("Saved new FAISS index at %s", index_dir)
            return vectorstore

        logger.info("Loading cached FAISS index from %s", index_dir)
        return FAISS.load_local(
            str(index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------

def stream_local_rag_answer(
    pdf_dirs: Sequence[str | Path],
    question: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    chat_model: str = DEFAULT_CHAT_MODEL,
    cache_root: Path | str = DEFAULT_CACHE_ROOT,
    top_k: int = 10,
    temperature: float = 0.2,
    embeddings_batch_size: int = 8,
) -> Generator[str, None, None]:
    """Convenience wrapper returning the streaming generator directly."""
    pipeline = LocalRAGPipeline(
        pdf_dirs,
        base_url=base_url,
        api_key=api_key,
        embedding_model=embedding_model,
        chat_model=chat_model,
        cache_root=cache_root,
        embeddings_batch_size=embeddings_batch_size,
    )
    return pipeline.stream_answer(
        question, top_k=top_k, temperature=temperature
    )


# ---------------------------------------------------------------------------
# CLI entry point for quick experiments
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local Lemonade-backed RAG chatbot over PDFs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Directories or PDF files to ingest. Defaults to the built-in demo PDFs when omitted.",
    )
    parser.add_argument(
        "-q",
        "--question",
        help="Question to ask the local chatbot. Defaults to the demo question when omitted.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of page chunks to send to the chat model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the local chat completion.",
    )
    parser.add_argument(
        "--cache-root",
        default=str(DEFAULT_CACHE_ROOT),
        help="Where to store cached FAISS indices.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Configure logging verbosity.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    pdf_inputs: List[str | Path] = list(args.paths) if args.paths else list(DEFAULT_DEMO_PATHS)
    if not args.paths:
        logger.info("No PDF paths provided; using demo files:\n%s", "\n".join(str(p) for p in pdf_inputs))

    provided_question = args.question.strip() if args.question else ""
    question = provided_question or DEFAULT_DEMO_QUESTION
    if not provided_question:
        logger.info("No question provided; using demo question: %s", question)

    pipeline = LocalRAGPipeline(
        pdf_inputs,
        cache_root=args.cache_root,
    )
    print("Streaming answer...\n")
    try:
        for token in pipeline.stream_answer(
            question,
            top_k=args.top_k,
            temperature=args.temperature,
        ):
            print(token, end="", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 1
    print("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
