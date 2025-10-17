# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator, Iterable, List, Sequence, Tuple

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.science.pipeline.utils import generate_file_id

# ---------------------------------------------------------------------------
# Environment ‑----------------------------------------------------------------
# We load the *.env* sitting next to this file so the subprocess inherits the
# required Azure credentials. The file must define the **backup** variables –
# we intentionally *do not* touch the primary ones.
# ---------------------------------------------------------------------------

DOTENV_PATH = Path(__file__).with_suffix(".env")

load_dotenv(DOTENV_PATH)  # no raise_if_missing – let missing vars fail later

logger = logging.getLogger("tutorpipeline.features_lab.codex_chatbot_test")


# ---------------------------------------------------------------------------
# PDF → Markdown workspace helpers
# ---------------------------------------------------------------------------

WorkspaceEntry = Tuple[Path, Path]


def _codex_workspace_root() -> Path:
    """Return the root directory where Codex workspaces are stored."""
    workspace_root = PROJECT_ROOT / "tmp" / "codex_workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    return workspace_root


def _combined_hash(file_ids: Iterable[str]) -> str:
    """Build a stable hash representing a set of file identifiers."""
    digest = hashlib.md5()
    for file_id in sorted(file_ids):
        digest.update(file_id.encode("utf-8"))
    return digest.hexdigest()


def _safe_markdown_name(pdf_path: Path, file_id: str, existing: set[str]) -> str:
    """Generate a readable, collision-resistant markdown filename."""
    base = pdf_path.stem or file_id[:8]
    candidate = f"{base}.md"
    if candidate not in existing:
        return candidate

    suffix = file_id[:8]
    counter = 0
    while True:
        tail = f"_{suffix}" if counter == 0 else f"_{suffix}_{counter}"
        candidate = f"{base}{tail}.md"
        if candidate not in existing:
            return candidate
        counter += 1


def _pdf_to_markdown(pdf_path: Path) -> str:
    """Extract simple markdown text from *pdf_path* using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError(
            "PyPDF2 is required to convert PDF files to markdown. "
            "Install it with `pip install PyPDF2`."
        ) from exc

    reader = PdfReader(str(pdf_path))
    pages: List[str] = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.replace("\x00", "").strip()
        if text:
            pages.append(f"## Page {index}\n\n{text}")
        else:
            pages.append(f"## Page {index}\n\n*(No extractable text found on this page.)*")

    markdown = "\n\n".join(pages).strip()
    if not markdown:
        markdown = "*No extractable text found in the provided PDF.*"

    return markdown + "\n"


def prepare_codex_workspace(
    pdf_paths: Sequence[str | Path],
    *,
    workspace_root: Path | None = None,
) -> tuple[Path, List[WorkspaceEntry]]:
    """Convert *pdf_paths* to markdown files inside a hashed Codex workspace."""

    if not pdf_paths:
        raise ValueError("pdf_paths must contain at least one PDF path.")

    normalized: List[Path] = []
    for raw_path in pdf_paths:
        pdf_path = Path(raw_path).expanduser().resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        normalized.append(pdf_path)

    file_ids: List[str] = []
    for path in normalized:
        try:
            file_ids.append(generate_file_id(str(path)))
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to generate file id for {path}") from exc

    workspace_root = workspace_root or _codex_workspace_root()
    workspace_dir = workspace_root / _combined_hash(file_ids)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    existing_names: set[str] = set()
    entries: List[WorkspaceEntry] = []

    for pdf_path, file_id in zip(normalized, file_ids):
        markdown_name = _safe_markdown_name(pdf_path, file_id, existing_names)
        existing_names.add(markdown_name)

        markdown_path = workspace_dir / markdown_name
        if not markdown_path.exists():
            logger.info("Converting %s to markdown at %s", pdf_path, markdown_path)
            markdown_text = _pdf_to_markdown(pdf_path)
            markdown_path.write_text(markdown_text, encoding="utf-8")
        else:
            logger.debug("Markdown already exists for %s at %s", pdf_path, markdown_path)

        entries.append((pdf_path, markdown_path))

    manifest = {
        "workspace_id": workspace_dir.name,
        "source_files": [
            {
                "pdf": str(pdf_path),
                "markdown": markdown_path.name,
                "file_id": file_id,
            }
            for (pdf_path, markdown_path), file_id in zip(entries, file_ids)
        ],
    }

    manifest_path = workspace_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Codex workspace ready at %s", workspace_dir)
    return workspace_dir, entries


# ---------------------------------------------------------------------------
# Helper that builds the *codex exec* command with the correct overrides.
# ---------------------------------------------------------------------------


def _build_codex_cmd(prompt: str, model: str = "o3-pro", workdir: Path | None = None) -> List[str]:
    """Return the command list for ``subprocess``.

    We override the *env_key* that Codex should look at so that it pulls the
    token from ``AZURE_OPENAI_API_KEY_BACKUP`` (see `config.toml` notes).
    """

    cmd = [
        "codex",
        "exec",
        "--json",
        "-m",
        model,
        # Override the env_key inside the azure provider section so Codex will
        # read AZURE_OPENAI_API_KEY_BACKUP instead of the default
        "-c",
        "model_providers.azure.env_key=\"AZURE_OPENAI_API_KEY_BACKUP\"",
    ]
    if workdir is not None:
        cmd.extend(["-C", str(workdir)])

    # The prompt itself must be last
    cmd.append(prompt)
    return cmd


# ---------------------------------------------------------------------------
# Streaming wrapper – exposed API
# ---------------------------------------------------------------------------


async def _iter_codex_events(
    question: str,
    model: str,
    *,
    workdir: Path | None = None,
) -> AsyncIterator[dict]:
    """Yield JSON events emitted by the Codex CLI for *question*."""

    process = await asyncio.create_subprocess_exec(
        *_build_codex_cmd(question, model, workdir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=os.environ.copy(),
    )

    if process.stdout is None:  # pragma: no cover – defensive guard
        raise RuntimeError("Failed to open subprocess stdout pipe.")

    try:
        while True:
            raw_line = await process.stdout.readline()
            if not raw_line:
                break

            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # Ignore malformed lines – Codex may interleave notices.
    finally:
        try:
            await process.wait()
        except Exception:  # pragma: no cover – best-effort cleanup
            pass


async def _format_codex_events(
    question: str,
    model: str,
    *,
    workdir: Path | None = None,
) -> AsyncGenerator[str, None]:
    """Stream formatted text chunks built from Codex JSON events."""

    think_open = True
    response_open = False

    yield "<think>\n"

    async for event in _iter_codex_events(question, model, workdir=workdir):
        event_type = event.get("type")

        if not event_type:
            continue

        if event_type.startswith("item."):
            item = event.get("item", {})
            item_type = item.get("type")

            if event_type == "item.started" and item_type == "command_execution":
                command = item.get("command")
                if command:
                    yield f"exec\n{command}\n"
                continue

            if event_type == "item.delta":
                delta = event.get("delta", {})
                text_fragment = (
                    delta.get("text")
                    or delta.get("aggregated_output")
                    or delta.get("output")
                )

                if not text_fragment:
                    for value in delta.values():
                        if isinstance(value, str):
                            text_fragment = value
                            break

                if not text_fragment:
                    continue

                if item_type == "agent_message":
                    if think_open:
                        yield "</think>\n"
                        think_open = False
                    if not response_open:
                        yield "<response>\n"
                        response_open = True

                if not text_fragment.endswith("\n"):
                    text_fragment += "\n"
                yield text_fragment
                continue

            if event_type == "item.completed":
                if item_type == "command_execution":
                    output = item.get("aggregated_output") or ""
                    if output and not output.endswith("\n"):
                        output += "\n"
                    if output:
                        yield output
                    continue

                if item_type == "reasoning":
                    text = item.get("text") or ""
                    if text and not text.endswith("\n"):
                        text += "\n"
                    if text:
                        yield text
                    continue

                if item_type == "agent_message":
                    if think_open:
                        yield "</think>\n"
                        think_open = False
                    if not response_open:
                        yield "<response>\n"
                        response_open = True
                    text = item.get("text") or ""
                    yield text
                    if text and not text.endswith("\n"):
                        yield "\n"
                    continue

    if think_open:
        yield "</think>\n"

    if response_open:
        yield "</response>"
    else:
        yield "<response>\n</response>"


async def get_codex_response(
    question: str,
    *,
    stream: bool = True,
    pdf_paths: Sequence[str | Path] | None = None,
    workdir: Path | None = None,
) -> AsyncGenerator[str, None]:
    """Call Codex with *question* and yield the response.

    The generator yields **strings** that can be forwarded directly to the
    client. Output is wrapped in ``<think>`` and ``<response>`` sections so
    callers can expose intermediate tool usage separately from the final
    answer.
    """

    if pdf_paths and workdir:
        raise ValueError("Provide either pdf_paths or workdir, not both.")

    resolved_workdir = workdir
    if pdf_paths:
        resolved_workdir, _ = prepare_codex_workspace(pdf_paths)

    if not stream:
        buffered: List[str] = []
        async for chunk in _format_codex_events(
            question,
            model="o3-pro",
            workdir=resolved_workdir,
        ):
            buffered.append(chunk)
        yield "".join(buffered)
        return

    async for chunk in _format_codex_events(
        question,
        model="o3-pro",
        workdir=resolved_workdir,
    ):
        yield chunk


# ---------------------------------------------------------------------------
# Demo / manual test
# ---------------------------------------------------------------------------


async def _demo() -> None:  # pragma: no cover – side‑effect entry‑point
    """Run a quick manual test when the file is executed directly."""

    test_pdfs = [
        "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/tmp/tutor_pipeline/input_files/e54b39fed418d7d27ae2761e7ca2d459.pdf"
    ]

    workspace_dir, entries = prepare_codex_workspace(test_pdfs)

    markdown_overview = "\n".join(
        f"- {md_path.name} (source: {pdf_path.name})" for pdf_path, md_path in entries
    )

    question = (
        "You are a tutor reviewing the markdown exports found in the current working directory. "
        "Each markdown file mirrors a source PDF that has already been converted for you:\n"
        f"{markdown_overview}\n\n"
        "Please provide a concise summary of the main contributions."
    )

    # Stream to stdout – coloured output is handled by Codex itself.
    async for token in get_codex_response(
        question,
        stream=True,
        workdir=workspace_dir,
    ):
        print(token, end="", flush=True)


if __name__ == "__main__":
    # The interactive runner – keep any event‑loop already running safe.
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:  # Graceful shutdown on Ctrl‑C
        print("\nInterrupted by user – exiting.")
