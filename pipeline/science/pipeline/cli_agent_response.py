"""codex_chatbot_refactored.py
================================

Minimal, self‑contained helpers to:

1.  **`pdfs_to_markdown_workspace`** – Convert one or more PDF files to plain
    markdown and store them in a deterministic *Codex* workspace directory that
    is named after the combined MD5 hash of the source files. The function
    returns the absolute path to the workspace.

2.  **`stream_codex_answer`** – Given a *workspace* directory (produced by the
    first helper) and a *question*, call the `codex` CLI and yield a streaming
    response formatted as

        ``<think>…</think><response>…</response>``.

These two top‑level functions are the only public API exposed by this module, as
requested. Any additional utilities are defined *inside* the functions to
avoid leaking extra symbols.

Notes
-----
* The implementation mirrors the behaviour of the original
  `codex_chatbot_test.py` experiment while dramatically reducing surface area.
* The Azure OpenAI credential is read from the backup environment variable
  `AZURE_OPENAI_API_KEY_BACKUP` on each invocation so that callers do not need
  to manipulate their primary key settings.
* The module relies on the external *PyPDF2* package for text extraction and
  on the *Codex* CLI being available on the `$PATH`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import logging
import subprocess
import sys
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator, Iterable, List, Sequence, Set

# Project‑local helper for reproducible file hashing
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    # `generate_file_id` lives in the main tutor pipeline utils and performs a
    # robust MD5 hash on the raw bytes of a file.
    from pipeline.science.pipeline.utils import generate_file_id
except Exception:  # pragma: no cover – fallback if dependency graph changes
    def generate_file_id(file_path: str | Path) -> str:  # type: ignore
        """Light‑weight MD5 helper used as a graceful fallback."""

        path = Path(file_path)
        digest = hashlib.md5()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()


# ---------------------------------------------------------------------------
# Public API – 1) PDF → Markdown workspace
# ---------------------------------------------------------------------------


def pdfs_to_markdown_workspace(pdf_paths: Sequence[str | Path]) -> Path:
    """Convert *pdf_paths* to markdown inside a hashed Codex workspace.

    Parameters
    ----------
    pdf_paths:
        A sequence of absolute or relative paths pointing to PDF documents.

    Returns
    -------
    Path
        Absolute path of the workspace directory that now contains the newly
        generated markdown files.
    """

    if not pdf_paths:
        raise ValueError("pdf_paths must contain at least one path.")

    # 1. Normalise & validate paths ------------------------------------------------
    resolved_pdfs: List[Path] = []
    for raw in pdf_paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        resolved_pdfs.append(path)

    # 2. Compute stable combined hash ---------------------------------------------
    file_ids: List[str] = [generate_file_id(p) for p in resolved_pdfs]

    digest = hashlib.md5()
    for fid in sorted(file_ids):
        digest.update(fid.encode())
    workspace_id = digest.hexdigest()

    # 3. Create workspace directory ------------------------------------------------
    workspace_root = PROJECT_ROOT / "tmp" / "codex_workspaces"
    workspace_dir = workspace_root / workspace_id
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # 4. Convert each PDF ----------------------------------------------------------
    try:
        from PyPDF2 import PdfReader  # Lazy import; heavy dependency
    except ImportError as exc:  # pragma: no cover – explicit for clarity
        raise RuntimeError(
            "PyPDF2 is required for PDF -> markdown conversion. "
            "Install with `pip install PyPDF2`."
        ) from exc

    existing_names: set[str] = set()

    def _safe_name(pdf: Path, fid: str) -> str:
        base = pdf.stem or fid[:8]
        candidate = f"{base}.md"
        if candidate not in existing_names:
            return candidate
        # Collision handling
        suffix = fid[:8]
        counter = 0
        while True:
            tail = f"_{suffix}" if counter == 0 else f"_{suffix}_{counter}"
            candidate = f"{base}{tail}.md"
            if candidate not in existing_names:
                return candidate
            counter += 1

    for pdf_path, fid in zip(resolved_pdfs, file_ids):
        md_name = _safe_name(pdf_path, fid)
        existing_names.add(md_name)
        md_path = workspace_dir / md_name

        if md_path.exists():
            # File already converted – skip work to keep the function idempotent
            continue

        # Extract raw text per page in a minimal markdown structure
        reader = PdfReader(str(pdf_path))
        sections: List[str] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\x00", "").strip()
            if text:
                sections.append(f"## Page {idx}\n\n{text}")
            else:
                sections.append(f"## Page {idx}\n\n*(No extractable text found on this page.)*")

        md_content = "\n\n".join(sections).strip() or "*No extractable text found in the provided PDF.*"
        md_path.write_text(md_content + "\n", encoding="utf-8")

    return workspace_dir


# ---------------------------------------------------------------------------
# Public API – 2) Codex Q&A
# ---------------------------------------------------------------------------

logger = logging.getLogger("tutorpipeline.science.cli_agent_response")


async def stream_codex_answer(
    workspace_dir: str | Path,
    question: str,
    *,
    model: str = "gpt-5-codex",
) -> AsyncGenerator[str, None]:
    """Yield a streaming Codex answer for *question* over *workspace_dir*.

    The generator emits incremental chunks, first surrounded by a ``<think>``
    block (agent reasoning/tool calls) followed by a ``<response>`` block
    (answer). Example consumer::

        async for chunk in stream_codex_answer("/path/to/ws", "What is ...?"):
            print(chunk, end="")
    """

    workspace = Path(workspace_dir).expanduser().resolve()
    if not workspace.exists() or not workspace.is_dir():
        raise FileNotFoundError(f"Workspace directory not found: {workspace}")

    # ---------------------------------------------------------------------
    # Helper: build CLI command
    # ---------------------------------------------------------------------
    def build_cmd(prompt: str) -> List[str]:
        return [
            "codex",
            "exec",
            "--json",
            "--skip-git-repo-check",
            "-m",
            model,
            # Ensure Codex reads from the backup Azure key
            "-c",
            "model_providers.azure.env_key=\"AZURE_OPENAI_API_KEY_BACKUP\"",
            "-C",
            str(workspace),
            prompt,
        ]

    # ------------------------------------------------------------------
    # Spawn Codex CLI as an async subprocess and stream JSON events
    # ------------------------------------------------------------------
    raw_logs: List[str] = []
    seen_event = False
    max_log_lines = 20

    async def iter_events() -> AsyncIterator[dict]:
        nonlocal seen_event, raw_logs
        proc = await asyncio.create_subprocess_exec(
            *build_cmd(question),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=os.environ.copy(),
        )

        assert proc.stdout is not None  # mypy: ignore[assert]

        async for raw in proc.stdout:  # type: ignore[attr-defined]
            line = raw.decode(errors="ignore")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # Codex sometimes emits non‑JSON logs; skip them
                stripped = line.strip()
                if stripped:
                    raw_logs.append(stripped)
                    if len(raw_logs) > max_log_lines:
                        raw_logs = raw_logs[-max_log_lines:]
                    logger.debug("Codex CLI raw stdout: %s", stripped)
                continue
            seen_event = True
            yield payload

        returncode = await proc.wait()
        if returncode != 0:
            tail = "\n".join(raw_logs[-5:])
            raise RuntimeError(
                f"Codex CLI exited with code {returncode}. "
                f"Last output:\n{tail or 'No additional output captured.'}"
            )
        if not seen_event:
            tail = "\n".join(raw_logs[-5:])
            raise RuntimeError(
                "Codex CLI produced no JSON events. "
                f"Last output:\n{tail or 'No additional output captured.'}"
            )

    # ------------------------------------------------------------------
    # Transform events -> contiguous text stream with tags
    # ------------------------------------------------------------------
    async def format_events() -> AsyncGenerator[str, None]:
        think_open = True
        response_open = False
        produced_text = False
        event_types: Set[str] = set()
        yield "<think>\n"

        async for event in iter_events():
            etype = event.get("type")
            if not etype:
                continue
            event_types.add(etype)

            if etype.startswith("item."):
                item = event.get("item", {})
                itype = item.get("type")

                # Command execution start – show executed command
                if etype == "item.started" and itype == "command_execution":
                    cmd = item.get("command")
                    if cmd:
                        yield f"exec\n{cmd}\n"
                    continue

                # Incremental delta updates for agent_message / reasoning / etc.
                if etype == "item.delta":
                    if itype == "command_execution":
                        continue

                    delta = event.get("delta", {})
                    fragment = (
                        delta.get("text")
                        or delta.get("aggregated_output")
                        or delta.get("output")
                        or next((v for v in delta.values() if isinstance(v, str)), "")
                    )
                    if not fragment:
                        continue

                    if itype == "agent_message":
                        if think_open:
                            yield "</think>\n"
                            think_open = False
                        if not response_open:
                            yield "<response>\n"
                            response_open = True

                    if not fragment.endswith("\n"):
                        fragment += "\n"
                    if fragment.strip():
                        produced_text = True
                    yield fragment
                    continue

                # Completed items – flush any remaining buffered output
                if etype == "item.completed":
                    # Skip full command_execution outputs for the same reason
                    # we ignore their deltas above – they often include the
                    # entire file content which would bloat the stream.
                    if itype == "command_execution":
                        continue

                    if itype == "reasoning":
                        text = item.get("text") or ""
                        if text:
                            if not text.endswith("\n"):
                                text += "\n"
                            yield text
                        continue

                    if itype == "agent_message":
                        if think_open:
                            yield "</think>\n"
                            think_open = False
                        if not response_open:
                            yield "<response>\n"
                            response_open = True
                        text = item.get("text") or ""
                        if text:
                            produced_text = produced_text or bool(text.strip())
                            yield text
                            if not text.endswith("\n"):
                                yield "\n"
                        continue

                    # For unhandled item types, attempt to forward any text content
                    fallback_text = item.get("text")
                    if fallback_text:
                        if think_open:
                            yield "</think>\n"
                            think_open = False
                        if not response_open:
                            yield "<response>\n"
                            response_open = True
                        produced_text = produced_text or bool(fallback_text.strip())
                        yield fallback_text
                        if not fallback_text.endswith("\n"):
                            yield "\n"
                    continue

            # Handle non-item events that still carry text payloads
            if etype.endswith(".delta"):
                delta = event.get("delta", {})
                fragment = (
                    delta.get("text")
                    or delta.get("output")
                    or next((v for v in delta.values() if isinstance(v, str)), "")
                )
                if not fragment:
                    continue
                if think_open:
                    yield "</think>\n"
                    think_open = False
                if not response_open:
                    yield "<response>\n"
                    response_open = True
                if not fragment.endswith("\n"):
                    fragment += "\n"
                if fragment.strip():
                    produced_text = True
                yield fragment
                continue

            if etype.endswith(".completed"):
                text = event.get("text") or ""
                if not text:
                    continue
                if think_open:
                    yield "</think>\n"
                    think_open = False
                if not response_open:
                    yield "<response>\n"
                    response_open = True
                produced_text = produced_text or bool(text.strip())
                yield text
                if not text.endswith("\n"):
                    yield "\n"
                continue

        if think_open:
            yield "</think>\n"

        if response_open:
            if not produced_text:
                tail = "\n".join(raw_logs[-5:])
                raise RuntimeError(
                    "Codex CLI completed without emitting assistant content. "
                    f"Event types observed: {sorted(event_types)}. "
                    f"Last output:\n{tail or 'No additional output captured.'}"
                )
            yield "</response>"
        else:
            if produced_text:
                yield "<response>\n</response>"
            else:
                tail = "\n".join(raw_logs[-5:])
                raise RuntimeError(
                    "Codex CLI did not emit any response text. "
                    f"Event types observed: {sorted(event_types)}. "
                    f"Last output:\n{tail or 'No additional output captured.'}"
                )

    # Finally, stream the formatted output to the caller
    async for chunk in format_events():
        yield chunk


# ---------------------------------------------------------------------------
# CLI helper for quick manual checks ----------------------------------------
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover – developer convenience entry‑point
    """Simple sanity‑check using a hard‑coded PDF and question.

    This helper mirrors the request from the IDE context so that developers can
    run the file directly to validate end‑to‑end behaviour without crafting
    bespoke scripts each time::

        python -m pipeline.science.features_lab.Codex_chatbot_test.codex_chatbot_refactored
    """

    # 1. Prepare workspace -----------------------------------------------------
    pdf_path_1 = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/(Benchmarks and evals, safety vs. capabilities, machine ethics) DecodingTrust A Comprehensive Assessment of Trustworthiness in GPT Models.pdf"
    pdf_path_2 = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/(Benchmarks and evals, safety vs. capabilities, machine ethics) DecodingTrust A Comprehensive Assessment of Trustworthiness in GPT Models.pdf"
    pdf_path_3 = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/(Benchmarks and evals, safety vs. capabilities, machine ethics) DecodingTrust A Comprehensive Assessment of Trustworthiness in GPT Models.pdf"
    # pdf_path_2 = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/(Benchmarks and evals, safety vs. capabilities, machine ethics) Do the Rewards Justify the Means Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark.pdf"
    # pdf_path_3 = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/(Chichi Thesis - Toroidal Half-Wave (λ2) Resonator - 2021 Da An) Electric-field noise scaling and wire-mediated ion-ion energy exchange in a novel elevator surface trap.pdf"
    pdf_path_4 = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/(Half-wave Resonator - 2017 Gorman) Noise sensing and quantum simulation with trapped atomic ions.pdf"
    pdf_path_5 = "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/(Helical Resonator - 2012 Sussex) On the application of radio frequency voltages to ion traps via helical resonators.pdf"

    workspace = pdfs_to_markdown_workspace([pdf_path_1, pdf_path_2, pdf_path_3, pdf_path_4, pdf_path_5])
    
    # 2. Ask Codex -------------------------------------------------------------
    # question = "review all the papers and give me a comparesion table"
    question = "review and list all the files in this folder"

    async def _runner() -> None:
        async for token in stream_codex_answer(workspace, question):
            print(token, end="", flush=True)

    try:
        asyncio.run(_runner())
    except KeyboardInterrupt:
        print("\nInterrupted by user – exiting.")


if __name__ == "__main__":
    main()
