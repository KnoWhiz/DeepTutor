# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator, List

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment ‑----------------------------------------------------------------
# We load the *.env* sitting next to this file so the subprocess inherits the
# required Azure credentials. The file must define the **backup** variables –
# we intentionally *do not* touch the primary ones.
# ---------------------------------------------------------------------------

DOTENV_PATH = Path(__file__).with_suffix(".env")
load_dotenv(DOTENV_PATH)  # no raise_if_missing – let missing vars fail later


# ---------------------------------------------------------------------------
# Helper that builds the *codex exec* command with the correct overrides.
# ---------------------------------------------------------------------------


def _build_codex_cmd(prompt: str, model: str = "o3-pro") -> List[str]:
    """Return the command list for ``subprocess``.

    We override the *env_key* that Codex should look at so that it pulls the
    token from ``AZURE_OPENAI_API_KEY_BACKUP`` (see `config.toml` notes).
    """

    return [
        "codex",
        "exec",
        "--json",
        "-m",
        model,
        # Override the env_key inside the azure provider section so Codex will
        # read AZURE_OPENAI_API_KEY_BACKUP instead of the default
        "-c",
        "model_providers.azure.env_key=\"AZURE_OPENAI_API_KEY_BACKUP\"",
        # The prompt itself
        prompt,
    ]


# ---------------------------------------------------------------------------
# Streaming wrapper – exposed API
# ---------------------------------------------------------------------------


async def _iter_codex_events(question: str, model: str) -> AsyncIterator[dict]:
    """Yield JSON events emitted by the Codex CLI for *question*."""

    process = await asyncio.create_subprocess_exec(
        *_build_codex_cmd(question, model),
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


async def _format_codex_events(question: str, model: str) -> AsyncGenerator[str, None]:
    """Stream formatted text chunks built from Codex JSON events."""

    think_open = True
    response_open = False

    yield "<think>\n"

    async for event in _iter_codex_events(question, model):
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


async def get_codex_response(question: str, stream: bool = True) -> AsyncGenerator[str, None]:
    """Call Codex with *question* and yield the response.

    The generator yields **strings** that can be forwarded directly to the
    client. Output is wrapped in ``<think>`` and ``<response>`` sections so
    callers can expose intermediate tool usage separately from the final
    answer.
    """

    if not stream:
        buffered: List[str] = []
        async for chunk in _format_codex_events(question, model="o3-pro"):
            buffered.append(chunk)
        yield "".join(buffered)
        return

    async for chunk in _format_codex_events(question, model="o3-pro"):
        yield chunk


# ---------------------------------------------------------------------------
# Demo / manual test
# ---------------------------------------------------------------------------


async def _demo() -> None:  # pragma: no cover – side‑effect entry‑point
    """Run a quick manual test when the file is executed directly."""

    test_pdf = (
        Path(__file__)
        .resolve()
        .parents[4]
        / "tmp/tutor_pipeline/input_files/"  # Navigate back to project root
        / "2503.16408v1_RoboFactory_Exploring_Embodied_Agent_Collaboration_with_Compositional_Constraints.pdf"
    )

    question = (
        "You are a tutor reading the paper at path:\n"
        f"{test_pdf}\n\n"
        "Please provide a concise summary of its main contributions."
    )

    # Stream to stdout – coloured output is handled by Codex itself.
    async for token in get_codex_response(question, stream=True):
        print(token, end="", flush=True)


if __name__ == "__main__":
    # The interactive runner – keep any event‑loop already running safe.
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:  # Graceful shutdown on Ctrl‑C
        print("\nInterrupted by user – exiting.")
