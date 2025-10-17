# -*- coding: utf-8 -*-
"""pipeline.science.features_lab.Codex_chatbot_test.codex_chatbot_test
-------------------------------------------------------------------------------
Utility script that demos how to call the **Codex CLI** from Python and expose
the output as a *streaming* generator – mirroring the behaviour of
`pipeline.science.pipeline.get_response` where the caller receives chunks that
can be forwarded directly to a web‑socket / Streamlit component.

The script intentionally keeps **all** Codex interaction behind a subprocess
so that:

1.  We do **not** import Codex as a library – we simply rely on the installed
    command‑line binary (`codex exec …`).
2.  Authentication is delegated to the CLI which, in turn, picks up the
    `AZURE_OPENAI_API_KEY_BACKUP` / `AZURE_OPENAI_ENDPOINT_BACKUP` values that
    we load from the local `.env` file.
3.  We override Codex’ default *env_key* so that it uses the **backup** key
    (per the user instruction – “do **not** use `AZURE_OPENAI_API_KEY`).

Running the file directly will:

1.  Ask Codex to summarise a sample PDF
   (``tmp/tutor_pipeline/input_files/2503.16408v1_RoboFactory_Exploring_Embodied_Agent_Collaboration_with_Compositional_Constraints.pdf``)
2.  Print the response incrementally to the terminal.

The *public* interface exposed for reuse by other modules is:

``get_codex_response(question: str, stream: bool = True) -> AsyncGenerator[str, None]``

It returns an **async** generator yielding text chunks, wrapped in
`<response> … </response>` tags – exactly the contract expected by the rest of
the tutoring pipeline.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, List

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


async def get_codex_response(question: str, stream: bool = True) -> AsyncGenerator[str, None]:
    """Call Codex with *question* and yield the response.

    The generator yields **strings** that can be forwarded directly to the
    client.  The very first chunk is ``"<response>\n\n"`` and the very last
    chunk is ``"\n\n</response>"`` so that the format matches
    *tutorpipeline* conventions.
    """

    if not stream:
        # Non‑streaming mode: capture the full output then yield once.
        import subprocess

        completed = subprocess.run(
            _build_codex_cmd(question), capture_output=True, text=True, check=True
        )
        yield "<response>\n\n"
        yield completed.stdout
        yield "\n\n</response>"
        return

    # --- Streaming path -----------------------------------------------------

    # We use *asyncio.create_subprocess_exec* so that we can await reads without
    # blocking the event‑loop. Codex prints tokens (or small chunks) separated
    # by **newlines**, so we read line‑by‑line and forward immediately.

    process = await asyncio.create_subprocess_exec(
        *_build_codex_cmd(question),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # Inherit the current env so the backup credentials are visible.
        env=os.environ.copy(),
    )

    if process.stdout is None:  # pragma: no cover – safety guard
        raise RuntimeError("Failed to open subprocess stdout pipe.")

    # Emit opening tag
    yield "<response>\n\n"

    try:
        # Iterate until EOF. We purposely use *read(1)* instead of readline to
        # propagate the output as soon as possible (Codex prints without
        # guaranteed newlines between tokens).
        while True:
            chunk = await process.stdout.read(1)
            if not chunk:
                break  # EOF
            yield chunk.decode(errors="ignore")
    finally:
        # Make sure the subprocess has terminated.
        try:
            await process.wait()
        except Exception:  # pragma: no cover – best‑effort cleanup
            pass

    # Emit closing tag
    yield "\n\n</response>"


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
