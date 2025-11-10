import shutil
import time
from typing import AsyncGenerator, Dict, Iterable, Optional, Union

from pipeline.science.pipeline.cli_agent_response import (
    pdfs_to_markdown_workspace,
    stream_codex_answer,
)
from pipeline.science.pipeline.content_translator import translate_content
from pipeline.science.pipeline.get_response import generate_follow_up_questions
from pipeline.science.pipeline.session_manager import ChatSession
from pipeline.science.pipeline.utils import clean_translation_prefix, format_time_tracking

import logging

logger = logging.getLogger("tutorpipeline.science.tutor_agent_server_agent_basic")


StreamChunk = Union[str, bytes]


async def tutor_agent_server_agent_basic(
    chat_session: ChatSession,
    file_path_list: Iterable[str],
    user_input: Optional[str],
    time_tracking: Optional[Dict[str, float]] = None,
    deep_thinking: bool = True,
    stream: bool = False,
) -> AsyncGenerator[StreamChunk, None]:
    """
    Public entry point mirroring other mode implementations. Returns the
    tracking wrapper so the caller receives an async generator.
    """
    if time_tracking is None:
        time_tracking = {}

    return tutor_agent_server_agent_basic_streaming_tracking(
        chat_session=chat_session,
        file_path_list=file_path_list,
        user_input=user_input,
        time_tracking=time_tracking,
        deep_thinking=deep_thinking,
        stream=stream,
    )


async def tutor_agent_server_agent_basic_streaming_tracking(
    chat_session: ChatSession,
    file_path_list: Iterable[str],
    user_input: Optional[str],
    time_tracking: Optional[Dict[str, float]] = None,
    deep_thinking: bool = True,
    stream: bool = False,
) -> AsyncGenerator[StreamChunk, None]:
    """Accumulate Codex output while streaming so session state stays in sync."""
    if time_tracking is None:
        time_tracking = {}

    async for chunk in tutor_agent_server_agent_basic_streaming(
        chat_session=chat_session,
        file_path_list=file_path_list,
        user_input=user_input,
        time_tracking=time_tracking,
        deep_thinking=deep_thinking,
        stream=stream,
    ):
        yield chunk
        if isinstance(chunk, str):
            chat_session.current_message += chunk
        else:
            try:
                chat_session.current_message += chunk.decode("utf-8")
            except Exception:
                chat_session.current_message += str(chunk)


async def tutor_agent_server_agent_basic_streaming(
    chat_session: ChatSession,
    file_path_list: Iterable[str],
    user_input: Optional[str],
    time_tracking: Optional[Dict[str, float]] = None,
    deep_thinking: bool = True,
    stream: bool = False,
) -> AsyncGenerator[StreamChunk, None]:
    """
    Stream Codex CLI output using the helper defined in cli_agent_response.py.
    """
    del deep_thinking, stream  # Parameters reserved for parity with other modes

    if time_tracking is None:
        time_tracking = {}

    overall_start = time.time()
    yield "<thinking>"

    if not user_input:
        logger.info("SERVER_AGENT_BASIC received empty user input.")
        yield "</thinking>"
        yield (
            "<response>\n\n"
            "Hi, I'm DeepTutor running in Server Agent Basic mode. "
            "Please enter a question about the uploaded document so I can help.\n\n"
            "</response>"
        )
        yield "<appendix>\n\n</appendix>"
        time_tracking["total_time"] = time.time() - overall_start
        logger.info("Server Agent Basic time tracking:\n%s", format_time_tracking(time_tracking))
        return

    if not file_path_list:
        logger.warning("SERVER_AGENT_BASIC requires at least one document but none were provided.")
        yield "</thinking>"
        yield (
            "<response>\n\n"
            "⚠️ Server Agent Basic mode requires at least one PDF document. "
            "Upload a file and try again.\n\n"
            "</response>"
        )
        yield "<appendix>\n\n</appendix>"
        time_tracking["total_time"] = time.time() - overall_start
        logger.info("Server Agent Basic time tracking:\n%s", format_time_tracking(time_tracking))
        return

    codex_binary = shutil.which("codex")
    if not codex_binary:
        logger.error("Codex CLI binary not found on PATH.")
        yield "</thinking>"
        yield (
            "<response>\n\n"
            "⚠️ The Codex CLI (`codex`) is not available on this system. "
            "Install the CLI or ensure it is on the PATH, then retry Server Agent Basic mode.\n\n"
            "</response>"
        )
        yield "<appendix>\n\n</appendix>"
        time_tracking["total_time"] = time.time() - overall_start
        logger.info("Server Agent Basic time tracking:\n%s", format_time_tracking(time_tracking))
        return

    try:
        workspace_start = time.time()
        workspace_dir = pdfs_to_markdown_workspace(file_path_list)
        time_tracking["workspace_preparation"] = time.time() - workspace_start
        logger.info("Server Agent Basic workspace prepared at %s", workspace_dir)
    except Exception as exc:
        logger.exception("Failed to prepare Codex workspace: %s", exc)
        yield "</thinking>"
        yield (
            "<response>\n\n"
            "⚠️ I could not prepare the document workspace for Codex processing. "
            f"Details: {exc}\n\n"
            "</response>"
        )
        yield "<appendix>\n\n</appendix>"
        time_tracking["total_time"] = time.time() - overall_start
        logger.info("Server Agent Basic time tracking:\n%s", format_time_tracking(time_tracking))
        return

    thinking_closed = False
    saw_think_block = False
    response_started = False
    response_closed = False
    response_start = time.time()

    try:
        async for chunk in stream_codex_answer(workspace_dir, user_input):
            chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)

            if "<think>" in chunk_str and not saw_think_block:
                saw_think_block = True

            if "<response>" in chunk_str:
                response_started = True
                response_closed = "</response>" in chunk_str
            elif "</response>" in chunk_str:
                response_closed = True

            if "</think>" in chunk_str and saw_think_block and not thinking_closed:
                yield chunk_str
                yield "</thinking>"
                thinking_closed = True
            else:
                yield chunk_str
    except Exception as exc:
        logger.exception("Codex CLI streaming failed: %s", exc)
        if not thinking_closed:
            yield "</thinking>"
            thinking_closed = True
        error_message = (
            "⚠️ Codex CLI encountered an error while generating the answer. "
            f"Details: {exc}\n"
        )
        if response_started and not response_closed:
            yield error_message
            yield "</response>"
            response_closed = True
        else:
            yield (
                "<response>\n\n"
                f"{error_message}\n"
                "</response>"
            )
    else:
        if not thinking_closed:
            yield "</thinking>"
    finally:
        time_tracking["response_generation"] = time.time() - response_start

    follow_up_questions = []
    try:
        message_content = chat_session.current_message or ""
        if isinstance(message_content, list) and message_content:
            message_content = message_content[0]
        follow_up_questions = generate_follow_up_questions(
            message_content,
            [],
            user_input or "",
        )
        for idx, question in enumerate(follow_up_questions):
            translated = translate_content(
                content=question,
                target_lang=chat_session.current_language or "English",
                stream=False,
            )
            follow_up_questions[idx] = clean_translation_prefix(translated).strip()
    except Exception as exc:
        logger.warning("Failed to generate follow-up questions: %s", exc)

    yield "<appendix>\n\n"

    if follow_up_questions:
        yield "**💬 Suggested follow-up questions:**\n\n"
        for question in follow_up_questions:
            cleaned = question.strip()
            if not cleaned:
                continue
            yield "<followup_question>"
            yield cleaned
            yield "</followup_question>\n\n"

    yield (
        "**ℹ️ Server Agent Basic mode currently provides Codex-powered answers "
        "without DeepTutor source extraction.**\n\n"
        "</appendix>"
    )

    time_tracking["total_time"] = time.time() - overall_start
    logger.info("Server Agent Basic time tracking:\n%s", format_time_tracking(time_tracking))
