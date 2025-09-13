from __future__ import annotations
from typing import Dict, Generator, Optional, List
import json, os, pathlib
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_INSTRUCTIONS = """\
你是严谨的研究助理。优先基于提供的context回答；若context不足，再使用web search补全证据。
使用了web信息时，请在答案结尾给出来源要点（站点名/标题）。"""

def _extract_text_from_final_obj(final_obj) -> str:
    # 1) try convenience property (may be absent)
    try:
        txt = getattr(final_obj, "output_text", None)
        if txt:
            return txt
    except Exception:
        pass
    # 2) traverse canonical structure
    try:
        d = final_obj.to_dict() if hasattr(final_obj, "to_dict") else final_obj
        chunks: List[str] = []
        for out in d.get("output", []):
            for part in out.get("content", []):
                # model output is usually under type "output_text"
                if part.get("type") in ("output_text", "text"):
                    if isinstance(part.get("text"), str):
                        chunks.append(part["text"])
        return "".join(chunks)
    except Exception:
        return ""

def stream_chat_with_context(
    query: str,
    context: str,
    *,
    model: str = "gpt-5",
    reasoning_effort: str = "medium",      # minimal | low | medium | high
    verbosity: str = "medium",             # low | medium | high
    max_output_tokens: int = 4000,
    enable_web_search: bool = True,
    debug_tool_calls: bool = False,
) -> Generator[Dict, None, None]:
    """
    Yield events:
      - {"type":"text.delta","text":...}
      - {"type":"tool.delta","delta":...}      # if debug_tool_calls=True
      - {"type":"error","error":{...}}
      - {"type":"final","output_text":str,"raw":dict|None}
    """
    tools = [{"type": "web_search"}] if enable_web_search else []
    try:
        with client.responses.stream(
            model=model,
            reasoning={"effort": reasoning_effort},
            text={"verbosity": verbosity},          # do NOT send temperature to gpt-5
            instructions=SYSTEM_INSTRUCTIONS,
            tools=tools,
            tool_choice="auto" if tools else "none",
            max_output_tokens=max_output_tokens,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text",
                         "text": f"[Context]\n{context}\n\n[Query]\n{query}"}
                    ],
                }
            ],
        ) as stream:
            saw_text = False

            for event in stream:
                et = getattr(event, "type", None)

                if et == "response.output_text.delta":
                    saw_text = True
                    yield {"type": "text.delta", "text": event.delta}

                elif et == "response.function_call_arguments.delta":
                    if debug_tool_calls:
                        yield {"type": "tool.delta", "delta": event.delta}

                elif et == "response.refusal.delta":
                    saw_text = True
                    yield {"type": "text.delta", "text": event.delta}

                elif et == "response.error":
                    yield {"type": "error", "error": event.error}

                # (Optional) marker when model finishes emitting text
                elif et == "response.output_text.done":
                    pass

            final = stream.get_final_response()
            # robust fallback: extract text from the canonical structure
            output_text = _extract_text_from_final_obj(final)
            raw = final.to_dict() if hasattr(final, "to_dict") else None

            # If you got no text deltas and still no text, dump raw for inspection
            if not saw_text and not output_text and raw:
                dump_path = pathlib.Path("/tmp/last_final.json")
                try:
                    dump_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2))
                except Exception:
                    pass  # best-effort debugging

            yield {"type": "final", "output_text": output_text, "raw": raw}

    except Exception as e:
        yield {"type": "error", "error": {"message": str(e), "exc_type": type(e).__name__}}

def load_context():
    with open("/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/Responses_API_test/context.txt", "r") as f:
        return f.read()

if __name__ == "__main__":
    # For context, load from "/Users/bingran_you/Documents/GitHub_MacBook/DeepTutor/pipeline/science/features_lab/PDF_translate_test/test_document.pdf"
    ctx = load_context()
    print(ctx)
    buf = []
    for ev in stream_chat_with_context(
        "联网搜索Bingran You的信息",
        ctx,
        reasoning_effort="high",
        verbosity="medium",
        enable_web_search=True,
        debug_tool_calls=True,
    ):
        t = ev["type"]
        if t == "text.delta":
            buf.append(ev["text"])
            print(ev["text"], end="", flush=True)
        elif t == "tool.delta":
            print(f"\n[TOOL ARGS] {ev['delta']}", flush=True)
        elif t == "error":
            print(f"\n[ERROR] {ev['error']}", flush=True)
        elif t == "final":
            if not buf and ev.get("output_text"):
                print(ev["output_text"], end="", flush=True)
            print("\n---\n[FINAL META READY]")