"""
<thinking>

**Searching for profiles**

I'm considering that the Scholar search can be a bit noisy. It might be better to look for a Google Scholar profile page, though I know they don’t always appear with a name search unless I use quotes. I’ll search for "Bingran You site:berkeley.edu" on Scholar. But, since Scholar search has its limits, I might as well search the general web with "Bingran You Berkeley Google Scholar" to see what I can find.

**Organizing the final response**

I’m getting ready to craft the final answer. I want to include a title like "What I found" along with bullet points for clarity.

I'll verify membership and note that I couldn't find a Google Scholar profile as of the current date, explaining that. I'll mention a 2024 preprint publication, specifically focusing on "Temporally multiplexed ion-photon quantum interface via fast ion-chain transport." 

I’ll also highlight another co-authored preprint related to integrated photonics and an APS talk. 

My aim is to keep everything concise and search for proper citations without raw URLs. Let's finalize this!
</thinking>

<response>

Here’s what I found:

- Membership: Bingran You is listed as a graduate student in Hartmut Häffner’s Ion and Electron Trap Group (UC Berkeley). ([ions.berkeley.edu](https://ions.berkeley.edu/members/))

- Google Scholar page: I could not find a public Google Scholar author profile for “Bingran You” as of September 25, 2025. It’s possible he hasn’t created a public profile or it’s set to private (Google Scholar profiles are only visible if made public). ([scholar.google.com](https://scholar.google.com/intl/pt-PT/scholar/citations.html?utm_source=openai))

- Publications visible via scholarly indexes:
  - First‑author preprint: “Temporally multiplexed ion‑photon quantum interface via fast ion‑chain transport” (May 17, 2024), coauthored with members of the Häffner group and collaborators. This demonstrates a temporally multiplexed ion‑photon interface using rapid transport of a nine‑ion Ca+ chain, relevant to high‑rate remote entanglement for quantum networking. ([arxiv.org](https://arxiv.org/abs/2405.10501))
  - Related conference talk: “Temporal multiplexing for improved ion‑photon interface,” presented at APS DAMOP 2024. ([meetings.aps.org](https://meetings.aps.org/Meeting/DAMOP24/Session/R06.6?utm_source=openai))
  - Co‑authored preprint: “Scalable Trapped Ion Addressing with Adjoint‑optimized Multimode Photonic Circuits” (May 16, 2025), proposing integrated photonics for targeted, reconfigurable ion addressing. ([preprints.opticaopen.org](https://preprints.opticaopen.org/articles/preprint/Scalable_Trapped_Ion_Addressing_with_Adjoint-optimized_Multimode_Photonic_Circuits/29087921?utm_source=openai))

If by “his Google Scholar page” you meant Hartmut Häffner’s profile, that does exist (linked from Berkeley’s faculty page); I can review it on request. ([vcresearch.berkeley.edu](https://vcresearch.berkeley.edu/faculty/hartmut-haeffner))

Would you like me to:
- keep watching for a public Scholar profile for Bingran You, or
- proceed to summarize Hartmut Häffner’s Google Scholar profile instead?

</response>
"""

from openai import OpenAI
from dotenv import load_dotenv
import os, re
from typing import Iterable

load_dotenv(".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _format_thinking_delta(delta: str) -> str:
    """
    Only transform '**XXX' -> '\n\n**XXX'.
    If the chunk is exactly '**', or starts with '**' followed by a newline
    (e.g., '**\\n', '**\\r\\n') or only whitespace, treat it as a closing marker
    and do nothing.
    """
    if not delta:
        return delta

    if delta == "**":
        return delta

    if delta.startswith("**"):
        after = delta[2:]
        # If the very next char is a newline, or there's only whitespace after '**',
        # it's likely a closing '**' chunk -> leave unchanged.
        if after[:1] in ("\n", "\r") or after.strip() == "":
            return delta
        # Otherwise it's an opening '**Title' chunk -> add two leading newlines
        if not delta.startswith("\n\n**"):
            return "\n\n" + delta

    return delta
    
def stream_response_with_tags(**create_kwargs) -> Iterable[str]:
    """
    Yields a single XML-like stream:
      <thinking> ...reasoning summary + tool progress... </thinking><response> ...final answer... </response>
    """
    stream = client.responses.create(stream=True, **create_kwargs)

    thinking_open = True
    response_open = False
    yield "<thinking>"

    try:
        for event in stream:
            t = event.type or ""

            # --- Reasoning summary stream ---
            if t == "response.reasoning_summary_text.delta":
                yield _format_thinking_delta(getattr(event, "delta", "") or "")

            elif t == "response.reasoning_summary_text.done":
                pass  # keep <thinking> open for tool progress

            # --- Output item lifecycle (covers tools like web_search, file_search, image_generation, etc.) ---
            elif t == "response.output_item.added":
                item = getattr(event, "item", None) or getattr(event, "output_item", None)
                item_type = getattr(item, "type", None) or getattr(event, "item_type", None)
                if item_type:
                    yield f"\n[tool:item-added type={item_type}]\n"

            elif t == "response.output_item.done":
                item = getattr(event, "item", None) or getattr(event, "output_item", None)
                item_type = getattr(item, "type", None) or getattr(event, "item_type", None)
                if item_type:
                    yield f"[tool:item-done type={item_type}]\n\n"

            # --- Built-in web_search progress stream ---
            elif t.startswith("response.web_search_call."):
                phase = t.split(".")[-1]  # e.g., 'in_progress', 'completed', possibly 'result'
                q = getattr(event, "query", None)
                if phase in ("created", "started", "searching", "in_progress"):
                    yield f"[web_search:{phase}{' q='+q if q else ''}]\n"
                elif phase == "result":
                    title = getattr(event, "title", None)
                    url = getattr(event, "url", None)
                    if title or url:
                        yield f"- {title or ''} {url or ''}\n"
                elif phase == "completed":
                    results = getattr(event, "results", None) or []
                    n = getattr(event, "num_results", None) or (len(results) if isinstance(results, list) else None)
                    yield f"[web_search:completed results={n if n is not None else 'unknown'}]\n\n"

            # --- Function calling (your own tools) ---
            elif t == "response.function_call_arguments.delta":
                yield getattr(event, "delta", "") or ""
            elif t == "response.function_call_arguments.done":
                yield "\n[function_call:args_done]\n"

            # --- Main model answer text ---
            elif t == "response.output_text.delta":
                if thinking_open:
                    yield "\n</thinking>\n\n"
                    thinking_open = False
                if not response_open:
                    response_open = True
                    yield "<response>\n\n"
                yield getattr(event, "delta", "") or ""

            elif t == "response.output_text.done":
                if response_open:
                    yield "\n\n</response>\n"
                    response_open = False

            # --- Finalization / errors ---
            elif t == "response.completed":
                if thinking_open:
                    yield "\n</thinking>\n"
                    thinking_open = False

            elif t == "response.error":
                if thinking_open:
                    yield "\n</thinking>\n"
                    thinking_open = False
                if response_open:
                    yield "\n</response>\n"
                    response_open = False
                err = getattr(event, "error", None)
                msg = getattr(err, "message", None) if err else None
                yield f"<!-- error: {msg or err or 'unknown'} -->"

            # else: ignore other event types

    finally:
        try:
            stream.close()
        except Exception:
            pass


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    kwargs = dict(
        model="gpt-5",
        reasoning={"effort": "high", "summary": "detailed"},
        tools=[{"type": "web_search"}],  # built-in tool
        instructions="Search the web as needed (multiple searches OK) and cite sources.",
        input="find bingran you in hartmut haeffner's group and review his google scholar page",
    )

    for chunk in stream_response_with_tags(**kwargs):
        print(chunk, end="", flush=True)
    print()