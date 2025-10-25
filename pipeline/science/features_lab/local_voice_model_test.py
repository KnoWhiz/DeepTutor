# app.py
# Streamlit demo for a local LIRA Whisper server
# Usage:
#   streamlit run app.py
#
# Defaults assume you started LIRA like:
#   lira serve --backend openai --model whisper-base --device cpu --host 127.0.0.1 --port 5000
#
# The OpenAI-compatible endpoint is:
#   POST http://127.0.0.1:5000/v1/audio/transcriptions
#
# References:
# - OpenAI audio transcription endpoint format (/v1/audio/transcriptions)
# - Streamlit file uploader + audio player

import io
import json
import mimetypes
import requests
import streamlit as st

st.set_page_config(page_title="LIRA Whisper Demo", page_icon="🎙️", layout="centered")

st.title("🎙️ LIRA Whisper (local) — Streamlit Demo")
st.caption("Upload an audio file, send it to your local LIRA server, and view the transcript.")

with st.sidebar:
    st.header("Server & Params")
    server_url = st.text_input(
        "Server URL",
        value="http://127.0.0.1:5000",
        help="Base URL of your LIRA server.",
    )

    endpoint = st.text_input(
        "Transcriptions endpoint",
        value="/v1/audio/transcriptions",
        help="OpenAI-compatible transcription route.",
    )

    # Your LIRA run showed curl using `-F model=whisper-onnx`
    # Keep this editable in case your server expects a different value (e.g., whisper-base).
    model = st.text_input(
        "model (form field)",
        value="whisper-onnx",
        help="Model name sent as a form field. Try 'whisper-onnx' or 'whisper-base'."
    )

    language = st.text_input(
        "language (optional)",
        value="",
        placeholder="e.g. en, zh, de",
        help="ISO language hint. Leave blank to let model auto-detect."
    )

    temperature = st.number_input(
        "temperature (optional)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.1,
        help="Sampling temperature (0 = deterministic)."
    )

    response_format = st.selectbox(
        "response_format",
        options=["json", "verbose_json", "text"],
        index=0,
        help="Most servers return JSON. 'verbose_json' may include segments if implemented."
    )

st.divider()

def guess_mime(filename: str, default: str = "application/octet-stream") -> str:
    # Good enough for common audio types
    mtype, _ = mimetypes.guess_type(filename)
    return mtype or default

uploaded = st.file_uploader(
    "Upload audio",
    type=["wav", "mp3", "m4a", "flac", "ogg", "webm"],
    accept_multiple_files=False,
    help="Max size depends on Streamlit config; WAV/MP3/M4A/FLAC/OGG/WEBM supported."
)

if uploaded:
    file_bytes = uploaded.getvalue()
    # Try to show a player
    fmt = guess_mime(uploaded.name)
    st.audio(file_bytes, format=fmt)

    st.caption(f"File: {uploaded.name} · {len(file_bytes)/1_000_000:.2f} MB · MIME: {fmt}")

    if st.button("Transcribe", type="primary"):
        try:
            url = server_url.rstrip("/") + endpoint
            # Build the multipart/form-data request:
            # - 'file' goes in the `files` dict
            # - other fields can go in `data`
            files = {
                "file": (uploaded.name, io.BytesIO(file_bytes), fmt),
            }
            data = {
                "model": model,
                "response_format": response_format,
            }
            if language.strip():
                data["language"] = language.strip()
            # Temperature often supported by Whisper-style servers
            if temperature is not None:
                data["temperature"] = str(temperature)

            with st.status("Sending to server…", expanded=False) as status:
                resp = requests.post(url, data=data, files=files, timeout=120)
                status.update(label="Received response", state="complete")

            if resp.status_code != 200:
                st.error(f"HTTP {resp.status_code}: {resp.text[:2000]}")
            else:
                # Try flexible parsing depending on response_format
                content_type = resp.headers.get("content-type", "")
                if "application/json" in content_type or response_format in ("json", "verbose_json"):
                    payload = resp.json()
                    # Common OpenAI-style field is 'text'
                    text = None
                    if isinstance(payload, dict):
                        text = payload.get("text")
                    if text is not None:
                        st.subheader("Transcript")
                        st.write(text)
                    else:
                        st.subheader("Raw JSON")
                        st.code(json.dumps(payload, ensure_ascii=False, indent=2))
                        # If segments exist, show them
                        segs = payload.get("segments") if isinstance(payload, dict) else None
                        if segs:
                            with st.expander("Segments"):
                                st.code(json.dumps(segs, ensure_ascii=False, indent=2))
                else:
                    # Some servers may return plain text when response_format='text'
                    st.subheader("Transcript (text)")
                    st.write(resp.text)

        except requests.exceptions.RequestException as e:
            st.error(f"Request error: {e}")

with st.expander("Show sample cURL (matches this app)"):
    curl_url = (server_url.rstrip("/") + endpoint)
    example = f"""curl -X POST "{curl_url}" \\
  -H "accept: application/json" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@{uploaded.name if uploaded else 'path/to/audio.wav'}" \\
  -F "model={model}" \\
  -F "response_format={response_format}"{f' \\\n  -F "language={language.strip()}"' if language.strip() else ""}{f' \\\n  -F "temperature={temperature}"' if temperature is not None else ""}"""
    st.code(example)