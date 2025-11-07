# app.py (additions marked ==== DIAGNOSTICS ====)
import io, json, time, inspect, requests, streamlit as st
import wave, contextlib
import numpy as np

st.set_page_config(page_title="🎙️ Voice → LIRA Transcribe", page_icon="🎙️", layout="centered")
st.title("🎙️ Record • Stop • Transcribe (LIRA local server)")

with st.sidebar:
    st.header("Server & options")
    base_url = st.text_input("Server URL", "http://127.0.0.1:5000")
    endpoint = st.text_input("Endpoint", "/v1/audio/transcriptions")
    model = st.text_input("model (form field)", "whisper-onnx")
    language = st.text_input("language (optional)", value="", placeholder="e.g. en, zh, de")
    response_format = st.selectbox("response_format", ["json", "verbose_json", "text"], index=0)
    temperature = st.number_input("temperature", 0.0, 1.0, 0.0, 0.1)
    auto = st.toggle("Auto-transcribe when recording finishes", value=True)

st.write("Press the microphone to **start**, then press again to **stop**. We'll transcribe automatically or when you click **Transcribe**.")

# Safe audio_input across Streamlit versions
audio_kwargs = {"key": "mic"}
try:
    if "sample_rate" in inspect.signature(st.audio_input).parameters:
        audio_kwargs["sample_rate"] = 16000
except Exception:
    pass

audio_file = st.audio_input("Record a voice message", **audio_kwargs)

def transcribe(bytes_data: bytes, filename: str = "recording.wav"):
    url = base_url.rstrip("/") + endpoint
    files = {"file": (filename, io.BytesIO(bytes_data), "audio/wav")}
    data = {"model": model, "response_format": response_format, "temperature": str(temperature)}
    if language.strip(): data["language"] = language.strip()
    with st.status("Transcribing…", expanded=False) as status:
        resp = requests.post(url, data=data, files=files, timeout=180)
        status.update(label="Received response", state="complete")
    if resp.status_code != 200:
        st.error(f"HTTP {resp.status_code}\n{resp.text[:2000]}")
        return
    ct = resp.headers.get("content-type", "")
    if "application/json" in ct or response_format in ("json", "verbose_json"):
        payload = resp.json()
        text = payload.get("text") if isinstance(payload, dict) else None
        st.subheader("Transcript" if text is not None else "Raw JSON")
        st.write(text) if text is not None else st.code(json.dumps(payload, ensure_ascii=False, indent=2))
        if isinstance(payload, dict) and "segments" in payload:
            with st.expander("Segments"):
                st.code(json.dumps(payload["segments"], ensure_ascii=False, indent=2))
    else:
        st.subheader("Transcript (text)")
        st.write(resp.text)

def _diagnose_wav(bytes_data: bytes):
    # ==== DIAGNOSTICS ====
    try:
        bio = io.BytesIO(bytes_data)
        with contextlib.closing(wave.open(bio, "rb")) as wf:
            fr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            n = wf.getnframes()
            duration = n / float(fr) if fr else 0.0
            wf.rewind()
            raw = wf.readframes(n)
        # convert to numpy for a simple RMS (normalized)
        dtype = {1: np.int8, 2: np.int16, 3: np.int32, 4: np.int32}.get(sw, np.int16)
        audio_np = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        # scale based on sample width (approx)
        max_val = float(2 ** (8*sw - 1))
        rms = float(np.sqrt(np.mean(np.square(audio_np / max_val)))) if audio_np.size else 0.0
        return {"duration_s": duration, "samplerate": fr, "channels": ch, "samplewidth": sw, "rms": rms}
    except Exception as e:
        return {"error": str(e)}

if audio_file is not None:
    bytes_data = audio_file.getvalue()
    st.audio(bytes_data, format="audio/wav")

    # ==== DIAGNOSTICS UI ====
    diag = _diagnose_wav(bytes_data)
    if "error" in diag:
        st.warning(f"Could not parse WAV header: {diag['error']}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Duration (s)", f"{diag['duration_s']:.2f}")
        col2.metric("Sample rate", f"{diag['samplerate']} Hz")
        col3.metric("Channels", str(diag['channels']))
        col4.metric("Loudness (RMS)", f"{diag['rms']:.3f}")
        if diag["duration_s"] < 0.3 or diag["rms"] < 0.005:
            st.info("It looks like this recording is either very short or nearly silent. "
                    "Check your mic device & gain in the browser’s site settings and macOS Sound input.")

    st.caption(f"{getattr(audio_file, 'name', 'recording.wav')} · {len(bytes_data)/1_000_000:.2f} MB")
    st.download_button("Download WAV", bytes_data, file_name="recording.wav", mime="audio/wav")

    if auto and not st.session_state.get("did_auto", False) and ("error" not in diag):
        time.sleep(0.15)
        transcribe(bytes_data, getattr(audio_file, "name", "recording.wav"))
        st.session_state["did_auto"] = True

    if st.button("Transcribe", type="primary"):
        transcribe(bytes_data, getattr(audio_file, "name", "recording.wav"))
        st.session_state["did_auto"] = True

if audio_file is None and st.session_state.get("did_auto"):
    st.session_state["did_auto"] = False

with st.expander("Show equivalent cURL"):
    curl = f"""curl -X POST "{base_url.rstrip('/') + endpoint}" \\
  -H "accept: application/json" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@recording.wav" \\
  -F "model={model}" \\
  -F "response_format={response_format}"{f' \\\\\n  -F "language={language.strip()}"' if language.strip() else ""}{f' \\\\\n  -F "temperature={temperature}"' if temperature is not None else ""}"""
    st.code(curl)