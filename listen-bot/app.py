import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import os
from pydub import AudioSegment
import tempfile

from llmware.models import ModelCatalog
from llmware.gguf_configs import GGUFConfigs

# Set WhisperCPP configs
GGUFConfigs().set_config("whisper_cpp_verbose", "OFF")
GGUFConfigs().set_config("whisper_cpp_realtime_display", False)
GGUFConfigs().set_config("whisper_language", "en")
GGUFConfigs().set_config("whisper_remove_segment_markers", True)

# Load models once
whisper_model = ModelCatalog().load_model("whisper-cpp-base-english")
text_llm = ModelCatalog().load_model("phi-3-onnx")  # or bling-tiny-llama-onnx

st.title("🎙️ Voice Assistant with LLMware")

st.markdown("Speak into the mic, and I'll transcribe your voice and generate a response using LLM!")

audio_buffer = []

# Audio frame callback
def audio_callback(frame: av.AudioFrame) -> av.AudioFrame:
    audio_data = frame.to_ndarray().flatten().astype(np.int16)
    audio_buffer.append(audio_data.tobytes())
    return frame

# Start recording
webrtc_ctx = webrtc_streamer(
    key="mic",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    audio_frame_callback=audio_callback,
)

if st.button("🛑 Stop Recording and Process"):
    if len(audio_buffer) == 0:
        st.warning("No audio captured.")
    else:
        st.success("Audio captured. Processing...")

        # Save to temp .wav file
        temp_wav_path = os.path.join(tempfile.gettempdir(), "recorded.wav")
        raw_audio = b"".join(audio_buffer)
        with open(temp_wav_path, "wb") as f:
            f.write(raw_audio)

        # Convert raw audio to proper WAV using pydub
        audio = AudioSegment.from_file(temp_wav_path, format="s16le", frame_rate=48000, channels=1, sample_width=2)
        converted_path = os.path.join(tempfile.gettempdir(), "converted.wav")
        audio.export(converted_path, format="wav")

        st.audio(converted_path, format="audio/wav")

        # Transcribe
        transcription = whisper_model.inference(converted_path)
        user_prompt = transcription["llm_response"]
        st.subheader("📝 Transcription")
        st.write(user_prompt)

        # Get LLM response
        response = text_llm.inference(user_prompt)
        st.subheader("🤖 LLM Response")
        st.write(response["llm_response"])
