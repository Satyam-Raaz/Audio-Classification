import io

import librosa
import numpy as np
import streamlit as st
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ----------------------------- Configuration -----------------------------
SAMPLE_RATE = 16000
STT_MODEL_NAME = "openai/whisper-small"
INTENT_MODEL_NAME = "Serj/intent-classifier"

FEW_SHOT_PROMPT = """\
Customer: I want to buy a new phone.
Intent: Shopping

Customer: How do I pay my electricity bill?
Intent: Bill Payment

Customer: My internet is not working properly.
Intent: Technical Issue

Customer: I need help resetting my bank password.
Intent: Customer Support

Customer: Can you recommend the best laptop for gaming?
Intent: Shopping

Customer: I was trying to make a payment, but I keep getting a 'Payment not authorized' error.
Intent: Billing Issue

Customer: I received the wrong product and need to return it for a refund.
Intent: Returns & Refunds

Customer: My internet is not working properly and keeps disconnecting frequently.
Intent: Technical Issue

Customer: I am looking to upgrade my phone to the latest model and need advice on the best option.
Intent: Product Inquiry

Customer: I need help resetting my password for my online banking account.
Intent: Customer Support

Customer: I want to track my order that was supposed to be delivered yesterday.
Intent: Delivery / Order Tracking

Customer: I need to change my shipping address for an order I placed yesterday.
Intent: Order Modification

Customer: I am considering subscribing to your premium plan but want to understand the differences between the Basic and Premium plans.
Intent: Subscription Inquiry

Customer: {}
Intent:"""


# ----------------------------- Model loading -----------------------------
@st.cache_resource(show_spinner="Loading speech-to-text model...")
def load_stt():
    processor = WhisperProcessor.from_pretrained(STT_MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(
        STT_MODEL_NAME, low_cpu_mem_usage=True
    )
    model.config.forced_decoder_ids = None
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Loading intent model...")
def load_intent():
    tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        INTENT_MODEL_NAME, low_cpu_mem_usage=True
    )
    model.eval()
    return tokenizer, model


# ----------------------------- Core functions -----------------------------
def load_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """Decode uploaded/recorded audio (wav, mp3, etc.) to 16 kHz mono float32."""
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    return audio


def transcribe_audio(audio: np.ndarray) -> str:
    processor, model = load_stt()
    features = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
    with torch.no_grad():
        ids = model.generate(features)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def classify_intent(text: str) -> str:
    tokenizer, model = load_intent()
    prompt = FEW_SHOT_PROMPT.format(text)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def run_pipeline(audio: np.ndarray):
    if audio is None or len(audio) < SAMPLE_RATE * 0.3:
        st.warning("The recording is too short. Try again and speak for a bit longer.")
        return
    with st.spinner("Transcribing speech..."):
        text = transcribe_audio(audio)
    if not text:
        st.warning("No speech was detected. Try again closer to the microphone.")
        return
    with st.spinner("Classifying intent..."):
        intent = classify_intent(text)
    st.session_state["result"] = {"text": text, "intent": intent}


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Voice Intent Detector", page_icon="🎙️", layout="centered")
st.title("🎙️ Voice Intent Detector")
st.caption("Speak or upload audio. Whisper transcribes it, then a few-shot model detects the intent.")

with st.sidebar:
    st.header("Models")
    st.write(f"**Speech model:** `{STT_MODEL_NAME}`")
    st.write(f"**Intent model:** `{INTENT_MODEL_NAME}`")
    st.write(f"**Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}")

tab_browser, tab_upload = st.tabs(["Record audio", "Upload audio"])

with tab_browser:
    st.write("Click the mic, speak, then click stop.")
    recorded = st.audio_input("Record your request")
    if recorded is not None and st.button("Detect intent", key="btn_record", type="primary"):
        run_pipeline(load_audio_bytes(recorded.getvalue()))

with tab_upload:
    uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])
    if uploaded is not None:
        st.audio(uploaded)
        if st.button("Detect intent", key="btn_upload", type="primary"):
            run_pipeline(load_audio_bytes(uploaded.getvalue()))

# ----------------------------- Results -----------------------------
result = st.session_state.get("result")
if result:
    st.divider()
    st.subheader("Results")
    st.text_area("Transcription", result["text"], height=100, disabled=True)
    st.metric("Detected intent", result["intent"])