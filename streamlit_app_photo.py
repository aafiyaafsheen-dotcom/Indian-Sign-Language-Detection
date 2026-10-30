
import streamlit as st
import numpy as np
import mediapipe as mp
import tensorflow as tf
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="ISL Recognition", page_icon="🤟")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "isl_model.keras"
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
LABELS_PATH = ROOT / "models" / "labels.npy"

st.title("🤟 Indian Sign Language Recognition")
st.write("Capture an ISL alphabet sign to recognise it.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    labels = np.load(LABELS_PATH, allow_pickle=True)
    return model, labels

@st.cache_resource
def load_detector():
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(HAND_MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)

def normalize_landmarks(hand):
    wrist = hand[0]
    values = []

    for point in hand:
        values.extend([
            point.x - wrist.x,
            point.y - wrist.y,
            point.z - wrist.z
        ])

    values = np.array(values, dtype=np.float32)
    maximum = np.max(np.abs(values))

    if maximum > 0:
        values = values / maximum

    return values.reshape(1, 63)

if "letters" not in st.session_state:
    st.session_state.letters = ""

if "prediction" not in st.session_state:
    st.session_state.prediction = None

try:
    model, labels = load_model()
    detector = load_detector()
except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()

photo = st.camera_input("Capture your hand sign")

if photo is not None:
    if st.button("Recognise Sign", type="primary"):
        image = Image.open(photo).convert("RGB")
        rgb = np.array(image)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        if result.hand_landmarks:
            features = normalize_landmarks(result.hand_landmarks[0])
            probabilities = model.predict(features, verbose=0)[0]
            index = int(np.argmax(probabilities))

            st.session_state.prediction = (
                str(labels[index]),
                float(probabilities[index])
            )
        else:
            st.session_state.prediction = None
            st.warning("No hand detected. Please try again.")

if st.session_state.prediction is not None:
    letter, confidence = st.session_state.prediction

    st.success(f"Predicted Alphabet: {letter}")
    st.metric("Confidence", f"{confidence * 100:.2f}%")

    if st.button("Add Letter"):
        st.session_state.letters += letter

st.divider()
st.subheader("Recognised Text")
st.text_input("Text", key="letters", disabled=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("Add Space"):
        st.session_state.letters += " "
        st.rerun()

with col2:
    if st.button("Clear Text"):
        st.session_state.letters = ""
        st.rerun()
