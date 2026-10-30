import streamlit as st
import numpy as np
import mediapipe as mp
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageOps

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
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def normalize_landmarks(hand):
    wrist = hand[0]
    values = []
    for point in hand:
        values.extend([point.x - wrist.x, point.y - wrist.y, point.z - wrist.z])
    values = np.array(values, dtype=np.float32)
    maximum = np.max(np.abs(values))
    if maximum > 0:
        values = values / maximum
    return values.reshape(1, 63)


# Callbacks (they run before the script reruns, so changing state is safe)
def add_text(ch):
    st.session_state.text += ch


def backspace():
    st.session_state.text = st.session_state.text[:-1]


def clear_text():
    st.session_state.text = ""


if "text" not in st.session_state:
    st.session_state.text = ""

try:
    model, labels = load_model()
    detector = load_detector()
except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()

mirror = st.checkbox("Mirror image (try this if predictions look wrong)", value=False)
photo = st.camera_input("Capture your hand sign")

if photo is not None:
    image = Image.open(photo).convert("RGB")
    if mirror:
        image = ImageOps.mirror(image)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        features = normalize_landmarks(result.hand_landmarks[0])
        probabilities = model.predict(features, verbose=0)[0]
        index = int(np.argmax(probabilities))
        letter = str(labels[index])
        confidence = float(probabilities[index])

        st.success(f"Predicted Alphabet: {letter}")
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        if confidence < 0.6:
            st.info("Low confidence. Try better lighting or a clearer pose.")

        st.button("Add Letter", type="primary", on_click=add_text, args=(letter,))
    else:
        st.warning("No hand detected. Please try again.")

st.divider()
st.subheader("Recognised Text")
st.text_input("Text", value=st.session_state.text, disabled=True)

c1, c2, c3 = st.columns(3)
c1.button("Add Space", on_click=add_text, args=(" ",))
c2.button("Backspace", on_click=backspace)
c3.button("Clear Text", on_click=clear_text)
