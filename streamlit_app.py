import collections
import threading

import av
import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import tensorflow as tf
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(page_title="ISL Recognition", page_icon="🤟")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "isl_model.keras"
HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
LABELS_PATH = ROOT / "models" / "labels.npy"

st.title("🤟 Indian Sign Language Recognition")
st.write("Click START, allow the camera, and show an ISL alphabet sign.")


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
    for p in hand:
        values.extend([p.x - wrist.x, p.y - wrist.y, p.z - wrist.z])
    values = np.array(values, dtype=np.float32)
    maximum = np.max(np.abs(values))
    if maximum > 0:
        values = values / maximum
    return values.reshape(1, 63)


try:
    model, labels = load_model()
    detector = load_detector()
except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()

# State shared between the video thread and the page
lock = threading.Lock()
shared = {"letter": None, "conf": 0.0}
recent = collections.deque(maxlen=8)  # smoothing over the last frames


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)  # mirror, like a normal webcam app
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with lock:
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            h, w = img.shape[:2]
            for p in hand:
                cv2.circle(img, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)

            probs = model(normalize_landmarks(hand), training=False).numpy()[0]
            idx = int(np.argmax(probs))
            recent.append((str(labels[idx]), float(probs[idx])))

            votes = collections.Counter(l for l, _ in recent)
            letter = votes.most_common(1)[0][0]
            confs = [c for l, c in recent if l == letter]
            shared["letter"] = letter
            shared["conf"] = sum(confs) / len(confs)

            cv2.putText(img, f"{letter} {shared['conf']*100:.0f}%", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            recent.clear()
            shared["letter"] = None
            shared["conf"] = 0.0

    return av.VideoFrame.from_ndarray(img, format="bgr24")


ctx = webrtc_streamer(
    key="isl",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if "text" not in st.session_state:
    st.session_state.text = ""


def add_letter():
    with lock:
        letter = shared["letter"]
    if letter:
        st.session_state.text += letter


def add_space():
    st.session_state.text += " "


def backspace():
    st.session_state.text = st.session_state.text[:-1]


def clear_text():
    st.session_state.text = ""


st.divider()
st.subheader("Recognised Text")
st.text_input("Text", value=st.session_state.text, disabled=True)

c1, c2, c3, c4 = st.columns(4)
c1.button("Add Letter", type="primary", on_click=add_letter)
c2.button("Add Space", on_click=add_space)
c3.button("Backspace", on_click=backspace)
c4.button("Clear", on_click=clear_text)

st.subheader("Live Prediction")
live = st.empty()

# Refresh the live letter/confidence while the camera is running
if ctx.state.playing:
    import time
    while ctx.state.playing:
        with lock:
            letter, conf = shared["letter"], shared["conf"]
        if letter:
            live.metric(f"Predicted: {letter}", f"{conf*100:.1f}% confidence")
        else:
            live.info("No hand detected")
        time.sleep(0.3)
