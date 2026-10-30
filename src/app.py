"""
Indian Sign Language Recognition System - Main Application

Complete ISL alphabet recognition with GUI, text formation,
and text-to-speech support.

Usage:
    python src/app.py

Requirements:
    - Trained model at models/isl_model.keras
    - Hand landmarker at models/hand_landmarker.task
    - Labels at models/labels.npy
"""

import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import messagebox
from collections import deque, Counter
from PIL import Image, ImageTk
import os
import sys


# ==================================================
# FILE PATHS
# ==================================================

MODEL_PATH = "models/isl_model.keras"
HAND_MODEL_PATH = "models/hand_landmarker.task"
LABELS_PATH = "models/labels.npy"


# ==================================================
# CHECK FILES
# ==================================================

def check_files():
    """Check that all required files exist."""
    missing = []

    if not os.path.exists(MODEL_PATH):
        missing.append(MODEL_PATH)

    if not os.path.exists(HAND_MODEL_PATH):
        missing.append(HAND_MODEL_PATH)

    if not os.path.exists(LABELS_PATH):
        missing.append(LABELS_PATH)

    if missing:
        print()
        print("ERROR: Required files missing:")
        for f in missing:
            print(f"  - {f}")
        print()
        print("Please train the model first:")
        print("  python src/train_model.py")
        print()
        sys.exit(1)


# ==================================================
# TEXT-TO-SPEECH
# ==================================================

class SpeechEngine:
    """Thread-safe text-to-speech using pyttsx3."""

    def __init__(self):
        self.available = False
        self.speaking = False

        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)
            self.engine.setProperty("volume", 0.9)
            self.available = True
            print("Text-to-speech: Available")
        except Exception as e:
            print(f"Text-to-speech: Not available ({e})")
            self.engine = None

    def speak(self, text):
        """Speak text in a background thread."""
        if not self.available or not text.strip():
            return

        if self.speaking:
            return

        thread = threading.Thread(
            target=self._speak_thread,
            args=(text,),
            daemon=True
        )
        thread.start()

    def _speak_thread(self, text):
        """Background speech thread."""
        try:
            self.speaking = True

            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 0.9)
            engine.say(text)
            engine.runAndWait()
            engine.stop()

        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            self.speaking = False


# ==================================================
# HAND DETECTOR
# ==================================================

class HandDetector:
    """MediaPipe hand landmark detector."""

    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=HAND_MODEL_PATH
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )

        self.detector = HandLandmarker.create_from_options(options)
        self.start_time = time.monotonic()

    def detect(self, rgb_frame):
        """Detect hand in frame. Returns landmarks or None."""
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (time.monotonic() - self.start_time) * 1000
        )

        result = self.detector.detect_for_video(
            mp_image, timestamp_ms
        )

        if result.hand_landmarks:
            return result.hand_landmarks[0]

        return None

    def close(self):
        self.detector.close()


# ==================================================
# LANDMARK NORMALIZER
# ==================================================

def normalize_landmarks(hand_landmarks):
    """Normalize hand landmarks using wrist as reference."""

    landmarks = []

    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    wrist_z = hand_landmarks[0].z

    for landmark in hand_landmarks:
        x = landmark.x - wrist_x
        y = landmark.y - wrist_y
        z = landmark.z - wrist_z
        landmarks.extend([x, y, z])

    landmarks = np.array(landmarks, dtype=np.float32)

    max_value = np.max(np.abs(landmarks))

    if max_value > 0:
        landmarks = landmarks / max_value

    return landmarks


# ==================================================
# DRAW HAND
# ==================================================

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]


def draw_hand(frame, hand_landmarks):
    """Draw hand landmarks and connections on frame."""
    height, width, _ = frame.shape

    points = []
    for lm in hand_landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        points.append((x, y))

    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start_idx],
            points[end_idx],
            (0, 255, 0), 2
        )

    # Draw landmarks
    for i, (x, y) in enumerate(points):
        color = (0, 0, 255) if i == 0 else (255, 0, 0)
        cv2.circle(frame, (x, y), 5, color, -1)


# ==================================================
# MAIN APPLICATION
# ==================================================

class ISLApp:
    """Main ISL Recognition Application."""

    def __init__(self):
        # --------------------------------------------------
        # LOAD MODEL
        # --------------------------------------------------
        print("Loading AI model...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.labels = np.load(LABELS_PATH, allow_pickle=True)
        print(f"Model loaded: {len(self.labels)} classes")
        print(f"Classes: {', '.join(self.labels)}")

        # --------------------------------------------------
        # INITIALIZE COMPONENTS
        # --------------------------------------------------
        self.hand_detector = HandDetector()
        self.speech = SpeechEngine()

        # --------------------------------------------------
        # STATE
        # --------------------------------------------------
        self.text_buffer = ""
        self.predicted_label = "..."
        self.confidence = 0.0
        self.hand_detected = False

        # Prediction smoothing
        self.prediction_history = deque(maxlen=10)
        self.CONFIDENCE_THRESHOLD = 0.70

        # Letter confirmation system
        self.CONFIRM_FRAMES = 15
        self.COOLDOWN_FRAMES = 20
        self.stable_count = 0
        self.last_stable_label = None
        self.cooldown_counter = 0
        self.need_change_before_repeat = False
        self.last_confirmed_label = None

        # Camera
        self.camera = None
        self.running = False

        # --------------------------------------------------
        # CREATE GUI
        # --------------------------------------------------
        self._create_gui()

    # ==================================================
    # GUI
    # ==================================================

    def _create_gui(self):
        """Create the Tkinter GUI."""

        self.root = tk.Tk()
        self.root.title("Indian Sign Language Recognition System")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        # Fonts
        title_font = ("Segoe UI", 16, "bold")
        heading_font = ("Segoe UI", 12, "bold")
        label_font = ("Segoe UI", 11)
        text_font = ("Consolas", 14)
        button_font = ("Segoe UI", 10, "bold")
        small_font = ("Segoe UI", 9)

        # Colors
        bg_color = "#1a1a2e"
        panel_color = "#16213e"
        accent_color = "#0f3460"
        text_color = "#e0e0e0"
        highlight_color = "#00d4ff"
        success_color = "#00e676"
        warning_color = "#ffd600"
        btn_color = "#0f3460"
        btn_hover = "#1a4a8a"

        # --------------------------------------------------
        # TITLE BAR
        # --------------------------------------------------

        title_frame = tk.Frame(
            self.root, bg=accent_color,
            padx=15, pady=10
        )
        title_frame.pack(fill="x")

        tk.Label(
            title_frame,
            text="🤟  INDIAN SIGN LANGUAGE RECOGNITION SYSTEM",
            font=title_font,
            bg=accent_color,
            fg="white"
        ).pack()

        # --------------------------------------------------
        # MAIN CONTAINER
        # --------------------------------------------------

        main_frame = tk.Frame(self.root, bg=bg_color, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)

        # --------------------------------------------------
        # LEFT PANEL - CAMERA
        # --------------------------------------------------

        left_frame = tk.Frame(main_frame, bg=panel_color, padx=5, pady=5)
        left_frame.pack(side="left", padx=(0, 10))

        tk.Label(
            left_frame,
            text="📷 LIVE CAMERA",
            font=heading_font,
            bg=panel_color,
            fg=highlight_color
        ).pack(pady=(5, 5))

        self.camera_label = tk.Label(
            left_frame,
            bg="black",
            width=640,
            height=480
        )
        self.camera_label.pack()

        # --------------------------------------------------
        # RIGHT PANEL
        # --------------------------------------------------

        right_frame = tk.Frame(
            main_frame, bg=panel_color,
            padx=15, pady=15, width=320
        )
        right_frame.pack(side="right", fill="y")
        right_frame.pack_propagate(False)

        # Current Sign
        tk.Label(
            right_frame,
            text="CURRENT SIGN",
            font=heading_font,
            bg=panel_color,
            fg=highlight_color
        ).pack(pady=(0, 5))

        self.sign_label = tk.Label(
            right_frame,
            text="...",
            font=("Segoe UI", 48, "bold"),
            bg=panel_color,
            fg=success_color,
            width=4
        )
        self.sign_label.pack(pady=(0, 5))

        # Confidence
        self.confidence_label = tk.Label(
            right_frame,
            text="Confidence: 0.0%",
            font=label_font,
            bg=panel_color,
            fg=warning_color
        )
        self.confidence_label.pack(pady=(0, 5))

        # Confirmation progress
        self.progress_frame = tk.Frame(
            right_frame, bg=panel_color
        )
        self.progress_frame.pack(fill="x", pady=(0, 5))

        self.progress_canvas = tk.Canvas(
            self.progress_frame,
            height=8, bg="#333",
            highlightthickness=0
        )
        self.progress_canvas.pack(fill="x")

        # Status
        self.status_label = tk.Label(
            right_frame,
            text="Status: Starting...",
            font=small_font,
            bg=panel_color,
            fg=text_color
        )
        self.status_label.pack(pady=(0, 15))

        # Separator
        tk.Frame(
            right_frame, bg="#444", height=2
        ).pack(fill="x", pady=5)

        # Recognized Text
        tk.Label(
            right_frame,
            text="RECOGNIZED TEXT",
            font=heading_font,
            bg=panel_color,
            fg=highlight_color
        ).pack(pady=(10, 5))

        self.text_display = tk.Text(
            right_frame,
            font=text_font,
            bg="#0d1117",
            fg="white",
            insertbackground="white",
            height=4,
            width=25,
            wrap="word",
            state="disabled",
            relief="flat",
            padx=10,
            pady=10
        )
        self.text_display.pack(fill="x", pady=(0, 15))

        # Separator
        tk.Frame(
            right_frame, bg="#444", height=2
        ).pack(fill="x", pady=5)

        # --------------------------------------------------
        # BUTTONS
        # --------------------------------------------------

        tk.Label(
            right_frame,
            text="CONTROLS",
            font=heading_font,
            bg=panel_color,
            fg=highlight_color
        ).pack(pady=(10, 8))

        button_style = {
            "font": button_font,
            "bg": btn_color,
            "fg": "white",
            "activebackground": btn_hover,
            "activeforeground": "white",
            "relief": "flat",
            "cursor": "hand2",
            "padx": 10,
            "pady": 6
        }

        # Row 1: Add Letter + Space
        row1 = tk.Frame(right_frame, bg=panel_color)
        row1.pack(fill="x", pady=3)

        self.btn_add = tk.Button(
            row1, text="✓ ADD LETTER",
            command=self.add_letter,
            **button_style
        )
        self.btn_add.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self.btn_space = tk.Button(
            row1, text="⎵ SPACE",
            command=self.add_space,
            **button_style
        )
        self.btn_space.pack(side="right", fill="x", expand=True, padx=(3, 0))

        # Row 2: Backspace + Clear
        row2 = tk.Frame(right_frame, bg=panel_color)
        row2.pack(fill="x", pady=3)

        self.btn_backspace = tk.Button(
            row2, text="⌫ BACKSPACE",
            command=self.backspace,
            **button_style
        )
        self.btn_backspace.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self.btn_clear = tk.Button(
            row2, text="✕ CLEAR",
            command=self.clear_text,
            **button_style
        )
        self.btn_clear.pack(side="right", fill="x", expand=True, padx=(3, 0))

        # Row 3: Speak
        speak_btn_style = button_style.copy()
        speak_btn_style["bg"] = "#1b5e20"
        speak_btn_style["activebackground"] = "#2e7d32"

        self.btn_speak = tk.Button(
            right_frame, text="🔊 SPEAK TEXT",
            command=self.speak_text,
            **speak_btn_style
        )
        self.btn_speak.pack(fill="x", pady=3)

        # Row 4: Exit
        exit_btn_style = button_style.copy()
        exit_btn_style["bg"] = "#b71c1c"
        exit_btn_style["activebackground"] = "#d32f2f"

        self.btn_exit = tk.Button(
            right_frame, text="✕ EXIT",
            command=self.quit_app,
            **exit_btn_style
        )
        self.btn_exit.pack(fill="x", pady=(3, 10))

        # --------------------------------------------------
        # FOOTER
        # --------------------------------------------------

        tk.Frame(
            right_frame, bg="#444", height=2
        ).pack(fill="x", pady=5)

        self.supported_label = tk.Label(
            right_frame,
            text=f"Supported: {', '.join(self.labels)}",
            font=("Segoe UI", 8),
            bg=panel_color,
            fg="#888",
            wraplength=280,
            justify="center"
        )
        self.supported_label.pack(pady=(5, 0))

        # Speech status
        speech_status = "🔊 TTS Available" if self.speech.available else "🔇 TTS Unavailable"
        tk.Label(
            right_frame,
            text=speech_status,
            font=("Segoe UI", 8),
            bg=panel_color,
            fg="#666"
        ).pack()

        # --------------------------------------------------
        # KEY BINDINGS
        # --------------------------------------------------

        self.root.bind("<space>", lambda e: self.add_letter())
        self.root.bind("<minus>", lambda e: self.add_space())
        self.root.bind("<BackSpace>", lambda e: self.backspace())
        self.root.bind("<Delete>", lambda e: self.clear_text())
        self.root.bind("<Escape>", lambda e: self.quit_app())
        self.root.bind("s", lambda e: self.speak_text())

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    # ==================================================
    # BUTTON ACTIONS
    # ==================================================

    def add_letter(self):
        """Add the current predicted letter to text buffer."""
        if self.predicted_label not in ("...", "Unknown"):
            self.text_buffer += self.predicted_label
            self._update_text_display()
            print(f"Added: {self.predicted_label} | Text: {self.text_buffer}")

    def add_space(self):
        """Add a space to the text buffer."""
        self.text_buffer += " "
        self._update_text_display()
        print(f"Space added | Text: '{self.text_buffer}'")

    def backspace(self):
        """Remove the last character."""
        if len(self.text_buffer) > 0:
            self.text_buffer = self.text_buffer[:-1]
            self._update_text_display()
            print(f"Backspace | Text: '{self.text_buffer}'")

    def clear_text(self):
        """Clear all text."""
        self.text_buffer = ""
        self._update_text_display()
        print("Text cleared.")

    def speak_text(self):
        """Speak the current text."""
        if not self.text_buffer.strip():
            return

        if not self.speech.available:
            messagebox.showwarning(
                "Speech Unavailable",
                "Text-to-speech is not available.\n"
                "Install pyttsx3: pip install pyttsx3"
            )
            return

        if self.speech.speaking:
            return

        print(f"Speaking: {self.text_buffer}")
        self.speech.speak(self.text_buffer)

    def quit_app(self):
        """Quit the application."""
        self.running = False

    def _update_text_display(self):
        """Update the text display widget."""
        self.text_display.config(state="normal")
        self.text_display.delete("1.0", tk.END)
        self.text_display.insert("1.0", self.text_buffer)
        self.text_display.config(state="disabled")

    # ==================================================
    # CAMERA LOOP
    # ==================================================

    def process_frame(self):
        """Process one camera frame."""
        if not self.running:
            return

        success, frame = self.camera.read()

        if not success:
            self.status_label.config(
                text="Status: Camera error",
                fg="#ff5252"
            )
            self.root.after(100, self.process_frame)
            return

        # Mirror
        frame = cv2.flip(frame, 1)

        # Detect hand
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_landmarks = self.hand_detector.detect(rgb_frame)

        if hand_landmarks:
            self.hand_detected = True

            # Normalize and predict
            features = normalize_landmarks(hand_landmarks)
            features = features.reshape(1, 63)

            prediction = self.model.predict(features, verbose=0)
            predicted_index = np.argmax(prediction[0])
            self.confidence = float(prediction[0][predicted_index])

            current_label = str(self.labels[predicted_index])

            # Confidence filter + smoothing
            if self.confidence >= self.CONFIDENCE_THRESHOLD:
                self.prediction_history.append(current_label)
                self.predicted_label = Counter(
                    self.prediction_history
                ).most_common(1)[0][0]
            else:
                self.prediction_history.clear()
                self.predicted_label = "..."

            # Letter confirmation (auto-confirm)
            self._process_confirmation()

            # Draw hand on frame
            draw_hand(frame, hand_landmarks)

        else:
            self.hand_detected = False
            self.prediction_history.clear()
            self.predicted_label = "..."
            self.confidence = 0.0
            self.stable_count = 0
            self.last_stable_label = None
            self.need_change_before_repeat = False

        # --------------------------------------------------
        # UPDATE GUI
        # --------------------------------------------------

        # Update sign label
        self.sign_label.config(text=self.predicted_label)

        if self.hand_detected:
            self.sign_label.config(fg="#00e676")
        else:
            self.sign_label.config(fg="#666")

        # Update confidence
        conf_text = f"Confidence: {self.confidence * 100:.1f}%"
        self.confidence_label.config(text=conf_text)

        if self.confidence >= 0.9:
            self.confidence_label.config(fg="#00e676")
        elif self.confidence >= 0.7:
            self.confidence_label.config(fg="#ffd600")
        else:
            self.confidence_label.config(fg="#ff5252")

        # Update status
        if self.hand_detected:
            self.status_label.config(
                text="Status: ✋ Hand Detected",
                fg="#00e676"
            )
        else:
            self.status_label.config(
                text="Status: Show your hand",
                fg="#ff5252"
            )

        # Update progress bar
        self.progress_canvas.delete("all")
        canvas_width = self.progress_canvas.winfo_width()

        if canvas_width > 1 and self.stable_count > 0 and not self.need_change_before_repeat:
            progress = min(self.stable_count / self.CONFIRM_FRAMES, 1.0)
            fill_width = int(canvas_width * progress)
            self.progress_canvas.create_rectangle(
                0, 0, fill_width, 8,
                fill="#00e676", outline=""
            )

        # Update camera display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)

        # Schedule next frame
        if self.running:
            self.root.after(15, self.process_frame)

    def _process_confirmation(self):
        """Process letter auto-confirmation."""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        if self.predicted_label in ("...", "Unknown"):
            return

        if self.predicted_label == self.last_stable_label:
            self.stable_count += 1
        else:
            self.stable_count = 1
            self.last_stable_label = self.predicted_label
            self.need_change_before_repeat = False

        # Auto-confirm after stable frames
        if self.stable_count >= self.CONFIRM_FRAMES:
            if not self.need_change_before_repeat:
                self.text_buffer += self.predicted_label
                self._update_text_display()
                print(
                    f"Auto-confirmed: {self.predicted_label} | "
                    f"Text: {self.text_buffer}"
                )

                self.last_confirmed_label = self.predicted_label
                self.need_change_before_repeat = True
                self.stable_count = 0
                self.cooldown_counter = self.COOLDOWN_FRAMES

    # ==================================================
    # RUN
    # ==================================================

    def run(self):
        """Start the application."""
        print()
        print("=" * 50)
        print("  Starting ISL Recognition System...")
        print("=" * 50)
        print()

        # Open camera
        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Could not open webcam.\n\n"
                "Please check:\n"
                "- Camera is connected\n"
                "- Camera is not used by another app\n"
                "- Camera drivers are installed"
            )
            return

        self.running = True

        print("Camera opened successfully.")
        print()
        print("Keyboard shortcuts:")
        print("  Space     = Add letter")
        print("  -         = Add space")
        print("  Backspace = Delete last")
        print("  Delete    = Clear all")
        print("  S         = Speak text")
        print("  Escape    = Exit")
        print()

        # Start processing
        self.root.after(100, self.process_frame)

        # Run main loop
        self.root.mainloop()

        # Cleanup
        self.running = False

        if self.camera:
            self.camera.release()

        self.hand_detector.close()
        cv2.destroyAllWindows()

        print()
        print("Application closed.")
        if self.text_buffer:
            print(f"Final text: {self.text_buffer}")


# ==================================================
# ENTRY POINT
# ==================================================

def main():

    check_files()

    print()
    print("=" * 50)
    print("  INDIAN SIGN LANGUAGE")
    print("  RECOGNITION SYSTEM")
    print("=" * 50)
    print()

    # Check pillow
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("ERROR: Pillow is required for the GUI.")
        print("Install it: pip install pillow")
        sys.exit(1)

    app = ISLApp()
    app.run()


if __name__ == "__main__":
    main()
