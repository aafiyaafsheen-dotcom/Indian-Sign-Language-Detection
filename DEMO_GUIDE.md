# ISL Recognition System — Demo Guide

## 3-5 Minute Demonstration Walkthrough

---

## Pre-Demo Checklist

Before the demo, ensure:

- [ ] Python virtual environment is activated: `venv\Scripts\activate`
- [ ] Webcam is connected and working
- [ ] Model is trained: `models/isl_model.keras` exists
- [ ] Labels exist: `models/labels.npy`
- [ ] MediaPipe model exists: `models/hand_landmarker.task`
- [ ] Room has adequate lighting
- [ ] Test the camera: `python src/camera_test.py`

---

## Demo Script

### Step 1: Launch the Application (30 seconds)

```bash
python src/app.py
```

**Say:** "This is our Indian Sign Language Recognition System. It uses a webcam, MediaPipe for hand detection, and a neural network trained on ISL alphabet signs."

**Show:** The GUI loading with the live camera feed.

---

### Step 2: Explain Hand Detection (30 seconds)

- Hold your hand in front of the camera
- Show the hand landmarks and skeleton being drawn

**Say:** "The system uses Google's MediaPipe to detect 21 hand landmarks in real time. You can see the hand skeleton being drawn on the video feed."

- Remove your hand to show "Show your hand" status
- Bring it back to show "Hand Detected" status

---

### Step 3: Demonstrate Individual Letters (60 seconds)

Show 3-5 different letter signs:

1. **Sign A** — "The system recognizes this as the letter A with high confidence."
2. **Sign B** — "When I change to B, the prediction updates smoothly."
3. **Sign L** — "L is easy to see — thumb and index finger form an L shape."
4. **Sign V** — "V is the peace sign."
5. **Sign Y** — "Y with thumb and pinky extended."

**Point out:**
- The current sign display updating
- The confidence percentage
- The prediction staying stable (not flickering)

---

### Step 4: Form a Word (60 seconds)

Form the word "HELLO":

1. Sign **H** — wait for auto-confirm or press Space
2. Sign **E** — wait for auto-confirm
3. Sign **L** — wait for auto-confirm
4. Sign **L** — change to a different sign first, then back to L
5. Sign **O** — wait for auto-confirm

**Say:** "Notice how the letter confirmation system works. I need to hold a sign steady for about half a second for it to auto-confirm. The system prevents accidental duplicates — I can't just hold one sign and get multiple letters."

**Point out:** The text buffer building up "HELLO"

---

### Step 5: Add a Space and Form "WORLD" (45 seconds)

1. Click the **SPACE** button (or press `-`)
2. Sign **W** → **O** → **R** → **L** → **D**

**Say:** "I press the space button to add a word break. Now I'll sign WORLD."

**Point out:** The text now reads "HELLO WORLD"

---

### Step 6: Demonstrate Controls (30 seconds)

1. **Backspace** — "I can delete the last character with backspace"
2. **Clear** — "Or clear all text with the clear button"
3. Re-form "HELLO"

**Say:** "The interface has all the text editing controls you'd expect."

---

### Step 7: Text-to-Speech (30 seconds)

1. Make sure text reads "HELLO WORLD" (or whatever word you formed)
2. Click the **SPEAK** button

**Say:** "The most powerful feature — the system can speak the text aloud using an offline text-to-speech engine. Press SPEAK and..." [let the computer speak]

**Say:** "This enables a complete communication flow: sign → text → speech."

---

### Step 8: Show Results (30 seconds)

If time permits, briefly show:

- `results/test_confusion_matrix.png` — "Here's our confusion matrix showing how well the model classifies each letter"
- `results/classification_report.txt` — "Detailed per-class precision, recall, and F1-score"

**Say:** "The model was evaluated on unseen test data using a stratified 80/20 split."

---

## Key Talking Points

During any part of the demo, you can mention:

- **Pipeline:** "The system extracts 21 hand landmarks, normalizes them into 63 features, and classifies using a neural network."
- **Normalization:** "We normalize coordinates relative to the wrist, making recognition work regardless of where the hand is in the frame."
- **Smoothing:** "We use majority voting over 10 frames to prevent flickering predictions."
- **No Internet:** "The entire system runs offline — no cloud API needed."
- **Dataset:** "The model was trained on a public ISL alphabet image dataset containing 26 classes with approximately 160 samples per class (total 4,150 samples)."

---

## Common Questions and Answers

**Q: What's the accuracy?**
A: The overall test accuracy on unseen data is 72.77% across all 26 classes.

**Q: How fast is it?**
A: The system processes frames in real time at approximately 15-30 FPS, depending on the hardware.

**Q: Can it recognize words directly?**
A: Currently, it recognizes individual letters. The user forms words letter by letter. Future scope includes word-level recognition using temporal models.

**Q: Why MediaPipe instead of a CNN on images?**
A: MediaPipe landmarks are invariant to skin tone, background, and lighting, making the system more robust. The model size is also much smaller (63 features vs. thousands of pixels).

**Q: Does it work with both hands?**
A: Currently, it processes the primary hand detected. Future versions could support two-hand signs.

**Q: What about J and Z which are dynamic?**
A: J and Z involve motion in the actual ISL alphabet. The current system treats them as static poses at their starting/ending positions. Dynamic gesture recognition is planned for future work.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not opening | Check camera connection, close other apps using camera |
| Model not found | Run `python src/train_model.py` first |
| Low accuracy | Ensure adequate lighting, hold hand clearly in frame |
| Speech not working | Check `pip install pyttsx3` is installed |
| GUI not appearing | Ensure tkinter is installed (comes with Python) |

---

## Exit

Press **Escape** or click the **EXIT** button to close the application gracefully.

```bash
# Final words
"Thank you for watching the demonstration. Any questions?"
```
