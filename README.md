# 🇮🇳 Indian Sign Language Recognition System

A real-time **Indian Sign Language (ISL) Alphabet Recognition System** built using **Python, OpenCV, MediaPipe, and TensorFlow/Keras**.

This project uses a webcam to detect hand landmarks and recognize **Indian Sign Language alphabet signs from A to Z** in real time. The recognized signs can be converted into text and used as a foundation for communication assistance.

---

## 📌 Project Overview

Communication can be difficult for people who use sign language when interacting with people who do not understand it.

This project aims to provide a computer-vision-based system that can recognize Indian Sign Language alphabet gestures using a webcam.

The system detects the user's hand, extracts **21 hand landmarks using MediaPipe**, converts them into numerical features, and uses a trained **TensorFlow/Keras neural network** to classify the gesture into one of the 26 alphabet classes.

### System Workflow

```text
                    Webcam
                       │
                       ▼
                OpenCV Video Capture
                       │
                       ▼
             MediaPipe Hand Detection
                       │
                       ▼
                21 Hand Landmarks
                       │
                       ▼
             Landmark Normalization
                       │
                       ▼
                 63 Features
                       │
                       ▼
             TensorFlow/Keras Model
                       │
                       ▼
                A-Z Classification
                       │
                       ▼
                 Recognized Text
                       │
                       ▼
                Communication Output
```

---

# ✨ Features

- 🎥 Real-time webcam-based recognition
- ✋ MediaPipe hand detection
- 📍 21 hand landmark extraction
- 🔢 63 numerical features per hand
- 🤖 TensorFlow/Keras neural network
- 🔤 Recognition of all **26 alphabet classes (A-Z)**
- 📊 Confidence score
- 🔄 Prediction smoothing
- 🖥️ Graphical user interface
- 📝 Letter and text formation
- ␣ Space support
- ⌫ Backspace support
- 🗑️ Clear text support
- 🔊 Text-to-speech support
- 📈 Accuracy and evaluation metrics
- 📊 Confusion matrix
- 📄 Classification report
- 📚 Complete training and evaluation pipeline

---

# 🔤 Supported Alphabet

The current trained model supports all 26 alphabet classes:

```text
A  B  C  D  E  F  G  H  I  J  K  L  M
N  O  P  Q  R  S  T  U  V  W  X  Y  Z
```

> **Note:** This project focuses on alphabet/static sign recognition. It is not a complete natural-language Indian Sign Language translator.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Main programming language |
| OpenCV | Webcam access and image processing |
| MediaPipe | Hand detection and landmark extraction |
| TensorFlow | Machine learning framework |
| Keras | Neural network implementation |
| NumPy | Numerical processing |
| Pandas | Dataset processing |
| Scikit-learn | Data splitting and evaluation |
| Matplotlib | Graphs and confusion matrix |
| Tkinter / CustomTkinter | Graphical interface |
| pyttsx3 | Text-to-speech |

---

# 💻 System Requirements

Before installing the project, make sure your computer has:

### Hardware

- Windows 10 or Windows 11
- Webcam
- At least 4 GB RAM
- Internet connection for initial setup
- A reasonably modern CPU

A dedicated GPU is **not required**.

The project can run using the CPU.

### Software

- Python 3.11
- Git
- pip
- Webcam drivers

---

# 🐍 Python Version

The project was developed and tested using:

```text
Python 3.11
```

Python 3.11 is recommended because TensorFlow and MediaPipe compatibility can vary with newer Python versions.

Check your Python version:

```bash
python --version
```

Expected:

```text
Python 3.11.x
```

---

# 📥 Installation and Setup

## Step 1 — Install Python

Download and install Python 3.11 from:

https://www.python.org/downloads/

During installation on Windows, make sure to enable:

```text
Add Python to PATH
```

Verify the installation:

```powershell
python --version
```

---

## Step 2 — Install Git

If Git is not already installed, download it from:

https://git-scm.com/downloads

Verify:

```powershell
git --version
```

---

# 📂 Step 3 — Clone the Repository

Open **PowerShell** or **Command Prompt**.

Clone the repository:

```powershell
git clone https://github.com/YOUR-USERNAME/Indian-Sign-Language-Detection.git
```

Replace `YOUR-USERNAME` with the actual GitHub username that owns this repository.

Enter the project directory:

```powershell
cd Indian-Sign-Language-Detection
```

---

# 🐍 Step 4 — Create a Virtual Environment

Creating a virtual environment is recommended so that the project dependencies do not interfere with other Python projects.

Run:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
venv\Scripts\activate
```

After activation, you should see:

```text
(venv)
```

at the beginning of your terminal.

Example:

```text
(venv) PS C:\Users\User\Indian-Sign-Language-Detection>
```

---

# 📦 Step 5 — Install Dependencies

Install all required Python packages:

```powershell
pip install -r requirements.txt
```

This installs the libraries required by the project.

If `pip` is outdated, you can update it first:

```powershell
python -m pip install --upgrade pip
```

Then:

```powershell
pip install -r requirements.txt
```

---

# 📁 Project Structure

```text
Indian-Sign-Language-Detection/
│
├── dataset/
│   ├── raw/
│   ├── processed/
│   ├── A/
│   ├── B/
│   ├── C/
│   ├── ...
│   ├── Z/
│   ├── isl_landmarks.csv
│   └── isl_landmarks_balanced.csv
│
├── models/
│   ├── hand_landmarker.task
│   ├── isl_model.keras
│   └── labels.npy
│
├── results/
│   ├── test_confusion_matrix.png
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── classification_report.txt
│
├── src/
│   ├── app.py
│   ├── camera_test.py
│   ├── check_dataset.py
│   ├── collect_data.py
│   ├── evaluate_model.py
│   ├── hand_detection.py
│   ├── predict.py
│   ├── prepare_kaggle_dataset.py
│   └── train_model.py
│
├── README.md
├── requirements.txt
├── .gitignore
├── PROJECT_REPORT.md
├── PRESENTATION_CONTENT.md
└── DEMO_GUIDE.md
```

---

# ▶️ Running the Application

After completing the installation, make sure the virtual environment is activated:

```powershell
venv\Scripts\activate
```

Then start the main application:

```powershell
python src\app.py
```

The application should open the graphical interface and access your webcam.

---

# 📷 How to Use the Application

## 1. Start the Application

Run:

```powershell
python src\app.py
```

## 2. Allow Camera Access

If Windows asks for camera permission, select:

```text
Allow
```

## 3. Position Your Hand

Place your hand clearly in front of the webcam.

For best results:

- Use good lighting.
- Keep the hand fully visible.
- Avoid covering your fingers.
- Keep the background relatively simple.
- Keep your hand at a reasonable distance from the camera.

## 4. Make an ISL Alphabet Sign

Make one of the supported alphabet signs:

```text
A B C D E F G H I J K L M
N O P Q R S T U V W X Y Z
```

The system will display the predicted letter and confidence.

Example:

```text
Sign: A
Confidence: 96.42%
```

---

# 🔤 Text Formation

The application can use recognized letters to form text.

For example:

```text
H → E → L → L → O
```

can produce:

```text
HELLO
```

The application provides controls for:

```text
Add Letter
Space
Backspace
Clear
Speak
Exit
```

This allows individual recognized signs to be combined into words and sentences.

---

# 🔊 Text-to-Speech

The application includes text-to-speech functionality.

After forming a word or sentence, use the **Speak** option.

For example:

```text
HELLO WORLD
```

can be converted into spoken audio.

Text-to-speech is intended to provide an additional communication output.

---

# 🧠 How the Machine Learning System Works

The system uses **MediaPipe Hand Landmarker** to detect a hand.

MediaPipe detects:

```text
21 hand landmarks
```

Each landmark contains:

```text
X coordinate
Y coordinate
Z coordinate
```

Therefore:

```text
21 landmarks × 3 coordinates = 63 features
```

The coordinates are normalized relative to the wrist.

The resulting 63-feature vector is passed to the TensorFlow/Keras model.

The model predicts one of 26 classes:

```text
A-Z
```

---

# 🏗️ Model Architecture

The neural network uses the normalized hand landmark features as input.

The architecture consists of multiple fully connected layers.

Conceptually:

```text
Input Layer
63 Features
     │
     ▼
Dense Layer
128 Neurons
ReLU
     │
     ▼
Dropout
     │
     ▼
Dense Layer
64 Neurons
ReLU
     │
     ▼
Dropout
     │
     ▼
Dense Layer
32 Neurons
ReLU
     │
     ▼
Output Layer
26 Classes
Softmax
     │
     ▼
A-Z Prediction
```

---

# 🤖 Trained Model Files

The trained model is stored in:

```text
models/isl_model.keras
```

The alphabet labels are stored in:

```text
models/labels.npy
```

The MediaPipe hand landmark model is stored in:

```text
models/hand_landmarker.task
```

These files allow users to run the trained system without retraining the model.

---

# 📊 Dataset

The project uses hand-sign data representing the alphabet classes A-Z.

The dataset is processed into numerical hand landmark features.

Each sample contains:

```text
63 landmark features
+
1 alphabet label
```

The dataset is organized into alphabet classes.

The training pipeline performs an **80/20 stratified train-test split**.

```text
80% → Training
20% → Testing
```

The test data is kept separate from the training process to provide an unbiased evaluation.

---

# 📈 Model Evaluation

The project includes a complete evaluation pipeline.

Run:

```powershell
python src\evaluate_model.py
```

The evaluation generates:

- Test accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

The results are stored inside:

```text
results/
```

---

# 🎯 Current Model Performance

The current full A-Z model achieved approximately:

```text
72.77% test accuracy
```

This value was obtained using the held-out test set.

The actual performance may vary when the model is used with different users, cameras, lighting conditions, backgrounds, and hand positions.

> **Important:** The 72.77% value is the measured performance of the current A-Z model. It should not be interpreted as the accuracy of a complete natural-language ISL translation system.

---

# 📊 Confusion Matrix

The evaluation process generates a confusion matrix showing how the model performs for each alphabet class.

The generated file is available in:

```text
results/test_confusion_matrix.png
```

The confusion matrix can be used to identify alphabet signs that the model frequently confuses.

---

# 🧪 Model Training

Most users do **not** need to train the model because a trained model is already included.

However, developers who want to retrain the model can use the training pipeline.

First validate the dataset:

```powershell
python src\check_dataset.py
```

If the dataset is valid, train the model:

```powershell
python src\train_model.py
```

After training, evaluate it:

```powershell
python src\evaluate_model.py
```

The newly trained model will be saved to:

```text
models/isl_model.keras
```

---

# 📊 Dataset Validation

Before training, check whether the required A-Z classes are available:

```powershell
python src\check_dataset.py
```

The checker verifies the dataset and confirms whether all required alphabet classes are present.

A successful dataset should contain:

```text
A
B
C
D
E
F
G
H
I
J
K
L
M
N
O
P
Q
R
S
T
U
V
W
X
Y
Z
```

---

# 📷 Camera Testing

If you want to test the webcam separately, run:

```powershell
python src\camera_test.py
```

This checks whether OpenCV can access the camera.

Press:

```text
Q
```

to exit the camera test.

---

# ✋ Hand Detection Testing

To test MediaPipe hand landmark detection independently, use:

```powershell
python src\hand_detection.py
```

This is useful when troubleshooting camera or hand-detection problems.

---

# 🔮 Future Scope

The current system focuses on alphabet/static sign recognition.

Future versions could include:

### Dynamic Sign Recognition

Some signs involve movement rather than a single static pose.

Future systems could use:

- LSTM
- GRU
- Transformer models
- Temporal landmark sequences

### Word Recognition

Instead of recognizing individual letters, the system could recognize complete ISL words.

### Sentence Recognition

A future system could translate complete sign-language sentences.

### Larger Vocabulary

The system could be extended from alphabet recognition to hundreds or thousands of ISL signs.

### Additional Visual Features

Future versions could combine:

- Hand landmarks
- Body pose
- Facial expressions
- Head movement

### Mobile Application

The recognition system could eventually be converted into an Android or iOS application.

### Web Application

A browser-based version could allow users to access the system without installing Python.

---

# ⚠️ Limitations

This project is primarily a **static alphabet recognition system**.

It should not be described as a complete Indian Sign Language translator.

Some signs involve movement and therefore cannot be fully represented by a single static frame.

For example:

```text
J
Z
```

may require temporal/movement information for reliable recognition.

Other factors that can affect recognition include:

- Poor lighting
- Low-quality cameras
- Complex backgrounds
- Occluded fingers
- Incorrect hand positioning
- Different hand orientations
- Differences between users

---

# 🛠️ Troubleshooting

## Camera Does Not Open

Check:

1. Camera is connected.
2. Windows camera permissions are enabled.
3. Another application is not using the camera.
4. Restart the application.

You can also test the camera:

```powershell
python src\camera_test.py
```

---

## `ModuleNotFoundError`

Make sure the virtual environment is activated:

```powershell
venv\Scripts\activate
```

Then reinstall dependencies:

```powershell
pip install -r requirements.txt
```

---

## TensorFlow / MediaPipe Compatibility Error

Use:

```text
Python 3.11
```

and reinstall the packages from:

```text
requirements.txt
```

Avoid randomly upgrading TensorFlow or MediaPipe because incompatible versions can cause native-library errors.

---

## Model Not Found

Make sure these files exist:

```text
models/
├── isl_model.keras
├── labels.npy
└── hand_landmarker.task
```

Then run the application again:

```powershell
python src\app.py
```

---

## Poor Recognition

Try:

- Better lighting
- Plain background
- Keep the entire hand visible
- Move closer to the camera
- Avoid rapid movement
- Keep fingers clearly separated when required
- Position the hand inside the camera view

---

# 🔐 GitHub / Repository Notes

The project uses a `.gitignore` file to prevent unnecessary files from being uploaded.

The following should not normally be committed:

```text
venv/
venv_old/
__pycache__/
*.pyc
.env
```

The trained model files can be included if their size is suitable for the repository.

If a model or dataset is too large for normal GitHub storage, Git Large File Storage (Git LFS) or an external download location may be required.

---

# 📄 Additional Documentation

The repository may also contain:

```text
PROJECT_REPORT.md
```

Detailed project report and methodology.

```text
PRESENTATION_CONTENT.md
```

Presentation material for academic demonstrations.

```text
DEMO_GUIDE.md
```

Instructions for demonstrating the project.

---

# 🎓 Academic Purpose

This project demonstrates the practical application of:

- Computer Vision
- Machine Learning
- Deep Learning
- Hand Landmark Detection
- Image Processing
- Human-Computer Interaction
- Real-Time AI Systems

It was developed as an academic/MCA project.

---

# 👨‍💻 Author

**Melvin Mathew**


Interests:

- Artificial Intelligence
- Machine Learning
- Computer Vision
- Cloud Computing
- Software Development

---

# ⭐ Project Goal

The goal of this project is to demonstrate how **Computer Vision and Machine Learning can be used to recognize Indian Sign Language alphabet gestures in real time**.

The current system provides an A-Z alphabet recognition foundation that can be extended in the future toward word-level, sentence-level, and dynamic Indian Sign Language translation.

---

## 📜 License

This project is intended primarily for educational and academic purposes.

Please check the licenses and usage terms of any third-party datasets, pretrained models, and libraries before redistributing them.

---

## ⭐ If You Find This Project Useful

If this project helps you learn about computer vision, machine learning, or sign-language recognition, consider giving the repository a ⭐ on GitHub.