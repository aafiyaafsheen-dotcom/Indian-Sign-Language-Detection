"""
ISL Model Training Script

Trains a neural network to recognize Indian Sign Language
alphabet signs (A-Z) from MediaPipe hand landmarks.

Usage:
    python src/train_model.py
    python src/train_model.py --dataset dataset/isl_landmarks.csv
    python src/train_model.py --epochs 50
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

DATASET_PATH = "dataset/isl_landmarks.csv"
MODEL_PATH = "models/isl_model.keras"
LABELS_PATH = "models/labels.npy"
TEST_DATA_PATH = "results/test_data.npz"
HISTORY_PATH = "results/training_history.json"
ACCURACY_PLOT = "results/training_accuracy.png"
LOSS_PLOT = "results/training_loss.png"


# --------------------------------------------------
# PARSE ARGUMENTS
# --------------------------------------------------

def parse_args():

    dataset = DATASET_PATH
    epochs = 50
    batch_size = 32

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg == "--dataset" and i + 1 < len(sys.argv):
            dataset = sys.argv[i + 1]
            i += 2
        elif arg == "--epochs" and i + 1 < len(sys.argv):
            epochs = int(sys.argv[i + 1])
            i += 2
        elif arg == "--batch-size" and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])
            i += 2
        elif arg == "--help":
            print()
            print("ISL Model Training")
            print("==================")
            print()
            print("Usage:")
            print("  python src/train_model.py")
            print("  python src/train_model.py --dataset path/to/data.csv")
            print("  python src/train_model.py --epochs 50 --batch-size 32")
            print()
            sys.exit(0)
        else:
            i += 1

    return dataset, epochs, batch_size


# --------------------------------------------------
# DATA AUGMENTATION
# --------------------------------------------------

def augment_landmarks(X, y, augment_factor=2):
    """
    Augment landmark data with safe transformations.

    Applied transformations:
    - Small translation (shift all coordinates)
    - Small scaling (resize hand)
    - Small coordinate noise (Gaussian)

    NOT applied:
    - Mirroring (left/right changes sign meaning in ISL)
    - Large rotations (would change sign semantics)
    """

    augmented_X = [X]
    augmented_y = [y]

    for _ in range(augment_factor):
        X_aug = X.copy()

        # Random small translation
        translation = np.random.uniform(-0.02, 0.02, size=(X_aug.shape[0], 1))
        X_aug = X_aug + translation

        # Random small scaling
        scale = np.random.uniform(0.95, 1.05, size=(X_aug.shape[0], 1))
        X_aug = X_aug * scale

        # Small Gaussian noise
        noise = np.random.normal(0, 0.01, size=X_aug.shape)
        X_aug = X_aug + noise

        # Re-normalize
        max_values = np.max(np.abs(X_aug), axis=1, keepdims=True)
        max_values[max_values == 0] = 1
        X_aug = X_aug / max_values

        augmented_X.append(X_aug.astype(np.float32))
        augmented_y.append(y.copy())

    return np.vstack(augmented_X), np.concatenate(augmented_y)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    dataset_path, max_epochs, batch_size = parse_args()

    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print()
    print("=" * 60)
    print("  INDIAN SIGN LANGUAGE - MODEL TRAINING")
    print("=" * 60)
    print()

    # --------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)

    print(f"Dataset: {dataset_path}")
    print(f"Shape: {df.shape}")
    print()

    # --------------------------------------------------
    # VALIDATE LABELS
    # --------------------------------------------------

    unique_labels = sorted(df["label"].unique())
    num_classes = len(unique_labels)

    print(f"Classes: {num_classes}")
    print(f"Labels: {', '.join(unique_labels)}")
    print()

    # --------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------

    print("Class distribution:")
    label_counts = df["label"].value_counts().sort_index()

    for label in unique_labels:
        count = label_counts[label]
        print(f"  {label}: {count}")

    print(f"  Total: {len(df)}")
    print()

    # --------------------------------------------------
    # SEPARATE FEATURES AND LABEL
    # --------------------------------------------------

    X = df.drop("label", axis=1).values
    y = df["label"].values

    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")

    # --------------------------------------------------
    # NORMALIZE LANDMARKS
    # --------------------------------------------------

    X = X.astype(np.float32)

    max_values = np.max(
        np.abs(X),
        axis=1,
        keepdims=True
    )

    # Prevent division by zero
    max_values[max_values == 0] = 1

    X = X / max_values

    # --------------------------------------------------
    # ENCODE LABELS
    # --------------------------------------------------

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    labels = label_encoder.classes_

    print()
    print(f"Encoded classes: {labels}")

    # --------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    print()
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # --------------------------------------------------
    # DATA AUGMENTATION (if dataset is small)
    # --------------------------------------------------

    samples_per_class = len(X_train) / num_classes

    if samples_per_class < 300:
        print()
        print("Dataset is small. Applying data augmentation...")

        augment_factor = max(1, int(300 / samples_per_class))
        augment_factor = min(augment_factor, 5)  # cap at 5x

        X_train, y_train = augment_landmarks(
            X_train, y_train,
            augment_factor=augment_factor
        )

        print(f"Augmentation factor: {augment_factor}x")
        print(f"Augmented training samples: {len(X_train)}")

    # --------------------------------------------------
    # SAVE TEST DATA
    # --------------------------------------------------

    np.savez(
        TEST_DATA_PATH,
        X_test=X_test,
        y_test=y_test
    )

    print()
    print(f"Test data saved: {TEST_DATA_PATH}")

    # --------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(63,)
        ),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.3
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.2
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    # --------------------------------------------------
    # COMPILE MODEL
    # --------------------------------------------------

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # --------------------------------------------------
    # DISPLAY MODEL
    # --------------------------------------------------

    print()
    print("Model architecture:")
    model.summary()

    # --------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    callbacks = [early_stopping, model_checkpoint]

    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("  TRAINING STARTED")
    print("=" * 60)
    print()

    history = model.fit(
        X_train,
        y_train,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_split=0.20,
        callbacks=callbacks,
        verbose=1
    )

    # --------------------------------------------------
    # EVALUATE ON UNSEEN TEST DATA
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("  TESTING ON UNSEEN DATA")
    print("=" * 60)
    print()

    # Load best model
    model = tf.keras.models.load_model(MODEL_PATH)

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Additional accuracy check
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"Verified Accuracy: {accuracy * 100:.2f}%")

    # --------------------------------------------------
    # SAVE LABELS
    # --------------------------------------------------

    np.save(LABELS_PATH, labels)
    print()
    print(f"Labels saved: {LABELS_PATH}")

    # --------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------

    history_dict = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }

    with open(HISTORY_PATH, "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"History saved: {HISTORY_PATH}")

    # --------------------------------------------------
    # GENERATE ACCURACY PLOT
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))
    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy",
        linewidth=2
    )
    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2
    )
    plt.title(
        "ISL Model - Training & Validation Accuracy",
        fontsize=14
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT, dpi=300)
    plt.close()

    print(f"Accuracy plot saved: {ACCURACY_PLOT}")

    # --------------------------------------------------
    # GENERATE LOSS PLOT
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))
    plt.plot(
        history.history["loss"],
        label="Training Loss",
        linewidth=2
    )
    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
        linewidth=2
    )
    plt.title(
        "ISL Model - Training & Validation Loss",
        fontsize=14
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT, dpi=300)
    plt.close()

    print(f"Loss plot saved: {LOSS_PLOT}")

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print()
    print(f"Classes:             {num_classes}")
    print(f"Training samples:    {len(X_train)}")
    print(f"Testing samples:     {len(X_test)}")
    print(f"Best val accuracy:   {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"Test accuracy:       {test_accuracy * 100:.2f}%")
    print(f"Epochs trained:      {len(history.history['loss'])}")
    print()
    print(f"Model saved:         {MODEL_PATH}")
    print(f"Labels saved:        {LABELS_PATH}")
    print(f"Test data saved:     {TEST_DATA_PATH}")
    print()
    print("Next step: python src/evaluate_model.py")
    print()


if __name__ == "__main__":
    main()