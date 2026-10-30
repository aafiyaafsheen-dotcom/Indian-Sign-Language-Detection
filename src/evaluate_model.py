"""
ISL Model Evaluation Script

Evaluates the trained model on unseen test data and generates
classification report, confusion matrix, and detailed metrics.

Usage:
    python src/evaluate_model.py
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

MODEL_PATH = "models/isl_model.keras"
LABELS_PATH = "models/labels.npy"
TEST_DATA_PATH = "results/test_data.npz"
CONFUSION_MATRIX_PATH = "results/test_confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = "results/classification_report.txt"


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    os.makedirs("results", exist_ok=True)

    print()
    print("=" * 60)
    print("  ISL MODEL EVALUATION")
    print("=" * 60)
    print()

    # --------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------

    if not os.path.exists(TEST_DATA_PATH):
        print(f"ERROR: Test data not found: {TEST_DATA_PATH}")
        print("Run train_model.py first.")
        return

    data = np.load(TEST_DATA_PATH)
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Test data loaded: {TEST_DATA_PATH}")
    print(f"Test samples: {len(X_test)}")

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")

    # --------------------------------------------------
    # LOAD LABELS
    # --------------------------------------------------

    if not os.path.exists(LABELS_PATH):
        print(f"ERROR: Labels not found: {LABELS_PATH}")
        return

    labels = np.load(LABELS_PATH, allow_pickle=True)
    print(f"Classes: {labels}")
    print(f"Number of classes: {len(labels)}")

    # --------------------------------------------------
    # MAKE PREDICTIONS
    # --------------------------------------------------

    predictions = model.predict(X_test, verbose=0)
    predicted_indices = np.argmax(predictions, axis=1)

    # --------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, predicted_indices)

    precision = precision_score(
        y_test, predicted_indices,
        average="weighted", zero_division=0
    )

    recall = recall_score(
        y_test, predicted_indices,
        average="weighted", zero_division=0
    )

    f1 = f1_score(
        y_test, predicted_indices,
        average="weighted", zero_division=0
    )

    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print()

    print(f"Test Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision:      {precision * 100:.2f}%")
    print(f"Recall:         {recall * 100:.2f}%")
    print(f"F1-Score:       {f1 * 100:.2f}%")
    print(f"Test Samples:   {len(X_test)}")
    print()

    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

    report = classification_report(
        y_test,
        predicted_indices,
        target_names=labels,
        zero_division=0
    )

    print("Classification Report:")
    print()
    print(report)

    # Save classification report
    with open(CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write("INDIAN SIGN LANGUAGE - CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Accuracy:  {accuracy * 100:.2f}%\n")
        f.write(f"Precision:      {precision * 100:.2f}%\n")
        f.write(f"Recall:         {recall * 100:.2f}%\n")
        f.write(f"F1-Score:       {f1 * 100:.2f}%\n")
        f.write(f"Test Samples:   {len(X_test)}\n")
        f.write(f"Classes:        {len(labels)}\n\n")
        f.write("Per-Class Report:\n\n")
        f.write(report)

    print(f"Classification report saved: {CLASSIFICATION_REPORT_PATH}")

    # --------------------------------------------------
    # MISCLASSIFICATION ANALYSIS
    # --------------------------------------------------

    misclassified = np.where(y_test != predicted_indices)[0]
    total_misclassified = len(misclassified)

    print()
    print("-" * 60)
    print("MISCLASSIFICATION ANALYSIS")
    print("-" * 60)
    print()
    print(f"Total misclassified: {total_misclassified} / {len(y_test)}")
    print()

    if total_misclassified > 0 and total_misclassified <= 50:
        print("Misclassified samples:")
        for idx in misclassified:
            true_label = labels[y_test[idx]]
            pred_label = labels[predicted_indices[idx]]
            confidence = predictions[idx][predicted_indices[idx]]
            print(
                f"  Sample {idx}: "
                f"True={true_label}, "
                f"Predicted={pred_label}, "
                f"Confidence={confidence:.3f}"
            )
    elif total_misclassified > 50:
        # Show most confused pairs
        print("Most confused class pairs:")
        cm = confusion_matrix(y_test, predicted_indices)

        confused_pairs = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i][j] > 0:
                    confused_pairs.append(
                        (labels[i], labels[j], cm[i][j])
                    )

        confused_pairs.sort(key=lambda x: x[2], reverse=True)

        for true_l, pred_l, count in confused_pairs[:15]:
            print(f"  {true_l} -> {pred_l}: {count} times")

    print()

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(y_test, predicted_indices)

    print("Confusion Matrix:")
    print(cm)
    print()

    # --------------------------------------------------
    # PLOT CONFUSION MATRIX
    # --------------------------------------------------

    num_classes = len(labels)
    fig_size = max(8, num_classes * 0.5)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    fig, ax = plt.subplots(
        figsize=(fig_size, fig_size)
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        "Indian Sign Language - Test Confusion Matrix",
        fontsize=14,
        pad=20
    )

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    # Rotate labels for readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Confusion matrix saved: {CONFUSION_MATRIX_PATH}")

    # --------------------------------------------------
    # PER-CLASS ACCURACY
    # --------------------------------------------------

    print()
    print("-" * 60)
    print("PER-CLASS ACCURACY")
    print("-" * 60)
    print()

    for i, label in enumerate(labels):
        mask = y_test == i
        class_total = mask.sum()

        if class_total > 0:
            class_correct = (predicted_indices[mask] == i).sum()
            class_acc = class_correct / class_total * 100

            bar_length = int(class_acc / 100 * 30)
            bar = "#" * bar_length

            status = ""
            if class_acc < 80:
                status = " [NEEDS IMPROVEMENT]"
            elif class_acc < 90:
                status = " [ACCEPTABLE]"

            print(
                f"  {label}: {class_acc:>6.1f}%  "
                f"({class_correct}/{class_total})  "
                f"{bar}{status}"
            )

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print()
    print(f"Overall Accuracy:   {accuracy * 100:.2f}%")
    print(f"Weighted Precision: {precision * 100:.2f}%")
    print(f"Weighted Recall:    {recall * 100:.2f}%")
    print(f"Weighted F1-Score:  {f1 * 100:.2f}%")
    print()
    print(f"Confusion matrix:   {CONFUSION_MATRIX_PATH}")
    print(f"Classification report: {CLASSIFICATION_REPORT_PATH}")
    print()


if __name__ == "__main__":
    main()