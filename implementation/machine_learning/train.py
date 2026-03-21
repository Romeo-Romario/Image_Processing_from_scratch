import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

# Import the model architecture
from custom_cnn import UkrainianOCRResNet

# --- CONFIGURATION ---
DATASET_DIR = r"D:\Source\Diplom\tryouts\tryout2_image_deskweing\implementation\machine_learning\dataset"
RUNS_DIR = r"runs"

BATCH_SIZE = 128
EPOCHS = 10  # Increased to 10 for Data Augmentation
LEARNING_RATE = 0.0005


class ExperimentTracker:
    def __init__(self, base_dir=RUNS_DIR):
        os.makedirs(base_dir, exist_ok=True)
        existing_runs = [d for d in os.listdir(base_dir) if d.startswith("exp_")]
        self.run_id = f"exp_{len(existing_runs) + 1:03d}"
        self.run_dir = os.path.join(base_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1_macro": [],
        }
        print(f"Tracking experiment in: {self.run_dir}")

    def log_epoch(self, t_loss, v_loss, t_acc, v_acc, v_f1):
        self.history["train_loss"].append(t_loss)
        self.history["val_loss"].append(v_loss)
        self.history["train_acc"].append(t_acc)
        self.history["val_acc"].append(v_acc)
        self.history["val_f1_macro"].append(v_f1)

    def save_run(self, model, class_names, final_report_dict):
        torch.save(model.state_dict(), os.path.join(self.run_dir, "model_weights.pth"))

        with open(
            os.path.join(self.run_dir, "class_mapping.txt"), "w", encoding="utf-8"
        ) as f:
            for idx, name in enumerate(class_names):
                f.write(f"{idx}:{name}\n")

        report = {
            "hyperparameters": {
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "augmentation": True,
                "weight_dampening": "Square Root",
            },
            "history": self.history,
            "final_metrics": final_report_dict,
        }
        with open(
            os.path.join(self.run_dir, "metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(report, f, indent=4)

        self._plot_curves()

    def _plot_curves(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label="Train Loss", marker="o")
        plt.plot(epochs, self.history["val_loss"], label="Validation Loss", marker="o")
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["train_acc"], label="Train Acc", marker="o")
        plt.plot(epochs, self.history["val_acc"], label="Validation Acc", marker="o")
        plt.plot(
            epochs,
            self.history["val_f1_macro"],
            label="Val F1 (Macro)",
            marker="x",
            linestyle="--",
        )
        plt.title("Accuracy & F1 Score")
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "learning_curves.png"))
        plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    tracker = ExperimentTracker()

    # --- 1. SEPARATE TRANSFORMATIONS ---
    # Training data gets distorted to artificially multiply our rare number/punctuation classes
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.RandomRotation(degrees=10),  # Tilt up to 10 degrees
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)
            ),  # Shift and Zoom 10%
            transforms.ToTensor(),
        ]
    )

    # Validation data stays perfect
    val_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )

    print("Loading dataset and splitting...")
    # Load the same folders twice, but with different transformation rules attached
    train_dataset_full = datasets.ImageFolder(
        root=DATASET_DIR, transform=train_transform
    )
    val_dataset_full = datasets.ImageFolder(root=DATASET_DIR, transform=val_transform)

    num_classes = len(train_dataset_full.classes)
    class_names = train_dataset_full.classes

    # --- 2. ADVANCED TRAIN/VAL SPLIT ---
    # We must ensure the exact same images go to Train and Val, just with different transforms
    num_samples = len(train_dataset_full)
    indices = torch.randperm(num_samples).tolist()
    train_size = int(0.8 * num_samples)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    # Apply the split
    train_dataset = Subset(train_dataset_full, train_idx)
    val_dataset = Subset(val_dataset_full, val_idx)

    # --- 3. DAMPENED CLASS WEIGHTS ---
    targets = train_dataset_full.targets
    class_counts = np.bincount(targets)

    # Apply the Square Root function to dampen the extreme weights!
    raw_weights = len(targets) / (num_classes * class_counts)
    dampened_weights = np.sqrt(raw_weights)
    class_weights_tensor = torch.FloatTensor(dampened_weights).to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = UkrainianOCRResNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )

    print("\nStarting Training (10 Epochs with Augmentation)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        # --- TRAINING ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        t_loss = running_loss / len(train_loader)
        t_acc = 100 * correct_train / total_train

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        v_loss = val_loss / len(val_loader)
        v_acc = 100 * correct_val / total_val
        v_f1 = f1_score(all_labels, all_preds, average="macro") * 100

        tracker.log_epoch(t_loss, v_loss, t_acc, v_acc, v_f1)

        scheduler.step(v_loss)

        # Grab the current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | LR: {current_lr:.6f} | "
            f"T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f} | "
            f"T-Acc: {t_acc:.2f}% | V-Acc: {v_acc:.2f}% | V-F1: {v_f1:.2f}%"
        )

    # --- FINAL REPORTING ---
    print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes.")
    final_report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    tracker.save_run(model, class_names, final_report_dict)
    print(f"Experiment saved successfully in {tracker.run_dir}!")


if __name__ == "__main__":
    main()
