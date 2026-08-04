"""
Script huấn luyện và đánh giá CNN trên MNIST.

Cách chạy:
    python train.py                      # cấu hình mặc định
    python train.py --epochs 15 --batch-size 64 --lr 0.0005

Kết quả sau khi chạy xong (lưu trong thư mục ./outputs):
    - best_model.pth        : trọng số mô hình tốt nhất (theo test accuracy)
    - training_curves.png   : biểu đồ loss/accuracy theo epoch
    - confusion_matrix.png  : ma trận nhầm lẫn trên tập test
    - classification_report.txt : precision/recall/f1 từng lớp
    - training_log.txt      : log chi tiết từng epoch
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")  # không cần hiển thị màn hình, chỉ lưu file
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from model import MnistCNN


def get_device() -> torch.device:
    """Tự động chọn thiết bị tốt nhất hiện có: CUDA > MPS (Mac) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dataloaders(data_dir: str, batch_size: int):
    # Chuẩn hóa theo mean/std chuẩn của MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


def plot_curves(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["test_loss"], label="Test Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss theo epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    axes[1].plot(epochs, history["test_acc"], label="Test Accuracy", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy theo epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - MNIST Test Set")

    for i in range(10):
        for j in range(10):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train CNN trên MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Sử dụng thiết bị: {device}")

    train_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)
    print(f"Số ảnh train: {len(train_loader.dataset)} | Số ảnh test: {len(test_loader.dataset)}")

    model = MnistCNN().to(device)
    param_counts = model.count_parameters()
    print("Số tham số theo khối:", param_counts)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    log_lines = []

    log_lines.append(f"Thiết bị: {device}")
    log_lines.append(f"Kiến trúc - số tham số: {param_counts}")
    log_lines.append(f"Cấu hình: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    log_lines.append("-" * 60)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        epoch_time = time.time() - epoch_start
        line = (f"Epoch {epoch:2d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
                f"{epoch_time:.1f}s")
        print(line)
        log_lines.append(line)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), out_dir / "best_model.pth")

    total_time = time.time() - start_time
    log_lines.append("-" * 60)
    log_lines.append(f"Tổng thời gian huấn luyện: {total_time:.1f}s")
    log_lines.append(f"Best test accuracy: {best_acc:.4f}")

    # Đánh giá lần cuối bằng model tốt nhất để lấy confusion matrix + classification report
    model.load_state_dict(torch.load(out_dir / "best_model.pth", map_location=device))
    _, final_acc, final_preds, final_labels = evaluate(model, test_loader, criterion, device)

    report = classification_report(final_labels, final_preds, digits=4)
    print("\nClassification Report:\n", report)
    log_lines.append("\nClassification Report:\n" + report)

    (out_dir / "training_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    plot_curves(history, out_dir / "training_curves.png")
    plot_confusion_matrix(final_labels, final_preds, out_dir / "confusion_matrix.png")

    print(f"\nHoàn tất! Kết quả đã lưu tại: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
