"""
Kiến trúc CNN cơ bản cho bài toán phân loại ảnh MNIST.

Kiến trúc (dạng LeNet cải tiến, thêm BatchNorm + Dropout):
    Input: 1x28x28
    -> Conv2d(1, 32, kernel=3, padding=1) -> BatchNorm -> ReLU
    -> Conv2d(32, 32, kernel=3, padding=1) -> BatchNorm -> ReLU
    -> MaxPool2d(2)                                    # 28x28 -> 14x14
    -> Conv2d(32, 64, kernel=3, padding=1) -> BatchNorm -> ReLU
    -> Conv2d(64, 64, kernel=3, padding=1) -> BatchNorm -> ReLU
    -> MaxPool2d(2)                                    # 14x14 -> 7x7
    -> Flatten -> Dropout(0.5)
    -> Linear(64*7*7, 128) -> ReLU -> Dropout(0.3)
    -> Linear(128, 10)                                  # 10 lớp (chữ số 0-9)
"""

import torch
import torch.nn as nn


class MnistCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Khối tích chập 1: trích xuất đặc trưng mức thấp (nét, cạnh)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
        )

        # Khối tích chập 2: trích xuất đặc trưng mức cao hơn (hình dạng số)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
        )

        # Phần phân loại (fully connected)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x  # logits, chưa qua softmax (dùng với CrossEntropyLoss)

    def count_parameters(self) -> dict:
        """Đếm số tham số theo từng khối - hữu ích cho báo cáo."""
        counts = {}
        for name, module in [
            ("conv_block1", self.conv_block1),
            ("conv_block2", self.conv_block2),
            ("classifier", self.classifier),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts["total"] = sum(counts.values())
        return counts


if __name__ == "__main__":
    # Sanity check nhanh: kiểm tra shape đầu ra và số tham số
    model = MnistCNN()
    dummy_input = torch.randn(4, 1, 28, 28)  # batch=4 ảnh MNIST giả
    output = model(dummy_input)
    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)  # kỳ vọng [4, 10]
    print("Số tham số theo khối:")
    for k, v in model.count_parameters().items():
        print(f"  {k}: {v:,}")
