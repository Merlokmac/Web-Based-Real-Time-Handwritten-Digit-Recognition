# Project III - Phân loại ảnh MNIST bằng CNN (PyTorch)

## 1. Cài đặt

```bash
cd mnist_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Chạy huấn luyện

```bash
python train.py
```

Lần chạy đầu sẽ tự tải bộ dữ liệu MNIST về thư mục `./data` (~10-15MB, cần internet).

Tùy chỉnh tham số:
```bash
python train.py --epochs 15 --batch-size 64 --lr 0.0005
```

## 3. Kết quả xuất ra (thư mục `./outputs`)

- `best_model.pth` - trọng số mô hình đạt test accuracy cao nhất
- `training_curves.png` - biểu đồ loss & accuracy theo epoch (train vs test)
- `confusion_matrix.png` - ma trận nhầm lẫn trên tập test
- `classification_report.txt` - precision/recall/F1-score từng lớp (0-9)
- `training_log.txt` - log chi tiết toàn bộ quá trình train

Các file này dùng trực tiếp cho phần "Kết quả" trong báo cáo.

## 4. Kiến trúc mô hình (xem chi tiết trong `model.py`)

CNN 2 khối tích chập (mỗi khối: 2x Conv-BN-ReLU + MaxPool) + 2 lớp Fully Connected,
có Dropout chống overfitting. Tổng ~468K tham số. Kỳ vọng đạt ~99% accuracy trên
test set sau 10 epoch.

## 5. Lỗi thường gặp

**`RuntimeError: Numpy is not available`** hoặc cảnh báo `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`:
Bản `torch` bạn cài chưa tương thích ABI với NumPy 2.x. Đã fix bằng cách ghim `numpy<2` trong `requirements.txt`. Nếu vẫn gặp (ví dụ cài lại từ trước), chạy:
```bash
pip install "numpy<2" --force-reinstall
```

## 6. Web demo: vẽ tay dự đoán realtime (hướng phát triển)

Chạy sau khi đã train xong và có file `outputs/best_model.pth`:

```bash
python app.py
```

Mở trình duyệt: `http://127.0.0.1:5000`. Vẽ 1 chữ số bằng chuột/trackpad (hoặc chạm nếu bạn mở trên điện thoại cùng mạng LAN), bấm "Dự đoán" để xem kết quả + độ tin cậy từng lớp.

**Ghi chú kỹ thuật (hữu ích cho báo cáo/bảo vệ):**
- Ảnh vẽ trên canvas được crop theo bounding box, resize về tối đa 20x20 rồi
  đặt giữa khung 28x28 nền đen — đây chính là cách dữ liệu MNIST gốc được xử lý
  (digit luôn được center trong khung 28x28), giúp tăng độ chính xác dự đoán
  đáng kể so với việc chỉ resize thô cả canvas 280x280 xuống 28x28.
- Ảnh sau đó được chuẩn hóa với cùng mean/std (0.1307, 0.3081) như lúc train,
  đảm bảo phân phối dữ liệu đầu vào giống hệt lúc huấn luyện.

## 7. Cấu trúc file

```
mnist_project/
├── model.py          # định nghĩa kiến trúc CNN
├── train.py           # script train + evaluate + xuất biểu đồ
├── app.py              # backend Flask cho web demo vẽ tay
├── templates/
│   └── index.html      # giao diện canvas vẽ tay + hiển thị kết quả
├── outputs/             # (tự tạo sau khi train) chứa best_model.pth, biểu đồ...
├── requirements.txt
└── README.md
```
