# Phân loại ảnh MNIST bằng mạng nơ ron tích chập (CNN) với PyTorch

## 1. Giới thiệu bài toán

Bài toán phân loại ảnh MNIST là một bài toán cơ bản và quan trọng trong lĩnh vực thị giác máy tính và học sâu. Bộ dữ liệu MNIST bao gồm các ảnh đen trắng kích thước 28x28 pixel, tương ứng với 10 lớp chữ số từ 0 đến 9. Nhiệm vụ của mô hình là dự đoán nhãn của từng ảnh đầu vào.

MNIST được coi là bộ dữ liệu chuẩn để kiểm tra hiệu quả của các mô hình học sâu vì:
- kích thước dữ liệu vừa phải và dễ quản lý
- hình ảnh đơn giản, rõ nét và đồng nhất
- phù hợp để đánh giá kiến trúc CNN, độ chính xác và khả năng tổng quát hóa

Trong dự án này, chúng tôi xây dựng một mô hình CNN để phân loại các chữ số 0-9, huấn luyện trên tập dữ liệu MNIST và đánh giá trên tập kiểm tra. Mô hình không chỉ đạt độ chính xác cao trên tập test mà còn được tích hợp vào một ứng dụng web cho phép người dùng vẽ tay và nhận dự đoán trực tiếp.

---

## 2. Bộ dữ liệu MNIST

Bộ dữ liệu MNIST được lấy từ kho dữ liệu chuẩn của Yann LeCun và đồng nghiệp, gồm:
- 60.000 ảnh huấn luyện
- 10.000 ảnh kiểm tra
- Mỗi ảnh là ảnh xám kích thước 28x28 pixel
- Mỗi ảnh tương ứng với một nhãn từ 0 đến 9

Trong quá trình huấn luyện, dữ liệu được xử lý theo chuẩn như sau:
- chuyển ảnh từ kiểu giá trị pixel [0,255] sang [0,1]
- chuẩn hóa theo mean và std của MNIST: mean = 0.1307, std = 0.3081

Công thức chuẩn hóa:

x_norm = (x - 0.1307) / 0.3081

Việc chuẩn hóa này giúp làm giảm độ lệch về sáng tối và tăng tốc độ hội tụ của mạng trong quá trình huấn luyện.

---

## 3. Ứng dụng học sâu được lựa chọn

Đối với bài toán phân loại ảnh, mạng nơ ron tích chập (Convolutional Neural Network - CNN) là lựa chọn phù hợp nhất. Lý do là:
- CNN có khả năng học các đặc trưng không gian như cạnh, góc, vòng tròn, đường cong của chữ số
- CNN giảm số lượng tham số so với mạng fully connected truyền thống
- CNN rất hiệu quả trong bài toán nhận dạng hình ảnh và xử lý dữ liệu 2D

Trong dự án này, mô hình CNN được triển khai bằng PyTorch vì PyTorch hỗ trợ linh hoạt trong việc xây dựng, huấn luyện, lưu trọng số và triển khai mô hình vào ứng dụng thực tế.

---

## 4. Kiến trúc mạng nơ ron tích chập được lựa chọn

Mô hình được xây dựng theo kiểu CNN 2 khối tích chập, kết hợp với các lớp chuẩn hóa BatchNorm, ReLU và Dropout để tăng độ ổn định và giảm hiện tượng overfitting.

### 4.1. Cấu trúc chi tiết

Input: 1 x 28 x 28

- Khối tích chập 1:
  - Conv2d(1, 32, kernel_size=3, padding=1)
  - BatchNorm2d(32)
  - ReLU
  - Conv2d(32, 32, kernel_size=3, padding=1)
  - BatchNorm2d(32)
  - ReLU
  - MaxPool2d(2)

- Khối tích chập 2:
  - Conv2d(32, 64, kernel_size=3, padding=1)
  - BatchNorm2d(64)
  - ReLU
  - Conv2d(64, 64, kernel_size=3, padding=1)
  - BatchNorm2d(64)
  - ReLU
  - MaxPool2d(2)

- Phần phân loại:
  - Flatten
  - Dropout(0.5)
  - Linear(64 x 7 x 7, 128)
  - ReLU
  - Dropout(0.3)
  - Linear(128, 10)

### 4.2. Số lượng tham số

Theo log huấn luyện thực tế, mô hình có tổng số tham số là:
- conv_block1: 9,696
- conv_block2: 55,680
- classifier: 402,826
- total: 468,202

Số lượng này vừa đủ để biểu diễn đặc trưng của chữ số mà không quá lớn, giúp mô hình hội tụ nhanh và độ chính xác cao.

---

## 5. Thiết lập huấn luyện và phần cứng

Trong quá trình huấn luyện, mô hình được chạy trên thiết bị có sẵn, ưu tiên theo thứ tự:
- CUDA nếu máy có GPU NVIDIA
- MPS nếu chạy trên MacBook Apple Silicon
- CPU nếu không có GPU

Kết quả log thu được cho thấy mô hình đang chạy trên nền tảng MPS (Apple Metal Performance Shaders), không phải GPU NVIDIA. Đây là một lựa chọn hợp lý cho môi trường MacBook.

### 5.1. Cấu hình huấn luyện

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss
- Batch size: 128
- Epochs: 10
- Seed: 42

### 5.2. Quy trình huấn luyện

Quá trình huấn luyện được thực hiện theo các bước sau:
1. Tải dữ liệu MNIST từ thư mục data
2. Chuyển đổi ảnh sang tensor và chuẩn hóa
3. Đưa batch dữ liệu lên thiết bị huấn luyện
4. Tính output của mô hình
5. Tính loss bằng CrossEntropyLoss
6. Backpropagation và cập nhật trọng số bằng Adam
7. Đánh giá trên tập test sau mỗi epoch
8. Lưu mô hình có độ chính xác tốt nhất vào outputs/best_model.pth

---

## 6. Kết quả thực nghiệm

Dựa trên log huấn luyện và báo cáo đánh giá thực tế, mô hình đạt kết quả rất tốt trên tập kiểm tra.

### 6.1. Độ chính xác theo epoch

Theo file log, kết quả tốt nhất đạt được là:
- Best test accuracy: 0.9948

Kết quả này cho thấy mô hình đạt khoảng 99.48% độ chính xác trên bộ dữ liệu MNIST test set, mức rất cao cho bài toán phân loại chữ số.

### 6.2. Báo cáo phân loại rõ ràng

Bảng đánh giá theo từng lớp trong file classification_report.txt cho thấy:

- class 0: precision 0.9980, recall 0.9990, f1-score 0.9985
- class 1: precision 0.9947, recall 0.9965, f1-score 0.9956
- class 2: precision 0.9952, recall 0.9981, f1-score 0.9966
- class 3: precision 0.9921, recall 0.9970, f1-score 0.9946
- class 4: precision 0.9959, recall 0.9949, f1-score 0.9954
- class 5: precision 0.9989, recall 0.9910, f1-score 0.9949
- class 6: precision 0.9989, recall 0.9896, f1-score 0.9942
- class 7: precision 0.9913, recall 0.9932, f1-score 0.9922
- class 8: precision 0.9938, recall 0.9949, f1-score 0.9944
- class 9: precision 0.9901, recall 0.9931, f1-score 0.9916

### 6.3. Chỉ số tổng hợp

- Accuracy: 0.9948
- Macro avg F1-score: 0.9948
- Weighted avg F1-score: 0.9948

Đây là kết quả rất ấn tượng, thể hiện mô hình học được các đặc trưng quan trọng của chữ số và tổng quát hóa tốt trên tập test.

---

## 7. Đánh giá và nhận xét

Mô hình CNN này phù hợp với bài toán phân loại chữ số MNIST vì các lý do sau:
- dữ liệu là ảnh có cấu trúc không gian rõ ràng
- CNN học được đặc trưng cục bộ và hình dạng của từng chữ số
- khối tích chập liên tiếp giúp trích xuất cả đặc trưng mức thấp và mức cao
- BatchNorm và Dropout giúp tăng độ ổn định và giảm overfitting

Nhìn chung, mô hình này là lựa chọn tối ưu cho một bài toán phân loại ảnh cơ bản nhưng hiệu quả, dễ triển khai và dễ hiểu.

---

## 8. Ứng dụng demo trên web

Ngoài việc huấn luyện và đánh giá, dự án còn tích hợp một demo web để người dùng có thể vẽ tay chữ số và xem dự đoán trực tiếp trên giao diện. Ứng dụng này được xây dựng bằng Flask và Python.

### Chức năng chính
- người dùng vẽ chữ số trên canvas
- hệ thống xử lý ảnh đầu vào
- ảnh được chuẩn hóa giống dữ liệu MNIST
- mô hình dự đoán nhãn tương ứng
- hiển thị kết quả và độ tin cậy

Điểm đáng chú ý là ảnh đầu vào được xử lý theo đúng logic tương tự MNIST: crop, resize về 20x20 rồi đặt vào khung 28x28, sau đó chuẩn hóa theo mean/std giống như khi huấn luyện. Điều này giúp tăng khả năng dự đoán chính xác trên dữ liệu vẽ tay.

---

## 9. Hướng dẫn chạy project

### 9.1. Cài đặt môi trường

```bash
cd /path/to/project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 9.2. Huấn luyện mô hình

```bash
python train.py
```

Mô hình sẽ tự động tải dữ liệu MNIST nếu chưa có trong thư mục data.

### 9.3. Chạy demo web

```bash
python app.py
```

Sau đó truy cập địa chỉ:

```text
http://127.0.0.1:5000
```

---

## 10. Cấu trúc thư mục project

```text
Project III/
├── app.py
├── model.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── MNIST/
├── outputs/
│   ├── best_model.pth
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── training_log.txt
├── templates/
│   └── index.html
└── LICENSE
```

---

## 11. Kết luận

Dự án này đã lựa chọn đúng hướng giải quyết bài toán phân loại ảnh MNIST bằng mô hình học sâu CNN. Mô hình được xây dựng rõ ràng, huấn luyện có hệ thống và đạt kết quả rất tốt trên tập kiểm tra. Đây là một dự án phù hợp cho mục tiêu học tập, nghiên cứu và demo ứng dụng máy học trong lĩnh vực thị giác máy tính.

Nếu tiếp tục phát triển, dự án có thể mở rộng sang các hướng như tăng cường dữ liệu, thử nghiệm CNN sâu hơn, so sánh với các kiến trúc khác như ResNet hoặc MobileNet, hoặc xây dựng giao diện nhận dạng ảnh thực tế trên web/mobile.

---

## 12. Tài liệu tham khảo

- LeCun, Y., Cortes, C., & Burges, C. J. C. (1998). The MNIST database of handwritten digits.
- PyTorch official documentation
- Deep Learning for Computer Vision with PyTorch
- OpenCV and Flask documentation

