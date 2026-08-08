"""
Backend Flask cho demo web: vẽ tay 1 chữ số -> model dự đoán realtime.

Cách chạy:
    pip install flask pillow
    python app.py
Sau đó mở trình duyệt: http://127.0.0.1:5000
"""

import base64
import io
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image

from model import MnistCNN

app = Flask(__name__)

MODEL_PATH = "outputs/best_model.pth"
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# Model chỉ có 10 lớp (0-9), không có khái niệm "không phải chữ số" - nên
# với ảnh có lẫn chữ cái/ký tự khác, model vẫn ép ra 1 số với 1 độ tin cậy
# nào đó. Ngưỡng này lọc bớt các dự đoán quá thiếu chắc chắn (nhiều khả
# năng không phải chữ số thật). KHÔNG loại bỏ hoàn toàn được vấn đề, vì
# vài ký tự có hình dạng trùng chữ số thật sự (vd. "S"≈5, "O"≈0) vẫn có
# thể được dự đoán với confidence cao - đây là giới hạn cố hữu của bài
# toán phân loại 10 lớp, không phải lỗi code.
CONFIDENCE_THRESHOLD = 60.0

device = torch.device("cpu")  # inference 1 ảnh nhỏ, CPU đủ nhanh, không cần MPS/CUDA
model = MnistCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Đã load model từ {MODEL_PATH}, sẵn sàng dự đoán.")


def preprocess_canvas_image(data_url: str) -> torch.Tensor:
    """
    Chuyển ảnh canvas (nét trắng, nền đen, do người dùng vẽ) thành tensor
    28x28 chuẩn hóa giống hệt cách MNIST gốc được xử lý:
      1. Decode base64 -> ảnh grayscale
      2. Tìm bounding box của nét vẽ, crop sát
      3. Resize phần nét vẽ về 20x20 (giữ tỉ lệ), dán vào giữa khung 28x28
         (đây chính là cách bộ MNIST gốc được tạo - digit được center theo
         trọng tâm khối lượng trong khung 28x28, ở đây mình dùng center theo
         bounding box - đơn giản hơn nhưng vẫn cho kết quả tốt)
      4. Chuẩn hóa theo mean/std của MNIST
    """
    # data_url dạng "data:image/png;base64,xxxxx"
    img_data = re.sub("^data:image/.+;base64,", "", data_url)
    img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("L")

    arr = np.array(img)

    # Nếu không vẽ gì (toàn đen) -> trả về ảnh rỗng, tránh lỗi chia cho 0
    if arr.max() == 0:
        empty = torch.zeros(1, 1, 28, 28)
        return empty

    # Tìm bounding box của vùng có nét vẽ (pixel sáng > ngưỡng)
    coords = np.argwhere(arr > 20)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    digit = img.crop((x0, y0, x1, y1))

    # Resize phần nét vẽ về tối đa 20x20, giữ tỉ lệ khung hình
    w, h = digit.size
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    digit = digit.resize((new_w, new_h), Image.LANCZOS)

    # Dán vào giữa khung 28x28 nền đen (giống format gốc của MNIST)
    canvas = Image.new("L", (28, 28), color=0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas.paste(digit, (paste_x, paste_y))

    # Chuẩn hóa: [0,255] -> [0,1] -> (x - mean) / std
    tensor = torch.from_numpy(np.array(canvas)).float() / 255.0
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # -> shape [1, 1, 28, 28]

    return tensor


def load_and_normalize_image(file_bytes: bytes):
    """
    Decode ảnh upload (jpg/png/...) -> (ảnh màu gốc BGR, ảnh xám đã chuẩn hóa hướng sáng/tối).

    Ảnh chụp giấy viết tay thường là nền sáng - chữ tối (ngược với format
    MNIST/canvas: nét sáng - nền tối), nên nếu phát hiện nền sáng thì đảo màu.
    """
    arr = np.frombuffer(file_bytes, np.uint8)
    img_color = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_color is None:
        return None, None

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 127:  # nền sáng chiếm đa số -> đảo màu
        gray = 255 - gray

    return img_color, gray


def normalize_gray_image(gray: np.ndarray) -> np.ndarray:
    """Chuẩn hóa ảnh xám để giảm nhiễu, cân bằng độ sáng và giữ nét chữ rõ hơn."""
    gray = gray.astype(np.uint8)

    # Nếu nền sáng chiếm đa số, đảo ảnh để chuyển về quy ước MNIST: chữ nét đậm trên nền tối.
    if np.mean(gray) > 127:
        gray = 255 - gray

    gray = cv2.medianBlur(gray, 3)

    # Tăng độ tương phản cục bộ để xử lý nền chụp mờ hoặc ảnh thiếu sáng.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Cân bằng lại độ sáng toàn cục nhưng tránh làm vỡ nét chữ.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def segment_digits(
    gray: np.ndarray,
    min_area_ratio: float = 0.00005,
    min_height_ratio: float = 0.008,
    max_area_ratio: float = 0.9,
    max_aspect_ratio: float = 4.0,
    min_extent: float = 0.03,
    min_solidity: float = 0.08,
):
    """Tách vùng chữ số cho cả ảnh cầm tay nhỏ và ảnh chụp thực tế. Mục tiêu là không lọc quá chặt như các phiên bản trước."""
    gray = normalize_gray_image(gray)
    img_h, img_w = gray.shape
    img_area = img_h * img_w

    def _collect_boxes(mask: np.ndarray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= 0:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            area_bbox = w * h
            if area_bbox <= 0:
                continue
            if w < 2 or h < 2:
                continue
            if area_bbox < img_area * min_area_ratio:
                continue
            if h < img_h * min_height_ratio:
                continue
            if area_bbox > img_area * max_area_ratio:
                continue
            if w / h > max_aspect_ratio:
                continue
            if h / w > 6.0:
                continue
            if area / area_bbox < min_extent:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0 and area / hull_area < min_solidity:
                continue

            boxes.append((x, y, w, h))
        return boxes

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    masks = []

    # Dùng nhiều ngưỡng để giữ trường hợp hình chụp có chữ rất nhòe hoặc chữ sáng/đậm khác nhau.
    thresholds = [
        int(np.clip(np.percentile(blurred, 10), 20, 140)),
        int(np.clip(np.percentile(blurred, 25), 30, 170)),
        int(np.clip(np.percentile(blurred, 40), 40, 200)),
        int(np.clip(np.percentile(blurred, 60), 60, 220)),
        128,
    ]

    for t in sorted(set(thresholds)):
        _, dark_on_light = cv2.threshold(blurred, t, 255, cv2.THRESH_BINARY_INV)
        _, light_on_dark = cv2.threshold(blurred, t, 255, cv2.THRESH_BINARY)
        masks.extend([dark_on_light, light_on_dark])

    adaptive_inv = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=31, C=8,
    )
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31, C=8,
    )
    masks.extend([adaptive_inv, adaptive])

    boxes = []
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        boxes.extend(_collect_boxes(mask))

    if not boxes:
        return []

    # Gộp các box chồng nhau nhưng không loại quá nhiều trường hợp.
    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    merged = []
    for box in boxes:
        x, y, w, h = box
        placed = False
        for i, (mx, my, mw, mh) in enumerate(merged):
            if x < mx + mw and x + w > mx and y < my + mh and y + h > my:
                if w * h > mw * mh:
                    merged[i] = (x, y, w, h)
                placed = True
                break
        if not placed:
            merged.append(box)

    segments = []
    for (x, y, w, h) in merged:
        pad = max(2, int(0.15 * max(w, h)))
        y0, y1 = max(0, y - pad), min(img_h, y + h + pad)
        x0, x1 = max(0, x - pad), min(img_w, x + w + pad)

        crop_gray = gray[y0:y1, x0:x1]
        crop_mask = np.zeros_like(crop_gray, dtype=np.uint8)
        crop_mask[max(0, y - y0):max(0, y - y0) + h, max(0, x - x0):max(0, x - x0) + w] = 255
        crop = cv2.bitwise_and(crop_gray, crop_mask)

        if crop.size == 0 or cv2.countNonZero(crop_mask) < 2:
            continue

        segments.append({"bbox": (x, y, w, h), "crop": crop})

    return sorted(segments, key=lambda s: s["bbox"][0])


def preprocess_digit_crop(crop: np.ndarray) -> torch.Tensor:
    """Center 1 chữ số đã crop (ảnh xám, nét sáng/nền tối) vào khung 28x28,
    dùng đúng logic resize-về-20x20-rồi-dán-giữa như preprocess_canvas_image,
    sau đó chuẩn hóa theo mean/std MNIST."""
    h, w = crop.shape
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized

    tensor = torch.from_numpy(canvas).float() / 255.0
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    return tensor.unsqueeze(0).unsqueeze(0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_tensor = preprocess_canvas_image(data["image"]).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_digit = int(np.argmax(probs))
    probabilities = {str(i): round(float(p) * 100, 2) for i, p in enumerate(probs)}

    return jsonify({
        "prediction": predicted_digit,
        "confidence": round(float(probs[predicted_digit]) * 100, 2),
        "probabilities": probabilities,
    })


@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "Không nhận được file ảnh"}), 400

    file = request.files["image"]
    img_color, gray = load_and_normalize_image(file.read())
    if gray is None:
        return jsonify({"error": "Không đọc được ảnh, hãy thử file khác"}), 400

    segments = segment_digits(gray)
    if not segments:
        return jsonify({"error": "Không tìm thấy chữ số nào trong ảnh"}), 200

    digits_result = []
    annotated = img_color.copy()
    box_color = (255, 140, 79)  # BGR - tương ứng #4f8cff (màu accent trên UI)
    rejected_color = (120, 120, 120)  # xám - box bị loại vì confidence thấp

    for seg in segments:
        x, y, w, h = seg["bbox"]
        tensor = preprocess_digit_crop(seg["crop"]).to(device)

        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
        digit = int(np.argmax(probs))
        conf = round(float(probs[digit]) * 100, 2)

        if conf < CONFIDENCE_THRESHOLD:
            # Không đủ tin cậy để tính là chữ số - vẫn vẽ khung xám để biết
            # vùng này đã bị phát hiện nhưng bị loại, không lặng lẽ bỏ qua
            cv2.rectangle(annotated, (x, y), (x + w, y + h), rejected_color, 2)
            cv2.putText(annotated, "?", (x + 5, y - 8 if y > 20 else y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, rejected_color, 2)
            continue

        digits_result.append({
            "digit": digit,
            "confidence": conf,
            "bbox": [int(x), int(y), int(w), int(h)],
        })

        # Vẽ khung + nhãn dự đoán (kèm độ tin cậy) đè lên ảnh gốc
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 3)
        label = f"{digit} - {conf:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        label_y = y - 10 if y - 10 - th > 0 else y + h + th + 10
        cv2.rectangle(annotated, (x, label_y - th - 6), (x + tw + 10, label_y + 6), box_color, -1)
        cv2.putText(annotated, label, (x + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    _, buf = cv2.imencode(".png", annotated)
    annotated_b64 = "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "digits": digits_result,
        "number_string": "".join(str(d["digit"]) for d in digits_result),
        "annotated_image": annotated_b64,
    })


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """
    PHẦN 2 (webcam real-time): nhận 1 frame từ webcam (base64 data URL, giống
    format canvas vẽ tay), tái sử dụng đúng pipeline tách chữ số của
    /predict_image (load_and_normalize_image, segment_digits,
    preprocess_digit_crop), nhưng CHỈ trả JSON tọa độ khung + số dự đoán
    (không encode + trả ảnh annotated) để giữ tốc độ phản hồi nhanh, vì
    endpoint này sẽ bị gọi lặp lại liên tục (nhiều lần/giây) từ Phần 3.

    Việc vẽ khung lên video sẽ do frontend tự làm trên canvas overlay
    (xem Phần 3), dựa theo tọa độ bbox trả về ở đây — bbox này tính theo
    đúng kích thước ảnh frame gửi lên, nên khớp trực tiếp với canvas overlay
    (đã set cùng kích thước video.videoWidth/videoHeight ở Phần 1).
    """
    data = request.get_json()
    data_url = data.get("image", "") if data else ""

    try:
        img_data = re.sub("^data:image/.+;base64,", "", data_url)
        img_bytes = base64.b64decode(img_data)
        if not img_bytes:
            raise ValueError("empty image bytes")
        img_color, gray = load_and_normalize_image(img_bytes)
    except Exception:
        # Frame lỗi/rỗng (thường gặp lúc mới bật camera) -> trả về rỗng,
        # KHÔNG để lỗi làm crash server, vì endpoint này bị gọi liên tục.
        return jsonify({"digits": []})

    if gray is None:
        return jsonify({"digits": []})

    segments = segment_digits(gray)

    digits_result = []
    for seg in segments:
        x, y, w, h = seg["bbox"]
        tensor = preprocess_digit_crop(seg["crop"]).to(device)

        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()

        digit = int(np.argmax(probs))
        conf = round(float(probs[digit]) * 100, 2)

        if conf < CONFIDENCE_THRESHOLD:
            continue  # không đủ tin cậy - bỏ qua, tránh nhấp nháy số sai trên overlay

        digits_result.append({
            "digit": digit,
            "confidence": conf,
            "bbox": [int(x), int(y), int(w), int(h)],
        })

    return jsonify({"digits": digits_result})


if __name__ == "__main__":
    app.run(debug=True, port=5000)