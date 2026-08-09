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
# Ngưỡng tin cậy tối thiểu để hiển thị 1 dự đoán trong chế độ webcam real-time.
# Dưới ngưỡng này model đang "phân vân" giữa nhiều số -> bỏ qua thay vì hiển
# thị đại 1 số, tránh nhấp nháy/nhảy loạn giữa các frame.
MIN_FRAME_CONFIDENCE = 55.0

device = torch.device("cpu")  # inference 1 ảnh nhỏ, CPU đủ nhanh, không cần MPS/CUDA
model = MnistCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Đã load model từ {MODEL_PATH}, sẵn sàng dự đoán.")


def preprocess_canvas_image(data_url: str):
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

    Trả về (tensor, raw_canvas_28x28): raw_canvas là ảnh PIL 28x28 CHƯA
    chuẩn hóa (giá trị pixel gốc 0-255) - dùng làm nền để vẽ đè heatmap
    Grad-CAM lên sau này.
    """
    # data_url dạng "data:image/png;base64,xxxxx"
    img_data = re.sub("^data:image/.+;base64,", "", data_url)
    img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("L")

    arr = np.array(img)

    # Nếu không vẽ gì (toàn đen) -> trả về ảnh rỗng, tránh lỗi chia cho 0
    if arr.max() == 0:
        empty_tensor = torch.zeros(1, 1, 28, 28)
        empty_canvas = Image.new("L", (28, 28), color=0)
        return empty_tensor, empty_canvas

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

    return tensor, canvas


def get_top_k(probs: np.ndarray, k: int = 3):
    """Trả về top-k dự đoán có xác suất cao nhất, sắp xếp giảm dần."""
    top_idx = np.argsort(probs)[::-1][:k]
    return [
        {"digit": int(i), "confidence": round(float(probs[i]) * 100, 2)}
        for i in top_idx
    ]


# Lớp Conv cuối cùng của conv_block2 (sau ReLU, trước MaxPool cuối) - dùng
# làm target layer cho Grad-CAM. Đây là lớp tích chập sâu nhất, giữ được
# nhiều thông tin không gian nhất (feature map 14x14) trước khi bị nén lại
# qua MaxPool + Flatten, nên phù hợp nhất để trực quan hóa "model nhìn vào đâu".
GRADCAM_TARGET_LAYER = model.conv_block2[5]


def forward_with_gradcam(input_tensor: torch.Tensor, target_layer=GRADCAM_TARGET_LAYER):
    """
    Chạy 1 lần forward + 1 lần backward để vừa lấy được dự đoán (probs),
    vừa tính được Grad-CAM heatmap cho đúng lớp được dự đoán - không cần
    forward 2 lần.

    Grad-CAM (Selvaraju et al., 2017): heatmap = ReLU( sum_c( w_c * A_c ) )
    trong đó A_c là feature map kênh thứ c của target_layer, w_c là trọng số
    = trung bình gradient của điểm số lớp dự đoán theo A_c (global average
    pooling của gradient) - kênh nào ảnh hưởng nhiều đến quyết định của
    model thì được nhân trọng số lớn hơn.

    Trả về (probs, predicted_class, cam) với cam là ảnh xám 2D giá trị
    trong [0,1], kích thước bằng feature map của target_layer (14x14).
    """
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out

    def bwd_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad()
        logits = model(input_tensor)
        predicted_class = int(logits.argmax(dim=1).item())

        score = logits[0, predicted_class]
        score.backward()

        weights = gradients["value"].mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = torch.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().cpu().numpy()

        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max  # normalize về [0, 1]

        probs = F.softmax(logits.detach(), dim=1).squeeze(0).cpu().numpy()
    finally:
        h_fwd.remove()
        h_bwd.remove()

    return probs, predicted_class, cam


def make_gradcam_overlay(raw_digit_img: Image.Image, cam: np.ndarray, out_size: int = 140) -> str:
    """
    Vẽ heatmap Grad-CAM (giá trị [0,1], kích thước nhỏ vd. 14x14) đè lên
    ảnh chữ số gốc 28x28, phóng to lên out_size x out_size cho dễ nhìn.
    Trả về data URL PNG base64.
    """
    cam_resized = cv2.resize(cam.astype(np.float32), (28, 28), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)
    cam_uint8 = np.uint8(255 * cam_resized)

    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  # BGR, đỏ = ảnh hưởng mạnh
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    raw_rgb = np.array(raw_digit_img.convert("RGB"))
    overlay = (0.5 * raw_rgb + 0.5 * heatmap_rgb).astype(np.uint8)

    overlay_big = cv2.resize(overlay, (out_size, out_size), interpolation=cv2.INTER_CUBIC)

    buf = io.BytesIO()
    Image.fromarray(overlay_big).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


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


def segment_digits(
    gray: np.ndarray,
    min_area_ratio: float = 0.0005,
    min_height_ratio: float = 0.03,
    max_area_ratio: float = 0.5,
):
    """
    Tách từng chữ số trong ảnh xám (đã chuẩn hóa nét sáng/nền tối).

    Dùng ADAPTIVE THRESHOLD (so sánh mỗi pixel với vùng lân cận) thay vì
    Otsu toàn cục - quan trọng khi chụp giấy kẻ ô ly hoặc ánh sáng không
    đều (đặc biệt qua webcam): Otsu dễ nhận nhầm cả lưới kẻ ô thành "nét vẽ",
    hoặc làm vỡ nét bút mảnh thành nhiều mảnh rời rạc (mỗi mảnh bị coi là
    1 "chữ số" khác nhau -> bounding box nhỏ vụn, dự đoán sai/nhảy loạn).

    Trả về danh sách dict {"bbox": (x, y, w, h), "crop": ảnh xám đã crop sát
    từng số}, sắp xếp theo thứ tự trái sang phải.
    """
    # CLAHE (cân bằng histogram thích nghi cục bộ): ảnh chụp qua webcam
    # thường có ánh sáng không đều (bóng đổ, loá 1 góc, auto-exposure dao
    # động) - CLAHE làm đều độ tương phản theo từng vùng nhỏ trước khi
    # threshold, giúp adaptiveThreshold ổn định hơn giữa các frame.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # blockSize=31: kích thước vùng lân cận để tính ngưỡng cục bộ (phải là số lẻ)
    # C=-15 (offset âm): pixel phải sáng hơn trung bình vùng lân cận ít nhất
    # 15 mức xám mới được coi là "nét vẽ" - giúp loại lưới kẻ ô nhạt, giữ
    # lại nét mực đậm
    mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31, C=-15,
    )

    # Phép mở (opening): xóa các đốm nhiễu rất nhỏ/mảnh (vd. lưới kẻ ô còn sót)
    kernel_open = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Phép đóng (closing): nối liền các đoạn nét chữ bị đứt quãng do threshold
    kernel_close = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = gray.shape
    img_area = img_h * img_w

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Lọc nhiễu: đốm quá nhỏ (bụi, vết bẩn, mảnh vụn còn sót của lưới kẻ)
        if w * h < img_area * min_area_ratio:
            continue
        # Lọc theo chiều cao tối thiểu so với khung hình: loại các mảnh vụn
        # li ti, giữ lại các nét đủ lớn để thực sự là 1 chữ số
        if h < img_h * min_height_ratio:
            continue
        # An toàn: loại box chiếm gần hết khung hình (dấu hiệu threshold
        # "vỡ trận", cả khung hình dính thành 1 khối - không phải chữ số)
        if w * h > img_area * max_area_ratio:
            continue
        # Loại box chạm sát viền khung hình: thường là mép giấy, ngón tay
        # cầm giấy, hoặc vệt sáng/bóng ở rìa ảnh webcam - không phải chữ số
        border_margin = 2
        if (x <= border_margin or y <= border_margin
                or x + w >= img_w - border_margin
                or y + h >= img_h - border_margin):
            continue
        # Loại box có tỉ lệ khung hình bất thường (quá dẹt ngang/dọc) -
        # thường là nét kẻ ô ly, mép giấy, hoặc vệt nhiễu, không phải 1 chữ số
        aspect = w / h
        if aspect < 0.15 or aspect > 3.0:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])  # trái -> phải

    segments = []
    for (x, y, w, h) in boxes:
        pad = max(3, int(0.15 * max(w, h)))
        y0, y1 = max(0, y - pad), min(img_h, y + h + pad)
        x0, x1 = max(0, x - pad), min(img_w, x + w + pad)

        # Crop từ ảnh xám gốc (giữ độ mượt của nét), chỉ giữ vùng thuộc
        # mask của chính chữ số này để tránh lẫn nét từ số bên cạnh
        crop_gray = gray[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        crop = cv2.bitwise_and(crop_gray, crop_mask)

        segments.append({"bbox": (x, y, w, h), "crop": crop})

    return segments


def preprocess_digit_crop(crop: np.ndarray) -> torch.Tensor:
    """Center 1 chữ số đã crop (ảnh xám, nét sáng/nền tối) vào khung 28x28,
    dùng đúng logic resize-về-20x20-rồi-dán-giữa như preprocess_canvas_image,
    sau đó chuẩn hóa theo mean/std MNIST."""
    # Nét bút trên giấy (đặc biệt bút bi/bút chì mảnh) MẢNH hơn nhiều so với
    # nét trong MNIST gốc (vốn khá dày do cách chuẩn hoá dữ liệu gốc: co về
    # 20x20 từ ảnh nhị phân rồi lấy mẫu lại). Model học trên nét dày nên gặp
    # nét mảnh dễ đoán sai/thiếu tự tin -> dãn nét (dilate) nhẹ trước khi
    # resize để đưa độ dày nét về gần với phân bố lúc huấn luyện.
    dilate_kernel = np.ones((3, 3), np.uint8)
    crop = cv2.dilate(crop, dilate_kernel, iterations=1)

    h, w = crop.shape
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized

    # Làm mượt nhẹ để mô phỏng anti-alias của ảnh MNIST gốc (nét không phải
    # nhị phân cứng 0/255 mà có gradient mềm ở viền) - giúp phân bố pixel
    # gần với dữ liệu huấn luyện hơn
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.6)

    tensor = torch.from_numpy(canvas).float() / 255.0
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    return tensor.unsqueeze(0).unsqueeze(0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_tensor, raw_canvas = preprocess_canvas_image(data["image"])
    image_tensor = image_tensor.to(device)

    # Nếu canvas trống (chưa vẽ gì) -> không chạy Grad-CAM (không có gì để giải thích)
    if raw_canvas.getextrema() == (0, 0):
        return jsonify({
            "prediction": None,
            "confidence": 0,
            "probabilities": {str(i): 0 for i in range(10)},
            "top3": [],
            "gradcam_image": None,
        })

    probs, predicted_digit, cam = forward_with_gradcam(image_tensor)
    gradcam_b64 = make_gradcam_overlay(raw_canvas, cam)

    probabilities = {str(i): round(float(p) * 100, 2) for i, p in enumerate(probs)}

    return jsonify({
        "prediction": predicted_digit,
        "confidence": round(float(probs[predicted_digit]) * 100, 2),
        "probabilities": probabilities,
        "top3": get_top_k(probs, k=3),
        "gradcam_image": gradcam_b64,
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

    for seg in segments:
        x, y, w, h = seg["bbox"]
        tensor = preprocess_digit_crop(seg["crop"]).to(device)

        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
        digit = int(np.argmax(probs))
        conf = round(float(probs[digit]) * 100, 2)
        digits_result.append({
            "digit": digit,
            "confidence": conf,
            "bbox": [int(x), int(y), int(w), int(h)],
            "top3": get_top_k(probs, k=3),
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
        if conf < MIN_FRAME_CONFIDENCE:
            continue  # model chưa đủ chắc chắn -> bỏ qua, không hiển thị
        digits_result.append({
            "digit": digit,
            "confidence": conf,
            "bbox": [int(x), int(y), int(w), int(h)],
            "top3": get_top_k(probs, k=3),
        })

    return jsonify({"digits": digits_result})


if __name__ == "__main__":
    app.run(debug=True, port=5000)