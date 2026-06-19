import cv2
import numpy as np
import base64
import json
import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO

app = Flask(__name__)

STRIP_DIR = "saved_strips"
os.makedirs(STRIP_DIR, exist_ok=True)


def decode_frame(data_url: str) -> np.ndarray:
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_frame(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def apply_color(img: np.ndarray) -> np.ndarray:
    """Vintage film look: warm shadows, faded highlights, slight vignette, grain."""
    result = img.copy().astype(np.float32)

    # Fade highlights and lift shadows (matte film look)
    result = result * 0.82 + 28

    # Warm tone: boost reds/greens, pull blue
    result[:, :, 2] = np.clip(result[:, :, 2] * 1.18, 0, 255)  # red
    result[:, :, 1] = np.clip(result[:, :, 1] * 1.05, 0, 255)  # green
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.80, 0, 255)  # blue

    result = result.astype(np.uint8)

    # Film grain
    h, w = result.shape[:2]
    grain = np.random.normal(0, 10, (h, w)).astype(np.int16)
    for c in range(3):
        ch = result[:, :, c].astype(np.int16) + grain
        result[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

    # Vignette
    cx, cy = w / 2, h / 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vignette = 1 - 0.55 * (dist / max_dist) ** 1.6
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c] * vignette, 0, 255).astype(np.uint8)

    return result


def apply_bw(img: np.ndarray) -> np.ndarray:
    """Vintage B&W film: high contrast, grain, vignette, slight silver halide tone."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Lift shadows, compress highlights (film stock feel)
    gray = gray * 0.78 + 22

    # Boost contrast with a soft S-curve via CLAHE
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_u8 = clahe.apply(gray_u8)

    # Film grain
    h, w = gray_u8.shape
    grain = np.random.normal(0, 14, (h, w)).astype(np.int16)
    gray_u8 = np.clip(gray_u8.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    # Vignette
    cx, cy = w / 2, h / 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vignette = 1 - 0.65 * (dist / max_dist) ** 1.5
    gray_u8 = np.clip(gray_u8 * vignette, 0, 255).astype(np.uint8)

    # Slight warm silver tone (sepia-ish tint, subtle)
    result = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
    result[:, :, 2] = np.clip(result[:, :, 2] * 1.04, 0, 255)  # slight warm
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.94, 0, 255)  # pull blue

    return result.astype(np.uint8)


def apply_pixel(img: np.ndarray) -> np.ndarray:
    """Cartoon pixel art: posterise colours, bold outlines, flat cel-shaded look."""
    h, w = img.shape[:2]

    # --- 1. Posterise to flat cartoon colours (8 levels per channel) ---
    levels = 8
    step = 256 // levels
    posterised = (img.astype(np.float32) // step * step + step // 2).astype(np.uint8)

    # --- 2. Pixelate: shrink then snap back (INTER_NEAREST = hard edges) ---
    block = max(8, w // 48)
    small_w, small_h = max(1, w // block), max(1, h // block)
    small = cv2.resize(posterised, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # --- 3. Bold cartoon outlines via edge detection on the pixelated image ---
    gray = cv2.cvtColor(pixelated, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    # Dilate edges so outlines are thick and blocky
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Draw black outlines
    outline_mask = edges > 0
    result = pixelated.copy()
    result[outline_mask] = [0, 0, 0]

    # --- 4. Boost saturation so colours pop like a cartoon ---
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


FILTERS = {
    "color": apply_color,
    "bw": apply_bw,
    "pixel": apply_pixel,
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process_frame():
    body = request.get_json()
    frame_data = body.get("frame")
    filter_name = body.get("filter", "color")

    img = decode_frame(frame_data)
    img = cv2.flip(img, 1)

    fn = FILTERS.get(filter_name, apply_color)
    result = fn(img)

    return jsonify({"frame": encode_frame(result)})


@app.route("/api/capture", methods=["POST"])
def capture():
    body = request.get_json()
    frame_data = body.get("frame")
    filter_name = body.get("filter", "color")

    img = decode_frame(frame_data)
    img = cv2.flip(img, 1)

    fn = FILTERS.get(filter_name, apply_color)
    result = fn(img)

    return jsonify({"frame": encode_frame(result)})


@app.route("/api/save_strip", methods=["POST"])
def save_strip():
    body = request.get_json()
    frames = body.get("frames", [])

    if len(frames) != 3:
        return jsonify({"error": "Need exactly 3 frames"}), 400

    imgs = [decode_frame(f) for f in frames]

    PAD = 20
    FRAME_W = 320
    FRAME_H = int(FRAME_W * 3 / 4)
    GAP = 12
    HEADER = 48
    FOOTER = 32
    STRIP_W = FRAME_W + PAD * 2
    STRIP_H = HEADER + FRAME_H * 3 + GAP * 2 + FOOTER + PAD * 2

    strip = np.full((STRIP_H, STRIP_W, 3), (30, 26, 10), dtype=np.uint8)

    cv2.rectangle(strip, (0, 0), (STRIP_W, STRIP_H), (83, 52, 15), 4)

    title = "PHOTOBOOTH.EXE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(title, font, 0.55, 1)
    tx = (STRIP_W - tw) // 2
    cv2.putText(strip, title, (tx, PAD + 22), font, 0.55, (94, 149, 233), 1, cv2.LINE_AA)

    date_str = datetime.now().strftime("%d/%m/%y")
    (dw, _), _ = cv2.getTextSize(date_str, font, 0.35, 1)
    dx = (STRIP_W - dw) // 2
    cv2.putText(strip, date_str, (dx, PAD + 38), font, 0.35, (83, 52, 15), 1, cv2.LINE_AA)

    for i, img in enumerate(imgs):
        resized = cv2.resize(img, (FRAME_W, FRAME_H))
        y = HEADER + PAD + i * (FRAME_H + GAP)
        strip[y : y + FRAME_H, PAD : PAD + FRAME_W] = resized
        cv2.rectangle(strip, (PAD, y), (PAD + FRAME_W, y + FRAME_H), (83, 52, 15), 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"strip_{timestamp}.png"
    filepath = os.path.join(STRIP_DIR, filename)
    cv2.imwrite(filepath, strip)

    _, buf = cv2.imencode(".png", strip)
    b64 = base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "strip": f"data:image/png;base64,{b64}",
        "filename": filename,
        "saved_to": filepath,
    })


@app.route("/api/download_strip", methods=["POST"])
def download_strip():
    body = request.get_json()
    strip_b64 = body.get("strip", "").split(",", 1)[-1]
    img_bytes = base64.b64decode(strip_b64)
    buf = BytesIO(img_bytes)
    buf.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name=f"strip_{timestamp}.png",
    )


if __name__ == "__main__":
    print("\n  PHOTOBOOTH.EXE — starting server...")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
