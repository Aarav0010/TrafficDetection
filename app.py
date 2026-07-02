import cv2
import numpy as np
import os
import sys

# HACK: Add PyTorch's bundled CUDA 12 DLLs to the PATH so ONNX Runtime GPU can find them!
torch_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "venv", "Lib", "site-packages", "torch", "lib"))
if os.path.exists(torch_lib_path):
    os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ["PATH"]

from flask import Flask, Response
from ultralytics import YOLO

app = Flask(__name__)

# Load Models
car_model = YOLO("yolov8n.pt")
sign_model = YOLO("best.onnx", task="detect")

# WHITELIST: Only traffic light classes
TRAFFIC_LIGHT_CLASSES = {0, 1, 2}
SPEED_LIMIT_CLASSES = {3: "30", 4: "50", 5: "60", 6: "80", 7: "100"}


def is_real_traffic_light(frame, x1, y1, x2, y2):
    """Dark-pixel analysis: real traffic light has dark housing, balloon is bright."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return False
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    dark_pixels = np.sum(v_channel < 80)
    return (dark_pixels / v_channel.size) >= 0.25


def get_video_source():
    """Auto-detect: use uploaded video if available, otherwise live dashcam."""
    VIDEO_FILE = os.environ.get("VIDEO_FILE", "spped.mp4")

    if os.path.exists(VIDEO_FILE):
        cap = cv2.VideoCapture(VIDEO_FILE)
        source = "VIDEO"
        print(f"[INFO] Using video file: {VIDEO_FILE}")
    else:
        # Try camera indexes 0, 1, 2 (USB dashcam or built-in webcam)
        cam_index = int(os.environ.get("CAMERA_INDEX", "0"))
        cap = cv2.VideoCapture(cam_index)
        source = "LIVE"
        print(f"[INFO] Video not found! Using LIVE dashcam (camera {cam_index})")

    return cap, source


def generate_frames():
    cap, source = get_video_source()

    if not cap.isOpened():
        print("[ERROR] Could not open video source!")
        return

    # Scale down upfront for performance (removes lag)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    width = 640
    height = int(orig_height * (width / orig_width))
    frame_area = width * height

    print(f"[INFO] Resolution: {width}x{height} | Source: {source}")

    # Backend Lane Polygon (Invisible to user, narrowed to stay in lane)
    lane_pts = np.array([[
        [int(width * 0.35), height],
        [int(width * 0.45), int(height * 0.60)],
        [int(width * 0.55), int(height * 0.60)],
        [int(width * 0.65), height]
    ]], np.int32)

    # Temporal smoothing counters
    red_count = 0
    green_count = 0
    yellow_count = 0
    REQUIRED_FRAMES = 3
    current_speed_limit = None

    import time
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    start_time = time.time()
    frame_idx = 0

    while cap.isOpened():
        if source == "VIDEO":
            elapsed = time.time() - start_time
            expected_frame = int(elapsed * fps)
            if frame_idx < expected_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frame)
                frame_idx = expected_frame
        
        success, frame = cap.read()
        frame_idx += 1

        if not success:
            if source == "VIDEO":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                red_count = green_count = yellow_count = 0
                start_time = time.time()
                frame_idx = 0
                continue
            else:
                break
        
        frame = cv2.resize(frame, (width, height))

        # Run Inference: Both models on GPU
        car_results = car_model(frame, classes=[2], verbose=False, device=0)
        sign_results = sign_model(frame, verbose=False, device=0)

        car_ahead_detected = False
        red_this_frame = False
        yellow_this_frame = False
        green_this_frame = False

        # 1. Filter Cars (Lane detection)
        for r in car_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cx = (x1 + x2) // 2
                cy = y2
                car_area = (x2 - x1) * (y2 - y1)

                # Car must be in lane AND near enough (e.g., area > 0.3% of frame)
                if cv2.pointPolygonTest(lane_pts[0], (cx, cy), False) >= 0 and car_area > (frame_area * 0.003):
                    car_ahead_detected = True

        # 2. Filter Traffic Signs & Lights
        for r in sign_results:
            for box in r.boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                if cls in SPEED_LIMIT_CLASSES:
                    if conf > 0.15:
                        current_speed_limit = SPEED_LIMIT_CLASSES[cls]
                    continue

                if cls not in TRAFFIC_LIGHT_CLASSES:
                    # General traffic sign logic (e.g. pedestrian, right turn)
                    if conf > 0.35:
                        pass # Drawing disabled per user request
                    continue

                box_area = (x2 - x1) * (y2 - y1)
                box_center_y = (y1 + y2) // 2

                # Confidence filter
                if conf < 0.20:
                    continue

                # Dark pixel analysis (balloon killer)
                if not is_real_traffic_light(frame, x1, y1, x2, y2):
                    continue

                if cls == 0:  # traffic_light_red
                    color = (0, 0, 255)
                    red_this_frame = True
                elif cls == 1:  # traffic_light_green
                    color = (0, 255, 0)
                    green_this_frame = True
                elif cls == 2:  # traffic_light_yellow
                    color = (0, 255, 255)
                    yellow_this_frame = True

                name = sign_model.names.get(cls, f"Sign {cls}")

                # Traffic light drawing disabled per user request

        # Temporal smoothing
        red_count = (red_count + 1) if red_this_frame else 0
        green_count = (green_count + 1) if green_this_frame else 0
        yellow_count = (yellow_count + 1) if yellow_this_frame else 0

        red_ok = red_count >= REQUIRED_FRAMES
        green_ok = green_count >= REQUIRED_FRAMES
        yellow_ok = yellow_count >= REQUIRED_FRAMES

        # 3. Dashboard HUD
        status_msg = "CLEAR"
        status_color = (0, 255, 0)

        if red_ok:
            status_msg = "RED LIGHT: STOP"
            status_color = (0, 0, 255)
        elif yellow_ok:
            status_msg = "YELLOW LIGHT: SLOW DOWN"
            status_color = (0, 255, 255)
        elif green_ok and car_ahead_detected:
            status_msg = "GREEN LIGHT BUT CAR AHEAD: Maintain Distance"
            status_color = (0, 165, 255)
        elif green_ok:
            status_msg = "GREEN LIGHT: GO AHEAD"
            status_color = (0, 255, 0)
        elif car_ahead_detected:
            status_msg = "CAR AHEAD: Maintain Distance"
            status_color = (0, 165, 255)

        # Scale back up for display BEFORE drawing HUD
        disp_width = 1024
        disp_height = int(height * (disp_width / width))
        frame = cv2.resize(frame, (disp_width, disp_height))

        # LIVE badge
        if source == "LIVE":
            cv2.circle(frame, (disp_width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "LIVE", (disp_width - 80, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.rectangle(frame, (10, 10), (850, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {status_msg}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)

        # Draw Speed Limit Sign HUD if detected
        if current_speed_limit:
            cv2.rectangle(frame, (disp_width - 150, 60), (disp_width - 10, 160), (255, 255, 255), -1)
            cv2.circle(frame, (disp_width - 80, 110), 45, (0, 0, 255), -1)
            cv2.circle(frame, (disp_width - 80, 110), 38, (255, 255, 255), -1)
            # Adjust text placement based on number of digits
            text_x = disp_width - 105 if len(current_speed_limit) == 2 else disp_width - 118
            cv2.putText(frame, f"{current_speed_limit}", (text_x, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)

        # Stream high-res frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Lane & Sign Detection</title>
        <style>
          body { text-align: center; font-family: sans-serif; background: #0f1115; color: #e5e7eb; margin: 0; padding: 20px;}
          img { max-width: 100%; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        </style>
      </head>
      <body>
        <h2>Live Lane & Traffic Light Detection</h2>
        <img src="/video_feed">
      </body>
    </html>
    """


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("\n=== Traffic Light & Car Detection ===")
    print("Open http://localhost:5000 in your browser\n")
    app.run(host='0.0.0.0', port=5000, threaded=True)
