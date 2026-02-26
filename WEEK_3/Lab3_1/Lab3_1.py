import cv2
import time
from ultralytics import YOLO
# import os

# print("Current working directory:", os.getcwd())
# print("Script directory:", os.path.dirname(os.path.abspath(__file__)))

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# VIDEO_PATH = os.path.join(BASE_DIR, "street.mp4")

# print("Looking for video at:", VIDEO_PATH)
# print("Exists:", os.path.exists(VIDEO_PATH))
# -----------------------------
# SETTINGS
# -----------------------------
VIDEO_PATH = VIDEO_PATH = r"C:\Users\HP\Documents\EY-ASSIGNMENTS\WEEK_3\Lab3_1\street.mp4"  # <-- put your video file here
OUTPUT_PATH = "output.mp4"

# -----------------------------
# Load YOLOv8 Nano
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Open Video File
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps_video = cap.get(cv2.CAP_PROP_FPS)

# Video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video,
                      (frame_width, frame_height))

prev_time = 0

# -----------------------------
# Processing Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster inference (optional)
    frame = cv2.resize(frame, (640, 480))

    # Run YOLO
    results = model(frame, verbose=False)

    person_count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])

        # COCO class 0 = person
        if cls == 0:
            person_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"Person {conf:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Overlay info
    cv2.putText(frame, f"People Count: {person_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,0,255), 3)

    cv2.putText(frame, f"FPS: {fps:.2f}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,0,0), 3)

    # Show frame
    cv2.imshow("YOLOv8 Person Detection", frame)

    # Save frame
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete.")
print("Saved output to:", OUTPUT_PATH)