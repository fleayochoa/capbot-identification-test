#!/usr/bin/env python3
# yolo_stream.py - YOLOv8n TRT + MJPEG streaming over WiFi, tuned for low power
import cv2, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from yolov8_trt import YoloV8TRT

# ---- power-oriented parameters -------------------------------------------
CAP_W, CAP_H   = 640, 360     # capture small: less ISP + memory traffic
FPS            = 5            # acquisition AND inference rate
JPEG_QUALITY   = 70           # lower = less CPU on encode + less WiFi airtime
PORT           = 8080

# Ask the sensor pipeline for 5 fps directly. IMX219 runs its native mode,
# but Argus delivers only 5 frames/s -> ISP and memory stay mostly idle.
GST = (f"nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,"
       f"framerate={FPS}/1 ! nvvidconv ! "
       f"video/x-raw,width={CAP_W},height={CAP_H},format=BGRx ! "
       f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1")

det = YoloV8TRT("yolov8n_fp16.engine", conf_th=0.25)

latest_jpeg = None
lock = threading.Condition()

def capture_loop():
    global latest_jpeg
    cap = cv2.VideoCapture(GST, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Camera pipeline failed to open")
    t_prev = time.time()
    while True:
        ok, frame = cap.read()          # blocks ~200 ms at 5 fps -> CPU idles
        if not ok:
            break
        dets = det.infer(frame)
        t_now = time.time()
        fps = 1.0 / (t_now - t_prev); t_prev = t_now
        for d in dets:
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "%.1f FPS" % fps, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with lock:
                latest_jpeg = buf.tobytes()
                lock.notify_all()       # wake only when a new frame exists
    cap.release()

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with lock:
                    lock.wait()          # push-driven: no polling, no busy CPU
                    jpg = latest_jpeg
                self.wfile.write(b"--frame\r\n"
                                 b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass                         # client closed the tab

    def log_message(self, *a):           # silence per-request logging
        pass

threading.Thread(target=capture_loop, daemon=True).start()
print(f"Streaming at http://0.0.0.0:{PORT}")
ThreadingHTTPServer(("0.0.0.0", PORT), MJPEGHandler).serve_forever()