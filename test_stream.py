# stream_server.py  (Python 3.6, stdlib only)
import cv2
import time
import threading
try:
    from http.server import ThreadingHTTPServer
except ImportError:                      # Python 3.6
    from socketserver import ThreadingMixIn
    from http.server import HTTPServer
    class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

from yolov8_trt import YoloV8TRT

# ---------------- shared state ----------------
class FrameStore(object):
    def __init__(self):
        self._jpg = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    def update(self, frame_bgr):
        ok, jpg = cv2.imencode(".jpg", frame_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._lock:
                self._jpg = jpg.tobytes()
            self._event.set()          # wake waiting viewers

    def get(self, timeout=1.0):
        self._event.wait(timeout)
        self._event.clear()
        with self._lock:
            return self._jpg

store = FrameStore()

# ---------------- perception loop ----------------
def perception_loop():
    GST = ("nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,"
           "framerate=5/1 ! nvvidconv ! video/x-raw,format=BGRx ! "
           "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1")
    cap = cv2.VideoCapture(GST, cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture(0)   # USB camera instead
    assert cap.isOpened(), "camera failed to open"

    det = YoloV8TRT("yolov8n_fp16.engine", conf_th=0.25)
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        dets = det.infer(frame)
        t_now = time.time()
        fps = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "%.2f" % d["conf"], (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "%.1f FPS  obstacles: %d" % (fps, len(dets)),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        store.update(frame)

# ---------------- HTTP / MJPEG ----------------
class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            html = (b"<html><body style='margin:0;background:#111'>"
                    b"<img src='/stream' style='width:100%'>"
                    b"</body></html>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = store.get()
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(("Content-Length: %d\r\n\r\n"
                                      % len(jpg)).encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass                     # viewer closed the tab

        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass                             # silence per-request logging

# ---------------- main ----------------
if __name__ == "__main__":
    t = threading.Thread(target=perception_loop, daemon=True)
    t.start()
    server = ThreadingHTTPServer(("0.0.0.0", 5000), StreamHandler)
    print("Streaming at http://<nano-ip>:5000")
    server.serve_forever()