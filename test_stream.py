# stream_server.py  (Python 3.6, stdlib only)
import cv2
import json
import math
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import pycuda.driver as cuda

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

from yolov8_trt import YoloV8TRT

# ---------------- camera geometry (SET THESE FOR YOUR ROBOT) ----------------
CAM_HEIGHT_CM = 9.0  # camera optical center above the ground, in cm
CAM_PITCH_DEG = -10.0    # downward tilt of the camera (0 = looking horizontal)
CAM_HFOV_DEG  = 62.2   # horizontal field of view (62.2 = Raspberry Pi cam v2)

class GroundPlaneMapper(object):
    """Approximate obstacle distance from a single camera.

    Model: ideal pinhole (no distortion), flat ground, camera at a fixed
    height. The bottom edge of a detection box is assumed to be the point
    where the obstacle touches the ground; back-projecting that pixel onto
    the ground plane yields forward / lateral distance in cm.
    """
    def __init__(self, img_w, img_h,
                 height_cm=CAM_HEIGHT_CM,
                 pitch_deg=CAM_PITCH_DEG,
                 hfov_deg=CAM_HFOV_DEG):
        self.cx = img_w / 2.0
        self.cy = img_h / 2.0
        self.fx = (img_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.fy = self.fx                      # square pixels assumed
        self.height = height_cm
        self.pitch = math.radians(pitch_deg)

    def locate(self, u, v):
        """Pixel (u, v) of a ground-contact point -> (forward_cm, lateral_cm)
        or None when the ray points at/above the horizon (no ground hit)."""
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        s, c = math.sin(self.pitch), math.cos(self.pitch)
        down = y * c + s                       # ray component toward ground
        if down <= 1e-6:
            return None
        t = self.height / down
        forward = t * (c - y * s)              # along the ground, ahead
        lateral = t * x                        # + right / - left
        return forward, lateral

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

class DetectionStore(object):
    """Latest detection metadata as a JSON payload for /detections."""
    def __init__(self):
        self._json = b'{"ts": 0, "fps": 0, "obstacles": []}'
        self._lock = threading.Lock()

    def update(self, obj):
        payload = json.dumps(obj).encode("utf-8")
        with self._lock:
            self._json = payload

    def get(self):
        with self._lock:
            return self._json

store = FrameStore()
det_store = DetectionStore()

# ---------------- perception loop ----------------
def perception_loop():
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()   # context lives in THIS thread
    try:
        GST = ("nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,"
            "framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1")
        cap = cv2.VideoCapture(GST, cv2.CAP_GSTREAMER)
        # cap = cv2.VideoCapture(0)   # USB camera instead
        assert cap.isOpened(), "camera failed to open"

        det = YoloV8TRT("yolov8n_fp16.engine", conf_th=0.25)
        mapper = None
        t_prev = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            if mapper is None:
                mapper = GroundPlaneMapper(frame.shape[1], frame.shape[0])
            dets = det.infer(frame)
            t_now = time.time()
            fps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            obstacles = []
            for d in dets:
                x1, y1, x2, y2 = [int(v) for v in d["box"]]
                # bottom-center of the box = assumed ground-contact point
                pos = mapper.locate((x1 + x2) / 2.0, y2)
                if pos is not None:
                    fwd, lat = pos
                    dist = math.hypot(fwd, lat)
                    label = "%.2f  %dcm" % (d["conf"], int(round(dist)))
                else:
                    fwd = lat = dist = None
                    label = "%.2f  far" % d["conf"]

                obstacles.append(dict(
                    box=[x1, y1, x2, y2],
                    conf=round(d["conf"], 3),
                    cls=d["cls"],
                    forward_cm=None if fwd is None else round(fwd, 1),
                    lateral_cm=None if lat is None else round(lat, 1),
                    distance_cm=None if dist is None else round(dist, 1)))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            det_store.update(dict(ts=t_now, fps=round(fps, 1),
                                  obstacles=obstacles))
            cv2.putText(frame, "%.1f FPS  obstacles: %d" % (fps, len(dets)),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            store.update(frame)
    finally:
        cuda_ctx.pop()

# ---------------- HTTP / MJPEG ----------------
class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            html = (b"<html><body style='margin:0;background:#111;"
                    b"color:#0f0;font-family:monospace'>"
                    b"<img src='/stream' style='width:100%'>"
                    b"<pre id='det' style='padding:8px'></pre>"
                    b"<script>"
                    b"setInterval(function(){"
                    b"fetch('/detections').then(function(r){return r.json();})"
                    b".then(function(j){"
                    b"var s=j.fps+' FPS  '+j.obstacles.length+' obstacle(s)\\n';"
                    b"j.obstacles.forEach(function(o,i){"
                    b"s+='#'+i+' cls '+o.cls+' conf '+o.conf+"
                    b"'  dist '+(o.distance_cm==null?'?':o.distance_cm+' cm')+"
                    b"'  fwd '+(o.forward_cm==null?'?':o.forward_cm)+"
                    b"'  lat '+(o.lateral_cm==null?'?':o.lateral_cm)+'\\n';});"
                    b"document.getElementById('det').textContent=s;});"
                    b"},200);"
                    b"</script></body></html>")
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

        elif self.path == "/detections":
            payload = det_store.get()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

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
