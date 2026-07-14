# capture_training.py  (Python 3.6, JetPack 4.6)
# Dataset capture: press SPACE in the terminal to save a photo to
# TrainingImages/, press q (or Ctrl-C) to quit.
# Reads keys from the terminal, so it works headless over SSH.
import os
import sys
import tty
import termios
import threading
import cv2

OUT_DIR = "TrainingImages"

GST = ("nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,"
       "framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
       "video/x-raw,format=BGR ! appsink drop=1 max-buffers=1")

class Grabber(threading.Thread):
    """Keeps reading frames so the pipeline stays live; holds the newest."""
    def __init__(self, cap):
        threading.Thread.__init__(self, daemon=True)
        self.cap = cap
        self.frame = None
        self.lock = threading.Lock()

    def run(self):
        while True:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

def next_index():
    """Continue numbering after whatever img_NNNN.jpg already exists."""
    n = 1
    for f in os.listdir(OUT_DIR):
        if f.startswith("img_") and f.endswith(".jpg"):
            try:
                n = max(n, int(f[4:-4]) + 1)
            except ValueError:
                pass
    return n

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(GST, cv2.CAP_GSTREAMER)     # CSI cam
    # cap = cv2.VideoCapture(0)                        # USB cam instead
    assert cap.isOpened(), "camera failed to open"

    grab = Grabber(cap)
    grab.start()

    idx = next_index()
    print("SPACE = save photo, q = quit  (saving to %s/)" % OUT_DIR)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)                      # single keypress, no Enter needed
    try:
        while True:
            ch = sys.stdin.read(1)
            if ch == " ":
                frame = grab.latest()
                if frame is None:
                    print("no frame from camera yet, try again")
                    continue
                path = os.path.join(OUT_DIR, "img_%04d.jpg" % idx)
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print("saved %s  (%dx%d)"
                      % (path, frame.shape[1], frame.shape[0]))
                idx += 1
            elif ch in ("q", "\x1b"):      # q or Esc
                break
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        cap.release()
        print("done, next run continues at img_%04d.jpg" % idx)

if __name__ == "__main__":
    main()
