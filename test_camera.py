import cv2, time
from yolov8_trt import YoloV8TRT

GST = ("nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,"
       "framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
       "video/x-raw,format=BGR ! appsink drop=1 max-buffers=1")

cap = cv2.VideoCapture(GST, cv2.CAP_GSTREAMER)     # CSI cam
# cap = cv2.VideoCapture(0)                        # USB cam instead

det = YoloV8TRT("yolov8n_fp16.engine", conf_th=0.25)   # was 0.40
t_prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    dets = det.infer(frame)
    t_now = time.time()
    fps = 1.0 / (t_now - t_prev); t_prev = t_now
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "%.1f FPS" % fps, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("yolov8n-trt", frame)
    if cv2.waitKey(1) == 27:
        break
cap.release(); cv2.destroyAllWindows()