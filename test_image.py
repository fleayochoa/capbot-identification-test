import cv2, time
from yolov8_trt import YoloV8TRT

COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
        "bird","cat","dog","horse","sheep","cow"]  # first 20; full list has 80

det = YoloV8TRT("yolov8n_fp16.engine", conf_th=0.25)   # was 0.40
img = cv2.imread("test.jpg")

for _ in range(3):                      # warm-up
    det.infer(img)

t0 = time.time()
N = 20
for _ in range(N):
    dets = det.infer(img)
print("%.1f FPS end-to-end" % (N / (time.time() - t0)))

for d in dets:
    x1, y1, x2, y2 = map(int, d["box"])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    name = COCO[d["cls"]] if d["cls"] < len(COCO) else str(d["cls"])
    cv2.putText(img, "%s %.2f" % (name, d["conf"]), (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imwrite("result.jpg", img)
print("saved result.jpg,", len(dets), "detections")