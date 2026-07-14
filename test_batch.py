# test_batch.py  (Python 3.6, JetPack 4.6)
# Runs YOLOv8-TRT on every image in ImageTests/ and writes annotated copies
# (bounding boxes + approximated ground-plane distance) to ResultsTest/.
import os
import math
import cv2
import pycuda.driver as cuda

from yolov8_trt import YoloV8TRT
from test_stream import GroundPlaneMapper   # reuse camera geometry / calibration

IN_DIR  = "ImageTests"
OUT_DIR = "ResultsTest"
ENGINE  = "yolov8n_fp16.engine"
EXTS    = (".png", ".jpg", ".jpeg", ".bmp")

COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter",
        "bench","bird","cat","dog","horse","sheep","cow","elephant","bear",
        "zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
        "frisbee","skis","snowboard","sports ball","kite","baseball bat",
        "baseball glove","skateboard","surfboard","tennis racket","bottle",
        "wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
        "cake","chair","couch","potted plant","bed","dining table","toilet",
        "tv","laptop","mouse","remote","keyboard","cell phone","microwave",
        "oven","toaster","sink","refrigerator","book","clock","vase",
        "scissors","teddy bear","hair drier","toothbrush"]

def main():
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()   # context lives in this thread
    try:
        det = YoloV8TRT(ENGINE, conf_th=0.4)
        os.makedirs(OUT_DIR, exist_ok=True)
        mappers = {}                           # (w, h) -> GroundPlaneMapper

        names = sorted(f for f in os.listdir(IN_DIR)
                       if f.lower().endswith(EXTS))
        if not names:
            print("no images found in %s/" % IN_DIR)
            return

        for name in names:
            img = cv2.imread(os.path.join(IN_DIR, name))
            if img is None:
                print("skipping %s (could not read)" % name)
                continue

            h, w = img.shape[:2]
            if (w, h) not in mappers:
                mappers[(w, h)] = GroundPlaneMapper(w, h)
                print("ground-plane for %dx%d: pitch %.1f deg down"
                      % (w, h, math.degrees(mappers[(w, h)].pitch)))
            mapper = mappers[(w, h)]

            dets = det.infer(img)
            for d in dets:
                x1, y1, x2, y2 = [int(v) for v in d["box"]]
                # bottom-center of the box = assumed ground-contact point;
                # if the box touches the frame bottom the contact point is
                # out of view, so the distance is only an upper bound
                clipped = y2 >= h - 3
                pos = mapper.locate((x1 + x2) / 2.0, y2)
                if pos is not None:
                    dist = math.hypot(pos[0], pos[1])
                    dist_txt = "%s%dcm" % ("<" if clipped else "",
                                           int(round(dist)))
                else:
                    dist_txt = "far"           # ray at/above the horizon

                cls_name = (COCO[d["cls"]] if d["cls"] < len(COCO)
                            else str(d["cls"]))
                label = "%s %.2f  %s" % (cls_name, d["conf"], dist_txt)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                print("  %-12s conf %.2f  dist %-6s box (%d,%d,%d,%d)"
                      % (cls_name, d["conf"], dist_txt, x1, y1, x2, y2))

            out_path = os.path.join(OUT_DIR, name)
            cv2.imwrite(out_path, img)
            print("%s: %d detection(s) -> %s" % (name, len(dets), out_path))
    finally:
        cuda_ctx.pop()

if __name__ == "__main__":
    main()
