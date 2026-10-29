"""
UNIT 5 - Task 3: Viola-Jones (Haar cascade) vs YOLO
Input : data/photo1.jpg, data/photo2.jpg and 6 frames taken from data/video.mp4
Output: outputs/task3/   (first run needs internet once to download yolov8n.pt)
"""
import os
import time
import cv2
import numpy as np
from utils import load_image, video_path, extract_frames, out_dir, save_csv, print_table, grid

YOLO_WEIGHTS = "yolov8n.pt"


def haar_detect(img, cascade):
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    t = time.time()
    rects, _, weights = cascade.detectMultiScale3(gray, scaleFactor=1.1, minNeighbors=5,
                                                  minSize=(40, 40), outputRejectLevels=True)
    dt = (time.time() - t) * 1000
    vis = img.copy()
    n = 0
    for (x, y, w, h), s in zip(rects, np.array(weights).ravel() if len(rects) else []):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(vis, f"face {float(s):.1f}", (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        n += 1
    return vis, n, dt


def yolo_detect(img, model):
    t = time.time()
    res = model(img, conf=0.25, verbose=False)[0]
    dt = (time.time() - t) * 1000
    vis = res.plot()  # draws boxes + class label + confidence
    labels = [model.names[int(c)] for c in res.boxes.cls]
    persons = labels.count("person")
    return vis, len(labels), persons, labels, dt


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def main():
    from ultralytics import YOLO
    out = out_dir("task3")
    model = YOLO(YOLO_WEIGHTS)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    items = [("photo1", load_image("photo1", 800)), ("photo2", load_image("photo2", 800))]
    frames, idxs = extract_frames(video_path(), n=6)
    items += [(f"frame_{i}", f) for f, i in zip(frames, idxs)]

    rows, notes = [], []
    for name, img in items:
        h_vis, h_n, h_t = haar_detect(img, cascade)
        y_vis, y_n, y_p, y_labels, y_t = yolo_detect(img, model)
        side = np.hstack([label(h_vis, f"Haar: {h_n} face(s)"), label(y_vis, f"YOLO: {y_n} object(s)")])
        cv2.imwrite(os.path.join(out, f"compare_{name}.png"), side)
        rows.append([name, h_n, round(h_t, 1), y_n, y_p, ", ".join(sorted(set(y_labels))) or "-", round(y_t, 1)])
        if y_p > 0 and h_n == 0:
            notes.append(f"{name}: YOLO found {y_p} person(s) but Haar found NO face -> Haar MISS (look at compare_{name}.png)")
        if h_n > y_p:
            notes.append(f"{name}: Haar found {h_n} face(s) but YOLO only {y_p} person(s) -> possible Haar FALSE POSITIVE")
        if y_n == 0:
            notes.append(f"{name}: YOLO detected nothing -> possible YOLO miss")

    header = ["Image", "Haar faces", "Haar ms", "YOLO objects", "YOLO persons", "YOLO classes", "YOLO ms"]
    print("\n=== Haar cascade vs YOLO ===")
    print_table(header, rows)
    save_csv(os.path.join(out, "detection_comparison.csv"), header, rows)

    print("\n=== Misses / wrong detections found automatically (use these in your report) ===")
    print("\n".join(notes) if notes else "None flagged automatically - check images manually for wrong labels.")
    with open(os.path.join(out, "failure_cases.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(notes) if notes else "No automatic flags; inspect images manually.")
    print(f"\nTask 3 done -> {out}")


if __name__ == "__main__":
    main()
