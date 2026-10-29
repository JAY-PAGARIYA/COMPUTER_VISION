"""
UNIT 5 - Task 4: Frame differencing vs GMM (MOG2) background subtraction vs Lucas-Kanade optical flow
Input : data/video.mp4
Output: outputs/task4/  (sample frames, masks, vector images, 3 result videos, comparison table)
"""
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import video_path, out_dir, save_csv, print_table

WIDTH = 640


def main():
    out = out_dir("task4")
    cap = cv2.VideoCapture(video_path())
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idx = set(np.linspace(int(total * 0.2), int(total * 0.9), 5).astype(int).tolist())

    mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=True)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    writers, samples = {}, []
    stats = {"diff": [0, 0.0], "mog": [0, 0.0], "lk": [0, 0.0]}   # [sum of metric, sum of ms]
    lk_points, n_proc = 0, 0
    prev = prev_gray = prev_blur = None
    i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h0, w0 = frame.shape[:2]
        frame = cv2.resize(frame, (WIDTH, int(h0 * WIDTH / w0)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # ---- GMM background subtraction (must see every frame, including the first)
        t = time.time()
        fg = mog.apply(frame)
        _, m_mog = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)   # drop shadow pixels (value 127)
        m_mog = cv2.dilate(cv2.morphologyEx(m_mog, cv2.MORPH_OPEN, k), k)
        t_mog = (time.time() - t) * 1000

        if prev is None:
            prev, prev_gray, prev_blur = frame, gray, blur
            i += 1
            continue

        # ---- Frame differencing
        t = time.time()
        _, m_diff = cv2.threshold(cv2.absdiff(prev_blur, blur), 25, 255, cv2.THRESH_BINARY)
        m_diff = cv2.dilate(m_diff, k, iterations=2)
        t_diff = (time.time() - t) * 1000

        # ---- Lucas-Kanade optical flow (sparse)
        t = time.time()
        vec = frame.copy()
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
        n_moving = 0
        if p0 is not None:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
            old, new = p0[st == 1].reshape(-1, 2), p1[st == 1].reshape(-1, 2)
            mag = np.linalg.norm(new - old, axis=1)
            for (x0, y0), (x1, y1), m in zip(old, new, mag):
                if m > 1.0:     # moving point -> red arrow
                    cv2.arrowedLine(vec, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2, tipLength=0.4)
                    n_moving += 1
                else:           # almost static point -> small green dot
                    cv2.circle(vec, (int(x1), int(y1)), 2, (0, 255, 0), -1)
        t_lk = (time.time() - t) * 1000

        # ---- record numbers
        area = lambda m: 100.0 * cv2.countNonZero(m) / m.size
        stats["diff"][0] += area(m_diff); stats["diff"][1] += t_diff
        stats["mog"][0] += area(m_mog);   stats["mog"][1] += t_mog
        stats["lk"][0] += n_moving;       stats["lk"][1] += t_lk
        n_proc += 1

        # ---- output videos
        outs = {"frame_differencing": cv2.cvtColor(m_diff, cv2.COLOR_GRAY2BGR),
                "gmm_background_subtraction": cv2.cvtColor(m_mog, cv2.COLOR_GRAY2BGR),
                "lucas_kanade_optical_flow": vec}
        for name, img in outs.items():
            if name not in writers:
                writers[name] = cv2.VideoWriter(os.path.join(out, name + ".mp4"),
                                                cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
            writers[name].write(img)

        if i in sample_idx:
            samples.append((i, prev.copy(), frame.copy(), m_diff, m_mog, vec))
            cv2.imwrite(os.path.join(out, f"frame_{i-1}.png"), prev)
            cv2.imwrite(os.path.join(out, f"frame_{i}.png"), frame)
            cv2.imwrite(os.path.join(out, f"diff_mask_{i}.png"), m_diff)
            cv2.imwrite(os.path.join(out, f"gmm_mask_{i}.png"), m_mog)
            cv2.imwrite(os.path.join(out, f"lk_vectors_{i}.png"), vec)

        prev, prev_gray, prev_blur = frame, gray, blur
        i += 1

    cap.release()
    for w in writers.values():
        w.release()
    if n_proc == 0:
        raise RuntimeError("Video could not be read / too short.")

    # ---- figure with selected frames
    if samples:
        fig, axes = plt.subplots(len(samples), 4, figsize=(16, 3.2 * len(samples)))
        axes = np.atleast_2d(axes)
        for r, (idx, _, fr, md, mg, vc) in enumerate(samples):
            for c, (im, t) in enumerate([(fr, f"Frame {idx}"), (md, "Frame differencing mask"),
                                          (mg, "GMM (MOG2) foreground mask"), (vc, "Lucas-Kanade vectors")]):
                axes[r, c].imshow(im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cmap="gray")
                axes[r, c].set_title(t, fontsize=9)
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "motion_comparison.png"), dpi=120)
        plt.close(fig)

    # ---- comparison table
    header = ["Method", "Avg result per frame", "Avg time (ms/frame)", "Output type", "Strength", "Weakness"]
    rows = [
        ["Frame differencing", f"{stats['diff'][0]/n_proc:.2f}% of pixels flagged", f"{stats['diff'][1]/n_proc:.2f}",
         "Binary mask", "Very fast, simple, no training", "Only edges of moving object; noise/camera shake gives false motion; slow objects missed"],
        ["GMM / MOG2 background subtraction", f"{stats['mog'][0]/n_proc:.2f}% of pixels flagged", f"{stats['mog'][1]/n_proc:.2f}",
         "Binary mask (full object)", "Learns background, fills whole object, handles lighting change slowly", "Needs static camera + warm-up; stopped objects fade into background; shadows/noise"],
        ["Lucas-Kanade optical flow", f"{stats['lk'][0]/n_proc:.1f} moving points/frame", f"{stats['lk'][1]/n_proc:.2f}",
         "Motion vectors (direction + speed)", "Gives direction and speed of motion", "Sparse only; fails on textureless areas / fast motion; camera motion moves all vectors"],
    ]
    print("\n=== Motion estimation comparison ===")
    print_table(header, rows)
    save_csv(os.path.join(out, "comparison_table.csv"), header, rows)
    print(f"\nProcessed {n_proc} frame pairs. Task 4 done -> {out}")


if __name__ == "__main__":
    main()
