"""Common helper functions used by all tasks."""
import os
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # figures are saved to files (no popup windows)
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "outputs")

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VID_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def out_dir(task_name):
    path = os.path.join(OUT, task_name)
    os.makedirs(path, exist_ok=True)
    return path


def find_file(stem, exts):
    """Find data/<stem>.<any ext> (case-insensitive). Returns path or None."""
    if not os.path.isdir(DATA):
        return None
    for f in os.listdir(DATA):
        name, ext = os.path.splitext(f)
        if name.lower() == stem.lower() and ext.lower() in exts:
            return os.path.join(DATA, f)
    return None


def load_image(stem, max_side=1000, required=True):
    """Load data/<stem>.jpg/png (BGR) and shrink if very large (keeps code fast)."""
    path = find_file(stem, IMG_EXT)
    if path is None:
        if required:
            raise FileNotFoundError(
                f"'{stem}' image not found in the data/ folder. "
                f"Put your photo there as data/{stem}.jpg")
        return None
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def video_path():
    path = find_file("video", VID_EXT)
    if path is None:
        raise FileNotFoundError("Video not found. Put your video as data/video.mp4")
    return path


def extract_frames(path, n=6, start=0.1, end=0.9, max_side=800):
    """Return (frames, indices): n frames evenly spaced between start..end of the video."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(int(total * start), int(total * end), n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return frames, [int(i) for i in idxs]


def grid(images, titles, path, cols=4, cell=4.0, suptitle=None):
    """Save a matplotlib grid. Accepts gray (2D) or BGR (3D) images."""
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell, rows * cell * 0.85))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, im, t in zip(axes, images, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def print_table(header, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = " | ".join(str(h).ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(x).ljust(w) for x, w in zip(r, widths)))
