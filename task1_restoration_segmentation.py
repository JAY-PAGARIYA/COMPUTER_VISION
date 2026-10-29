"""
UNIT 3 - Task 1: Image Restoration, Morphology, Edge Detection, Watershed
Input : data/photo1.jpg   (optional: data/objects.jpg for watershed - touching objects)
Output: outputs/task1/
"""
import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils import load_image, out_dir, grid, save_csv, print_table


def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount=0.05):
    out = img.copy()
    r = np.random.rand(*img.shape[:2])
    out[r < amount / 2] = 0          # pepper
    out[r > 1 - amount / 2] = 255    # salt
    return out


def metrics(clean, test):
    return round(psnr(clean, test, data_range=255), 2), round(ssim(clean, test, channel_axis=2, data_range=255), 4)


def part_a_restoration(img, out):
    np.random.seed(42)
    g_noisy = add_gaussian_noise(img, 25)
    sp_noisy = add_salt_pepper(img, 0.05)

    results = {
        "Gaussian noise": g_noisy,
        "Gaussian noise -> Median (5)": cv2.medianBlur(g_noisy, 5),
        "Gaussian noise -> Gaussian filter": cv2.GaussianBlur(g_noisy, (5, 5), 1.2),
        "Salt&Pepper noise": sp_noisy,
        "Salt&Pepper -> Median (5)": cv2.medianBlur(sp_noisy, 5),
        "Salt&Pepper -> Gaussian filter": cv2.GaussianBlur(sp_noisy, (5, 5), 1.2),
    }
    rows = []
    for name, im in results.items():
        p, s = metrics(img, im)
        rows.append([name, p, s])
        cv2.imwrite(os.path.join(out, name.replace(" ", "_").replace("&", "and").replace(">", "").replace("(", "").replace(")", "") + ".png"), im)

    header = ["Image", "PSNR (dB)", "SSIM"]
    print("\n=== Image restoration metrics (vs. original clean image) ===")
    print_table(header, rows)
    save_csv(os.path.join(out, "restoration_metrics.csv"), header, rows)

    names = list(results.keys())
    imgs = [img] + [results[n] for n in names[:3]] + [img] + [results[n] for n in names[3:]]
    titles = ["Original"] + [f"{n}\nPSNR={r[1]} SSIM={r[2]}" for n, r in zip(names[:3], rows[:3])] + \
             ["Original"] + [f"{n}\nPSNR={r[1]} SSIM={r[2]}" for n, r in zip(names[3:], rows[3:])]
    grid(imgs, titles, os.path.join(out, "restoration_comparison.png"), cols=4)

    best_g = max(rows[1:3], key=lambda r: r[1])[0]
    best_sp = max(rows[4:6], key=lambda r: r[1])[0]
    print(f"\nBest for Gaussian noise    : {best_g}")
    print(f"Best for Salt&Pepper noise : {best_sp}")


def part_b_morphology_edges(img, out):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(otsu, k, iterations=1)
    dilated = cv2.dilate(otsu, k, iterations=1)
    opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, k)
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k)

    med = np.median(blur)
    lo, hi = int(max(0, 0.66 * med)), int(min(255, 1.33 * med))
    canny_auto = cv2.Canny(blur, lo, hi)
    canny_fixed = cv2.Canny(blur, 100, 200)

    items = [
        (gray, "Grayscale"), (otsu, "Otsu threshold (binary mask)"), (adaptive, "Adaptive threshold"),
        (eroded, "Erosion"), (dilated, "Dilation"), (opened, "Opening (erode->dilate)"),
        (closed, "Closing (dilate->erode)"), (canny_auto, f"Canny auto ({lo},{hi})"), (canny_fixed, "Canny fixed (100,200)"),
    ]
    for im, t in items:
        cv2.imwrite(os.path.join(out, t.split(" (")[0].replace(" ", "_").lower() + ".png"), im)
    grid([i for i, _ in items], [t for _, t in items], os.path.join(out, "morphology_edges.png"), cols=3)
    print("Saved binary mask, morphology results and Canny edge maps.")


def watershed_seg(img):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if th.mean() > 127:          # objects should be the smaller (white) region
        th = 255 - th
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img.copy(), markers)
    result = img.copy()
    result[markers == -1] = (0, 0, 255)
    dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return opening, dist_vis, sure_fg, result, n - 1


def part_b_watershed(out):
    img = load_image("objects", required=False)
    src = "objects.jpg"
    if img is None:
        img = load_image("photo1")
        src = "photo1.jpg"
    th, dist, sure_fg, result, count = watershed_seg(img)
    print(f"Watershed on {src}: {count} segments found.")
    grid([img, th, dist, sure_fg, result],
         [f"Input ({src})", "Binary mask", "Distance transform", "Sure foreground (markers)", f"Watershed result ({count} regions)"],
         os.path.join(out, "watershed.png"), cols=3)
    cv2.imwrite(os.path.join(out, "watershed_result.png"), result)


def main():
    out = out_dir("task1")
    img = load_image("photo1")
    cv2.imwrite(os.path.join(out, "original.png"), img)
    part_a_restoration(img, out)
    part_b_morphology_edges(img, out)
    part_b_watershed(out)
    print(f"\nTask 1 done -> {out}")


if __name__ == "__main__":
    main()
