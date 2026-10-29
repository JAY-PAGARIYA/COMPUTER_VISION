"""
UNIT 4 - Task 2: HOG, SIFT keypoints/descriptors, feature matching
Input : data/photo1.jpg and data/photo2.jpg (same object/scene, 2 different shots)
Output: outputs/task2/
"""
import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
from utils import load_image, out_dir, grid, save_csv, print_table

sift = cv2.SIFT_create()


def hog_visualize(img, name, out):
    gray = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2GRAY)
    feats, hog_img = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                         visualize=True, block_norm="L2-Hys")
    hog_img = exposure.rescale_intensity(hog_img, in_range=(0, np.percentile(hog_img, 99.5) + 1e-6))
    hog_u8 = (hog_img * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out, f"hog_{name}.png"), hog_u8)
    print(f"HOG {name}: feature vector length = {len(feats)}")
    return gray, hog_u8, len(feats)


def rotate_bound(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh))


def match_pair(a, b, ratio=0.75):
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    k1, d1 = sift.detectAndCompute(ga, None)
    k2, d2 = sift.detectAndCompute(gb, None)
    stats = dict(kp1=len(k1), kp2=len(k2), good=0, inliers=0)
    if d1 is None or d2 is None or len(k2) < 2:
        return None, stats
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    stats["good"] = len(good)
    mask = None
    if len(good) >= 4:
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            stats["inliers"] = int(mask.sum())
    draw_mask = mask.ravel().tolist() if mask is not None else None
    vis = cv2.drawMatches(a, k1, b, k2, good, None, matchColor=(0, 255, 0),
                          matchesMask=draw_mask,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return vis, stats


def main():
    out = out_dir("task2")
    img1 = load_image("photo1", max_side=800)
    img2 = load_image("photo2", max_side=800)

    # ---- 1. HOG
    g1, h1, n1 = hog_visualize(img1, "photo1", out)
    g2, h2, n2 = hog_visualize(img2, "photo2", out)
    grid([g1, h1, g2, h2], ["Photo 1 (256x256 gray)", "HOG photo 1", "Photo 2 (256x256 gray)", "HOG photo 2"],
         os.path.join(out, "hog_visualization.png"), cols=4)

    # ---- 2. SIFT keypoints + descriptors
    vis_kps = []
    for name, im in (("photo1", img1), ("photo2", img2)):
        kp, des = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
        print(f"SIFT {name}: {len(kp)} keypoints, descriptor shape = {des.shape}")
        vis_kps.append(cv2.drawKeypoints(im, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    grid(vis_kps, ["SIFT keypoints - photo 1", "SIFT keypoints - photo 2"],
         os.path.join(out, "sift_keypoints.png"), cols=2, cell=6)

    # ---- 3 & 4. Matching + rotated / resized experiments
    experiments = [
        ("Original pair", img2),
        ("Photo 2 rotated 45 deg", rotate_bound(img2, 45)),
        ("Photo 2 rotated 90 deg", rotate_bound(img2, 90)),
        ("Photo 2 resized to 50%", cv2.resize(img2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)),
    ]
    rows, vis_list, titles = [], [], []
    for name, im in experiments:
        vis, st = match_pair(img1, im)
        ratio = round(100 * st["inliers"] / st["good"], 1) if st["good"] else 0
        rows.append([name, st["kp1"], st["kp2"], st["good"], st["inliers"], f"{ratio}%"])
        if vis is not None:
            cv2.imwrite(os.path.join(out, "match_" + name.replace(" ", "_").replace("%", "pct") + ".png"), vis)
            vis_list.append(vis)
            titles.append(f"{name}: {st['good']} good, {st['inliers']} RANSAC inliers")

    header = ["Experiment", "KP img1", "KP img2", "Good matches (ratio 0.75)", "RANSAC inliers", "Inlier ratio"]
    print("\n=== SIFT matching comparison ===")
    print_table(header, rows)
    save_csv(os.path.join(out, "matching_comparison.csv"), header, rows)
    grid(vis_list, titles, os.path.join(out, "matching_comparison.png"), cols=2, cell=8)
    print(f"\nTask 2 done -> {out}")


if __name__ == "__main__":
    main()
