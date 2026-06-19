import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
import sys
import os

COLOR_NAMES = {
    "Red":        ([150,  20,  20], [255,  80,  80]),
    "Dark Red":   ([ 80,   0,   0], [149,  40,  40]),
    "Green":      ([ 20, 120,  20], [ 80, 200,  80]),
    "Dark Green": ([  0,  60,   0], [ 19, 119,  19]),
    "Blue":       ([ 20,  20, 150], [ 80,  80, 255]),
    "Dark Blue":  ([  0,   0,  80], [ 19,  19, 149]),
    "Yellow":     ([200, 200,   0], [255, 255, 100]),
    "Orange":     ([200, 100,   0], [255, 170,  50]),
    "Purple":     ([100,   0, 150], [180,  80, 255]),
    "Pink":       ([220, 100, 150], [255, 180, 220]),
    "Cyan":       ([  0, 180, 180], [100, 255, 255]),
    "Brown":      ([ 80,  40,   0], [160, 100,  60]),
    "White":      ([200, 200, 200], [255, 255, 255]),
    "Light Gray": ([150, 150, 150], [199, 199, 199]),
    "Gray":       ([ 80,  80,  80], [149, 149, 149]),
    "Dark Gray":  ([ 30,  30,  30], [ 79,  79,  79]),
    "Black":      ([  0,   0,   0], [ 29,  29,  29]),
}

def rgb_to_color_name(rgb):
    r, g, b = rgb
    min_dist = float('inf')
    best_name = "Unknown"
    for name, (low, high) in COLOR_NAMES.items():
        ref = np.array([(low[0]+high[0])//2, (low[1]+high[1])//2, (low[2]+high[2])//2])
        dist = np.linalg.norm(np.array([r, g, b]) - ref)
        if dist < min_dist:
            min_dist = dist
            best_name = name
    return best_name


def extract_colors(frame_rgb, n_colors=10):
    h, w = frame_rgb.shape[:2]
    if max(h, w) > 400:
        scale = 400 / max(h, w)
        frame_rgb = cv2.resize(frame_rgb, (int(w*scale), int(h*scale)))

    pixels = frame_rgb.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_.astype(int)
    counts  = Counter(kmeans.labels_)
    total   = len(pixels)

    results = []
    for label, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        rgb = tuple(centers[label])
        results.append({
            "rgb":     rgb,
            "hex":     "#{:02X}{:02X}{:02X}".format(*rgb),
            "percent": (count / total) * 100,
            "name":    rgb_to_color_name(rgb)
        })
    return results


def show_palette(img_rgb, colors, title="Color Analysis"):
    n = len(colors)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("#1a1a2e")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Image / Frame", color="white", fontsize=13, pad=10)
    axes[0].axis("off")

    ax = axes[1]
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n)
    ax.axis("off")
    ax.set_title("Dominant Colors", color="white", fontsize=13, pad=10)

    for i, c in enumerate(colors):
        y = n - i - 1
        r, g, b = c["rgb"]
        ax.add_patch(plt.Rectangle((0, y+0.1), 2.5, 0.8,
                                   color=(r/255, g/255, b/255), ec="white", lw=0.5))
        ax.text(2.8, y+0.55, c["name"], color="white", va="center", fontsize=9, fontweight="bold")
        ax.text(2.8, y+0.15, f'{c["hex"]}  |  RGB{c["rgb"]}  |  {c["percent"]:.1f}%',
                color="#aaaaaa", va="center", fontsize=7)

    plt.tight_layout()
    plt.suptitle(title, color="#f0c040", fontsize=14, y=1.01)
    return fig


def print_summary(colors, label=""):
    if label:
        print(f"\n  {label}")
    print(f"{'#':<4} {'Color Name':<14} {'HEX':<10} {'RGB':<22} {'%'}")
    print("-" * 55)
    for i, c in enumerate(colors, 1):
        print(f"{i:<4} {c['name']:<14} {c['hex']:<10} {str(c['rgb']):<22} {c['percent']:.1f}%")


def analyze_image(path, n_colors=10):
    img = cv2.imread(path)
    if img is None:
        print(f"Cannot open: {path}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    colors  = extract_colors(img_rgb, n_colors)
    print_summary(colors, label=os.path.basename(path))

    fig      = show_palette(img_rgb, colors, title=f"Image: {os.path.basename(path)}")
    out_path = os.path.splitext(path)[0] + "_colors.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=150)
    print(f"\nResult saved: {out_path}")
    plt.show()


def analyze_video(path, n_colors=10, sample_every=30):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Cannot open: {path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {os.path.basename(path)} | Frames: {total_frames} | FPS: {fps:.1f}")

    out_dir    = os.path.splitext(path)[0] + "_color_frames"
    os.makedirs(out_dir, exist_ok=True)
    all_pixels = []
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_pixels.append(cv2.resize(frame_rgb, (200, 200)).reshape(-1, 3))

            colors    = extract_colors(frame_rgb, n_colors)
            timestamp = frame_idx / fps if fps > 0 else frame_idx
            label     = f"Frame {frame_idx} | Time {timestamp:.1f}s"
            print_summary(colors, label=label)

            fig      = show_palette(frame_rgb, colors, title=label)
            out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.png")
            fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=120)
            plt.close(fig)

        frame_idx += 1

    cap.release()

    if all_pixels:
        combined = np.vstack(all_pixels).astype(np.float32)
        kmeans   = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(combined)
        centers  = kmeans.cluster_centers_.astype(int)
        counts   = Counter(kmeans.labels_)
        total    = len(combined)
        overall  = []
        for lbl, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            rgb = tuple(centers[lbl])
            overall.append({
                "rgb":     rgb,
                "hex":     "#{:02X}{:02X}{:02X}".format(*rgb),
                "percent": (cnt / total) * 100,
                "name":    rgb_to_color_name(rgb)
            })
        print_summary(overall, label="Overall Video Summary")

        palette = np.zeros((100, n_colors * 80, 3), dtype=np.uint8)
        for i, c in enumerate(overall):
            palette[:, i*80:(i+1)*80] = c["rgb"]
        cv2.imwrite(os.path.join(out_dir, "overall_palette.png"),
                    cv2.cvtColor(palette, cv2.COLOR_RGB2BGR))

    print(f"\nAll results saved in: {out_dir}/")


if __name__ == "__main__":
    FILE_PATH    = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    N_COLORS     = 10
    SAMPLE_EVERY = 30

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

    if os.path.splitext(FILE_PATH)[1].lower() in VIDEO_EXTS:
        analyze_video(FILE_PATH, N_COLORS, SAMPLE_EVERY)
    else:
        analyze_image(FILE_PATH, N_COLORS)