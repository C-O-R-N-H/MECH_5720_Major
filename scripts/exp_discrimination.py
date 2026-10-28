"""
Experiment: material / spectral discrimination
================================================
Success criterion: "demonstrate that regions or materials with visibly
different spectral responses can be distinguished from the resulting
hyperspectral data."

Suggested test object: patches that are easy (red vs blue paper) AND hard
— metamer-ish pairs that look the same colour to the eye/RGB camera but
differ spectrally: a real leaf vs a green plastic/paper leaf (chlorophyll's
"red edge" at ~700 nm), two green paints from different brands, white paper
vs white plastic.

Works with or without calibration. Uncalibrated, the spectra are dark-
subtracted counts per spectral column: they still include the lamp's
spectrum and the camera's response, but those are the same for every
material in the scan, and the spectral angle ignores brightness, so
differences in SHAPE between materials are real material differences.

Two modes
  --region name:az0:az1:row0:row1   (degrees, cube rows; repeatable), or
  --pick                            drag boxes on the panorama:
      1. per class: mean spectrum ± std; spectral angle between class means
      2. classify held-out pixels (train on even azimuth columns, test on
         odd) by nearest class mean (spectral angle), using
           a) the full spectrum
           b) a 3-band RGB-like version of the same data
         (a) vs (b) is the "what does hyperspectral buy us" number
      3. classify the whole panorama -> class map
  --auto K   no labels: group pixels into K spectral clusters (k-means on
             spectral shape) and map them, to show which surfaces the
             spectra alone tell apart. Good for picking regions afterwards.

Output: <scan>/discrimination/  discrimination.png / clusters.png, results*.json

    python exp_discrimination.py <scan> --auto 6
    python exp_discrimination.py <scan> --region leaf:10:40:200:400 --region fake:60:90:200:400
"""

import argparse
import os

import cv2
import numpy as np
from scipy.cluster.vq import kmeans2

from hsi import colour, plotting, spectral, store
from hsi.plotting import plt


def parse_region(s, n_az):
    name, a0, a1, h0, h1 = s.split(":")
    c0, c1 = int(float(a0) / 360 * n_az), int(float(a1) / 360 * n_az)
    return name, (c0, c1, int(h0), int(h1))


def pick_regions(rgb_img):
    print("drag a box, press Enter/Space to accept it; Esc when done")
    boxes = cv2.selectROIs("label regions", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()
    out = []
    for i, (x, y, w, h) in enumerate(boxes):
        name = input(f"name for region {i + 1}: ").strip() or f"region{i + 1}"
        out.append((name, (int(x), int(x + w), int(y), int(y + h))))
    return out


def nearest_class(pixels, means):
    """pixels (n, d), means (k, d) -> (labels, min angle)."""
    ang = np.stack([spectral.spectral_angle(pixels, m[None]) for m in means], axis=1)
    return np.argmin(ang, axis=1), np.min(ang, axis=1)


def evaluate(train, test, k):
    """train/test: lists of (n_i, d) per class -> accuracy, confusion."""
    means = np.array([np.nanmean(t, axis=0) for t in train])
    conf = np.zeros((k, k), int)
    for c, t in enumerate(test):
        lab, _ = nearest_class(t, means)
        for l in lab:
            conf[c, l] += 1
    return float(np.trace(conf) / conf.sum()), conf, means


def three_band(calibrated, grid, n_b):
    """Projection to an RGB-like 3-channel camera, as a (bands, 3) matrix.
    Calibrated: CIE colour matching functions. Uncalibrated: the usable
    spectrum split into thirds, each averaged — a broadband 3-filter camera."""
    if calibrated:
        return colour.cie_cmf(grid)
    m = np.zeros((n_b, 3))
    for j, idx in enumerate(np.array_split(np.arange(n_b), 3)):
        m[idx, j] = 1.0 / len(idx)
    return m


def class_map_image(lab):
    cmap = np.array([[0.93, 0.93, 0.92]] + [tuple(int(h[j:j + 2], 16) / 255 for j in (1, 3, 5))
                                            for h in plotting.SERIES])
    return cmap[(lab + 1) % len(cmap)]


def plot_shapes(ax, grid, spectra, names, calibrated, xlabel, colours=None):
    """Class mean spectra. Uncalibrated counts are divided by their own mean,
    so the plot shows spectral SHAPE (what the spectral angle compares)."""
    for i, (n, px) in enumerate(zip(names, spectra)):
        c = colours[i] if colours else plotting.SERIES[i % len(plotting.SERIES)]
        m, s = np.nanmean(px, 0), np.nanstd(px, 0)
        if not calibrated:
            k = np.nanmean(m)
            m, s = m / k, s / k
        ax.fill_between(grid, m - s, m + s, color=c, alpha=0.2, lw=0)
        ax.plot(grid, m, color=c, label=n)
    ax.set(xlabel=xlabel, ylabel="reflectance" if calibrated else "relative shape (÷ own mean)",
           title="Class spectra (band = ±1σ)")
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", help="scan folder containing cube.npz (from scan.py / reconstruct.py)")
    ap.add_argument("--region", action="append", default=[], help="name:az0_deg:az1_deg:row0:row1")
    ap.add_argument("--pick", action="store_true", help="draw regions interactively")
    ap.add_argument("--auto", type=int, default=0, help="unsupervised: number of spectral clusters")
    ap.add_argument("--max-angle", type=float, default=8.0, help="deg; beyond this a pixel is unclassified")
    ap.add_argument("--min-signal", type=float, default=0.08,
                    help="ignore pixels dimmer than this fraction of the bright ones (their spectra are noise)")
    args = ap.parse_args()

    d = np.load(os.path.join(args.scan, "cube.npz"))
    cube, grid = d["cube"], d["grid_nm"]            # (az, height, bands)
    has_nm = bool(d["wavelength"]) if "wavelength" in d else not np.allclose(grid, np.arange(len(grid)))
    calibrated = bool(d["reflectance"]) and has_nm
    # keep bands that are mostly valid AND carry real signal (uncalibrated:
    # columns the lamp barely reaches are noise and would swamp the angles)
    good = np.mean(np.isfinite(cube), axis=(0, 1)) > 0.5
    level = np.nanmedian(cube.reshape(-1, cube.shape[-1]), axis=0)
    if not calibrated:
        good &= level >= 0.2 * np.nanmax(level)
    cube, grid = cube[..., good], grid[good]
    n_az, n_h, n_b = cube.shape
    img = np.transpose(cube, (1, 0, 2))             # (height, az, bands)
    rgb = colour.reflectance_to_srgb(img, grid) if calibrated else colour.quick_colour(img)
    xlabel = "wavelength (nm)" if has_nm else "spectral column (blue → red, uncalibrated)"
    print(f"{'calibrated reflectance' if calibrated else 'uncalibrated counts'}, "
          f"{n_b} usable spectral bands ({grid[0]:.0f}–{grid[-1]:.0f} {'nm' if has_nm else 'columns'})")

    out = os.path.join(args.scan, "discrimination")
    os.makedirs(out, exist_ok=True)
    flat = img.reshape(-1, n_b)
    # Near-black pixels (black paint, shadows) have almost no signal, so their
    # spectral SHAPE is just noise/stray light; leave them out (unclassified).
    total = np.nansum(flat, axis=1)
    ok = ~np.isnan(flat).any(1) & (total > args.min_signal * np.nanpercentile(total, 99))

    if args.auto:
        return auto_clusters(args, img, flat, ok, rgb, grid, xlabel, calibrated, out)

    regions = [parse_region(r, n_az) for r in args.region]
    if args.pick:
        regions += pick_regions(rgb)
    if len(regions) < 2:
        raise SystemExit("need at least two regions (--region ... or --pick), or use --auto K")
    names = [r[0] for r in regions]
    k = len(names)

    proj = three_band(calibrated, grid, n_b)
    train_h, test_h, train_3, test_3 = [], [], [], []
    for _, (c0, c1, h0, h1) in regions:
        block = img[h0:h1, c0:c1]                   # (h, az, bands)
        even, odd = block[:, 0::2].reshape(-1, n_b), block[:, 1::2].reshape(-1, n_b)
        even, odd = even[~np.isnan(even).any(1)], odd[~np.isnan(odd).any(1)]
        train_h.append(even)
        test_h.append(odd)
        train_3.append(even @ proj)
        test_3.append(odd @ proj)

    acc_h, conf_h, means = evaluate(train_h, test_h, k)
    acc_3, conf_3, _ = evaluate(train_3, test_3, k)
    sam_matrix = np.array([[float(spectral.spectral_angle(a, b)) for b in means] for a in means])

    res = {"calibrated": calibrated, "classes": names, "accuracy_hyperspectral": acc_h, "accuracy_rgb_like": acc_3,
           "confusion_hyperspectral": conf_h, "confusion_rgb_like": conf_3,
           "class_mean_angle_deg": sam_matrix,
           "regions": {n: list(map(int, b)) for n, b in regions}}
    print(f"held-out accuracy: hyperspectral {100 * acc_h:.1f}%   RGB-like 3-band {100 * acc_3:.1f}%")
    print("spectral angle between class means (deg):")
    print("        " + "".join(f"{n[:9]:>10}" for n in names))
    for n, row in zip(names, sam_matrix):
        print(f"{n[:8]:<8}" + "".join(f"{v:10.2f}" for v in row))

    lab = np.full(len(flat), -1)
    l, a = nearest_class(flat[ok], means)
    l[a > args.max_angle] = -1
    lab[ok] = l
    lab = lab.reshape(n_h, n_az)
    res["fraction_classified"] = float(np.mean(lab >= 0))
    store.save_json(os.path.join(out, "results.json"), res)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0, 0]
    ax.imshow(rgb, aspect="auto", extent=[0, 360, n_h, 0])
    for i, (n, (c0, c1, h0, h1)) in enumerate(regions):
        c = plotting.SERIES[i % len(plotting.SERIES)]
        ax.add_patch(plt.Rectangle((c0 * 360 / n_az, h0), (c1 - c0) * 360 / n_az, h1 - h0, fill=False, ec=c, lw=2))
        ax.annotate(n, (c0 * 360 / n_az, h0), textcoords="offset points", xytext=(2, 4), color=plotting.INK,
                    fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    ax.set_title("Panorama with labelled regions")
    ax.grid(False)
    plot_shapes(axes[0, 1], grid, [np.vstack([a, b]) for a, b in zip(train_h, test_h)], names, calibrated, xlabel)
    ax = axes[1, 0]
    ax.imshow(class_map_image(lab), aspect="auto", extent=[0, 360, n_h, 0], interpolation="nearest")
    ax.set_title(f"Class map (grey = unclassified, > {args.max_angle:g}°)")
    ax.set_xlabel("azimuth (deg)")
    ax.grid(False)
    ax = axes[1, 1]
    x = np.arange(2)
    ax.bar(x, [100 * acc_h, 100 * acc_3], width=0.5, color=[plotting.SERIES[0], plotting.MUTED])
    ax.set_xticks(x, ["full spectrum", "RGB-like (3 bands)"])
    for i, v in enumerate([acc_h, acc_3]):
        ax.annotate(f"{100 * v:.1f}%", (i, 100 * v), textcoords="offset points", xytext=(0, 4), ha="center")
    ax.set(ylabel="held-out accuracy (%)", ylim=(0, 110), title="Discrimination accuracy")
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "discrimination.png"))


def auto_clusters(args, img, flat, ok, rgb, grid, xlabel, calibrated, out):
    n_h, n_az, n_b = img.shape
    # cluster on SHAPE: each spectrum scaled to unit length (that's what the
    # spectral angle measures), from a random subsample for speed
    unit = flat[ok] / np.linalg.norm(flat[ok], axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    sample = unit[rng.choice(len(unit), min(30000, len(unit)), replace=False)]
    centres, _ = kmeans2(sample, args.auto, minit="++", seed=0)
    l, a = nearest_class(unit, centres)
    lab = np.full(len(flat), -1)
    lab[ok] = l
    lab = lab.reshape(n_h, n_az)
    names = [f"cluster {i + 1}" for i in range(args.auto)]
    members = [flat[ok][l == i] for i in range(args.auto)]
    sam_matrix = np.array([[float(spectral.spectral_angle(a_, b_)) for b_ in centres] for a_ in centres])
    sizes = [100 * len(m) / ok.sum() for m in members]
    res = {"mode": "auto", "calibrated": calibrated, "k": args.auto, "cluster_pct": sizes,
           "centre_angle_deg": sam_matrix,
           "median_angle_to_own_centre_deg": float(np.median(a))}
    store.save_json(os.path.join(out, "results_auto.json"), res)
    print(f"{args.auto} clusters; sizes " + ", ".join(f"{s:.0f}%" for s in sizes))
    print(f"median angle of a pixel to its own cluster centre: {res['median_angle_to_own_centre_deg']:.2f} deg")
    print("spectral angle between cluster centres (deg):")
    for n, row in zip(names, sam_matrix):
        print(f"{n:<10}" + "".join(f"{v:7.2f}" for v in row))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={"width_ratios": [1.4, 1]})
    axes[0, 0].imshow(rgb, aspect="auto", extent=[0, 360, n_h, 0])
    axes[0, 0].set_title("Panorama")
    axes[1, 0].imshow(class_map_image(lab), aspect="auto", extent=[0, 360, n_h, 0], interpolation="nearest")
    axes[1, 0].set(title=f"{args.auto} spectral clusters (no labels; grey = too dark to use)",
                   xlabel="azimuth (deg)")
    for ax in axes[:, 0]:
        ax.grid(False)
    shown = [i for i in range(args.auto) if sizes[i] >= 0.5]  # tiny clusters are noise
    plot_shapes(axes[0, 1], grid, [members[i] for i in shown], [names[i] for i in shown], calibrated, xlabel,
                colours=[plotting.SERIES[i % len(plotting.SERIES)] for i in shown])  # match the map
    ax = axes[1, 1]
    im = ax.imshow(sam_matrix, cmap="Blues")
    ax.set_xticks(range(args.auto), [str(i + 1) for i in range(args.auto)])
    ax.set_yticks(range(args.auto), [str(i + 1) for i in range(args.auto)])
    for i in range(args.auto):
        for j in range(args.auto):
            ax.text(j, i, f"{sam_matrix[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if sam_matrix[i, j] > sam_matrix.max() * 0.6 else plotting.INK)
    ax.set(title="Spectral angle between clusters (deg)", xlabel="cluster", ylabel="cluster")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "clusters.png"))


if __name__ == "__main__":
    main()
