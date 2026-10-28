"""
Experiment: turntable speed comparison
========================================
One scan per speed setting, same object and lighting, then a summary:

    python scan.py --experiment speed_comparison --tag setting1   (repeat per setting)
    python exp_consistency.py <each scan>
    python exp_speed_comparison.py ../captures/speed_comparison

Writes into that folder:
    summary.csv / summary.json   one row per scan (numbers for the report)
    summary_panoramas.png        the panoramas aligned + a zoomed crop
    summary_metrics.png          angular step, exposure smear and turn wobble by speed

Per scan:
  period_s              measured from the scan
  turns                 revolutions captured
  lines_per_turn        frame rate x period
  deg_per_line_1turn    angular step of a single revolution
  max_gap_deg_merged    largest angular gap with all revolutions merged
  blur_deg              rotation during one exposure (360 * exposure / period)
  rev12_offset_deg      revolution 2 vs 1 misalignment (exp_consistency.py)
  turn_wobble_deg_rms   scatter of every turn's alignment vs turn 1
  rev12_sam_deg         spectral difference rev 2 vs 1, usable columns only
  zipper_ratio          neighbour-column roughness merged / single turn (exp_consistency.py)
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

from hsi import colour, pipeline, plotting, spectral, turntable
from hsi.plotting import plt


def usable_columns(lines, min_frac=0.2):
    level = np.nanmedian(lines.reshape(-1, lines.shape[-1]), axis=0)
    return level >= min_frac * np.nanmax(level)


def analyse(scan):
    info = json.load(open(os.path.join(scan, "scan.json")))
    d = pipeline.load_calibrated_scan(scan, "estimate")
    lines, theta, period = d["lines"], d["theta"], d["period"]
    rate = info["n_frames"] / info["duration_s"]
    offsets = turntable.turn_offsets(lines, theta)
    cube = np.load(os.path.join(scan, "cube.npz"))
    cons_path = os.path.join(scan, "consistency", "results.json")
    cons = json.load(open(cons_path))["estimate"] if os.path.exists(cons_path) else {}

    sam = None
    if theta[-1] >= 720:
        use = usable_columns(lines)
        n_az = int(round(len(theta) / (theta[-1] / 360)))
        p0, _, _ = turntable.stitch(lines[..., use], theta, n_az, rev=0)
        p1, _, _ = turntable.stitch(lines[..., use], theta, n_az, rev=1)
        sam = float(np.nanmean(spectral.spectral_angle(p0, p1)))

    row = {
        "scan": os.path.basename(scan.rstrip("/")),
        "period_s": round(period, 4),
        "turns": round(theta[-1] / 360, 2),
        "lines": len(theta),
        "rate_hz": round(rate, 2),
        "lines_per_turn": round(rate * period, 1),
        "deg_per_line_1turn": round(360 / (rate * period), 2),
        "max_gap_deg_merged": round(float(cube["max_gap_deg"]), 2),
        "blur_deg": round(360 * (info.get("exposure_us") or 0) * 1e-6 / period, 2),
        "rev12_offset_deg": round(cons.get("azimuth_offset_deg", np.nan), 2),
        "turn_wobble_deg_rms": round(float(np.sqrt(np.mean(offsets ** 2))), 2) if len(offsets) else None,
        "rev12_sam_deg": round(sam, 2) if sam is not None else None,
        "zipper_ratio": round(cons.get("zipper_ratio", np.nan), 2),
    }
    return row, cube["cube"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--width", type=int, default=360, help="common azimuth samples for the figure")
    args = ap.parse_args()

    scans = sorted(p for p in glob.glob(os.path.join(args.folder, "*")) if os.path.exists(os.path.join(p, "cube.npz"))
                   and not os.path.basename(p).startswith("REJECTED"))
    results = [analyse(s) for s in scans]
    results.sort(key=lambda r: -r[0]["period_s"])  # slowest first
    rows = [r for r, _ in results]

    with open(os.path.join(args.folder, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(os.path.join(args.folder, "summary.json"), "w"), indent=2)
    for r in rows:
        print(r)

    imgs, h = aligned_panoramas([c for _, c in results], args.width)

    # --- figure 1: the panoramas + a zoomed crop -------------------------
    n = len(rows)
    fig = plt.figure(figsize=(14, 3.0 * n))
    gs = fig.add_gridspec(n, 2, width_ratios=[3, 1])
    z0, z1 = zoom_window(args.width)
    for i, (r, img) in enumerate(zip(rows, imgs)):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img, aspect="auto", extent=[0, 360, h, 0])
        ax.add_patch(plt.Rectangle((z0 * 360 / args.width, 2), (z1 - z0) * 360 / args.width, h - 4,
                                   fill=False, ec="white", lw=1.2))
        ax.set_title(f"{r['period_s']:.2f} s/turn — {r['lines_per_turn']:.0f} lines/turn, "
                     f"{r['turns']:.1f} turns merged, largest gap {r['max_gap_deg_merged']:.1f}°", loc="left")
        ax.set_yticks([])
        ax.grid(False)
        if i == n - 1:
            ax.set_xlabel("azimuth (deg)")
        axz = fig.add_subplot(gs[i, 1])
        axz.imshow(img[:, z0:z1], aspect="auto", interpolation="nearest")
        axz.set_title("zoom (boxed region)", loc="left")
        axz.set_xticks([])
        axz.set_yticks([])
        axz.grid(False)
    fig.tight_layout()
    plotting.save(fig, os.path.join(args.folder, "summary_panoramas.png"))

    # --- figure 2: the sampling / smear / wobble numbers -------------------
    labels = [f"{r['period_s']:.1f} s" for r in rows]
    x = np.arange(n)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    ax.bar(x - 0.18, [r["deg_per_line_1turn"] for r in rows], 0.36, color=plotting.SERIES[1], label="one turn")
    ax.bar(x + 0.18, [r["max_gap_deg_merged"] for r in rows], 0.36, color=plotting.SERIES[0], label="all turns merged")
    ax.set_xticks(x, labels)
    ax.set(ylabel="degrees", title="Angular step (lower = finer)")
    ax.legend()
    ax = axes[1]
    ax.bar(x, [r["blur_deg"] for r in rows], 0.5, color=plotting.SERIES[0])
    ax.set_xticks(x, labels)
    ax.set(ylabel="degrees", title="Smear during one exposure")
    ax = axes[2]
    wob = [r["turn_wobble_deg_rms"] or 0 for r in rows]
    ax.bar(x, wob, 0.5, color=plotting.SERIES[0])
    ax.set_xticks(x, labels)
    ax.set(ylabel="degrees RMS", title="Turn-to-turn alignment scatter")
    for i, v in enumerate(wob):
        ax.annotate(f"{v:.2f}°", (i, v), textcoords="offset points", xytext=(0, 3), ha="center")
    fig.tight_layout()
    plotting.save(fig, os.path.join(args.folder, "summary_metrics.png"))


def zoom_window(width):
    """Azimuth columns of the zoomed crop shown next to each panorama."""
    return int(width * 0.30), int(width * 0.45)


def aligned_panoramas(cubes, width=360):
    """False-colour renders of several panorama cubes on a common grid:
    cropped to the same height, resampled to `width` azimuth samples and
    rotated to line up with the first (slowest) scan. Returns (images, height)."""
    h = min(c.shape[1] for c in cubes)
    imgs, ref = [], None
    for cube in cubes:
        c = cube[:, :h]                                       # same height
        idx = np.linspace(0, c.shape[0], width, endpoint=False).astype(int)
        c = c[idx]                                            # same width
        broad = np.nan_to_num(np.nanmean(c, axis=2))
        if ref is None:
            ref = broad
        else:
            c = np.roll(c, -int(round(turntable.circ_shift(ref, broad))), axis=0)
        imgs.append(colour.quick_colour(np.transpose(c, (1, 0, 2))))
    return imgs, h


if __name__ == "__main__":
    main()
