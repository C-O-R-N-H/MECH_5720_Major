"""
Experiment: processing comparisons
====================================
The brief: "where alternative calibration, reconstruction, or processing
methods are investigated, they will be compared quantitatively". Each
comparison below re-processes data that already exists in two or more ways
and measures the difference. No hardware needed.

  1 period      turntable period: assumed label vs period search vs
                search + all-turn alignment fit -> turn-to-turn misalignment
  2 merging     one turn vs all turns merged -> angular step, roughness
  3 dark        dark subtraction off vs on -> contrast, separation between
                materials (found automatically by spectral shape)
  4 averaging   1..25 captures averaged -> noise vs N (expect 1/sqrt(N))
  5 binning     rows / spectral columns binned 1..16 -> SNR vs resolution
  6 stitching   nearest line vs linear interpolation -> roughness, rev-to-rev
  7 colour      single column per channel vs averaged thirds -> image noise

Inputs (defaults): captures/speed_comparison (three scans),
captures/repeatability (runs with segment_spectra.npz), calibration/dark.npz.
Output: captures/processing_comparisons/<stamp>/  one PNG per comparison,
results.json, summary.csv (comparison, option, metric, value).

    python exp_processing_comparisons.py              # all
    python exp_processing_comparisons.py --only 3,5   # some
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

from hsi import colour, plotting, spectral, store, turntable
from hsi.plotting import plt

ROWS = []  # summary.csv rows


def record(comparison, option, metric, value):
    ROWS.append({"comparison": comparison, "option": option, "metric": metric,
                 "value": None if value is None else round(float(value), 4)})


# ---------------------------------------------------------------- loading ---

class Scan:
    """A speed-comparison scan with and without dark subtraction, and its
    measured (refined) period."""

    def __init__(self, path, dark_full):
        self.path = path
        self.info = json.load(open(os.path.join(path, "scan.json")))
        self.raw = np.load(os.path.join(path, "frames.npy"), mmap_mode="r")
        self.times = np.load(os.path.join(path, "times.npy"))
        rb = self.info.get("row_bin", 1)
        self.dark = spectral.bin_rows(spectral.crop(dark_full, self.info["roi"]), rb)
        self.lines = np.asarray(self.raw, np.float32) - self.dark
        self.feats = turntable.frame_features(self.lines)
        self.period_search, _ = turntable.find_period(self.times, self.feats)
        self.period = turntable.refine_period(self.times, self.feats, self.period_search)
        self.theta = turntable.angles(self.times, self.period)
        self.revs = self.theta[-1] / 360
        self.lines_per_turn = len(self.times) / self.revs
        phi = np.sort(np.mod(self.theta, 360))
        self.max_gap = float(np.max(np.diff(np.concatenate([phi, phi[:1] + 360]))))
        self.name = f"{self.period:.1f} s/turn"

    def theta_for(self, period):
        return turntable.angles(self.times, period)


def roughness(pano):
    """Median neighbour difference along azimuth / median signal."""
    d = np.nanmedian(np.abs(np.diff(pano, axis=0)))
    return float(d / (np.nanmedian(np.abs(pano)) + 1e-9))


def rev_noise(p0, p1):
    """Noise from two independent passes over the same surface: robust std of
    their difference / sqrt(2). MAD keeps edges (slight misalignment) from
    dominating."""
    d = (p1 - p0).ravel()
    d = d[np.isfinite(d)]
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def bin_axis(a, b, axis):
    n = a.shape[axis] - a.shape[axis] % b
    a = np.take(a, np.arange(n), axis=axis)
    shape = list(a.shape)
    shape[axis:axis + 1] = [n // b, b]
    return a.reshape(shape).mean(axis=axis + 1)


# ------------------------------------------------------------ comparisons ---

def c1_period(scans, out):
    labels = (30.0, 15.0, 8.0, 5.0)
    res = {}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    methods = ["assumed label", "period search", "search + turn fit"]
    width = 0.26
    for j, sc in enumerate(scans):
        label = min(labels, key=lambda l: abs(l - sc.period))
        r = {}
        for m, p in zip(methods, (label, sc.period_search, sc.period)):
            off = turntable.turn_offsets(sc.lines, sc.theta_for(p))
            rms = float(np.sqrt(np.mean(off ** 2))) if len(off) else np.nan
            r[m] = {"period_s": p, "turn_offset_rms_deg": rms,
                    "turn_offset_max_deg": float(np.max(np.abs(off))) if len(off) else np.nan}
            record("1 period", f"{sc.name}: {m}", "turn_offset_rms_deg", rms)
        res[sc.name] = r
        for i, m in enumerate(methods):
            v = max(r[m]["turn_offset_rms_deg"], 1e-3)
            x = j + (i - 1) * width
            ax.bar(x, v, width * 0.9, color=plotting.SERIES[i], label=m if j == 0 else None)
            ax.annotate(f"{r[m]['turn_offset_rms_deg']:.2f}°", (x, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(scans)), [sc.name for sc in scans])
    ax.set(ylabel="turn-to-turn misalignment, RMS (deg, log)",
           title="1 · Turntable period method")
    ax.legend()
    plotting.save(fig, os.path.join(out, "1_period.png"))
    return res


def c2_merging(scans, out):
    res = {}
    fig, axes = plt.subplots(len(scans), 2, figsize=(12, 2.6 * len(scans)))
    for k, sc in enumerate(scans):
        n1 = int(round(sc.lines_per_turn))
        nm = int(360 / sc.max_gap)
        single, _, _ = turntable.stitch(sc.lines, sc.theta, n1, rev=0)
        merged, _, _ = turntable.stitch(sc.lines, sc.theta, nm)
        merged_same, _, _ = turntable.stitch(sc.lines, sc.theta, n1)
        r = {"deg_per_line_one_turn": 360 / sc.lines_per_turn, "max_gap_merged_deg": sc.max_gap,
             "turns": sc.revs, "zipper_ratio": roughness(merged_same) / roughness(single)}
        res[sc.name] = r
        record("2 merging", f"{sc.name}: one turn", "angular_step_deg", r["deg_per_line_one_turn"])
        record("2 merging", f"{sc.name}: merged", "angular_step_deg", sc.max_gap)
        record("2 merging", f"{sc.name}: merged", "zipper_ratio", r["zipper_ratio"])
        # same 90 deg of azimuth from each, upsampled to the merged grid for display
        for j, (p, n, title) in enumerate(((single, n1, f"one turn ({360 / sc.lines_per_turn:.1f}°/line)"),
                                           (merged, nm, f"{sc.revs:.1f} turns merged "
                                                        f"(largest gap {sc.max_gap:.1f}°)"))):
            a0, a1 = int(0.25 * n), int(0.5 * n)
            img = colour.quick_colour(np.transpose(p[a0:a1], (1, 0, 2)))
            ax = axes[k, j]
            ax.imshow(img, aspect="auto", interpolation="nearest", extent=[90, 180, img.shape[0], 0])
            ax.set_title(f"{sc.name} — {title}", loc="left", fontsize=9)
            ax.set_yticks([])
            ax.grid(False)
    axes[-1, 0].set_xlabel("azimuth (deg)")
    axes[-1, 1].set_xlabel("azimuth (deg)")
    fig.suptitle("2 · One turn vs all turns merged (same 90° of the object)", fontweight="bold")
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "2_merging.png"))
    return res


def material_labels(cube, k=4, min_signal=0.15):
    """Group the panorama's pixels into k materials by spectral SHAPE
    (k-means on unit-length spectra), ignoring near-black pixels. Returns
    (labels (az, height) with -1 = too dark, cluster sizes). Done once on the
    dark-subtracted cube so every option is compared on the same pixels."""
    from scipy.cluster.vq import kmeans2

    flat = cube.reshape(-1, cube.shape[-1]).astype(np.float64)
    total = np.nansum(flat, axis=1)
    ok = np.isfinite(flat).all(1) & (total > min_signal * np.nanpercentile(total, 99))
    unit = flat[ok] / np.linalg.norm(flat[ok], axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    centres, _ = kmeans2(unit[rng.choice(len(unit), min(30000, len(unit)), replace=False)], k, minit="++", seed=0)
    lab = np.full(len(flat), -1)
    lab[ok] = np.argmax(unit @ centres.T, axis=1)
    return lab.reshape(cube.shape[:2])


def c3_dark(sc, out):
    nm = int(360 / sc.max_gap)
    with_dark, _, _ = turntable.stitch(sc.lines, sc.theta, nm)
    raw = np.asarray(sc.raw, np.float32)
    no_dark, _, _ = turntable.stitch(raw, sc.theta, nm)
    labels = material_labels(with_dark)        # same pixels for both options
    res = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, (name, cube) in enumerate((("no dark subtraction", no_dark), ("dark subtracted", with_dark))):
        img = np.transpose(cube, (1, 0, 2))
        means = {f"material {m + 1}": np.nanmean(cube[labels == m], axis=0)
                 for m in range(labels.max() + 1) if np.any(labels == m)}
        names = list(means)
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        angles = {f"{a}-{b}": float(spectral.spectral_angle(means[a], means[b])) for a, b in pairs}
        broad = np.nanmean(img, axis=2)
        lo, hi = np.percentile(broad, [1, 99])
        r = {"pair_angles_deg": angles, "min_pair_angle_deg": min(angles.values()),
             "mean_pair_angle_deg": float(np.mean(list(angles.values()))),
             "contrast": float((hi - lo) / hi)}
        res[name] = r
        for k in ("min_pair_angle_deg", "mean_pair_angle_deg", "contrast"):
            record("3 dark", name, k, r[k])
        ax = axes[j]
        for i, m in enumerate(names):
            ax.plot(means[m] / np.mean(means[m]), color=plotting.SERIES[i], label=m)
        ax.set(xlabel="spectral column (blue → red)", ylabel="relative shape (÷ own mean)",
               title=f"{name}: mean pair angle {r['mean_pair_angle_deg']:.1f}°")
        ax.legend(fontsize=8)
    ax = axes[2]
    keys = list(res["dark subtracted"]["pair_angles_deg"])
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [res["no dark subtraction"]["pair_angles_deg"][k] for k in keys], 0.4,
           color=plotting.MUTED, label="no dark subtraction")
    ax.bar(x + 0.2, [res["dark subtracted"]["pair_angles_deg"][k] for k in keys], 0.4,
           color=plotting.SERIES[0], label="dark subtracted")
    ax.set_xticks(x, keys, rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="spectral angle between materials (deg)", title="Material separation")
    ax.legend(fontsize=8)
    fig.suptitle(f"3 · Dark subtraction ({sc.name} scan)", fontweight="bold")
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "3_dark.png"))
    return res


def c4_averaging(rep_runs, out):
    Ns = [1, 2, 5, 10, 25]
    res = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (seg, label) in enumerate((("whole_line", "whole line (bright)"), ("segment_4", "bottom segment (dim)"))):
        cvs = {N: [] for N in Ns}
        for run in rep_runs:
            d = np.load(run)
            spec, use = d[seg], d["use"]
            for N in Ns:
                g = len(spec) // N
                if g < 2:
                    continue
                means = spec[: g * N].reshape(g, N, -1).mean(1)[:, use]
                cvs[N].append(np.median(100 * means.std(0, ddof=1) / np.abs(means.mean(0))))
        cv = np.array([np.mean(cvs[N]) for N in Ns])
        res[seg] = {f"N={N}": float(c) for N, c in zip(Ns, cv)}
        for N, c in zip(Ns, cv):
            record("4 averaging", f"{seg}, N={N}", "variation_pct", c)
        c = plotting.SERIES[i]
        ax.plot(Ns, cv, "o-", ms=8, color=c, label=f"{label}: measured")
        ax.plot(Ns, cv[0] / np.sqrt(Ns), "--", color=c, lw=1.5, label=f"{label}: 1/√N")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.set(xlabel="captures averaged (N)", ylabel="variation per spectral column (%)",
           title="4 · Averaging captures of a still target")
    ax.legend(fontsize=8)
    plotting.save(fig, os.path.join(out, "4_averaging.png"))
    return res


def c5_binning(sc, out):
    n1 = int(round(sc.lines_per_turn))
    p0, _, _ = turntable.stitch(sc.lines, sc.theta, n1, rev=0)   # (az, rows, cols)
    p1, _, _ = turntable.stitch(sc.lines, sc.theta, n1, rev=1)
    bins = [1, 2, 4, 8, 16]
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (axis, label, unit) in enumerate(((1, "rows along the slit", "height"),
                                             (2, "spectral columns", "spectral"))):
        snr = []
        for b in bins:
            a0, a1 = bin_axis(p0, b, axis), bin_axis(p1, b, axis)
            s = float(np.nanmedian((a0 + a1) / 2))
            snr.append(s / rev_noise(a0, a1))
            record("5 binning", f"{label} x{b}", "snr", snr[-1])
        res[label] = {f"x{b}": v for b, v in zip(bins, snr)}
        c = plotting.SERIES[i]
        axes[0].plot(bins, snr, "o-", ms=8, color=c, label=f"binning {label}")
        axes[0].plot(bins, snr[0] * np.sqrt(bins), "--", color=c, lw=1.5, label=f"ideal √bin ({unit})")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(bins, [str(b) for b in bins])
    axes[0].set(xlabel="bin factor", ylabel="signal-to-noise (rev 1 vs rev 2)",
                title="SNR vs binning")
    axes[0].legend(fontsize=8)
    # what spectral binning costs: the brightest material's spectrum at several bins
    labels = material_labels(p0)
    m = max(range(labels.max() + 1), key=lambda k: np.nansum(np.nanmean(p0[labels == k], axis=0)))
    spec = np.nanmean(p0[labels == m], axis=0)
    for k, b in enumerate((1, 4, 16)):
        sb = bin_axis(spec[None], b, 1)[0]
        axes[1].plot((np.arange(len(sb)) + 0.5) * b, sb, color=plotting.SERIES[k], lw=1.5 if b == 1 else 2,
                     label=f"x{b} ({len(sb)} bands)")
    axes[1].set(xlabel="spectral column (blue → red)", ylabel="counts",
                title="Brightest material's spectrum at different spectral binning")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"5 · Binning ({sc.name} scan)", fontweight="bold")
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "5_binning.png"))
    return res


def c6_stitching(scans, out):
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for j, sc in enumerate(scans):
        n1 = int(round(sc.lines_per_turn))
        r = {}
        for m in ("nearest", "linear"):
            p0, _, _ = turntable.stitch(sc.lines, sc.theta, n1, rev=0, method=m)
            p1, _, _ = turntable.stitch(sc.lines, sc.theta, n1, rev=1, method=m)
            merged, _, _ = turntable.stitch(sc.lines, sc.theta, int(360 / sc.max_gap), method=m)
            r[m] = {"rev_to_rev_noise": rev_noise(p0, p1) / float(np.nanmedian(p0)),
                    "roughness_merged": roughness(merged)}
            record("6 stitching", f"{sc.name}: {m}", "rev_to_rev_noise_rel", r[m]["rev_to_rev_noise"])
            record("6 stitching", f"{sc.name}: {m}", "roughness_merged", r[m]["roughness_merged"])
        res[sc.name] = r
        for i, (k, title) in enumerate((("rev_to_rev_noise", "Rev 1 vs rev 2 noise (÷ signal)"),
                                        ("roughness_merged", "Neighbour-column roughness, merged"))):
            for q, m in enumerate(("nearest", "linear")):
                axes[i].bar(j + (q - 0.5) * 0.38, r[m][k], 0.36, color=[plotting.MUTED, plotting.SERIES[0]][q],
                            label=m if j == 0 else None)
            axes[i].set_title(title)
    for ax in axes:
        ax.set_xticks(range(len(scans)), [sc.name for sc in scans])
        ax.legend()
    fig.suptitle("6 · Stitching: nearest line vs linear interpolation", fontweight="bold")
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "6_stitching.png"))
    return res


def c7_colour(sc, out):
    nm = int(360 / sc.max_gap)
    cube, _, _ = turntable.stitch(sc.lines, sc.theta, nm)
    img = np.transpose(cube, (1, 0, 2))
    grid = np.arange(img.shape[-1])
    renders = {"single column per channel": colour.false_colour(img, grid, np.percentile(grid, [80, 50, 20])),
               "averaged thirds": colour.quick_colour(img)}
    res = {}
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, (name, rgb) in zip(axes, renders.items()):
        # pixel noise: median jump between vertically neighbouring pixels
        # (within a square the true colour barely changes down the slit)
        noise = float(np.median(np.abs(np.diff(rgb.astype(float), axis=0))) / 255 * 100)
        res[name] = {"pixel_noise_pct_of_range": noise}
        record("7 colour", name, "pixel_noise_pct", noise)
        ax.imshow(rgb, aspect="auto", extent=[0, 360, rgb.shape[0], 0])
        ax.set_title(f"{name}: pixel noise {noise:.1f}% of range", loc="left")
        ax.set_yticks([])
        ax.grid(False)
    axes[-1].set_xlabel("azimuth (deg)")
    fig.suptitle(f"7 · False-colour rendering ({sc.name} scan)", fontweight="bold")
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "7_colour.png"))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scans", default=os.path.join(store.CAPTURES_DIR, "speed_comparison"))
    ap.add_argument("--repeatability", default=os.path.join(store.CAPTURES_DIR, "repeatability"))
    ap.add_argument("--only", default="1,2,3,4,5,6,7")
    ap.add_argument("--out", default=None, help="existing results folder to add these comparisons to")
    args = ap.parse_args()
    only = {int(x) for x in args.only.split(",")}

    # --out adds to an existing results folder, so the comparisons can be run
    # in batches (each batch loads every scan; all at once can exhaust RAM)
    out = args.out or store.new_run_dir("processing_comparisons")
    previous = {}
    if args.out and os.path.exists(os.path.join(out, "results.json")):
        previous = json.load(open(os.path.join(out, "results.json")))
        with open(os.path.join(out, "summary.csv")) as f:
            ROWS.extend(r for r in csv.DictReader(f) if int(r["comparison"].split()[0]) not in only)
    dark_full = np.load(store.calib_path("dark.npz"))["dark"]
    scans = []
    if only & {1, 2, 3, 5, 6, 7}:
        for p in sorted(glob.glob(os.path.join(args.scans, "*"))):
            if os.path.exists(os.path.join(p, "frames.npy")):
                print(f"loading {os.path.basename(p)}")
                scans.append(Scan(p, dark_full))
        scans.sort(key=lambda s: -s.period)
    slow = scans[0] if scans else None
    multi = [s for s in scans if s.revs >= 2]

    results = previous
    steps = {1: ("period", lambda: c1_period(multi, out)),
             2: ("merging", lambda: c2_merging(scans, out)),
             3: ("dark", lambda: c3_dark(slow, out)),
             4: ("averaging", lambda: c4_averaging(
                 sorted(glob.glob(os.path.join(args.repeatability, "*", "segment_spectra.npz"))), out)),
             5: ("binning", lambda: c5_binning(slow, out)),
             6: ("stitching", lambda: c6_stitching(multi, out)),
             7: ("colour", lambda: c7_colour(slow, out))}
    for k in sorted(only):
        name, fn = steps[k]
        print(f"--- {k} {name}")
        results[name] = fn()

    store.save_json(os.path.join(out, "results.json"), results)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["comparison", "option", "metric", "value"])
        w.writeheader()
        w.writerows(ROWS)
    print()
    ROWS.sort(key=lambda r: int(r["comparison"].split()[0]))
    for r in ROWS:
        print(f"{r['comparison']:<12} {r['option']:<42} {r['metric']:<24} {r['value']}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
