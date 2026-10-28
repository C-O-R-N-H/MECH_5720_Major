"""
Report figures
===============
Regenerates every figure in the paper (latex/figures/*.pdf) from the data in
captures/ and calibration/. Each figure is sized for the two-column ICCP
layout: one column is 3.45 in wide, a full-width figure 7.1 in.

    python make_report_figures.py            # all figures
    python make_report_figures.py lsf snr    # just some

Inputs (the 26 Oct session):
    captures/snapshot/*_white_card/          white-card snapshot
    captures/cellophane/<run>/               cellophane filter captures
    captures/line_spread/<run>/edge_best.npz edge for the line spread function
    captures/exposure_sweep/<run>/           exposure / gain sweep
    captures/speed_comparison/               the three turntable scans
    captures/processing_comparisons/<run>/   period-method comparison
    calibration/dark.npz                     session dark frame
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

from hsi import colour, pipeline, plotting, store, turntable
from hsi.plotting import plt

import cal_cellophane
import exp_line_spread
import exp_speed_comparison

OUT = os.path.join(store.REPO_ROOT, "latex", "figures")
COL = 3.45                                    # one column, inches
FULL = 7.1                                    # both columns
# identity colours where a curve IS a colour (pixel type, filter, spectral third)
HUE = {"blue": "#2a6fd6", "green": "#2e9e4f", "red": "#d03b3b", "all": plotting.INK}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.3,
    "axes.titleweight": "normal", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def latest(pattern):
    paths = sorted(glob.glob(os.path.join(store.CAPTURES_DIR, pattern)))
    if not paths:
        raise SystemExit(f"no data matching captures/{pattern}")
    return paths[-1]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def session_dark():
    return np.load(store.calib_path("dark.npz"))["dark"]


# ---------------------------------------------------------------- figures ---

def fig_snapshot():
    """Raw spectrometer frame of the white card (colour) and the spectrum at
    four heights along the slit."""
    run = latest("snapshot/*_white_card")
    rgb = cv2.cvtColor(cv2.imread(os.path.join(run, "frame_0.png")), cv2.COLOR_BGR2RGB)
    frames = np.load(os.path.join(run, "frames.npz"))["frames"]
    mean = frames.mean(axis=0)
    h, w = mean.shape
    rows = (np.arange(1, 5) * h / 5).astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(COL, 4.0), gridspec_kw={"height_ratios": [1, 1.1]})
    ax = axes[0]
    ax.imshow(rgb, aspect="auto", extent=[0, w, h, 0])
    for i, r in enumerate(rows):
        ax.axhline(r, color=plotting.SERIES[i], lw=1)
    ax.set(xlabel="Spectral column (blue → red)", ylabel="Row along slit")
    ax.grid(False)
    ax = axes[1]
    for i, r in enumerate(rows):
        ax.plot(mean[r - 4: r + 4].mean(axis=0), color=plotting.SERIES[i], lw=1, label=f"Row {r}")
    ax.set(xlabel="Spectral column (blue → red)", ylabel="Signal (counts)", xlim=(0, w))
    ax.legend(ncol=2)
    fig.tight_layout(h_pad=0.6)
    save(fig, "snapshot")


def fig_lsf():
    """Line spread function from the knife edge, whole spectrum and per third."""
    run = latest("line_spread/*")
    img = np.load(os.path.join(run, "edge_best.npz"))["frame"] - session_dark()
    level = np.median(np.sort(img, axis=0)[-20:], axis=0)
    use = np.where(level >= 0.2 * level.max())[0]
    groups = {"all": use, **dict(zip(("blue", "green", "red"), np.array_split(use, 3)))}
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for name, cols in groups.items():
        edges, _, slope = exp_line_spread.find_edges(img[:, cols].mean(axis=1))
        e = edges[0]
        k = np.arange(len(slope)) - e["row"]
        win = np.abs(k) <= 60
        label = ("Whole spectrum" if name == "all" else f"{name.capitalize()} third") + f": {e['fwhm_px']:.0f} px"
        ax.plot(k[win], slope[win] / slope[win].max(), color=HUE[name], lw=1.6 if name == "all" else 1.1,
                ls="-" if name == "all" else "--", label=label)
    ax.axhline(0.5, color=plotting.MUTED, lw=0.8, ls=":")
    ax.set(xlabel="Rows from the edge (along the slit)", ylabel="Normalised LSF", xlim=(-60, 60))
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28), title="FWHM", title_fontsize=7)
    save(fig, "lsf")


def fig_snr():
    """Single-pixel signal-to-noise against signal for the exposure and gain sweeps."""
    run = latest("exposure_sweep/*")
    rows = json.load(open(os.path.join(run, "results.json")))["settings"]
    ex = sorted((r for r in rows if r["sweep"] == "exposure"), key=lambda r: r["exposure_ms"])
    gn = sorted((r for r in rows if r["gain"] != 8 or r["exposure_ms"] == 100), key=lambda r: r["gain"])
    gn = [r for r in gn if r["exposure_ms"] == 100]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    for series, key, c, m, lab in ((ex, "exposure_ms", plotting.SERIES[0], "o", "Exposure 25–200 ms (gain 8×)"),
                                   (gn, "gain", plotting.SERIES[1], "s", "Gain 1–16× (100 ms)")):
        # saturated settings are left out: clipping removes noise, so their SNR is meaningless
        ok = [r for r in series if r["near_sat_pct"] < 0.1]
        ax.plot([r["signal"] for r in ok], [r["snr"] for r in ok], m + "-", color=c, ms=4, label=lab)
        for r in ok:
            if key == "gain" and r[key] in (2, 8):
                continue  # 2x crowds 1x/4x; 8x is the same setting as 100 ms
            txt = f"{r[key]:g} ms" if key == "exposure_ms" else f"{r[key]:g}×"
            ax.annotate(txt, (r["signal"], r["snr"]), textcoords="offset points",
                        xytext=(4, -8) if key == "exposure_ms" else (2, 4), fontsize=6, color=plotting.INK)
    ax.set(xlabel="Signal (counts)", ylabel="Single-pixel SNR")
    ax.legend(loc="lower right")
    save(fig, "snr")


def fig_cellophane():
    """Filter transmission, and method A (shifted green) vs method B (measured)."""
    run = latest("cellophane/*")
    _, _, spec, total, x, lit = cal_cellophane.load_spectra(run)
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    for f in cal_cellophane.FILTERS:
        t = np.where(lit, total[f] / np.maximum(total["white"], 1e-9), np.nan)
        ax.plot(x, 100 * t, color=HUE[f], label=f"{f.capitalize()} cellophane")
    ax.set(xlabel="Spectral column (blue → red)", ylabel="Transmission (%)", xlim=(0, x[lit][-1]), ylim=(0, 100))
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    save(fig, "cellophane_transmission")

    meth = cal_cellophane.method_a_vs_b(spec, x, lit)
    names = {"R": "red", "G": "green", "B": "blue"}
    xmax = x[lit][-1]                         # beyond this the light had no signal
    fig, axes = plt.subplots(3, 1, figsize=(COL, 4.0), sharex=True)
    for ax, ch in zip(axes, ("B", "G", "R")):
        m = meth[ch]
        ax.plot(x, m["measured"], color=HUE[names[ch]], label="Measured (method B)")
        if ch != "G":
            ax.plot(x, m["model"], color=plotting.INK, ls="--", lw=1.1, label="Shifted green (method A)")
            note = f"shape angle {m['shape_angle_deg']:.1f}°"
        else:
            note = "source of the shifted model"
        ax.set_ylabel("Counts")
        ax.set_xlim(0, xmax)
        ax.set_title(f"{names[ch].capitalize()} pixels, {names[ch]} filter: {note}", loc="left")
    axes[0].legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.22), fontsize=6.5)
    axes[-1].set_xlabel("Spectral column (blue → red)")
    fig.tight_layout(h_pad=0.3)
    save(fig, "cellophane_methods")


def speed_scans():
    """The speed-comparison scans, slowest first."""
    rows = json.load(open(os.path.join(store.CAPTURES_DIR, "speed_comparison", "summary.json")))
    return [(r, os.path.join(store.CAPTURES_DIR, "speed_comparison", r["scan"])) for r in rows]


def fig_panoramas():
    """The three panoramas, aligned, each with a zoomed crop."""
    scans = speed_scans()
    width = 360
    imgs, h = exp_speed_comparison.aligned_panoramas(
        [np.load(os.path.join(p, "cube.npz"))["cube"] for _, p in scans], width)
    z0, z1 = exp_speed_comparison.zoom_window(width)
    fig = plt.figure(figsize=(FULL, 3.4))
    gs = fig.add_gridspec(len(scans), 2, width_ratios=[3.2, 1], hspace=0.45, wspace=0.04)
    for i, ((r, _), img) in enumerate(zip(scans, imgs)):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img, aspect="auto", extent=[0, 360, h, 0])
        ax.add_patch(plt.Rectangle((z0, 3), z1 - z0, h - 6, fill=False, ec="white", lw=1))
        ax.set_title(f"{r['period_s']:.1f} s per turn ({r['lines_per_turn']:.0f} lines per turn, "
                     f"{r['turns']:.1f} turns merged)", loc="left")
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xticks([0, 90, 180, 270, 360])
        if i < len(scans) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Azimuth (deg)")
        axz = fig.add_subplot(gs[i, 1])
        axz.imshow(img[:, z0:z1], aspect="auto", interpolation="nearest")
        axz.set_xticks([])
        axz.set_yticks([])
        axz.grid(False)
    save(fig, "panoramas")


def fig_turns():
    """Turn-to-turn misalignment for the three ways of getting the period."""
    run = latest("processing_comparisons/*")
    res = json.load(open(os.path.join(run, "results.json")))["period"]
    scans = list(res)
    methods = [("assumed label", "Speed label"), ("period search", "Period search"),
               ("search + turn fit", "Search + turn fit")]
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    width = 0.27
    for i, (key, lab) in enumerate(methods):
        vals = [res[s][key]["turn_offset_rms_deg"] for s in scans]
        xs = np.arange(len(scans)) + (i - 1) * width
        ax.bar(xs, vals, width * 0.92, color=plotting.SERIES[i], label=lab)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.2f}" if v < 10 else f"{v:.0f}", (x, v), textcoords="offset points", xytext=(0, 1.5),
                        ha="center", fontsize=6, color=plotting.INK)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 400)
    ax.set_xticks(range(len(scans)), [s.replace("/turn", " per turn") for s in scans])
    ax.set_ylabel("RMS misalignment (deg)")
    # legend below the axis so it can never cover the bars
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=6.5)
    save(fig, "turns")


def fig_merging():
    """Same 90° of the object from one turn vs all turns merged, per speed."""
    scans = speed_scans()
    merged_all = [np.load(os.path.join(p, "cube.npz"))["cube"] for _, p in scans]
    # rotate every scan to line up with the slowest, so each row shows the same 90 deg
    ref = None
    shifts = []
    for cube in merged_all:
        idx = np.linspace(0, cube.shape[0], 360, endpoint=False).astype(int)
        broad = np.nan_to_num(np.nanmean(cube[idx], axis=2))
        ref = broad if ref is None else ref
        shifts.append(turntable.circ_shift(ref, broad))           # degrees (360 samples)
    fig, axes = plt.subplots(len(scans), 2, figsize=(FULL, 3.3))
    for i, ((r, path), merged, shift) in enumerate(zip(scans, merged_all, shifts)):
        d = pipeline.load_calibrated_scan(path, "estimate")
        lines, theta = d["lines"], d["theta"]
        n1 = int(round(len(theta) / (theta[-1] / 360)))
        single, _, _ = turntable.stitch(lines, theta, n1, rev=0)
        del d, lines
        for j, (pano, title) in enumerate(((single, f"one turn ({r['deg_per_line_1turn']:.1f}° per line)"),
                                           (merged, f"{r['turns']:.1f} turns merged "
                                                    f"(largest gap {r['max_gap_deg_merged']:.1f}°)"))):
            n = pano.shape[0]
            pano = np.roll(pano, -int(round(shift * n / 360)), axis=0)
            img = colour.quick_colour(np.transpose(pano, (1, 0, 2)))
            a0, a1 = int(round(0.25 * n)), int(round(0.5 * n))
            ax = axes[i, j]
            ax.imshow(img[:, a0:a1], aspect="auto", interpolation="nearest", extent=[90, 180, img.shape[0], 0])
            ax.set_title(f"{r['period_s']:.1f} s per turn, {title}", loc="left")
            ax.set_yticks([])
            ax.grid(False)
            if i < len(scans) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Azimuth (deg)")
    fig.tight_layout(h_pad=0.5, w_pad=0.6)
    save(fig, "merging")


FIGURES = {"snapshot": fig_snapshot, "lsf": fig_lsf, "snr": fig_snr, "cellophane": fig_cellophane,
           "panoramas": fig_panoramas, "turns": fig_turns, "merging": fig_merging}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", choices=list(FIGURES) + [[]], default=[])
    args = ap.parse_args()
    for name in args.which or FIGURES:
        print(name)
        FIGURES[name]()


if __name__ == "__main__":
    main()
