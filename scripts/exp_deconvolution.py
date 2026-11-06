"""
Experiment: deconvolving the blur along the slit
=================================================
The knife-edge test (exp_line_spread.py) measured how much one point on the
object is smeared along the slit, separately for the blue, green and red
thirds of the spectrum. If that blur is the same everywhere along the slit,
it can be undone: deconvolution of every spectral column of a panorama
along its height axis, each column with the LSF for its part of the
spectrum (interpolated between the three measured thirds). Two methods:
Richardson-Lucy (iterative, keeps the image positive) or Wiener (one
filter, regularised by a noise-to-signal ratio).

Only the height axis is sharpened. Across the slit (azimuth) the blur is
set by the slit width and the angular sampling, which this doesn't measure.

The LSF and the scan must come from the same optical setup and focus, so
this uses the 26 Oct session: its knife edge, its dark frame, and one
revolution of one of its speed-comparison scans.

    python exp_deconvolution.py                       # slowest scan, revolution 0
    python exp_deconvolution.py --scan <dir> --rev 1 --iters 20
    python exp_deconvolution.py --method wiener --nsr 0.01

Output: captures/deconvolution/<time>/
    lsf.png          the measured LSFs used as kernels
    edge_check.png   the knife edge itself, before / after (sanity check)
    panorama.png     one revolution, before / after, with a zoom
    profiles.png     brightness along the height at a few azimuths
    results.json     LSFs (FWHM), sharpness and noise before / after
"""

import argparse
import glob
import json
import os

import numpy as np

from hsi import colour, plotting, store, sweep, turntable
from hsi.plotting import plt

import exp_line_spread

DARK = "dark_20261026.npz"          # the session the LSF and scans share
THIRDS = ("blue", "green", "red")


def load_lsfs(edge_path, dark, half=60):
    """Measured LSF of each spectral third from the knife edge: the slope of
    the edge profile, cut to where it is above 5% of its peak (the rest is
    noise), centred on the 50% crossing and normalised to unit sum.
    Returns {third: (kernel, column centre, fwhm_px)} and the lit columns."""
    img = np.load(edge_path)["frame"] - dark
    level = np.median(np.sort(img, axis=0)[-20:], axis=0)
    use = np.where(level >= 0.2 * level.max())[0]
    k = np.arange(-half, half + 1)
    out = {}
    for name, cols in zip(THIRDS, np.array_split(use, 3)):
        edges, _, slope = exp_line_spread.find_edges(img[:, cols].mean(axis=1))
        e = edges[0]
        c = int(round(e["row"])) + half
        seg = np.pad(slope, half)[c - half: c + half + 1]            # zeros past the slit ends
        peak = seg.max()
        # walk out from the peak; everything beyond the first drop below 5% is noise
        i0 = i1 = int(np.argmax(seg))
        while i0 > 0 and seg[i0 - 1] > 0.05 * peak:
            i0 -= 1
        while i1 < len(seg) - 1 and seg[i1 + 1] > 0.05 * peak:
            i1 += 1
        seg[:i0] = 0
        seg[i1 + 1:] = 0
        # centre on the sub-pixel 50% crossing so deconvolution doesn't shift the image
        seg = np.interp(k + (e["row"] - round(e["row"])), k, seg, left=0, right=0)
        out[name] = (seg / seg.sum(), float(cols.mean()), float(e["fwhm_px"]))
    return out, use


def kernel_for(col, lsfs):
    """LSF for one spectral column: linear between the measured thirds."""
    centres = np.array([lsfs[t][1] for t in THIRDS])
    j = np.clip(np.searchsorted(centres, col) - 1, 0, len(centres) - 2)
    w = np.clip((col - centres[j]) / (centres[j + 1] - centres[j]), 0, 1)
    k = (1 - w) * lsfs[THIRDS[j]][0] + w * lsfs[THIRDS[j + 1]][0]
    return k / k.sum()


def _padded(data, kernel, axis):
    """Move `axis` last and reflect-pad it by the kernel half-width, so the
    ends of the slit don't wrap into each other under the FFT. Returns the
    padded data, the kernel's transfer function and a function to undo both."""
    data = np.moveaxis(data, axis, -1)
    n, h = data.shape[-1], len(kernel) // 2
    pad = np.concatenate([data[..., h:0:-1], data, data[..., -2:-h - 2:-1]], axis=-1)
    kp = np.zeros(pad.shape[-1])
    kp[:len(kernel)] = kernel
    K = np.fft.rfft(np.roll(kp, -h))
    return pad, K, lambda x: np.moveaxis(x[..., h:h + n], -1, axis)


def richardson_lucy(data, kernel, iters, axis):
    """Richardson-Lucy along one axis, by FFT. Keeps the image positive and
    the total signal unchanged."""
    pad, K, unpad = _padded(np.clip(data, 1e-3, None), kernel, axis)
    m = pad.shape[-1]

    def conv(x, F):
        return np.fft.irfft(np.fft.rfft(x, axis=-1) * F, m, axis=-1)

    est = pad.copy()
    for _ in range(iters):
        ratio = pad / np.maximum(conv(est, K), 1e-6)
        est *= conv(ratio, np.conj(K))
    return unpad(est)


def wiener(data, kernel, nsr, axis):
    """Wiener deconvolution along one axis: divide by the kernel's transfer
    function K where it is strong, and roll off to zero where |K|^2 falls
    below the noise-to-signal ratio `nsr` (K is 1 at zero frequency, so nsr
    is relative to the image's overall level). Smaller nsr = sharper, noisier."""
    pad, K, unpad = _padded(data, kernel, axis)
    G = np.conj(K) / (np.abs(K) ** 2 + nsr)
    return unpad(np.fft.irfft(np.fft.rfft(pad, axis=-1) * G, pad.shape[-1], axis=-1))


def deconvolve(cube, lsfs, method, axis, groups=24):
    """Deconvolve each block of spectral columns with its own kernel;
    method(data, kernel, axis) does one block."""
    out = np.empty_like(cube, dtype=np.float32)
    for cols in np.array_split(np.arange(cube.shape[-1]), groups):
        out[..., cols] = method(cube[..., cols], kernel_for(cols.mean(), lsfs), axis)
    return out


def sharpness(img):
    """Mean squared height-gradient of the broadband image (bigger = sharper)."""
    b = np.nanmean(img, axis=-1)
    return float(np.nanmean(np.diff(b, axis=-2) ** 2))


def noise(img):
    """Pixel noise: robust spread of the column-to-column differences, which
    removes real spectral shape (smooth) and keeps per-pixel noise."""
    d = np.diff(img, axis=-1)
    return float(1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d))) / np.sqrt(2))


def one_revolution(scan, rev, dark):
    frames, times, info = sweep.load_scan(scan)
    lines = np.asarray(frames, np.float32) - dark
    period = float(np.load(os.path.join(scan, "cube.npz"))["period_s"])
    theta = turntable.angles(times, period)
    n = int(round(np.sum((theta >= 360 * rev) & (theta < 360 * (rev + 1)))))
    pano, az, gap = turntable.stitch(lines, theta, n, rev=rev)
    return pano, az, period, gap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", default=None, help="scan folder (default: the slowest 26 Oct speed-comparison scan)")
    ap.add_argument("--edge", default=None, help="knife-edge capture (default: latest line_spread/*/edge_best.npz)")
    ap.add_argument("--rev", type=int, default=0, help="which revolution to use")
    ap.add_argument("--half", type=int, default=60, help="kernel half-width in rows (raise it for a blurrier edge)")
    ap.add_argument("--method", choices=["rl", "wiener"], default="rl", help="Richardson-Lucy or Wiener")
    ap.add_argument("--iters", type=int, default=15, help="Richardson-Lucy iterations (more = sharper, noisier)")
    ap.add_argument("--nsr", type=float, default=0.01, help="Wiener noise-to-signal ratio (smaller = sharper, noisier)")
    args = ap.parse_args()
    if args.method == "rl":
        method = lambda d, k, axis: richardson_lucy(d, k, args.iters, axis)
        label = f"Richardson-Lucy, {args.iters} iterations"
    else:
        method = lambda d, k, axis: wiener(d, k, args.nsr, axis)
        label = f"Wiener, NSR {args.nsr:g}"

    dark = np.load(store.calib_path(DARK))["dark"]
    scan = args.scan or sorted(glob.glob(os.path.join(store.CAPTURES_DIR, "speed_comparison", "*_setting3_*")))[-1]
    edge = args.edge or sorted(glob.glob(os.path.join(store.CAPTURES_DIR, "line_spread", "*", "edge_best.npz")))[-1]
    out = store.new_run_dir("deconvolution")
    print(f"scan {scan}\nedge {edge}\n-> {out}")

    lsfs, _ = load_lsfs(edge, dark, args.half)
    res = {"scan": scan, "edge": edge, "rev": args.rev, "method": label,
           "lsf_fwhm_px": {t: lsfs[t][2] for t in THIRDS}}

    # --- LSFs ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3))
    for t in THIRDS:
        k = lsfs[t][0]
        ax.plot(np.arange(len(k)) - len(k) // 2, k / k.max(), label=f"{t} third (FWHM {lsfs[t][2]:.0f} px)")
    ax.set(xlabel="Rows along the slit", ylabel="Normalised LSF (kernel)")
    ax.legend()
    fig.savefig(os.path.join(out, "lsf.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- sanity check on the knife edge itself --------------------------------
    img = (np.load(edge)["frame"] - dark)[None]                       # (1, rows, cols)
    sharp = deconvolve(img, lsfs, method, axis=1)[0]
    img = img[0]
    level = np.median(np.sort(img, axis=0)[-20:], axis=0)
    use = np.where(level >= 0.2 * level.max())[0]
    check = {}
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    for ax, (t, cols) in zip(axes, zip(THIRDS, np.array_split(use, 3))):
        e0 = exp_line_spread.find_edges(img[:, cols].mean(axis=1))[0][0]
        e1 = exp_line_spread.find_edges(sharp[:, cols].mean(axis=1))[0][0]
        check[t] = {"rise_px_before": e0["rise_px"], "rise_px_after": e1["rise_px"]}
        r = slice(max(int(e0["row"]) - 80, 0), min(int(e0["row"]) + 80, len(img)))
        for data, lab in ((img, f"before (10-90% {e0['rise_px']:.0f} px)"),
                          (sharp, f"after (10-90% {e1['rise_px']:.0f} px)")):
            p = data[r, cols].mean(axis=1)
            ax.plot(np.arange(r.start, r.stop), p / np.median(np.sort(p)[-10:]), label=lab)
        ax.set(title=f"{t} third", xlabel="Row along slit")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Edge profile (normalised)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "edge_check.png"), dpi=150)
    plt.close(fig)
    res["edge_check"] = check
    print("knife edge 10-90% rise (px): " + ", ".join(
        f"{t} {c['rise_px_before']:.0f} -> {c['rise_px_after']:.0f}" for t, c in check.items()))

    # --- one revolution of the scan --------------------------------------------
    pano, az, period, gap = one_revolution(scan, args.rev, dark)
    print(f"revolution {args.rev}: {len(az)} lines, period {period:.2f} s, largest gap {gap:.1f} deg")
    sharp = deconvolve(pano, lsfs, method, axis=1)
    res["panorama"] = {"lines": len(az), "period_s": period, "max_gap_deg": gap,
                       "sharpness_before": sharpness(pano), "sharpness_after": sharpness(sharp),
                       "noise_before": noise(pano), "noise_after": noise(sharp)}
    p = res["panorama"]
    print(f"sharpness x{p['sharpness_after'] / p['sharpness_before']:.2f}, "
          f"pixel noise x{p['noise_after'] / p['noise_before']:.2f}")

    # one colour scale for both, so the comparison is fair
    both = colour.quick_colour(np.concatenate([np.transpose(pano, (1, 0, 2)), np.transpose(sharp, (1, 0, 2))], axis=1))
    before, after = both[:, :len(az)], both[:, len(az):]
    h = before.shape[0]
    z = slice(int(0.25 * len(az)), int(0.4 * len(az)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [3, 1]})
    for i, (im, lab) in enumerate(((before, "measured"), (after, f"deconvolved ({label})"))):
        axes[i, 0].imshow(im, aspect="auto", extent=[0, 360, h, 0])
        axes[i, 0].axvspan(az[z.start], az[z.stop - 1], fill=False, ec="white", lw=1)
        axes[i, 0].set(title=f"Revolution {args.rev}, {lab}", ylabel="Row along slit")
        axes[i, 1].imshow(im[:, z], aspect="auto", interpolation="nearest")
        axes[i, 1].set(xticks=[], yticks=[])
        for a in axes[i]:
            a.grid(False)
    axes[1, 0].set_xlabel("Azimuth (deg)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "panorama.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    for ax, a in zip(axes, np.linspace(0, len(az), 3, endpoint=False).astype(int) + len(az) // 6):
        for data, lab in ((pano, "measured"), (sharp, "deconvolved")):
            ax.plot(np.nanmean(data[a], axis=-1), label=lab)
        ax.set(title=f"Azimuth {az[a]:.0f} deg", xlabel="Row along slit")
    axes[0].set_ylabel("Mean signal (counts)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "profiles.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
