"""
Experiment: reconstruction consistency
========================================
Success criterion: "evaluate spatial alignment between neighbouring line
scans and quantify any discontinuity or reconstruction error."

Needs a scan of at least 2 full revolutions (scan.py's default 70 s). The same
surface is then imaged twice, so the two revolutions are independent
reconstructions of the same panorama and any disagreement is
reconstruction error. For each way of getting the turntable period
(nominal if recorded / estimated from this scan / --period-s stopwatch)
it reports:

  azimuth offset  rev 2 vs rev 1, by circular cross-correlation (deg).
                  A wrong period shows up as exactly this: it grows by
                  360*(error/period) per revolution.
  height offset   rev 2 vs rev 1 along the slit (rows) — should be ~0;
                  non-zero means the object wobbles/shifts on the table.
  RMSE, SAM       reflectance and spectral-angle difference rev 2 vs rev 1
  seam ratio      a single revolution has a seam at 0/360 where its first
                  and last lines meet. Compare it with a panorama built from
                  a window offset by half a revolution (where 0 deg is
                  continuous data, no seam): their difference in the
                  columns next to 0 deg / their difference elsewhere.
                  ~1 = invisible seam; >>1 = visible seam. Needs 1.5 revs.
  zipper ratio    neighbour-column roughness of the 2-rev merged panorama /
                  that of a single revolution. ~1 = revolutions interleave
                  cleanly; >1 = they're misregistered (zig-zag edges).

This is also the quantitative comparison of the alternative angle-mapping
methods the brief asks for.
"""

import argparse
import os

import cv2
import numpy as np

from hsi import colour, pipeline, plotting, spectral, store, turntable
from hsi.plotting import plt


def lin_shift(a, b, max_shift=20):
    """Sub-sample linear shift of 1-D profile b relative to a."""
    lags = np.arange(-max_shift, max_shift + 1)
    a = a - a.mean()
    b = b - b.mean()
    cc = np.array([np.sum(a[max(0, -l): len(a) - max(0, l)] * b[max(0, l): len(b) - max(0, -l)]) for l in lags])
    return float(lags[0] + spectral.subpixel_peak(cc, int(np.argmax(cc))))


def roughness(pano):
    """Median neighbour-column difference of a panorama (n_az, rows, bands)."""
    d = np.nanmean(np.abs(np.diff(pano, axis=0)), axis=(1, 2))
    return float(np.median(d)), d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", help="scan folder from scan.py (>= 2 revolutions)")
    ap.add_argument("--direction", type=int, choices=[1, -1], default=1)
    ap.add_argument("--period-s", type=float, default=None,
                    help="also score a hand-measured period (e.g. stopwatch 10 turns / 10)")
    ap.add_argument("--n-az", type=int, default=None)
    ap.add_argument("--step-nm", type=float, default=10.0, help="coarser bands are fine for alignment")
    args = ap.parse_args()

    base = pipeline.load_calibrated_scan(args.scan, "estimate", args.direction, args.step_nm)
    lines, times, info = base["lines"], base["times"], base["info"]
    feats = turntable.frame_features(lines)

    methods = {"nominal": float(info["nominal_period_s"])} if info.get("nominal_period_s") else {}
    if args.period_s:
        methods["stopwatch"] = args.period_s
    for m in ("estimate",):
        try:
            methods[m] = pipeline.period_for(times, lines, info, m, feats)[0]
        except (SystemExit, ValueError) as e:
            print(f"skipping {m}: {e}")

    out = os.path.join(args.scan, "consistency")
    os.makedirs(out, exist_ok=True)
    res = {}
    renders = {}
    for name, period in methods.items():
        theta = turntable.angles(times, period, direction=args.direction)
        revs = theta[-1] / 360
        n_az = args.n_az or int(round(len(theta) / max(revs, 1e-9)))
        p0, az, _ = turntable.stitch(lines, theta, n_az, rev=0)
        r = {"period_s": period}
        if revs >= 1.5:
            ph, _, _ = turntable.stitch(lines, theta, n_az, window=(180.0, 540.0))
            diff = np.nanmean(np.abs(p0 - ph), axis=(1, 2))
            near = np.r_[0:2, n_az - 2:n_az]
            far = np.setdiff1d(np.arange(n_az), np.r_[0:4, n_az - 4:n_az])
            r["seam_ratio"] = float(np.mean(diff[near]) / (np.median(diff[far]) + 1e-12))
        if revs >= 2.0:
            p1, _, _ = turntable.stitch(lines, theta, n_az, rev=1)
            merged, _, _ = turntable.stitch(lines, theta[theta < 720], n_az)
            # broadband per (azimuth, height) for alignment
            b0 = np.nan_to_num(np.nanmean(p0, axis=2))
            b1 = np.nan_to_num(np.nanmean(p1, axis=2))
            r["azimuth_offset_deg"] = float(turntable.circ_shift(b0, b1) * 360 / n_az)
            r["height_offset_rows"] = lin_shift(b0.mean(0), b1.mean(0))
            r["rmse"] = float(np.sqrt(np.nanmean((p1 - p0) ** 2)))
            sam = spectral.spectral_angle(p0, p1)
            r["sam_mean_deg"] = float(np.nanmean(sam))
            r["sam_p95_deg"] = float(np.nanpercentile(sam, 95))
            r["zipper_ratio"] = roughness(merged)[0] / roughness(p0)[0]
            renders[name] = (p0, p1, sam)
        else:
            print(f"{name}: scan covers {revs:.2f} rev (< 2) — rev-to-rev metrics need 2 full revolutions")
        res[name] = r

    hdr = f"{'method':<10}{'period s':>10}{'az off °':>10}{'h off px':>10}{'RMSE':>9}{'SAM °':>8}{'seam':>7}{'zipper':>8}"
    print(hdr)
    for name, r in res.items():
        f = lambda k, fmt: format(r[k], fmt) if k in r else "-"
        print(f"{name:<10}{r['period_s']:>10.4f}{f('azimuth_offset_deg', '>10.2f')}{f('height_offset_rows', '>10.2f')}"
              f"{f('rmse', '>9.4f')}{f('sam_mean_deg', '>8.2f')}{f('seam_ratio', '>7.2f')}{f('zipper_ratio', '>8.2f')}")
    store.save_json(os.path.join(out, "results.json"), res)

    if renders:
        best = min(renders, key=lambda k: abs(res[k]["azimuth_offset_deg"]))
        p0, p1, sam = renders[best]
        grid = base["grid"]

        def render(p):
            img = np.transpose(p, (1, 0, 2))
            if base["calibrated"] and base["wavelength"]:
                return colour.reflectance_to_srgb(img, grid)
            return colour.quick_colour(img, grid)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10),
                                 gridspec_kw={"height_ratios": [1, 1, 1, 1.1]})
        axes[0].imshow(render(p0), aspect="auto", extent=[0, 360, p0.shape[1], 0])
        axes[0].set_title(f"Revolution 1 ({best} period)", loc="left")
        axes[1].imshow(render(p1), aspect="auto", extent=[0, 360, p0.shape[1], 0])
        axes[1].set_title("Revolution 2", loc="left")
        im = axes[2].imshow(sam.T, aspect="auto", cmap="Blues", extent=[0, 360, p0.shape[1], 0],
                            vmin=0, vmax=np.nanpercentile(sam, 99))
        axes[2].set_title("Spectral angle rev 1 vs rev 2 (deg)", loc="left")
        fig.colorbar(im, ax=axes[2], fraction=0.02)
        for ax in axes[:3]:
            ax.grid(False)
            ax.set_yticks([])
        names = [k for k in res if "azimuth_offset_deg" in res[k]]
        vals = [abs(res[k]["azimuth_offset_deg"]) for k in names]
        axes[3].bar(names, vals, color=plotting.SERIES[0], width=0.5)
        for i, v in enumerate(vals):
            axes[3].annotate(f"{v:.2f}°", (i, v), textcoords="offset points", xytext=(0, 4),
                             ha="center", color=plotting.INK)
        axes[3].set_ylabel("|azimuth offset| (deg)")
        axes[3].set_title("Rev-to-rev misalignment by period method")
        fig.tight_layout()
        plotting.save(fig, os.path.join(out, "consistency.png"))
        cv2.imwrite(os.path.join(out, "rev1.png"), cv2.cvtColor(render(p0), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out, "rev2.png"), cv2.cvtColor(render(p1), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
