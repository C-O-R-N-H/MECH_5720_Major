"""
Experiment: spectral repeatability (and the single-line minimum outcome)
==========================================================================
Success criterion: "repeatedly measure the same stationary target under
identical illumination and quantify the variation between measured spectra."

This is also the brief's fallback minimum outcome — "capture and calibrate
the spectrum of a single line on a test object" — so run it on a target
with a few different coloured patches stacked ALONG the slit (e.g. strips
of coloured paper taped vertically on a card) and keep the figure.

Procedure
  1. Dark frame at the scan settings (take_dark.py). A white reference
     (cal_white_reference.py) is optional: with it the numbers are in
     reflectance, without it in dark-subtracted counts.
  2. Stop the turntable with the target in view. Stay still — moving
     around the room showed up as a 6% spike.
  3. Script captures --n frames, --interval s apart, and reports:
       per-band coefficient of variation (CV %)      noise vs wavelength
       spectral angle of each capture vs the mean    shape stability (deg)
       trend of the band-integrated reflectance      slow drift (%/min)
     for the whole line and for each --segments chunk along the slit,
     plus robust (median-based) versions that ignore brief disturbances,
     and the single-pixel variation. Per-capture spectra are saved
     (segment_spectra.npz) for exp_processing_comparisons.py.

    python exp_repeatability.py --n 50
  Optional --sessions K: repeat K times, and between sessions remove and
  replace the target (and/or re-take the reference). Session-to-session
  spread vs within-session spread separates "camera noise" from
  "setup repeatability".
"""

import argparse
import os
import time

import numpy as np

from hsi import camera, pipeline, plotting, spectral, store
from hsi.plotting import plt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    camera.add_camera_args(ap)
    ap.add_argument("--n", type=int, default=30, help="captures per session")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between captures")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--segments", type=int, default=4, help="chunks along the slit to report separately")
    ap.add_argument("--step-nm", type=float, default=5.0)
    ap.add_argument("--reference", default=None, help="reference dir (default: latest)")
    ap.add_argument("--experiment", default="repeatability", help="folder under captures/")
    args = ap.parse_args()

    cam = camera.open_camera(args)
    h, w = cam.grab(args.method)[0].shape
    roi = [0, h, 0, w]  # whole frame
    roi_shape = (roi[1] - roi[0], roi[3] - roi[2])
    ref = pipeline.load_reference(args.reference)
    dark = None
    if ref is None:
        dark = pipeline.load_dark(roi, 1, args.exposure_us, args.gain, shape=roi_shape)
        print("no white reference — reporting " + ("dark-subtracted counts" if dark is not None
                                                     else "raw counts (no dark frame either: take_dark.py)"))
    calib = pipeline.LineCalibrator(roi_shape, reference=ref, exposure_us=args.exposure_us, step_nm=args.step_nm,
                                    dark=dark)
    out = store.new_run_dir(args.experiment)

    lines, times, session = [], [], []
    for s in range(args.sessions):
        msg = "Place the stationary target on the stopped turntable."
        if s:
            msg = "Remove and re-place the target (re-run cal_white_reference.py first if testing re-referencing)."
        cam.prompt(msg, "lamp-patches")
        for i in range(args.n):
            f, m = cam.grab(args.method)
            lines.append(calib(spectral.crop(f, roi)[None])[0])
            times.append(m["t"])
            session.append(s)
            print(f"\rsession {s + 1}/{args.sessions} capture {i + 1}/{args.n}", end="")
            time.sleep(max(0.0, args.interval - (time.time() - m["t"])))
        print()
    lines = np.array(lines)        # (n, rows, bands)
    times = np.array(times)
    session = np.array(session)
    grid = calib.grid
    mean_img = np.nanmean(lines, axis=0)
    std_img = np.nanstd(lines, axis=0)
    np.savez_compressed(os.path.join(out, "summary_images.npz"), mean=mean_img, std=std_img, times=times,
                        session=session, grid=grid)
    # Only spectral columns with real signal count in the summary numbers;
    # near-zero columns would turn noise / ~0 into huge percentages.
    level = np.nanmedian(mean_img, axis=0)
    use = level >= 0.2 * np.nanmax(level)
    lit = (mean_img > 0.2 * np.nanpercentile(mean_img, 99)) & use[None, :]
    pixel_cv = 100 * std_img[lit] / mean_img[lit]

    seg_edges = np.linspace(0, lines.shape[1], args.segments + 1).astype(int)
    segs = {"whole line": (0, lines.shape[1])}
    segs.update({f"segment {k + 1}": (seg_edges[k], seg_edges[k + 1]) for k in range(args.segments)})

    res = {"n_captures": len(lines), "sessions": args.sessions, "reflectance": calib.calibrated,
           "usable_columns": [int(np.argmax(use)), int(len(use) - np.argmax(use[::-1]))],
           "single_pixel_cv_pct_median": float(np.median(pixel_cv)),
           "single_pixel_snr_median": float(np.median(100 / pixel_cv)),
           "segments": {}}
    print(f"single pixel: median variation {res['single_pixel_cv_pct_median']:.1f}% "
          f"(signal-to-noise {res['single_pixel_snr_median']:.0f}) over {lit.sum()} lit pixels")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    all_spec = {}
    for k, (name, (a, b)) in enumerate(segs.items()):
        spec = np.nanmean(lines[:, a:b], axis=1)                # (n, bands)
        mean = np.nanmean(spec, axis=0)
        within = np.nanmean([np.nanstd(spec[session == s], axis=0) for s in range(args.sessions)], axis=0)
        cv = 100 * within / np.abs(mean)
        sam = spectral.spectral_angle(spec, mean[None])
        total = np.nansum(spec, axis=1)
        tmin = (times - times[0]) / 60
        slope = float(np.polyfit(tmin, 100 * total / total.mean(), 1)[0]) if tmin[-1] > 0 else 0.0
        sam = spectral.spectral_angle(spec[:, use], mean[None, use])
        # Robust version: median absolute deviation instead of std, so a brief
        # disturbance (torch flicker, someone's shadow) doesn't dominate.
        mad = 1.4826 * np.nanmedian(np.abs(spec - np.nanmedian(spec, axis=0)), axis=0)
        cv_robust = 100 * mad / np.abs(mean)
        total_dev = 100 * (np.nansum(spec[:, use], axis=1) / np.nanmedian(np.nansum(spec[:, use], axis=1)) - 1)
        outliers = np.abs(total_dev) > 5 * 1.4826 * np.median(np.abs(total_dev)) + 0.5
        all_spec[name] = spec
        r = {"rows": [int(a), int(b)],
             "median_cv_pct": float(np.nanmedian(cv[use])),
             "median_cv_robust_pct": float(np.nanmedian(cv_robust[use])),
             "sam_median_deg": float(np.median(sam)),
             "outlier_captures": np.where(outliers)[0].tolist(),
             "cv_pct_by_band": cv,
             "sam_mean_deg": float(np.mean(sam)),
             "sam_p95_deg": float(np.percentile(sam, 95)),
             "drift_pct_per_min": slope}
        if args.sessions > 1:
            sess_means = np.array([np.nanmean(spec[session == s], axis=0) for s in range(args.sessions)])
            r["between_session_cv_pct"] = float(np.nanmedian((100 * np.nanstd(sess_means, axis=0) / np.abs(mean))[use]))
            r["between_session_sam_deg"] = float(np.mean(spectral.spectral_angle(sess_means[:, use], mean[None, use])))
        res["segments"][name] = r
        print(f"{name:<12} CV {r['median_cv_pct']:5.2f}% (robust {r['median_cv_robust_pct']:.2f}%)   "
              f"SAM median {r['sam_median_deg']:.3f}, mean {r['sam_mean_deg']:.3f} deg "
              f"(p95 {r['sam_p95_deg']:.3f})   drift {slope:+.3f} %/min"
              + (f"   between-session CV {r['between_session_cv_pct']:.2f}%" if args.sessions > 1 else "")
              + (f"   disturbed captures: {r['outlier_captures']}" if r["outlier_captures"] else ""))

        if k == 0:
            axes[0, 1].plot(grid[use], cv[use], color=plotting.SERIES[0])
            axes[1, 0].plot(np.arange(len(sam)), sam, ".", ms=8, color=plotting.SERIES[0])
            axes[1, 1].plot(tmin, 100 * (total / total.mean() - 1), color=plotting.SERIES[0])
        else:
            c = plotting.SERIES[(k - 1) % len(plotting.SERIES)]
            sd = np.nanstd(spec, axis=0)
            axes[0, 0].fill_between(grid, mean - 2 * sd, mean + 2 * sd, color=c, alpha=0.25, lw=0)
            axes[0, 0].plot(grid, mean, color=c, label=name)

    xl = "wavelength (nm)" if calib.resampler is not None else "column (px)"
    axes[0, 0].set(xlabel=xl, ylabel="reflectance" if calib.calibrated else "counts",
                   title="Mean spectrum per segment (band = ±2σ)")
    axes[0, 0].legend()
    axes[0, 1].set(xlabel=xl, ylabel="CV (%)", title="Capture-to-capture variation, whole line (usable columns)")
    axes[1, 0].set(xlabel="capture #", ylabel="spectral angle to mean (deg)", title="Shape stability")
    axes[1, 1].set(xlabel="time (min)", ylabel="change (%)", title="Band-integrated drift")
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "repeatability.png"))
    store.save_json(os.path.join(out, "results.json"), res)
    np.savez_compressed(os.path.join(out, "segment_spectra.npz"), times=times, grid=grid, use=use,
                        **{n.replace(" ", "_"): v for n, v in all_spec.items()})


if __name__ == "__main__":
    main()
