"""
Calibration stage 2: white reference + lamp drift (not yet run)
===================================================================
The two calibration stages fix different things and are needed at
different times:

  stage 1 (cal_wavelength.py)  WAVELENGTH  which column is which nm.
               Depends only on the optics geometry. Do it once (sunlight),
               redo if anything moves.
  stage 2 (this)  RADIOMETRIC  what the light + camera do to the signal at each
               (row, wavelength). Depends on the lamp, its warm-up, its
               position, the exposure, the room light ... so it is captured
               EVERY scan session, under exactly the scan conditions:
                   dark  D  lens capped / lamp off, same exposure
                   white W  a white reference card where the object will be
               then every scan frame S becomes reflectance  R = (S-D)/(W-D).
               The lamp spectrum, the camera's spectral response, slit
               width variations and vignetting all cancel.

Use the scan settings (the defaults) so the reference matches the scans
(the script records them; scans warn if they differ). A white reference
replaces the plain dark frame in scans: they then report reflectance.

    python cal_white_reference.py
    python cal_white_reference.py --drift-minutes 15    (also measure lamp drift)
    python cal_white_reference.py --sun --exposure-us 2000 --gain 1   (camera response; see cal_wavelength.py)
White target: matte, flat, spectrally flat — PTFE / Spectralon is ideal,
plain white printer paper is acceptable (but optical brighteners fluoresce
blue under UV-rich light like sunlight, so prefer brightener-free paper).

Extras
  --sun:   if the reference is taken in sunlight, divide by a reference solar
           spectrum to get the CAMERA's own relative spectral response
           (calibration/response.npz). Default model is a 5778 K blackbody;
           pass --solar-csv with ASTM G173 data (NREL) for a better one.
  If response.npz exists and this is a lamp reference, the lamp's spectrum
  is characterised as  (W-D) / response  — the brief's "illumination will be
  characterised during calibration".
  --drift-minutes M: after the reference, keep capturing the white card
           every --drift-interval s for M minutes and report how much the
           lamp drifts. This tells you how often you must re-reference
           (and how long the lamp needs to warm up).

Output: calibration/references/<stamp>/reference.npz (+ 'latest' pointer)
        captures/white_reference/<stamp>/ figures + results.json
"""

import argparse
import datetime
import os
import time

import numpy as np

from hsi import camera, pipeline, plotting, spectral, store
from hsi.plotting import plt


def mean_spectrum(roi_img, calib):
    """Mean over the slit, on the wavelength grid if calibrated."""
    line = calib(roi_img[None])[0]
    return np.nanmean(line, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    camera.add_camera_args(ap)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sun", action="store_true", help="reference taken in sunlight -> camera response")
    ap.add_argument("--solar-csv", default=None, help="nm,value CSV (e.g. ASTM G173 global tilt)")
    ap.add_argument("--drift-minutes", type=float, default=0.0)
    ap.add_argument("--drift-interval", type=float, default=10.0)
    ap.add_argument("--label", default="", help="free text stored with the reference (lamp, distance ...)")
    args = ap.parse_args()
    cam = camera.open_camera(args)
    h, w = cam.grab(args.method)[0].shape
    roi = [0, h, 0, w]  # whole frame
    out = store.new_run_dir("white_reference", "sun" if args.sun else "lamp")

    cam.prompt("Cap the lens (same exposure as the scan) for the dark reference.", "dark")
    dark = spectral.crop(cam.average(args.n, args.method)[0], roi)
    scene = "sun" if args.sun else "lamp-white"
    cam.prompt("Put the WHITE reference card where the object will sit, lit as for the scan.", scene)
    white_full, white_std, _ = cam.average(args.n, args.method)
    white = spectral.crop(white_full, roi)
    white_std = spectral.crop(white_std, roi)

    sat = float(np.percentile(white, 99.9))
    info = {"exposure_us": args.exposure_us or 0, "gain": args.gain, "source": "sun" if args.sun else "lamp",
            "label": args.label, "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "roi": np.asarray(roi)}
    ref_path = pipeline.save_reference(dark, white, info)
    print(f"saved reference {ref_path}")

    calib = pipeline.LineCalibrator(white.shape)
    grid = calib.grid
    sig = mean_spectrum(white - dark, calib)
    snr = np.nanmean(calib((white - dark)[None])[0] / np.maximum(calib(white_std[None])[0], 1e-6), axis=0)
    res = {"reference": ref_path, "white_p99_9": sat,
           "median_single_frame_snr": float(np.nanmedian(snr))}
    print(f"white p99.9 = {sat:.0f} counts (keep well below saturation); median single-frame SNR {res['median_single_frame_snr']:.0f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(grid, sig / np.nanmax(sig), color=plotting.SERIES[0], label="white − dark (lamp × camera)")
    xlabel = "wavelength (nm)" if calib.resampler is not None else "column (px)"

    resp_path = store.calib_path("response.npz")
    if args.sun and calib.resampler is not None:
        sun = (spectral.load_reference_spectrum(args.solar_csv, grid) if args.solar_csv
               else spectral.planck(grid, 5778))
        response = sig / sun
        response /= np.nanmax(response)
        np.savez(resp_path, grid=grid, response=response,
                 solar_model=args.solar_csv or "blackbody 5778 K")
        ax.plot(grid, response, color=plotting.SERIES[1], label="camera relative response")
        print(f"wrote camera spectral response {resp_path}")
    elif os.path.exists(resp_path) and calib.resampler is not None:
        r = np.load(resp_path)
        response = np.interp(grid, r["grid"], r["response"], left=np.nan, right=np.nan)
        lamp = sig / np.maximum(response, 0.05)
        lamp /= np.nanmax(lamp)
        np.savez(os.path.join(out, "illumination_spectrum.npz"), grid=grid, spectrum=lamp)
        ax.plot(grid, lamp, color=plotting.SERIES[2], label="illumination spectrum (÷ response)")
        res["illumination_spectrum"] = os.path.join(out, "illumination_spectrum.npz")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("relative")
    ax.set_title("Reference spectra")
    ax.legend()
    plotting.save(fig, os.path.join(out, "reference.png"))

    if args.drift_minutes > 0:
        print(f"monitoring drift for {args.drift_minutes} min, leave everything untouched...")
        t_end = time.time() + 60 * args.drift_minutes
        ts, specs = [], []
        while time.time() < t_end:
            f, m = cam.grab(args.method)
            ts.append(m["t"])
            specs.append(mean_spectrum(spectral.crop(f, roi) - dark, calib))
            time.sleep(max(0.0, args.drift_interval - (time.time() - m["t"])))
        ts = (np.array(ts) - ts[0]) / 60
        specs = np.array(specs)
        total = np.nansum(specs, axis=1)
        rel = 100 * (total / total[0] - 1)
        shape = 100 * (specs / specs[:1] / (total / total[0])[:, None] - 1)  # colour change, brightness removed
        slope = float(np.polyfit(ts, rel, 1)[0]) if len(ts) > 2 else 0.0
        res["drift"] = {"minutes": float(ts[-1]), "total_change_pct": float(rel[-1]),
                        "slope_pct_per_min": slope, "max_abs_pct": float(np.max(np.abs(rel))),
                        "max_band_shape_change_pct": float(np.nanmax(np.abs(shape)))}
        print(f"drift: {rel[-1]:+.2f}% over {ts[-1]:.1f} min ({slope:+.3f} %/min); "
              f"worst per-band colour change {res['drift']['max_band_shape_change_pct']:.2f}%")
        if abs(slope) > 0:
            print(f"-> 1% error budget is used up after ~{1 / abs(slope):.1f} min: re-reference at least that often")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(ts, rel, color=plotting.SERIES[0])
        axes[0].set_xlabel("time (min)")
        axes[0].set_ylabel("total signal change (%)")
        axes[0].set_title("Lamp brightness drift")
        im = axes[1].imshow(shape.T, aspect="auto", origin="lower", cmap="RdBu_r",
                            extent=[ts[0], ts[-1], grid[0], grid[-1]],
                            vmin=-np.nanmax(np.abs(shape)), vmax=np.nanmax(np.abs(shape)))
        axes[1].grid(False)
        axes[1].set_xlabel("time (min)")
        axes[1].set_ylabel(xlabel)
        axes[1].set_title("Per-band colour drift (%)")
        fig.colorbar(im, ax=axes[1])
        plotting.save(fig, os.path.join(out, "drift.png"))

    store.save_json(os.path.join(out, "results.json"), res)


if __name__ == "__main__":
    main()
