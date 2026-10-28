"""
Calibration stage 1: wavelength (not yet run)
===============================================
Success criterion: "establish a mapping between sensor position and
wavelength using known spectral reference features, and quantify the
wavelength calibration error in nm."

Method
  Sunlight has dozens of narrow absorption lines (Fraunhofer lines) at
  wavelengths known to 0.01 nm — Na D at 589.3, H-alpha at 656.3, H-beta at
  486.1, the Mg b triplet at 517.3 ... Image sunlit white paper, find the
  dips, match them to the catalogue, fit pixel -> nm with a low-order
  polynomial. This only depends on the optics geometry, so it stays valid
  until the slit/grating/lens/camera are moved — re-run it if they are.

  Smile (curved spectral lines along the slit) is measured from the same
  frame and stored, so each row gets its own wavelength mapping.

Error is reported two ways
  * fit RMS   — residuals of the fitted points (optimistic)
  * LOO RMS   — leave-one-out: each line predicted by a fit WITHOUT it.
                Quote this one as the calibration error.
  * --validate: apply the saved calibration to a DIFFERENT source (a
    fluorescent lamp, whose Hg/Eu/Tb emission lines are independent of the
    Fraunhofer lines) and report the error on those. This is the strongest
    test, and the "compare calibration methods quantitatively" item: run
    the calibration with --source sun and --source fluorescent and validate
    each against the other.

Procedure
  1. Pick an exposure first: sunlight is far brighter than the torch, so
     the scan settings (100 ms, 8x) will saturate. Try
        python snapshot.py --exposure-us 2000 --gain 1
     and adjust until the brightest pixels are ~70% of full scale (~2700).
     This calibration only depends on the optics, so it doesn't need to
     use the scan settings — but use the same settings for both steps here.
  2. Run with those settings; it asks for the lens cap (dark), then:
     outdoors, aim at white paper in direct sun (not at the sun!), square-on,
     no shadows. (Fluorescent: aim at a CFL/tube, or paper lit only by one.)
  3. Script fits and writes calibration/wavelength.json. Every later
     scan / reconstruct / experiment then works in nm automatically.

    python cal_wavelength.py --exposure-us 2000 --gain 1
    python cal_wavelength.py --exposure-us 20000 --gain 1 --source fluorescent --validate

If automatic matching picks the wrong lines (check the figure), pass
anchors explicitly, e.g.  --anchors 312.4:589.3,401.0:656.3,150.2:486.1
(spectral column : nm). Two or three obvious lines are enough.

Also: --reanalyse <run dir> to re-fit saved frames without the camera.
"""

import argparse
import datetime
import os

import numpy as np

from hsi import camera, plotting, spectral, store
from hsi.plotting import plt


def desmile(roi_img, smile_coeffs):
    """Shift every row so its features line up with the reference row."""
    x = np.arange(roi_img.shape[1], dtype=np.float64)
    out = np.empty_like(roi_img, dtype=np.float64)
    for y in range(roi_img.shape[0]):
        out[y] = np.interp(x, x - np.polyval(smile_coeffs, y), roi_img[y])
    return out


def parse_anchors(s):
    pairs = [p.split(":") for p in s.split(",") if p]
    return [(float(a), float(b), f"anchor {b}") for a, b in pairs]


def capture(args, out):
    cam = camera.open_camera(args)
    cam.prompt("Cover the lens for a dark frame.", "dark")
    dark, _, _ = cam.average(args.n, args.method)
    msg = ("Aim at white paper in direct sunlight." if args.source == "sun"
           else "Aim at the fluorescent lamp (or paper lit only by it).")
    cam.prompt(msg, args.source)
    light, std, _ = cam.average(args.n, args.method)
    cam.set_controls(AeEnable=True)
    roi = [0, light.shape[0], 0, light.shape[1]]  # whole frame
    print(f"brightest pixels {np.percentile(light - dark, 99.9):.0f} counts (full scale ~3800; "
          "aim for ~2700, never saturated)")
    np.savez_compressed(os.path.join(out, "frames.npz"), dark=dark, light=light, std=std, roi=np.asarray(roi))
    return dark, light, roi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    camera.add_camera_args(ap)
    ap.add_argument("--source", choices=["sun", "fluorescent"], default="sun")
    ap.add_argument("--n", type=int, default=30, help="frames to average (more = less noise)")
    ap.add_argument("--degree", type=int, default=2, help="polynomial degree pixel -> nm")
    ap.add_argument("--prominence", type=float, default=0.02, help="line detection threshold")
    ap.add_argument("--tol-nm", type=float, default=4.0, help="matching tolerance")
    ap.add_argument("--anchors", default="", help="manual col:nm pairs, skips auto-matching")
    ap.add_argument("--validate", action="store_true",
                    help="don't fit — test the EXISTING calibration on this source's lines")
    ap.add_argument("--reanalyse", default=None, help="run dir of a previous capture")
    args = ap.parse_args()

    if args.reanalyse:
        out = args.reanalyse
        d = np.load(os.path.join(out, "frames.npz"))
        dark, light, roi = d["dark"], d["light"], d["roi"].tolist()
    else:
        out = store.new_run_dir("wavelength_calibration", args.source + ("_validate" if args.validate else ""))
        dark, light, roi = capture(args, out)

    roi_img = spectral.crop(light - dark, roi).astype(np.float64)
    kind = "absorption" if args.source == "sun" else "emission"
    catalogue = spectral.CATALOGUES[args.source]

    # --- smile ------------------------------------------------------------
    rows, shifts = spectral.measure_smile(roi_img)
    ref_row = roi_img.shape[0] / 2
    smile_coeffs = np.polyfit(rows, shifts, 2)
    smile_coeffs[-1] -= np.polyval(smile_coeffs, ref_row)  # zero at the reference row
    smile_pp = float(np.ptp(np.polyval(smile_coeffs, rows)))
    print(f"smile: {smile_pp:.2f} px peak-to-peak along the slit")

    # --- lines on the de-smiled mean spectrum -------------------------------
    spec = desmile(roi_img, smile_coeffs).mean(axis=0)
    pos, prom = spectral.find_lines(spec, kind, prominence=args.prominence)
    print(f"detected {len(pos)} {kind} features")

    if args.validate:
        model = spectral.WavelengthModel.from_json(store.load_wavelength())
        # positions are in de-smiled (reference-row) coordinates -> use the
        # model's reference-row mapping
        pred = np.polyval(model.coeffs, pos)
        names = list(catalogue)
        lams = np.array([catalogue[n] for n in names])
        rows_out = []
        for p, lp in zip(pos, pred):
            k = int(np.argmin(np.abs(lams - lp)))
            if abs(lams[k] - lp) < args.tol_nm:
                rows_out.append((p, lams[k], names[k], lp - lams[k]))
        errs = np.array([r[3] for r in rows_out])
        res = {
            "mode": "validate",
            "validated_source": args.source,
            "calibration_source": model.extra.get("source"),
            "n_lines": len(rows_out),
            "lines": [{"col": r[0], "nm": r[1], "name": r[2], "error_nm": r[3]} for r in rows_out],
            "rms_error_nm": float(np.sqrt(np.mean(errs ** 2))) if len(errs) else None,
            "max_abs_error_nm": float(np.max(np.abs(errs))) if len(errs) else None,
        }
        for r in rows_out:
            print(f"  {r[2]:<18} true {r[1]:7.1f}  predicted {r[1] + r[3]:7.2f}  error {r[3]:+.2f} nm")
        print(f"VALIDATION RMS error on independent {args.source} lines: {res['rms_error_nm']} nm")
        store.save_json(os.path.join(out, "results.json"), res)
        return

    matches = parse_anchors(args.anchors) if args.anchors else spectral.match_lines(pos, catalogue, args.tol_nm)
    if len(matches) < args.degree + 2:
        raise SystemExit(f"only {len(matches)} lines matched — try --anchors, a lower --prominence, "
                         "or a better-exposed capture. See the figure.")
    px = np.array([m[0] for m in matches])
    nm = np.array([m[1] for m in matches])
    coeffs, resid, loo = spectral.fit_dispersion(px, nm, args.degree)
    fit_rms = float(np.sqrt(np.mean(resid ** 2)))
    loo_rms = float(np.sqrt(np.mean(loo ** 2))) if len(loo) else None
    disp = float(np.polyval(np.polyder(coeffs), roi_img.shape[1] / 2))
    lam_ends = np.polyval(coeffs, [0, roi_img.shape[1] - 1])

    res = {
        "source": args.source,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "roi": roi,
        "coeffs": coeffs,
        "smile_coeffs": smile_coeffs,
        "smile_peak_to_peak_px": smile_pp,
        "degree": args.degree,
        "nm_per_px": disp,
        "range_nm": sorted(map(float, lam_ends)),
        "fit_rms_nm": fit_rms,
        "loo_rms_nm": loo_rms,
        "lines": [{"col": m[0], "nm": m[1], "name": m[2], "resid_nm": float(r)}
                  for m, r in zip(matches, resid)],
    }

    if kind == "emission":
        fwhm_px = [spectral.line_fwhm(spec, m[0]) for m in matches]
        res["resolution_fwhm_nm"] = float(np.median(fwhm_px) * abs(disp))
        print(f"spectral resolution (median line FWHM): {res['resolution_fwhm_nm']:.2f} nm")

    for m, r in zip(matches, resid):
        print(f"  {m[2]:<18} col {m[0]:7.2f}  {m[1]:7.1f} nm  residual {r:+.2f} nm")
    print(f"dispersion {disp:.3f} nm/px, range {res['range_nm'][0]:.0f}-{res['range_nm'][1]:.0f} nm")
    print(f"fit RMS {fit_rms:.2f} nm   leave-one-out RMS {loo_rms if loo_rms is None else round(loo_rms, 2)} nm  <- quote this")

    store.save_json(os.path.join(out, "results.json"), res)
    store.save_json(store.calib_path("wavelength.json"), res)
    print(f"wrote {store.calib_path('wavelength.json')}")

    # --- figures ----------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [2, 1, 1]})
    lam_axis = np.polyval(coeffs, np.arange(len(spec)))
    ax = axes[0]
    ax.plot(lam_axis, spec, color=plotting.SERIES[0], lw=1.2)
    for m in matches:
        y = np.interp(m[1], lam_axis, spec) if lam_axis[0] < lam_axis[-1] else np.interp(m[1], lam_axis[::-1], spec[::-1])
        ax.axvline(m[1], color=plotting.MUTED, lw=0.8, ls="--")
        ax.annotate(m[2], (m[1], y), textcoords="offset points", xytext=(2, 6), fontsize=8, color=plotting.INK, rotation=90)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("counts (dark-subtracted)")
    ax.set_title(f"{args.source} spectrum, reference row, with matched lines")

    ax = axes[1]
    ax.axhline(0, color=plotting.MUTED, lw=1)
    ax.plot(nm, resid, "o", ms=8, color=plotting.SERIES[0], label="fit residual")
    if len(loo):
        ax.plot(nm, loo, "s", ms=8, mfc="none", color=plotting.SERIES[1], label="leave-one-out")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("error (nm)")
    ax.set_title(f"Calibration error — LOO RMS {loo_rms if loo_rms is None else f'{loo_rms:.2f}'} nm")
    ax.legend()

    ax = axes[2]
    ax.plot(rows, shifts, "o", ms=8, color=plotting.SERIES[0], label="measured")
    yy = np.linspace(0, roi_img.shape[0], 100)
    ax.plot(yy, np.polyval(smile_coeffs, yy) + (shifts.mean() - np.polyval(smile_coeffs, rows).mean()),
            color=plotting.SERIES[1], label="quadratic fit")
    ax.set_xlabel("row along slit (px)")
    ax.set_ylabel("column shift (px)")
    ax.set_title(f"Smile: {smile_pp:.2f} px peak-to-peak")
    ax.legend()
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "wavelength_calibration.png"))


if __name__ == "__main__":
    main()
