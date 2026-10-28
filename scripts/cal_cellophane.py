"""
Calibration: cellophane filters + the camera's colour pixels
=============================================================
A broadband light shone through red, green and blue cellophane, one sheet
at a time, plus once with no sheet. Because the sensor has a known Bayer
pattern, every capture gives three spectra: one from the red pixels, one
from the green and one from the blue.

Run one step per command, changing the setup in between (each step adds to
the most recent run; `--new` starts a fresh one):

    python cal_cellophane.py dark       light OFF (or lens capped)
    python cal_cellophane.py white      light on, no cellophane
    python cal_cellophane.py blue       blue sheet over the light
    python cal_cellophane.py green
    python cal_cellophane.py red
    python cal_cellophane.py analyse    [--nm-blue 450 --nm-green 530 --nm-red 620] [--install]

Keep the light, its position and the camera settings identical for every
step; only the sheet changes. Make sure the cellophane covers the light
completely, so no unfiltered light gets past the edges.

What it reports
  band mapping   each filter's transmission (filtered / white) along the
                 spectrum, and which spectral columns are its band
                 (centroid and half-maximum range).
  method B       measured response of every pixel type to every filter:
                 a 3x3 matrix (e.g. blue pixels under green light = x% of
                 their white-light signal) and each pixel type's response
                 curve along the spectrum under each filter.
  method A       quick model: the green pixels' response to green light,
                 SHIFTED to where the red / blue pixels peak under white
                 light, taken as the red / blue pixels' response to their
                 own filters. Compared with method B: peak position, width
                 and shape (spectral angle).
  rough nm scale the three band centroids at the given nominal filter
                 wavelengths, straight-line fit -> nm per column. Cellophane
                 bands are broad and their exact centres unknown, so this is
                 APPROXIMATE (good to tens of nm). --install makes it the
                 active wavelength calibration (calibration/wavelength.json)
                 so scans and panoramas switch to approximate nm; the sunlight
                 calibration (cal_wavelength.py) supersedes it.

Output: captures/cellophane/<stamp>/  <step>.npz, spectra.png,
        methods.png, results.json, wavelength_approx.json
"""

import argparse
import datetime
import os

import numpy as np

from hsi import camera, plotting, spectral, store
from hsi.plotting import plt

STEPS = ("dark", "white", "blue", "green", "red")
FILTERS = ("blue", "green", "red")
CHANNELS = ("R", "G", "B")
PROMPTS = {
    "dark": ("Light OFF (or lens capped) for the dark frame.", "dark"),
    "white": ("Light ON, no cellophane.", "lamp-white"),
    "blue": ("BLUE cellophane over the light (fully covering it).", "filter-blue"),
    "green": ("GREEN cellophane over the light (fully covering it).", "filter-green"),
    "red": ("RED cellophane over the light (fully covering it).", "filter-red"),
}
SATURATED = 0.95 * 1023  # raw 10-bit value


def capture(args):
    run = None if args.new else store.latest_run_dir("cellophane")
    if run is None or (args.step == "dark" and os.path.exists(os.path.join(run, "dark.npz")) and not args.redo):
        run = store.new_run_dir("cellophane")
    if os.path.exists(os.path.join(run, f"{args.step}.npz")) and not args.redo:
        raise SystemExit(f"{args.step} already captured in {run}; add --redo to replace it, or --new for a new run")
    cam = camera.open_camera(args)
    cam.prompt(*PROMPTS[args.step])
    rgb, peak = camera.average_rgb(cam, args.n)
    cam.set_controls(AeEnable=True)
    np.savez_compressed(os.path.join(run, f"{args.step}.npz"), rgb=rgb, peak=peak, n=args.n,
                        exposure_us=args.exposure_us, gain=args.gain,
                        time=datetime.datetime.now().isoformat(timespec="seconds"))
    print(f"{args.step}: brightest raw pixel {peak:.0f} / 1023"
          + ("  WARNING: saturating — lower --exposure-us (and redo every step at the new setting)"
             if peak >= SATURATED and args.step != "dark" else ""))
    done = [s for s in STEPS if os.path.exists(os.path.join(run, f"{s}.npz"))]
    todo = [s for s in STEPS if s not in done]
    print(f"saved in {run}\ndone: {', '.join(done)}" + (f"   still to do: {', '.join(todo)}" if todo
                                                          else "   -> python cal_cellophane.py analyse"))


def centroid_and_band(x, t):
    """Transmission-weighted centre and half-maximum range of a band."""
    t = np.clip(np.nan_to_num(t), 0, None)
    above = np.where(t >= 0.5 * t.max())[0]
    w = t * (t >= 0.5 * t.max())
    return float(np.sum(x * w) / np.sum(w)), (int(above[0]), int(above[-1]))


def peak_and_fwhm(y):
    y = np.nan_to_num(y)
    k = int(np.argmax(y))
    return spectral.subpixel_peak(y, k), spectral.line_fwhm(y, k)


OWN_FILTER = {"R": "red", "G": "green", "B": "blue"}


def load_spectra(run):
    """Dark-subtracted spectra of one cellophane run.
    Returns (captures, rows used, spec {step: (cols, 3) R/G/B}, total
    {step: (cols,) R + 2G + B}, column index, columns with usable light)."""
    caps = {s: np.load(os.path.join(run, f"{s}.npz")) for s in STEPS}
    settings = {(float(c["exposure_us"]), float(c["gain"])) for c in caps.values()}
    if len(settings) > 1:
        print(f"WARNING: steps were captured at different settings {settings} — results are not comparable")
    dark = caps["dark"]["rgb"]
    img = {s: caps[s]["rgb"] - dark for s in STEPS[1:]}          # (rows, cols, 3)
    # rows along the slit that the light actually reaches (white, summed)
    row_level = np.median(img["white"].sum(axis=2), axis=1)
    rows = row_level >= 0.2 * row_level.max()
    spec = {s: img[s][rows].mean(axis=0) for s in img}          # (cols, 3) per step
    total = {s: spec[s][:, 0] + 2 * spec[s][:, 1] + spec[s][:, 2] for s in spec}
    x = np.arange(len(total["white"]))
    lit = total["white"] >= 0.1 * total["white"].max()           # columns with usable white light
    return caps, rows, spec, total, x, lit


def method_a_vs_b(spec, x, lit):
    """Method A: the green pixels' response to green light, shifted to where
    each pixel type peaks under white light, scaled to the same total, vs
    method B: that pixel type's measured response to its own filter."""
    white_peaks = {ch: peak_and_fwhm(spec["white"][:, i])[0] for i, ch in enumerate(CHANNELS)}
    g_resp = spec["green"][:, 1]                                   # green pixels, green light
    meth = {}
    for i, ch in enumerate(CHANNELS):
        measured = spec[OWN_FILTER[ch]][:, i]
        shift = white_peaks[ch] - white_peaks["G"]
        model = np.interp(x - shift, x, g_resp, left=0, right=0)
        model *= np.nansum(measured[lit]) / max(np.nansum(model[lit]), 1e-9)  # compare shapes
        pm, wm = peak_and_fwhm(measured)
        pa, wa = peak_and_fwhm(model)
        meth[ch] = {"shift_cols": float(shift), "peak_measured_col": pm, "peak_model_col": pa,
                    "peak_error_cols": pa - pm, "fwhm_measured_cols": wm, "fwhm_model_cols": wa,
                    "shape_angle_deg": float(spectral.spectral_angle(measured[lit], model[lit])),
                    "model": model, "measured": measured}
    return meth


def analyse(args):
    run = args.run or store.latest_run_dir("cellophane")
    if run is None:
        raise SystemExit("no cellophane run yet — start with: python cal_cellophane.py dark")
    missing = [s for s in STEPS if not os.path.exists(os.path.join(run, f"{s}.npz"))]
    if missing:
        raise SystemExit(f"{run} is missing: {', '.join(missing)}")
    caps, rows, spec, total, x, lit = load_spectra(run)
    cols = np.where(lit)[0]

    res = {"run": run, "rows_used": int(rows.sum()), "columns_used": [int(cols[0]), int(cols[-1])],
           "saturated_steps": [s for s in STEPS[1:] if float(caps[s]["peak"]) >= SATURATED]}
    if res["saturated_steps"]:
        print(f"WARNING: saturated in {res['saturated_steps']} — numbers there are unreliable")

    # --- band mapping ----------------------------------------------------
    trans = {f: np.where(lit, total[f] / np.maximum(total["white"], 1e-9), np.nan) for f in FILTERS}
    bands = {}
    for f in FILTERS:
        c, (a, b) = centroid_and_band(x, trans[f])
        bands[f] = {"centroid_col": c, "half_max_cols": [a, b], "peak_transmission": float(np.nanmax(trans[f]))}
    order_ok = bands["blue"]["centroid_col"] < bands["green"]["centroid_col"] < bands["red"]["centroid_col"]
    res["bands"] = bands
    res["band_order_blue_green_red"] = bool(order_ok)
    if not order_ok:
        print("WARNING: bands are not in blue < green < red order along the columns — check the filters "
              "and that frames are transposed the right way (--transpose / --no-transpose)")

    # --- method B: response matrix ----------------------------------------
    white_int = spec["white"][lit].sum(axis=0)                     # per channel
    M = np.array([[spec[f][lit, c].sum() / white_int[c] for f in FILTERS] for c in range(3)])  # (channel, filter)
    share = np.array([[spec[f][lit, c].sum() for f in FILTERS] for c in range(3)])
    share = share / share.sum(axis=0, keepdims=True)              # each filter: split among pixel types
    res["method_B"] = {
        "fraction_of_white_signal": {ch: {f: float(M[i, j]) for j, f in enumerate(FILTERS)}
                                     for i, ch in enumerate(CHANNELS)},
        "share_of_filtered_light": {f: {ch: float(share[i, j]) for i, ch in enumerate(CHANNELS)}
                                    for j, f in enumerate(FILTERS)},
    }

    # --- method A vs B ----------------------------------------------------
    own = OWN_FILTER
    meth = method_a_vs_b(spec, x, lit)
    res["method_A_vs_B"] = {ch: {k: v for k, v in m.items() if k not in ("model", "measured")}
                            for ch, m in meth.items()}

    # --- rough nm scale ---------------------------------------------------
    nominal = {"blue": args.nm_blue, "green": args.nm_green, "red": args.nm_red}
    cx = np.array([bands[f]["centroid_col"] for f in FILTERS])
    ny = np.array([nominal[f] for f in FILTERS])
    coeffs = np.polyfit(cx, ny, 1)
    resid = ny - np.polyval(coeffs, cx)
    approx = {"coeffs": coeffs, "smile_coeffs": [0.0], "roi": [0, len(rows), 0, len(x)],
              "source": "cellophane filters (APPROXIMATE)", "approximate": True,
              "nominal_nm": nominal, "anchor_cols": cx, "residuals_nm": resid,
              "nm_per_px": float(coeffs[0]), "range_nm": sorted(np.polyval(coeffs, [0, len(x) - 1]).tolist()),
              "date": datetime.datetime.now().isoformat(timespec="seconds")}
    res["approx_nm"] = {k: approx[k] for k in ("nominal_nm", "anchor_cols", "residuals_nm", "nm_per_px", "range_nm")}
    store.save_json(os.path.join(run, "wavelength_approx.json"), approx)
    if args.install:
        store.save_json(store.calib_path("wavelength.json"), approx)
        print(f"installed as {store.calib_path('wavelength.json')} (approximate)")
    store.save_json(os.path.join(run, "results.json"), res)

    # --- print --------------------------------------------------------------
    print(f"rows used {res['rows_used']}, spectral columns {cols[0]}-{cols[-1]}")
    print("bands (centroid column, half-max range, peak transmission):")
    for f in FILTERS:
        b = bands[f]
        print(f"  {f:<6} {b['centroid_col']:7.1f}   {b['half_max_cols'][0]}-{b['half_max_cols'][1]}   "
              f"{100 * b['peak_transmission']:.0f}%")
    print("method B — fraction of each pixel type's white-light signal passed by each filter:")
    print("          " + "".join(f"{f:>8}" for f in FILTERS))
    for i, ch in enumerate(CHANNELS):
        print(f"  {ch} pixels" + "".join(f"{100 * M[i, j]:7.0f}%" for j in range(3)))
    print("method A (shifted green) vs B (measured), each pixel type under its own filter:")
    for ch in CHANNELS:
        m = meth[ch]
        print(f"  {ch}: shift {m['shift_cols']:+.0f} cols, peak error {m['peak_error_cols']:+.1f} cols, "
              f"FWHM {m['fwhm_model_cols']:.0f} vs {m['fwhm_measured_cols']:.0f} cols, "
              f"shape angle {m['shape_angle_deg']:.1f} deg")
    print(f"rough nm scale: {coeffs[0]:.3f} nm/column, {approx['range_nm'][0]:.0f}-{approx['range_nm'][1]:.0f} nm "
          f"across the sensor; residuals {', '.join(f'{r:+.1f}' for r in resid)} nm (APPROXIMATE)")

    # --- figures ------------------------------------------------------------
    chan_col = {"R": "#d03b3b", "G": "#2e9e4f", "B": "#2a6fd6"}  # pixel colour = identity here
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    for i, ch in enumerate(CHANNELS):
        ax.plot(x, spec["white"][:, i], color=chan_col[ch], label=f"{ch} pixels")
    ax.set(xlabel="spectral column (blue → red)", ylabel="counts", title="White light, by pixel type")
    ax.legend()
    ax = axes[0, 1]
    for f in FILTERS:
        ax.plot(x, 100 * trans[f], color=chan_col[f[0].upper()], label=f"{f} cellophane")
        a, b = bands[f]["half_max_cols"]
        ax.axvspan(a, b, color=chan_col[f[0].upper()], alpha=0.08, lw=0)
        ax.axvline(bands[f]["centroid_col"], color=chan_col[f[0].upper()], ls="--", lw=1)
    ax.set(xlabel="spectral column", ylabel="transmission (%)", title="Filter transmission (shaded = band)")
    ax.legend()
    ax = axes[1, 0]
    im = ax.imshow(100 * M, cmap="Blues", vmin=0)
    ax.set_xticks(range(3), [f"{f}\nlight" for f in FILTERS])
    ax.set_yticks(range(3), [f"{ch} pixels" for ch in CHANNELS])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{100 * M[i, j]:.0f}%", ha="center", va="center",
                    color="white" if M[i, j] > 0.6 * M.max() else plotting.INK)
    ax.set_title("Method B: % of white-light signal passed")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1, 1]
    ax.plot(cx, ny, "o", ms=9, color=plotting.SERIES[0], label="filter band centres (nominal nm)")
    ax.plot(x, np.polyval(coeffs, x), color=plotting.MUTED, ls="--", lw=1.5, label="straight-line fit")
    ax.set(xlabel="spectral column", ylabel="wavelength (nm, approximate)",
           title=f"Rough nm scale: {coeffs[0]:.2f} nm/column")
    ax.legend()
    fig.tight_layout()
    plotting.save(fig, os.path.join(run, "spectra.png"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for ax, ch in zip(axes, CHANNELS):
        m = meth[ch]
        ax.plot(x, m["measured"], color=chan_col[ch], label="B: measured")
        ax.plot(x, m["model"], color=plotting.INK, ls="--", lw=1.5, label="A: shifted green")
        ax.set(xlabel="spectral column", ylabel="counts",
               title=f"{ch} pixels, {own[ch]} light: " + ("the model's source" if ch == "G"
                                                         else f"angle {m['shape_angle_deg']:.1f}°"))
        ax.legend(fontsize=8)
    fig.suptitle("Method A (green response shifted) vs method B (measured)", fontweight="bold")
    fig.tight_layout()
    plotting.save(fig, os.path.join(run, "methods.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ("analyse",))
    camera.add_camera_args(ap)
    ap.add_argument("--n", type=int, default=20, help="frames averaged per step")
    ap.add_argument("--new", action="store_true", help="start a new run instead of adding to the latest")
    ap.add_argument("--redo", action="store_true", help="replace this step in the latest run")
    ap.add_argument("--run", default=None, help="analyse this run folder instead of the latest")
    ap.add_argument("--nm-blue", type=float, default=450.0)
    ap.add_argument("--nm-green", type=float, default=530.0)
    ap.add_argument("--nm-red", type=float, default=620.0)
    ap.add_argument("--install", action="store_true",
                    help="make the rough nm scale the active wavelength calibration")
    args = ap.parse_args()
    if args.step == "analyse":
        analyse(args)
    else:
        capture(args)


if __name__ == "__main__":
    main()
