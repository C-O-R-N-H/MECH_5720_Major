"""
Experiment: exposure & gain sweep
===================================
How signal, noise and capture rate depend on the camera settings, to
justify the exposure/gain used for scans.

Phase 1 (object still, lit as for scanning): --n frames at each setting
    exposure sweep at --sweep-gain:   --exposures (ms)
    gain sweep at --sweep-exposure:   --gains
Phase 2 (lens covered): the same settings again, so every setting gets its
    own dark frame (the dark level changes with exposure and gain).

Per setting:
    signal       dark-subtracted mean over the bright part of the frame
    noise        frame-to-frame std of a pixel (median over that region)
    snr          signal / noise for a single pixel in a single frame
    read_noise   frame-to-frame std of a dark pixel
    near_sat_pct % of superpixels where any colour pixel is >= 90% of 1023
                 (settings above 0.1% are left out of the linearity fits)
    rate_hz      frames per second the server delivered (exposure-limited)
Plus:
    linearity    signal vs exposure / gain, straight-line fit through zero,
                 worst deviation from it (%)
    photon transfer  noise^2 vs signal along the exposure sweep. Shot noise
                 makes variance grow linearly with signal; the slope is
                 counts per detected photon, so signal / slope = photons.

    python exp_exposure_sweep.py                       (both phases, prompts in between)
    python exp_exposure_sweep.py --phase light         (then cover the lens)
    python exp_exposure_sweep.py --phase dark --run <folder>
Output: captures/exposure_sweep/<stamp>/  sweep.png, results.json, sweep.csv
"""

import argparse
import csv
import os
import time

import numpy as np

from hsi import camera, plotting, store
from hsi.plotting import plt

# superpixel = sum of four 10-bit pixels
PIXEL_MAX = 1023  # 10-bit raw


def grab_set(cam, exposure_ms, gain, n, method):
    """n frames at one setting -> (summed superpixel frames, frames/s,
    % of superpixels where any colour pixel is near its 10-bit limit).
    Saturation is checked per colour pixel: one green pixel can hit 1023
    while the 2x2 sum still looks well below 4 x 1023."""
    cam.lock_exposure(exposure_ms * 1000, gain)
    t0 = time.time()
    rgb = np.array([cam.grab_rgb()[0] for _ in range(n)])
    rate = n / (time.time() - t0)
    frames = rgb[..., 0] + 2 * rgb[..., 1] + rgb[..., 2]
    sat_pct = 100 * float(np.mean(rgb.max(axis=(0, 3)) >= 0.9 * PIXEL_MAX))
    return frames, rate, sat_pct


def save_light(out, light):
    np.savez(os.path.join(out, "light.npz"), keys=np.array([k for k in light]),
             frames=np.array([v[0] for v in light.values()]), rates=np.array([v[1] for v in light.values()]),
             sat_pct=np.array([v[2] for v in light.values()]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    camera.add_camera_args(ap)
    ap.add_argument("--exposures", default="25,50,100,200,400", help="ms, swept at --sweep-gain")
    ap.add_argument("--gains", default="1,2,4,8,16", help="swept at --sweep-exposure")
    ap.add_argument("--sweep-gain", type=float, default=8.0)
    ap.add_argument("--sweep-exposure", type=float, default=100.0, help="ms")
    ap.add_argument("--n", type=int, default=5, help="frames per setting")
    ap.add_argument("--phase", choices=["both", "light", "dark"], default="both",
                    help="light/dark = run the two halves separately (cover the lens in between)")
    ap.add_argument("--run", default=None, help="folder of the light phase, for --phase dark")
    ap.add_argument("--redo", default="",
                    help='with --phase dark --run: recapture just these LIGHT settings, e.g. "100:1" '
                         "(exposure_ms:gain), reusing the saved darks")
    args = ap.parse_args()

    exposures = [float(x) for x in args.exposures.split(",")]
    gains = [float(x) for x in args.gains.split(",")]
    settings = [(e, args.sweep_gain, "exposure") for e in exposures]
    settings += [(args.sweep_exposure, g, "gain") for g in gains
                 if not (g == args.sweep_gain and args.sweep_exposure in exposures)]

    cam = camera.open_camera(args)
    if args.phase in ("light", "both"):
        out = store.new_run_dir("exposure_sweep")
        cam.prompt("Object still and lit as for scanning.", "lamp-patches")
        light = {}
        for e, g, _ in settings:
            light[(e, g)] = grab_set(cam, e, g, args.n, args.method)
            print(f"  light {e:g} ms x{g:g}: {light[(e, g)][1]:.1f} frames/s, "
                  f"{light[(e, g)][2]:.2f}% saturating")
        cam.set_controls(AeEnable=True)
        save_light(out, light)
        if args.phase == "light":
            print(f"light phase saved. Cover the lens, then run:\n  python exp_exposure_sweep.py "
                  f"--phase dark --run {out}")
            return
    else:
        if not args.run:
            raise SystemExit("--phase dark needs --run <folder from the light phase>")
        out = args.run
        L = np.load(os.path.join(out, "light.npz"))
        light = {tuple(k): (f, r, sp) for k, f, r, sp in zip(L["keys"], L["frames"], L["rates"], L["sat_pct"])}
    if args.redo:
        # recapture only these settings' LIGHT frames (e.g. one that went wrong)
        redo = [tuple(float(v) for v in x.split(":")) for x in args.redo.split(",")]
        cam.prompt("Uncover the lens; object still and lit as before.", "lamp-patches")
        for e, g in redo:
            light[(e, g)] = grab_set(cam, e, g, args.n, args.method)
            print(f"  light {e:g} ms x{g:g} (redone)")
        cam.set_controls(AeEnable=True)
        save_light(out, light)
        D = np.load(os.path.join(out, "dark.npz"))
        dark = {tuple(k): f for k, f in zip(D["keys"], D["frames"])}
    else:
        cam.prompt("Cover the lens completely (cap or black cloth).", "dark")
        dark = {}
        for e, g, _ in settings:
            dark[(e, g)] = grab_set(cam, e, g, args.n, args.method)[0]
            print(f"  dark  {e:g} ms x{g:g}")
        cam.set_controls(AeEnable=True)
        np.savez(os.path.join(out, "dark.npz"), keys=np.array([k for k in dark]),
                 frames=np.array([v for v in dark.values()]))

    # The bright region is fixed once (at the reference setting) so every
    # setting is measured on the same pixels.
    ref = (args.sweep_exposure, args.sweep_gain)
    ref_sig = light[ref][0].mean(0) - dark[ref].mean(0)
    bright = ref_sig > 0.5 * np.percentile(ref_sig, 99)

    rows = []
    for e, g, sweep in settings:
        frames, rate, sat_pct = light[(e, g)]
        d = dark[(e, g)]
        sig_img = frames.mean(0) - d.mean(0)
        # Noise from the MEAN per-pixel variance (unbiased), not the median
        # per-pixel std: with only a few frames the latter reads ~20% low,
        # which would throw off the photon transfer slope. Top 1% trimmed
        # so hot pixels don't count.
        def rms_noise(var):
            return float(np.sqrt(np.mean(np.sort(var.ravel())[: int(0.99 * var.size)])))
        signal = float(np.mean(sig_img[bright]))  # mean, to match the mean variance below
        noise = rms_noise(frames.var(0, ddof=1)[bright])
        read_noise = rms_noise(d.var(0, ddof=1))
        rows.append({
            "sweep": sweep, "exposure_ms": e, "gain": g,
            "signal": round(signal, 2),
            "noise": round(noise, 2),
            "snr": round(signal / noise, 2),
            "read_noise": round(read_noise, 2),
            "dark_level": round(float(np.median(d.mean(0))), 1),
            "near_sat_pct": round(sat_pct, 3),
            "rate_hz": round(rate, 2),
        })

    def fit_through_zero(x, y):
        """Slope of y = k x, and the worst % deviation from that line.
        Needs >= 2 unsaturated points; otherwise (nan, nan)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) < 2:
            return float("nan"), float("nan"), np.array([])
        k = float(np.sum(x * y) / np.sum(x * x))
        dev = 100 * (y - k * x) / (k * x)
        return k, float(np.max(np.abs(dev))), dev

    ex = [r for r in rows if r["gain"] == args.sweep_gain and r["exposure_ms"] in exposures]
    ex.sort(key=lambda r: r["exposure_ms"])
    gn = [r for r in rows if r["exposure_ms"] == args.sweep_exposure and r["gain"] in gains]
    gn.sort(key=lambda r: r["gain"])
    unsat_ex = [r for r in ex if r["near_sat_pct"] < 0.1]
    unsat_gn = [r for r in gn if r["near_sat_pct"] < 0.1]
    k_e, worst_e, _ = fit_through_zero([r["exposure_ms"] for r in unsat_ex], [r["signal"] for r in unsat_ex])
    # Also a line WITH an offset: a constant few-count offset (e.g. darks taken
    # at a different time to the light frames) looks like non-linearity in a
    # through-zero fit at short exposures, but isn't the sensor bending.
    xe_u = np.array([r["exposure_ms"] for r in unsat_ex], float)
    ye_u = np.array([r["signal"] for r in unsat_ex], float)
    if len(xe_u) >= 3:
        slope_o, icpt_o = np.polyfit(xe_u, ye_u, 1)
        worst_o = float(np.max(np.abs(100 * (ye_u - (slope_o * xe_u + icpt_o)) / ye_u)))
    else:
        slope_o = icpt_o = worst_o = float("nan")
    k_g, worst_g, _ = fit_through_zero([r["gain"] for r in unsat_gn], [r["signal"] for r in unsat_gn])

    # photon transfer along the exposure sweep: var_shot = var_total - var_read
    sig = np.array([r["signal"] for r in unsat_ex])
    var_shot = np.array([r["noise"] ** 2 - r["read_noise"] ** 2 for r in unsat_ex])
    counts_per_photon = float(np.sum(sig * var_shot) / np.sum(sig * sig)) if len(sig) >= 2 else float("nan")
    if len(unsat_ex) < 2 or len(unsat_gn) < 2:
        print("WARNING: fewer than 2 unsaturated settings in a sweep — lower the exposures/gains")
    res = {
        "settings": rows,
        "bright_region_pixels": int(bright.sum()),
        "linearity_exposure": {"counts_per_ms": k_e, "worst_deviation_pct": worst_e,
                               "with_offset": {"counts_per_ms": float(slope_o), "offset_counts": float(icpt_o),
                                               "worst_deviation_pct": worst_o},
                               "points_used": len(unsat_ex)},
        "linearity_gain": {"counts_per_gain_unit": k_g, "worst_deviation_pct": worst_g,
                           "points_used": len(unsat_gn)},
        "photon_transfer": {"gain": args.sweep_gain, "counts_per_photon": counts_per_photon,
                            "photons_per_pixel_at_reference": float(
                                [r for r in rows if (r["exposure_ms"], r["gain"]) == ref][0]["signal"]
                                / counts_per_photon) if counts_per_photon > 0 else None},
    }
    store.save_json(os.path.join(out, "results.json"), res)
    with open(os.path.join(out, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'sweep':<9}{'exp ms':>7}{'gain':>6}{'signal':>9}{'noise':>8}{'SNR':>7}{'read':>7}"
          f"{'dark':>7}{'sat %':>7}{'fps':>6}")
    for r in rows:
        print(f"{r['sweep']:<9}{r['exposure_ms']:>7g}{r['gain']:>6g}{r['signal']:>9.1f}{r['noise']:>8.1f}"
              f"{r['snr']:>7.1f}{r['read_noise']:>7.1f}{r['dark_level']:>7.0f}{r['near_sat_pct']:>7.2f}"
              f"{r['rate_hz']:>6.1f}")
    print(f"linearity vs exposure: worst deviation {worst_e:.1f}% through zero; "
          f"{worst_o:.1f}% allowing an offset of {icpt_o:+.1f} counts ({len(unsat_ex)} unsaturated points)")
    print(f"linearity vs gain:     worst deviation {worst_g:.1f}% ({len(unsat_gn)} unsaturated points)")
    print(f"photon transfer at x{args.sweep_gain:g}: {counts_per_photon:.2f} counts per photon -> "
          f"~{res['photon_transfer']['photons_per_pixel_at_reference']:.0f} photons per pixel at "
          f"{ref[0]:g} ms x{ref[1]:g}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    xe = np.array([r["exposure_ms"] for r in ex])
    ax.plot(xe, [r["signal"] for r in ex], "o", ms=8, color=plotting.SERIES[0], label="measured")
    ax.plot([0, xe.max()], [0, k_e * xe.max()], color=plotting.MUTED, lw=1.5, ls="--", label="straight line")
    ax.set(xlabel="exposure (ms)", ylabel="signal (counts)", title=f"Signal vs exposure (gain x{args.sweep_gain:g})")
    ax.legend()
    ax = axes[0, 1]
    xg = np.array([r["gain"] for r in gn])
    ax.plot(xg, [r["signal"] for r in gn], "o", ms=8, color=plotting.SERIES[0], label="measured")
    ax.plot([0, xg.max()], [0, k_g * xg.max()], color=plotting.MUTED, lw=1.5, ls="--", label="straight line")
    ax.set(xlabel="analogue gain", ylabel="signal (counts)", title=f"Signal vs gain ({args.sweep_exposure:g} ms)")
    ax.legend()
    ax = axes[1, 0]
    # SNR against signal level puts both sweeps on one axis and shows whether
    # signal bought with gain is worth as much as signal bought with exposure.
    ax.plot([r["signal"] for r in ex], [r["snr"] for r in ex], "o-", ms=8, color=plotting.SERIES[0],
            label=f"exposure sweep (x{args.sweep_gain:g})")
    ax.plot([r["signal"] for r in gn], [r["snr"] for r in gn], "s-", ms=8, color=plotting.SERIES[1],
            label=f"gain sweep ({args.sweep_exposure:g} ms)")
    for r in ex:
        ax.annotate(f"{r['exposure_ms']:g} ms", (r["signal"], r["snr"]), textcoords="offset points",
                    xytext=(4, -12), fontsize=8, color=plotting.INK)
    for r in gn:
        ax.annotate(f"x{r['gain']:g}", (r["signal"], r["snr"]), textcoords="offset points",
                    xytext=(4, 4), fontsize=8, color=plotting.INK)
    ax.set(xlabel="signal (counts)", ylabel="single-pixel SNR", title="Signal-to-noise vs signal")
    ax.legend()
    ax = axes[1, 1]
    ax.plot(sig, var_shot, "o", ms=8, color=plotting.SERIES[0])
    ax.plot([0, sig.max()], [0, counts_per_photon * sig.max()], color=plotting.MUTED, lw=1.5, ls="--")
    ax.set(xlabel="signal (counts)", ylabel="shot-noise variance (counts²)",
           title=f"Photon transfer: {counts_per_photon:.2f} counts per photon")
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "sweep.png"))


if __name__ == "__main__":
    main()
