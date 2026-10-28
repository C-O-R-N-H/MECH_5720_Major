"""
Experiment: line spread function along the slit
=================================================
A small white light source, further away than the collecting lens's focal
length, is imaged onto the slit as a short bright spot. The spectrometer
spreads it into a horizontal stripe (its spectrum) whose height is the blur
along the slit: the line spread function (LSF). Its full width at half
maximum (FWHM) is the system's spatial resolution along the slit.

Measured at several heights along the slit (does focus change across the
field?) and in the blue, green and red thirds of the spectrum separately
(does focus change with colour?).

Two ways to measure it (both can go in the same run):

  point source  a small bright source imaged as a thin line on the slit;
                its profile along the slit IS the LSF.
                    python exp_line_spread.py capture middle
  edge          a straight sharp edge across a lit white card, at right
                angles to the slit (e.g. black tape across the card), so the
                brightness steps from bright to dark along the slit. That
                step's blur is the edge spread function (ESF); its slope is
                the LSF. Static and easy to set up — the standard
                "knife-edge" method — and independent of the source size.
                    python exp_line_spread.py edge middle

Run one step per command (each adds to the most recent run; `--new` starts
a fresh one):

    python exp_line_spread.py dark          optional: lens capped. Without it
                                            the session dark (take_dark.py) is
                                            used, if taken at the same settings
    python exp_line_spread.py edge middle   (or capture <name> for a point source;
                                            any names, as many as you like)
    python exp_line_spread.py analyse

Tips: check with snapshot.py first; nothing should saturate (lower
--exposure-us if the brightest raw pixel nears 1023, and use the same
setting for the dark). For the edge, the card must be evenly lit across it
and the edge straight and sharp (tape or a cut card edge, not a torn one).

Output: captures/line_spread/<stamp>/  <step>.npz, line_spread.png, results.json
"""

import argparse
import datetime
import glob
import os

import numpy as np

from hsi import camera, plotting, spectral, store
from hsi.plotting import plt

SATURATED = 0.95 * 1023
BAND_NAMES = ("blue", "green", "red")


def capture(args):
    run = None if args.new else store.latest_run_dir("line_spread")
    if run is None or (args.step == "dark" and os.path.exists(os.path.join(run, "dark.npz")) and not args.redo):
        run = store.new_run_dir("line_spread")
    name = {"dark": "dark", "capture": f"pos_{args.position}", "edge": f"edge_{args.position}"}[args.step]
    if os.path.exists(os.path.join(run, f"{name}.npz")) and not args.redo:
        raise SystemExit(f"{name} already captured in {run}; add --redo to replace it, or --new for a new run")
    cam = camera.open_camera(args)
    frac = {"top": 0.2, "middle": 0.5, "bottom": 0.8}.get(args.position, 0.5)
    if args.step == "dark":
        cam.prompt("Source OFF (or lens capped) for the dark frame.", "dark")
    elif args.step == "edge":
        cam.prompt(f"Lit card with a sharp edge across the slit, near the {args.position}.", f"edge:{frac}")
    else:
        cam.prompt(f"Source on, imaged at the {args.position} of the slit.", f"point:{frac}")
    rgb, peak = camera.average_rgb(cam, args.n)
    cam.set_controls(AeEnable=True)
    frame = rgb[..., 0] + 2 * rgb[..., 1] + rgb[..., 2]           # summed superpixel
    np.savez_compressed(os.path.join(run, f"{name}.npz"), frame=frame, peak=peak, n=args.n,
                        exposure_us=args.exposure_us, gain=args.gain,
                        time=datetime.datetime.now().isoformat(timespec="seconds"))
    print(f"{name}: brightest raw pixel {peak:.0f} / 1023"
          + ("  WARNING: saturating — lower --exposure-us (and retake the dark at the new setting)"
             if peak >= SATURATED and args.step != "dark" else ""))
    print(f"saved in {run}")


def fwhm_profile(p):
    """Peak position (sub-pixel) and FWHM of a profile with its baseline
    removed (median of the rows away from the peak)."""
    p = np.asarray(p, np.float64)
    k = int(np.argmax(p))
    far = np.r_[0:max(k - 40, 0), min(k + 40, len(p)):len(p)]
    base = np.median(p[far]) if len(far) > 10 else np.percentile(p, 10)
    q = p - base
    return spectral.subpixel_peak(q, k), spectral.line_fwhm(q, k), float(q[k]), q


def find_edges(profile, rows=None, max_edges=2):
    """Edges in a profile along the slit (e.g. the two sides of a tape strip).

    The LSF is the slope of the edge spread function, so: lightly smooth the
    profile (pixel noise would dominate the derivative otherwise), take
    |slope|, and treat each strong peak as one edge. Each edge's extent is
    found by walking out from its peak until the slope drops below 5% of it,
    so the window adapts to however blurred the edge is. The search ignores
    the outer 8% of the slit at each end (the end of the lit field would give
    a false edge) unless `rows` (first, last) says otherwise.

    Returns a list of dicts (strongest first): row (50% crossing), rise_px
    (10-90%), fwhm_px (of the LSF), contrast (bright / dark), rising (bool),
    plus the smoothed profile and |slope| for plotting."""
    from scipy.signal import savgol_filter

    p = savgol_filter(np.asarray(profile, np.float64), 5, 2)
    g = np.gradient(p)
    a = np.abs(g)
    m = int(0.08 * len(p))
    r0, r1 = rows if rows else (m, len(p) - m)
    search = a.copy()
    search[:r0] = 0
    search[r1:] = 0
    edges = []
    while len(edges) < max_edges:
        k = int(np.argmax(search))
        if search[k] <= 0 or (edges and search[k] < 0.25 * edges[0]["slope"]):
            break
        lo = k
        while lo > 0 and a[lo] > 0.05 * a[k]:
            lo -= 1
        hi = k
        while hi < len(a) - 1 and a[hi] > 0.05 * a[k]:
            hi += 1
        seg = p[lo:hi + 1]
        rising = bool(g[k] > 0)
        if not rising:
            seg = seg[::-1]
        dark, bright = np.median(seg[:3]), np.median(seg[-3:])

        def crossing(frac):
            level = dark + frac * (bright - dark)
            i = int(np.argmax(seg >= level))
            if i == 0:
                return 0.0
            return i - 1 + (level - seg[i - 1]) / max(seg[i] - seg[i - 1], 1e-9)

        c50 = crossing(0.5)
        edges.append({"row": float(lo + c50 if rising else hi - c50),
                      "rise_px": float(crossing(0.9) - crossing(0.1)),
                      "fwhm_px": float(spectral.line_fwhm(a, k)),
                      "contrast": float(bright / max(dark, 1e-9)),
                      "rising": rising, "slope": float(a[k])})
        search[lo:hi + 1] = 0
    return edges, p, a


def analyse(args):
    run = args.run or store.latest_run_dir("line_spread")
    if run is None:
        raise SystemExit("no line-spread run yet — e.g. python exp_line_spread.py edge middle")
    if os.path.exists(os.path.join(run, "dark.npz")):
        dark = dict(np.load(os.path.join(run, "dark.npz")))
    elif os.path.exists(store.calib_path("dark.npz")):
        d = np.load(store.calib_path("dark.npz"))  # the session dark from take_dark.py
        dark = {"frame": d["dark"], "exposure_us": d["exposure_us"], "gain": d["gain"]}
        print("no dark in this run — using the session dark (calibration/dark.npz)")
    else:
        raise SystemExit("no dark frame — run: python exp_line_spread.py dark")
    captures = sorted(glob.glob(os.path.join(run, "pos_*.npz")) + glob.glob(os.path.join(run, "edge_*.npz")),
                      key=os.path.getmtime)
    if not captures:
        raise SystemExit(f"nothing captured in {run} — python exp_line_spread.py edge middle")
    search_rows = tuple(int(v) for v in args.rows.split(":")) if args.rows else None

    res = {"run": run, "strip_mm": args.strip_mm, "captures": {}}
    curves = []                                              # (name, rows, lsf, kind)
    for path in captures:
        d = np.load(path)
        base = os.path.basename(path)[:-4]
        is_edge = base.startswith("edge_")
        name = ("edge " + base[5:]) if is_edge else base[4:]
        if (float(d["exposure_us"]), float(d["gain"])) != (float(dark["exposure_us"]), float(dark["gain"])):
            print(f"WARNING: {name} and the dark frame were taken at different settings")
        img = d["frame"] - dark["frame"]                       # (rows along slit, spectral cols)
        # spectral columns with real signal, split into blue / green / red thirds
        level = np.median(np.sort(img, axis=0)[-20:], axis=0)  # brightest rows per column
        use = np.where(level >= 0.2 * level.max())[0]
        thirds = np.array_split(use, 3)
        r = {"method": "edge" if is_edge else "point source", "saturated": bool(float(d["peak"]) >= SATURATED)}

        if is_edge:
            edges, smooth, slope = find_edges(img[:, use].mean(axis=1), search_rows)
            if not edges or max(e["contrast"] for e in edges) < 2:
                print(f"WARNING: {name}: no clear edge (bright/dark under 2x) — is the edge in view?")
            r["edges"] = [{k: v for k, v in e.items() if k != "slope"} for e in edges]
            r["fwhm_px"] = float(np.mean([e["fwhm_px"] for e in edges])) if edges else np.nan
            r["rise_px"] = float(np.mean([e["rise_px"] for e in edges])) if edges else np.nan
            for bname, cols in zip(BAND_NAMES, thirds):
                be, _, _ = find_edges(img[:, cols].mean(axis=1), search_rows)
                r[bname] = {"fwhm_px": float(np.mean([e["fwhm_px"] for e in be])) if be else np.nan,
                            "rise_px": float(np.mean([e["rise_px"] for e in be])) if be else np.nan}
            if len(edges) == 2 and edges[0]["rising"] != edges[1]["rising"]:
                r["strip_width_px"] = abs(edges[0]["row"] - edges[1]["row"])
                if args.strip_mm:
                    mm = args.strip_mm / r["strip_width_px"]
                    r["mm_per_row"] = mm
                    r["fwhm_mm"] = r["fwhm_px"] * mm
                    r["rise_mm"] = r["rise_px"] * mm
                    r["slit_length_mm"] = img.shape[0] * mm
            curves.append((name, np.arange(len(slope)), slope, smooth, edges))
            print(f"{name:<14}" + "   ".join(
                f"edge at row {e['row']:.0f}: 10-90% {e['rise_px']:.1f} px, FWHM {e['fwhm_px']:.1f} px, "
                f"contrast {e['contrast']:.0f}x" for e in edges))
            print(" " * 14 + "by colour (mean FWHM): " + "   ".join(f"{b} {r[b]['fwhm_px']:.1f}" for b in BAND_NAMES)
                  + (f"   strip {r['strip_width_px']:.0f} px wide" if "strip_width_px" in r else ""))
            if "mm_per_row" in r:
                print(" " * 14 + f"with a {args.strip_mm:g} mm strip: {r['mm_per_row'] * 1000:.1f} um per row, "
                      f"blur FWHM {r['fwhm_mm']:.2f} mm (10-90% {r['rise_mm']:.2f} mm), "
                      f"slit sees {r['slit_length_mm']:.1f} mm of the object")
        else:
            pk, w, amp, q = fwhm_profile(img[:, use].mean(axis=1))
            r.update({"peak_row": pk, "fwhm_px": w, "peak_counts": amp})
            for bname, cols in zip(BAND_NAMES, thirds):
                r[bname] = {"fwhm_px": fwhm_profile(img[:, cols].mean(axis=1))[1]}
            curves.append((name, np.arange(len(q)), q, None, [{"row": pk, "fwhm_px": w}]))
            print(f"{name:<14}point at row {pk:.0f}: FWHM {w:.2f} px   by colour: "
                  + "   ".join(f"{b} {r[b]['fwhm_px']:.2f}" for b in BAND_NAMES))
        res["captures"][name] = r

    # --- figure: LSF of every edge / point, centred; and FWHM by colour ---
    span = max(30.0, 1.5 * max(e["fwhm_px"] for _, _, _, _, es in curves for e in es))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    n = 0
    for name, x, lsf, _, es in curves:
        for e in es:
            c = plotting.SERIES[n % len(plotting.SERIES)]
            n += 1
            k = int(round(e["row"]))
            w = np.abs(x - k) <= span
            axes[0].plot(x[w] - e["row"], lsf[w] / lsf[w].max(), color=c,
                         label=f"{name}, row {e['row']:.0f}: FWHM {e['fwhm_px']:.1f} px")
    axes[0].axhline(0.5, color=plotting.MUTED, lw=1, ls=":")
    axes[0].set(xlim=(-span, span), xlabel="rows from the edge / point (along the slit)",
                ylabel="normalised LSF", title="Line spread function")
    axes[0].legend(fontsize=8)
    names = list(res["captures"])
    xs = np.arange(len(names))
    for j, b in enumerate(BAND_NAMES):
        axes[1].bar(xs + (j - 1) * 0.26, [res["captures"][nm][b]["fwhm_px"] for nm in names], 0.24,
                    color=["#2a6fd6", "#2e9e4f", "#d03b3b"][j], label=f"{b} third")
    top = np.nanmax([res["captures"][nm][b]["fwhm_px"] for nm in names for b in BAND_NAMES])
    axes[1].set_xticks(xs, names)
    axes[1].set(ylabel="FWHM (px along the slit)", title="Blur by position and colour", ylim=(0, 1.3 * top))
    axes[1].legend(fontsize=8, ncol=3, loc="upper center")
    fig.tight_layout()
    plotting.save(fig, os.path.join(run, "line_spread.png"))

    edge_curves = [cv for cv in curves if cv[3] is not None]
    if edge_curves:  # the raw edge profiles along the whole slit, edges marked
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, (name, x, _, smooth, es) in enumerate(edge_curves):
            c = plotting.SERIES[i % len(plotting.SERIES)]
            ax.plot(x, smooth, color=c, label=name)
            for e in es:
                ax.axvline(e["row"], color=c, ls="--", lw=1)
        ax.set(xlabel="row along the slit", ylabel="counts (dark-subtracted)",
               title="Edge profiles (dashed = 50% point of each edge)")
        ax.legend(fontsize=8)
        plotting.save(fig, os.path.join(run, "edge_profiles.png"))
    store.save_json(os.path.join(run, "results.json"), res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("dark", "capture", "edge", "analyse"))
    ap.add_argument("position", nargs="?", default="middle", help="name of the position for 'capture'")
    camera.add_camera_args(ap)
    ap.add_argument("--n", type=int, default=20, help="frames averaged per step")
    ap.add_argument("--new", action="store_true", help="start a new run instead of adding to the latest")
    ap.add_argument("--redo", action="store_true", help="replace this step in the latest run")
    ap.add_argument("--run", default=None, help="analyse this run folder instead of the latest")
    ap.add_argument("--strip-mm", type=float, default=0.0,
                    help="width of the tape strip on the object (mm): converts pixels to mm")
    ap.add_argument("--rows", default="", help="edge mode: only look for the edge between these rows, e.g. 100:400")
    args = ap.parse_args()
    if args.step == "analyse":
        analyse(args)
    else:
        capture(args)


if __name__ == "__main__":
    main()
