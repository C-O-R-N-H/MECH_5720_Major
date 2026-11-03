"""
Experiment: real vs fake fruit (material comparison)
=====================================================
Each object is scanned on its own, centred on the turntable, so its
panorama is an unwrapped map of its whole side surface. A real fruit and
a convincing fake are the same colour to the eye; this asks whether their
spectra still tell them apart, and whether a 3-band (RGB-like) camera
could have done the same.

Capture, one scan per object (same lamp, same settings, nothing moved):

    python scan.py --experiment material_comparison --tag white --speed 28.15 --revs 2.5
    python scan.py --experiment material_comparison --tag real_apple --speed 28.15 --revs 2.5
    ... fake_apple, real_lemon, fake_lemon (any real_X / fake_X names pair up)
    python exp_material_comparison.py

`white` is optional: white paper wrapped round a can, scanned like the
fruit (or just a snapshot of it: it looks the same from every side, so
scans fall back to the white snapshot when there is no white scan).
With it, every spectrum is divided by the white one (per height), which
removes the lamp and the camera's colour response and gives relative
reflectance. Without it the comparison still works on counts, since the
spectral angle only compares shape and the lamp is the same in every scan.

Analysis (latest scan of each tag in captures/material_comparison/):
  * object pixels: bright enough to have a usable spectrum, minus the
    brightest 1% (specular highlights show the lamp, not the material);
    --rows limits the height, --pick draws a box per object instead
  * each object split into 12 azimuth sectors (30 deg of its surface);
    sector-to-sector angles within an object are the natural variation,
    the angle between objects is the material difference
  * leave-one-sector-out classification: every pixel in sector s is
    classified by nearest class mean (spectral angle) using means built
    from the other sectors only, so it is always tested on surface it
    wasn't trained on. Full spectrum vs the same data cut to 3 bands.

Snapshot mode (--snapshots): the same comparison from ONE still frame
per object instead of a scan, i.e. a single vertical line of each fruit.
Take them with snapshot.py into the same folder (white optional, as above):

    python snapshot.py --experiment material_comparison --tag real_apple --n 10
    python exp_material_comparison.py --snapshots

The 12 "sectors" are then 12 blocks of rows along the slit (different
heights of the fruit) instead of 30 deg azimuth sectors.

Output: captures/material_comparison/analysis/ (analysis_snapshot/ with --snapshots)
    panoramas.png    false colour (common scale) | what each pixel was classified as
    spectra.png      real vs fake spectra per fruit, and their ratio
    separation.png   angle matrices (full / 3-band) and accuracies
    results.json

    python exp_material_comparison.py                  # every tag found
    python exp_material_comparison.py real_apple fake_apple --rows 100:450
"""

import argparse
import glob
import os
import re

import cv2
import numpy as np

from exp_discrimination import nearest_class, three_band
from hsi import plotting, spectral, store
from hsi.plotting import plt

EXPERIMENT = "material_comparison"
N_SECTORS = 12


def find_scans(tags=None, fname="cube.npz"):
    """Latest run folder (with `fname`: cube.npz for scans, frames.npz for
    snapshots) for each tag, in capture order."""
    found = {}
    for d in sorted(glob.glob(os.path.join(store.CAPTURES_DIR, EXPERIMENT, "*", fname))):
        name = os.path.basename(os.path.dirname(d))
        m = re.match(r"\d{8}_\d{6}_(.+)", name)
        if m and (not tags or m.group(1) in tags):
            found[m.group(1)] = os.path.dirname(d)   # later scans replace earlier ones
    missing = set(tags or []) - set(found)
    if missing:
        raise SystemExit(f"no {fname} for {', '.join(sorted(missing))} in "
                         f"{os.path.join(store.CAPTURES_DIR, EXPERIMENT)}")
    return found


def load_cube(scan, bin_px):
    if not os.path.exists(os.path.join(scan, "cube.npz")):          # a snapshot: one line, (height, 1, bands)
        f = np.load(os.path.join(scan, "frames.npz"))["frames"].mean(0).astype(np.float32)
        return f[:, None, :], np.arange(f.shape[1], dtype=float), False
    d = np.load(os.path.join(scan, "cube.npz"))
    img = np.transpose(d["cube"], (1, 0, 2)).astype(np.float32)   # (height, az, bands)
    if bin_px > 1:
        h, a = img.shape[0] // bin_px * bin_px, img.shape[1] // bin_px * bin_px
        img = np.nanmean(img[:h, :a].reshape(h // bin_px, bin_px, a // bin_px, bin_px, -1), axis=(1, 3))
    return img, d["grid_nm"], bool(d["wavelength"])


def object_mask(img, min_signal, rows=None):
    """Pixels with a usable spectrum: bright enough, not a specular highlight."""
    total = np.nansum(img, axis=-1)
    ok = np.isfinite(img).all(-1)
    hi = np.nanpercentile(total[ok], 99)
    mask = ok & (total > min_signal * hi) & (total < hi)
    if rows is not None:
        keep = np.zeros(img.shape[0], bool)
        keep[rows[0]:rows[1]] = True
        mask &= keep[:, None]
    return mask


def pick_box(rgb, tag):
    print(f"{tag}: drag a box round the object, Enter to accept")
    x, y, w, h = cv2.selectROI(f"{tag}: select object", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()
    box = np.zeros(rgb.shape[:2], bool)
    box[y:y + h, x:x + w] = True
    return box


def white_per_row(img, mask, smooth=15):
    """White spectrum at each height (median over azimuth), smoothed along
    the slit; heights the white didn't cover get the overall median."""
    rows = np.full((img.shape[0], img.shape[-1]), np.nan, np.float32)
    for r in range(img.shape[0]):
        if mask[r].sum() >= min(5, img.shape[1]):   # a snapshot has one pixel per row
            rows[r] = np.median(img[r, mask[r]], axis=0)
    overall = np.nanmedian(rows, axis=0)
    rows[np.isnan(rows).any(1)] = overall
    k = np.ones(smooth) / smooth
    pad = np.pad(rows, ((smooth // 2, smooth // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, b], k, mode="valid") for b in range(rows.shape[1])], axis=1)


def false_colour(imgs, mask_list):
    """Same false-colour scale for every object, so their colours compare:
    the usable spectrum in thirds (short -> B, long -> R), one shared
    stretch from the pixels of all the objects."""
    thirds = np.array_split(np.arange(imgs[0].shape[-1]), 3)
    rgbs = [np.stack([np.nanmean(im[..., idx], -1) for idx in thirds[::-1]], -1) for im in imgs]
    hi = np.nanpercentile(np.concatenate([c[m] for c, m in zip(rgbs, mask_list)]), 99.5, axis=0)
    return [(np.clip(np.nan_to_num(c / hi), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8) for c in rgbs]


def sector_means(px, sec, min_px=20):
    return np.array([px[sec == s].mean(0) for s in range(N_SECTORS) if (sec == s).sum() >= min_px])


def shape_curves(px, sec):
    """Spectral shape (each sector mean ÷ its own mean, averaged) and the
    10-90% spread of the sectors around it: (mean, p10, p90)."""
    sm = sector_means(px, sec)
    sm = sm / sm.mean(1, keepdims=True)
    return (sm.mean(0), *np.percentile(sm, [10, 90], axis=0))


def angle_matrix(a, b):
    return np.array([[float(spectral.spectral_angle(x, y)) for y in b] for x in a])


def within_spread(means):
    """Median angle between different sectors of the same object."""
    m = angle_matrix(means, means)
    return float(np.median(m[np.triu_indices(len(m), 1)])) if len(m) > 1 else float("nan")


def leave_one_sector_out(pix, secs, proj=None):
    """Confusion matrix (true class x predicted) and per-pixel predictions."""
    k = len(pix)
    pix = [p if proj is None else p @ proj for p in pix]
    conf = np.zeros((k, k), int)
    preds = [np.full(len(p), -1) for p in pix]
    for s in range(N_SECTORS):
        train = [p[sc != s] for p, sc in zip(pix, secs)]
        if any(len(t) == 0 for t in train):
            continue
        means = np.array([t.mean(0) for t in train])
        for c, (p, sc) in enumerate(zip(pix, secs)):
            if (sc == s).any():
                lab, _ = nearest_class(p[sc == s], means)
                preds[c][sc == s] = lab
                np.add.at(conf[c], lab, 1)
    return conf, preds


def pairs_of(tags):
    """real_X / fake_X (either order of words) -> [(X, real_tag, fake_tag)]."""
    def key(t, word):
        return re.sub(rf"(^{word}_|_{word}$)", "", t) if re.search(rf"(^{word}_|_{word}$)", t) else None
    real = {key(t, "real"): t for t in tags if key(t, "real")}
    fake = {key(t, "fake"): t for t in tags if key(t, "fake")}
    return [(x, real[x], fake[x]) for x in real if x in fake]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="*", help="scan tags to compare (default: all; 'white' is the reference)")
    ap.add_argument("--rows", default=None, help="r0:r1, only use these heights (cube rows) for every object")
    ap.add_argument("--pick", action="store_true", help="draw a box round each object instead of the auto mask")
    ap.add_argument("--min-signal", type=float, default=0.15,
                    help="object pixels must be at least this fraction of the object's brightest")
    ap.add_argument("--bin", type=int, default=1, help="average NxN pixels first (less noise per pixel)")
    ap.add_argument("--no-white", action="store_true", help="ignore the white scan even if there is one")
    ap.add_argument("--snapshots", action="store_true", help="compare single snapshots instead of scans")
    args = ap.parse_args()

    fname = "frames.npz" if args.snapshots else "cube.npz"
    scans = find_scans(fname=fname)
    white_scan = None if args.no_white else scans.pop("white", None)
    if white_scan is None and not args.no_white and not args.snapshots:
        # a white target looks the same all the way round, so a still
        # snapshot of it is as good a reference as a scan
        white_scan = find_scans(fname="frames.npz").get("white")
    if args.tags:
        scans = find_scans(args.tags, fname)
    tags = list(scans)
    if len(tags) < 2:
        raise SystemExit(f"need at least two object {'snapshots' if args.snapshots else 'scans'} in captures/{EXPERIMENT}/, found {tags}")
    rows = tuple(int(v) for v in args.rows.split(":")) if args.rows else None

    loaded = [load_cube(scans[t], args.bin) for t in tags]
    imgs = [im for im, _, _ in loaded]
    grid, has_nm = loaded[0][1], loaded[0][2]
    if any(im.shape[0] != imgs[0].shape[0] or im.shape[-1] != imgs[0].shape[-1] for im in imgs):
        raise SystemExit("the scans have different heights or spectral sizes; were the settings changed?")
    masks = [object_mask(im, args.min_signal, rows) for im in imgs]

    # Usable bands: where the lamp actually reaches. The white shows that
    # directly; without it, the brightest object at each band stands in.
    if white_scan:
        wimg = load_cube(white_scan, args.bin)[0]
        wmask = object_mask(wimg, args.min_signal, rows)
        level = np.median(wimg[wmask], axis=0)
    else:
        level = np.max([np.median(im[m], axis=0) / np.median(im[m]) for im, m in zip(imgs, masks)], axis=0)
    good = level >= 0.2 * level.max()
    grid = grid[good]
    imgs = [im[..., good] for im in imgs]
    if white_scan:
        white = white_per_row(wimg[..., good], wmask)
        imgs = [im / np.maximum(white[:, None, :], 1e-6) for im in imgs]   # relative reflectance
    units = "nm" if has_nm else "columns"
    print(f"{len(tags)} objects: {', '.join(tags)}; {good.sum()} usable bands "
          f"({grid[0]:.0f}-{grid[-1]:.0f} {units}); "
          + ("relative reflectance (÷ white scan)" if white_scan else "dark-subtracted counts (no white scan)"))

    rgbs = false_colour(imgs, masks)
    if args.pick:
        masks = [m & pick_box(c, t) for m, c, t in zip(masks, rgbs, tags)]
    pix = [im[m] for im, m in zip(imgs, masks)]
    secs = []
    for im, m in zip(imgs, masks):
        if args.snapshots:   # one line per object: sectors are blocks of height along the slit
            secs.append(np.broadcast_to(np.arange(im.shape[0])[:, None], m.shape)[m] * N_SECTORS // im.shape[0])
        else:
            az = np.broadcast_to(np.arange(im.shape[1]), m.shape)[m]
            secs.append(az * N_SECTORS // im.shape[1])
    for t, p in zip(tags, pix):
        print(f"  {t:<14} {len(p):6d} pixels")

    proj = three_band(False, grid, len(grid))
    res = {"scans": {t: scans[t] for t in tags}, "white": white_scan, "bands": int(good.sum()),
           "relative_reflectance": bool(white_scan), "pixels": {t: len(p) for t, p in zip(tags, pix)},
           "grid": grid, "spectra": {t: dict(zip(("mean", "p10", "p90"), shape_curves(p, s)))
                                     for t, p, s in zip(tags, pix, secs)}}
    for name, pr in (("full", None), ("rgb_like", proj)):
        sm = [sector_means(p if pr is None else p @ pr, s) for p, s in zip(pix, secs)]
        means = np.array([(p if pr is None else p @ pr).mean(0) for p in pix])
        conf, preds = leave_one_sector_out(pix, secs, pr)
        res[name] = {"angle_deg": angle_matrix(means, means),
                     "within_object_deg": {t: within_spread(m) for t, m in zip(tags, sm)},
                     "confusion": conf, "accuracy": float(np.trace(conf) / conf.sum())}
        if pr is None:
            full_preds = preds
    pairs = pairs_of(tags)
    res["pairs"] = {}
    for x, rt, ft in pairs:
        i, j = tags.index(rt), tags.index(ft)
        res["pairs"][x] = {}
        for name in ("full", "rgb_like"):
            r = res[name]
            c = np.array(r["confusion"])[[i, j]][:, [i, j]]
            within = max(r["within_object_deg"][rt], r["within_object_deg"][ft])
            res["pairs"][x][name] = {"angle_deg": r["angle_deg"][i, j], "within_deg": within,
                                     "separation_ratio": r["angle_deg"][i, j] / within,
                                     "pair_accuracy": float(np.trace(c) / max(c.sum(), 1))}
    store_dir = os.path.join(store.CAPTURES_DIR, EXPERIMENT, "analysis_snapshot" if args.snapshots else "analysis")
    store.save_json(os.path.join(store_dir, "results.json"), res)

    print(f"\nleave-one-sector-out accuracy, all {len(tags)} objects: full spectrum "
          f"{100 * res['full']['accuracy']:.1f}%, RGB-like 3 bands {100 * res['rgb_like']['accuracy']:.1f}%")
    print("spectral angle between objects (deg, full spectrum):")
    print(" " * 14 + "".join(f"{t[:11]:>12}" for t in tags))
    for t, row in zip(tags, res["full"]["angle_deg"]):
        print(f"{t[:13]:<14}" + "".join(f"{v:12.2f}" for v in row))
    print(f"within-object spread (median angle between {'height blocks' if args.snapshots else 'sectors'}, deg): "
          + ", ".join(f"{t} {v:.2f}" for t, v in res["full"]["within_object_deg"].items()))
    for x, p in res["pairs"].items():
        f, g = p["full"], p["rgb_like"]
        print(f"\n{x}: real vs fake {f['angle_deg']:.2f} deg vs {f['within_deg']:.2f} within an object "
              f"({f['separation_ratio']:.1f}x); real/fake pixels told apart {100 * f['pair_accuracy']:.1f}%")
        print(f"{' ' * len(x)}  RGB-like:    {g['angle_deg']:.2f} deg vs {g['within_deg']:.2f} "
              f"({g['separation_ratio']:.1f}x); {100 * g['pair_accuracy']:.1f}%")

    xlabel = "wavelength (nm)" if has_nm else "spectral column (blue → red, uncalibrated)"
    if not args.snapshots:
        plot_panoramas(tags, rgbs, masks, full_preds, store_dir)
    plot_spectra(tags, pix, secs, pairs, grid, xlabel, bool(white_scan), store_dir)
    plot_separation(tags, res, store_dir)


def class_colours(k):
    return [tuple(int(h[j:j + 2], 16) / 255 for j in (1, 3, 5)) for h in plotting.SERIES[:k]]


def plot_panoramas(tags, rgbs, masks, preds, out):
    cols = class_colours(len(tags))
    fig, axes = plt.subplots(len(tags), 2, figsize=(13, 2.2 * len(tags)), squeeze=False)
    for i, (t, c, m, p) in enumerate(zip(tags, rgbs, masks, preds)):
        axes[i, 0].imshow(c, aspect="auto", extent=[0, 360, c.shape[0], 0])
        axes[i, 0].contour(np.linspace(0, 360, m.shape[1]), np.arange(m.shape[0]), m.astype(float),
                           levels=[0.5], colors="white", linewidths=0.6)
        cmap = np.full(m.shape + (3,), 0.93)
        cmap[m] = np.array(cols)[p]
        axes[i, 1].imshow(cmap, aspect="auto", extent=[0, 360, c.shape[0], 0], interpolation="nearest")
        axes[i, 0].set_ylabel(t, color=cols[i], fontweight="bold")
        for ax in axes[i]:
            ax.grid(False)
            ax.set_yticks([])
    axes[0, 0].set_title("False colour, common scale (outline = pixels used)")
    axes[0, 1].set_title("Classified as (colour = object, held-out sectors)")
    for ax in axes[-1]:
        ax.set_xlabel("azimuth (deg)")
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "panoramas.png"))


def plot_spectra(tags, pix, secs, pairs, grid, xlabel, reflectance, out):
    """Per fruit: real and fake spectra (band = spread over the 12 sectors),
    and fake ÷ real, each scaled to its own mean so it shows shape only."""
    groups = [(x, [rt, ft]) for x, rt, ft in pairs] or [("all objects", tags)]
    cols = class_colours(len(tags))
    fig, axes = plt.subplots(2, len(groups), figsize=(6.5 * len(groups), 7), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    for g, (x, members) in enumerate(groups):
        shapes = {}
        for t in members:
            i = tags.index(t)
            shapes[t], lo, hi = shape_curves(pix[i], secs[i])
            axes[0, g].fill_between(grid, lo, hi, color=cols[i], alpha=0.2, lw=0)
            axes[0, g].plot(grid, shapes[t], color=cols[i], label=t)
        axes[0, g].set(title=x, ylabel=("relative reflectance" if reflectance else "counts") + " (÷ own mean)")
        axes[0, g].legend()
        if len(members) == 2:
            axes[1, g].plot(grid, shapes[members[1]] / shapes[members[0]], color=plotting.INK, lw=1.5)
            axes[1, g].axhline(1, color=plotting.MUTED, lw=1)
            axes[1, g].set(ylabel=f"{members[1]} ÷ {members[0]}")
        axes[1, g].set_xlabel(xlabel)
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "spectra.png"))


def plot_separation(tags, res, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    vmax = max(np.max(res["full"]["angle_deg"]), np.max(res["rgb_like"]["angle_deg"]))
    for ax, name, title in ((axes[0], "full", "Full spectrum"), (axes[1], "rgb_like", "RGB-like (3 bands)")):
        m = np.array(res[name]["angle_deg"])
        np.fill_diagonal(m, [res[name]["within_object_deg"][t] for t in tags])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(len(tags)):
            for j in range(len(tags)):
                ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=9,
                        color="white" if m[i, j] > 0.6 * vmax else plotting.INK)
        ax.set_xticks(range(len(tags)), tags, rotation=30, ha="right")
        ax.set_yticks(range(len(tags)), tags)
        ax.set_title(f"{title}: angle (deg)\ndiagonal = within-object spread")
        ax.grid(False)
    ax = axes[2]
    names = list(res["pairs"]) or ["all objects"]
    x = np.arange(len(names))
    for k, (name, label) in enumerate((("full", "full spectrum"), ("rgb_like", "RGB-like"))):
        v = [100 * res["pairs"][n][name]["pair_accuracy"] for n in res["pairs"]] or [100 * res[name]["accuracy"]]
        ax.bar(x + (k - 0.5) * 0.35, v, width=0.33, color=[plotting.SERIES[0], plotting.MUTED][k], label=label)
        for xi, vi in zip(x, v):
            ax.annotate(f"{vi:.0f}%", (xi + (k - 0.5) * 0.35, vi), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8)
    ax.axhline(50, color=plotting.MUTED, lw=1, ls="--")
    ax.set_xticks(x, [f"real vs fake\n{n}" for n in res["pairs"]] or names)
    ax.set(ylim=(0, 125), ylabel="held-out pixels correct (%)", title="Telling real from fake (dashed = chance)")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "separation.png"))


if __name__ == "__main__":
    main()
