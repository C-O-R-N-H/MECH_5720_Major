"""
Snapshot: what ONE frame from the spectrometer looks like
===========================================================
Grabs --n frames at the scan settings, subtracts the dark frame
(take_dark.py) if it matches, and saves to captures/snapshot/<stamp>/.
Use it to check the light before scanning: the brightest pixels should be
well above ~100 counts, across the whole slit height, and nowhere near
full scale (~3800).

    frame_<k>.png   the sensor image in colour (the camera's own red/green/
                    blue pixels, dark-subtracted, contrast-stretched):
                    up/down = position along the slit (object height),
                    left/right = spectrum (blue -> red)
    snapshot.png    that image plus the spectrum at a few heights
    frames.npz      the numbers (summed superpixel counts, as the analysis uses)

    python snapshot.py
"""

import argparse
import os

import cv2
import numpy as np

from hsi import camera, pipeline, plotting, store
from hsi.plotting import plt


def colour_view(rgb):
    """(h, w, 3) dark-subtracted R, G, B -> uint8 for display. Each channel is
    stretched to its own 99.5th percentile (a rough white balance, since the
    sensor is much less sensitive in blue), then gamma'd for the screen."""
    hi = np.percentile(rgb.reshape(-1, 3), 99.5, axis=0)
    v = np.clip(rgb / np.maximum(hi, 1e-9), 0, 1) ** (1 / 2.2)
    return (v * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    camera.add_camera_args(ap)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tag", default="", help="name added to the folder, e.g. white_card")
    args = ap.parse_args()

    cam = camera.open_camera(args)
    rgb = np.array([cam.grab_rgb()[0] for _ in range(args.n)])     # (n, h, w, 3)
    cam.set_controls(AeEnable=True)
    out = store.new_run_dir("snapshot", args.tag or None)

    h, w = rgb.shape[1:3]
    dark = pipeline.load_dark([0, h, 0, w], 1, args.exposure_us, args.gain, shape=(h, w))
    if dark is not None:
        rgb = rgb - dark[None, ..., None] / 4  # the dark is a 4-pixel sum; ~equal per pixel
    frames = rgb[..., 0] + 2 * rgb[..., 1] + rgb[..., 2]          # = the summed superpixel
    np.savez_compressed(os.path.join(out, "frames.npz"), frames=frames, dark_subtracted=dark is not None)
    for k, f in enumerate(rgb):
        cv2.imwrite(os.path.join(out, f"frame_{k}.png"), cv2.cvtColor(colour_view(f), cv2.COLOR_RGB2BGR))

    mean = frames.mean(axis=0)
    view = colour_view(rgb.mean(axis=0))
    rows = np.linspace(0, h, 6)[1:-1].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.3]})
    axes[0].imshow(view, aspect="auto")
    for i, r in enumerate(rows):
        axes[0].axhline(r, color=plotting.SERIES[i], lw=1.5)
    axes[0].set(xlabel="spectral column (blue → red)", ylabel="position along slit (px)",
                title=f"One frame (mean of {args.n})" + (", dark-subtracted" if dark is not None else ""))
    axes[0].grid(False)
    for i, r in enumerate(rows):
        band = mean[max(0, r - 4): r + 4].mean(axis=0)  # a few rows for less noise
        axes[1].plot(band, color=plotting.SERIES[i], lw=1.5, label=f"row {r}")
    axes[1].set(xlabel="spectral column (blue → red, uncalibrated)", ylabel="counts",
                title="Spectrum at the marked heights")
    axes[1].legend()
    fig.tight_layout()
    plotting.save(fig, os.path.join(out, "snapshot.png"))
    print(f"signal: median {np.median(mean):.0f}, brightest 0.5% {np.percentile(mean, 99.5):.0f} counts "
          f"(full scale ~{4 * 1023 - 4 * 64})")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
