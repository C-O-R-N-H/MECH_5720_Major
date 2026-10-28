"""
Reconstruct the 360° hyperspectral panorama
=============================================
Success criterion: "automatically combine line scans acquired over a
complete 360° rotation into a continuous hyperspectral panorama using the
known turntable angle."

scan.py runs this automatically at the end of every scan; run it again by
hand to redo a scan with different settings:  python reconstruct.py <scan>

Pipeline (hsi/pipeline.py + hsi/turntable.py):
  1. every line: subtract the dark frame, or with a white reference
     (cal_white_reference.py) reflectance = (S - D) / (W - D)
  2. every line: onto a uniform wavelength grid, if cal_wavelength.py has
     been run (otherwise stays in spectral columns)
  3. every line: angle from its timestamp, theta = 360 (t - t0) / period
       --period auto      measured from the scan if reliable, else nominal
       --period estimate | nominal   force one
       --period-s P       use a hand-timed period (stopwatch 10 turns / 10)
  4. resample all lines onto a uniform azimuth grid (all revolutions merged,
     linear interpolation, wraps at 360)

Output (in the scan folder):
  cube.npz       cube (azimuth, height, wavelength) float32, grid_nm, azimuth_deg ...
  panorama.png   true-colour render (sRGB from reflectance), upscaled for viewing
  panorama_native.png  the same at one pixel per cube sample
  bands.png      single-wavelength images
"""

import argparse
import os

import cv2
import numpy as np

from hsi import colour, pipeline, plotting, turntable
from hsi.plotting import plt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", help="scan folder from scan.py")
    ap.add_argument("--period", choices=["auto", "nominal", "estimate"], default="auto")
    ap.add_argument("--period-s", type=float, default=None,
                    help="use this turntable period (s), e.g. stopwatch time for 10 turns / 10")
    ap.add_argument("--direction", type=int, choices=[1, -1], default=1,
                    help="-1 if the panorama comes out mirrored")
    ap.add_argument("--n-az", type=int, default=None, help="azimuth samples (default: as many as the angular sampling supports)")
    ap.add_argument("--step-nm", type=float, default=5.0)
    ap.add_argument("--reference", default=None, help="override the scan's reference")
    ap.add_argument("--view-width", type=int, default=1800, help="width of the upscaled panorama.png")
    args = ap.parse_args()

    d = pipeline.load_calibrated_scan(args.scan, args.period, args.direction, args.step_nm, args.reference,
                                      period_s=args.period_s)
    lines, grid, theta = d["lines"], d["grid"], d["theta"]
    revs = (theta.max() - theta.min()) / 360
    if revs < 1.0:
        print(f"WARNING: scan covers only {revs:.2f} rev — the panorama will have a gap")
    # Default azimuth sampling: as fine as the data supports. With several
    # revolutions merged the lines interleave, so the largest angular gap
    # (not lines per rev) is the limit.
    phi = np.sort(np.mod(theta, 360))
    max_gap = np.max(np.diff(np.concatenate([phi, phi[:1] + 360])))
    n_az = args.n_az or int(360 / max_gap)

    cube, az, gap = turntable.stitch(lines, theta, n_az)
    print(f"period {d['period']:.4f} s ({d['period_method']}), {revs:.2f} revs, {len(theta)} lines")
    print(f"cube {cube.shape} (azimuth, height, bands); max angular gap {gap:.2f} deg "
          f"vs grid step {360 / n_az:.2f} deg")
    np.savez_compressed(os.path.join(args.scan, "cube.npz"), cube=cube, grid_nm=grid, azimuth_deg=az,
                        period_s=d["period"], period_method=d["period_method"], max_gap_deg=gap,
                        reflectance=d["calibrated"], wavelength=d["wavelength"])

    # height runs down the image, azimuth across: (height, azimuth, ...)
    img_cube = np.transpose(cube, (1, 0, 2))
    if d["calibrated"] and d["wavelength"]:
        rgb = colour.reflectance_to_srgb(img_cube, grid)
    else:
        rgb = colour.quick_colour(img_cube, grid)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(args.scan, "panorama_native.png"), bgr)  # one pixel per cube sample
    # Viewing copy: the cube is only a few hundred samples each way, so
    # scale it up to a screen-friendly size (same proportions as bands.png).
    big = cv2.resize(bgr, (args.view_width, int(args.view_width * 0.3)), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(args.scan, "panorama.png"), big)
    print(f"  figure: {os.path.join(args.scan, 'panorama.png')}")

    picks = np.linspace(0, len(grid) - 1, 6).astype(int)
    fig, axes = plt.subplots(len(picks) + 1, 1, figsize=(12, 2.0 * (len(picks) + 1)))
    axes[0].imshow(rgb, aspect="auto", extent=[0, 360, img_cube.shape[0], 0])
    axes[0].set_title("True-colour render" if d["calibrated"] else "False colour (uncalibrated)")
    for ax, b in zip(axes[1:], picks):
        ax.imshow(img_cube[..., b], aspect="auto", cmap="gray", extent=[0, 360, img_cube.shape[0], 0],
                  vmin=np.nanpercentile(img_cube[..., b], 0.5), vmax=np.nanpercentile(img_cube[..., b], 99.5))
        unit = "nm" if d["wavelength"] else "col"
        ax.set_title(f"{grid[b]:.0f} {unit}", loc="left")
    for ax in axes:
        ax.grid(False)
        ax.set_yticks([])
    axes[-1].set_xlabel("azimuth (deg)")
    fig.tight_layout()
    plotting.save(fig, os.path.join(args.scan, "bands.png"))


if __name__ == "__main__":
    main()
