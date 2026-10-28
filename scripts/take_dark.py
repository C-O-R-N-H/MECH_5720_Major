"""
Dark frame
===========
Cap the lens (or wrap it in black cloth, room lights off) and run this at
the same settings you scan with (the defaults). Saves calibration/dark.npz;
scans, snapshots and the experiments subtract it automatically. It removes
the sensor's black level and fixed pattern, without which every spectrum is
flattened toward the same shape (see exp_processing_comparisons.py #3).

Retake it whenever exposure, gain or resolution change. A good dark frame at
100 ms / 8x reads ~120 counts median with ~12 counts noise; noticeably
higher, or a spectrum visible in it, means light is leaking in.

    python take_dark.py
"""

import argparse
import datetime
import os

import numpy as np

from hsi import camera, store


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    camera.add_camera_args(ap)
    ap.add_argument("--n", type=int, default=20, help="frames to average")
    args = ap.parse_args()

    cam = camera.open_camera(args)
    cam.prompt("Cover the lens completely.", "dark")
    dark, std, _ = cam.average(args.n, args.method)
    cam.set_controls(AeEnable=True)  # hand the camera back in auto mode
    path = store.calib_path("dark.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, dark=dark, exposure_us=args.exposure_us or 0, gain=args.gain,
                        transpose=args.transpose, flip_spectrum=args.flip_spectrum, date=datetime.datetime.now().isoformat(timespec="seconds"))
    print(f"dark level: median {np.median(dark):.1f}, brightest 0.5% {np.percentile(dark, 99.5):.1f}, "
          f"noise {np.median(std):.1f} counts per frame")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
