"""
Turntable scan (capture app with live panorama)
=================================================
Start the turntable, run this, and it grabs line scans as fast as the
server allows (~5.4 lines/s at 100 ms), drawing the panorama live as it
builds up, then reconstructs and shows the final panorama.

    python scan.py                                   # 70 s into captures/scans/
    python scan.py --experiment speed_comparison --tag setting1

You don't need to tell it the turntable speed: reconstruct.py measures the
real period from the scan (the speed labels on the turntable are wrong).
The default 70 s is >= 2 full turns even at the slowest setting (28 s/turn);
two or more turns let exp_consistency.py compare turn 1 against turn 2,
and later turns sample different angles, which fills in the panorama.

Before scanning, same session and settings:
  * take_dark.py         dark frame (needed; retake if settings change)
  * snapshot.py          check the light covers the slit and isn't too dim
  * cal_white_reference.py  optional, gives reflectance instead of counts

Live preview: true colour if a white reference and wavelength calibration
exist, otherwise false colour. Its angles use a placeholder period, so it
may smear; the saved data and final panorama aren't affected. Press q / Esc
in the window to stop early.

Fallback (no motor): --manual-step-deg D captures one averaged line per
Enter press while you rotate the object by hand D degrees each time.
"""

import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

from hsi import camera, colour, pipeline, spectral, store, sweep, turntable


class LivePanorama:
    def __init__(self, calib, n_rows, n_az, period, title="hyperspectral scan"):
        self.calib = calib
        self.period = period
        self.n_az = n_az
        self.canvas = np.zeros((n_rows, n_az, 3), np.uint8)
        self.title = title
        self.t0 = None
        self.prev_col = None
        self.scale = max(1.0, 700 / max(n_rows, 1))
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def line_rgb(self, roi_frame):
        line = self.calib(roi_frame[None])[0]                    # (rows, bands)
        if self.calib.calibrated:
            return colour.reflectance_to_srgb(line, self.calib.grid)
        return colour.quick_colour(line[None], self.calib.grid)[0]

    def add(self, roi_frame, t):
        self.t0 = t if self.t0 is None else self.t0
        theta = turntable.angles(np.array([t]), self.period, self.t0)[0]
        col = int(np.mod(theta, 360) / 360 * self.n_az) % self.n_az
        rgb = self.line_rgb(roi_frame)
        # fill the columns since the previous line so the preview has no gaps
        n = (col - self.prev_col) % self.n_az if self.prev_col is not None else 1
        n = n if 0 < n < self.n_az // 8 else 1
        for k in range(n):
            self.canvas[:, (col - k) % self.n_az] = rgb
        self.prev_col = col
        return theta

    def show(self, status):
        img = cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imshow(self.title, img)
        return cv2.waitKey(1) & 0xFF


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    camera.add_camera_args(ap)
    ap.add_argument("--speed", type=float, default=None,
                    help="nominal seconds per revolution, if known (the real period is measured from the scan)")
    ap.add_argument("--revs", type=float, default=2.2, help="revolutions to capture when --speed is given")
    ap.add_argument("--seconds", type=float, default=70.0,
                    help="capture time when --speed isn't given (70 s = 2+ turns even at ~30 s/turn)")
    ap.add_argument("--row-bin", type=int, default=1,
                    help="combine this many rows along the slit at capture time (1 = keep all 540)")
    ap.add_argument("--reference", default=None, help="reference dir (default: latest)")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--manual-step-deg", type=float, default=0.0, help="hand-rotated fallback mode")
    ap.add_argument("--tag", default="", help="name for the run folder (object name)")
    ap.add_argument("--experiment", default="scans",
                    help="folder under captures/ to save into, e.g. speed_comparison")
    args = ap.parse_args()

    cam = camera.open_camera(args)
    if args.manual_step_deg:
        cam.prompt("Object in front of the camera at 0 deg, lights on.", "lamp-patches")
    else:
        cam.prompt("Object on the turntable, lights on, turntable running.",
                   f"turntable:{args.speed or 5:g}")
    # Whole frame: the spectrum fills the sensor. (An automatic crop from one
    # frame once cut off 60% of the slit when a dark patch was in view.)
    first = cam.grab(args.method)[0]
    roi = [0, first.shape[0], 0, first.shape[1]]
    try:
        spectral.find_roi(first)  # just checks that light gets in
    except SystemExit as e:
        print(f"WARNING: {e}")
    ref = pipeline.load_reference(args.reference)
    if ref is None:
        print("no white reference (cal_white_reference.py) — showing uncalibrated false colour"
              + ("" if os.path.exists(store.calib_path("dark.npz")) else "; no dark frame either (take_dark.py)"))
    out = store.new_run_dir(args.experiment, args.tag or (f"{args.speed:g}s" if args.speed else "scan"))
    roi_rows = (roi[1] - roi[0]) // args.row_bin
    roi_shape = (roi_rows, roi[3] - roi[2])
    dark = None if ref else pipeline.load_dark(roi, args.row_bin, args.exposure_us, args.gain,
                                                         shape=roi_shape)
    calib = pipeline.LineCalibrator(roi_shape, row_bin=args.row_bin, reference=ref, exposure_us=args.exposure_us,
                                    dark=dark)

    info = {"nominal_period_s": args.speed, "exposure_us": args.exposure_us or 0, "gain": args.gain,
            "reference": ref["path"] if ref else None, "tag": args.tag}

    if args.manual_step_deg:
        return manual_scan(cam, args, out, roi, calib, info)

    # The real period is only known after the scan (reconstruct.py measures
    # it); the live preview just needs a rough one.
    period = args.speed or 28.2
    duration = args.revs * period if args.speed else args.seconds
    preview = None if args.no_preview else LivePanorama(calib, roi_rows, 360, period)

    sw = sweep.Sweep(cam, out, duration, roi=roi, row_bin=args.row_bin,
                     method=args.method, info=info).start()
    last_shown = -1
    while sw.running:
        try:
            i, frame, t = sw.latest.get(timeout=0.2)
        except Exception:
            continue
        if preview is None:
            print(f"\r{sw.n} frames", end="")
            continue
        theta = preview.add(frame, t)
        if i != last_shown:
            last_shown = i
            key = preview.show(f"{sw.n} lines  {theta / 360:.2f} rev  {sw.n / max(t - sw.t0, 1e-3):.1f} Hz")
            if key in (ord("q"), 27):
                sw.stop()
    summary = sw.finish()
    if preview is not None:
        cv2.imwrite(os.path.join(out, "live_preview.png"), cv2.cvtColor(preview.canvas, cv2.COLOR_RGB2BGR))
        cv2.destroyAllWindows()
    print(f"\n{summary['n_frames']} lines in {summary['duration_s']:.1f} s ({summary['mean_rate_hz']:.2f} Hz)")
    print(f"saved {out}")
    reconstruct(out, show=not args.no_preview)


def manual_scan(cam, args, out, roi, calib, info):
    n_steps = int(round(360 / args.manual_step_deg))
    frames, angles, times = [], [], []
    for k in range(n_steps):
        cam.prompt(f"Step {k + 1}/{n_steps}: rotate the object to {k * args.manual_step_deg:.1f} deg.",
                   "lamp-patches")
        f, _, _ = cam.average(5, args.method)
        frames.append(spectral.bin_rows(spectral.crop(f, roi), args.row_bin))
        angles.append(k * args.manual_step_deg)
        times.append(time.time())
    np.save(os.path.join(out, "frames.npy"), np.array(frames, np.float32))
    np.save(os.path.join(out, "times.npy"), np.array(times))
    np.save(os.path.join(out, "angles.npy"), np.array(angles))
    store.save_json(os.path.join(out, "scan.json"), {**info, "roi": roi, "row_bin": args.row_bin,
                                                     "method": args.method, "n_frames": n_steps,
                                                     "angle_source": "manual"})
    print(f"saved {out}")
    reconstruct(out, show=not args.no_preview)


def reconstruct(out, show):
    """Run reconstruct.py on the scan just taken and show the panorama."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reconstruct.py")
    subprocess.run([sys.executable, script, out], check=True)
    if show:
        img = cv2.imread(os.path.join(out, "panorama.png"))
        cv2.namedWindow("panorama (any key to close)", cv2.WINDOW_NORMAL)
        cv2.imshow("panorama (any key to close)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
