"""
Continuous capture for turntable scans (scan.py).

The turntable free-runs at a constant speed, so nothing is triggered: we
grab frames as fast as the server allows, timestamp every one, and work out
the angle afterwards from the timestamps (turntable.py). Frames are cropped
to the ROI and written straight into a memory-mapped .npy so a long scan
never has to fit in RAM.

Scan folder layout (what reconstruct.py and the exp_*.py analyses read back):
    frames.npy   float32 (n_frames, rows, cols)   ROI-cropped superpixel frames
    times.npy    float64 (n_frames,)              capture time, seconds (epoch)
    scan.json    settings: roi, row_bin, method, exposure, nominal period ...
"""

import json
import os
import queue
import threading
import time

import numpy as np

from . import spectral


class Sweep:
    """Capture loop in a background thread. The main thread is free to draw
    a live preview from `latest` (a queue of (index, frame, t))."""

    def __init__(self, cam, out_dir, duration_s, roi=None, row_bin=1, method="raw",
                 info=None, max_rate_hz=None):
        self.cam = cam
        self.out_dir = out_dir
        self.duration_s = duration_s
        self.roi = roi
        self.row_bin = row_bin
        self.method = method
        self.info = dict(info or {})
        self.min_dt = 1.0 / max_rate_hz if max_rate_hz else 0.0
        self.latest = queue.Queue(maxsize=4)
        self.n = 0
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def _prep(self, frame):
        return spectral.bin_rows(spectral.crop(frame, self.roi), self.row_bin)

    def start(self):
        first, meta = self.cam.grab(self.method)
        first = self._prep(first)
        # Allow for the rate fluctuating upward of what the first frame suggests.
        dt = max(meta["t_response"] - meta["t_request"], 0.02)
        cap = int(self.duration_s / max(dt, self.min_dt) * 2) + 50
        self.frames = np.lib.format.open_memmap(
            os.path.join(self.out_dir, "frames.npy"), mode="w+", dtype=np.float32,
            shape=(cap, *first.shape))
        self.times = np.zeros(cap)
        self._store(first, meta["t"])
        self.t0 = meta["t"]
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _store(self, frame, t):
        if self.n >= len(self.times):
            self._stop.set()
            return
        self.frames[self.n] = frame
        self.times[self.n] = t
        try:
            self.latest.put_nowait((self.n, frame, t))
        except queue.Full:
            pass  # preview is behind; it'll catch the next one
        self.n += 1

    def _run(self):
        try:
            last = time.time()
            while not self._stop.is_set() and time.time() - self.t0 < self.duration_s:
                wait = self.min_dt - (time.time() - last)
                if wait > 0:
                    time.sleep(wait)
                last = time.time()
                f, m = self.cam.grab(self.method)
                self._store(self._prep(f), m["t"])
        except Exception as e:  # surfaced to the main thread via .error
            self.error = e
        finally:
            self._stop.set()

    @property
    def running(self):
        return not self._stop.is_set()

    def wait(self, timeout):
        """Block until the capture ends or timeout s pass."""
        self._stop.wait(timeout)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def finish(self):
        """Wait for the capture to end, trim the memmap and write metadata."""
        self.stop()
        n = self.n
        self.frames.flush()
        frames = np.load(os.path.join(self.out_dir, "frames.npy"), mmap_mode="r")[:n]
        # Rewrite trimmed (np.save of a memmap slice streams it to disk).
        tmp = os.path.join(self.out_dir, "frames_trim.npy")
        np.save(tmp, frames)
        del frames, self.frames
        os.replace(tmp, os.path.join(self.out_dir, "frames.npy"))
        np.save(os.path.join(self.out_dir, "times.npy"), self.times[:n])
        dts = np.diff(self.times[:n])
        summary = {
            **self.info,
            "roi": self.roi,
            "row_bin": self.row_bin,
            "method": self.method,
            "n_frames": n,
            "duration_s": float(self.times[n - 1] - self.times[0]) if n > 1 else 0.0,
            "mean_rate_hz": float(1.0 / dts.mean()) if len(dts) else 0.0,
            "dt_std_s": float(dts.std()) if len(dts) else 0.0,
        }
        with open(os.path.join(self.out_dir, "scan.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if self.error:
            print(f"capture stopped early: {self.error!r}")
        return summary


def load_scan(scan_dir):
    frames = np.load(os.path.join(scan_dir, "frames.npy"), mmap_mode="r")
    times = np.load(os.path.join(scan_dir, "times.npy"))
    with open(os.path.join(scan_dir, "scan.json")) as f:
        info = json.load(f)
    return frames, times, info
