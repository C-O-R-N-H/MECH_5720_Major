"""
Pi camera server client for the experiments (headless, no GL — the viewer's
own raw_convert.py imports OpenGL, so it isn't reused here).

Both capture paths return the same thing: a float32 "superpixel" frame where
each output pixel is the SUM of one 2x2 Bayer cell (R + G + G + B). The
spectrometer puts one wavelength on each column, so demosaicing would only
smear neighbouring wavelengths together; summing the cell instead gives a
single broadband value per (row, wavelength) that is independent of which
CFA pattern the sensor has. The cost is half resolution in both axes.

No black level is subtracted here. Every experiment captures its own dark
frames and subtracts those, which removes black level and fixed-pattern
offset in one step and works identically for the raw and DNG paths.
"""

import argparse
import io
import re
import time

import numpy as np
import requests

DEFAULT_URL = "http://localhost:8089"


class Camera:
    def __init__(self, base_url=DEFAULT_URL, transpose=False, flip_spectrum=False):
        self.base_url = base_url.rstrip("/")
        self.transpose = transpose
        self.flip_spectrum = flip_spectrum
        self.session = requests.Session()
        self._exposure_us = 100000  # unknown at start: assume a long-ish one

    # ---- server config / controls -------------------------------------
    def config(self):
        r = self.session.get(f"{self.base_url}/config", timeout=5)
        r.raise_for_status()
        return r.json()

    def controls(self):
        r = self.session.get(f"{self.base_url}/controls", timeout=5)
        r.raise_for_status()
        return r.json()

    def set_controls(self, **values):
        r = self.session.post(f"{self.base_url}/controls", json=values, timeout=5)
        r.raise_for_status()

    def ensure_resolution(self, width, height):
        """Switch the server to width x height if it isn't already (it
        resets to its own default whenever the Pi server restarts)."""
        cfg = self.config()
        if (cfg.get("width"), cfg.get("height")) != (width, height):
            print(f"camera was {cfg.get('width')}x{cfg.get('height')}; setting {width}x{height}")
            r = self.session.post(f"{self.base_url}/config",
                                  json={"width": width, "height": height, "format": cfg.get("format", "RGB888")},
                                  timeout=20)
            r.raise_for_status()
            time.sleep(2)

    def lock_exposure(self, exposure_us, gain=1.0):
        """Manual exposure. Auto-exposure would silently rescale every frame,
        which breaks dark/white referencing — always lock it for science."""
        # The frame duration caps the exposure (auto mode stops at ~33 ms),
        # so stretch it to fit.
        self.set_controls(AeEnable=False, ExposureTime=int(exposure_us), AnalogueGain=float(gain),
                          FrameDurationLimits=[int(exposure_us), int(exposure_us) + 10000])
        # Frames already in the camera's pipeline still carry the OLD settings
        # (measured: the first frame ~0.6 s after the change is stale), so
        # wait several frame times before anything is captured.
        # Wait on the LONGER of the old and new exposure: after e.g. 400 ms,
        # the queued 400 ms frames take seconds to clear even if the new
        # setting is short (measured: 1.5 s was not enough after 400 ms).
        slowest = max(exposure_us, getattr(self, "_exposure_us", 0))
        time.sleep(max(1.5, 8 * slowest * 1e-6))
        self._exposure_us = exposure_us

    # ---- captures ------------------------------------------------------
    def grab_raw(self):
        """One /capture_raw frame -> (superpixel frame, meta). Fast path."""
        t0 = time.time()
        r = self.session.get(f"{self.base_url}/capture_raw", timeout=10)
        t1 = time.time()
        r.raise_for_status()
        bayer = decode_raw(r.content, r.headers)
        meta = {
            "t_request": t0,
            "t_response": t1,
            # Midpoint of the HTTP round trip. If the server ever sends the
            # sensor's own timestamp (see README "Server changes"), prefer it.
            "t": _header_timestamp(r.headers, default=(t0 + t1) / 2),
            "format": r.headers.get("X-Format"),
        }
        return self._orient(superpixel(bayer)), meta

    def grab_rgb(self):
        """One /capture_raw frame -> ((H/2, W/2, 3) R, G, B, meta): the
        sensor's own colour filters, one value each per 2x2 cell (G is the
        mean of the two greens). For LOOKING at frames — the analysis uses
        the summed superpixel, which equals R + 2G + B."""
        r = self.session.get(f"{self.base_url}/capture_raw", timeout=10)
        r.raise_for_status()
        fmt = r.headers.get("X-Format", "")
        rgb = bayer_rgb(decode_raw(r.content, r.headers), fmt)
        return self._orient(rgb), {"format": fmt}

    def grab_dng(self):
        """One /capture_dng frame -> (superpixel frame, meta). Slow but the
        decode is done by rawpy from the file's own metadata, so it is the
        ground truth for checking the raw decode."""
        import rawpy

        t0 = time.time()
        r = self.session.get(f"{self.base_url}/capture_dng", timeout=20)
        t1 = time.time()
        r.raise_for_status()
        with rawpy.imread(io.BytesIO(r.content)) as raw:
            bayer = raw.raw_image_visible.astype(np.uint16).copy()
            meta = {
                "t_request": t0,
                "t_response": t1,
                "t": (t0 + t1) / 2,
                "black_level": list(map(float, raw.black_level_per_channel)),
                "white_level": float(raw.white_level),
                "cfa": raw.color_desc.decode(),
            }
        return self._orient(superpixel(bayer)), meta

    def grab(self, method="raw"):
        return self.grab_raw() if method == "raw" else self.grab_dng()

    def prompt(self, message, scene=None):
        """Ask the operator to change the setup. `scene` only matters for
        the simulator (sim.py), which switches scene instead of waiting."""
        input(f"\n>>> {message}\n    press Enter when ready... ")

    def average(self, n, method="raw"):
        """Mean and per-pixel std of n frames. Also returns the frames' metas."""
        acc = acc2 = None
        metas = []
        for _ in range(n):
            f, m = self.grab(method)
            f = f.astype(np.float64)
            acc = f if acc is None else acc + f
            acc2 = f * f if acc2 is None else acc2 + f * f
            metas.append(m)
        mean = acc / n
        std = np.sqrt(np.maximum(acc2 / n - mean * mean, 0.0))
        return mean.astype(np.float32), std.astype(np.float32), metas

    def _orient(self, frame):
        """Rows along the slit, columns along the spectrum, blue at column 0.
        Works on (h, w) frames and (h, w, 3) colour frames."""
        if self.transpose:
            frame = frame.swapaxes(0, 1)
        if self.flip_spectrum:
            frame = frame[:, ::-1]
        return np.ascontiguousarray(frame)


def _header_timestamp(headers, default):
    ts = headers.get("X-Sensor-Timestamp")
    return float(ts) * 1e-9 if ts else default


def average_rgb(cam, n):
    """Mean of n colour frames (h, w, 3) and the brightest raw value seen
    (10-bit, so >= ~970 means some pixels are saturating)."""
    acc, peak = None, 0.0
    for _ in range(n):
        f, _ = cam.grab_rgb()
        acc = f.astype(np.float64) if acc is None else acc + f
        peak = max(peak, float(f.max()))
    return (acc / n).astype(np.float32), peak


def bayer_rgb(bayer, fmt):
    """Split a Bayer mosaic into (H/2, W/2, 3) R, G, B planes. The filter
    layout comes from the raw format name, e.g. SGBRG10_CSI2P -> GBRG:
    top-left G, top-right B, bottom-left R, bottom-right G."""
    m = re.match(r"S([RGB]{4})", fmt)
    if not m:
        raise ValueError(f"can't tell the colour filter layout from raw format {fmt!r}")
    h, w = bayer.shape
    b = bayer[: h - h % 2, : w - w % 2].astype(np.float32)
    cells = {(0, 0): b[0::2, 0::2], (0, 1): b[0::2, 1::2], (1, 0): b[1::2, 0::2], (1, 1): b[1::2, 1::2]}
    planes = {"R": [], "G": [], "B": []}
    for (y, x), letter in zip(((0, 0), (0, 1), (1, 0), (1, 1)), m.group(1)):
        planes[letter].append(cells[(y, x)])
    return np.stack([np.mean(planes[c], axis=0) for c in "RGB"], axis=-1)


def superpixel(bayer):
    """Sum each 2x2 CFA cell -> (H/2, W/2) float32."""
    h, w = bayer.shape
    b = bayer[: h - h % 2, : w - w % 2].astype(np.float32)
    return b[0::2, 0::2] + b[0::2, 1::2] + b[1::2, 0::2] + b[1::2, 1::2]


def decode_raw(raw_bytes, headers):
    """Turn a /capture_raw response into a 2D uint16 Bayer mosaic.

    picamera2 hands back the raw stream as uint8 rows of `stride` bytes.
    Supported: unpacked (SRGGB10/12/16 etc. — little-endian uint16) and
    MIPI CSI-2 packed (*_CSI2P, 10 or 12 bit). The Pi 5's compressed
    PISP_COMP formats can't be decoded client-side; switch the server's raw
    format to an unpacked one or use the DNG path.
    """
    width = int(headers["X-Width"])
    height = int(headers["X-Height"])
    stride = int(headers["X-Stride"])
    fmt = headers["X-Format"]
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    rows = buf[: stride * height].reshape(height, stride)

    if "PISP" in fmt or "COMP" in fmt:
        raise ValueError(f"raw format {fmt} is compressed; use --method dng or an unpacked raw format")

    bits = int(re.search(r"(\d+)", fmt).group(1)) if re.search(r"\d+", fmt) else 8

    if fmt.endswith("_CSI2P"):
        if bits == 10:
            p = rows[:, : width * 5 // 4].reshape(height, width // 4, 5).astype(np.uint16)
            out = np.empty((height, width // 4, 4), np.uint16)
            for i in range(4):
                out[..., i] = (p[..., i] << 2) | ((p[..., 4] >> (2 * i)) & 0x3)
            return out.reshape(height, width)
        if bits == 12:
            p = rows[:, : width * 3 // 2].reshape(height, width // 2, 3).astype(np.uint16)
            out = np.empty((height, width // 2, 2), np.uint16)
            out[..., 0] = (p[..., 0] << 4) | (p[..., 2] & 0xF)
            out[..., 1] = (p[..., 1] << 4) | (p[..., 2] >> 4)
            return out.reshape(height, width)
        raise ValueError(f"unsupported packed raw format {fmt}")

    if bits > 8:
        return rows[:, : width * 2].copy().view("<u2").reshape(height, width)
    return rows[:, :width].astype(np.uint16)


# Settings used for every capture on this rig (chosen by exp_exposure_sweep.py:
# 100 ms balances signal-to-noise against lines per turn; gain barely
# affects SNR, 8x just keeps counts well above the 10-bit quantisation).
# The dark frame must be taken at the same settings (take_dark.py).
DEFAULT_EXPOSURE_US = 100000
DEFAULT_GAIN = 8.0
# Camera resolution every capture uses (the server forgets it on restart).
# 1920x1080 -> 960 x 540 superpixels; the capture rate is exposure-limited,
# so the extra pixels cost nothing in speed.
DEFAULT_RESOLUTION = (1920, 1080)
# Orientation (measured from the raw colour pixels after the camera was
# rotated on 26 Oct): the spectrum already runs along the sensor's columns,
# so no transpose, but the RED end is at column 0, so the columns are
# reversed to put BLUE at column 0. Frames are 540 rows along the slit x 960
# spectral columns. (The server's video feed swaps red and blue —
# picamera2's "RGB888" arrays are stored B, G, R — so don't judge the
# direction from the feed.) Re-check if the camera is rotated again.
DEFAULT_TRANSPOSE = False
DEFAULT_FLIP_SPECTRUM = True


def add_camera_args(parser):
    parser.add_argument("--url", default=DEFAULT_URL, help="Pi camera server base URL, or sim:// for the simulator")
    parser.add_argument("--method", choices=["raw", "dng"], default="raw",
                        help="capture path: raw (fast) or dng (slow, rawpy-decoded)")
    parser.add_argument("--exposure-us", type=float, default=DEFAULT_EXPOSURE_US,
                        help=f"manual exposure in microseconds (default {DEFAULT_EXPOSURE_US})")
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN, help=f"analogue gain (default {DEFAULT_GAIN:g})")
    parser.add_argument("--transpose", action=argparse.BooleanOptionalAction, default=DEFAULT_TRANSPOSE,
                        help="swap rows/columns so wavelength runs along the columns (off for this rig since 26 Oct)")
    parser.add_argument("--flip-spectrum", action=argparse.BooleanOptionalAction, default=DEFAULT_FLIP_SPECTRUM,
                        help="reverse the spectral axis so blue is at column 0 (default on for this rig)")


def open_camera(args):
    """Camera (or simulator) with exposure locked to args' settings."""
    if args.url.startswith("sim:"):
        from .sim import SimCamera

        cam = SimCamera()
    else:
        cam = Camera(args.url, transpose=args.transpose, flip_spectrum=args.flip_spectrum)
        cam.ensure_resolution(*DEFAULT_RESOLUTION)
    cam.lock_exposure(args.exposure_us, args.gain)
    return cam
