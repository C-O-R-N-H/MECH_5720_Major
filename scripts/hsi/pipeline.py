"""
The processing chain shared by scan.py (live preview), reconstruct.py
(panorama) and the exp_*.py analyses:

    raw ROI frame --(dark/white reference)--> reflectance
                  --(wavelength model)------> uniform wavelength grid
                  --(turntable angle)-------> panorama cube

References (stage-2 calibration, cal_white_reference.py) are saved as
    calibration/references/<stamp>/reference.npz   dark, white, exposure ...
and calibration/references/latest points at the newest one.
"""

import datetime
import os

import numpy as np

from . import spectral, store, turntable

REF_DIR = os.path.join(store.CALIB_DIR, "references")
LATEST_REF = os.path.join(REF_DIR, "latest")


def save_reference(dark, white, info):
    path = os.path.join(REF_DIR, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(path, exist_ok=True)
    np.savez_compressed(os.path.join(path, "reference.npz"), dark=dark, white=white,
                        **{k: np.asarray(v) for k, v in info.items()})
    with open(LATEST_REF, "w") as f:
        f.write(path)
    return path


def load_reference(path=None):
    """Returns dict(dark, white, exposure_us, ...) or None if none exists."""
    if path is None:
        if not os.path.exists(LATEST_REF):
            return None
        with open(LATEST_REF) as f:
            path = f.read().strip()
    if os.path.isdir(path):
        path = os.path.join(path, "reference.npz")
    d = dict(np.load(path, allow_pickle=False))
    d = {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}
    d["path"] = path
    return d


def load_dark(roi, row_bin=1, exposure_us=None, gain=None, shape=None):
    """The quick dark frame from take_dark.py, cropped/binned like a scan
    frame, or None if there isn't one (or it doesn't match `shape`, e.g.
    after the camera resolution changed)."""
    path = store.calib_path("dark.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    if exposure_us and float(d["exposure_us"]) and abs(float(d["exposure_us"]) - exposure_us) > 1:
        print(f"WARNING: dark frame was taken at {float(d['exposure_us']):g} us, scan at {exposure_us:g} us "
              "— re-run take_dark.py with the scan settings")
    if gain and abs(float(d["gain"]) - gain) > 1e-3:
        print(f"WARNING: dark frame gain {float(d['gain']):g} != scan gain {gain:g}")
    if roi is not None and (roi[1] > d["dark"].shape[0] or roi[3] > d["dark"].shape[1]):
        print(f"WARNING: dark frame is {d['dark'].shape}, smaller than this scan's frames "
              "(resolution changed?) — re-run take_dark.py; not using it")
        return None
    dark = spectral.bin_rows(spectral.crop(d["dark"], roi), row_bin)
    if shape is not None and dark.shape != tuple(shape):
        print(f"WARNING: dark frame {dark.shape} doesn't match the scan {tuple(shape)} — "
              "re-run take_dark.py; not using it")
        return None
    return dark


class LineCalibrator:
    """Turns one ROI frame (rows along slit x columns) into a calibrated
    line scan (rows x wavelength bands). Without a reference it falls back
    to dark-free counts, without a wavelength model to raw columns."""

    def __init__(self, roi_shape, row_bin=1, reference=None, exposure_us=None, step_nm=5.0,
                 lo_nm=None, hi_nm=None, dark=None):
        self.reference = reference
        # dark: an already cropped/binned dark frame (load_dark), used when
        # there's no full reference — subtracts black level + stray light.
        self.dark, self.white = (dark if reference is None else None), None
        if reference is not None:
            self.dark = spectral.bin_rows(reference["dark"], row_bin)
            self.white = spectral.bin_rows(reference["white"], row_bin)
            ref_exp = reference.get("exposure_us")
            if exposure_us and ref_exp and abs(exposure_us - ref_exp) > 1:
                # Scale the white's signal (not its dark) to the scan exposure.
                k = exposure_us / ref_exp
                print(f"note: reference exposure {ref_exp} us != scan {exposure_us} us, scaling white by {k:.3f}")
                self.white = self.dark + (self.white - self.dark) * k
            if self.dark.shape != tuple(roi_shape):
                raise ValueError(f"reference shape {self.dark.shape} != scan ROI {tuple(roi_shape)}: "
                                 "frame size changed since the reference was taken — retake it (cal_white_reference.py)")
        try:
            self.resampler, self.model = spectral.load_resampler(roi_shape, step_nm, lo_nm, hi_nm, row_bin)
            self.grid = self.resampler.grid
        except SystemExit:
            print("note: no wavelength calibration yet — output stays in column units")
            self.resampler, self.model = None, None
            self.grid = np.arange(roi_shape[1], dtype=np.float64)

    @property
    def calibrated(self):
        return self.reference is not None and self.resampler is not None

    def __call__(self, roi_frames):
        f = np.asarray(roi_frames, np.float32)
        if self.white is not None:
            f = spectral.reflectance(f, self.dark, self.white)
        elif self.dark is not None:
            f = f - self.dark
        if self.resampler is not None:
            f = self.resampler(f)
        return f


def calibrated_lines(frames, calib, chunk=32):
    """Apply a LineCalibrator to every frame of a (memmapped) scan."""
    out = None
    for s in range(0, len(frames), chunk):
        part = calib(frames[s: s + chunk])
        if out is None:
            out = np.empty((len(frames), *part.shape[1:]), np.float32)
        out[s: s + len(part)] = part
    return out


def period_for(times, lines, info, source="auto", feats=None):
    """Turntable period by the requested method. Returns (period, method).

    "estimate" finds the period from the scan itself (1.5-40 s search, then
    fitted to every turn), so it doesn't need to know which speed setting
    was used — the labels on the turntable turned out not to match reality.
    "nominal" uses the speed recorded at scan time (--speed). "auto" uses the
    estimate when it is reliable, else the nominal speed."""
    nominal = info.get("nominal_period_s")
    nominal = float(nominal) if nominal else None
    if source == "nominal":
        if nominal is None:
            raise SystemExit("scan has no nominal speed recorded; use --period estimate or --period-s")
        return nominal, "nominal"
    if feats is None:
        feats = turntable.frame_features(lines)
    try:
        p, score = turntable.find_period(times, feats)
        p = turntable.refine_period(times, feats, p)  # sharper, if >= 2 turns
    except (ValueError, IndexError):
        p, score = None, np.inf
    reliable = score < 0.6
    if source == "estimate" or reliable:
        if not reliable:
            print(f"WARNING: period estimate {p} s is unreliable (PDM {score:.2f}) — time 10 turns "
                  "with a stopwatch and pass --period-s")
        return p, "estimate"
    if nominal:
        print(f"WARNING: couldn't estimate the period from this scan (PDM {score:.2f}); using the nominal speed")
        return nominal, "nominal"
    raise SystemExit("no usable period — time 10 turns with a stopwatch and pass --period-s")


def load_calibrated_scan(scan_dir, period_source="auto", direction=1, step_nm=5.0, reference=None,
                         period_s=None):
    """Load a scan.py scan and return everything the panorama scripts need:
    dict(lines (n, rows, bands), grid, times, theta (deg, unwrapped),
    period, period_method, info, calibrated)."""
    from . import sweep

    frames, times, info = sweep.load_scan(scan_dir)
    ref = load_reference(reference or info.get("reference"))
    dark = None
    if ref is None:
        dark = load_dark(info.get("roi"), info.get("row_bin", 1), info.get("exposure_us") or None,
                         info.get("gain"), shape=frames.shape[1:])
        if dark is not None:
            print("subtracting dark frame (calibration/dark.npz)")
    calib = LineCalibrator(frames.shape[1:], row_bin=info.get("row_bin", 1), reference=ref,
                           exposure_us=info.get("exposure_us") or None, step_nm=step_nm, dark=dark)
    lines = calibrated_lines(frames, calib)
    angles_path = os.path.join(scan_dir, "angles.npy")
    if os.path.exists(angles_path):
        theta = np.load(angles_path)
        period, how = float("nan"), "manual"
    else:
        if period_s:  # measured by hand, e.g. stopwatch over 10 turns
            period, how = float(period_s), "given"
        else:
            period, how = period_for(times, lines, info, period_source)
        theta = turntable.angles(times, period, direction=direction)
    return {"lines": lines, "grid": calib.grid, "times": times, "theta": theta, "period": period,
            "period_method": how, "info": info, "calibrated": calib.calibrated,
            "wavelength": calib.resampler is not None}
