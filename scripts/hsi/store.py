"""
Where things live on disk.

    captures/<experiment>/<timestamp>[_<tag>]/   one folder per run
    calibration/dark.npz                        dark frame at the scan settings (take_dark.py)
    calibration/wavelength.json                 pixel -> nm model           (cal_wavelength.py)
    calibration/references/<stamp>/             dark + white reference      (cal_white_reference.py)
    calibration/response.npz                    camera spectral response    (cal_white_reference.py --sun)

Calibration files are the hand-off between scripts: each reads whatever
has been written so far and falls back (uncalibrated) or stops with a
pointer to the right script if something it needs is missing.
"""

import datetime as _dt
import json
import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# HSI_DATA_ROOT redirects captures/ + calibration/ elsewhere, e.g. so a
# simulator dry run doesn't overwrite the real calibration.
DATA_ROOT = os.environ.get("HSI_DATA_ROOT", REPO_ROOT)
CAPTURES_DIR = os.path.join(DATA_ROOT, "captures")
CALIB_DIR = os.path.join(DATA_ROOT, "calibration")


def new_run_dir(experiment, tag=None):
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{tag}" if tag else stamp
    path = os.path.join(CAPTURES_DIR, experiment, name)
    os.makedirs(path, exist_ok=True)
    return path


def latest_run_dir(experiment):
    """Most recent run folder of an experiment, or None. Lets step-by-step
    experiments (one capture per command) add to the same run."""
    root = os.path.join(CAPTURES_DIR, experiment)
    runs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))) if os.path.isdir(root) else []
    return os.path.join(root, runs[-1]) if runs else None


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def calib_path(name):
    return os.path.join(CALIB_DIR, name)


def require_calib(name, made_by):
    path = calib_path(name)
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run {made_by} first.")
    return path


def load_wavelength():
    return load_json(require_calib("wavelength.json", "cal_wavelength.py"))
