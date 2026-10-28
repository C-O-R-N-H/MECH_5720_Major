"""
Save actions for the capture buttons. Never touches disk until one of these
fires — the captures directory is created lazily, on first save.

All three actions share the filename typed into the "Save name" textbox
(state.save_name_buf); each appends its own extension. An empty box falls
back to a `<kind>_<timestamp>` name, same as the original script's default.

Each function here is a blocking target meant to be run via
threading.Thread(target=..., daemon=True).start() from the UI, same
convention as server_api.py.
"""

import os

import cv2

import raw_convert
import server_api

# <repo>/captures/viewer/, resolved from this file so it doesn't depend on cwd.
CAPTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "captures", "viewer"
)


def _resolve_path(state, kind, ext):
    name = state.save_name_buf.strip()
    if not name:
        name = state.timestamp_name(kind)
    if not name.lower().endswith(ext):
        name += ext
    # A save name like "run1/frame" lands in a subfolder of CAPTURES_DIR.
    path = os.path.join(CAPTURES_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def capture_raw_thread(state):
    state.set_status("Downloading raw frame...", (0.9, 0.6, 0.2, 1.0))
    try:
        raw_bytes, headers = server_api.fetch_raw_bytes(state.base_url)
        arr, meta = raw_convert.decode_raw_response(raw_bytes, headers)
        path = _resolve_path(state, "raw", ".raw")
        # Plain raw binary dump (no header/metadata) so it can be
        # reinterpreted later using the printed shape/dtype/format.
        arr.tofile(path)
        print(f"Saved raw bytes to {path} (shape={arr.shape}, dtype={arr.dtype})")
        state.set_status(f"Saved {path}: {meta['width']}x{meta['height']} {meta['format']}", (0.2, 0.8, 0.2, 1.0))
    except Exception as e:
        state.set_status(f"Capture error: {e}", (0.9, 0.2, 0.2, 1.0))


def capture_dng_thread(state):
    state.set_status("Downloading DNG frame...", (0.9, 0.6, 0.2, 1.0))
    try:
        dng_bytes = server_api.fetch_dng_bytes(state.base_url)
        path = _resolve_path(state, "frame", ".dng")
        with open(path, "wb") as f:
            f.write(dng_bytes)
        print(f"Saved DNG to {path} ({len(dng_bytes)} bytes)")
        state.set_status(f"Saved {path}", (0.2, 0.8, 0.2, 1.0))
    except Exception as e:
        state.set_status(f"DNG capture error: {e}", (0.9, 0.2, 0.2, 1.0))


def save_current_thread(state):
    frame_to_save = state.active_display_frame()
    if frame_to_save is None:
        state.set_status("No frame to save yet", (0.9, 0.2, 0.2, 1.0))
        return
    try:
        # frame_to_save is RGB-ordered (as uploaded to the GL texture);
        # cv2.imwrite expects BGR, so reverse the channel order to save a
        # PNG that matches exactly what's currently on screen.
        bgr_for_save = frame_to_save[:, :, ::-1]
        path = _resolve_path(state, "frame", ".png")
        cv2.imwrite(path, bgr_for_save)
        state.set_status(f"Saved {path}", (0.2, 0.8, 0.2, 1.0))
    except Exception as e:
        state.set_status(f"Save error: {e}", (0.9, 0.2, 0.2, 1.0))
