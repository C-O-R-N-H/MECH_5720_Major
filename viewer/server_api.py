"""
HTTP client functions for talking to the Pi camera server, plus the
threaded action wrappers the UI fires off on button clicks / control edits.

Every `*_thread` function here is meant to be run via
`threading.Thread(target=..., daemon=True).start()` from the UI so network
latency never blocks a frame of the render loop. They report progress/errors
through `state.set_status(...)`.
"""

import requests


def get_config(base_url):
    r = requests.get(f"{base_url}/config", timeout=5)
    r.raise_for_status()
    return r.json()


def change_config_thread(state, width, height, fmt):
    state.set_status(f"Changing config to {width}x{height} {fmt}...", (0.9, 0.6, 0.2, 1.0))
    try:
        r = requests.post(
            f"{state.base_url}/config",
            json={"width": width, "height": height, "format": fmt},
            timeout=10,
        )
        if r.status_code == 200:
            state.set_status("Config updated successfully!", (0.2, 0.8, 0.2, 1.0))
        else:
            state.set_status(f"Error changing config: {r.status_code}", (0.9, 0.2, 0.2, 1.0))
        print("Config change response:", r.status_code, r.json())
    except Exception as e:
        state.set_status(f"Error: {e}", (0.9, 0.2, 0.2, 1.0))


def fetch_camera_controls(state):
    try:
        r = requests.get(f"{state.base_url}/controls", timeout=5)
        if r.status_code == 200:
            state.camera_controls = r.json()
            print("Fetched camera controls from server:", len(state.camera_controls))
    except Exception as e:
        print("Failed to fetch camera controls:", e)


def set_camera_control_thread(state, name, value):
    try:
        r = requests.post(f"{state.base_url}/controls", json={name: value}, timeout=5)
        if r.status_code == 200:
            if name in state.camera_controls:
                state.camera_controls[name]["current"] = value
            state.set_status(f"Updated {name} to {value}", (0.2, 0.8, 0.2, 1.0))
            fetch_camera_controls(state)
        else:
            state.set_status(f"Error setting {name}: {r.status_code}", (0.9, 0.2, 0.2, 1.0))
    except Exception as e:
        state.set_status(f"Error: {e}", (0.9, 0.2, 0.2, 1.0))


def fetch_raw_bytes(base_url):
    """Single-shot fetch of /capture_raw. Returns (raw_bytes, headers)."""
    r = requests.get(f"{base_url}/capture_raw", timeout=10)
    r.raise_for_status()
    return r.content, r.headers


def fetch_dng_bytes(base_url):
    """Single-shot fetch of /capture_dng. Returns the raw DNG file bytes."""
    r = requests.get(f"{base_url}/capture_dng", timeout=15)
    r.raise_for_status()
    return r.content
