"""
Background readers that keep a latest-frame buffer fed from a producer
thread so the render loop never blocks on network/decode I/O.

FrameReader wraps cv2.VideoCapture on the server's MJPEG /video_feed — a
real stream.

RawStreamPoller and DngStreamPoller are the honest substitute for a raw/DNG
stream: the server only exposes those one-shot (/capture_raw, /capture_dng),
so each loops its single-shot fetch as fast as it can (DNG being far slower,
since it's a full raw file download + rawpy demosaic per "frame"). Both sit
idle (blocked on an Event) unless a caller resume()s them, so they don't
hammer the server while their checkbox is off.
"""

import threading
import time

import cv2

import raw_convert
import server_api


class FrameReader:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.frame = None
        self.new_frame_available = False
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.fps = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        fps_count = 0
        fps_start_time = time.time()

        while self.running:
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                with self.lock:
                    self.connected = False
                time.sleep(1.0)
                continue

            with self.lock:
                self.connected = True

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    with self.lock:
                        self.connected = False
                    break

                with self.lock:
                    self.frame = frame
                    self.new_frame_available = True
                    self.connected = True

                fps_count += 1
                now = time.time()
                if now - fps_start_time >= 1.0:
                    self.fps = fps_count / (now - fps_start_time)
                    fps_count = 0
                    fps_start_time = now

            cap.release()
            time.sleep(0.5)

    def get_frame(self):
        with self.lock:
            is_new = self.new_frame_available
            if is_new:
                self.new_frame_available = False
            return self.frame, is_new

    def get_status(self):
        with self.lock:
            return self.connected

    def stop(self):
        self.running = False


class _PolledReader:
    """Shared scaffolding for a background thread that repeatedly polls a
    one-shot endpoint into a latest-frame buffer, idling on a
    threading.Event unless resume()d."""

    def __init__(self):
        self.frame = None
        self.new_frame_available = False
        self.lock = threading.Lock()

        self.running = True
        self.active = threading.Event()  # cleared = idle, no polling
        self.connected = False
        self.fps = 0.0
        self.last_error = None

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def resume(self):
        self.active.set()

    def pause(self):
        self.active.clear()
        with self.lock:
            self.connected = False

    def _fetch_one(self):
        """Fetch+decode one frame and return it. Any extra per-subclass
        state should be stashed on self under self.lock before returning."""
        raise NotImplementedError

    def _run(self):
        fps_count = 0
        fps_start_time = time.time()

        while self.running:
            self.active.wait()
            if not self.running:
                break

            try:
                frame = self._fetch_one()
                with self.lock:
                    self.frame = frame
                    self.new_frame_available = True
                    self.connected = True
                    self.last_error = None

                fps_count += 1
                now = time.time()
                if now - fps_start_time >= 1.0:
                    self.fps = fps_count / (now - fps_start_time)
                    fps_count = 0
                    fps_start_time = now
            except Exception as e:
                with self.lock:
                    self.connected = False
                    self.last_error = str(e)
                time.sleep(0.5)

            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            is_new = self.new_frame_available
            if is_new:
                self.new_frame_available = False
            return self.frame, is_new

    def get_status(self):
        with self.lock:
            return self.connected

    def stop(self):
        self.running = False
        self.active.set()


class RawStreamPoller(_PolledReader):
    def __init__(self, base_url, bayer_pattern_getter):
        self.base_url = base_url
        self.bayer_pattern_getter = bayer_pattern_getter
        self.raw_arr = None  # last frame's undemosaiced array, for tabs
        self.raw_meta = None
        super().__init__()

    def _fetch_one(self):
        raw_bytes, headers = server_api.fetch_raw_bytes(self.base_url)
        arr, meta = raw_convert.decode_raw_response(raw_bytes, headers)
        bgr = raw_convert.raw_to_bgr(arr, meta, self.bayer_pattern_getter())
        with self.lock:
            self.raw_arr = arr
            self.raw_meta = meta
        return bgr

    def get_raw(self):
        with self.lock:
            return self.raw_arr, self.raw_meta


class DngStreamPoller(_PolledReader):
    """Polls /capture_dng and demosaics via rawpy — slow (full raw file
    download + demosaic per "frame") but correct-by-construction, since
    rawpy reads the sensor's real CFA pattern from the DNG's own metadata
    instead of guessing."""

    def __init__(self, base_url):
        self.base_url = base_url
        super().__init__()

    def _fetch_one(self):
        dng_bytes = server_api.fetch_dng_bytes(self.base_url)
        return raw_convert.dng_to_bgr(dng_bytes)
