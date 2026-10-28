"""
AppState: the one object every window and tab reads/writes. Keeping it all
in one place means tabs never need to reach into app.py or each other.
"""

import time


class AppState:
    def __init__(self, base_url):
        self.base_url = base_url

        # --- server config controls ---
        self.resolutions = ["640x480", "1280x720", "1920x1080"]
        self.formats = ["RGB888", "BGR888", "RGB565", "YUV420"]
        self.current_res_idx = 0
        self.current_fmt_idx = 0

        # --- display processing (applies to every stream) ---
        # Defaults: color_order_idx=0 ("RGB", i.e. BGR->RGB is applied so
        # colors look correct) and rotation_deg=0 (upright, no rotation).
        # See CLAUDE.md.
        self.color_order_options = ["BGR", "RGB"]
        self.color_order_idx = 0
        self.rotation_deg = 90.0
        self.settings_dirty = False  # forces reprocessing of cached frames

        # --- stream selection: exactly one of these is true at a time ---
        self.stream_mode = "processed"  # "processed" | "raw" | "dng"

        # --- raw bayer demosaic guess (cv2's own BG/GB/RG/GR naming). Only
        # relevant to the "raw" stream/tab -- "dng" always demosaics
        # correctly via the file's own metadata, no guessing needed. ---
        self.bayer_pattern = "BG"

        # --- readers, attached by app.py after construction ---
        self.reader = None  # stream.FrameReader (processed /video_feed)
        self.raw_reader = None  # stream.RawStreamPoller (/capture_raw loop)
        self.dng_reader = None  # stream.DngStreamPoller (/capture_dng loop)

        # --- GL textures, created by app.py after GL context exists ---
        self.processed_texture_id = None
        self.raw_texture_id = None
        self.dng_texture_id = None
        self.has_processed_texture = False
        self.has_raw_texture = False
        self.has_dng_texture = False
        self.processed_tex_size = (1, 1)
        self.raw_tex_size = (1, 1)
        self.dng_tex_size = (1, 1)

        # pre-display-processing BGR source frames, cached so settings_dirty
        # can redo color-order/rotation without waiting on a new frame
        self.last_processed_bgr_source = None
        self.last_raw_bgr_source = None
        self.last_dng_bgr_source = None

        # post-display-processing RGB frames (what's on screen / what Save
        # Current writes out); also what the analysis tabs read from
        self.processed_display_frame = None
        self.raw_display_frame = None
        self.dng_display_frame = None

        # bumped whenever any display frame is refreshed, so tabs can
        # cheaply tell "is there new data since I last computed my plot"
        self.frame_seq = 0

        # --- camera controls (Advanced Camera Controls panel) ---
        self.camera_controls = {}

        # --- save filename textbox ---
        self.save_name_buf = ""

        # --- status line ---
        self.status_message = "Ready"
        self.status_color = (0.5, 0.7, 1.0, 1.0)

    def set_status(self, msg, color=(0.5, 0.7, 1.0, 1.0)):
        self.status_message = msg
        self.status_color = color

    def active_display_frame(self):
        """The RGB frame currently on screen in the main video window."""
        if self.stream_mode == "raw":
            return self.raw_display_frame
        if self.stream_mode == "dng":
            return self.dng_display_frame
        return self.processed_display_frame

    def set_stream_mode(self, mode):
        if mode == self.stream_mode:
            return
        self.stream_mode = mode
        # Only "raw" and "dng" are pollers that need pause/resume; the
        # processed FrameReader always runs (it's a real push stream).
        for name, reader in (("raw", self.raw_reader), ("dng", self.dng_reader)):
            if reader is None:
                continue
            reader.resume() if name == mode else reader.pause()

    def bump_frame_seq(self):
        self.frame_seq += 1

    def timestamp_name(self, kind):
        return f"{kind}_{int(time.time())}"
