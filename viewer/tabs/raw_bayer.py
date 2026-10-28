"""
Side-by-side comparison of the processed stream, a raw sensor capture
demosaiced with a selectable Bayer pattern guess, and a DNG capture
demosaiced by rawpy (ground truth, since rawpy reads the sensor's real CFA
pattern from the DNG's own metadata instead of guessing — same technique as
lab1/1.2/line_profile.py's CFA-aware read, not its single-band extraction;
see CLAUDE.md).

One button fetches both a raw and a DNG capture together so all three panes
can be compared against the same moment in time.
"""

import threading

import OpenGL.GL as gl
from imgui_bundle import imgui

import raw_convert
import server_api
from tabs.base import Tab


class RawBayerTab(Tab):
    name = "Raw / DNG Compare"

    def __init__(self):
        self._raw_tex_id = None
        self._raw_tex_size = (1, 1)
        self._raw_dirty_key = None

        self._dng_tex_id = None
        self._dng_tex_size = (1, 1)
        self._dng_source_id = None

        self._lock = threading.Lock()
        self._raw_arr = None
        self._raw_meta = None
        self._dng_bgr = None
        self._detected_tile = ""
        self._fetch_status = "Nothing fetched yet."

    def _ensure_textures(self):
        for attr in ("_raw_tex_id", "_dng_tex_id"):
            if getattr(self, attr) is not None:
                continue
            tex_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            setattr(self, attr, tex_id)

    def _fetch_both(self, state):
        self._fetch_status = "Fetching raw..."
        try:
            raw_bytes, headers = server_api.fetch_raw_bytes(state.base_url)
            arr, meta = raw_convert.decode_raw_response(raw_bytes, headers)
            with self._lock:
                self._raw_arr, self._raw_meta = arr, meta
        except Exception as e:
            self._fetch_status = f"Raw fetch error: {e}"
            return

        self._fetch_status = "Raw OK. Fetching DNG..."
        try:
            dng_bytes = server_api.fetch_dng_bytes(state.base_url)
            dng_bgr = raw_convert.dng_to_bgr(dng_bytes)
            tile = raw_convert.detect_cfa_tile(dng_bytes)
        except Exception as e:
            self._fetch_status = f"Raw OK; DNG error: {e}"
            return

        with self._lock:
            self._dng_bgr = dng_bgr
            self._detected_tile = tile
        self._fetch_status = "Raw + DNG fetched."

    def render(self, state):
        self._ensure_textures()

        if imgui.button("Fetch Raw + DNG"):
            threading.Thread(target=self._fetch_both, args=(state,), daemon=True).start()
        imgui.same_line()
        imgui.text_disabled(self._fetch_status)

        with self._lock:
            raw_arr, raw_meta = self._raw_arr, self._raw_meta
            dng_bgr, tile = self._dng_bgr, self._detected_tile

        status_line = f"cv2 pattern guess: {state.bayer_pattern}"
        if tile:
            status_line += f"   |   DNG ground-truth CFA tile: {tile}"
        imgui.text_wrapped(status_line)

        # -- keep the raw preview texture in sync with the pattern combo --
        if raw_arr is not None and raw_meta.get("bayer"):
            key = (id(raw_arr), state.bayer_pattern, state.color_order_idx, state.rotation_deg)
            if key != self._raw_dirty_key:
                self._raw_dirty_key = key
                bgr = raw_convert.raw_to_bgr(raw_arr, raw_meta, state.bayer_pattern)
                rgb = raw_convert.process_frame_for_display(bgr, state.color_order_idx, state.rotation_deg)
                w, h = raw_convert.upload_texture(self._raw_tex_id, rgb)
                self._raw_tex_size = (w, h)

        # -- keep the DNG preview texture in sync with the last DNG fetch --
        if dng_bgr is not None and id(dng_bgr) != self._dng_source_id:
            self._dng_source_id = id(dng_bgr)
            rgb = raw_convert.process_frame_for_display(dng_bgr, state.color_order_idx, state.rotation_deg)
            w, h = raw_convert.upload_texture(self._dng_tex_id, rgb)
            self._dng_tex_size = (w, h)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        avail = imgui.get_content_region_avail()
        col_w = avail.x / 3 - 8

        imgui.begin_group()
        imgui.text("Processed stream")
        if state.has_processed_texture:
            pw, ph = state.processed_tex_size
            scale = col_w / pw if pw else 1.0
            imgui.image(imgui.ImTextureRef(state.processed_texture_id), imgui.ImVec2(col_w, ph * scale))
        else:
            imgui.text_disabled("No processed frame yet.")
        imgui.end_group()

        imgui.same_line()

        imgui.begin_group()
        imgui.text("Raw, demosaiced (guess)")
        if raw_arr is not None and raw_meta.get("bayer"):
            rw, rh = self._raw_tex_size
            scale = col_w / rw if rw else 1.0
            imgui.image(imgui.ImTextureRef(self._raw_tex_id), imgui.ImVec2(col_w, rh * scale))
        elif raw_arr is not None:
            imgui.text_disabled("(Not raw Bayer sensor data)")
        else:
            imgui.text_disabled("Fetch above.")
        imgui.end_group()

        imgui.same_line()

        imgui.begin_group()
        imgui.text("DNG, demosaiced (ground truth)")
        if dng_bgr is not None:
            dw, dh = self._dng_tex_size
            scale = col_w / dw if dw else 1.0
            imgui.image(imgui.ImTextureRef(self._dng_tex_id), imgui.ImVec2(col_w, dh * scale))
        else:
            imgui.text_disabled("Fetch above.")
        imgui.end_group()
