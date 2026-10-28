"""Splits the currently displayed frame into its R/G/B channels, shown
side by side as grayscale images."""

import numpy as np
import OpenGL.GL as gl
from imgui_bundle import imgui

from tabs.base import Tab

LABELS = ("Red", "Green", "Blue")


class ChannelsTab(Tab):
    name = "RGB Channels"

    def __init__(self):
        self._tex_ids = None
        self._last_seq = -1
        self._size = (1, 1)

    def _ensure_textures(self):
        if self._tex_ids is not None:
            return
        self._tex_ids = list(gl.glGenTextures(3))
        for tex_id in self._tex_ids:
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    def render(self, state):
        frame = state.active_display_frame()
        if frame is None:
            imgui.text("No frame yet.")
            return

        self._ensure_textures()

        if state.frame_seq != self._last_seq:
            self._last_seq = state.frame_seq
            h, w = frame.shape[:2]
            self._size = (w, h)
            for i, tex_id in enumerate(self._tex_ids):
                channel = np.ascontiguousarray(frame[:, :, i])
                gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D, 0, gl.GL_LUMINANCE, w, h, 0, gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE, channel
                )

        w, h = self._size
        avail = imgui.get_content_region_avail()
        col_w = avail.x / 3 - 8
        scale = col_w / w if w else 1.0
        draw_h = h * scale

        for i, tex_id in enumerate(self._tex_ids):
            imgui.begin_group()
            imgui.text(LABELS[i])
            imgui.image(imgui.ImTextureRef(tex_id), imgui.ImVec2(col_w, draw_h))
            imgui.end_group()
            if i < 2:
                imgui.same_line()
