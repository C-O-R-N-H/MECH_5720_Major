"""Intensity histograms of the currently displayed frame, in RGB or HSV."""

import cv2
import numpy as np
from imgui_bundle import imgui, implot

from tabs.base import Tab

_RGB_COLORMAP = "CamViewRGB"
_HSV_COLORMAP = "CamViewHSV"
_colormaps_registered = False


def _ensure_colormaps():
    global _colormaps_registered
    if _colormaps_registered:
        return
    implot.add_colormap(
        _RGB_COLORMAP,
        np.array(
            [
                [0.90, 0.25, 0.25, 1.0],  # R
                [0.30, 0.85, 0.35, 1.0],  # G
                [0.30, 0.55, 0.95, 1.0],  # B
            ],
            dtype=np.float32,
        ),
        True,
    )
    implot.add_colormap(
        _HSV_COLORMAP,
        np.array(
            [
                [0.90, 0.70, 0.15, 1.0],  # H - gold
                [0.65, 0.35, 0.85, 1.0],  # S - purple
                [0.80, 0.80, 0.80, 1.0],  # V - gray
            ],
            dtype=np.float32,
        ),
        True,
    )
    _colormaps_registered = True


# A pixel counts as clipped at >= SAT_LEVEL (254 rather than 255 so JPEG
# ringing on a blown-out region doesn't hide it). The verdict is based on the
# fraction of pixels with *any* channel clipped.
SAT_LEVEL = 254
SAT_WARN_FRAC = 0.005
SAT_BAD_FRAC = 0.02

_SAT_OK_COLOR = imgui.ImVec4(0.30, 0.85, 0.35, 1.0)
_SAT_WARN_COLOR = imgui.ImVec4(0.95, 0.75, 0.15, 1.0)
_SAT_BAD_COLOR = imgui.ImVec4(0.90, 0.25, 0.25, 1.0)


def _saturation_stats(frame):
    clipped = frame >= SAT_LEVEL
    per_channel = clipped.reshape(-1, clipped.shape[-1]).mean(axis=0)
    any_channel = clipped.any(axis=-1).mean()
    return per_channel, float(any_channel)


def _channel_hist(data, nbins, vmax):
    hist, _ = np.histogram(data, bins=nbins, range=(0, vmax))
    return hist.astype(np.float32)


class HistogramTab(Tab):
    name = "Histogram"

    SPACES = ["RGB", "HSV"]

    def __init__(self):
        self._space_idx = 0
        self._last_key = None
        self._hists = []  # list of (label, values)
        self._last_sat_seq = -1
        self._sat = None  # (per-channel fractions, any-channel fraction)

    def _render_saturation(self):
        per_channel, any_channel = self._sat
        if any_channel >= SAT_BAD_FRAC:
            color, verdict = _SAT_BAD_COLOR, "SATURATED - reduce exposure/gain"
        elif any_channel >= SAT_WARN_FRAC:
            color, verdict = _SAT_WARN_COLOR, "Near saturation"
        else:
            color, verdict = _SAT_OK_COLOR, "Not saturated"
        imgui.text_colored(color, verdict)
        imgui.same_line()
        imgui.text_disabled(
            f"clipped (>= {SAT_LEVEL}): "
            f"R {per_channel[0]:.2%}  G {per_channel[1]:.2%}  B {per_channel[2]:.2%}  "
            f"any {any_channel:.2%}"
        )

    def render(self, state):
        frame = state.active_display_frame()
        if frame is None:
            imgui.text("No frame yet.")
            return

        _ensure_colormaps()

        _, self._space_idx = imgui.combo("Color space", self._space_idx, self.SPACES)
        space = self.SPACES[self._space_idx]

        if state.frame_seq != self._last_sat_seq:
            self._last_sat_seq = state.frame_seq
            self._sat = _saturation_stats(frame)
        self._render_saturation()

        key = (state.frame_seq, space)
        if key != self._last_key:
            self._last_key = key
            if space == "RGB":
                self._hists = [
                    ("R", _channel_hist(frame[:, :, 0], 256, 256)),
                    ("G", _channel_hist(frame[:, :, 1], 256, 256)),
                    ("B", _channel_hist(frame[:, :, 2], 256, 256)),
                ]
            else:
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                self._hists = [
                    ("H", _channel_hist(hsv[:, :, 0], 180, 180)),  # OpenCV hue range
                    ("S", _channel_hist(hsv[:, :, 1], 256, 256)),
                    ("V", _channel_hist(hsv[:, :, 2], 256, 256)),
                ]

        implot.push_colormap(_RGB_COLORMAP if space == "RGB" else _HSV_COLORMAP)
        if implot.begin_plot(f"{space} Histogram", imgui.ImVec2(-1, -1)):
            implot.setup_axes("Value", "Pixel count")
            for label, values in self._hists:
                implot.plot_line(label, values)
            implot.end_plot()
        implot.pop_colormap()
