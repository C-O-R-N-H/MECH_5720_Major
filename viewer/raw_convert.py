"""
Turns whatever /capture_raw or /capture_dng hands back into pixels, and the
shared color-order + rotation + GL-upload pipeline used by every displayed
frame (processed stream, raw stream, DNG stream, and any tab that wants to
show an image).
"""

import io

import cv2
import numpy as np
import OpenGL.GL as gl
import rawpy

# Bytes-per-pixel for the pixel formats the server supports.
BYTES_PER_PIXEL = {
    "RGB888": 3,
    "BGR888": 3,
    "RGB565": 2,
    "YUV420": 1,
}

# cv2's own Bayer-pattern naming (note: this is transposed from the physical
# CFA name — see CLAUDE.md). Default guess for Pi camera modules.
BAYER_PATTERNS = ["BG", "GB", "RG", "GR"]
DEFAULT_BAYER_PATTERN = "BG"

ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def decode_raw_response(raw_bytes, headers):
    """Parse a /capture_raw response into (arr, meta).

    meta always has width/height/format. If the frame is raw Bayer sensor
    data (server sent X-Shape/X-Dtype), meta["bayer"] is True and arr is a
    2D single-channel array; otherwise arr is HxWxbpp for one of
    BYTES_PER_PIXEL's simple pixel formats and meta["bayer"] is False.
    """
    width = int(headers["X-Width"])
    height = int(headers["X-Height"])
    fmt = headers["X-Format"]

    if "X-Shape" in headers and "X-Dtype" in headers:
        shape = tuple(map(int, headers["X-Shape"].split(",")))
        dtype = np.dtype(headers["X-Dtype"])
        arr = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
        return arr, {"width": width, "height": height, "format": fmt, "bayer": True}

    stride = int(headers["X-Stride"])
    if fmt == "YUV420":
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        return arr, {"width": width, "height": height, "format": fmt, "bayer": False, "stride": stride}

    bpp = BYTES_PER_PIXEL[fmt]
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    rows = len(raw_bytes) // stride
    arr = arr.reshape((rows, stride))
    arr = arr[:, : width * bpp].reshape((height, width, bpp))
    return arr, {"width": width, "height": height, "format": fmt, "bayer": False, "stride": stride}


def raw_to_bgr(arr, meta, bayer_pattern=DEFAULT_BAYER_PATTERN):
    """Normalize a decoded /capture_raw frame into a standard BGR888 image."""
    fmt = meta["format"]

    if meta.get("bayer"):
        img = arr
        if img.dtype != np.uint8:
            # Preview-only normalization: we don't know the sensor's real
            # bit depth reliably, so scale each frame by its own max rather
            # than assume e.g. 10-bit. Fine for a live comparison view, not
            # for radiometric analysis (use the DNG capture for that).
            peak = float(img.max()) or 1.0
            img = (img.astype(np.float32) * (255.0 / peak)).astype(np.uint8)
        cv_const = getattr(cv2, f"COLOR_Bayer{bayer_pattern}2BGR")
        return cv2.cvtColor(img, cv_const)

    if fmt == "RGB888":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if fmt == "BGR888":
        return arr

    if fmt == "RGB565":
        width, height = meta["width"], meta["height"]
        packed = arr[: height * meta["stride"]].reshape((height, meta["stride"]))[:, : width * 2]
        packed16 = packed.reshape((height, width, 2)).view(np.uint16).reshape((height, width))
        return cv2.cvtColor(packed16, cv2.COLOR_BGR5652BGR)

    if fmt == "YUV420":
        width, height = meta["width"], meta["height"]
        yuv = arr[: width * height * 3 // 2].reshape((height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    raise ValueError(f"Unsupported raw format for display: {fmt}")


def dng_to_bgr(dng_bytes, half_size=True):
    """Decode a /capture_dng capture into a BGR888 preview using rawpy's own
    debayering. This is the ground-truth demosaic — rawpy reads the file's
    real CFA pattern/white balance/color matrix from its embedded metadata,
    unlike raw_to_bgr()'s pattern guess for bare /capture_raw bytes.

    half_size=True demosaics at quarter resolution, which is the difference
    between "usable as a live-ish DNG stream" and "unusably slow" — a full
    decode of a several-MP raw frame is far too slow to poll repeatedly.
    """
    with rawpy.imread(io.BytesIO(dng_bytes)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=half_size)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_cfa_tile(dng_bytes):
    """Read the sensor's real 2x2 CFA layout off a DNG via rawpy metadata
    (raw_colors_visible / color_desc) -- the same technique
    lab1/1.2/line_profile.py uses, minus its single-band extraction. Returns
    a 4-character string like "BGGR" in row-major top-left-2x2 order, so it
    can be compared against the cv2 pattern guess (see BAYER_PATTERNS)."""
    with rawpy.imread(io.BytesIO(dng_bytes)) as raw:
        colors = raw.raw_colors_visible
        desc = raw.color_desc.decode()
        return "".join(desc[colors[r, c]] for r in (0, 1) for c in (0, 1))


def process_frame_for_display(frame_bgr, color_order_idx, rotation_deg):
    """
    Takes a BGR frame and returns a frame ready to be uploaded to an RGB
    OpenGL texture, with:
      - color_order_idx == 0 ("RGB"): normal BGR->RGB conversion (correct colors)
      - color_order_idx == 1 ("BGR"): channels left as-is, so red/blue are
        swapped in the displayed image (useful for spotting/undoing a wrong
        RGB/BGR assumption somewhere in the pipeline)
      - rotated by rotation_deg (0/90/180/270) so the stream can be made upright
    """
    if color_order_idx == 0:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        img = frame_bgr.copy()

    rot_code = ROTATIONS.get(rotation_deg)
    if rot_code is not None:
        img = cv2.rotate(img, rot_code)

    return np.ascontiguousarray(img)


def upload_texture(texture_id, rgb_frame):
    h, w = rgb_frame.shape[:2]
    channels = rgb_frame.shape[2] if rgb_frame.ndim == 3 else 1
    gl_format = gl.GL_RGB if channels == 3 else gl.GL_LUMINANCE

    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl_format,
        w,
        h,
        0,
        gl_format,
        gl.GL_UNSIGNED_BYTE,
        rgb_frame,
    )
    return w, h
