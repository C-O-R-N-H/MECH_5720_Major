"""
Spectrum -> sRGB for previews and panorama renders.

CIE 1931 2-degree colour matching functions use the analytic multi-lobe
Gaussian fit of Wyman, Sloan & Shirley (2013, JCGT 2(2)). It's accurate to
a few percent, which is plenty for a preview, and needs no data file.
"""

import numpy as np


def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cie_cmf(lam_nm):
    lam = np.asarray(lam_nm, np.float64)
    x = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


_XYZ_TO_SRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])


def reflectance_to_srgb(refl, grid):
    """(..., bands) reflectance -> (..., 3) sRGB uint8, white-balanced so a
    perfect white reflector (all 1.0) renders as (255, 255, 255).
    Bands outside the camera's range simply don't contribute."""
    cmf = cie_cmf(grid)
    r = np.nan_to_num(np.asarray(refl, np.float64), nan=0.0)
    xyz = r @ cmf
    white = np.ones(len(grid)) @ cmf
    rgb = xyz @ _XYZ_TO_SRGB.T
    rgb_w = white @ _XYZ_TO_SRGB.T
    rgb = np.clip(rgb / rgb_w, 0, 1)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return (rgb * 255 + 0.5).astype(np.uint8)


def false_colour(cube, grid, bands_nm=(620, 550, 460)):
    """Pick three single bands as R, G, B, each stretched from its 1st to
    99th percentile. Useful when there's no white reference yet."""
    idx = [int(np.argmin(np.abs(np.asarray(grid) - b))) for b in bands_nm]
    out = np.stack([cube[..., i] for i in idx], axis=-1).astype(np.float64)
    axes = tuple(range(out.ndim - 1))
    lo = np.nanpercentile(out, 1, axis=axes)
    hi = np.nanpercentile(out, 99, axis=axes)
    return (np.clip(np.nan_to_num((out - lo) / np.maximum(hi - lo, 1e-9)), 0, 1) * 255).astype(np.uint8)


def quick_colour(cube, grid=None, min_frac=0.2):
    """False colour for UNCALIBRATED data, built to be easy on the eye.

    Only the spectral columns that actually carry signal are used (median
    at least min_frac of the brightest column), so a part of the spectrum
    the lamp doesn't reach, e.g. a torch with no blue, doesn't turn into
    pure noise. That usable range is split into thirds, and each third is
    AVERAGED into one channel: short -> B, middle -> G, long -> R.
    Averaging ~100 columns per channel removes most of the per-column noise.
    `grid` is unused; it's accepted so the call matches false_colour's."""
    c = np.asarray(cube, np.float64)
    flat = c.reshape(-1, c.shape[-1])
    level = np.nanmedian(flat, axis=0)
    usable = np.where(level >= min_frac * np.nanmax(level))[0]
    if len(usable) < 9:
        usable = np.arange(c.shape[-1])
    thirds = np.array_split(usable, 3)
    out = np.stack([np.nanmean(c[..., idx], axis=-1) for idx in thirds[::-1]], axis=-1)  # R, G, B
    axes = tuple(range(out.ndim - 1))
    lo = np.nanpercentile(out, 1, axis=axes)
    hi = np.nanpercentile(out, 99, axis=axes)
    return (np.clip(np.nan_to_num((out - lo) / np.maximum(hi - lo, 1e-9)), 0, 1) * 255).astype(np.uint8)
