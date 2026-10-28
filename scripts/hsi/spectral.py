"""
Spectral processing: line catalogues, line finding, the pixel -> wavelength
model (including smile), resampling onto a common wavelength grid, and
reflectance.

Wavelength model (what cal_wavelength.py writes to
calibration/wavelength.json):

    lambda(x, y) = polyval(coeffs, x - smile(y))      x, y in ROI pixel coords
    smile(y)     = polyval(smile_coeffs, y)           column shift of row y
                                                       relative to the reference row

Smile is the curvature of a straight slit's spectral lines across the slit
height, a normal property of grating spectrometers; ignoring it puts a
wavelength error that grows towards the top and bottom of the image.
"""

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import find_peaks, savgol_filter

# --------------------------------------------------------------------------
# Reference lines (nm, air wavelengths)
# --------------------------------------------------------------------------

# Solar Fraunhofer absorption features that stay visible at low resolution
# (a few nm). Close multiplets are given as one blended feature, since a
# home-built slit spectrometer won't split them.
FRAUNHOFER = {
    "G (Fe/CH band)": 430.8,
    "F (H-beta)": 486.1,
    "b (Mg triplet)": 517.3,
    "E2 (Fe)": 527.0,
    "D (Na doublet)": 589.3,
    "C (H-alpha)": 656.3,
    "B (O2 telluric)": 686.7,
    "A (O2 telluric)": 759.4,
}

# Compact fluorescent / fluorescent tube emission lines (Hg + rare-earth
# phosphors). Used as an INDEPENDENT validation set for the sunlight
# calibration, and to measure spectral resolution (they are narrow).
FLUORESCENT = {
    "Hg 404.7": 404.7,
    "Hg 435.8": 435.8,
    "Tb 487.7": 487.7,
    "Tb 542.4": 542.4,
    "Hg 546.1": 546.1,
    "Hg/Eu 577-579": 578.0,
    "Eu 611.6": 611.6,
}

CATALOGUES = {"sun": FRAUNHOFER, "fluorescent": FLUORESCENT}


# --------------------------------------------------------------------------
# Basic frame helpers
# --------------------------------------------------------------------------

def crop(frame, roi):
    """roi = [row0, row1, col0, col1] (half-open). None -> whole frame."""
    if roi is None:
        return frame
    r0, r1, c0, c1 = roi
    return frame[..., r0:r1, c0:c1]


def find_roi(img, frac=0.2, margin=2):
    """Bounding box [row0, row1, col0, col1] of the lit part of a frame
    (the slit's spectrum). Works with or without dark subtraction: the
    frame's darkest 5% is taken as the background level. Raises SystemExit
    if nothing stands clearly above the noise, i.e. no light gets through."""
    s = np.asarray(img, np.float64) - np.percentile(img, 5)
    # pixel-to-pixel noise from neighbour differences, so a frame that is
    # mostly lit doesn't count its own structure as noise
    noise = 1.4826 * np.median(np.abs(np.diff(s, axis=1))) / np.sqrt(2) + 1e-9
    peak = np.percentile(s, 99.5)
    if peak < 10 * noise:
        raise SystemExit(f"no light on the sensor (brightest {peak:.0f} counts above background, "
                         f"noise {noise:.0f}) — lens cap? slit blocked? optics not aligned?")
    lit = s > frac * peak
    # a row/col counts as lit if a few percent of it is (ignores hot pixels)
    rows = np.where(lit.sum(axis=1) > max(3, 0.02 * s.shape[1]))[0]
    cols = np.where(lit.sum(axis=0) > max(3, 0.02 * s.shape[0]))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise SystemExit("no lit region found — is the light reaching the sensor?")
    r0, r1 = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, s.shape[0])
    c0, c1 = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, s.shape[1])
    return [int(r0), int(r1), int(c0), int(c1)]


def bin_rows(frame, n):
    """Average every n rows (spatial binning along the slit) to trade
    height resolution for SNR."""
    if n <= 1:
        return frame
    h = frame.shape[-2] - frame.shape[-2] % n
    f = frame[..., :h, :]
    return f.reshape(*f.shape[:-2], h // n, n, f.shape[-1]).mean(axis=-2)


def reflectance(sample, dark, white, min_frac=0.03):
    """Flat-field: (S - D) / (W - D). Removes illumination spectrum, camera
    spectral response, slit width non-uniformity and vignetting in one go,
    PROVIDED the white reference was taken under the same light and exposure.

    Pixels where the white reference has less than min_frac of its peak
    signal (outside the camera's range, past the IR-cut filter, off the
    end of the slit) become NaN: dividing by ~0 there would only produce
    huge noisy values that swamp every later statistic."""
    den = np.asarray(white, np.float32) - dark
    ok = den > min_frac * np.nanmax(den)
    out = (np.asarray(sample, np.float32) - dark) / np.where(ok, den, 1.0)
    return np.where(ok, out, np.nan).astype(np.float32)


def spectral_angle(a, b, axis=-1):
    """Spectral angle (degrees) between spectra: insensitive to overall
    brightness, so it measures SHAPE differences only."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    dot = np.nansum(a * b, axis=axis)
    na = np.sqrt(np.nansum(a * a, axis=axis))
    nb = np.sqrt(np.nansum(b * b, axis=axis))
    cos = np.clip(dot / np.maximum(na * nb, 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


# --------------------------------------------------------------------------
# Line finding
# --------------------------------------------------------------------------

def continuum_normalise(spec, window=61):
    """Divide by a smooth upper envelope so absorption lines become dips
    below 1.0 regardless of the overall (sun x response) shape."""
    s = np.asarray(spec, np.float64)
    env = maximum_filter1d(s, size=window)
    env = uniform_filter1d(env, size=window)
    return s / np.maximum(env, 1e-9)


def find_lines(spec, kind="absorption", prominence=0.02, distance=4, smooth=5):
    """Sub-pixel positions of absorption dips (sunlight) or emission peaks
    (lamps). Returns (positions, prominences)."""
    s = np.asarray(spec, np.float64)
    if smooth >= 5:
        s = savgol_filter(s, smooth | 1, 2)
    if kind == "absorption":
        sig = -continuum_normalise(s)
    else:
        sig = s / max(np.max(s), 1e-9)
    idx, props = find_peaks(sig, prominence=prominence, distance=distance)
    pos = np.array([subpixel_peak(sig, i) for i in idx])
    return pos, props["prominences"]


def subpixel_peak(sig, i):
    """Parabolic interpolation of a peak at integer index i."""
    if i <= 0 or i >= len(sig) - 1:
        return float(i)
    a, b, c = sig[i - 1], sig[i], sig[i + 1]
    den = a - 2 * b + c
    return float(i) if den == 0 else float(i + 0.5 * (a - c) / den)


def line_fwhm(spec, pos):
    """FWHM in pixels of an emission line at (sub-pixel) position pos."""
    s = np.asarray(spec, np.float64)
    i = int(round(pos))
    peak = s[i]
    base = np.percentile(s, 10)
    half = base + 0.5 * (peak - base)
    lo = i
    while lo > 0 and s[lo] > half:
        lo -= 1
    hi = i
    while hi < len(s) - 1 and s[hi] > half:
        hi += 1
    # linear interpolation on each flank
    left = lo + (half - s[lo]) / (s[lo + 1] - s[lo]) if s[lo + 1] != s[lo] else lo
    right = hi - 1 + (half - s[hi - 1]) / (s[hi] - s[hi - 1]) if s[hi] != s[hi - 1] else hi
    return float(right - left)


# --------------------------------------------------------------------------
# Matching detected lines to a catalogue
# --------------------------------------------------------------------------

def match_lines(pixels, catalogue, tol_nm=3.0, nm_per_px=(0.05, 5.0)):
    """Find which detected pixel positions correspond to which catalogue
    wavelengths, without an initial guess.

    Brute force: every (pixel pair, wavelength pair) defines a linear
    pixel->nm model; keep the model that puts the most detected lines within
    tol_nm of a catalogue line. Both dispersion directions are tried, since
    blue could land on either side of the sensor. Returns a list of
    (pixel, wavelength, name) for the best model's matches.
    """
    px = np.sort(np.asarray(pixels, np.float64))
    names = list(catalogue)
    lams = np.array([catalogue[n] for n in names])
    best = (0, np.inf, None)
    for i in range(len(px)):
        for j in range(i + 1, len(px)):
            for a in range(len(lams)):
                for b in range(len(lams)):
                    if a == b:
                        continue
                    slope = (lams[b] - lams[a]) / (px[j] - px[i])
                    if not (nm_per_px[0] <= abs(slope) <= nm_per_px[1]):
                        continue
                    pred = lams[a] + slope * (px - px[i])
                    err = np.abs(pred[:, None] - lams[None, :])
                    hits = err.min(axis=1) < tol_nm
                    n = int(hits.sum())
                    rms = float(np.sqrt(np.mean(err.min(axis=1)[hits] ** 2))) if n else np.inf
                    if n > best[0] or (n == best[0] and rms < best[1]):
                        best = (n, rms, (lams[a], slope, px[i]))
    if best[2] is None:
        return []
    l0, slope, p0 = best[2]
    pred = l0 + slope * (px - p0)
    out = []
    used = set()
    for p, lp in zip(px, pred):
        k = int(np.argmin(np.abs(lams - lp)))
        if abs(lams[k] - lp) < tol_nm and k not in used:
            used.add(k)
            out.append((float(p), float(lams[k]), names[k]))
    return out


def fit_dispersion(pixels, wavelengths, degree=2):
    """Polynomial pixel -> nm. Returns coeffs, residuals (nm) and the
    leave-one-out residuals, which are the honest calibration error: an
    n-point fit with n-1 coefficients always looks good on its own points."""
    x = np.asarray(pixels, np.float64)
    y = np.asarray(wavelengths, np.float64)
    deg = min(degree, len(x) - 1)
    c = np.polyfit(x, y, deg)
    resid = y - np.polyval(c, x)
    loo = []
    if len(x) > deg + 1:
        for k in range(len(x)):
            m = np.arange(len(x)) != k
            ck = np.polyfit(x[m], y[m], deg)
            loo.append(y[k] - np.polyval(ck, x[k]))
    return c, resid, np.asarray(loo)


# --------------------------------------------------------------------------
# Smile
# --------------------------------------------------------------------------

def measure_shift(ref, spec, max_shift=20):
    """Sub-pixel column shift that best aligns spec to ref (x-corr of the
    high-passed spectra, so the smooth continuum doesn't dominate)."""
    def hp(s):
        s = np.asarray(s, np.float64)
        return s - uniform_filter1d(s, 31)

    a, b = hp(ref), hp(spec)
    lags = np.arange(-max_shift, max_shift + 1)
    cc = np.array([np.sum(a[max(0, -l): len(a) - max(0, l)] * b[max(0, l): len(b) - max(0, -l)]) for l in lags])
    return float(lags[0] + subpixel_peak(cc, int(np.argmax(cc))))


def measure_smile(roi_frame, block=16, ref_row=None):
    """Column shift of each block of rows relative to the reference block.
    Returns (row_centres, shifts)."""
    h = roi_frame.shape[0]
    ref_row = h // 2 if ref_row is None else ref_row
    r0 = max(0, ref_row - block // 2)
    ref = roi_frame[r0: r0 + block].mean(axis=0)
    centres, shifts = [], []
    for start in range(0, h - block + 1, block):
        centres.append(start + block / 2)
        shifts.append(measure_shift(ref, roi_frame[start: start + block].mean(axis=0)))
    return np.array(centres), np.array(shifts)


# --------------------------------------------------------------------------
# Wavelength model + resampling
# --------------------------------------------------------------------------

class WavelengthModel:
    def __init__(self, coeffs, smile_coeffs=(0.0,), **extra):
        self.coeffs = np.asarray(coeffs, np.float64)
        self.smile_coeffs = np.asarray(smile_coeffs, np.float64)
        self.extra = extra

    @classmethod
    def from_json(cls, d):
        return cls(d["coeffs"], d.get("smile_coeffs", [0.0]),
                   **{k: v for k, v in d.items() if k not in ("coeffs", "smile_coeffs")})

    def to_json(self):
        return {"coeffs": self.coeffs, "smile_coeffs": self.smile_coeffs, **self.extra}

    def wavelengths(self, n_rows, n_cols):
        """lambda for every ROI pixel, shape (n_rows, n_cols)."""
        y = np.arange(n_rows, dtype=np.float64)[:, None]
        x = np.arange(n_cols, dtype=np.float64)[None, :]
        return np.polyval(self.coeffs, x - np.polyval(self.smile_coeffs, y))

    def default_grid(self, n_cols, step=5.0, lo=None, hi=None):
        ends = np.polyval(self.coeffs, [0, n_cols - 1])
        lo = np.ceil(min(ends) / step) * step if lo is None else lo
        hi = np.floor(max(ends) / step) * step if hi is None else hi
        return np.arange(lo, hi + step / 2, step)


class Resampler:
    """Precomputed mapping from ROI columns onto a uniform wavelength grid,
    so resampling a whole scan is one sparse matrix product.

    Because the grid step is usually coarser than the pixel pitch, each
    output band is the MEAN of the columns whose wavelength falls inside it
    (box filter) rather than a point sample, which keeps the noise down.
    Smile is handled automatically: each row has its own column->band map."""

    def __init__(self, model, n_rows, n_cols, grid):
        from scipy import sparse

        self.grid = np.asarray(grid, np.float64)
        self.n_rows, self.n_cols, self.n_bands = n_rows, n_cols, len(self.grid)
        lam = model.wavelengths(n_rows, n_cols)
        step = np.median(np.diff(self.grid)) if len(self.grid) > 1 else 1.0
        edges = np.concatenate([self.grid - step / 2, [self.grid[-1] + step / 2]])
        band = np.searchsorted(edges, lam) - 1
        valid = (band >= 0) & (band < self.n_bands)
        rows = np.broadcast_to(np.arange(n_rows)[:, None], lam.shape)
        src = (rows * n_cols + np.arange(n_cols)[None, :])[valid]
        dst = (rows * self.n_bands + band)[valid]
        counts = np.bincount(dst, minlength=n_rows * self.n_bands).astype(np.float64)
        w = 1.0 / counts[dst]
        self.matrix = sparse.csr_matrix(
            (w, (src, dst)), shape=(n_rows * n_cols, n_rows * self.n_bands))
        self.empty = (counts == 0).reshape(n_rows, self.n_bands)

    def __call__(self, roi_frame):
        """(..., rows, cols) -> (..., rows, bands)."""
        f = np.asarray(roi_frame, np.float32)
        lead = f.shape[:-2]
        flat = f.reshape(-1, self.n_rows * self.n_cols)
        out = np.asarray(flat @ self.matrix).reshape(-1, self.n_rows, self.n_bands)
        out[:, self.empty] = np.nan
        return out.reshape(*lead, self.n_rows, self.n_bands).astype(np.float32)


def load_resampler(roi_shape, step=5.0, lo=None, hi=None, row_bin=1):
    """Build a Resampler from calibration/wavelength.json for an ROI of
    roi_shape (rows, cols). If rows have been binned by row_bin, the smile
    polynomial is evaluated at the binned row centres."""
    from . import store

    d = store.load_wavelength()
    model = WavelengthModel.from_json(d)
    n_rows, n_cols = roi_shape
    if row_bin > 1:
        # rescale smile from unbinned row coords to binned row coords
        y = np.arange(n_rows) * row_bin + (row_bin - 1) / 2
        sm = np.polyfit(np.arange(n_rows), np.polyval(model.smile_coeffs, y), min(2, n_rows - 1))
        model = WavelengthModel(model.coeffs, sm)
    grid = model.default_grid(n_cols, step, lo, hi)
    return Resampler(model, n_rows, n_cols, grid), model


# --------------------------------------------------------------------------
# Reference illuminants
# --------------------------------------------------------------------------

def planck(lam_nm, temp_k):
    """Relative blackbody spectral radiance (peak-normalised)."""
    lam = np.asarray(lam_nm, np.float64) * 1e-9
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    b = 1.0 / (lam ** 5 * (np.exp(h * c / (lam * k * temp_k)) - 1.0))
    return b / b.max()


def load_reference_spectrum(path, grid):
    """Two-column CSV (nm, value) -> interpolated onto grid, peak-normalised.
    e.g. the ASTM G173 global-tilt solar spectrum from NREL."""
    data = np.loadtxt(path, delimiter=",", comments="#", usecols=(0, 1))
    v = np.interp(grid, data[:, 0], data[:, 1])
    return v / v.max()
