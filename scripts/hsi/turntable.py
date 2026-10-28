"""
Turntable timing -> angle, and stitching line scans into a panorama.

The turntable free-runs, so angle comes from time:

    theta(t) = direction * 360 * (t - t0) / period        (degrees, unwrapped)

The speed labels on the turntable don't match reality (every setting runs
6-9% fast), so the period is measured from each scan itself:
  find_period    phase dispersion minimisation (PDM): fold the frames at a
                 trial period and pick the period where frames at the same
                 phase look most alike. Needs > ~1.2 turns.
  refine_period  line every full turn's panorama up with the first and fit
                 the drift, which sharpens the period ~10-50x. Needs >= 2 turns.
"""

import warnings

import numpy as np


def angles(times, period_s, t0=None, direction=1, theta0=0.0):
    t0 = times[0] if t0 is None else t0
    return theta0 + direction * 360.0 * (np.asarray(times) - t0) / period_s


def frame_features(frames, n_blocks=64):
    """Compact per-frame descriptor for period finding: the broadband
    brightness profile along the slit (sum over wavelength), block-averaged
    to n_blocks values and normalised. frames: (n, rows, cols) or (n, rows, bands)."""
    n, rows = frames.shape[0], frames.shape[1]
    prof = np.empty((n, rows), np.float64)
    for i in range(n):  # loop so a memmap isn't pulled into RAM at once
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN, dropped below
            prof[i] = np.nanmean(np.asarray(frames[i], np.float64), axis=1)
    k = max(1, rows // n_blocks)
    prof = prof[:, : rows - rows % k].reshape(n, -1, k).mean(axis=2)
    # blocks that are NaN in any frame (masked rows off the slit ends) carry
    # no information and would poison every statistic
    prof = prof[:, np.all(np.isfinite(prof), axis=0)]
    prof -= prof.mean(axis=0)
    prof /= prof.std() + 1e-12
    return prof


def pdm_statistic(times, feats, period):
    phase = np.mod(times - times[0], period) / period
    order = np.argsort(phase)
    f = feats[order]
    d = np.diff(f, axis=0)
    # wrap the last frame round to the first so the seam counts too
    d = np.vstack([d, f[:1] - f[-1:]])
    return float(np.mean(d * d) / (2 * np.var(feats) + 1e-12))


def stitch(frames, theta_deg, n_az, rev=None, window=None, method="linear"):
    """Resample line scans onto a uniform 0..360 azimuth grid.

    frames: (n, rows, bands) — one line scan per frame, already calibrated.
    theta_deg: unwrapped angle of each frame. rev: if given, only frames from
    that revolution (floor(theta/360) == rev) are used — exp_consistency.py
    compares the panorama from revolution 0 with revolution 1. Otherwise ALL revolutions
    are merged: because the turntable free-runs, each revolution samples
    slightly different angles, so merging gives denser angular sampling.

    Linear interpolation in angle with wrap-around at 360 (method="nearest"
    takes the closest line instead, for comparison). Returns
    (panorama (n_az, rows, bands), azimuth grid, max angular gap in degrees).
    """
    theta = np.asarray(theta_deg, np.float64)
    sel = np.arange(len(theta))
    if rev is not None:
        window = (360.0 * rev, 360.0 * (rev + 1))
    if window is not None:
        sel = sel[(theta >= window[0]) & (theta < window[1])]
    if len(sel) < 2:
        raise ValueError("not enough frames for this revolution")
    phi = np.mod(theta[sel], 360.0)
    order = np.argsort(phi)
    phi_s, idx_s = phi[order], sel[order]
    grid = (np.arange(n_az) + 0.5) * 360.0 / n_az

    right = np.searchsorted(phi_s, grid) % len(phi_s)
    left = (right - 1) % len(phi_s)
    a_l = phi_s[left]
    a_r = phi_s[right]
    span = np.mod(a_r - a_l, 360.0)
    span[span == 0] = 360.0 if len(phi_s) == 1 else 1e-9
    w = np.mod(grid - a_l, 360.0) / span
    if method == "nearest":  # take the closest line instead of blending two
        w = (w >= 0.5).astype(float)

    pano = np.empty((n_az, *frames.shape[1:]), np.float32)
    for k in range(n_az):
        fl = np.asarray(frames[idx_s[left[k]]], np.float32)
        fr = np.asarray(frames[idx_s[right[k]]], np.float32)
        pano[k] = (1 - w[k]) * fl + w[k] * fr
    gaps = np.diff(np.concatenate([phi_s, phi_s[:1] + 360.0]))
    return pano, grid, float(gaps.max())


def find_period(times, feats, lo=1.5, hi=40.0, n=3000, tol=1.15):
    """Period search when the turntable speed setting is unknown.

    PDM is also low at multiples of the true period (and sometimes at
    fractions, if the object looks similar every quarter turn), so instead
    of the global minimum take the SHORTEST period whose statistic is within
    `tol` of the best one, then refine around it. Only periods the scan
    covers at least 1.2 times are considered.

    Returns (period, pdm_value). pdm_value near 0 = confident; above ~0.6
    the scan doesn't contain enough detail per turn to trust the result."""
    span = times[-1] - times[0]
    hi = min(hi, span / 1.2)
    trials = np.geomspace(lo, hi, n)
    stat = np.array([pdm_statistic(times, feats, p) for p in trials])
    best = stat.min()
    k = int(np.argmax(stat <= tol * best))   # first (shortest) near-best period
    # walk downhill to that dip's own minimum
    while k + 1 < len(stat) and stat[k + 1] < stat[k]:
        k += 1
    fine = np.linspace(trials[k] * 0.98, trials[k] * 1.02, 800)
    fstat = np.array([pdm_statistic(times, feats, p) for p in fine])
    # With coarse angular sampling the minimum can be a flat-bottomed range
    # of equally good periods; take its middle, not its edge.
    flat = fine[fstat <= fstat.min() * 1.002 + 1e-6]
    return float(np.median(flat)), float(fstat.min())


def circ_shift(a, b):
    """Sub-sample circular shift (in samples) of b relative to a along axis 0."""
    fa = np.fft.rfft(a - a.mean(0), axis=0)
    fb = np.fft.rfft(b - b.mean(0), axis=0)
    cc = np.fft.irfft(np.sum(np.conj(fa) * fb, axis=1), n=a.shape[0])
    k = int(np.argmax(cc))
    c = np.concatenate([cc[-1:], cc, cc[:1]])
    from .spectral import subpixel_peak

    sub = subpixel_peak(c, k + 1) - 1
    n = a.shape[0]
    return ((sub + n / 2) % n) - n / 2


def refine_period(times, feats, period, iters=4):
    """Sharpen a period estimate using every full revolution in the scan.

    PDM can only resolve the period to within a flat-bottomed range when
    there are few lines per turn. If the period is slightly wrong, the
    panorama from revolution r comes out rotated relative to revolution 0
    by r * delta degrees, with delta = 360 * (true/period - 1), and each
    shift can be measured to a fraction of a line by cross-correlation.
    Fitting delta to ALL revolutions (not just the first two, which can
    agree by chance) and correcting:
        true = period * (1 + delta / 360)
    Needs >= 2 revolutions; otherwise the period is returned unchanged."""
    f = feats[:, :, None]
    for _ in range(iters):
        theta = angles(times, period)
        revs = theta[-1] / 360.0
        n_rev = int(revs)
        if n_rev < 2:
            return period
        n_az = int(round(len(times) / revs))
        p0, _, _ = stitch(f, theta, n_az, rev=0)
        r = np.arange(1, n_rev)
        shifts = np.array([circ_shift(p0[..., 0], stitch(f, theta, n_az, rev=k)[0][..., 0]) * 360.0 / n_az
                           for k in r])
        delta = float(np.sum(r * shifts) / np.sum(r * r))  # slope through the origin
        period = period * (1 + delta / 360.0)
    return float(period)


def turn_offsets(lines, theta, n_az=40):
    """Rotation (deg) of every full turn's panorama relative to turn 1, by
    cross-correlating the broadband brightness profiles. With the right
    period these scatter around 0; a wrong period makes them grow by a
    constant step per turn."""
    f = frame_features(lines)[:, :, None]
    n_rev = int(theta[-1] // 360)
    if n_rev < 2:
        return np.array([])
    p0, _, _ = stitch(f, theta, n_az, rev=0)
    return np.array([circ_shift(p0[..., 0], stitch(f, theta, n_az, rev=r)[0][..., 0]) * 360 / n_az
                     for r in range(1, n_rev)])
