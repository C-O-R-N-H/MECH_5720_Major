"""
Simulated spectrometer camera, so every experiment script can be run
end-to-end before the hardware exists (and so the analysis code has a test
with known answers). Select it with  --url sim://
The truth values (dispersion, smile, turntable period) are in SIM_TRUTH;
compare a script's output against them.

Each script asks the operator to change the setup between captures
(cover the lens, swap the target, ...) via cam.prompt(message, scene). On
the real camera that waits for Enter; here it just switches the scene.
"""

import time

import numpy as np

from . import spectral

SIM_TRUTH = {
    "roi": [20, 280, 30, 370],          # rows, cols on the superpixel frame
    "lambda0_nm": 395.0,                # wavelength at ROI col 0, ref row
    "nm_per_px": 1.0,
    "quad_nm_per_px2": -2.5e-4,
    "smile_px_per_row2": 1.5e-4,        # col shift = k * (row - mid)^2
    "period_true_factor": 1.013,        # real period = nominal * this
    "rate_hz": 10.0,
    "lsf_sigma_px": 3.0,                # point-source blur along the slit (FWHM = 2.355 x this)
}

H, W = 300, 400  # superpixel frame size


def _lambda_map():
    r0, r1, c0, c1 = SIM_TRUTH["roi"]
    y = np.arange(r1 - r0)[:, None]
    x = np.arange(c1 - c0)[None, :]
    shift = SIM_TRUTH["smile_px_per_row2"] * (y - (r1 - r0) / 2) ** 2
    xs = x - shift
    return (SIM_TRUTH["lambda0_nm"] + SIM_TRUTH["nm_per_px"] * xs
            + SIM_TRUTH["quad_nm_per_px2"] * xs ** 2)


def _sun(lam):
    s = spectral.planck(lam, 5778)
    for l0, depth in zip(spectral.FRAUNHOFER.values(), [0.35, 0.45, 0.4, 0.25, 0.5, 0.45, 0.3, 0.6]):
        s = s * (1 - depth * np.exp(-0.5 * ((lam - l0) / 1.6) ** 2))
    return s


def _fluorescent(lam):
    s = 0.03 * np.ones_like(lam)
    for l0, a in zip(spectral.FLUORESCENT.values(), [0.3, 0.7, 0.4, 0.5, 1.0, 0.4, 0.9]):
        s = s + a * np.exp(-0.5 * ((lam - l0) / 1.4) ** 2)
    return s


def _halogen(lam):
    return spectral.planck(lam, 3000)


def _gauss(c, w):
    return lambda lam: np.exp(-0.5 * ((lam - c) / w) ** 2)


# Bayer filter responses (R, G, B) and cellophane transmissions for the
# cellophane calibration test; the SUM R + 2G + B is roughly _qe below.
_CHANNEL_QE = [lambda lam: 0.9 * _gauss(605, 45)(lam) / (1 + np.exp((lam - 690) / 12)),
               _gauss(535, 40), _gauss(460, 35)]
_CELLOPHANE = {"red": lambda lam: 0.05 + 0.85 / (1 + np.exp(-(lam - 595) / 10)),
               "green": lambda lam: 0.05 + 0.7 * _gauss(530, 30)(lam),
               "blue": lambda lam: 0.05 + 0.75 * _gauss(455, 30)(lam)}


def _qe(lam):
    # summed-CFA silicon response with an IR-cut filter rolling off past 680 nm
    return np.exp(-0.5 * ((lam - 540) / 110) ** 2) / (1 + np.exp((lam - 690) / 12))


# reflectance spectra of the turntable test object's materials
def _materials(lam):
    sig = lambda c, w: 1 / (1 + np.exp(-(lam - c) / w))
    return {
        "white": 0.9 * np.ones_like(lam),
        "red paint": 0.05 + 0.8 * sig(595, 12),
        "leaf": 0.05 + 0.10 * np.exp(-0.5 * ((lam - 550) / 25) ** 2) + 0.5 * sig(700, 10),
        "green paint": 0.05 + 0.12 * np.exp(-0.5 * ((lam - 545) / 35) ** 2),
        "blue paint": 0.08 + 0.5 * np.exp(-0.5 * ((lam - 455) / 35) ** 2),
    }


class SimCamera:
    def __init__(self, transpose=False):
        self.scene = "dark"
        self.exposure_scale = 1.0
        self.nominal_period = 5.0
        self.t_start = None
        self._lam = _lambda_map()
        self._last = 0.0
        self.rng = np.random.default_rng(0)

    # interface shared with camera.Camera
    def config(self):
        return {"sim": True, "width": W * 2, "height": H * 2}

    def lock_exposure(self, exposure_us, gain=1.0):
        self.exposure_scale = exposure_us / 10000.0 * gain

    def set_controls(self, **values):
        pass  # auto exposure etc. don't exist in the simulator

    def prompt(self, message, scene):
        print(f"[sim] {message}  -> scene '{scene}'")
        self.scene = scene
        if scene.startswith("turntable"):
            # "turntable:<nominal period s>"
            self.nominal_period = float(scene.split(":")[1]) if ":" in scene else 5.0
            self.t_start = time.time()

    def grab(self, method="raw"):
        return self.grab_raw()

    def grab_raw(self):
        rgb, meta = self.grab_rgb()
        return rgb[..., 0] + 2 * rgb[..., 1] + rgb[..., 2], meta

    def grab_rgb(self):
        """(H, W, 3) R, G, B per 2x2 cell, like camera.Camera.grab_rgb: each
        pixel type sees the light through its own colour filter (_CHANNEL_QE)."""
        wait = 1 / SIM_TRUTH["rate_hz"] - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = t = time.time()
        signal = np.zeros((H, W, 3))
        r0, r1, c0, c1 = SIM_TRUTH["roi"]
        lam = self._lam
        if self.scene != "dark":
            rows = np.ones(r1 - r0)
            if self.scene == "sun":
                light, refl = _sun(lam), 0.9
            elif self.scene == "fluorescent":
                light, refl = _fluorescent(lam), 0.9
            elif self.scene == "lamp-white":
                light, refl = _halogen(lam), 0.9
            elif self.scene.startswith("filter-"):       # cellophane over the lamp
                light, refl = _halogen(lam) * _CELLOPHANE[self.scene[7:]](lam), 0.9
            elif self.scene.startswith("point:"):        # small source at a fraction along the slit
                light, refl = _halogen(lam), 0.9
                y = np.arange(r1 - r0)
                rows = 0.02 + np.exp(-0.5 * ((y - float(self.scene[6:]) * (r1 - r0)) / SIM_TRUTH["lsf_sigma_px"]) ** 2)
            elif self.scene.startswith("edge:"):         # lit card, dark above an edge across the slit
                from scipy.special import erf

                light, refl = _halogen(lam), 0.9
                y = np.arange(r1 - r0)
                y0 = float(self.scene[5:]) * (r1 - r0)
                rows = 0.03 + 0.97 * 0.5 * (1 + erf((y - y0) / (np.sqrt(2) * SIM_TRUTH["lsf_sigma_px"])))
            elif self.scene == "lamp-patches":
                light, refl = _halogen(lam), self._patches(lam)
            elif self.scene.startswith("turntable"):
                light, refl = _halogen(lam), self._turntable(lam, t)
            else:
                raise ValueError(self.scene)
            # slit edges fall off smoothly along the slit (vignetting)
            vig = rows * (1 - 0.3 * ((np.arange(r1 - r0) - (r1 - r0) / 2) / (r1 - r0)) ** 2)
            base = 12000 * self.exposure_scale * light * refl * vig[:, None]
            for c in range(3):
                # R + 2G + B summed reproduces the overall _qe response
                signal[r0:r1, c0:c1, c] = base * _CHANNEL_QE[c](lam) * (0.5 if c == 1 else 1.0)
        noisy = self.rng.poisson(np.maximum(signal, 0)) + 64.0 + self.rng.normal(0, 3, signal.shape)
        return noisy.astype(np.float32), {"t": t, "t_request": t - 0.01, "t_response": t, "format": "SGBRG10"}

    def _patches(self, lam):
        mats = list(_materials(lam).values())
        n = lam.shape[0]
        out = np.empty_like(lam)
        for k, m in enumerate(mats):
            out[k * n // len(mats):(k + 1) * n // len(mats)] = m[k * n // len(mats):(k + 1) * n // len(mats)]
        return out

    def _turntable(self, lam, t):
        period = self.nominal_period * SIM_TRUTH["period_true_factor"]
        phi = (360.0 * (t - self.t_start) / period) % 360.0
        mats = _materials(lam)
        names = list(mats)
        # 8 vertical stripes around the object; top third is always white
        k = int(phi // 45) % 8
        stripe = mats[names[[0, 1, 2, 3, 4, 2, 3, 1][k]]]
        out = stripe.copy()
        out[: lam.shape[0] // 3] = mats["white"][: lam.shape[0] // 3]
        # a black marker stripe for timing experiments
        if 200 <= phi < 210:
            out *= 0.05
        return out
