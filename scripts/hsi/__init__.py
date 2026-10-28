"""
Shared code for the rotational pushbroom hyperspectral imager experiments.

    camera.py     talk to the Pi camera server, decode raw/DNG into a
                  CFA-free "superpixel" frame (2x2 Bayer cell summed)
    store.py      where captures + calibration files live, save/load helpers
    pipeline.py   dark / white reference -> reflectance -> wavelength grid,
                  loading a scan with its turntable angles
    spectral.py   Fraunhofer/lamp line catalogues, wavelength fitting, smile
                  correction, reflectance, resampling, spectral angle
    sweep.py      continuous capture loop used by scan.py
    turntable.py  time -> angle, period measurement (PDM + turn fit),
                  stitching lines into a panorama
    colour.py     spectrum -> sRGB, and false colour for uncalibrated data
    plotting.py   consistent matplotlib styling for the experiment figures
    sim.py        simulated camera (--url sim://) for testing without hardware

Frame convention everywhere: a frame is a 2D float array indexed
[row, col] where rows run ALONG the slit (spatial, object height) and cols
run ACROSS the dispersion (spectral). On this rig the grating spreads the
spectrum down the sensor's rows, so frames are transposed on capture
(camera.DEFAULT_TRANSPOSE; --no-transpose to turn it off).
"""

import warnings

# Masked (NaN) pixels off the slit ends / outside the camera's wavelength
# range are expected; nanmean over an all-NaN row correctly gives NaN.
warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)
