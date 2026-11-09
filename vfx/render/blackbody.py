"""Planck blackbody -> linear sRGB lookup table (computed once on the CPU at init).

Uses the Wyman/Sloan/Shirley (2013) multi-lobe Gaussian fit of the CIE 1931 2-degree observer.
Each LUT entry is chromaticity (normalized so max channel = 1) times a compressed luminance,
so 800 K reads as a dim deep red and 3000 K as a bright yellow-white without blowing out.
"""
from __future__ import annotations

import numpy as np

_XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                         [-0.9689, 1.8758, 0.0415],
                         [0.0557, -0.2040, 1.0570]])


def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def _cie_xyz(lam_nm):
    x = 1.056 * _g(lam_nm, 599.8, 37.9, 31.0) + 0.362 * _g(lam_nm, 442.0, 16.0, 26.7) - 0.065 * _g(lam_nm, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam_nm, 568.8, 46.9, 40.5) + 0.286 * _g(lam_nm, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam_nm, 437.0, 11.8, 36.0) + 0.681 * _g(lam_nm, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


def _planck(lam_nm, T):
    lam = lam_nm[None, :] * 1e-9
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    return (2 * h * c * c) / (lam ** 5 * np.expm1(h * c / (lam * k * T[:, None])))


def blackbody_lut(t_lo: float = 700.0, t_hi: float = 3000.0, n: int = 256,
                  t_ref: float = 1800.0, compress: float = 0.33, onset: float = 150.0) -> np.ndarray:
    """(n, 3) float32 linear-sRGB emission for temperatures linspace(t_lo, t_hi, n).

    Luminance is (Y(T)/Y(t_ref))**compress (a perceptual squash of the ~1e6 dynamic range of
    Planck luminance over this range); a smoothstep over [t_lo, t_lo+onset] fades the onset to 0.
    """
    lam = np.linspace(380.0, 780.0, 401)
    xyz_bar = _cie_xyz(lam)
    T = np.linspace(t_lo, t_hi, n)
    spec = _planck(lam, np.concatenate([T, [t_ref]]))
    XYZ = spec @ xyz_bar                       # (n+1, 3)
    Y = XYZ[:, 1]
    rgb = XYZ @ _XYZ_TO_SRGB.T
    rgb = np.clip(rgb, 0.0, None)
    chroma = rgb / rgb.max(axis=1, keepdims=True)
    lum = (Y[:-1] / Y[-1]) ** compress
    s = np.clip((T - t_lo) / onset, 0.0, 1.0)
    fade = s * s * (3 - 2 * s)
    return (chroma[:-1] * (lum * fade)[:, None]).astype(np.float32)


if __name__ == "__main__":
    lut = blackbody_lut()
    for t in (700, 900, 1200, 1500, 1800, 2200, 2600, 3000):
        i = int(round((t - 700) / 2300 * 255))
        print(t, np.round(lut[i], 3))
