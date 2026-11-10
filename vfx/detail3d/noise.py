"""Periodic, band-limited 3D noise tile (built once with an FFT; sampled with hardware trilinear + WRAP).

Channels (fp16, unit RMS each):
  0..2  curl noise: v = curl(psi) of a random band-limited vector potential psi, computed spectrally
        (i k x psi_hat), so v is exactly divergence-free on the tile -> displacing a density by it
        neither creates nor destroys smoke to first order (it is a volume-preserving map).
  3     scalar noise with the same band (multiplicative density detail).

The band is an annulus |k| in [k_lo, k_hi) cycles per tile with a smooth (raised-cosine) edge, so one
tap of the tile carries exactly the frequencies one detail octave should add and nothing below it:
detail never changes the coarse sim's own scales (the "frequency cutoff" of Houdini Gas Up Res /
wavelet turbulence; Cook & DeRose wavelet noise is the same idea: a band-limited, tileable texture).
"""
from __future__ import annotations

import math

import torch


def band_noise_tile(n: int = 64, k_lo: float = 5.0, k_hi: float = 10.0, seed: int = 7,
                    device="cuda") -> torch.Tensor:
    """(n, n, n, 4) float32: curl noise xyz + scalar noise, periodic with period n texels.
    Layout [z, y, x, c]; channel 0 = x component (world x), 1 = y, 2 = z."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.fft.fftfreq(n, d=1.0 / n)                       # integer cycles per tile
    kz, ky, kx = torch.meshgrid(k, k, k, indexing="ij")
    kk = torch.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    # smooth annulus: 1 inside [k_lo, k_hi), raised-cosine roll-off over 1 cycle each side
    def ramp(x):
        return 0.5 - 0.5 * torch.cos(math.pi * x.clamp(0, 1))
    band = ramp(kk - (k_lo - 0.5)) * ramp((k_hi + 0.5) - kk)
    band[0, 0, 0] = 0.0

    def rand_field(nc):
        w = torch.randn(nc, n, n, n, generator=g)
        return torch.fft.fftn(w, dim=(-3, -2, -1)) * band

    psi = rand_field(3)                                       # vector potential (x, y, z) spectra
    two_pi_i = 2j * math.pi
    # curl: (dpz/dy - dpy/dz, dpx/dz - dpz/dx, dpy/dx - dpx/dy); 1/|k| keeps each band's amplitude flat
    inv = torch.where(kk > 0, 1.0 / kk.clamp(min=1e-6), torch.zeros_like(kk))
    vx = two_pi_i * (ky * psi[2] - kz * psi[1]) * inv
    vy = two_pi_i * (kz * psi[0] - kx * psi[2]) * inv
    vz = two_pi_i * (kx * psi[1] - ky * psi[0]) * inv
    s = rand_field(1)[0]
    out = torch.stack([torch.fft.ifftn(f, dim=(-3, -2, -1)).real for f in (vx, vy, vz, s)], -1)
    out[..., :3] = out[..., :3] / out[..., :3].pow(2).mean().sqrt()     # unit RMS per component
    out[..., 3] = out[..., 3] / out[..., 3].pow(2).mean().sqrt()
    return out.float().to(device)


def rotation(seed: int = 3) -> list:
    """A fixed 'irrational' rotation (row-major 3x3) applied between octaves so lattice/tile axes of
    successive octaves never line up (avoids axis-aligned repetition artefacts)."""
    a, b, c = 0.6435, 1.1071, 0.4636          # rad: atan(3/4), atan(2), atan(1/2) — no common period
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    rz = [[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]
    ry = [[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]]
    rx = [[1, 0, 0], [0, cc, -sc], [0, sc, cc]]

    def mm(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return mm(rz, mm(ry, rx))
