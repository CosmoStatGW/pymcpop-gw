from __future__ import annotations

from typing import Tuple
import numpy as np

import pytensor.tensor as at
from pytensor.graph.op import Op, Apply
from constants import _PI


# ---------------------------------------------------------------------
# Utilities
# 
# ---------------------------------------------------------------------


def logsumexp(bk, x, axis=None):
    """
    Backend-agnostic logsumexp.
    Assumes bk has max, exp, log, sum.
    """
    m = bk.max(x, axis=axis, keepdims=True)
    s = bk.sum(bk.exp(x - m), axis=axis, keepdims=True)
    out = m + bk.log(s)
    if axis is None:
        return out.reshape(())
    return bk.squeeze(out, axis=axis)


def _logsumexp_np(x: np.ndarray) -> float:
    """Stable logsumexp for 1D numpy arrays."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))

def logsumexp2(bk, a, b):
    """Stable log(exp(a)+exp(b)) for two terms."""
    m = bk.maximum(a, b)
    return m + bk.log(bk.exp(a - m) + bk.exp(b - m))


def logaddexp(bk, a, b):
    """Stable logaddexp using logsumexp2."""
    return logsumexp2(bk, a, b)

def sigmoid(bk, x, x0, s, eps=1e-12, clip=1e-15):
    s_pos = bk.maximum(s, eps)
    t = (x - x0) / s_pos
    y = 0.5 * (bk.tanh(0.5 * t) + 1.0)
    if clip is not None:
        y = bk.clip(y, clip, 1.0 - clip)
    return y


def log_sigmoid(bk, x, m, sig):
    return bk.log(sigmoid(bk, x, m, sig))


def safe_sigmoid(bk, x, x0, eps):
    return sigmoid(bk, x, x0, eps, clip=1e-15)


def logdiffexp(bk, a, b, *, eps=1e-16):
    """
    Stable log(exp(a) - exp(b)) elementwise.
    Returns -inf where b >= a.
    """
    delta = bk.minimum(b - a, 0.0)     # <= 0
    ed = bk.exp(delta)                # in [0,1]

    out = a + bk.log1p(-bk.minimum(ed, 1.0 - eps))
    return bk.where(b < a, out, -np.inf)


# ---------------------------------------------------------------------
# Interpolation
# 
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Uniform grid interpolation 


def atinterp_uniform(bk, x, x0, x1, n, yp):
    """
    Uniform-grid linear interpolation: xp must be uniformly spaced and increasing.
    Backend-agnostic version (works for NumPy + PyTensor).

    Inputs:
      x: query points
      x0, x1: grid endpoints
      n: number of grid points (same meaning as your legacy)
      yp: values on the grid (length n)
    """
    dx = (x1 - x0) / (n - 1)

    t = (x - x0) / dx
    t = bk.clip(t, 0.0, n - 1)

    # integer index
    j = bk.asarray(bk.floor(t), dtype="int32") if hasattr(bk, "asarray") else bk.floor(t).astype("int32")
    j = bk.clip(j, 0, n - 2)

    r = t - j

    y0 = yp[j]
    y1 = yp[j + 1]
    return (1.0 - r) * y0 + r * y1


# ---------------------------------------------------------------------
# interpolation


def atinterp(bk, x, xp, fp_const, eps=1e-12, side="right"):
    n = xp.shape[0]
    idx = bk.searchsorted(xp, x, side=side)
    idx = bk.clip(idx, 1, n - 1)
    idx = bk.stop_grad(idx)  # <- matches your stop_grad(idxs)

    xl = xp[idx - 1]; xh = xp[idx]
    yl = fp_const[idx - 1]; yh = fp_const[idx]

    denom = bk.maximum(xh - xl, eps)
    r = (x - xl) / denom
    return (1.0 - r) * yl + r * yh



    
# ---------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------

def attrapzvec(bk, y, x, *, axis=-1):
    """
    Backend-agnostic trapezoid integral along `axis`.

    Equivalent to:
      ∫ y(x) dx ≈ sum_i 0.5 * (x[i+1]-x[i]) * (y[i+1]+y[i])

    Gradients:
      - w.r.t y: correct
      - w.r.t x: caller should pass stop_grad(x) if x is constant

    Example: norm = attrapzvec(bk, integrand, bk.stop_grad(z))
    """
    d = bk.diff(x, axis=axis)

    # broadcast d if x is 1D and y is ND
    if getattr(x, "ndim", None) == 1 and getattr(y, "ndim", None) != 1:
        shape = [1] * y.ndim
        shape[axis] = d.shape[0]
        d = bk.reshape(d, shape)

    nd = y.ndim
    sl1 = [slice(None)] * nd
    sl2 = [slice(None)] * nd
    sl1[axis] = slice(1, None)
    sl2[axis] = slice(None, -1)

    return bk.sum(d * (y[tuple(sl1)] + y[tuple(sl2)]) * 0.5, axis=axis)


def atcumtrapz(bk, y, x, *, axis=-1):
    """
    Backend-agnostic cumulative trapezoid integral.

    Returns:
      cumulative integral with shape shortened by 1 along `axis`
      (same convention as scipy.integrate.cumtrapz without `initial`)

    Example: cdf  = atcumtrapz(bk, p2, bk.stop_grad(xx))
    """
    d = bk.diff(x, axis=axis)

    if getattr(x, "ndim", None) == 1 and getattr(y, "ndim", None) != 1:
        shape = [1] * y.ndim
        shape[axis] = d.shape[0]
        d = bk.reshape(d, shape)

    nd = y.ndim
    sl1 = [slice(None)] * nd
    sl2 = [slice(None)] * nd
    sl1[axis] = slice(1, None)
    sl2[axis] = slice(None, -1)

    pieces = d * (y[tuple(sl1)] + y[tuple(sl2)]) * 0.5
    return bk.cumsum(pieces, axis=axis)