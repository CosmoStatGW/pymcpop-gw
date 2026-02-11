from __future__ import annotations

from typing import Tuple
import numpy as np

import pytensor.tensor as at
from pytensor.graph.op import Op, Apply
from constants import _PI




def pack1d(L):
    """Flatten each entry (scalar -> length-1) and concatenate into one 1D tensor."""
    flats = []
    for v in L:
        v = at.as_tensor_variable(v)
        v = v[None] if v.ndim == 0 else v.ravel()
        flats.append(v)
    return at.concatenate(flats, axis=0)


def pack1d_with_layout(L):
    flats, lens = [], []
    for v in L:
        v = at.as_tensor_variable(v)
        v = v[None] if v.ndim == 0 else v.ravel()
        flats.append(v)
        lens.append(v.shape[0])
    return at.concatenate(flats, axis=0), at.stack(lens).astype("int64")

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


def atinterp_uniform(bk, x, *args, eps=1e-12, side="left"):
    # New style: (xp, fp)
    if len(args) == 2:
        xp, fp_const = args
        n = xp.shape[0]
        x0 = xp[0]
        dx = xp[1] - xp[0]
    # Legacy style: (x0, x1, nU, fp)
    elif len(args) == 4:
        x0, x1, nU, fp_const = args
        n = nU
        dx = (x1 - x0) / bk.maximum(nU - 1, 1)
    else:
        raise TypeError("atinterp_uniform expects (x, xp, fp) or (x, x0, x1, nU, fp)")

    dx = bk.maximum(dx, eps)
    t = (x - x0) / dx
    t = bk.clip(t, 0.0, (n - 1) * 1.0)

    if side == "right":
        j = bk.ceil(t) - 1.0
    else:
        j = bk.floor(t)

    if hasattr(bk, "asarray"):
        j = bk.asarray(j, dtype="int32")
    else:
        j = j.astype("int32")
    j = bk.clip(j, 0, n - 2)

    xl = x0 + j * dx
    yl = fp_const[j]
    yh = fp_const[j + 1]
    r = (x - xl) / dx
    r = bk.clip(r, 0.0, 1.0)
    return (1.0 - r) * yl + r * yh


# ---------------------------------------------------------------------
# interpolation


def atinterp_clip(bk, x, xp, fp_const, eps=1e-12, side="left"):
    n = xp.shape[0]
    idx = bk.searchsorted(xp, x, side=side)
    idx = bk.clip(idx, 1, n - 1)
    #idx = bk.stop_grad(idx) 

    xl = xp[idx - 1]; xh = xp[idx]
    yl = fp_const[idx - 1]; yh = fp_const[idx]

    denom = bk.maximum(xh - xl, eps)
    r = (x - xl) / denom
    return (1.0 - r) * yl + r * yh


def atinterp(bk, x, xs, ys):

  idxs = bk.searchsorted(xs, x, side='left')
  idxs = bk.clip(idxs, 1, xs.shape[0] - 1) # out of index case

  xl = xs[idxs-1]
  yl = ys[idxs-1]
  xh = xs[idxs]
  yh = ys[idxs]

  r = (x-xl)/(xh-xl)

  return r*yh + (1.0-r)*yl

    
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




def make_dL_to_z_table(
    bk,
    dL_grid,
    zgrid,
    *,
    NdL=1024,
    logspace=True,
):
    """
    Build a uniform (or log-uniform) dL axis and the corresponding inverse table z(dL_u),
    using your existing atinterp(dL_u, dL_grid, zgrid).

    This does NOT change cosmology: dL_grid is still computed from zgrid.
    It only amortizes searchsorted by moving it to a small table (NdL points).

    Returns:
      dL_u (uniform in dL or log(dL))
      z_u  (same length)
    """
    dLmin = dL_grid[0]
    dLmax = dL_grid[-1]

    # Build uniform axis in dL (or in log dL)
    if logspace:
        # guard against <=0
        dLmin_g = bk.maximum(dLmin, eps)
        dLmax_g = bk.maximum(dLmax, dLmin_g * (1.0 + 1e-12))
        # need exp/log; bk should provide these (JAXBackend does)
        u0 = bk.log(dLmin_g)
        u1 = bk.log(dLmax_g)
        u = bk.linspace(u0, u1, NdL)
        dL_u = bk.exp(u)
    else:
        dL_u = bk.linspace(dLmin, dLmax, NdL)

    # Invert once using your existing general interp (searchsorted on NdL points only)
    z_u = atinterp(bk, dL_u, dL_grid, zgrid)

    return dL_u, z_u

