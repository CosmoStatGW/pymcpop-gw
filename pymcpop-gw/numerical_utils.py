from __future__ import annotations

from typing import Tuple
import numpy as np
import jax.numpy as jnp
from constants import _PI
import jax


# ---------------------------------------------------------------------
# GW stuff
# 
# ---------------------------------------------------------------------

def Mcq_from_m1m2(m1, m2):
   
    Mc  = ((m1*m2)**(3./5.))/((m1+m2)**(1./5.))
    q = m2/m1
    
    return Mc, q



# ---------------------------------------------------------------------
# Utilities
# 
# ---------------------------------------------------------------------


def logit(bk, p):
    return bk.log(p) - bk.log(1. - p)



def safe_logsumexp_jax(a, axis=0):
    finite = jnp.isfinite(a)
    all_bad = jnp.all(~finite, axis=axis, keepdims=True)
    a_safe = jnp.where(finite, a, -jnp.inf)
    m = jnp.max(jnp.where(all_bad, 0.0, a_safe), axis=axis, keepdims=True)
    s = jnp.sum(jnp.exp(a_safe - m), axis=axis, keepdims=True)
    out = m + jnp.log(s)
    out = jnp.where(all_bad, -jnp.inf, out)
    return jnp.squeeze(out, axis=axis)

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

def logsumexp2_unsafe(bk, a, b):
    """Stable log(exp(a)+exp(b)) for two terms."""
    m = bk.maximum(a, b)
    return m + bk.log(bk.exp(a - m) + bk.exp(b - m))


def logsumexp2(bk, a, b):
    both_neg_inf = bk.logical_and(
        bk.eq(a, -jnp.inf),
        bk.eq(b, -jnp.inf)
    )

    m = bk.maximum(a, b)

    # safe shifts: replace -inf with large negative number BEFORE subtraction
    a_safe = bk.where(bk.eq(a, -jnp.inf), -1e30, a)
    b_safe = bk.where(bk.eq(b, -jnp.inf), -1e30, b)

    m_safe = bk.maximum(a_safe, b_safe)

    out = m_safe + bk.log(
        bk.exp(a_safe - m_safe) + bk.exp(b_safe - m_safe)
    )

    return bk.where(both_neg_inf, -jnp.inf, out)


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


# def logdiffexp(bk, a, b, *, eps=1e-16):
#     """
#     Stable log(exp(a) - exp(b)) elementwise.
#     Returns -inf where b >= a.
#     """
#     delta = bk.minimum(b - a, 0.0)     # <= 0
#     ed = bk.exp(delta)                # in [0,1]

#     out = a + bk.log1p(-bk.minimum(ed, 1.0 - eps))
#     return bk.where(b < a, out, -np.inf)

def logdiffexp_nosafe(bk, a, b):
    # returns log(exp(a) - exp(b)); -inf if b >= a
    return bk.where(
        a > b,
        a + bk.log1p(-bk.exp(b - a)),
        -np.inf,
    )


def logdiffexp(bk, a, b, eps=1e-16):
    delta = bk.minimum(b - a, 0.0)           # <= 0
    ed = bk.exp(delta)                        # in (0,1]
    ed = bk.minimum(ed, 1.0 - eps)
    out = a + bk.log1p(-ed)
    return bk.where(b < a, out, -np.inf)


def log_power_gate(bk, x, low, high, p=16.0, eps=1e-12, clip=1e-15):
    """
    Smooth log gate on positive x:

      low cutoff:  -log(1 + (low/x)^p)
      high cutoff: -log(1 + (x/high)^p)

    Returns log gate <= 0.
    """

    logx = bk.log(bk.maximum(x, eps))

    t_low  = p * (bk.log(low)  - logx)
    t_high = p * (logx - bk.log(high))

    log_gate = -bk.logaddexp(0.0, t_low) - bk.logaddexp(0.0, t_high)

    if clip is not None:
        log_gate = bk.maximum(log_gate, bk.log(clip))

    return log_gate

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






# ---------------------------------------------------------------------
# Interpolation
# 
# ---------------------------------------------------------------------

def atinterp(bk, x, xs, ys):

  idxs = bk.searchsorted(xs, x, side='left')
  idxs = bk.clip(idxs, 1, xs.shape[0] - 1) # out of index case

  xl = xs[idxs-1]
  yl = ys[idxs-1]
  xh = xs[idxs]
  yh = ys[idxs]

  r = (x-xl)/(xh-xl)

  return r*yh + (1.0-r)*yl


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




def _interp_indices_nonuniform_safe(bk, x, x_grid):
    """
    Robust index+weight for non-uniform 1D interpolation.

    Returns:
      j  in [1, N-1]
      r  in [0, 1]
    such that:
      xL = x_grid[j-1], xR = x_grid[j]
      y(x) ~ (1-r)*y[j-1] + r*y[j]
    """
    N = x_grid.shape[0]

    # clip x into grid domain (avoid out-of-bounds indices)
    x_clip = bk.clip(x, x_grid[0], x_grid[-1])

    # searchsorted gives insertion index in [0..N]
    j = bk.searchsorted(x_grid, x_clip, side="left")

    # clamp to valid interpolation interval [1..N-1]
    j = bk.clip(j, 1, N - 1)

    xL = x_grid[j - 1]
    xR = x_grid[j]
    denom = bk.maximum(xR - xL, 1e-30)

    r = (x_clip - xL) / denom
    r = bk.clip(r, 0.0, 1.0)

    return j, r



def interp_1d_nonuniform_multiY(bk, x, x_grid, Y, side="left", eps=1e-30):
    """
    Simple multi-Y nonuniform 1D linear interpolation.

    Inputs:
      x      : (...,)
      x_grid : (N,)
      Y      : (K, N)

    Output:
      out    : (K, ...)   (same as your Op)
    """
    N = x_grid.shape[0]

    # clip x into grid domain
    x_clip = bk.clip(x, x_grid[0], x_grid[-1])

    # indices of right/left bracket
    j = bk.searchsorted(x_grid, x_clip, side=side)
    j = bk.clip(j, 1, N - 1)

    xL = x_grid[j - 1]
    xR = x_grid[j]
    denom = bk.maximum(xR - xL, eps)
    r = (x_clip - xL) / denom
    r = bk.clip(r, 0.0, 1.0)

    # gather Y at j-1 and j for all K
    jm1 = (j - 1)[None, ...]  # (1, ...)
    j0  = j[None, ...]        # (1, ...)

    # prefer take_along_axis if backend has it
    if hasattr(bk, "take_along_axis"):
        YL = bk.take_along_axis(Y, jm1, axis=1)  # (K, ...)
        YR = bk.take_along_axis(Y, j0,  axis=1)  # (K, ...)
    else:
        # works for numpy-like backends
        YL = Y[:, (j - 1)]
        YR = Y[:, j]

    out = (1.0 - r)[None, ...] * YL + r[None, ...] * YR
    return out



