# jax_utils.py
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _interp_prepare_bk(bk, x, xp, eps=0, side="left"):
    idx = bk.searchsorted( xp, x, side=side)
    idx = bk.clip(idx, 1, xp.shape[0] - 1) #bk.stop_grad( bk.clip(idx, 1, xp.shape[0] - 1) )
    x0 = xp[idx - 1]
    x1 = xp[idx]
    denom = x1 - x0 #bk.maximum(x1 - x0, eps)
    t = (x - x0) / denom
    return idx, t


def _interp_apply_bk(bk, idx, t, fp):
    y0 = fp[idx - 1]
    y1 = fp[idx]
    return (1.0 - t) * y0 + t * y1



def _interp_apply_multi_bk(bk, idx, t, fps):
    """
    fps: array of shape (K, Ngrid)  (stacked tables)
    returns: array of shape (K, ...) matching idx/t broadcast
    """
    y0 = fps[:, idx - 1]
    y1 = fps[:, idx]
    return (1.0 - t) * y0 + t * y1



def _interp_prepare_uniform_bk(bk, x, xp, eps=1e-12):
    """
    Prepare (idx, t) for linear interpolation on a *uniform* 1D grid xp.

    Assumes:
      - xp is 1D, increasing, and uniformly spaced.
    Returns:
      idx: int32 indices in [1, n-1]  (to match your existing _interp_apply_bk convention)
      t:   fraction in [0, 1] between xp[idx-1] and xp[idx]
    """
    n = xp.shape[0]
    x0 = xp[0]
    dx = xp[1] - xp[0]
    #dx = bk.maximum(dx, eps)

    # continuous index in grid units
    s = (x - x0) / dx

    # Clip so that idx in [1, n-1] and idx-1 in [0, n-2]
    # We clip s to [0, n-1], then compute k=floor(s) in [0, n-2]
    s = bk.clip(s, 0.0, (n - 1) * 1.0)

    k = bk.floor(s)
    if hasattr(bk, "asarray"):
        k = bk.asarray(k, dtype="int32")
    else:
        k = k.astype("int32")
    k = bk.clip(k, 0, n - 2)

    # We want idx = k+1 so that _interp_apply_bk uses fp[idx-1], fp[idx]
    idx = k + 1

    # stop grads through discrete indices if available
    # if hasattr(bk, "stop_grad"):
    #     idx = bk.stop_grad(idx)

    # t = (x - xp[k]) / dx ; note xp[k] = x0 + k*dx
    xl = x0 + k * dx
    t = (x - xl) / dx
    t = bk.clip(t, 0.0, 1.0)

    return idx, t






