# jax_utils.py
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _searchsorted_bk(bk, xp, x, side="right"):
    # Selection Op is JAX; still keep a safe fallback
    if jnp is not None and (type(xp).__module__.startswith("jax") or type(x).__module__.startswith("jax")):
        return jnp.searchsorted(xp, x, side=side)
    # numpy fallback (mostly for tests)
    return np.searchsorted(np.asarray(xp), np.asarray(x), side=side)

def _interp_prepare_bk(bk, x, xp, eps=1e-12, side="right"):
    idx = _searchsorted_bk(bk, xp, x, side=side)
    idx = bk.clip(idx, 1, xp.shape[0] - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    denom = bk.maximum(x1 - x0, eps)
    t = (x - x0) / denom
    return idx, t

def _interp_apply_bk(bk, idx, t, fp):
    y0 = fp[idx - 1]
    y1 = fp[idx]
    return (1.0 - t) * y0 + t * y1


def make_interp_pt(*, eps: float = 1e-12, side: str = "right"):
    """
    JAX interp that matches the PyTensor 'standard' atinterp semantics:

      idx = stop_grad(clip(searchsorted(xp, x, side), 1, n-1))
      denom = max(x1 - x0, eps)
      r = (x - x0)/denom
      y = (1-r)*y0 + r*y1

    Gradients:
      - No gradient through idx (discrete selection).
      - Piecewise-linear gradient wrt query x and fp values.
      - No gradient wrt xp (knot positions) by default (returns None),
        matching common PyTensor use where xp is constant.

    Returns
    -------
    interp_pt(x, xp, fp) -> y
    """
    side = str(side)
    eps = float(eps)

    def _forward(x, xp, fp):
        n = xp.shape[0]
        idx = jnp.searchsorted(xp, x, side=side)
        idx = jnp.clip(idx, 1, n - 1)

        x0 = xp[idx - 1]
        x1 = xp[idx]
        y0 = fp[idx - 1]
        y1 = fp[idx]

        denom = jnp.maximum(x1 - x0, eps)
        r = (x - x0) / denom
        y = (1.0 - r) * y0 + r * y1

        return y, (idx, r, denom, y0.shape, y1.shape, fp.shape)

    @jax.custom_vjp
    def interp_pt(x, xp, fp):
        y, _ = _forward(x, xp, fp)
        return y

    def fwd(x, xp, fp):
        y, res = _forward(x, xp, fp)
        return y, res

    def bwd(res, g):
        idx, r, denom, _y0_shape, _y1_shape, fp_shape = res

        # Need dy/dx = (y1 - y0)/denom, but we don't have y0/y1 cached.
        # Reconstruct dy via fp gathers (cheap; no second searchsorted).
        # NOTE: idx is treated as constant (PyTensor stop_grad semantics).
        # We have access to fp only through closure? No; custom_vjp bwd
        # signature has only (res, g). So we must cache dy or fp slices.
        #
        # => Cache dy in res to avoid needing fp. Update _forward accordingly.
        raise RuntimeError("Internal error: make_interp_pt needs dy cached. "
                           "Use make_interp_pt_cached_dy below.")

    # We implement a corrected version that caches dy and x0/x1 relation.
    # Keep API stable by returning that fixed implementation.
    return make_interp_pt_cached_dy(eps=eps, side=side)


def make_interp_pt_cached_dy(*, eps: float = 1e-12, side: str = "right"):
    """
    Same as make_interp_pt, but caches dy=(y1-y0) and idx for backward.
    This avoids needing fp inside bwd (custom_vjp constraint).
    """
    side = str(side)
    eps = float(eps)

    def _forward(x, xp, fp):
        n = xp.shape[0]
        idx = jnp.searchsorted(xp, x, side=side)
        idx = jnp.clip(idx, 1, n - 1)

        x0 = xp[idx - 1]
        x1 = xp[idx]
        y0 = fp[idx - 1]
        y1 = fp[idx]

        denom = jnp.maximum(x1 - x0, eps)
        r = (x - x0) / denom
        y = (1.0 - r) * y0 + r * y1

        dy = (y1 - y0)
        # cache minimal info; idx is treated as constant in bwd
        return y, (idx, r, denom, dy, fp.shape)

    @jax.custom_vjp
    def interp_pt(x, xp, fp):
        y, _ = _forward(x, xp, fp)
        return y

    def fwd(x, xp, fp):
        y, res = _forward(x, xp, fp)
        return y, res

    def bwd(res, g):
        idx, r, denom, dy, fp_shape = res

        # dy/dx = (y1 - y0)/denom
        gx = g * dy / denom

        # grads wrt fp values at the two knots
        g_y0 = g * (1.0 - r)
        g_y1 = g * r

        gfp = jnp.zeros(fp_shape, dtype=g.dtype)
        gfp = gfp.at[idx - 1].add(g_y0)
        gfp = gfp.at[idx].add(g_y1)

        # No grad wrt xp (knot locations) for the forward-interp helper
        return (gx, None, gfp)

    interp_pt.defvjp(fwd, bwd)
    return interp_pt


def make_inv_interp_const_fp_wrt_xp(*, eps: float = 1e-12, side: str = "right"):
    """
    Inverse interpolation:
        y = interp(x; xp, fp_const)
    where fp_const is treated as constant (no gradient), BUT xp receives gradients.

    This matches your JAX inverse map for dL -> z with cosmology gradients
    flowing through dL_grid(theta) (xp).

    Semantics:
      idx = stop_grad(clip(searchsorted(xp, x, side), 1, n-1))
      denom = max(xh-xl, eps)
      r = (x-xl)/denom
      y = (1-r)*yl + r*yh

    Gradients returned:
      - wrt x (query)
      - wrt xp (knot positions)
      - fp_const: None (constant)
    """
    side = str(side)
    eps = float(eps)

    def _forward(x, xp, fp_const):
        n = xp.shape[0]
        idx = jnp.searchsorted(xp, x, side=side)
        idx = jnp.clip(idx, 1, n - 1)

        xl = xp[idx - 1]
        xh = xp[idx]
        yl = fp_const[idx - 1]
        yh = fp_const[idx]

        denom = jnp.maximum(xh - xl, eps)
        r = (x - xl) / denom
        y = (1.0 - r) * yl + r * yh

        # cache for bwd
        return y, (idx, x, xl, xh, yl, yh, denom, xp.shape)

    @jax.custom_vjp
    def inv_interp(x, xp, fp_const):
        y, _ = _forward(x, xp, fp_const)
        return y

    def fwd(x, xp, fp_const):
        y, res = _forward(x, xp, fp_const)
        return y, res

    def bwd(res, g):
        idx, x, xl, xh, yl, yh, denom, xp_shape = res

        dz = (yh - yl)

        # dy/dx = dz/denom
        gx = g * dz / denom

        # grads wrt xp knots (xl, xh)
        denom2 = denom * denom
        g_xl = g * dz * (x - xh) / denom2
        g_xh = g * dz * (xl - x) / denom2

        gxp = jnp.zeros(xp_shape, dtype=g.dtype)
        gxp = gxp.at[idx - 1].add(g_xl)
        gxp = gxp.at[idx].add(g_xh)

        # fp_const treated as constant
        return (gx, gxp, None)

    inv_interp.defvjp(fwd, bwd)
    return inv_interp
