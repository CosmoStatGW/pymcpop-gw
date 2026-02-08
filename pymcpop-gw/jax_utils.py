# jax_utils.py
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _interp_prepare_bk(bk, x, xp, eps=1e-12, side="right"):
    idx = bk.searchsorted( xp, x, side=side)
    idx = bk.clip(idx, 1, xp.shape[0] - 1) #bk.stop_grad( bk.clip(idx, 1, xp.shape[0] - 1) )
    x0 = xp[idx - 1]
    x1 = xp[idx]
    denom = bk.maximum(x1 - x0, eps)
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


def make_interp_pt_like_multiY(eps=1e-12, side="right"):
    eps = float(eps)
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    @jax.custom_vjp
    def interp_multiY(x, xg, Y):
        # x: (...), xg: (N,), Y: (K,N)
        x  = jnp.asarray(x)
        xg = jnp.asarray(xg)
        Y  = jnp.asarray(Y)

        x_flat = jnp.ravel(x)
        x_clip = jnp.clip(x_flat, xg[0], xg[-1])

        j = jnp.searchsorted(xg, x_clip, side=side)
        j = jnp.clip(j, 1, xg.shape[0] - 1)

        xL = xg[j - 1]
        xR = xg[j]
        denom = jnp.maximum(xR - xL, eps)
        r = (x_clip - xL) / denom
        r = jnp.clip(r, 0.0, 1.0)

        # gather Y at left/right: (K, M)
        YL = Y[:, j - 1]
        YR = Y[:, j]

        out_flat = (1.0 - r)[None, :] * YL + r[None, :] * YR  # (K,M)
        out = jnp.reshape(out_flat, (Y.shape[0],) + x.shape)   # (K, ...)

        return out

    def fwd(x, xg, Y):
        x  = jnp.asarray(x)
        xg = jnp.asarray(xg)
        Y  = jnp.asarray(Y)

        x_flat = jnp.ravel(x)
        x_clip = jnp.clip(x_flat, xg[0], xg[-1])

        j = jnp.searchsorted(xg, x_clip, side=side)
        j = jnp.clip(j, 1, xg.shape[0] - 1)

        xL = xg[j - 1]
        xR = xg[j]
        denom = jnp.maximum(xR - xL, eps)
        r = (x_clip - xL) / denom
        r = jnp.clip(r, 0.0, 1.0)

        # slopes per K: (K,M)
        YL = Y[:, j - 1]
        YR = Y[:, j]
        slopes = (YR - YL) / denom[None, :]

        out_flat = (1.0 - r)[None, :] * YL + r[None, :] * YR
        out = jnp.reshape(out_flat, (Y.shape[0],) + x.shape)

        # stash everything needed
        return out, (x.shape, j, r, slopes, xg, Y.shape[1])

    def bwd(res, g_out):
        x_shape, j, r, slopes, xg, N = res

        # g_out: (K, ...) -> (K,M)
        g_flat = jnp.reshape(g_out, (g_out.shape[0], -1))

        # dx: sum_k g_k * slope_k  -> (M,)
        dx_flat = jnp.sum(g_flat * slopes, axis=0)
        dx = jnp.reshape(dx_flat, x_shape)

        # dxg forced zero
        dxg = jnp.zeros_like(xg, dtype=g_out.dtype)

        # dY: (K,N)
        dY = jnp.zeros((g_flat.shape[0], N), dtype=g_out.dtype)
        dY = dY.at[:, j - 1].add(g_flat * (1.0 - r)[None, :])
        dY = dY.at[:, j].add(g_flat * r[None, :])

        return (dx, dxg, dY)

    interp_multiY.defvjp(fwd, bwd)
    return interp_multiY



def make_interp_pt_like(eps=1e-12, side="right"):
    eps = float(eps)
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    @jax.custom_vjp
    def interp(x, xg, yg):
        # x: (...), xg: (N,), yg: (N,)
        x = jnp.asarray(x)
        xg = jnp.asarray(xg)
        yg = jnp.asarray(yg)

        x_flat = jnp.ravel(x)
        x_clip = jnp.clip(x_flat, xg[0], xg[-1])

        j = jnp.searchsorted(xg, x_clip, side=side)
        j = jnp.clip(j, 1, xg.shape[0] - 1)

        xL = xg[j - 1]
        xR = xg[j]
        denom = jnp.maximum(xR - xL, eps)
        r = (x_clip - xL) / denom
        r = jnp.clip(r, 0.0, 1.0)

        yL = yg[j - 1]
        yR = yg[j]
        out_flat = (1.0 - r) * yL + r * yR
        return jnp.reshape(out_flat, x.shape)

    def fwd(x, xg, yg):
        # stash what we need for VJP
        x = jnp.asarray(x)
        xg = jnp.asarray(xg)
        yg = jnp.asarray(yg)

        x_flat = jnp.ravel(x)
        x_clip = jnp.clip(x_flat, xg[0], xg[-1])

        j = jnp.searchsorted(xg, x_clip, side=side)
        j = jnp.clip(j, 1, xg.shape[0] - 1)

        xL = xg[j - 1]
        xR = xg[j]
        denom = jnp.maximum(xR - xL, eps)
        r = (x_clip - xL) / denom
        r = jnp.clip(r, 0.0, 1.0)

        yL = yg[j - 1]
        yR = yg[j]
        out_flat = (1.0 - r) * yL + r * yR
        out = jnp.reshape(out_flat, x.shape)

        # slopes for dx
        slopes = (yR - yL) / denom  # (M,)
        return out, (x.shape, j, r, slopes)

    def bwd(res, g_out):
        x_shape, j, r, slopes = res
        g_flat = jnp.ravel(g_out)

        # dx: sum g * slope
        dx_flat = g_flat * slopes
        dx = jnp.reshape(dx_flat, x_shape)

        # dxg: forced to zero (PT behavior)
        # Note: return a zeros array with same shape/dtype as xg
        # We cannot reconstruct xg here, so return None and let caller stop-grad xg,
        # OR pass xg shape in res. We'll do the robust way: pass xg shape via closure
        # by capturing in fwd would be too heavy. So simplest: return a "shape-only" zero
        # by requiring xg to be treated as nondiff arg in your usage OR stop_grad(xg).
        # Better: include xg in res:
        raise RuntimeError("Use the safer 4-tuple version below that stashes xg.")

    # safer: include xg and yg for correct zero and scatter
    def fwd2(x, xg, yg):
        out, (x_shape, j, r, slopes) = fwd(x, xg, yg)
        return out, (x_shape, j, r, slopes, xg, yg)

    def bwd2(res, g_out):
        x_shape, j, r, slopes, xg, yg = res
        g_flat = jnp.ravel(g_out)

        dx_flat = g_flat * slopes
        dx = jnp.reshape(dx_flat, x_shape)

        # dxg forced zero
        dxg = jnp.zeros_like(xg, dtype=g_out.dtype)

        # dyg: scatter add like np.add.at
        N = yg.shape[0]
        dyg = jnp.zeros_like(yg, dtype=g_out.dtype)
        dyg = dyg.at[j - 1].add(g_flat * (1.0 - r))
        dyg = dyg.at[j].add(g_flat * r)

        return (dx, dxg, dyg)

    interp.defvjp(fwd2, bwd2)
    return interp


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
        dx = (xh - xl)
    
        # unclamped mask
        unclamped = dx > eps
    
        # dy/dx: dz/denom works in both regimes (since denom is the actual used value)
        gx = g * dz / denom
    
        # If unclamped:
        #   dy/dxl = dz*(x - xh)/dx^2
        #   dy/dxh = dz*(xl - x)/dx^2
        # If clamped (denom = eps const):
        #   y = yl + (x - xl)/eps * dz
        #   dy/dxl = -dz/eps
        #   dy/dxh = 0
        dx2 = dx * dx
        g_xl_uncl = g * dz * (x - xh) / dx2
        g_xh_uncl = g * dz * (xl - x) / dx2
    
        g_xl_clmp = g * (-dz) / eps
        g_xh_clmp = jnp.zeros_like(g_xh_uncl)
    
        g_xl = jnp.where(unclamped, g_xl_uncl, g_xl_clmp)
        g_xh = jnp.where(unclamped, g_xh_uncl, g_xh_clmp)
    
        gxp = jnp.zeros(xp_shape, dtype=g.dtype)
        gxp = gxp.at[idx - 1].add(g_xl)
        gxp = gxp.at[idx].add(g_xh)
    
        return (gx, gxp, None)

    inv_interp.defvjp(fwd, bwd)
    return inv_interp
