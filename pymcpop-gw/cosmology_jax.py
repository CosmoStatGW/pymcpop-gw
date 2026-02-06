# cosmology_jax.py
from __future__ import annotations

import jax.numpy as jnp

import jax_utils
import jax
from pytensor_utils import atinterp



def make_z_from_dL_interp(bk, *, eps=1e-12, side="right", param="vanilla"):
    

    def z_from_dL(dL, theta5, zgrid, dL_grid, x01, w01):
        return atinterp(bk, dL, dL_grid, zgrid, eps=eps, side=side)

    return z_from_dL



def make_z_from_dL_interp_long(bk, *, eps=1e-12, side="right", param="vanilla"):
    """
    JAX-only inverse map dL -> z that matches your PyTensor 'standard' atinterp semantics.

    Signature preserved (matches your existing call sites):
        z_from_dL(dL, theta5, zgrid, dL_grid, x01, w01) -> z

    Additionally attaches:
        z_from_dL.interp_pt(x, xp, fp) -> y
    which is the matching forward interpolation helper you should use
    for dc(z) / log_ddL_dz(z) lookups to keep semantics identical.

    Notes:
      - bk/param are accepted for API compatibility; not used in pure-interp version.
      - No Newton steps; purely interpolation-based.
      - Gradients wrt cosmology parameters flow through dL_grid (xp) because the inverse
        returns gradients wrt xp.
    """
    eps = float(eps)
    side = str(side)

    # forward helper (query x + fp grads; no xp grads)
    interp_pt = jax_utils.make_interp_pt_cached_dy(eps=eps, side=side)

    # inverse helper (query x grads + xp grads; fp_const treated constant)
    inv_interp = jax_utils.make_inv_interp_const_fp_wrt_xp(eps=eps, side=side)

    def z_from_dL(dL, theta5, zgrid, dL_grid, x01, w01):
        # y = interp(dL; xp=dL_grid, fp=zgrid), with xp grads enabled.
        return inv_interp(dL, dL_grid, zgrid)

    # attach helper for dc/log_ddL_dz interpolation on zgrid
    z_from_dL.interp_pt = interp_pt
    return z_from_dL




# def make_z_from_dL_interp(bk, *, eps=1e-12, side="right", param="vanilla"):
#     """
#     EXACT PyTensor 'standard' atinterp semantics, but for BOTH:
#       (A) inverse map  dL -> z   (xp=dL_grid, fp=zgrid)
#       (B) forward map  z  -> f(z) (xp=zgrid, fp=fp_grid) used later for dc/log_ddL_dz

#     IMPORTANT:
#       - This function keeps the SAME signature you already use:
#             z_from_dL(dL, theta5, zgrid, dL_grid, x01, w01) -> z
#       - Additionally, it attaches a helper:
#             z_from_dL.interp_pt(x, xp, fp) -> y
#         which you should use instead of (_interp_prepare_jax/_interp_apply_jax).

#     PyTensor semantics matched:
#       idx = stop_grad(clip(searchsorted(xp, x, side), 1, n-1))
#       denom = max(xh-xl, eps)
#       r = (x-xl)/denom
#       y = (1-r)*yl + r*yh

#     Gradients:
#       - No gradient through the discrete interval choice (searchsorted / idx)
#       - Piecewise-linear gradients inside the chosen interval
#       - For inverse: gradients w.r.t. dL (query) and dL_grid (knot positions)
#       - For forward interp: gradients w.r.t. x (query) and fp (values);
#         (xp gradients are returned as None, matching the usual PyTensor case where xp is constant)
#     """
#     side = str(side)

#     # ------------------------------------------------------------------
#     # Generic forward interpolation with PyTensor semantics:
#     #   y = interp(x; xp, fp), with idx clipped (no clamp of x)
#     #   and stop-grad through idx.
#     # ------------------------------------------------------------------
#     def _interp_forward(x, xp, fp):
#         n = xp.shape[0]
#         idx = jnp.searchsorted(xp, x, side=side)
#         idx = jnp.clip(idx, 1, n - 1)

#         x0 = xp[idx - 1]
#         x1 = xp[idx]
#         y0 = fp[idx - 1]
#         y1 = fp[idx]

#         denom = jnp.maximum(x1 - x0, eps)
#         r = (x - x0) / denom
#         y = (1.0 - r) * y0 + r * y1

#         # cache; idx is treated as constant in backward (PyTensor stop_grad)
#         return y, (idx, x, x0, x1, y0, y1, denom, r, fp.shape)

#     @jax.custom_vjp
#     def interp_pt(x, xp, fp):
#         y, _ = _interp_forward(x, xp, fp)
#         return y

#     def interp_pt_fwd(x, xp, fp):
#         y, res = _interp_forward(x, xp, fp)
#         return y, res

#     def interp_pt_bwd(res, g):
#         idx, x, x0, x1, y0, y1, denom, r, fp_shape = res

#         dy = (y1 - y0)

#         # dy/dx = (y1 - y0)/denom
#         gx = g * dy / denom

#         # grads wrt fp (values at knots)
#         g_y0 = g * (1.0 - r)
#         g_y1 = g * r

#         gfp = jnp.zeros(fp_shape, dtype=g.dtype)
#         gfp = gfp.at[idx - 1].add(g_y0)
#         gfp = gfp.at[idx].add(g_y1)

#         # No grad wrt xp (interval locations) for forward interp helper
#         gxp = None
#         return (gx, gxp, gfp)

#     interp_pt.defvjp(interp_pt_fwd, interp_pt_bwd)

#     # ------------------------------------------------------------------
#     # Your inverse: z = interp(dL; xp=dL_grid, fp=zgrid)
#     # BUT with gradient also wrt dL_grid (knot positions) because dL_grid(theta).
#     # This matches your existing implementation, with PyTensor semantics.
#     # ------------------------------------------------------------------
#     def _inv_forward(dL, dL_grid, zgrid):
#         n = dL_grid.shape[0]
#         idx = jnp.searchsorted(dL_grid, dL, side=side)
#         idx = jnp.clip(idx, 1, n - 1)

#         dl_lo = dL_grid[idx - 1]
#         dl_hi = dL_grid[idx]
#         z_lo  = zgrid[idx - 1]
#         z_hi  = zgrid[idx]

#         denom = jnp.maximum(dl_hi - dl_lo, eps)
#         r = (dL - dl_lo) / denom
#         z = (1.0 - r) * z_lo + r * z_hi

#         return z, (idx, dL, dl_lo, dl_hi, z_lo, z_hi, denom, dL_grid.shape)

#     @jax.custom_vjp
#     def z_from_dL(dL, theta5, zgrid, dL_grid, x01, w01):
#         z, _ = _inv_forward(dL, dL_grid, zgrid)
#         return z

#     def z_from_dL_fwd(dL, theta5, zgrid, dL_grid, x01, w01):
#         z, res = _inv_forward(dL, dL_grid, zgrid)
#         return z, res

#     def z_from_dL_bwd(res, g_z):
#         idx, dL, dl_lo, dl_hi, z_lo, z_hi, denom, dL_grid_shape = res

#         dz = (z_hi - z_lo)

#         # dz/ddL = dz/denom
#         g_dL = g_z * dz / denom

#         # grads wrt knot positions (dl_lo, dl_hi)
#         denom2 = denom * denom
#         g_dl_lo = g_z * dz * (dL - dl_hi) / denom2
#         g_dl_hi = g_z * dz * (dl_lo - dL) / denom2

#         g_dL_grid = jnp.zeros(dL_grid_shape, dtype=g_z.dtype)
#         g_dL_grid = g_dL_grid.at[idx - 1].add(g_dl_lo)
#         g_dL_grid = g_dL_grid.at[idx].add(g_dl_hi)

#         # grads for (dL, theta5, zgrid, dL_grid, x01, w01)
#         return (g_dL, None, None, g_dL_grid, None, None)

#     z_from_dL.defvjp(z_from_dL_fwd, z_from_dL_bwd)

#     # expose helper so you can do:
#     #   dc_evt = z_from_dL.interp_pt(z_evt, zgrid, dc_grid)
#     #   log_dd_evt = z_from_dL.interp_pt(z_evt, zgrid, log_ddL_dz_grid)
#     z_from_dL.interp_pt = interp_pt

#     return z_from_dL