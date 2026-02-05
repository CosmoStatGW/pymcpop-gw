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
# Fused interpolation helper (NumPy; used inside perform())

def _interp1d_from_sorted_x(x, xg, yg, *, clip=True):
    """
    Vectorized linear interpolation y(x) with xg sorted ascending.
    x: (...,)
    xg: (M,)
    yg: (M,) or (M, K)
    returns: same shape as x plus optional trailing K
    """
    x = np.asarray(x)
    xg = np.asarray(xg)
    yg = np.asarray(yg)

    idx = np.searchsorted(xg, x, side="right") - 1
    if clip:
        idx = np.clip(idx, 0, xg.size - 2)

    x0 = xg[idx]
    x1 = xg[idx + 1]
    t = (x - x0) / (x1 - x0)

    y0 = np.take(yg, idx, axis=0)
    y1 = np.take(yg, idx + 1, axis=0)
    return y0 + (y1 - y0) * t



# caches
_INV_INTERP_CONSTY_CACHE = {}
_FWD_INTERP_CONSTX_CACHE = {}

def _as_float64(x):
    return np.asarray(x, dtype=np.float64)

# -----------------------------
# Case A: ys is constant (your zgrid)
# y = interp(x, xs, ys_const)
# -----------------------------

class _InterpConstYGradOp(Op):
    """
    NumPy VJP for y = interp(x, xs, ys_const).
    Inputs: x, xs, g_out
    Outputs: dx, gxs
    """
    def __init__(self, ys_const, eps=1e-12, side="right"):
        self.ys = _as_float64(ys_const)
        self.eps = float(eps)
        self.side = side

    def make_node(self, x, xs, g_out):
        x = at.as_tensor_variable(x)
        xs = at.as_tensor_variable(xs)
        g_out = at.as_tensor_variable(g_out)
        return Apply(self, [x, xs, g_out], [x.type(), xs.type()])

    def perform(self, node, inputs, outputs):
        x, xs, g = inputs
        ys = self.ys
        eps = self.eps
        side = self.side

        x_arr = np.asarray(x)
        xs_arr = np.asarray(xs)
        g_arr = np.asarray(g)

        x_flat = x_arr.ravel()
        g_flat = g_arr.ravel()

        idx = np.searchsorted(xs_arr, x_flat, side=side)
        idx = np.clip(idx, 1, xs_arr.shape[0] - 1)

        xl = xs_arr[idx - 1]
        xh = xs_arr[idx]
        yl = ys[idx - 1]
        yh = ys[idx]

        denom = np.maximum(xh - xl, eps)
        dy_dx = (yh - yl) / denom

        # dx
        dx = (g_flat * dy_dx).reshape(x_arr.shape)

        # gxs scatter (only hits idx-1 and idx)
        # y = (1-r) yl + r yh, r=(x-xl)/(xh-xl)
        # ∂y/∂xl = (yh-yl) * (x - xh) / (xh-xl)^2
        # ∂y/∂xh = (yh-yl) * (xl - x) / (xh-xl)^2
        coeff = g_flat * (yh - yl) / (denom * denom)
        g_xl = coeff * (x_flat - xh)
        g_xh = coeff * (xl - x_flat)

        gxs = np.zeros_like(xs_arr, dtype=xs_arr.dtype)
        np.add.at(gxs, idx - 1, g_xl)
        np.add.at(gxs, idx,     g_xh)

        outputs[0][0] = dx
        outputs[1][0] = gxs


class _InterpConstYOp(Op):
    """
    Forward Op for y = interp(x, xs, ys_const) with VJP in NumPy.
    """
    def __init__(self, ys_const, eps=1e-12, side="right"):
        self.ys = _as_float64(ys_const)
        self.eps = float(eps)
        self.side = side
        self._grad_op = _InterpConstYGradOp(self.ys, eps=self.eps, side=self.side)

    def make_node(self, x, xs):
        x = at.as_tensor_variable(x)
        xs = at.as_tensor_variable(xs)
        return Apply(self, [x, xs], [x.type()])

    def perform(self, node, inputs, outputs):
        x, xs = inputs
        ys = self.ys
        eps = self.eps
        side = self.side

        x_arr = np.asarray(x)
        xs_arr = np.asarray(xs)

        x_flat = x_arr.ravel()
        idx = np.searchsorted(xs_arr, x_flat, side=side)
        idx = np.clip(idx, 1, xs_arr.shape[0] - 1)

        xl = xs_arr[idx - 1]
        xh = xs_arr[idx]
        yl = ys[idx - 1]
        yh = ys[idx]

        denom = np.maximum(xh - xl, eps)
        r = (x_flat - xl) / denom
        y_flat = (1.0 - r) * yl + r * yh

        outputs[0][0] = y_flat.reshape(x_arr.shape)

    def grad(self, inputs, gout):
        x, xs = inputs
        (g,) = gout
        dx, gxs = self._grad_op(x, xs, g)
        return [dx, gxs]


# -----------------------------
# Case B: xs is constant (your zgrid)
# y = interp(x, xs_const, ys)
# -----------------------------

class _InterpConstXGradOp(Op):
    """
    NumPy VJP for y = interp(x, xs_const, ys).
    Inputs: x, ys, g_out
    Outputs: dx, dys
    """
    def __init__(self, xs_const, eps=1e-12, side="right"):
        self.xs = _as_float64(xs_const)
        self.eps = float(eps)
        self.side = side

    def make_node(self, x, ys, g_out):
        x = at.as_tensor_variable(x)
        ys = at.as_tensor_variable(ys)
        g_out = at.as_tensor_variable(g_out)
        return Apply(self, [x, ys, g_out], [x.type(), ys.type()])

    def perform(self, node, inputs, outputs):
        x, ys, g = inputs
        xs = self.xs
        eps = self.eps
        side = self.side

        x_arr = np.asarray(x)
        ys_arr = np.asarray(ys)
        g_arr = np.asarray(g)

        x_flat = x_arr.ravel()
        g_flat = g_arr.ravel()

        idx = np.searchsorted(xs, x_flat, side=side)
        idx = np.clip(idx, 1, xs.shape[0] - 1)

        xl = xs[idx - 1]
        xh = xs[idx]
        yl = ys_arr[idx - 1]
        yh = ys_arr[idx]

        denom = np.maximum(xh - xl, eps)
        r = (x_flat - xl) / denom

        # dx = g * (yh-yl)/(xh-xl)
        dx = (g_flat * (yh - yl) / denom).reshape(x_arr.shape)

        # grads wrt ys: y = (1-r) yl + r yh
        dys = np.zeros_like(ys_arr, dtype=ys_arr.dtype)
        np.add.at(dys, idx - 1, g_flat * (1.0 - r))
        np.add.at(dys, idx,     g_flat * r)

        outputs[0][0] = dx
        outputs[1][0] = dys


class _InterpConstXOp(Op):
    """
    Forward Op for y = interp(x, xs_const, ys) with VJP in NumPy.
    """
    def __init__(self, xs_const, eps=1e-12, side="right"):
        self.xs = _as_float64(xs_const)
        self.eps = float(eps)
        self.side = side
        self._grad_op = _InterpConstXGradOp(self.xs, eps=self.eps, side=self.side)

    def make_node(self, x, ys):
        x = at.as_tensor_variable(x)
        ys = at.as_tensor_variable(ys)
        return Apply(self, [x, ys], [x.type()])

    def perform(self, node, inputs, outputs):
        x, ys = inputs
        xs = self.xs
        eps = self.eps
        side = self.side

        x_arr = np.asarray(x)
        ys_arr = np.asarray(ys)

        x_flat = x_arr.ravel()
        idx = np.searchsorted(xs, x_flat, side=side)
        idx = np.clip(idx, 1, xs.shape[0] - 1)

        xl = xs[idx - 1]
        xh = xs[idx]
        yl = ys_arr[idx - 1]
        yh = ys_arr[idx]

        denom = np.maximum(xh - xl, eps)
        r = (x_flat - xl) / denom
        y_flat = (1.0 - r) * yl + r * yh

        outputs[0][0] = y_flat.reshape(x_arr.shape)

    def grad(self, inputs, gout):
        x, ys = inputs
        (g,) = gout
        dx, dys = self._grad_op(x, ys, g)
        return [dx, dys]


# -----------------------------
# Public helper: backend-agnostic dispatcher
# -----------------------------

def atinterp(x, xs, ys, eps=1e-12, side="right"):
    """
    Differentiable interpolation without symbolic SearchsortedOp when either grid is constant.

    Dispatch rules:
      - if ys is TensorConstant -> use Const-Y Op: interp(x, xs, ys_const)
      - elif xs is TensorConstant -> use Const-X Op: interp(x, xs_const, ys)
      - else fallback (symbolic searchsorted; avoid if you care about speed)
    """
    xs_var = at.as_tensor_variable(xs)
    ys_var = at.as_tensor_variable(ys)

    if isinstance(ys_var, at.TensorConstant):
        key = (ys_var.data.tobytes(), float(eps), side)
        op = _INV_INTERP_CONSTY_CACHE.get(key)
        if op is None:
            op = _InterpConstYOp(ys_const=ys_var.data, eps=eps, side=side)
            _INV_INTERP_CONSTY_CACHE[key] = op
        return op(x, xs)

    if isinstance(xs_var, at.TensorConstant):
        key = (xs_var.data.tobytes(), float(eps), side)
        op = _FWD_INTERP_CONSTX_CACHE.get(key)
        if op is None:
            op = _InterpConstXOp(xs_const=xs_var.data, eps=eps, side=side)
            _FWD_INTERP_CONSTX_CACHE[key] = op
        return op(x, ys)

    # fallback (try hard not to hit this in your model)
    idxs = at.searchsorted(xs, x, side=side)
    idxs = at.clip(idxs, 1, xs.shape[0] - 1)
    xl = xs[idxs - 1]
    xh = xs[idxs]
    yl = ys[idxs - 1]
    yh = ys[idxs]
    denom = at.maximum(xh - xl, eps)
    r = (x - xl) / denom
    return (1 - r) * yl + r * yh


    
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