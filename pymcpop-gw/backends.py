from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp



class NPBackend:
    floatX = np.float64

    # ---- arrays ----
    @staticmethod
    def asarray(x, dtype=None):
        return np.asarray(x, dtype=dtype)

    @staticmethod
    def constant(x, dtype=None):
        return np.asarray(x, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(shape, dtype=None):
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def arange(*args, dtype=None):
        return np.arange(*args, dtype=dtype)

    @staticmethod
    def linspace(start, stop, num, dtype=None):
        out = np.linspace(start, stop, int(num))
        return out.astype(dtype) if dtype is not None else out

    # ---- math ----
    exp = staticmethod(np.exp)
    log = staticmethod(np.log)
    log1p = staticmethod(np.log1p)
    sqrt = staticmethod(np.sqrt)
    abs = staticmethod(np.abs)
    tanh = staticmethod(np.tanh)

    @staticmethod
    def erf(x):
        # numpy may not expose erf depending on version; scipy.special.erf would be fallback
        if hasattr(np, "erf"):
            return np.erf(x)
        raise AttributeError("NumPy has no erf; use scipy.special.erf or add a fallback here.")

    @staticmethod
    def maximum(a, b):
        return np.maximum(a, b)

    @staticmethod
    def minimum(a, b):
        return np.minimum(a, b)

    @staticmethod
    def clip(x, a_min, a_max):
        return np.clip(x, a_min, a_max)

    @staticmethod
    def where(cond, x, y):
        return np.where(cond, x, y)

    # ---- reductions ----
    @staticmethod
    def sum(x, axis=None, keepdims=False):
        return np.sum(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def max(x, axis=None, keepdims=False):
        return np.max(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def any(x, axis=None, keepdims=False):
        return np.any(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def all(x, axis=None, keepdims=False):
        return np.all(x, axis=axis, keepdims=keepdims)

    # ---- utils ----
    @staticmethod
    def stop_grad(x):
        return x

    @staticmethod
    def searchsorted(a, v, side="left"):
        return np.searchsorted(a, v, side=side)

    @staticmethod
    def take(a, idx, axis=0):
        return np.take(a, idx, axis=axis)

    @staticmethod
    def sort(x, axis=-1):
        return np.sort(x, axis=axis)

    @staticmethod
    def diff(x, n=1, axis=-1):
        return np.diff(x, n=n, axis=axis)

    @staticmethod
    def cumsum(x, axis=None):
        return np.cumsum(x, axis=axis)

    @staticmethod
    def reshape(x, shape):
        return np.reshape(x, shape)

    @staticmethod
    def expm1(x):
        return np.expm1(x)

    @staticmethod
    def squeeze(x):
        return np.squeeze(x)

    @staticmethod
    def logsumexp(x, axis=None, keepdims=False):
        m = np.max(x, axis=axis, keepdims=True)
        s = np.sum(np.exp(x - m), axis=axis, keepdims=True)
        out = np.log(s) + m
        if not keepdims and axis is not None:
            out = np.squeeze(out, axis=axis)
        return out


class ATBackend:
    """
    Lazy-import PyTensor so importing backends.py doesn't force pytensor import
    in contexts where you only want NumPy.
    """

    floatX = "float64"

    @staticmethod
    def _at():
        import pytensor.tensor as at
        return at

    # ---- arrays ----
    @staticmethod
    def asarray(x, dtype=None):
        at = ATBackend._at()
        out = at.as_tensor_variable(x)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    @staticmethod
    def constant(x, dtype=None):
        at = ATBackend._at()
        x = np.asarray(x)
        if dtype is not None:
            x = x.astype(dtype)
        return at.constant(x)

    @staticmethod
    def zeros(shape, dtype=None):
        at = ATBackend._at()
        return at.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(shape, dtype=None):
        at = ATBackend._at()
        return at.ones(shape, dtype=dtype)

    @staticmethod
    def arange(*args, dtype=None):
        at = ATBackend._at()
        return at.arange(*args, dtype=dtype)

    @staticmethod
    def linspace(start, stop, num, dtype=None):
        at = ATBackend._at()
        out = at.linspace(start, stop, num)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    # ---- math ----
    @staticmethod
    def exp(x):
        at = ATBackend._at()
        return at.exp(x)

    @staticmethod
    def log(x):
        at = ATBackend._at()
        return at.log(x)

    @staticmethod
    def log1p(x):
        at = ATBackend._at()
        return at.log1p(x)

    @staticmethod
    def sqrt(x):
        at = ATBackend._at()
        return at.sqrt(x)

    @staticmethod
    def abs(x):
        at = ATBackend._at()
        return at.abs(x)

    @staticmethod
    def tanh(x):
        at = ATBackend._at()
        return at.tanh(x)

    @staticmethod
    def erf(x):
        from pytensor.tensor import special as tspecial
        return tspecial.erf(x)

    @staticmethod
    def maximum(a, b):
        at = ATBackend._at()
        return at.maximum(a, b)

    @staticmethod
    def minimum(a, b):
        at = ATBackend._at()
        return at.minimum(a, b)

    @staticmethod
    def clip(x, a_min, a_max):
        at = ATBackend._at()
        return at.clip(x, a_min, a_max)

    @staticmethod
    def where(cond, x, y):
        at = ATBackend._at()
        return at.where(cond, x, y)

    # ---- reductions ----
    @staticmethod
    def sum(x, axis=None, keepdims=False):
        at = ATBackend._at()
        return at.sum(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def max(x, axis=None, keepdims=False):
        at = ATBackend._at()
        return at.max(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def any(x, axis=None, keepdims=False):
        at = ATBackend._at()
        return at.any(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def all(x, axis=None, keepdims=False):
        at = ATBackend._at()
        return at.all(x, axis=axis, keepdims=keepdims)

    # ---- utils ----
    @staticmethod
    def stop_grad(x):
        from pytensor.gradient import disconnected_grad
        return disconnected_grad(x)

    @staticmethod
    def searchsorted(a, v, side="left"):
        at = ATBackend._at()
        return at.searchsorted(a, v, side=side)

    @staticmethod
    def take(a, idx, axis=0):
        at = ATBackend._at()
        return at.take(a, idx, axis=axis)

    @staticmethod
    def sort(x, axis=-1):
        at = ATBackend._at()
        return at.sort(x, axis=axis)

    @staticmethod
    def diff(x, n=1, axis=-1):
        at = ATBackend._at()
        return at.diff(x, n=n, axis=axis)

    @staticmethod
    def cumsum(x, axis=None):
        at = ATBackend._at()
        return at.cumsum(x, axis=axis)

    @staticmethod
    def reshape(x, shape):
        at = ATBackend._at()
        return at.reshape(x, shape)

    @staticmethod
    def expm1(x):
        at = ATBackend._at()
        return at.expm1(x)

    @staticmethod
    def squeeze(x):
        at = ATBackend._at()
        return at.squeeze(x)


    @staticmethod
    def logsumexp(x, axis=None, keepdims=False):
        from pytensor.tensor import special as tspecial
        return tspecial.logsumexp(x, axis=axis, keepdims=keepdims)




class JAXBackend:
    # elementary
    exp = staticmethod(jnp.exp)
    log = staticmethod(jnp.log)
    log1p = staticmethod(jnp.log1p)
    sqrt = staticmethod(jnp.sqrt)
    abs = staticmethod(jnp.abs)
    tanh = staticmethod(jnp.tanh)
    expm1 = staticmethod(jnp.expm1)

    erf = staticmethod(jax.scipy.special.erf)
    erfc = staticmethod(jax.scipy.special.erfc)

    # reductions / elementwise
    sum = staticmethod(jnp.sum)
    max = staticmethod(jnp.max)
    minimum = staticmethod(jnp.minimum)
    maximum = staticmethod(jnp.maximum)
    clip = staticmethod(jnp.clip)
    where = staticmethod(jnp.where)

    # array creation / shape
    asarray = staticmethod(jnp.asarray)
    linspace = staticmethod(jnp.linspace)
    reshape = staticmethod(jnp.reshape)
    concatenate = staticmethod(jnp.concatenate)
    stack = staticmethod(jnp.stack)

    # misc
    floor = staticmethod(jnp.floor)
    cumsum = staticmethod(jnp.cumsum)
    diff = staticmethod(jnp.diff)
    matmul = staticmethod(jnp.matmul)

    # logsumexp / logaddexp (often needed)
    logaddexp = staticmethod(jnp.logaddexp)

    stop_grad = staticmethod(jax.lax.stop_gradient)

