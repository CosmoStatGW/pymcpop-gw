from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from numerical_utils import atinterp, attrapzvec

class NPBackend:
    floatX = np.float64

    # ---- arrays ----
    @staticmethod
    def asarray(x, dtype=None):
        return np.asarray(x, dtype=dtype)
    
    @staticmethod
    def atleast_1d(x, ):
        return np.atleast_1d(x)

    @staticmethod
    def shape(x):
        return x.shape

    @staticmethod
    def sign(x):
        return np.sign(x)

    @staticmethod
    def tile(x, reps):
        return np.tile(x, reps)

    @staticmethod
    def repeat(a, reps, axis=None):
        return np.repeat(a, reps, axis=axis)

    @staticmethod
    def interp(x, xp,fp, left=None, right=None, period=None):
        return np.interp(x, xp, fp, left=left, right=right, period=period )

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
    eye = staticmethod(np.eye)

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
    def any(x, axis=None, keepdims=False):
        return np.any(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def all(x, axis=None, keepdims=False):
        return np.all(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def zeros_like(x):
        return np.zeros_like(x)
    
    @staticmethod
    def ceil(x):
        return np.ceil(x)

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
    def square(x):
        return np.square(x)

    @staticmethod
    def pow(x1, x2):
        return np.pow(x1, x2)
        
    @staticmethod
    def squeeze(x):
        return np.squeeze(x)



    # ---- comparisons (elementwise) ----
    @staticmethod
    def gt(a, b):  # >
        return np.greater(a, b)

    @staticmethod
    def ge(a, b):  # >=
        return np.greater_equal(a, b)

    @staticmethod
    def lt(a, b):  # <
        return np.less(a, b)

    @staticmethod
    def le(a, b):  # <=
        return np.less_equal(a, b)

    @staticmethod
    def eq(a, b):  # ==
        return np.equal(a, b)

    @staticmethod
    def ne(a, b):  # !=
        return np.not_equal(a, b)

    # ---- logical (elementwise) ----
    @staticmethod
    def logical_and(a, b):
        return np.logical_and(a, b)

    @staticmethod
    def logical_or(a, b):
        return np.logical_or(a, b)

    @staticmethod
    def logical_not(a):
        return np.logical_not(a)

    # ---- control-flow style helpers ----
    @staticmethod
    def switch(cond, x, y):
        # your code uses at.switch semantics; np.where matches elementwise
        return np.where(cond, x, y)

    # ---- extra array ops commonly needed in your code ----
    @staticmethod
    def concatenate(xs, axis=0):
        return np.concatenate(xs, axis=axis)

    @staticmethod
    def stack(xs, axis=0):
        return np.stack(xs, axis=axis)

    @staticmethod
    def floor(x):
        return np.floor(x)

    @staticmethod
    def arctan2(y, x):
        return np.arctan2(y, x)

    @staticmethod
    def maximum_accumulate(x, axis=0):
        return np.maximum.accumulate(x, axis=axis)


    @staticmethod
    def interp(x, xp,fp, left=None, right=None, period=None):
        return np.interp(x, xp, fp, left=None, right=None, period=period )

    @staticmethod
    def trapezoid(y, x=None, dx=1.0, axis=-1):
        return np.trapezoid(y, x=x, dx=dx, axis=axis )

    logspace = staticmethod(np.logspace)
    log10 = staticmethod(np.log10)

    isfinite = staticmethod(np.isfinite)
    isnan = staticmethod(np.isnan)
    isinf = staticmethod(np.isinf)

    power = staticmethod(np.power)
    
    @staticmethod
    def finite_or(x, fill):
        return np.where(np.isfinite(x), x, fill)







class JAXBackend:
    # elementary
    exp = staticmethod(jnp.exp)
    log = staticmethod(jnp.log)
    log10 = staticmethod(jnp.log10)
    log1p = staticmethod(jnp.log1p)
    sqrt = staticmethod(jnp.sqrt)
    abs = staticmethod(jnp.abs)
    tanh = staticmethod(jnp.tanh)
    expm1 = staticmethod(jnp.expm1)
    square = staticmethod(jnp.square)
    erf = staticmethod(jax.scipy.special.erf)
    erfc = staticmethod(jax.scipy.special.erfc)
    power = staticmethod(jnp.power)
    
    @staticmethod
    def pow(x1, x2):
        return jnp.pow(x1, x2)

    # reductions / elementwise
    sum = staticmethod(jnp.sum)
    
    @staticmethod
    def max(x, axis=None, keepdims=False):
        return jnp.max(x, axis=axis, keepdims=keepdims)
        
    minimum = staticmethod(jnp.minimum)
    min = staticmethod(jnp.min)
    maximum = staticmethod(jnp.maximum)
    clip = staticmethod(jnp.clip)
    where = staticmethod(jnp.where)
    ceil = staticmethod(jnp.ceil)
    sign = staticmethod(jnp.sign)
    squeeze = staticmethod(jnp.squeeze)
    
    @staticmethod
    def searchsorted(a, v, side="left"):
        return jnp.searchsorted(a, v, side=side)

    # array creation / shape
    asarray = staticmethod(jnp.asarray)
    linspace = staticmethod(jnp.linspace)
    reshape = staticmethod(jnp.reshape)
    concatenate = staticmethod(jnp.concatenate)
    stack = staticmethod(jnp.stack)
    atleast_1d = staticmethod(jnp.atleast_1d)

    zeros_like = staticmethod(jnp.zeros_like)

    @staticmethod
    def tile(x, reps):
        return jnp.tile(x, reps)

    @staticmethod
    def shape(x):
        return x.shape

    @staticmethod
    def broadcast_to(x, shape):
        return jnp.broadcast_to(x,shape)

    @staticmethod
    def interp(x, xp,fp, left=None, right=None, period=None):
        return jnp.interp(x, xp, fp, left=left, right=right, period=period )

    @staticmethod
    def trapezoid(y, x=None, dx=1.0, axis=-1):
        return jnp.trapezoid(y, x=x, dx=dx, axis=axis )
    
    
    @staticmethod
    def repeat(a, reps, axis=None):
        return jnp.repeat(a, reps, axis=axis)
    
    @staticmethod
    def sort(x, axis=-1):
        return jnp.sort(x, axis=axis)

    @staticmethod
    def full_like(x, fill_value):
        return jnp.full_like(x, fill_value)

    # misc
    floor = staticmethod(jnp.floor)
    cumsum = staticmethod(jnp.cumsum)
    diff = staticmethod(jnp.diff)
    matmul = staticmethod(jnp.matmul)

    # logsumexp / logaddexp (often needed)
    logaddexp = staticmethod(jnp.logaddexp)
    logsumexp = staticmethod(jax.scipy.special.logsumexp)

    stop_grad = staticmethod(jax.lax.stop_gradient)

    logspace = staticmethod(jnp.logspace)


    # comparisons
    gt = staticmethod(jnp.greater)
    ge = staticmethod(jnp.greater_equal)
    lt = staticmethod(jnp.less)
    le = staticmethod(jnp.less_equal)
    eq = staticmethod(jnp.equal)
    ne = staticmethod(jnp.not_equal)
    nanmin = staticmethod(jnp.nanmin)
    nanmax = staticmethod(jnp.nanmax)
    isneginf = staticmethod(jnp.isneginf)
    finfo = staticmethod(jnp.finfo)

    # logical
    logical_and = staticmethod(jnp.logical_and)
    logical_or  = staticmethod(jnp.logical_or)
    logical_not = staticmethod(jnp.logical_not)

    # switch
    switch = staticmethod(jnp.where)

    # shape / creation
    zeros = staticmethod(jnp.zeros)
    ones  = staticmethod(jnp.ones)
    arange = staticmethod(jnp.arange)

    sigmoid = staticmethod(jax.nn.sigmoid)
    eye = staticmethod(jnp.eye)

    # needed by your grid helpers
    any = staticmethod(jnp.any)
    all = staticmethod(jnp.all)

    # monotone enforcement helper
    @staticmethod
    def maximum_accumulate(x, axis=0):
        # jnp.maximum.accumulate exists and is JIT-friendly
        return jnp.maximum.accumulate(x, axis=axis)


    isfinite = staticmethod(jnp.isfinite)
    isnan = staticmethod(jnp.isnan)
    isinf = staticmethod(jnp.isinf)
    
    @staticmethod
    def finite_or(x, fill):
        return jnp.where(jnp.isfinite(x), x, fill)


    @staticmethod
    def take_along_axis(arr, indices, axis):
        return jnp.take_along_axis(arr, indices, axis=axis)