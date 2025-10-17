#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import pytensor.tensor as at
import jax.numpy as np
import pymc as pm
import jax
from pytensor.graph import Apply, Op
import pytensor
import numpy as onp

from jax.numpy import array
from jax.numpy import concatenate
from jax.numpy import ones
from jax.numpy import zeros


import pade_cosmo as pc

p, q = pc.flat_wcdm_pade_coefficients(w0=-1.0, zpower=0)  # arrays of floats



c_light = 299792458*1e-03
c_light_at = at.as_tensor_variable(c_light)
NINF = at.as_tensor_variable(-np.inf)  
INF = at.as_tensor_variable(np.inf)



# EPS32 = at.as_tensor_variable(1e-30)  #  use 1e-30 for float32
# BIG32 = at.as_tensor_variable(1e20) 

MIN = -np.inf #NINF # your "effectively -inf" : NINF or EPS
MAX = np.inf #INF


 
try:
        zGridGlobals_at = at.sort(at.unique(at.concatenate([ at.logspace(start=-100, stop=-15, base=10, steps=50), at.logspace(start=-30, stop=-4, base=10, steps=100), 
                     #at.linspace(start=1.1e-03, end=10, steps=50),
                     at.logspace(start=-4, stop=1, base=10, steps=1000), 
                     at.logspace(start=1, stop=2, base=10, steps=100), at.logspace(start=2, stop=5, base=10, steps=50) ])))

except:
    
    zGridGlobals_at = at.sort(at.unique(at.concatenate([ at.logspace(start=-100, end=-15, base=10, steps=50), at.logspace(start=-30, end=-4, base=10, steps=100), 
                     #at.linspace(start=1.1e-03, end=10, steps=50),
                     at.logspace(start=-4, end=1, base=10, steps=1000), 
                     at.logspace(start=1, end=2, base=10, steps=100), at.logspace(start=2, end=5, base=10, steps=50) ])))

zGridGlobals = np.array(zGridGlobals_at.eval())


# try:
#     zGridGlobals_at = at.sort(at.unique(at.concatenate([
#         at.logspace(start=-100, stop=-15, base=10, steps=50),
#         at.logspace(start=-30, stop=-4, base=10, steps=100),
#         at.logspace(start=-4, stop=1, base=10, steps=1000),
#         at.logspace(start=1, stop=2, base=10, steps=100),
#         at.logspace(start=2, stop=5, base=10, steps=50),
#         at.logspace(start=5, stop=6, base=10, steps=20),   # aggiunta: redshift molto alto
#         at.logspace(start=6, stop=7, base=10, steps=10)    # aggiunta: bordo massimo sicuro
#     ])))
# except:
#     zGridGlobals_at = at.sort(at.unique(at.concatenate([
#         at.logspace(start=-100, end=-15, base=10, steps=50),
#         at.logspace(start=-30, end=-4, base=10, steps=100),
#         at.logspace(start=-4, end=1, base=10, steps=1000),
#         at.logspace(start=1, end=2, base=10, steps=100),
#         at.logspace(start=2, end=5, base=10, steps=50),
#         at.logspace(start=5, end=6, base=10, steps=20),   # aggiunta
#         at.logspace(start=6, end=7, base=10, steps=10)    # aggiunta
#     ])))

# zGridGlobals = np.array(zGridGlobals_at.eval())




# zGridGlobals = onp.sort(onp.unique(onp.concatenate([ onp.logspace(start=-100, stop=-15, base=10, num=50), np.logspace(start=-30, stop=-4, base=10, num=100), 
#                      #at.linspace(start=1.1e-03, end=10, steps=50),
#                      onp.logspace(start=-4, stop=1, base=10, num=1000), 
#                      onp.logspace(start=1, stop=2, base=10, num=100), onp.logspace(start=2, stop=5, base=10, num=50) ])))

def log_cheb(a, b, N):
    """
    Chebyshev nodes mapped to log10-space between a and b (a<b).
    Clusters near both endpoints in *log z*.
    """
    la, lb = onp.log10(a), onp.log10(b)
    k = onp.arange(N)
    theta = (k + 0.5) * onp.pi / N
    logz = 0.5 * (la + lb) + 0.5 * (lb - la) * onp.cos(theta)
    return 10 ** logz

def make_z_grid(total=150, hi_boost=0.20):
    """
    Generic grid builder:
      total   : total number of points (e.g., 150, 500)
      hi_boost: fraction of points allocated to (3,10]; default 20%
    Remaining points are split 10% / 45% / 45% across the first three bands.
    """
    total = int(total)
    zmin_a, zmin_b, zmid_b, zmax_c = 1e-5, 1e-3, 3.0, 10.0

    # allocate counts
    N3  = int(round(total * hi_boost))
    rem = total - N3
    N1  = int(round(rem * 0.10))
    N2a = int(round(rem * 0.45))
    N2b = rem - N1 - N2a  # remainder

    g1  = onp.logspace(onp.log10(zmin_a), onp.log10(zmin_b), max(N1,1), endpoint=False)
    g2a = log_cheb(1e-3, 1e-1,            max(N2a,1))
    g2b = log_cheb(1e-1, zmid_b,          max(N2b,1))
    g3  = onp.logspace(onp.log10(zmid_b), onp.log10(zmax_c), max(N3,1))

    z = onp.unique(onp.concatenate([g1, g2a, g2b, g3]))
    z.sort()
    return z



zGridGlobals_low = make_z_grid()


zGridGlobals_at_low = at.as_tensor_variable(zGridGlobals_low)



zGridGlobals_high = make_z_grid(1000)

zGridGlobals_at_high = at.as_tensor_variable(zGridGlobals_high)


max_m = 500.


_mass_grid_np = onp.unique(
    onp.concatenate([
        onp.linspace(1.0e-3, 15.0, 500, ),
        onp.linspace(15.01, 100.0, 1000, ),
        onp.linspace(101.1, max_m, 500, ),
    ])
)
_mass_grid_np.sort()
_mass_grid_at = at.as_tensor_variable(_mass_grid_np)

def _get_mass_grid():
    return _mass_grid_at



_tgrid  = onp.linspace(0.0, 1.0, 2000)
_tgrid_at = at.as_tensor_variable(_tgrid)

def _get_t_grid():
    return _tgrid_at
    


##########################
####### Auxiliary functions ########
##########################


# safe_pos = lambda x: at.clip(x, EPS32.astype(x.dtype), BIG32.astype(x.dtype))   # >0, finite
# safe_div = lambda a,b: a / safe_pos(b)
# safe_log = lambda x: at.log(safe_pos(x))
# safe_sqrt= lambda x: at.sqrt(safe_pos(x))
# clip_unit= lambda p: at.clip(p, 1e-12, 1 - 1e-7)  # probs in (0,1)



def uniform_unconstrained(name, low, high, init=None):
    # Optional: set z init to hit desired x init
    if init is not None:
        # nudge inside bounds to avoid logit at the edges
        eps = 1e-8 * (high - low)
        x0 = float(np.clip(init, low + eps, high - eps))
        z_init = float(np.log((x0 - low) / (high - x0)))  # logit((x-low)/(high-low))
    else:
        z_init = 0.0

    z = pm.Logistic(name + "_z", mu=0.0, s=1.0, initval=z_init)   # free RV on ℝ
    x = pm.Deterministic(name, low + (high - low) * pm.math.sigmoid(z))
    return x




#def logdiffexp(x, y):
#    """`log(exp(x)-exp(y))` """
#    return x + at.log1p(-at.exp(y-x))

# 1) Real-only version: returns -inf when x <= y
def logdiffexp(x, y, neg_inf=-np.inf):
    x = at.as_tensor_variable(x); y = at.as_tensor_variable(y)
    m = at.maximum(x, y)
    d = at.abs(x - y)
    tiny = at.as_tensor_variable(1e-300).astype(m.dtype)   # avoid 0 in masked branches
    d = at.maximum(d, tiny)
    # log|exp(x)-exp(y)| = max(x,y) + log(1 - exp(-|x-y|))
    logabs = m + at.log1p(-at.exp(-d))
    return at.where(x >= y, logabs, at.as_tensor_variable(neg_inf).astype(m.dtype))


def logdiffexp32(x,y):
    m = at.maximum(x,y); d = at.maximum(at.abs(x-y), EPS32)
    out = m + at.log1p(-at.exp(-d))
    return at.where(x>y, out, NINF)   # or your MIN

def logsumexp(x, y):
    """`log(exp(x)+exp(y))` """
    #return x + at.log1p(at.exp(y-x))
    return at.logaddexp(x, y)

def logitat(p):
    return at.log(p) - at.log(1. - p)

def inv_logitat(p):
    return 1. / (1 + at.exp(-p))

def inv_flogitat(p):
    return (at.exp(p) - 1. ) / (1. + at.exp(p))

 
def flogitat(p):
    return at.log(1 + p) - at.log(1 - p)


def logit(p):
    return np.log(p) - np.log(1. - p)

def inv_logit(p):
    return 1. / (1 + np.exp(-p))

def inv_flogit(p):
    return (np.exp(p) - 1. ) / (1. + np.exp(p))

 
def flogit(p):
    return np.log(1 + p) - np.log(1 - p)


def normal_cdf(x):
    # Phi(x) = 0.5 * (1 + erf(x/sqrt(2)))
    return 0.5 * (1.0 + at.erf(x / at.sqrt(2.0)))



def normal_ppf(u):
    return at.sqrt(2.0) * at.erfinv(2.0*u - 1.0)


def m1m2_from_Mcq_at(Mc, q):
    
    m1 = Mc*(1+q)**(1./5.)/q**(3./5.)
    m2 = q*m1

    return m1, m2

def Mcq_from_m1m2_at(m1, m2):
   
    Mc  = ((m1*m2)**(3./5.))/((m1+m2)**(1./5.))
    q = m2/m1
    
    return Mc, q


def get_sample_from_cho_lMclqld(x, mu, L):
    
    
    # for cholesky rules see 
    # https://www.cs.helsinki.fi/u/ahonkela/teaching/compstats1/book/multivariate-normal-distributions-and-numerical-linear-algebra.html
    
    # x, mu have shape 3
    # L has shape 3x3
    # nd = mu.shape[0]

    sample = mu + (L @ x[:, None])[:, 0]   # instead of at.dot(L, x)

    # Log probability of standard normal x
    logp = (
    -0.5 * at.sum(x**2)   # instead of at.dot(x.T, x)
    - 0.5 * mu.shape[0] * at.log(2 * atools.PI)
    - at.sum(at.log(at.diagonal(L)))  # log determinant of L
    )
    return sample, logp


def stick_breaking(beta):
    portion_remaining = at.concatenate([[1], at.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining



#######################
# sigmoids
########################

#######################
# working 

def sigmoid(x, x0, s, eps=1e-12, clip=1e-15):
    # ensure positive scale
    s = at.maximum(s, at.as_tensor_variable(eps).astype(x.dtype))
    t = (x - x0) / s
    y = 0.5 * (at.tanh(0.5 * t) + 1.0)   # stable sigmoid
    if clip is not None:
        lo = at.as_tensor_variable(clip).astype(x.dtype)
        hi = at.as_tensor_variable(1.0 - clip).astype(x.dtype)
        y = at.clip(y, lo, hi)            # only if you really need interior (0,1)
    return y


# def sigmoid(x, m, sig):
#     return 1/(1+at.exp((-(x-m)/sig)))

def log_sigmoid(x, m, sig):
    return at.log(sigmoid(x, m, sig)) 


def safe_sigmoid(x, x0, eps):

    # works, older
    #s = 1.0 / (1.0 + at.exp(-(x - x0) / eps))
    #return at.clip(s, 1e-15, 1 - 1e-15)
    
    return sigmoid(x, x0, eps, clip=1e-15)


#######################


# def _softplus_stable(z):
#     zero = at.as_tensor_variable(0.0).astype(z.dtype)
#     return at.log1p(at.exp(-at.abs(z))) + at.maximum(z, zero)

# def log_sigmoid(x, m, s, eps=EPS32):
#     epsv = at.as_tensor_variable(eps).astype(x.dtype)
#     s = at.maximum(s, epsv)                  # ensure positive scale
#     t = (x - m) / s
#     return -_softplus_stable(-t)             # = log(sigmoid(t))

# def log1m_sigmoid_stable(x, m, s, eps=EPS32):
#     epsv = at.as_tensor_variable(eps).astype(x.dtype)
#     s = at.maximum(s, epsv)
#     t = (x - m) / s
#     return -_softplus_stable(t)              # = log(1 - sigmoid(t))

# def sigmoid(x, m, s, eps=EPS32):     # (in case you also need the sigmoid itself)
#     epsv = at.as_tensor_variable(eps).astype(x.dtype)
#     s = at.maximum(s, epsv)
#     t = (x - m) / s
#     return 0.5 * (at.tanh(0.5 * t) + 1.0)

#######################


# def _softplus_stable(z):
#     # log(1 + exp(z)) computed without overflow/underflow
#     zero = at.as_tensor_variable(0.0).astype(z.dtype)
#     z = at.clip(z, -1e20 if str(z.dtype).endswith("32") else -1e300,
#                    1e20  if str(z.dtype).endswith("32") else  1e300)
#     return at.log1p(at.exp(-at.abs(z))) + at.maximum(z, zero)


# def _guard_scale(s, x, eps_f32=1e-12, eps_f64=1e-300):
#     eps = eps_f32 if str(x.dtype).endswith("32") else eps_f64
#     epsv = at.as_tensor_variable(eps).astype(x.dtype)
#     # replace NaN with eps, then ensure >= eps
#     s = at.where(at.isnan(s), epsv, s)
#     return at.maximum(s, epsv)

# def log_sigmoid(x, m, s):
#      #s = _guard_scale(s, x)
#      t = (x - m) / s
#      return -at.logaddexp(0.0, -t)          # == log(sigmoid(t))

# def log1m_sigmoid_stable(x, m, s):
#     #s = _guard_scale(s, x)
#     t = (x - m) / s
#     return -at.logaddexp(0.0, t)           # == log(1 - sigmoid(t))

#######################


def softplus(x):
    # log(1 + exp(x)) with good numerical stability
    return at.maximum(x, 0) + at.log1p(at.exp(-at.abs(x)))


##########################
####### Interpolators and integrators ########
##########################

def meshgrid_at(x, y):
    x = at.as_tensor_variable(x)
    y = at.as_tensor_variable(y)
    nx = x.shape[0]
    ny = y.shape[0]

    X = at.alloc(x, nx, ny)      # Broadcast x along columns
    Y = at.alloc(y, nx, ny).T    # Broadcast y along rows, then transpose

    return X.T, Y.T


def atinterp(x, xs, ys, eps=1e-12, side="right"):

    # was atinterp_safe_fast
    
    # Assume xs, ys are tensors; cast if needed outside for speed
    idxs = at.searchsorted(xs, x, side=side)
    # Clamp indices to valid interior [1, N-1]
    idxs = at.clip(idxs, 1, xs.shape[0] - 1)

    xl = xs[idxs - 1]; xh = xs[idxs]
    yl = ys[idxs - 1]; yh = ys[idxs]

    denom = at.maximum(xh - xl, eps)  # protect against accidental ties
    r = (x - xl) / denom
    return (1 - r) * yl + r * yh

def atinterp_minimal(x, xs, ys):

  idxs = at.searchsorted(xs, x,  side='left', sorter=None)

  xl = xs[idxs-1]
  yl = ys[idxs-1]
  xh = xs[idxs]
  yh = ys[idxs]

  r = (x-xl)/(xh-xl);

  return r*yh + (1.0-r)*yl;


def atinterp_safe(x, xs, ys, eps=1e-12, mode="clip"):
    # x  = at.as_tensor_variable(x)
    # xs = at.as_tensor_variable(xs)
    # ys = at.as_tensor_variable(ys)

    x0, x1 = xs[0], xs[-1]
    if mode == "clip":
        xq = at.clip(x, x0 + eps, x1 - eps)
    elif mode == "error":
        # Optional penalty in a model context:
        # pm.Potential("interp_oob", at.switch(at.any((x < x0) | (x > x1)), -np.inf, 0.))
        xq = at.clip(x, x0 + eps, x1 - eps)
    else:
        raise ValueError("mode must be 'clip' or 'error'.")

    N = xs.shape[0]
    # Use 'right' then subtract 1 so exact knots fall to the left interval
    idxs = at.searchsorted(xs, xq, side="right")
    # Keep indices in [1, N-1]
    idxs = at.clip(idxs, 1, N-1)

    xl = xs[idxs - 1]
    xh = xs[idxs]
    yl = ys[idxs - 1]
    yh = ys[idxs]

    denom = at.clip(xh - xl, eps, np.inf)   # protect against ties in xs
    r = (xq - xl) / denom
    y = (1.0 - r) * yl + r * yh
    return y


def invert_monotone_binary_at(y, y_grid, x_grid, eps=1e-12, mode="clip"):
    """
    Invert a strictly increasing tabulation y_grid(x_grid) with binary search.
    y:      (M,)
    y_grid: (NZ,)
    x_grid: (NZ,)
    mode: "clip" | "error"
      - "clip": clamp queries to [y_grid[0], y_grid[-1]] (robust default)
      - "error": add -inf Potential if any query is out of range (you can wrap outside)
    returns x: (M,)
    """
    # Ensure tensors
    y       = at.as_tensor_variable(y)
    y_grid  = at.as_tensor_variable(y_grid)
    x_grid  = at.as_tensor_variable(x_grid)

    # Bounds & optional handling
    y0, y1 = y_grid[0], y_grid[-1]
    oob_low  = at.lt(y, y0)
    oob_high = at.gt(y, y1)

    if mode == "clip":
        yq = at.clip(y, y0 + eps, y1 - eps)
    elif mode == "error":
        # You can uncomment this Potential inside a model context:
        # pm.Potential("invert_oob", at.switch(at.any(oob_low | oob_high), -np.inf, 0.0))
        yq = at.clip(y, y0 + eps, y1 - eps)
    else:
        raise ValueError("mode must be 'clip' or 'error'.")

    NZ = x_grid.shape[0]
    # We search for k such that y_grid[k] <= yq < y_grid[k+1]
    # Initialize integer bounds
    lo = at.zeros_like(yq, dtype="int64")
    hi = at.fill(lo, NZ - 2)  # last valid left index

    # Number of bisection steps: ceil(log2(NZ))
    n_steps = int(np.ceil(np.log2(1 + (np.array(1) if isinstance(NZ, int) else 1024))))  # fallback if NZ not concrete
    # If NZ is a constant tensor, try to get its value
    try:
        nZ_val = int(y_grid.shape[0].eval())  # optional: if in interactive mode
        n_steps = int(np.ceil(np.log2(nZ_val)))
    except Exception:
        pass

    # Fixed-count loop (Python-level, creates ~log2(NZ) graph layers)
    for _ in range(max(1, n_steps)):
        mid = (lo + hi) // 2  # (M,) int64
        # Compare with the right edge of the mid bin: y_grid[mid+1]
        right = y_grid[mid + 1]
        go_right = at.le(right, yq)  # if yq >= y_grid[mid+1], move right
        lo = at.where(go_right, mid + 1, lo)
        hi = at.where(go_right, hi, mid)

    k = lo  # (M,) left index of bin

    # Linear inverse within the bin
    ygL = y_grid[k]           # (M,)
    ygR = y_grid[k + 1]       # (M,)
    xL  = x_grid[k]           # (M,)
    xR  = x_grid[k + 1]       # (M,)

    dyg = at.clip(ygR - ygL, eps, np.inf)
    t   = (yq - ygL) / dyg
    x   = xL + t * (xR - xL)
    return x


def invert_monotone_linear(y, y_grid, x_grid, eps=1e-12, mode="clip"):
    """
    Vectorized inverse for strictly increasing y_grid(x_grid).

    y:      (M,)     query values (e.g., distances)
    y_grid: (NZ,)    tabulated monotone y(z)
    x_grid: (NZ,)    corresponding z grid
    mode: "clip" | "extrapolate" | "error" | "nan"
    """
    y0, y1 = y_grid[0], y_grid[-1]
    x0, x1 = x_grid[0], x_grid[-1]

    # Precompute per-bin slopes dx/dy
    dyg = y_grid[1:] - y_grid[:-1]              # (NZ-1,)
    dxg = x_grid[1:] - x_grid[:-1]              # (NZ-1,)
    dxdy_bins = dxg / at.clip(dyg, eps, np.inf) # (NZ-1,)

    # Masks for out-of-bounds
    oob_low  = at.lt(y, y0)
    oob_high = at.gt(y, y1)
    inside   = at.eq(oob_low + oob_high, 0)

    if mode == "clip":
        yq = at.clip(y, y0 + eps, y1 - eps)

    elif mode == "extrapolate":
        # linear extrapolation using edge slopes
        # left edge slope: use first bin
        dxdy_left  = dxdy_bins[0]
        # right edge slope: use last bin
        dxdy_right = dxdy_bins[-1]

        # Start by clipping to avoid undefined bin-finding; we'll replace OOB later
        yq = at.clip(y, y0 + eps, y1 - eps)

    elif mode == "error":
        # Hard fail: add a -inf Potential if any y is OOB
        pm.Potential(
            "invert_oob_penalty",
            at.switch(at.any(oob_low | oob_high), -np.inf, 0.0)
        )
        yq = at.clip(y, y0 + eps, y1 - eps)

    elif mode == "nan":
        # We'll compute inverse for inside points; fill NaN for OOB after
        yq = at.clip(y, y0 + eps, y1 - eps)

    else:
        raise ValueError("mode must be one of: clip, extrapolate, error, nan")

    # Find left bin index for the (possibly clipped) queries
    mask = at.cast(y_grid[None, :] <= yq[:, None], "int64")   # (M, NZ)
    k = at.sum(mask, axis=1) - 1                              # (M,)
    k = at.clip(k, 0, x_grid.shape[0] - 2)

    ygL = y_grid[k]             # (M,)
    xL  = x_grid[k]             # (M,)
    slope = dxdy_bins[k]        # (M,)

    x_in = xL + (yq - ygL) * slope  # inverse for clamped/inside points

    if mode == "clip":
        return x_in

    if mode == "extrapolate":
        # Left extrapolation: x0 + (y - y0) * dx/dy at left edge
        x_left  = x0 + (y - y0) * dxdy_left
        # Right extrapolation: x1 + (y - y1) * dx/dy at right edge
        x_right = x1 + (y - y1) * dxdy_right
        # Stitch
        return at.where(oob_low, x_left, at.where(oob_high, x_right, x_in))

    if mode == "error":
        return x_in  # Potential above will kill OOB cases

    if mode == "nan":
        nanv = at.as_tensor_variable(np.nan)
        return at.where(inside, x_in, nanv)



def jnptinterp(x, xs, ys):

  idxs = jax.numpy.searchsorted(xs, x,  side='left', sorter=None)

  xl = xs[idxs-1]
  yl = ys[idxs-1]
  xh = xs[idxs]
  yh = ys[idxs]

  r = (x-xl)/(xh-xl);

  return r*yh + (1.0-r)*yl;




def atcumtrapz(y, x=None, dx=1.0, axis=-1, initial=None):

    
    if x.ndim == 1:
        d = at.diff(x)
        # reshape to correct shape
        shape = [1] * y.ndim
        shape[axis] = -1
        d = d.reshape(shape)
    elif len(x.shape) != len(y.shape):
        raise ValueError("If given, shape of x must be 1-d or the "
                         "same as y.I got: d.shape=%s, y.shape=%s"%(d.shape.eval(), x.shape.eval()))
    else:
        d = at.diff(x, axis=axis)

    nd = y.ndim
    
    if x.ndim==1:
        res = at.cumsum(d * (y[1:] + y[:-1]) / 2.0, axis=axis)
    elif (x.ndim==2) and ((axis==1) or (axis==-1)):        
        res = at.sum( d * (y[:, 1: ]+y[:, :-1])/2.0, axis )

    return res


def attrapzvec11(y, x,  axis=-1):

    # works in 1D and 2D

    if True:
        if x.ndim == 1:
            d = at.diff(x)
            # reshape to correct shape
            shape = [1]*y.ndim
            shape[axis] = d.shape[0]
            d = at.reshape(d, shape)
        else:
            d = at.diff(x, axis=axis)
    nd = y.ndim
    
    if x.ndim == 1:
        ret = at.sum(d * (y[1:] + y[:-1]) / 2.0)#.sum(axis)
    elif (x.ndim==2) and ((axis==1) or (axis==-1)):
        # Operations didn't work, cast to ndarray
        # d = np.asarray(d)
        # y = np.asarray(y)        
        ret = at.sum( d * (y[:, 1: ]+y[:, :-1])/2.0, axis )    
    else:
      raise NotImplementedError()
    return ret


def attrapzvec(y, x,  dx=1., axis=-1):
        if x is None:
                d = dx
        else:
                #x = asanyarray(x)
                if x.ndim == 1:
                    d = at.diff(x)
                    # reshape to correct shape
                    shape = [1]*y.ndim
                    shape[axis] = d.shape[0]
                    d = at.reshape(d, shape)
                else:
                    d = at.diff(x, axis=axis)
        
        nd = y.ndim
        slice1 = [slice(None)]*nd
        slice2 = [slice(None)]*nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        try:
            ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
        except ValueError:
            # Operations didn't work, cast to ndarray
            d = np.asarray(d)
            y = np.asarray(y)
            ret = add.reduce(d * (y[tuple(slice1)]+y[tuple(slice2)])/2.0, axis)
        return ret



class TrapzOp(Op):
    itypes = [at.dmatrix, at.dmatrix]   # y: (M,N), x: (1,N)
    otypes = [at.dvector]               # output: (M,)

    def __init__(self, axis=1):
        self.axis = axis

    def perform(self, node, inputs, outputs):
        y, x = inputs                    # y: (M,N), x: (1,N)
        # Broadcast x to (M,N)
        x_b = np.broadcast_to(x, y.shape)
        out = np.trapz(y, x_b, axis=self.axis)  # (M,)
        outputs[0][0] = out

    def grad(self, inputs, output_grads):
        y, x = inputs                    # y: (M,N), x: (1,N)
        (gz,) = output_grads             # (M,)

        class JaxTrapzGrad(Op):
            itypes = [at.dmatrix, at.dmatrix]   # y: (M,N), x: (1,N)
            otypes = [at.dmatrix, at.dmatrix]   # dy: (M,N), dx: (1,N)

            def perform(inner_self, node, inputs, outputs):
                yv, xv = inputs                  # yv: (M,N), xv: (1,N)
                # Broadcast x to (M,N)
                xv_b = jnp.broadcast_to(xv, yv.shape)

                def trapz_sum(y_, x_):
                    return jnp.sum(jnp.trapz(y_, x_, axis=1))

                dy = jax.grad(trapz_sum, argnums=0)(yv, xv_b)  # (M,N)
                dx_full = jax.grad(trapz_sum, argnums=1)(yv, xv_b)  # (M,N)

                # Sum dx over M dimension to get (N,)
                dx = jnp.sum(dx_full, axis=0)   # (N,)
                dx = dx[None, :]                # (1,N)

                outputs[0][0] = np.asarray(dy)
                outputs[1][0] = np.asarray(dx)

        jax_grad_op = JaxTrapzGrad()
        dy, dx = jax_grad_op(y, x)

        return [gz[:, None] * dy, gz[:, None] * dx]  # dx broadcasted

##########################
####### Distances and cosmology ########
##########################


PI = np.pi #at.as_tensor_variable(np.pi)


# Precompute n-point Gauss–Legendre nodes/weights on [0,1]
def gauss_legendre_01(n=32, dtype="float64"):
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)                 # on [-1, 1]
    x01 = (x + 1.0) * 0.5              # map to [0, 1]
    w01 = w * 0.5
    return x01.astype(dtype), w01.astype(dtype)

_x01_np, _w01_np = gauss_legendre_01(n=32)  # 16–64 usually plenty
x01_at = at.as_tensor_variable(_x01_np)     # shape (n,)
w01_at = at.as_tensor_variable(_w01_np)     # shape (n,)


def dcfun_at(z, H0, Om, w0, interp=False):
    """Comoving distance at redshift ``z``, in Gpc, H0 in km/s/Mpc"""
    if interp:
        return pc.comoving_distance_pade_at(z, H0, Om, w0=-1.0, p=p, q=q) 
    else:
        
        # zz = at.linspace(0, z, steps=500).T
        # E = Efun_at(zz,Om,w0 )
        # return c_light/H0 * attrapzvec(1/E, zz)*1e-03
        
        z = at.as_tensor_variable(z)
        z_nodes = z[..., None] * x01_at  # shape (..., n)
        integrand = 1.0 / Efun_at(z_nodes, Om, w0)  # shape (..., n)
        I = at.sum(w01_at * integrand, axis=-1)     # shape (...)
        return (c_light / H0) * z * I * 1e-03

def Xifun_at(z, Xi0, n):
    return Xi0+(1-Xi0)/(1+z)**n


def dLfun_at(z, H0, Om, w0, Xi0, n, interp=False):
    """Luminosity distance at redshift ``z``."""
    return Xifun_at(z, Xi0, n)*(z+1.0)*dcfun_at(z, H0, Om, w0, interp=interp)


def Efun_at(z, Om, w0):
    # E(z) = sqrt( Om (1+z)^3 + (1-Om) (1+z)^{3(1+w0)} )
    a = 1.0 + z
    return at.sqrt(Om * a**3 + (1.0 - Om) * a**(3.0 * (1.0 + w0)))





def z_from_dL_at( r, H0, Om, w0, Xi0, n , interp=False):
    dLGrid_at = dLfun_at( zGridGlobals_at, H0, Om, w0, Xi0, n , interp=interp)
    z2dL = atinterp( r, dLGrid_at, zGridGlobals_at ) 
    return z2dL 


    
def log_j_at(z, Om, H0=70, dc=None,  interp=False):
    if dc is None:
        dc = dcfun_at(z, H0, Om, interp=interp)
    dc*=H0/c_light*1e03
    return at.log(4*PI)+2*at.log(dc)-at.log(Efun_at(z, Om=Om))


def log_dV_dz_at(z, H0, Om0, w0, dc=None, interp=False):
    if dc is None:
        dc = dcfun_at(z, H0, Om0, w0, interp=interp)    
    res =  at.log(4*PI)+at.log(c_light)-at.log(H0)+2*at.log(dc)-at.log(Efun_at(z, Om0, w0))-3*at.log(10)
    return res


def log_ddL_dz(z, H0, Om0,  w0, Xi0, n, dc=None, interp=False):
    
    # H0 in Mpc, dLs in Gpc
    if dc is None:
        dc = dcfun_at(z, H0, Om0,  w0, Xi0, n, interp=interp) # Gpc
    
    Xi = Xifun_at(z, Xi0, n)
    res = at.log( ( Xi - n*(1-Xi0)/(1+z)**n ) * dc + Xi * c_light_at * (1+z)/(1e03*H0*Efun_at(z,Om0,  w0)) )  
        
    return res


# no dependence on H0 (as in Finke et.al.)
# dc * H0/c
def u_z_at(z, Om, w0):
    zz = at.linspace(0, z, 100).T
    E = Efun_at(zz, Om, w0)
    u = attrapzvec(1./E, zz)
    return u

# dV/dzdOm * H0^3/c^3/4pi
def log_j_z_at(z, Om, w0, ):
    E = Efun_at(z, Om, w0)
    u = u_z_at(z, Om, w0).T
    logj = 2*at.log(u) - at.log(E)
    return logj

def log_j_z_at_norm(z, Om, w0, zmax):
    logj = log_j_z_at(z, Om, w0)
    zz = at.geomspace(1e-7, zmax, 10000) # fixed (zmin, zmax)
    log_norm = at.log(attrapzvec(at.exp(log_j_z_at(zz, Om, w0)), zz))
    return logj - log_norm


##########################
####### Redshift distributions ########
##########################

def zdist_at(z, gamma, kappa):
  return z**2*(1+z)**gamma*at.exp(-z**2/kappa)


def p_z_at(z, gamma, kappa, normalize=True, zmax=15):
    
    if normalize:
        zz = at.linspace(0, zmax, steps=500).T
        pz =  zdist_at(zz, gamma, kappa)
        norm = attrapzvec(pz, zz)
    else:
        norm=1

    return  zdist_at(z, gamma, kappa)/norm



def zdist_at_MD(z, gamma, kappa, zp):
    return at.exp(log_zdist_at_MD(z, gamma, kappa, zp))


def log_zdist_at_MD(z, gamma, kappa, zp):
    lrate =  gamma*at.log1p(z)-at.log(1+((1+z)/(1+zp))**(gamma+kappa))
    lC0 = at.log( 1+(1+zp)**(-gamma-kappa))
    return lC0+lrate


def psi_MD(z, gamma, kappa, zp, normalize=True, zmax=15):
    
    if normalize:
        zz = at.linspace(0, zmax, steps=500).T
        pz =  zdist_at_MD(zz, gamma, kappa, zp)
        norm = attrapzvec(pz, zz)
    else:
        norm=1

    return  zdist_at_MD(z, gamma, kappa, zp)/norm



def p_z_MD(z, gamma, kappa, zp, Om, normalize=True, zmax=20, dc=None):
    
    psiz = psi_MD(z, gamma, kappa, zp, normalize=False, zmax=zmax)
    dVdz = at.exp(log_j_at(z, Om, H0=70, dc=dc, ))
    
    if normalize:
        zz = at.linspace(0, zmax, steps=500).T
        pz =  psi_MD(zz, gamma, kappa, zp, normalize=False,)*at.exp(log_j_at(zz, Om, H0=70, dc=None, ))/(1+zz)
        norm = attrapzvec(pz, zz)
    else:
        norm=1
        
    return psiz*dVdz/(1+z)/norm


def log_p_z_MD_unnorm(z, gamma, kappa, zp, H0, Om, w0, dc=None):
    #lC0 = at.log( 1+(1+zp)**(-gamma-kappa))
    
    log_psiz = log_psi_z_MD(z, gamma, kappa, zp) #gamma*at.log1p(z)-at.log(1+((1+z)/(1+zp))**(gamma+kappa))
    
    log_dVdz = log_dV_dz_at(z, H0, Om, w0, dc=dc )

    return log_psiz+log_dVdz


def N_per_year( gamma, kappa, zp, H0, Om, w0, R0=1., dc=None, z_max = 100, res=1000):

    zgrid = at.linspace(0, z_max, steps=res) 
    pz = R0*at.exp( log_p_z_MD_unnorm(zgrid, gamma, kappa, zp, H0, Om, w0, dc=dc))
    norm = attrapzvec(pz, zgrid)
    return norm

def log_psi_z_MD(z, gamma, kappa, zp):
    lC0 = at.log( 1+(1+zp)**(-gamma-kappa))
    log_psiz = lC0+gamma*at.log1p(z)-at.log(1+((1+z)/(1+zp))**(gamma+kappa))
    return log_psiz-at.log1p(z)


def log_p_z_PL_unnorm(z, gamma, H0, Om, w0, dc=None):
    log_psiz = gamma*at.log1p(z)
    log_dVdz = log_dV_dz_at(z, H0, Om, w0, dc=dc )

    return log_psiz+log_dVdz-at.log1p(z)


def log_p_z_PL_norm(z, gamma, H0, Om, w0, dc=None):
    log_psiz = gamma*at.log1p(z)
    log_dVdz = log_dV_dz_at(z, H0, Om, w0, dc=dc )

    zz = at.geomspace(1e-07, 500, steps=2000).T #at.linspace(0, 5, steps=2000).T
    pz = at.exp( gamma*at.log1p(zz)+log_dV_dz_at(zz, H0, Om, w0,dc=None )-at.log1p(zz) )
    norm = attrapzvec(pz, zz)
    
    return log_psiz+log_dVdz-at.log1p(z)-at.log(norm)






##########################
####### Spin distributions ########
##########################


# def logpdf_multivariate_trunc_2D( x1, x2, m1, m2, s1, s2, rho, l1, u1, l2, u2 ):

    
#     where_inf =  ( x1 < l1 ) | ( x1 > u1 ) | ( x2 < l2 ) | ( x2 > u2 )

#     mean = at.as_tensor_variable([m1, m2])
#     x = at.as_tensor_variable([x1, x2]).T

#     sEsP = rho*s1*s2 

    
#     detC = s1**2* s2**2 - sEsP**2
#     logdetC = at.log(detC)

#     Cinv = at.zeros( (2, 2) )
#     Cinv = at.set_subtensor( Cinv[0,0], s2**2/detC )
#     Cinv = at.set_subtensor( Cinv[1,1], s1**2/detC )
#     Cinv = at.set_subtensor( Cinv[0,1], -sEsP/detC )
#     Cinv = at.set_subtensor( Cinv[1,0], -sEsP/detC )


#     return at.where( where_inf, MIN, pm.logp( pm.MvNormal.dist( mu=mean, tau=Cinv, shape=(x.shape[0], 3)), x ))


def logpdf_multivariate_trunc_2D(x1, x2, m1, m2, s1, s2, rho, l1, u1, l2, u2):
    # mask outside the box (no renormalization; just -inf outside)
    where_inf = (x1 < l1) | (x1 > u1) | (x2 < l2) | (x2 > u2)

    # mean and data as 2-vectors
    mean = at.stack([m1, m2], axis=0)                  # (2,)
    x    = at.stack([x1, x2], axis=0).T                # (n,2)

    # covariance pieces
    sEsP   = rho * s1 * s2
    detC   = at.clip(s1**2 * s2**2 - sEsP**2, 1e-300, np.inf)  # det Σ = s1^2 s2^2 (1-ρ^2)
    logdetC = at.log(detC)

    # precision Σ^{-1} via stacks (no set_subtensor)
    Cinv00 = (s2**2) / detC
    Cinv11 = (s1**2) / detC
    Cinv01 = -sEsP    / detC
    Cinv = at.stack([
        at.stack([Cinv00, Cinv01], axis=0),
        at.stack([Cinv01, Cinv11], axis=0)
    ], axis=0)                                         # (2,2)

    # quadratic form (x-μ)^T Σ^{-1} (x-μ), vectorized over rows of x
    delta = x - mean                                   # (n,2)
    Fd    = at.dot(Cinv, delta.T)                      # (2,n)
    quad  = at.sum(delta * Fd.T, axis=1)               # (n,)

    logpdf = -0.5 * (2.0 * at.log(2.0 * atools.PI) + logdetC + quad)  # (n,)

    return at.where(where_inf, MIN, logpdf)



def logpdf_default_spin(theta, lambdaBBHspin):

    chi1, chi2, cost1, cost2 = theta
    alphaChi, betaChi, zeta, sigmat = lambdaBBHspin
  
    normBeta =  at.gammaln(alphaChi) + at.gammaln(betaChi) - at.gammaln(alphaChi + betaChi)
        
    lpdfs1 = (alphaChi-1.0)*at.log(chi1) + (betaChi-1.0)*at.log1p(-chi1)
    lpdfs2 = (alphaChi-1.0)*at.log(chi2) + (betaChi-1.0)*at.log1p(-chi2)

    logpdfampl = lpdfs1 + lpdfs2 - 2*normBeta
   
  
    lpdfcos1_gauss = -0.5*(1.0-cost1)**2/(sigmat**2)-at.log(sigmat)-at.log(at.erf(at.sqrt(2.)/sigmat))
    lpdfcos2_gauss = -0.5*(1.0-cost2)**2/(sigmat**2)-at.log(sigmat)-at.log(at.erf(at.sqrt(2.)/sigmat))

    return logpdfampl + logsumexp( at.log(2.0)+at.log(zeta)-at.log(PI) + lpdfcos1_gauss + lpdfcos2_gauss, at.log(1.0-zeta)-at.log(4.0) )


def logpdf_default_spin_gauss(theta, lambdaBBHspin):

    chi1, chi2, cost1, cost2 = theta
    muChi, sigmaChi, zeta, sigmat = lambdaBBHspin
  
        
    lpdfs1 = truncGausslowerupper_at_lpdf_nonly(chi1, muChi, sigmaChi, xmin=0, xmax=1)
    lpdfs2 = truncGausslowerupper_at_lpdf_nonly(chi2, muChi, sigmaChi, xmin=0, xmax=1)

    logpdfampl = lpdfs1 + lpdfs2
   
  
    lpdfcos1_gauss = -0.5*(1.0-cost1)**2/(sigmat**2)-at.log(sigmat)-at.log(at.erf(at.sqrt(2.)/sigmat))
    lpdfcos2_gauss = -0.5*(1.0-cost2)**2/(sigmat**2)-at.log(sigmat)-at.log(at.erf(at.sqrt(2.)/sigmat))

    return logpdfampl + logsumexp( at.log(2.0)+at.log(zeta)-at.log(PI) + lpdfcos1_gauss + lpdfcos2_gauss, at.log(1.0-zeta)-at.log(4.0) )

    
        

##########################
####### Mass distributions ########
##########################


####### Uncorrelated flat ########


def logpdf_flat_sharp(theta, lambdaBBHmass):  
    m1, m2 = theta
    ml, mh = lambdaBBHmass

    return at.where( (m1>=ml) & (m1<=mh) & (m2>=ml) & (m2<=mh) & (m2<=m1), -2*at.log( mh-ml ) , MIN  )


def logpdf_flat(theta, lambdaBBHmass):  
    m1, m2 = theta
    ml, mh = lambdaBBHmass

    return -2*at.log( mh-ml ) + at.log(1-sigmoid(m1, mh, 0.05))+log_sigmoid(m1, ml, 0.05)+ at.log(1-sigmoid(m2, mh, 0.05))+log_sigmoid(m2, ml, 0.05)

    
    
    
####### Uncorrelated gaussian ########

def truncGausslower_at(x, loc, scale, xmin=0, ):    

    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    return at.where(x>xmin, 1./(at.sqrt(2.*PI)*scale)/(1.-Phialpha) * at.exp(-(x-loc)**2/(2*scale**2)) , 0.)


def truncGaussLowerUpper_at(x, loc, scale, xmin=0, xmax=1 ):    

    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    Phibeta = 0.5*(1.+at.erf((xmax-loc)/(at.sqrt(2.)*scale)))
    return at.where(  at.le(xmin,x) & at.le(x,xmax), 1./(at.sqrt(2.*PI)*scale)/(Phibeta-Phialpha) * at.exp(-(x-loc)**2/(2*scale**2)) , 0.)


def truncGausslowerupper_at_lpdf(x, loc, scale, xmin=0, xmax=1):    

    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    Phibeta = 0.5*(1.+at.erf((xmax-loc)/(at.sqrt(2.)*scale)))
    
    return at.where( (x>=xmin) & (x<=xmax), 
                    -at.log(scale)-0.5*at.log(2*PI)-at.log(Phibeta-Phialpha) + 0.5*(-(x-loc)**2/(scale**2)) , MIN)


def truncGausslowerupper_at_lpdf_nonly(x, loc, scale, xmin=0, xmax=1):    

    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    Phibeta = 0.5*(1.+at.erf((xmax-loc)/(at.sqrt(2.)*scale)))
    
    return -at.log(scale)-0.5*at.log(2*PI)-at.log(Phibeta-Phialpha) + 0.5*(-(x-loc)**2/(scale**2)) 

def truncGausslower_at_lpdf(x, loc, scale, xmin=0):    

    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    #Phibeta = 0.5*(1.+at.erf((xmax-loc)/(at.sqrt(2.)*scale)))
    
    return at.where( x>=xmin, 
                    -at.log(scale)-0.5*at.log(2*PI)-at.log(1.-Phialpha) + 0.5*(-(x-loc)**2/(scale**2)) , MIN)


def double_gauss_norm(mu, sigma):
    z = -mu / sigma
    C = 0.5 * (1 + at.erf(z / at.sqrt(2)))
    return 0.5 - C + 0.5 * C**2


def truncGausslower_at_logpdf(x, loc, scale, xmin=0):  
    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    return at.where(x>xmin, at.log(1./(at.sqrt(2.*PI)*scale)/(1.-Phialpha)) + -(x-loc)**2/(2*scale**2) , MIN )
    #return -at.log(scale)-0.5*at.log(2.*PI) -0.5*(x-loc)**2/(scale**2)

def truncGausslower_at_pdf(x, loc, scale, xmin=0):  
    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    return at.where(x>xmin, at.exp( -(x-loc)**2/(2*scale**2))/(at.sqrt(2.*PI)*scale)/(1.-Phialpha) , at.as_tensor_variable(0.) )
    #return -at.log(scale)-0.5*at.log(2.*PI) -0.5*(x-loc)**2/(scale**2)



def logpdf_gauss(theta, lambdaBBHmass):  
    m1, m2 = theta
    loc, scale = lambdaBBHmass
    
    return truncGausslower_at_logpdf(m1, loc, scale, xmin=0) + truncGausslower_at_logpdf(m2, loc, scale, xmin=0) -at.log(double_gauss_norm(loc, scale))

def logpdf_gauss_cond(theta, lambdaBBHmass):  
    m1, m2 = theta
    loc, scale = lambdaBBHmass
    
    logpdfm1 = truncGausslower_at_lpdf( m1, xmin=0., loc=loc, scale=scale)
    logpdfm2 = truncGausslowerupper_at_lpdf( m2, xmin=0., xmax=m1, loc=loc, scale=scale)
    return logpdfm1+logpdfm2



####### Power Law + Peak ########


def truncated_power_law(m, alpha, ml, mh):
        
        where_compute = (ml < m) & (m < mh )

        result = at.where(where_compute, at.log(m)*(-alpha), MIN)
        
        return result



def logpdf_PLP(theta, lambdaBBHmass, pairing=True):
    
        m1, m2 = theta
        lambdaPeak, alpha, beta, deltam, ml, mh, muMass, sigmaMass = lambdaBBHmass
                
        where_compute = (m2 <= m1) & (ml <= m2) & (m1 <= mh ) 

        lpdfm1 = at.where(where_compute, logpdfm1_PLP(m1,  lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass ), MIN )
        lpdfm2 = at.where(where_compute,logpdfm2_PLP(m2, beta, deltam, ml), MIN )
        if pairing:
            lC = at.where(where_compute, logC_PLP(m1, beta, deltam,  ml, ), MIN )
        ln = at.where(where_compute, logNorm_PLP( lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass), MIN )
        
        return at.where( where_compute, lpdfm1+lpdfm2+lC-ln, MIN )
        

def logS_PLP(m, deltam, ml, eps=1e-12):
    """
    Smoothly goes from -inf (log 0) below ml to 0 (log 1) above ml+deltam,
    with a C^1 transition (smoothstep) in between. Numerically robust.
    """
    # normalize position in the window and clamp to [0, 1]
    t = (m - ml) / at.maximum(deltam, at.as_tensor_variable(eps).astype(m.dtype))
    t = at.clip(t, 0.0, 1.0)

    # smoothstep: S(t) = 3t^2 - 2t^3, monotone from 0→1 with zero slope at ends
    S = t * t * (3.0 - 2.0 * t)

    # log S, safely (avoid log(0) at the lower edge)
    return at.log(at.clip(S, at.as_tensor_variable(eps).astype(m.dtype), 1.0))
    
def logS_PLP_LVK(m, deltam, ml,):
        
        maskL = m <= ml 
        maskU = m >= (ml + deltam) 
        
        maskM = ~(maskL | maskU)
        
        s = at.where( maskL, MIN, at.as_tensor_variable(0.)  )
        
        s1 = at.where( maskM,  at.log(1/(1+ at.exp(deltam/(m-ml) + deltam/(m-ml - deltam) ) ))  , s  )
        
        return s1   



def logpdfm1_PLP(m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass):

    where_compute = (ml <= m) & (m <= mh )

    norm = norm_truncated_pl_num(alpha, ml, mh)
    trunc_component = at.where(where_compute, 1./m**alpha/norm, MIN)
    gauss_component = at.where(where_compute, at.exp(-(m-muMass)**2/(2*sigmaMass**2))/(at.sqrt(2*PI)*sigmaMass), MIN)

    lS = logS_PLP(m, deltam, ml) 
        
    result =  at.where( where_compute, at.log( (1-lambdaPeak)*trunc_component+lambdaPeak*gauss_component)+lS
                       , MIN )
    return result

    

def logpdfm2_PLP(m2, beta, deltam, ml):

    where_compute = (ml<= m2) #& (~where_nan)
    res = at.log(m2)*(beta)+logS_PLP(m2, deltam, ml)
    result = at.where( where_compute, res, MIN )
           
    return result

        

def logC_PLP( m, beta, deltam, ml, res=100):
    '''
    Gives inverse log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
    '''
    

    # max_m = at.as_tensor_variable(500)
  
    
    # x2 = at.linspace(ml, 15, res )
    # x3 = at.linspace(15.01, 100, res )
    # x4 = at.linspace(101.1, max_m, int(res/2) )
    # xx = at.concatenate([ x2, x3, x4 ] )
    # p2 = at.exp(logpdfm2_PLP( xx , beta, deltam, ml))
    # cdf = atcumtrapz(p2, xx, )
    # itr = atinterp( m, xx[1:], at.log(cdf))
    # return itr

    _tgrid = _get_t_grid()
    
    xx = ml + (max_m - ml) * _tgrid

    # Evaluate log-pdf on the fixed grid, then zero-out below ml
    logp2 = logpdfm2_PLP(xx, beta, deltam, ml)          # (NM,)
    p2    = at.exp(logp2)                                # (NM,)

    # CDF via trapezoid from the fixed grid (below-ml bins contribute 0)
    cdf = atcumtrapz(p2, xx)                             # (NM-1,)

    # Interpolate log C at m
    return atinterp(m, xx[1:], at.log(cdf))



    

# def logNorm_PLP( lambdaPeak, alpha,  deltam, ml, mh, muMass, sigmaMass  , res=1000 ):
    
#     '''
#         Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

#     '''
    
#     ms = at.linspace(ml, mh, res)
#     ps = at.exp( logpdfm1_PLP( ms , lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass  ))
#     p1 = at.where( (ms>=ml) & (ms<=mh), ps, 0.)
#     return at.log(attrapzvec(p1,ms))


def logNorm_PLP(lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, res=1000):
    """
    Log integral of p(m1, m2) dm1 dm2 (total normalization of the mass function).
    Uses a cached global grid; ml and mh can be stochastic.
    """
    
    _tgrid = _get_t_grid()
    
    xx = ml + (mh - ml) * _tgrid

    # Evaluate log-pdf on fixed grid
    logp = logpdfm1_PLP(xx, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass)  # (NM,)
    p    = at.exp(logp)

    Z = attrapzvec(p, xx)                              # (scalar)
    return at.log(at.clip(Z, 1e-300, np.inf))
            
    
            
def norm_truncated_pl_num(alpha, mmin, mmax):

    return 1/(1-alpha)*(mmax**(1-alpha)-mmin**(1-alpha))


def log_norm_truncated_pl_num(alpha, mmin, mmax, eps=1e-12):
    """
    log ∫_{mmin}^{mmax} m^{-alpha} dm
    = log( (mmax^(1-α) - mmin^(1-α)) / (1-α) ), with a stable α≈1 branch.
    """
    # tensors + guards
    epsv  = at.as_tensor_variable(eps).astype(mmin.dtype)

    mmin_c = at.clip(mmin, epsv, INF)
    mmax_c = at.maximum(at.clip(mmax, epsv,INF), mmin_c * (1.0 + 1e-12))

    t = at.as_tensor_variable(1.0).astype(alpha.dtype) - alpha  # t = 1 - α
    close = at.abs(t) < 1e-6

    # α ≠ 1: log( |mmax^t - mmin^t| ) - log( |t| )
    num = at.pow(mmax_c, t) - at.pow(mmin_c, t)
    log_not1 = at.log(at.abs(num)) - at.log(at.abs(t))

    # α = 1: log( log(mmax/mmin) )
    log_ratio = at.log(mmax_c / mmin_c)
    log_eq1   = at.log(at.clip(log_ratio, epsv, np.inf))

    return at.switch(close, log_eq1, log_not1)
    

####### Power Law + Peak smooth edges , LVK low-end ########



def logpdf_PLP_reg(theta, lambdaBBHmass,  smoothing='LVK'):
    
        m1, m2 = theta
        lambdaPeak, alpha, beta, deltam, ml, mh, muMass, sigmaMass = lambdaBBHmass
                

        lpdfm1 = logpdfm1_PLP_reg( m1, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing=smoothing )
        
        lpdfm2 = logpdfm2_PLP_reg(m2, beta, deltam, ml, smoothing=smoothing)
        
        ln = logNorm_PLP_reg( lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing=smoothing )
        
        return lpdfm1 +lpdfm2-ln-logC_PLP_reg(m1, beta, deltam,  ml, smoothing=smoothing) 
        


 
def logpdfm1_PLP_reg(m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, sl=0.05, sh=0.05, smoothing='LVK'):

    return logpdfm1_PLP_noreg(m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing=smoothing)  + log_sigmoid(m, ml, sl) + at.log(1-safe_sigmoid(m, mh, sh)) 
    
    # at.log(1-sigmoid(m, mh, sh))  #log1m_sigmoid_stable(m, mh, sh)
    #at.log(1-safe_sigmoid(m, mh, sh)) 
    #+ log1m_sigmoid_stable(m, mh, sh)

def logpdfm1_PLP_noreg(m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing='LVK'):

    log_norm = log_norm_truncated_pl_num(alpha, ml, mh) #norm_truncated_pl_num(alpha, ml, mh)
    log_trunc_component =  -alpha*at.log(m) - log_norm #1./(m**alpha)/norm
    log_gauss_component = -0.5 * at.square((m - muMass) / sigmaMass) - at.log(sigmaMass) - 0.5 * at.log(2*PI)

    if smoothing=='LVK':
        lS = logS_PLP_LVK(m, deltam, ml)
    else:
        lS = logS_PLP(m, deltam, ml)
        
    #result =  at.log( (1-lambdaPeak)*trunc_component+lambdaPeak*gauss_component) + lS

    result = logsumexp( at.log1p(-lambdaPeak) + log_trunc_component, at.log(lambdaPeak) + log_gauss_component ) + lS
 
    return result


def logpdfm2_PLP_reg(m, beta, deltam, ml, sig_l=0.05, m_g=45, w_g = 80, sig_g_low = 5., sig_g_high = 5. , has_m2_break=False,  smoothing='LVK'):

    return logpdfm2_PLP_noreg(m, beta, deltam, ml,  m_g=m_g, w_g = w_g, sig_g_low = sig_g_low, sig_g_high = sig_g_high, has_m2_break=has_m2_break, smoothing=smoothing)+log_sigmoid(m, ml, sig_l) 
    

def logpdfm2_PLP_noreg(m, beta, deltam, ml,  m_g=45, w_g = 80, sig_g_low = 5., sig_g_high = 5. ,  has_m2_break=False, smoothing='LVK'):

    if smoothing=='LVK':
        lS = logS_PLP_LVK(m, deltam, ml) 
    else:
        lS = logS_PLP(m, deltam, ml) 
    
    lpdfval = beta*at.log(m) + lS
    
    if not has_m2_break:
        return lpdfval
    else:
        #eval = at.and_(m2 <= m_g, m2 >=  m_g+w_g )
        #return at.where(eval, lpdfval, MIN)
        
        # Define two sigmoid edges: one increasing at m_g, one decreasing at m_g + w_g
        left_edge  = 1 - safe_sigmoid(m, m_g, sig_g_low )
        right_edge = safe_sigmoid(m, m_g + w_g, sig_g_high )
        
        # Smooth mask transitions from 1 to 0 over the window [m_g, m_g + w_g]
        mask = at.log( left_edge + right_edge )
        
        # Smoothly blend between lpdfval and MIN
        return mask + lpdfval
        




def logC_PLP_reg( m, beta, deltam, ml, res=1000, smoothing='LVK'):
    '''
    Gives log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
    '''

    #max_m = at.as_tensor_variable(500)
  
   
    # lower edge
    #ms1 = at.linspace(ml, 15, res)
    
    # before gaussian peak
    #ms2 = at.linspace( 15.1, 25, res )
    
    # around gaussian peak
    #ms3= at.linspace( 25.1, 40, res)
    
    # after gaussian peak
    #ms4 = at.linspace(40.1, 100, res )

    # after gaussian peak
    #ms5 = at.linspace(100.1, max_m, int(res/2) )
    
    #xx=at.concatenate([ms1,ms2, ms3, ms4, ms5] )

    #xx = at.linspace(ml, 500, res)

    _tgrid = _get_t_grid()
    
    xx = ml + (max_m - ml) * _tgrid 

    # --- DEBUG PRINTS ---
    # print("DEBUG logC_PLP_reg:")
    # print("xx shape:", xx.shape.eval())         # dimensione della griglia
    # print("xx min/max:", xx[0].eval(), xx[-1].eval())  # min e max della griglia
    # print("m min/max:", at.min(m).eval(), at.max(m).eval())  # min/max delle masse che passano
    # print("number of injections:", m.shape.eval())  # quante masse stiamo passando
    # -------------------
    
    p2 = at.exp(logpdfm2_PLP_noreg( xx , beta, deltam, ml, smoothing=smoothing))
    cdf = atcumtrapz(p2, xx, )
    itr = atinterp( m, xx[1:], at.log(cdf) )

    return itr



def logNorm_PLP_reg( lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing='LVK', res=1000):
    
    '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

    '''


    _tgrid = _get_t_grid()
    ms = ml + (mh - ml) * _tgrid 
    
    ps = at.exp( logpdfm1_PLP_noreg( ms , lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing=smoothing  ))

    return  at.log( attrapzvec(ps, ms) )


            


####### double Power Law + double Peak  LVK low-end ########


def log_broken_power_law_DPLDP_pdf(m1, alpha1, alpha2, mb, m1_low, m_high, sh=0.05, sl=0.05, epsilon=0.01):
    """
    Log of the broken power-law PDF 
    """    
    
    # Compute log normalization constant
    norm1 = (m_high * (m_high / mb) ** (-alpha2) - mb) / (-alpha2 + 1)
    norm2 = (mb - m1_low * (m1_low / mb) ** (-alpha1)) / (-alpha1 + 1)
    log_N = at.log(norm1 + norm2)


    # log(pdf) in each regime
    log_val1 = -alpha1 * at.log(m1 / mb)
    log_val2 = -alpha2 * at.log(m1 / mb)

  
    # Smooth weight function (sigmoid transition)
    w = safe_sigmoid( -m1, -mb, epsilon)

    # Use log-sum-exp to compute:
    # log(w * exp(log_val1) + (1-w) * exp(log_val2))
    log_mix_val = logsumexp(
        at.log(w) + log_val1,
        at.log1p(-w) + log_val2
    )

    
    s1 = at.log1p(-safe_sigmoid(m1, m_high, sh))
    s2 = at.log(safe_sigmoid(m1, m1_low, sl))

    return log_mix_val - log_N + s1 + s2


def logpdfm1_DPLDP(
    m1, alpha1, alpha2, mb,
    mu1, sigma1, mu2, sigma2,
    m1_low, m_high, delta_m1,
    lambda0, lambda1,
    epsilon, 
    smoothing='LVK'
    ):
    """
    Log of the mixture model. Assumes other components return log-probabilities.
    """
    
    log_lambda0 = at.log(lambda0)
    log_lambda1 = at.log(lambda1)
    log_lambda2 = at.log1p(-lambda0 - lambda1)  # log(1 - λ0 - λ1)

    log_ppl = log_broken_power_law_DPLDP_pdf(m1, alpha1, alpha2, mb, m1_low, m_high, epsilon=epsilon)
    
    log_pnorm1 = truncGausslowerupper_at_lpdf(m1, mu1, sigma1, xmin=m1_low, xmax=m_high) # low-mass peak
    log_pnorm2 = truncGausslowerupper_at_lpdf(m1, mu2, sigma2, xmin=m1_low, xmax=m_high)   # mid-mass peak
    
    if smoothing=='LVK':
        log_S = logS_PLP_LVK(m1, delta_m1, m1_low,)
    else:
        log_S = logS_PLP(m1, delta_m1, m1_low,)

    # logsumexp of the weighted logs
    log_mix = logsumexp(
        logsumexp(log_lambda0 + log_ppl, log_lambda1 + log_pnorm1),
        log_lambda2 + log_pnorm2
    )

    return log_S + log_mix 


def logpdf_DPLDP(theta, lambdaBBHmass, force_m2_less_than_m1=False, has_m2_break=False, smoothing='LVK'):
    
        m1, m2 = theta
        alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, beta, m2_low, delta_m2, epsilon, m_g, w_g, sig_g_low, sig_g_high = lambdaBBHmass
                

        lpdfm1 = logpdfm1_DPLDP( m1, alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, epsilon, smoothing=smoothing)
    
        lpdfm2 = logpdfm2_PLP_reg(m2, beta, delta_m2, m2_low, m_g=m_g, w_g=w_g, sig_g_low = sig_g_low, sig_g_high = sig_g_high, has_m2_break=has_m2_break, smoothing=smoothing)
        
        lC = logC_DPLDP(m1, beta, delta_m2,  m2_low, m_g=m_g, w_g=w_g, sig_g_low = sig_g_low, sig_g_high = sig_g_high, has_m2_break=has_m2_break, smoothing=smoothing) 
   
        ln = logNorm_DPLDP(  alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, epsilon, smoothing=smoothing)
    
        lpdf = lpdfm1 + lpdfm2 -lC -ln
    
        #lpdf = at.switch(
        #                    (at.isinf(lpdfm1) & (lpdfm1 < 0)) | (at.isinf(lpdfm2) & (lpdfm2 < 0)),
        #                    MIN,
        #                    lpdfm1 + lpdfm2 - lC - ln
        #                )
        

        if force_m2_less_than_m1:
            eval = at.and_(at.and_(m2 <= m1, m2 > 0), m1 > 0)
            return at.where(eval, lpdf, MIN)
        else:
            return lpdf
        
     

def logC_DPLDP( m, beta, deltam, m2_low, m_g=45, w_g=80, sig_g_low=5, sig_g_high = 5, has_m2_break=False, res=5000, smoothing='LVK'):
    '''
    Gives log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
    '''

    _tgrid = _get_t_grid()
    
    xx = m2_low + (max_m - m2_low) * _tgrid 
        
    p2 = at.exp( logpdfm2_PLP_noreg( xx , beta, deltam, m2_low, m_g=m_g, w_g=w_g, sig_g_low=sig_g_low, sig_g_high = sig_g_high, has_m2_break=has_m2_break, smoothing=smoothing))
    
    cdf = atcumtrapz( p2, xx, )

    itr = atinterp( m, xx[1:], at.log(cdf) )
    
    return itr





def logNorm_DPLDP( alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, epsilon, res=2000, smoothing='LVK'):
    
    '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )
    '''
    
    _tgrid = _get_t_grid()
    
    ms = m1_low + (m_high - m1_low) * _tgrid 
            
    lpdf = logpdfm1_DPLDP( ms , alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, epsilon, smoothing=smoothing  )
    ps = at.exp( lpdf)
    
    return at.log( attrapzvec(ps, ms) )





####### FullPop-4.0 ########


def log1mexp(x):
    """
    Numerically stable log(1 - exp(x)) for x < 0
    """
    log2 = at.log(2.0)
    return at.switch(
        x <= -log2,
        at.log1p(-at.exp(x)),
        at.log(-at.expm1(x))
    )

# --- Broken power law ---
def log_broken_power_law_FP_pdf(m, λ_p, norm=False, maxm=100, epsilon=0.1):
    m_NSmax, m_BHmin, α1, α2, α_dip = λ_p

    log_region1 = α1 * at.log(m)
    log_region2 = (α1 - α_dip) * at.log(m_NSmax) + α_dip * at.log(m)
    log_region3 = (α1 - α_dip) * at.log(m_NSmax) + (α_dip - α2) * at.log(m_BHmin) + α2 * at.log(m)

    s1 = safe_sigmoid(m, m_NSmax, epsilon)
    s2 = safe_sigmoid(m, m_BHmin, epsilon)

    log_part1 = log_region1 + at.log1p(-s1)
    log_part2 = log_region2 + at.log(s1) + at.log1p(-s2)
    log_part3 = log_region3 + at.log(s2)

    result = logsumexp(
        logsumexp(log_part1, log_part2),
        log_part3
    )

    if norm:
        mgrid = at.logspace(at.log10(1.0), at.log10(maxm), 2000)
        log_vals = log_broken_power_law_FP_pdf(mgrid, λ_p, norm=False, maxm=maxm)
        vals = at.exp(log_vals)
        norm_factor = attrapzvec(vals, mgrid)
        return result - at.log(norm_factor)
    else:
        return result


def log_l_filter_at(m, m0, η):
    log_x = η * (at.log(m0) - at.log(m))
    return -logsumexp(0.0, log_x)

def log_h_filter_at(m, m0, η):
    log_x = η * (at.log(m0) - at.log(m))
    return log_x - logsumexp(0.0, log_x)


def log_notch_filter_at(m, γlow, γhigh, ηlow, ηhigh, A):
    log_l = log_l_filter_at(m, γlow, ηlow)
    log_h = log_h_filter_at(m, γhigh, ηhigh)
    log_prod = log_l + log_h + at.log(A)
    return log1mexp(log_prod)  # safe: log(1 - A * l * h)

def log_f_q_FP(q, m2, Λ_q, has_m2_break=False, epsilon=0.1):
    
    beta_low, beta_high, m_break, m_g, w_g, sig_g_low, sig_g_high = Λ_q
    
    s = safe_sigmoid(m2, m_break, epsilon)

    log_s = at.log(s)
    log1m_s = at.log1p(-s)
    log_q = at.log(q)

    log_term1 = log1m_s + beta_low * log_q
    log_term2 = log_s + beta_high * log_q

    lpdfval = logsumexp(log_term1, log_term2)

    if not has_m2_break:
        return lpdfval
    else:
        #eval = at.and_(m2 <= m_g, m2 >=  m_g+w_g )
        #return at.where(eval, lpdfval, MIN)
        
        # Define two sigmoid edges: one increasing at m_g, one decreasing at m_g + w_g
        left_edge  = 1 - safe_sigmoid(m2, m_g, sig_g_low )
        right_edge = safe_sigmoid(m2, m_g + w_g, sig_g_high )
        
        # Smooth mask transitions from 1 to 0 over the window [m_g, m_g + w_g]
        mask = at.log( left_edge + right_edge )
        
        # Smoothly blend between lpdfval and MIN
        return mask + lpdfval


def log_B_notches(m, λ_b):
    γlow_1, γhigh_1, ηlow_1, ηhigh_1, A1 = λ_b[0:5]
    γlow_2, γhigh_2, ηlow_2, ηhigh_2, A2 = λ_b[5:10]
    η_NSmin, m_NSmin = λ_b[10:12]
    η_BHmax, m_BHmax = λ_b[12:14]

    log_n1 = log_notch_filter_at(m, γlow_1, γhigh_1, ηlow_1, ηhigh_1, A1)
    log_n2 = log_notch_filter_at(m, γlow_2, γhigh_2, ηlow_2, ηhigh_2, A2)
    log_l = log_l_filter_at(m, m_NSmin, η_NSmin)
    log_h = log_h_filter_at(m, m_BHmax, η_BHmax)

    return log_l + log_h + log_n1 + log_n2


def logpdfm1_FP(m, λ_m, norm=False):
    m_BHmax = λ_m[11:][-1]

    def my_logp(m, λ_m):
        c1, c2, μ1, σ1, μ2, σ2 = λ_m[0:6]
        λ_p = λ_m[6:11]
        λ_b = λ_m[11:]
        _, m_NSmin = λ_b[10:12]
        _, m_BHmax = λ_b[12:14]

        log_G1 = truncGausslowerupper_at_lpdf(m, μ1, σ1, xmin=m_NSmin, xmax=m_BHmax)
        log_G2 = truncGausslowerupper_at_lpdf(m, μ2, σ2, xmin=m_NSmin, xmax=m_BHmax)

        logP = log_broken_power_law_FP_pdf(m, λ_p, norm=False, maxm=m_BHmax * 1.1)
        logB = log_B_notches(m, λ_b)

        log_terms1 = 0.0  # log(1)
        log_terms2 = at.log(c1) + log_G1
        log_terms3 = at.log(c2) + log_G2
            
        logsum = logsumexp(
                logsumexp(log_terms1, log_terms2),
                log_terms3
            )

        return logsum + logP + logB

    log_unnorm = my_logp(m, λ_m)

    if norm:
        mgrid = at.logspace(at.log10(1.0), at.log10(m_BHmax * 1.1), 2000)
        log_vals = my_logp(mgrid, λ_m)
        vals = at.exp(log_vals)
        norm_factor = attrapzvec(vals, mgrid)
        return log_unnorm - at.log(norm_factor)
    else:
        return log_unnorm


def logpdf_FP(theta, lambdaBBHmass, norm=True, norm_p1=False, res=1000, force_m2_less_than_m1=False, has_m2_break=False):
    
    m1, m2 = theta
    λ_m, Λ_q = lambdaBBHmass
    
    logp1 = logpdfm1_FP(m1, λ_m, norm=norm_p1)
    logp2 = logpdfm1_FP(m2, λ_m, norm=norm_p1)
    q = m2 / m1
    logf = log_f_q_FP(q, m2, Λ_q, has_m2_break=has_m2_break)
    lpdfval = logp1 + logp2 + logf

    if force_m2_less_than_m1:
        eval = at.and_(at.and_(m2 <= m1, m2 > 0), m1 > 0)
        joint = at.where(eval, lpdfval, MIN)
    else:
        joint = lpdfval

    if norm:
        #m_min = 1e-05
        λ_b = λ_m[11:]
        _, m_NSmin = λ_b[10:12]
        m_max = λ_m[11:][-1] * 1.5
        m_min = m_NSmin * 0.5

        m1_grid_ = at.geomspace(m_min, m_max, res)
        m2_grid_ = at.geomspace(m_min, m_max, res)
        m1_vals_, m2_vals_ = meshgrid_at(m1_grid_, m2_grid_)

        m1_stack = at.flatten(m1_vals_)
        logp1_grid = logpdfm1_FP(m1_stack, λ_m, norm=norm_p1)

        m2_stack = at.flatten(m2_vals_)
        logp2_grid = logpdfm1_FP(m2_stack, λ_m, norm=norm_p1)

        q_grid = m2_stack / m1_stack
        logf_grid = log_f_q_FP(q_grid, m2_stack, Λ_q, has_m2_break=has_m2_break)

        joint_grid = logp1_grid + logp2_grid + logf_grid

        joint_grid = at.where(m2_stack <= m1_stack, at.exp(joint_grid), 0.0)

        trapz = TrapzOp(axis=1)
        inner = trapz(at.reshape(joint_grid, m2_vals_.shape), m2_grid_[None, :])

        trapz0 = TrapzOp(axis=0)
        norm_factor = trapz0(inner.dimshuffle(0, 'x'), m1_grid_[:, None])

        return joint - at.log(norm_factor)
    else:
        return joint

