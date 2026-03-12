#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import pytensor.tensor as at
import jax.numpy as np
import numpy as onp
import pymc as pm
import jax
from pytensor.graph import Apply, Op
import pytensor
from pytensor.gradient import grad
from pymc.distributions.dist_math import check_parameters
from pytensor.gradient import disconnected_grad as stop_grad


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

LOG_10_ZMIN = -5

 

try:
        zGridGlobals_at_long = at.sort(at.unique(at.concatenate([ at.logspace(start=-100, stop=-15, base=10, steps=50), at.logspace(start=-30, stop=-4, base=10, steps=100), 
                     #at.linspace(start=1.1e-03, end=10, steps=50),
                     at.logspace(start=-4, stop=1, base=10, steps=1000), 
                     at.logspace(start=1, stop=2, base=10, steps=100), at.logspace(start=2, stop=5, base=10, steps=50) ])))

except:
    
    zGridGlobals_at_long = at.sort(at.unique(at.concatenate([ at.logspace(start=-100, end=-15, base=10, steps=50), at.logspace(start=-30, end=-4, base=10, steps=100), 
                     #at.linspace(start=1.1e-03, end=10, steps=50),
                     at.logspace(start=-4, end=1, base=10, steps=1000), 
                     at.logspace(start=1, end=2, base=10, steps=100), at.logspace(start=2, end=5, base=10, steps=50) ])))



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



def make_z_grid(total=150, zmin_a=1e-05, zmin_b=1e-03, zmid_b=3.0, zmax_c=10.0, hi_boost=0.15, low_boost=0.15):
    """
    Generic grid builder:
      total   : total number of points (e.g., 150, 500)
      hi_boost: fraction of points allocated to (3,10]; default 15%
      low_boost: fraction of points allocated to (1e-05,1e-03]; default 15%  
      Remaining points are in 1e-03, 3
    """
    total = int(total)
    #zmin_a, zmin_b, zmid_b, zmax_c = 1e-5, 1e-3, 3.0, 10.0

    # allocate counts
    N3  = int(round(total * hi_boost))
    rem = total - N3
    N1  = int(round(rem * low_boost))
    #N2a = int(round(rem * 0.45))
    N2 = rem - N1 #- N2a  # remainder
    #print("z grid built. N1=%s, N2=%s, N3=%s"%(N1, N2, N3))

    g1  = onp.logspace(onp.log10(zmin_a), onp.log10(zmin_b), max(N1,1), endpoint=False)
    #g2a = log_cheb(1e-3, 1e-1,            max(N2a,1))
    g2 = log_cheb(zmin_b, zmid_b,          max(N2,1))
    g3  = onp.logspace(onp.log10(zmid_b), onp.log10(zmax_c), max(N3,1))

    z = onp.unique(onp.concatenate([g1, g2, g3]))
    z.sort()
    return z


def make_z_grids_GP(
    zmin=1e-6,
    z0=1e-2,          # ramp up to 0.01
    zmax=80.0,
    n_nodes=160,
    n_fine=900,
    n_ramp_nodes=12,  # extra points in [zmin, z0)
    n_ramp_fine=20,
):
    # ---- log-uniform base grids in x=log(1+z) ----
    x_min = onp.log1p(zmin)
    x_max = onp.log1p(zmax)

    z_nodes = onp.expm1(onp.linspace(x_min, x_max, n_nodes))
    z_fine  = onp.expm1(onp.linspace(x_min, x_max, n_fine))

    # ---- add low-z ramp (geometric in z) to densify [zmin, z0) ----
    def ramp(zmin, z_target, n):
        if n <= 0 or z_target <= zmin:
            return onp.array([zmin])
        # include zmin, exclude z_target (z_target will already exist in base grid if close)
        return onp.geomspace(zmin, z_target, n + 1)[:-1]

    z_nodes = onp.concatenate([ramp(zmin, z0, n_ramp_nodes), z_nodes])
    z_fine  = onp.concatenate([ramp(zmin, z0, n_ramp_fine),  z_fine])

    # ---- clean up ----
    z_nodes = onp.unique(z_nodes)
    z_fine  = onp.unique(z_fine)

    # ensure coverage includes z0 exactly (helpful for plotting/anchors)
    z_nodes = onp.unique(onp.concatenate([z_nodes, onp.array([z0])]))
    z_fine  = onp.unique(onp.concatenate([z_fine,  onp.array([z0])]))

    return z_nodes, z_fine


def make_z_grids_GP_uniform(
    zmin=1e-6, zmid=5.0, zmax=80.0,
    dz_low_nodes=0.05, n_high_nodes=60,
    dz_low_fine=0.01, n_high_fine=300,
    n_ramp_nodes=8,            # NEW: # extra points in (zmin, dz_low_nodes)
    n_ramp_fine=12,            # NEW: # extra points in (zmin, dz_low_fine)
):
    # ---- helper: geometric ramp from zmin up to z_target ----
    def ramp(zmin, z_target, n):
        if n <= 0 or z_target <= zmin:
            return onp.array([zmin])
        # include zmin, exclude z_target (it will appear in the uniform grid)
        r = onp.geomspace(zmin, z_target, n + 1)[:-1]
        return r

    # --- GP nodes: add ramp then uniform low-z, log(1+z) high-z ---
    z_ramp_nodes = ramp(zmin, dz_low_nodes, n_ramp_nodes)
    z_low_nodes  = onp.arange(dz_low_nodes, zmid + 1e-12, dz_low_nodes)  # start at dz_low_nodes
    t_nodes      = onp.linspace(onp.log1p(zmid), onp.log1p(zmax), n_high_nodes)
    z_high_nodes = onp.expm1(t_nodes)

    z_nodes = onp.unique(onp.concatenate([z_ramp_nodes, z_low_nodes, z_high_nodes]))

    # --- integration grid: add ramp then finer uniform low-z, log(1+z) high-z ---
    z_ramp_fine = ramp(zmin, dz_low_fine, n_ramp_fine)
    z_low_fine  = onp.arange(dz_low_fine, zmid + 1e-12, dz_low_fine)     # start at dz_low_fine
    t_fine      = onp.linspace(onp.log1p(zmid), onp.log1p(zmax), n_high_fine)
    z_high_fine = onp.expm1(t_fine)

    z_fine = onp.unique(onp.concatenate([z_ramp_fine, z_low_fine, z_high_fine]))

    return z_nodes, z_fine


def make_z_grids_GP_noramp(zmin=1e-6, zmid=5.0, zmax=80.0,
                 dz_low_nodes=0.05, n_high_nodes=60,
                 dz_low_fine=0.01, n_high_fine=300):
    # --- GP nodes: uniform low-z, log(1+z) high-z ---
    z_low_nodes = onp.arange(zmin, zmid + 1e-12, dz_low_nodes)
    # logspace in (1+z)
    t_nodes = onp.linspace(onp.log1p(zmid), onp.log1p(zmax), n_high_nodes)
    z_high_nodes = onp.expm1(t_nodes)
    z_nodes = onp.unique(onp.concatenate([z_low_nodes, z_high_nodes]))

    # --- integration grid: finer uniform low-z, moderate log high-z ---
    z_low_fine = onp.arange(zmin, zmid + 1e-12, dz_low_fine)
    t_fine = onp.linspace(onp.log1p(zmid), onp.log1p(zmax), n_high_fine)
    z_high_fine = onp.expm1(t_fine)
    z_fine = onp.unique(onp.concatenate([z_low_fine, z_high_fine]))

    return z_nodes, z_fine

zGrid500_at = make_z_grid(total=1000)

zGridGlobals = make_z_grid()


zGridGlobals_at = at.as_tensor_variable(zGridGlobals)


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


def safe_exp(x, lo=-60.0, hi=60.0):        # avoid exp(±inf)->{inf,0} and NaN backprop
    return at.exp(at.clip(x, lo, hi))

def safe_exp1(x, margin=5.0):
    #x = at.as_tensor_variable(x)
    # pick finfo from the tensor's dtype
    dt = onp.float64 if x.dtype == "float64" else onp.float32
    finfo = onp.finfo(dt)
    hi = onp.log(finfo.max)  - margin    # avoid overflow
    lo = onp.log(finfo.tiny) + margin    # avoid underflow to 0
    return at.exp(at.clip(x, at.as_tensor_variable(lo), at.as_tensor_variable(hi)))


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
    #tiny = at.as_tensor_variable(1e-300).astype(m.dtype)   # avoid 0 in masked branches
    tiny=1e-300
    d = at.maximum(d, tiny)
    # log|exp(x)-exp(y)| = max(x,y) + log(1 - exp(-|x-y|))
    logabs = m + at.log1p(-at.exp(-d))
    return at.where(x >= y, logabs, neg_inf) #at.as_tensor_variable(neg_inf).astype(m.dtype))


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
    s = at.maximum(s, eps )#at.as_tensor_variable(eps).astype(x.dtype))
    t = (x - x0) / s
    y = 0.5 * (at.tanh(0.5 * t) + 1.0)   # stable sigmoid
    if clip is not None:
        lo = at.as_tensor_variable(clip)#.astype(x.dtype)
        hi = at.as_tensor_variable(1.0 - clip)#.astype(x.dtype)
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


def poly_hinge_neg(x, tau=1e-3):
    # Penalize x<0; 0 for x≥0; ~(-x) for x≤-tau; C¹ smooth in [-tau, 0]
    t = at.clip(-x / tau, 0.0, 1.0)             # t ∈ [0,1]
    mid = tau * (2*t*t - t*t*t)                 # cubic: r(-tau)=tau, r(0)=0, slopes match
    tail = at.maximum(-(x + tau), 0.0)          # adds linear extension for x < -tau
    return mid + tail


def softplus_stable(x):
    # log(1 + exp(x)) computed stably for any x
    return at.log1p(at.exp(-at.abs(x))) + at.maximum(x, 0.0)

def softplus(x):
    # log(1 + exp(x)) with good numerical stability
    return at.maximum(x, 0) + at.log1p(at.exp(-at.abs(x)))



##########################
####### Interpolators and integrators ########
##########################

def at_isfinite(x):
    x = at.as_tensor_variable(x)
    return ~(at.isnan(x) | at.isinf(x))

def meshgrid_at(x, y):
    x = at.as_tensor_variable(x)
    y = at.as_tensor_variable(y)
    nx = x.shape[0]
    ny = y.shape[0]

    X = at.alloc(x, nx, ny)      # Broadcast x along columns
    Y = at.alloc(y, nx, ny).T    # Broadcast y along rows, then transpose

    return X.T, Y.T

from pytensor.gradient import disconnected_grad  # <- correct import



def _interp_indices_nonuniform_safe(x, x_grid):
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

    # clip x into grid domain (avoid out-of-bounds indices and extrapolation)
    x_clip = at.clip(x, x_grid[0], x_grid[-1])

    # insertion index in [0..N]
    j = at.searchsorted(x_grid, x_clip, side="right")

    # clamp to valid interpolation interval [1..N-1]
    j = at.clip(j, 1, N - 1)

    xL = x_grid[j - 1]
    xR = x_grid[j]
    denom = at.maximum(xR - xL, 1e-30)

    r = (x_clip - xL) / denom
    r = at.clip(r, 0.0, 1.0)

    return j, r


def atinterp(x, xs, ys, eps=1e-30):
    """
    1D linear interpolation of y(x) given monotone non-uniform grid xs and values ys.

    - Clips x to [xs[0], xs[-1]] to avoid extrapolation.
    - Safe for non-uniform grids.
    - eps prevents division by zero if xs has repeated points.
    """
    # Ensure tensors
    #x  = at.as_tensor_variable(x)
    #xs = at.as_tensor_variable(xs)
    #ys = at.as_tensor_variable(ys)

    print("\nTHis is the new atinterp")

    # Get safe interval indices and weights
    j, r = _interp_indices_nonuniform_safe(x, xs)

    yL = ys[j - 1]
    yR = ys[j]

    # Linear interp
    return (1.0 - r) * yL + r * yR



def atinterp1(x, xs, ys, return_grad=False):
    #x  = at.as_tensor_variable(x).astype("float64").ravel()
    #xs = at.as_tensor_variable(xs).astype("float64").ravel()
    #ys = at.as_tensor_variable(ys).astype("float64").ravel()

    # Detach indexing path from autodiff
    x_det  = disconnected_grad(x)
    xs_det = disconnected_grad(xs)

    # Sort by detached xs
    order  = at.argsort(xs_det)
    xs     = xs[order]
    ys     = ys[order]
    xs_det = xs_det[order]

    # Clamp x to grid range
    x     = at.clip(x,     xs[0],     xs[-1])
    x_det = at.clip(x_det, xs_det[0], xs_det[-1])

    # Build strictly-increasing surrogate grid for indexing
    eps      = 1e-12 #at.as_tensor_variable(1e-12).astype("float64")
    dx       = xs_det[1:] - xs_det[:-1]
    dx_safe  = at.maximum(dx, eps)

    xs0    = xs_det[0]                         # <-- scalar offset fixes broadcasting
    xs_idx = at.concatenate([xs_det[:1], xs0 + at.cumsum(dx_safe)])

    # Indices on detached path
    idxs = at.searchsorted(xs_idx, x_det, side="left")
    idxs = at.clip(idxs, 1, xs.shape[0] - 1)

    xl = xs[idxs - 1];  xh = xs[idxs]
    yl = ys[idxs - 1];  yh = ys[idxs]

    denom      = xh - xl
    safe_denom = at.maximum(denom, eps)
    r          = at.clip((x - xl) / safe_denom, 0.0, 1.0)

    y_interp = r * yh + (1.0 - r) * yl

    if return_grad:
        dy_dx = (yh - yl) / safe_denom
        dy_dx = at.switch(at.le(denom, eps), 0.0, dy_dx)
        return y_interp, dy_dx
    else:
        return y_interp



def atinterp_uniform(x, x0, x1, n, yp):
    """
    Uniform-grid linear interpolation: xp must be uniformly spaced and increasing.
    JAX-friendly: uses int32 indices.
    """
    # x = at.as_tensor_variable(x)
    # xp = at.as_tensor_variable(xp)
    # yp = at.as_tensor_variable(yp)

    #n = xp.shape[0]
    # dx = (xp[-1] - xp[0]) / (n-1)
    dx = (x1 - x0)  / (n - 1)

    # t in [0, n-1]
    t = (x - x0) / dx
    t = at.clip( t, 0.0,  n - 1 )

    j = at.floor(t).astype("int32") #.astype("int32")
    j = at.clip(j, 0, n - 2)

    r = t - j #.astype(t.dtype)

    y0 = yp[j]
    y1 = yp[j + 1]
    return (1.0 - r) * y0 + r * y1




def atinterp_00(x, xs, ys, return_grad=False):
    """
    Linearly interpolate ys(x) from (xs, ys) to x.
    Optionally returns gradient dy/dx.

    Args:
        x: TensorVariable (N,) — interpolation points
        xs: TensorVariable (M,) — fixed grid (sorted)
        ys: TensorVariable (M,) — values on the grid
        return_grad: bool — whether to return dy/dx

    Returns:
        y_interp: interpolated values at x
        (optional) grad: dy/dx at x
    """
    x = x.ravel()
    xs = xs.ravel()
    ys = ys.ravel()

    # Inject NaN if out-of-bounds
    #out_of_bounds = ~at.all((x >= xs[0]) & (x <= xs[-1]))
    #_ = at.switch(out_of_bounds, float("nan"), 0.0)

    # Interpolation indices
    idxs = at.searchsorted(xs, x, side='left')
    idxs = at.clip(idxs, 1, xs.shape[0] - 1)

    xl = xs[idxs - 1]
    xh = xs[idxs]
    yl = ys[idxs - 1]
    yh = ys[idxs]

    r = (x - xl) / (xh - xl)
    y_interp = r * yh + (1.0 - r) * yl

    if return_grad:
        dy_dx = (yh - yl) / (xh - xl)
        return y_interp, dy_dx
    else:
        return y_interp

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
    return x01, w01 #.astype(dtype), w01.astype(dtype)

_x01_np, _w01_np = gauss_legendre_01(n=32)  # 16–64 usually plenty
x01_at = at.as_tensor_variable(_x01_np)     # shape (n,)
w01_at = at.as_tensor_variable(_w01_np)     # shape (n,)



def dcfun_at(z, H0, Om, w0, interp=False):
    """Comoving distance at redshift ``z``, in Gpc, H0 in km/s/Mpc"""
    if interp:
        return pc.comoving_distance_pade_at(z, H0, Om, w0=-1.0, p=p, q=q) 
    else:
        
        # zz = at.linspace(0, z, steps=100).T
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

def safe_sqrt_pos(x, tiny=1e-12):
    return at.sqrt(at.maximum(x, tiny))
    


def Efun_at(z, Om, w0):
    # E(z) = sqrt( Om (1+z)^3 + (1-Om) (1+z)^{3(1+w0)} )
    a = 1.0 + z
    return at.sqrt(Om * a**3 + (1.0 - Om) * a**(3.0 * (1.0 + w0)))



def z_from_dL_at(
    r,
    H0,
    Om,
    w0,
    Lambda_MG,
    is_GP_dL,
    z_grid,
    z_grid_fine=None,
    out_type="fine",
    gp_mode="mono_reparam",     # "direct" (old) or "mono_reparam" (new)
    taper_z0=None,        # kept for API compatibility (unused in new mono_reparam)
):
    """
    If is_GP_dL:
      Returns dL grid plus (logXi, g=dlogXi/dz) on both coarse and fine grids.

    gp_mode:
      - "direct": GP draw is g(z)=d/dz log Xi (old behavior; compute_gp_interp_dist_ratio)
      - "mono_reparam": NEW stable monotone construction:
            GP is a latent perturbation f(x) with x=log1p(z),
            q(z)=d/dz log dL_GW(z)=eps + softplus(inv_softplus(b(z)-eps) + f)
            g(z)=d/dz log Xi = q - b
        where b(z)=d/dz log dL_EM(z).
        This enforces monotonicity by construction and is GR-centered.

    out_type:
      - "fine": build dLGrid_at on fine grid
      - else:  build dLGrid_at on coarse grid
    """
    if not is_GP_dL:
        Xi0, n = Lambda_MG
        dLGrid_at = dLfun_at(z_grid, H0, Om, w0, Xi0, n)
        return atinterp(r, dLGrid_at, z_grid)

    gp = Lambda_MG[0]
    Z_nodes = z_grid

    if z_grid_fine is None:
        z_grid_fine = zGrid500_at

    # --- get logXi and g on grids ---
    if gp_mode == "direct":
        logXi_nodes, g_nodes, logXi_fine, g_fine = compute_gp_interp_dist_ratio(
            Z_nodes, gp, name="f", z_fine=z_grid_fine, reparameterize=True
        )

        # build dLGrid_at on requested grid (same as your old code)
        if out_type == "fine":
            z_out = z_grid_fine
            Xi_out = at.exp(logXi_fine)
        else:
            z_out = Z_nodes
            Xi_out = at.exp(logXi_nodes)

        dCGrid_at = dcfun_at(z_out, H0, Om, w0)
        dLGrid_EM_at = dCGrid_at * (1.0 + z_out)
        dLGrid_at = Xi_out * dLGrid_EM_at

        return dLGrid_at, logXi_nodes, g_nodes, logXi_fine, g_fine

    elif gp_mode == "mono_reparam":
        # NEW: delegate to the stable monotone constructor (already returns dLGrid_at)
        # NOTE: taper_z0 is intentionally ignored here (no low-z taper trick).
        dLGrid_at, logXi_nodes, g_nodes, logXi_fine, g_fine = z_from_dL_at_monotone(
            r=None,
            H0=H0,
            Om=Om,
            w0=w0,
            Lambda_MG=Lambda_MG,
            is_GP_dL=True,
            z_grid=Z_nodes,
            z_grid_fine=z_grid_fine,
            out_type=out_type,          # <-- compatible with your call
            eps_q=1e-6,
            use_log1p_input=True,
        )
        return dLGrid_at, logXi_nodes, g_nodes, logXi_fine, g_fine

    else:
        raise ValueError("gp_mode must be 'direct' or 'mono_reparam'")


def z_from_dL_at_0(
    r,
    H0,
    Om,
    w0,
    Lambda_MG,
    is_GP_dL,
    z_grid,
    z_grid_fine=None,
    out_type="fine",
    gp_mode="direct",     # "direct" (old) or "mono_reparam" (new)
    taper_z0=None,        # e.g. 0.02; None disables taper
):
    """
    If is_GP_dL:
      Returns dL grid plus (logXi, g=dlogXi/dz) on both coarse and fine grids.

    gp_mode:
      - "direct": GP draw is g(z)=d/dz log Xi (old behavior)
      - "mono_reparam": GP draw is latent u(z); enforce monotonicity by construction:
            g(z) = b(z) * (exp(u(z)) - 1)   (optionally tapered at low z)
        where b(z)=d/dz log dL_EM(z).

    out_type:
      - "fine": use fine grid to build dLGrid_at
      - else:  use coarse grid
    """
    if not is_GP_dL:
        Xi0, n = Lambda_MG
        dLGrid_at = dLfun_at(z_grid, H0, Om, w0, Xi0, n)
        return atinterp(r, dLGrid_at, z_grid)

    gp = Lambda_MG[0]
    Z_nodes = z_grid

    if z_grid_fine is None:
        z_grid_fine = zGrid500_at

    # --- get logXi and g on grids ---
    if gp_mode == "direct":
        logXi_nodes, g_nodes, logXi_fine, g_fine = compute_gp_interp_dist_ratio(
            Z_nodes, gp, name="f", z_fine=z_grid_fine, reparameterize=True
        )

    elif gp_mode == "mono_reparam":
        # b(z)=d/dz log dL_EM on nodes + fine
        b_nodes = d_log_dLEM_dz(Z_nodes, H0, Om, w0)
        b_fine  = d_log_dLEM_dz(z_grid_fine, H0, Om, w0)

        logXi_nodes, g_nodes, logXi_fine, g_fine = compute_gp_interp_dist_ratio_monotone(
            Z_nodes,
            gp,
            b_nodes=b_nodes,
            name="f",                  # keep compatibility
            z_fine=z_grid_fine,
            b_fine=b_fine,
            reparameterize=True,
            taper_z0=taper_z0,         # e.g. 0.02
        )

    else:
        raise ValueError("gp_mode must be 'direct' or 'mono_reparam'")

    # --- choose which grid to output ---
    if out_type == "fine":
        z_out = z_grid_fine
        Xi_out = at.exp(logXi_fine)
        logXi_out = logXi_fine
        g_out = g_fine
    else:
        z_out = Z_nodes
        Xi_out = at.exp(logXi_nodes)
        logXi_out = logXi_nodes
        g_out = g_nodes

    # --- build dL grid (GW) ---
    dCGrid_at = dcfun_at(z_out, H0, Om, w0)
    dLGrid_EM_at = dCGrid_at * (1.0 + z_out)
    dLGrid_at = Xi_out * dLGrid_EM_at

    # Return with the same structure you already expect downstream
    return dLGrid_at, logXi_nodes, g_nodes, logXi_fine, g_fine



def _inv_softplus_stable(y):
    """
    Stable inverse of softplus for y>0:
      softplus(x)=log(1+exp(x))
      inv_softplus(y)= y + log(-expm1(-y))
    Works well for very small and very large y.
    """
    return y + at.log(-at.expm1(-y))


def z_from_dL_at_monotone(
    r,
    H0,
    Om,
    w0,
    Lambda_MG,
    is_GP_dL,
    z_grid,
    z_grid_fine=None,
    out_type="fine",
    eps_q=1e-6,
    use_log1p_input=True,
):
    """
    Monotone-by-construction GP model using:
        q(z) = d/dz log dL_GW(z) = eps_q + softplus( mu_u(z) + f(z) )   > 0
        g(z) = d/dz log Xi(z) = q(z) - b(z)
    where b(z)=d/dz log dL_EM(z), and mu_u(z)=inv_softplus(b(z)-eps_q)
    centers the prior near GR (q≈b => g≈0).

    Returns same tuple as your existing functions:
      dLGrid_at, logXi_nodes, g_nodes, logXi_fine, g_fine

    Notes:
    - The sampled latent variable name is still "f" (compat).
    - If use_log1p_input=True, the GP lives on x=log(1+z) (recommended).
    """
    if not is_GP_dL:
        Xi0, n = Lambda_MG
        dLGrid_at = dLfun_at(z_grid, H0, Om, w0, Xi0, n)
        return atinterp(r, dLGrid_at, z_grid)

    gp = Lambda_MG[0]
    Z_nodes = at.as_tensor_variable(z_grid)

    if z_grid_fine is None:
        z_grid_fine = zGrid500_at
    Z_fine = at.as_tensor_variable(z_grid_fine)

    # --- GP latent f on nodes (optionally in x=log1p(z)) ---
    if use_log1p_input:
        X_nodes = at.log1p(Z_nodes)[:, None]
        x_fine = at.log1p(Z_fine)
        x_nodes_1d = at.log1p(Z_nodes)
    else:
        X_nodes = Z_nodes[:, None]
        x_fine = Z_fine
        x_nodes_1d = Z_nodes

    f_nodes = gp.prior("f", X=X_nodes, reparameterize=True)  # (N_nodes,)
    f_fine = atinterp(x_fine, x_nodes_1d, f_nodes)           # (N_fine,)

    # --- b(z)=d/dz log dL_EM on nodes + fine ---
    b_nodes = d_log_dLEM_dz(Z_nodes, H0, Om, w0)
    b_fine  = d_log_dLEM_dz(Z_fine,  H0, Om, w0)

    # --- GR-centered mean function mu_u = inv_softplus(b-eps) ---
    y_nodes = at.maximum(b_nodes - eps_q, 1e-12)
    y_fine  = at.maximum(b_fine  - eps_q, 1e-12)
    mu_u_nodes = _inv_softplus_stable(y_nodes)
    mu_u_fine  = _inv_softplus_stable(y_fine)

    # --- monotone slope q and implied g ---
    u_nodes = mu_u_nodes + f_nodes
    u_fine  = mu_u_fine  + f_fine

    q_nodes = eps_q + at.softplus(u_nodes)
    q_fine  = eps_q + at.softplus(u_fine)

    g_nodes = q_nodes - b_nodes
    g_fine  = q_fine  - b_fine

    # --- integrate g to get logXi, with logXi(0)=0 on the chosen grid ---
    def _integrate_midpoint(z, g):
        dz_raw = z[1:] - z[:-1]
        dz = at.clip(dz_raw, 1e-18, np.inf)
        g_mid = 0.5 * (g[:-1] + g[1:])
        inc = g_mid * dz
        h0 = at.as_tensor_variable(0.0)
        return at.concatenate([h0[None], h0 + at.cumsum(inc)])

    logXi_fine = _integrate_midpoint(Z_fine, g_fine)
    logXi_nodes = atinterp(Z_nodes, Z_fine, logXi_fine)

    # --- choose output grid for dLGrid_at ---
    if out_type == "fine":
        z_out = Z_fine
        Xi_out = at.exp(logXi_fine)
    else:
        z_out = Z_nodes
        Xi_out = at.exp(logXi_nodes)

    dCGrid_at = dcfun_at(z_out, H0, Om, w0)
    dLGrid_EM_at = dCGrid_at * (1.0 + z_out)
    dLGrid_at = Xi_out * dLGrid_EM_at

    return dLGrid_at, logXi_nodes, g_nodes, logXi_fine, g_fine


def z_from_dL_at_monotone_v0(r, H0, Om, w0, Lambda_MG, is_GP_dL, z_grid, z_grid_fine=None, out_type='fine'):
    if not is_GP_dL:
        Xi0, n = Lambda_MG
        dLGrid_at = dLfun_at(z_grid, H0, Om, w0, Xi0, n)
        return atinterp(r, dLGrid_at, z_grid)

    gp = Lambda_MG[0]
    Z_nodes = z_grid

    if z_grid_fine is None:
        z_grid_fine = zGrid500_at

    # b on nodes + fine
    b_nodes = d_log_dLEM_dz(Z_nodes, H0, Om, w0)
    b_fine  = d_log_dLEM_dz(z_grid_fine, H0, Om, w0)

    logXi_nodes, g_nodes, logXi_fine, g_fine = compute_gp_interp_dist_ratio_monotone(
        Z_nodes,
        gp,
        b_nodes=b_nodes,
        name="f",
        z_fine=z_grid_fine,
        b_fine=b_fine,
        reparameterize=True,
        taper_z0=0.02,   # match what you decided
    )

    if out_type == 'fine':
        z_grid_out = z_grid_fine
        dr_grid_out = at.exp(logXi_fine)   # Xi
        g_out = g_fine
        logXi_out = logXi_fine
    else:
        z_grid_out = Z_nodes
        dr_grid_out = at.exp(logXi_nodes)
        g_out = g_nodes
        logXi_out = logXi_nodes

    dCGrid_at = dcfun_at(z_grid_out, H0, Om, w0)
    dLGrid_EM_at = dCGrid_at * (1.0 + z_grid_out)
    dLGrid_at = dr_grid_out * dLGrid_EM_at

    return dLGrid_at, logXi_nodes, g_nodes, logXi_fine, g_fine



def z_from_dL_at_nomonotone(r, H0, Om, w0, Lambda_MG, is_GP_dL, z_grid, z_grid_fine=None, out_type='fine'): 
    
    if not is_GP_dL:
        Xi0, n = Lambda_MG
        dLGrid_at = dLfun_at( z_grid, H0, Om, w0, Xi0, n )
        return atinterp( r, dLGrid_at, z_grid)   
    
    else:
        gp = Lambda_MG[0]
        Z_nodes      = z_grid         # coarse, used for the GP

        if z_grid_fine is None:
            z_grid_fine = zGrid500_at
    
        log_distance_ratio, grad_log_distance_ratio, log_distance_ratio_fine, grad_log_distance_ratio_fine = compute_gp_interp_dist_ratio(
    Z_nodes, gp, name="f", z_fine=z_grid_fine, reparameterize=True
)

        if out_type=='fine':
            z_grid_out = z_grid_fine
            dr_grid_out=at.exp(log_distance_ratio_fine)
        else:
            z_grid_out = z_grid
            dr_grid_out=at.exp(log_distance_ratio)
        
        dCGrid_at  = dcfun_at( z_grid_out, H0, Om, w0 )
        dLGrid_EM_at = dCGrid_at*(1+z_grid_out)
        dLGrid_at = dr_grid_out*dLGrid_EM_at

        return dLGrid_at, log_distance_ratio, grad_log_distance_ratio, log_distance_ratio_fine, grad_log_distance_ratio_fine




    
def log_j_at(z, Om, H0=70, dc=None,  interp=False):
    if dc is None:
        dc = dcfun_at(z, H0, Om, interp=interp)
    dc*=H0/c_light*1e03
    return at.log(4*PI)+2*at.log(dc)-at.log(Efun_at(z, Om=Om))


def log_dV_dz_at(z, Lambdac, dc=None, interp=False):
    H0, Om0, w0 = Lambdac
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


def ddL_dz_EM(z, H0, Om0,  w0, dc=None):
    
    # H0 in Mpc, ds in Gpc
    
    if dc is None:
        dc = dcfun_at(z, H0, Om0,  w0, interp=False) # in Gpc
    
    
    res =  dc + c_light*(1+z)/(1e03*H0*Efun_at(z, Om0,  w0)) 


    return res


def d_log_dLEM_dz(z, H0, Om0,  w0, dc=None, safe=False):

    if dc is None:
        dc = dcfun_at(z, H0, Om0,  w0, interp=False) # in Gpc

    # print("In d_log_dLEM_dz")
    # print("z is ")
    # print(z.eval())
    # print("dc is")
    # print(dc.eval())

    E_ = Efun_at(z ,Om0,  w0)

    # print("Efun is ")
    # print(E_.eval())

    if not safe:
        first_term = 1/(1+z)
        second_term = c_light/dc/(1e03*H0*E_) 
    else:

        chi_prime = (c_light/1e03/H0) * safe_div(1.0, E_)
        second_term = safe_div(chi_prime , dc )
        first_term = safe_div(1.0, 1.0 + z) 
    
    # print("1/(E*dc is)")
    # print(second_term.eval())

    return first_term + second_term


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


def log_p_z_MD_unnorm(z, gamma, kappa, zp, Lambda_c, dc=None):
    #lC0 = at.log( 1+(1+zp)**(-gamma-kappa))
    
    log_psiz = log_psi_z_MD(z, gamma, kappa, zp) #gamma*at.log1p(z)-at.log(1+((1+z)/(1+zp))**(gamma+kappa))

    log_dVdz = log_dV_dz_at(z, Lambda_c, dc=dc )
    
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
    pz = at.exp( gamma*at.log1p(zz)+log_dV_dz_at(zz, H0, Om, w0,dc=dc )-at.log1p(zz) )
    norm = attrapzvec(pz, zz)
    
    return log_psiz+log_dVdz-at.log1p(z)-at.log(norm)



#####################################################
# Gaussian processes for d
#####################################################



def min_max_scaler(X_raw, data_range, feature_range=(0, 1)):
    data_min, data_max = data_range
    feature_min, feature_max = feature_range

    X_std = (X_raw - data_min) / (data_max - data_min)
    X_scaled = X_std * (feature_max - feature_min) + feature_min
    return X_scaled



def min_max_inverse_transform(X_scaled, data_range, feature_range=(0, 1)):
    data_min, data_max = data_range
    feature_min, feature_max = feature_range

    X_std = (X_scaled - feature_min) / (feature_max - feature_min)
    X_raw = X_std * (data_max - data_min) + data_min
    return X_raw





def frechet_logp_full(value, lambda_ell, d):
    """
    Fréchet-like kernel:
      log f(x) = log(alpha*lambda) - (alpha+1) log x - lambda * x^{-alpha},  x>0
    with alpha = d/2 > 0, lambda>0.
    """
    x   = at.as_tensor_variable(value)
    lam = at.as_tensor_variable(lambda_ell)
    d_  = at.as_tensor_variable(d)
    alpha = d_ / 2.0

    # core logp
    logp = (
        at.log(alpha * lam)
        - (alpha + 1.0) * at.log(x)
        - lam * at.power(x, -alpha)   # use at.power for JAX friendliness
    )

    # single boolean condition (no 'alltrue' needed)
    ok = (x > 0) & (lam > 0) & (d_ > 0)

    # return -inf outside support; keeps graph differentiable
    return check_parameters(logp, ok, msg="Frechet requires x>0, lambda>0, d>0")


def frechet_random(lambda_ell, d, size=None, rng=None):
    # Sample via the same reparam: U ~ Exp(lambda), ℓ = U^{-2/d}
    rng = onp.random.default_rng() if rng is None else rng
    lam = onp.asarray(lambda_ell, dtype=onp.float64)
    d_  = onp.asarray(d, dtype=onp.float64)
    alpha = d_ / 2.0
    u = rng.exponential(scale=1.0/lam, size=size)
    return u ** (-1.0 / alpha)


def frechet_logp_full_0(l, lambda_ell, d):
    return at.log(d * lambda_ell / 2) \
         - (d / 2 + 1) * at.log(l) \
         - lambda_ell * l ** (-d / 2)


def find_beta(L, alpha, p0=0.01):
    import scipy.stats as stats
    from scipy.optimize import bisect
    # Define function for root-finding: GammaCDF(L; alpha, beta) - p0 = 0
    def func(beta):
        return stats.gamma.cdf(L, a=alpha, scale=1/beta) - p0

    # beta must be positive, try searching between a small number and a large number
    beta_opt = bisect(func, 1e-6, 100)
    return beta_opt

def find_al(L, beta, p0=0.01):
    import scipy.stats as stats
    from scipy.optimize import bisect
    # Define function for root-finding: GammaCDF(L; alpha, beta) - p0 = 0
    def func(al):
        return stats.invgamma.cdf(L, a=al, scale=1/beta) - p0

    # beta must be positive, try searching between a small number and a large number
    alpha_opt = bisect(func, 1e-6, 100)
    return alpha_opt


#####################################################


def compute_gp_interp_dist_ratio_monotone(
    z_nodes,
    gp,
    b_nodes,                 # b(z)=d/dz log dL_EM on nodes (tensor)
    name="f",
    z_fine=None,
    b_fine=None,             # b(z) on fine grid (tensor). If None, interpolated from nodes.
    reparameterize=True,
    nugget=1e-4,             # must match WhiteNoise(1e-4)
    taper_z0=None,           # e.g. 0.02; None disables taper
    eps=0.0,                 # optional tiny buffer in exp argument (usually 0)
):
    """
    Samples u(z) ~ GP on nodes, builds monotone GW slope:
        q = b * exp(u)  > 0
        g = d/dz log Xi = q - b = b*(exp(u)-1)
    Integrates g to get logXi with logXi(0)=0 (=> Xi(0)=1).

    Returns (same structure as your original):
      logXi_nodes, g_nodes, logXi_fine, g_fine
    """
    z_nodes = at.as_tensor_variable(z_nodes)
    b_nodes = at.as_tensor_variable(b_nodes)

    # GP draw: u_raw on nodes
    X_nodes = z_nodes[:, None]
    u_nodes = gp.prior(name, X=X_nodes, reparameterize=reparameterize)  # (N_nodes,)

    # helper to compute g given (u, b, z)
    def _g_from_u(u, b, z):
        # GR-centered in median if u has median 0
        g = b * (at.exp(u + eps) - 1.0)

        if taper_z0 is not None:
            z0 = at.as_tensor_variable(float(taper_z0))
            w = 1.0 - at.exp(- (z / z0) ** 2)   # w(0)=0, w->1
            g = g * w
        return g

    # If no fine grid: integrate on nodes
    if z_fine is None:
        g_nodes = _g_from_u(u_nodes, b_nodes, z_nodes)

        dz_raw = z_nodes[1:] - z_nodes[:-1]
        dz     = at.clip(dz_raw, 1e-18, np.inf)
        g_mid  = 0.5 * (g_nodes[:-1] + g_nodes[1:])
        inc    = g_mid * dz
        h0     = at.as_tensor_variable(0.0)
        logXi_nodes = at.concatenate([h0[None], h0 + at.cumsum(inc)])
        return logXi_nodes, g_nodes

    # Fine grid path
    z_fine = at.as_tensor_variable(z_fine)

    # interpolate u to fine
    u_fine = atinterp(z_fine, z_nodes, u_nodes)

    # b on fine
    if b_fine is None:
        b_fine = atinterp(z_fine, z_nodes, b_nodes)
    else:
        b_fine = at.as_tensor_variable(b_fine)

    g_fine = _g_from_u(u_fine, b_fine, z_fine)

    dz_raw_f = z_fine[1:] - z_fine[:-1]
    dz_f     = at.clip(dz_raw_f, 1e-18, np.inf)
    g_mid_f  = 0.5 * (g_fine[:-1] + g_fine[1:])
    inc_f    = g_mid_f * dz_f
    h0       = at.as_tensor_variable(0.0)
    logXi_f  = at.concatenate([h0[None], h0 + at.cumsum(inc_f)])

    # map back to nodes
    logXi_nodes = atinterp(z_nodes, z_fine, logXi_f)
    g_nodes     = _g_from_u(u_nodes, b_nodes, z_nodes)

    return logXi_nodes, g_nodes, logXi_f, g_fine
    

def compute_gp_interp_dist_ratio(
    z_nodes,                 # (N_nodes,)  coarse GP grid (e.g., 150) — outputs will be here
    gp,                      # pm.gp.Latent already defined
    name="f",
    z_fine=None,             # (N_fine,)   optional finer grid (e.g., 500) used only for integration
    reparameterize=True,
    return_fine=False
):
    """
    1) Draw GP on coarse nodes z_nodes (cheap Cholesky).
    2) Interpolate GP to z_fine (if provided) for accurate integration.
    3) Integrate on z_fine with midpoint rule to get log_distance_ratio(z_fine).
    4) Interpolate the integrated curve back to z_nodes and RETURN there.

    Returns:
      log_distance_ratio_nodes : (N_nodes,)
      grad_log_distance_ratio  : (N_nodes,)  (= GP value at nodes)
    """
    z_nodes = at.as_tensor_variable(z_nodes)

    # --- GP on coarse nodes (this is the only GP call) ---
    X_nodes = z_nodes[:, None]
    g_nodes = gp.prior(name, X=X_nodes, reparameterize=reparameterize)  # (N_nodes,)

    # If no fine grid provided, integrate on nodes directly (your original behavior)
    if z_fine is None:
        dz_raw = z_nodes[1:] - z_nodes[:-1]
        dz     = at.clip(dz_raw, 1e-18, np.inf)
        g_mid  = 0.5 * (g_nodes[:-1] + g_nodes[1:])
        inc    = g_mid * dz
        h0     = at.as_tensor_variable(0.0)
        log_dr_nodes = at.concatenate([h0[None], h0 + at.cumsum(inc)])  # (N_nodes,)
        return log_dr_nodes, g_nodes

    # --- otherwise: integrate on fine grid and map back to nodes ---
    z_fine = at.as_tensor_variable(z_fine)

    # 1) interpolate GP to fine grid (cheap O(N_fine))
    #    assumes you already have a monotone atinterp(xq, x, y)
    g_fine = atinterp(z_fine, z_nodes, g_nodes)  # (N_fine,)

    # 2) midpoint integration on fine grid
    dz_raw_f = z_fine[1:] - z_fine[:-1]
    dz_f     = at.clip(dz_raw_f, 1e-18, np.inf)     # (N_fine-1,)
    g_mid_f  = 0.5 * (g_fine[:-1] + g_fine[1:])     # (N_fine-1,)
    inc_f    = g_mid_f * dz_f
    h0       = at.as_tensor_variable(0.0)
    log_dr_f = at.concatenate([h0[None], h0 + at.cumsum(inc_f)])  # (N_fine,)

    # 3) interpolate the integrated curve back to the coarse nodes
    log_dr_nodes = atinterp(z_nodes, z_fine, log_dr_f)            # (N_nodes,)
                
    # Derivative at nodes is exactly the GP at nodes
    return log_dr_nodes, g_nodes, log_dr_f, g_fine


def compute_gp_interp_dist_ratio_0( z_grid, gp, data_range=None, name="f", res=1000, GP_zero_point='y' , dense_grad = False , eta=None, ell=None, nu=None, sgn=None, b_full=None):

    
        
    # X_test = z_grid[:, None]
    
    # g  = gp.prior(name, X=X_test, reparameterize=True)  # (N,)
    # s_floor = at.as_tensor_variable(1e-4).astype(g.dtype)
    # s = s_floor + softplus_stable(g)                    # (N,), strictly > 0
    
    # dz = z_grid[1:] - z_grid[:-1]                       # (N-1,)
    
    # # NEW: h'(z) = nu * s(z)  (no lb involved)
    # hprime_L = nu * s[:-1]                              # (N-1,)
    # hprime_R = nu * s[1:]                               # (N-1,)
    # inc = 0.5 * (hprime_L + hprime_R) * dz             # trapezoid increments
    
    # # Integrate with h(z0)=0 ⇒ Xi(z0)=1
    # h0 = at.constant(0.0)
    # log_distance_ratio = at.concatenate([h0[None], h0 + at.cumsum(inc)])   # (N,)
    
    # # Node-aligned derivative (length-weighted average on irregular grid)
    # grad_log_distance_ratio = at.concatenate([
    #     hprime_L[:1],
    #     (dz[:-1]*hprime_R[:-1] + dz[1:]*hprime_L[1:]) / (dz[:-1] + dz[1:]),
    #     hprime_R[-1:]
    # ])


    
    # X_test = z_grid[:, None]

    # g  = gp.prior(name, X=X_test, reparameterize=True)  # (N,)
    # s_floor = at.as_tensor_variable(1e-4).astype(g.dtype)
    # s = s_floor + softplus_stable(g)                    # (N,), strictly > 0
    
    # dz = z_grid[1:] - z_grid[:-1]                       # (N-1,)
    
    # # --- NEW PART: allow h' to be ± while keeping d log dL^GW/dz > 0 ---
    
    # # EM slope b_full = d/dz log dL_EM  (all > 0)
    
    # # Midpoint versions (more accurate & matches dz)
    # b_mid  = 0.5 * (b_full[:-1] + b_full[1:])           # (N-1,)
    # s_mid  = 0.5 * (s[:-1] + s[1:])                     # (N-1,)
    
    # # Total slope of log dL^GW at midpoints; sgn∈(-1,+1) lets h go up or down.
    # # softplus keeps total slope strictly positive ⇒ d_L^GW monotone increasing.
    # q_mid = softplus_stable(b_mid + sgn * nu * s_mid)   # (N-1,)
    
    # # delta = sgn * nu * s_mid  (same as before)
    # delta_mid = sgn * nu * s_mid                    # (N-1,)
    
    # # Stable exact formula: h'(z) = softplus(b+delta) - b = logaddexp(-b, delta)
    # hprime_mid = at.logaddexp(-b_mid, delta_mid)    # (N-1,)
    
    
    # # Integrate with midpoint rule
    # inc = hprime_mid * dz                                  # (N-1,)
    # h0  = at.constant(0.0)
    # log_distance_ratio = at.concatenate([h0[None], h0 + at.cumsum(inc)])  # (N,)
    
    # # Optional: node-aligned derivative from midpoint values (length-weighted avg)
    # grad_log_distance_ratio = at.concatenate([
    #     hprime_mid[:1],
    #     (dz[:-1]*hprime_mid[:-1] + dz[1:]*hprime_mid[1:]) / (dz[:-1] + dz[1:]),
    #     hprime_mid[-1:]
    # ])


    # X_test = z_grid[:, None]

    # g  = gp.prior(name, X=X_test, reparameterize=True)  # (N,)
   
    # dz = z_grid[1:] - z_grid[:-1]                       # (N-1,)
        
    # # EM slope is b_full = d/dz log dL_EM  (all > 0)
       
    # # Midpoint versions (more accurate & matches dz)
    # b_mid  = 0.5 * (b_full[:-1] + b_full[1:])           # (N-1,)
    # g_mid  = 0.5 * (g[:-1] + g[1:])                     # (N-1,)
    
    # # Stable exact formula: h'(z) = g 
    # hprime_mid =  g_mid #*b_mid
    
    
    # # Integrate with midpoint rule
    # inc = hprime_mid * dz                                  # (N-1,)
    # h0  = at.constant(0.0)
    # log_distance_ratio = at.concatenate([h0[None], h0 + at.cumsum(inc)])  # (N,)
    
    # # Optional: node-aligned derivative from midpoint values (length-weighted avg)
    # grad_log_distance_ratio = g #at.concatenate([hprime_mid[:1], (dz[:-1]*hprime_mid[:-1] + dz[1:]*hprime_mid[1:]) / (dz[:-1] + dz[1:]),hprime_mid[-1:]])


    # --- inputs: z_grid (N,), gp object already defined ---
    z_grid = at.as_tensor_variable(z_grid)              # ensure tensor
    # small safety to avoid zero/negative spacings due to any numerical issues
    dz_raw = z_grid[1:] - z_grid[:-1]
    dz = at.clip(dz_raw, 1e-18, np.inf)                 # (N-1,)
    
    # Midpoints for integration (more accurate on nonuniform grids)
    z_mid = 0.5 * (z_grid[1:] + z_grid[:-1])           # (N-1,)
    X_node = z_grid[:, None]                            # (N, 1)
    X_mid  = z_mid[:, None]                             # (N-1, 1)
    
    # ---- OPTION SELECTOR ----
    USE_MIDPOINT_GP = False  # True → draw GP at midpoints too (more accurate)
    
    # Draw GP at nodes (always needed for the derivative reported at nodes)
    g_nodes = gp.prior(name, X=X_node, reparameterize=True)  # (N,)
    
    if USE_MIDPOINT_GP:
        # Option B: evaluate integrand at true midpoints
        g_mid = gp.prior(name+"_mid", X=X_mid, reparameterize=True)   # (N-1,)
    else:
        # Option A (default): midpoint average of node values
        g_mid = 0.5 * (g_nodes[:-1] + g_nodes[1:])                # (N-1,)
    
    # Midpoint rule integral: h'(z) = g(z)
    inc = g_mid * dz                                              # (N-1,)
    h0  = at.as_tensor_variable(0.0)
    log_distance_ratio = at.concatenate([h0[None], h0 + at.cumsum(inc)])  # (N,)
    
    # Node-aligned derivative (the GP itself at nodes)
    grad_log_distance_ratio = g_nodes                             # (N,)
    
    
    return log_distance_ratio, grad_log_distance_ratio




# derivative cross-cov for Matérn-5/2 (1D), stable (no 1/r)
def matern52_dcov_dx_1d(Xd, Xc, eta, ell):
    xd   = at.as_tensor_variable(Xd).ravel()[:, None]
    xc   = at.as_tensor_variable(Xc).ravel()[None, :]
    diff = xd - xc
    r    = at.abs(diff)
    a    = at.sqrt(5.0) / ell
    expm = at.exp(-a * r)
    coef = -(5.0 * (eta**2) / 3.0) * expm
    return coef * ( diff / (ell**2) + at.sqrt(5.0) * r * diff / (ell**3) )


def matern52_1d(X, Y, eta, ell):
    X = np.atleast_2d(X)#.astype(np.float64)
    Y = np.atleast_2d(Y)#.astype(np.float64)
    d = np.abs(X - Y.T)
    a = np.sqrt(5.0) / ell
    return (eta**2) * (1.0 + a*d + 5.0*(d**2)/(3.0*ell**2)) * np.exp(-a*d)


def make_gp_mapper(gp, Xc, eta, ell):
    """
    Prepare a callable that, given any X_new, returns linear maps
    T_new, A_new so that:
      f(X_new)  = T_new @ f_c
      f'(X_new) = A_new @ f_c
    Reuses the same Kcc factorization from the coarse grid Xc.
    """
    Xc = at.as_tensor_variable(Xc)
    Kcc = gp.cov_func(Xc[:, None])                  # includes WhiteNoise on the diagonal
    Lcc = at.linalg.cholesky(Kcc)                   # cached factor

    # n = Kcc.shape[0]
    # #diag = at.diag(Kcc)
    # # scale-aware jitter: tie it to the average variance
    # #base = at.switch(at.all(at.isfinite(diag)), at.mean(diag), 1.0)
    # jitter = 1e-6   # tweak 1e-10 ↔ 1e-6 if needed
    
    # Lcc = at.linalg.cholesky(Kcc + jitter * at.eye(n) )

    def maps(X_new):
        X_new = at.as_tensor_variable(X_new)
        # Values map: T = K(X_new, Xc) Kcc^{-1}
        Kvc = gp.cov_func(X_new[:, None], Xc[:, None])   # no white-noise cross terms
        Yv  = at.linalg.solve(Lcc,  Kvc.T)
        Tv  = at.linalg.solve(Lcc.T, Yv).T
        # Derivative map: A = K'(X_new, Xc) Kcc^{-1}
        Kdc = matern52_dcov_dx_1d(X_new, Xc, eta, ell)
        Yd  = at.linalg.solve(Lcc,  Kdc.T)
        Ad  = at.linalg.solve(Lcc.T, Yd).T
        return Tv, Ad

    return maps


SQRT2   = onp.sqrt(2.0)
SQRTPI  = onp.sqrt(np.pi)
LOGHALF = onp.log(0.5)

def log_ndtr_at(x):
    """
    Stable log Φ(x) for all x.
    Uses:
      - direct: log(0.5*erfc(-x/sqrt(2))) when safe
      - asymptotic: for large negative x to avoid underflow
    Works elementwise on tensors.
    """
    y = -x / SQRT2                      # y >= 0 when x <= 0
    # direct (safe for y not too large)
    direct = LOGHALF + at.log(at.erfc(y) + 1e-300)

    # asymptotic for large y: erfc(y) ~ exp(-y^2)/(sqrt(pi)*y) * (1 - 1/(2y^2) + 3/(4y^4))
    invy  = 1.0 / at.clip(y, 1e-300, np.inf)
    invy2 = invy * invy
    corr  = 1.0 - 0.5 * invy2 + 0.75 * invy2 * invy2  # 1 - 1/(2y^2) + 3/(4y^4)
    asym  = (LOGHALF
             - y*y
             - 0.5 * at.log(np.pi)
             - at.log(at.clip(y, 1e-300, np.inf))
             + at.log(at.clip(corr, 1e-300, np.inf)))

    # switch to asymptotics when y is large (threshold ~5 is plenty for float64)
    use_asym = at.gt(y, 5.0)
    return at.where(use_asym, asym, direct)

def log_norm_cdf_at(x, sigma, clip=True):
    """
    log Φ(x / sigma). sigma > 0 (can be scalar/broadcastable).
    """
    if clip:
        return log_ndtr_at( at.clip( x / sigma, -10, 10) ) 
    else:
        log_ndtr_at( x / at.clip(sigma, 1e-300, np.inf)) 


#####################################################
#####################################################


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


def _log_ndtr(z):
    """
    log Phi(z) computed as log(0.5*erfc(-z/sqrt(2))).
    This is reasonably stable and returns -inf in extreme left tail (OK).
    """
    sqrt2 = at.sqrt(2.0)
    return at.log(0.5) + at.log(at.erfc(-z / sqrt2))


def truncGausslowerupper_at_lpdf_safe(x, loc, scale, xmin=0.0, xmax=1.0,
                                     eps_scale=1e-12, eps_Z=1e-300):
    """
    Logpdf of N(loc, scale) truncated to [xmin, xmax], returning -inf outside.
    Numerically stable normalizer using log-space CDF difference.

    - eps_scale prevents division by 0 in gradients
    - eps_Z prevents log(0) or log(negative) from producing NaN;
      when the truncation mass is truly ~0, you get a large negative logZ,
      which correctly makes the density extremely small / problematic but finite.
    """
    # ensure positive scale (still keep gradients sane)
    scale_pos = at.maximum(scale, eps_scale)

    # standardized bounds
    za = (xmin - loc) / scale_pos
    zb = (xmax - loc) / scale_pos

    logPhia = _log_ndtr(za)
    logPhib = _log_ndtr(zb)

    # enforce ordering for numerical safety (should already be true since xmax>=xmin)
    hi = at.maximum(logPhib, logPhia)
    lo = at.minimum(logPhib, logPhia)

    logZ = logdiffexp(hi, lo)

    # floor logZ to avoid NaNs when Z underflows / becomes 0
    logZ = at.maximum(logZ, at.log(eps_Z))

    # base normal logpdf
    z = (x - loc) / scale_pos
    logp = (-at.log(scale_pos)
            - 0.5 * at.log(2.0 * PI)
            - 0.5 * z**2
            - logZ)

    in_bounds = (x >= xmin) & (x <= xmax)
    return at.where(in_bounds, logp, -np.inf)


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
    t = (m - ml) / at.maximum(deltam, eps) #at.as_tensor_variable(eps).astype(m.dtype))
    t = at.clip(t, 0.0, 1.0)

    # smoothstep: S(t) = 3t^2 - 2t^3, monotone from 0→1 with zero slope at ends
    S = t * t * (3.0 - 2.0 * t)

    # log S, safely (avoid log(0) at the lower edge)
    return at.log(at.clip(S, eps, 1.0))
    
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



def log_norm_truncated_pl_num_alpha1_safe(alpha, mmin, mmax, eps=1e-12, t_floor=1e-12):
    """
    Wrapper around your log_norm_truncated_pl_num that is well-defined at alpha==1.
    Uses a tiny floor on t = 1 - alpha only when |t| is extremely small.
    This preserves the correct continuous limit and avoids log(0).
    """
    # sanitize bounds similarly to your function
    mmin_c = at.clip(mmin, eps, np.inf)
    mmax_c = at.clip(mmax, eps, np.inf)
    mmax_c = at.maximum(mmax_c, mmin_c * (1.0 + 1e-12))

    t = 1.0 - alpha
    # ensure t is never exactly 0 (for numerical definition)
    t_safe = at.where(at.abs(t) < t_floor, at.sign(t) * t_floor + t_floor, t)

    b = at.log(mmin_c)
    delta = at.log(mmax_c) - b

    return (t_safe * b
            + at.log(at.abs(at.expm1(t_safe * delta)))
            - at.log(at.abs(t_safe)))


def log_norm_truncated_pl_num(alpha, mmin, mmax, eps=1e-12):
    """
    log ∫_{mmin}^{mmax} m^{-alpha} dm
    = log( (mmax^(1-α) - mmin^(1-α)) / (1-α) ), with a stable α≈1 branch.
    """
    # tensors + guards
    epsv  = at.as_tensor_variable(eps).astype(mmin.dtype)

    mmin_c = at.clip(mmin, epsv, INF)
    mmax_c = at.maximum(at.clip(mmax, epsv,INF), mmin_c * (1.0 + 1e-12))

    t = 1.0 - alpha  # t = 1 - α
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
    
    p2 = at.exp(logpdfm2_PLP_noreg( xx , beta, deltam, ml, smoothing=smoothing))
    cdf = atcumtrapz(p2, xx, )
    itr = atinterp( m, xx[1:], at.log(cdf) )
    return itr



def logNorm_PLP_reg( lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing='LVK', res=1000):
    
    '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

    '''
     
            
    # lower edge
    #ms1 = at.linspace(ml, 15, res)
    
    # before gaussian peak
    #ms2 = at.linspace( 15.1, 25, res )
    
    # around gaussian peak
    #ms3= at.linspace( 25.1, 40, res)
    
    # after gaussian peak
    #ms4 = at.linspace(40.1, mh, int(res/2) )
    
    #ms=at.concatenate([ms1,ms2, ms3, ms4] )
    #ms = at.linspace(ml, mh, res)

    _tgrid = _get_t_grid()
    ms = ml + (mh - ml) * _tgrid 
    
    ps = at.exp( logpdfm1_PLP_noreg( ms , lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing=smoothing  ))
    return at.log(attrapzvec(ps,ms))



####### double Power Law + double Peak  LVK low-end ########


def log_broken_pl_norm_DPLDP(alpha1, alpha2, mb, m1_low, m_high,
                            eps=1e-12, t_floor=1e-12):
    """
    Log normalization for broken PL defined as:
      p(m) ∝ (m/mb)^(-alpha1) for m <= mb
      p(m) ∝ (m/mb)^(-alpha2) for m >= mb
    integrated over m in [m1_low, m_high].

    Uses u = m/mb, so:
      N = mb * [ ∫_{u_low}^{1} u^{-alpha1} du + ∫_{1}^{u_high} u^{-alpha2} du ]
    with stable log integrals and a well-defined alpha==1 case (via t_floor).
    """
    mb_pos = at.maximum(mb, eps)

    u_low = at.maximum(m1_low / mb_pos, eps)
    u_high = at.maximum(m_high / mb_pos, u_low * (1.0 + 1e-12))

    one = at.as_tensor_variable(1.0, dtype=getattr(mb, "dtype", "float64"))

    # Left side exists only if u_low < 1
    logI1 = at.switch(
        u_low < one,
        log_norm_truncated_pl_num_alpha1_safe(alpha1, u_low, one, eps=eps, t_floor=t_floor),
        -np.inf
    )

    # Right side exists only if u_high > 1
    logI2 = at.switch(
        u_high > one,
        log_norm_truncated_pl_num_alpha1_safe(alpha2, one, u_high, eps=eps, t_floor=t_floor),
        -np.inf
    )

    # log( mb * (I1 + I2) )
    return at.log(mb_pos) + at.logaddexp(logI1, logI2)


def log_broken_power_law_DPLDP_pdf(m1, alpha1, alpha2, mb, m1_low, m_high,
                                  sh=0.05, sl=0.05, epsilon=0.01,
                                  eps=1e-12, eps_w=1e-12, t_floor=1e-12):
    """
    Broken-PL component logpdf (no outer low/high gating here).
    Uses safe log normalization that is well-defined at alpha==1.

    NOTE:
      If you have moved the low/high sigmoid envelope to logpdfm1_DPLDP (recommended),
      do NOT re-add s1/s2 here (to avoid double gating).
    """
    # Keep logs safe (masses positive by construction, but guard anyway)
    mb_pos = at.maximum(mb, eps)
    m1_pos = at.maximum(m1, eps)

    # log normalization (safe)
    log_N = log_broken_pl_norm_DPLDP(alpha1, alpha2, mb_pos, m1_low, m_high,
                                    eps=eps, t_floor=t_floor)

    # log(m1/mb)
    log_m1_over_mb = at.log(m1_pos / mb_pos)
    log_val1 = -alpha1 * log_m1_over_mb
    log_val2 = -alpha2 * log_m1_over_mb

    # Smooth transition weight (already clipped in your sigmoid, but clip again here)
    w = safe_sigmoid(-m1_pos, -mb_pos, epsilon)
    w = at.clip(w, eps_w, 1.0 - eps_w)

    log_w = at.log(w)
    log_1mw = at.log1p(-w)

    # stable mixture
    log_mix_val = at.logaddexp(log_w + log_val1, log_1mw + log_val2)

    return log_mix_val - log_N 

def logpdfm1_DPLDP(m1, alpha1, alpha2, mb,
    mu1, sigma1, mu2, sigma2,
    m1_low, m_high, delta_m1,
    lambda0, lambda1, lambda2,
    epsilon,
    smoothing='LVK', simplex_repair=False, eps_w=1e-15, sl=0.05, sh=0.05,
    norm_gauss='uplow'
                  ):


    #work_dtype = getattr(m1, "dtype", "float64")

    #one = at.as_tensor_variable(1.0, dtype=work_dtype)

    # eps_w = at.as_tensor_variable(
    #     1e-6 if str(work_dtype) == "float32" else 1e-12,
    #     dtype=work_dtype
    # )

    if not simplex_repair:
        log_lambda0 = at.log(lambda0)
        log_lambda1 = at.log(lambda1)
        log_lambda2 = at.log(lambda2)

        #log_lambda0 = at.log(at.clip(lambda0, eps_w, 1.0-eps_w))
        #log_lambda1 = at.log(at.clip(lambda1, eps_w, 1.0-eps_w))
        #log_lambda2 = at.log(at.clip(lambda2, eps_w, 1.0-eps_w))

        #lambda2_raw  = 1. - lambda0 - lambda1
        #lambda2_safe = at.clip(lambda2_raw, eps_w, 1.0-eps_w)
        #log_lambda2  = at.log(lambda2_safe)

    else:
        # ---- Simplex repair (same math as your version; just dtype-safe) ----
        lam0 = at.clip(lambda0, eps_w, 1.-eps_w)
        lam1 = at.clip(lambda1, eps_w, 1.-eps_w)
        lam2_raw = 1. - lam0 - lam1

        lam2 = eps_w + at.softplus(lam2_raw - eps_w)

        denom = lam0 + lam1 + lam2
        lam0 = lam0 / denom
        lam1 = lam1 / denom
        lam2 = lam2 / denom

        log_lambda0 = at.log(lam0)
        log_lambda1 = at.log(lam1)
        log_lambda2 = at.log(lam2)

    log_ppl    = log_broken_power_law_DPLDP_pdf(m1, alpha1, alpha2, mb, m1_low, m_high, epsilon=epsilon)

    if norm_gauss=='uplow':
        print("gaussian components truncated and normalized at lower and upper end")
        log_pnorm1 = truncGausslowerupper_at_lpdf_safe(m1, mu1, sigma1, xmin=m1_low, xmax=m_high)
        log_pnorm2 = truncGausslowerupper_at_lpdf_safe(m1, mu2, sigma2, xmin=m1_low, xmax=m_high)
    elif norm_gauss=='low':
        print("gaussian components normalized only at lower end")
        raise NotImplementedError()
    elif norm_gauss=='none':
        print("gaussian components not truncated and not normalized")
        log_pnorm1 = -0.5*((m1-mu1)/sigma1)**2 - at.log(sigma1) - 0.5*at.log(2*PI)
        log_pnorm2 = -0.5*((m1-mu2)/sigma2)**2 - at.log(sigma2) - 0.5*at.log(2*PI)
    else:
        raise ValueError("norm_gauss can be uplow, low, or none")
        

    if smoothing == 'LVK':
        log_S = logS_PLP_LVK(m1, delta_m1, m1_low)
    else:
        log_S = logS_PLP(m1, delta_m1, m1_low)

    term0 = log_lambda0 + log_ppl
    term1 = log_lambda1 + log_pnorm1
    term2 = log_lambda2 + log_pnorm2

    #log_mix = safe_logsumexp3(term0, term1, term2)
    
    log_mix = logsumexp(
        logsumexp(term0, term1),
        term2
    )

    log_gate = log_sigmoid(m1, m1_low, sl) + at.log1p(-safe_sigmoid(m1, m_high, sh)) 

    return log_S + log_mix + log_gate




def logpdf_DPLDP(theta, lambdaBBHmass, force_m2_less_than_m1=False, has_m2_break=False, smoothing='LVK', resC=100, resN=500, interp_vals=None, interp_grids=None, norm=True, simplex_repair=False, norm_gauss='uplow'):
    
        m1, m2 = theta
        alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, lambda2, beta, m2_low, delta_m2, epsilon, m_g, w_g, sig_g_low, sig_g_high = lambdaBBHmass
                

        if interp_vals is None:
            
            lpdfm1 = logpdfm1_DPLDP( m1, alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, lambda2, epsilon, smoothing=smoothing, simplex_repair=simplex_repair, norm_gauss=norm_gauss)
        
            lpdfm2 = logpdfm2_PLP_reg(m2, beta, delta_m2, m2_low, m_g=m_g, w_g=w_g, sig_g_low = sig_g_low, sig_g_high = sig_g_high, has_m2_break=has_m2_break, smoothing=smoothing)
            
            lC = logC_DPLDP(m1, beta, delta_m2,  m2_low, m_g=m_g, w_g=w_g, sig_g_low = sig_g_low, sig_g_high = sig_g_high, has_m2_break=has_m2_break, smoothing=smoothing, res=resC) 
            if norm:
                ln = logNorm_DPLDP(  alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, lambda2, epsilon, smoothing=smoothing, res=resN, norm_gauss=norm_gauss )
            else:
                ln=0.
        else:

            m1_grid = interp_grids[0]
            m2_grid = interp_grids[1]
            lp_m1_grid, lp_m2_grid, lC_of_m1, ln = interp_vals


            ############ Faster, v3
            # ----- M1 interpolation indices computed once -----
            x0_1 = m1_grid[0]
            x1_1 = m1_grid[-1]
            nU_1 = m1_grid.shape[0]
            
            j1, r1 = uniform_interp_indices(m1, x0_1, x1_1, nU_1)
            
            # interpolate logpdf(m1)
            lpdfm1 = (1 - r1) * lp_m1_grid[j1] + r1 * lp_m1_grid[j1 + 1]
            
            # interpolate C(m1)
            lC     = (1 - r1) * lC_of_m1[j1]   + r1 * lC_of_m1[j1 + 1]
            
            # ----- M2 interpolation indices computed once -----
            x0_2 = m2_grid[0]
            x1_2 = m2_grid[-1]
            nU_2 = m2_grid.shape[0]
            
            j2, r2 = uniform_interp_indices(m2, x0_2, x1_2, nU_2)
            
            # interpolate logpdf(m2)
            lpdfm2 = (1 - r2) * lp_m2_grid[j2] + r2 * lp_m2_grid[j2 + 1]
        


        lpdf = lpdfm1 + lpdfm2 -lC -ln
    
        #lpdf = at.switch(
        #                    (at.isinf(lpdfm1) & (lpdfm1 < 0)) | (at.isinf(lpdfm2) & (lpdfm2 < 0)),
        #                    MIN,
        #                    lpdfm1 + lpdfm2 - lC - ln
        #                )
        

        if force_m2_less_than_m1:
            eval = at.and_(at.and_(m2 <= m1, m2 > 0), m1 > 0)
            return at.where(eval, lpdf, -np.inf)
        else:
            return lpdf        
     

def logC_DPLDP(m, beta, deltam, m2_low,
              m_g=45, w_g=80, sig_g_low=5, sig_g_high=5,
              has_m2_break=False, res=500, smoothing='LVK'):

    if res != 500:
        _tgrid = at.linspace(0, 1, res)
    else:
        _tgrid = _get_t_grid()

    xx = m2_low + (max_m - m2_low) * _tgrid

    l2 = logpdfm2_PLP_reg(xx, beta, deltam, m2_low,
                         m_g=m_g, w_g=w_g, sig_g_low=sig_g_low, sig_g_high=sig_g_high,
                         has_m2_break=has_m2_break, smoothing=smoothing)

    a = at.max(l2)
    p2 = at.exp(l2 - a)

    cdf = atcumtrapz(p2, xx)
    cdf = at.clip(cdf, 1e-300, np.inf)

    x0 = xx[1]
    x1 = xx[-1]
    nU = xx.shape[0] - 1

    # log(cdf_scaled) + a gives log(cdf_original)
    itr = atinterp_uniform(m, x0, x1, nU, at.log(cdf) + a)

    return itr





def logNorm_DPLDP(alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2,
                  m1_low, m_high, delta_m1,
                  lambda0, lambda1, lambda2,
                  epsilon,
                  res=500, smoothing='LVK', simplex_repair=False,
                  eps_int=1e-300, norm_gauss='uplow'):
    """
    Overflow-safe log normalization:
      log ∫ exp(logpdfm1_DPLDP(ms)) dms
    using max-subtraction.
    """
    if res != 500:
        _tgrid = at.linspace(0, 1, res)
    else:
        _tgrid = _get_t_grid()

    ms = m1_low + (m_high - m1_low) * _tgrid

    lpdf = logpdfm1_DPLDP(
        ms, alpha1, alpha2, mb,
        mu1, sigma1, mu2, sigma2,
        m1_low, m_high, delta_m1,
        lambda0, lambda1, lambda2,
        epsilon,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
        norm_gauss=norm_gauss
    )

    a = at.max(lpdf)
    ps = at.exp(lpdf - a)                 # <= 1, avoids overflow
    integ = attrapzvec(ps, ms)
    integ = at.clip(integ, eps_int, np.inf)

    return a + at.log(integ)






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

