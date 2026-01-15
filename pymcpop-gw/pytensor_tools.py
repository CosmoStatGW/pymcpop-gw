#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import pytensor.tensor as at

import pymc as pm

from pytensor.graph import Apply, Op
import pytensor
import numpy as onp
import numpy as np


from pytensor.gradient import disconnected_grad as stop_grad
from pymc.distributions.dist_math import check_parameters



# def _const_like(x, v):
#     """Create a scalar constant with dtype matching x (no Cast op)."""
#     return at.constant(v, dtype=getattr(x, "dtype", "float64"))


import pade_cosmo as pc

p, q = pc.flat_wcdm_pade_coefficients(w0=-1.0, zpower=0)  # arrays of floats



c_light = 299792458*1e-03
#c_light_at = at.as_tensor_variable(c_light)
NINF = at.as_tensor_variable(-np.inf)  
INF = at.as_tensor_variable(np.inf)


MIN = -np.inf #NINF # your "effectively -inf" : NINF or EPS
MAX = np.inf #INF


tiny = 1e-300
_eps_for_div = 1E-30
neg_inf = MIN

PI = np.pi

 
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



zGridGlobals_low = make_z_grid()


zGridGlobals_at_low = stop_grad(at.as_tensor_variable(zGridGlobals_low))



zGridGlobals_high = make_z_grid(1000)

zGridGlobals_at_high = stop_grad(at.as_tensor_variable(zGridGlobals_high))


max_m = 500.


_mass_grid_np = onp.unique(
    onp.concatenate([
        onp.linspace(1.0e-3, 15.0, 500, ),
        onp.linspace(15.01, 100.0, 1000, ),
        onp.linspace(101.1, max_m, 500, ),
    ])
)
_mass_grid_np.sort()
_mass_grid_at = stop_grad(at.as_tensor_variable(_mass_grid_np))

def _get_mass_grid():
    return _mass_grid_at



_tgrid  = onp.linspace(0.0, 1.0, 500)
_tgrid_at = stop_grad(at.as_tensor_variable(_tgrid))

_tgrid_1000  = onp.linspace(0.0, 1.0, 1000)
_tgrid_at_1000 = stop_grad(at.as_tensor_variable(_tgrid_1000))

_tgrid_100  = onp.linspace(0.0, 1.0, 100)
_tgrid_at_100 = stop_grad(at.as_tensor_variable(_tgrid_100))

def _get_t_grid():
    return _tgrid_at

def _get_t_grid_100():
    return _tgrid_at_100

def _get_t_grid_1000():
    return _tgrid_at_1000
    


##########################
####### Auxiliary functions ########
##########################


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




def soft_constraint_leq(x, y, k=50.0):
    """
    Smooth barrier enforcing x <= y.
    Returns ~0 when x<=y and ~-(x-y) when violated (scaled by k).
    """
    return -at.softplus(k * (x - y)) / k

    

def logdiffexp(a, b, eps=1e-16):
    """
    Stable log(exp(a) - exp(b)) elementwise.
    Returns -inf where b >= a (rather than NaN).
    """
    # ensure b-a <= 0 to keep exp <= 1
    delta = at.minimum(b - a, 0.0 )  # <= 0
    # exp(delta) in [0,1]
    ed = at.exp(delta)

    # protect log1p( -1 ) when ed==1 (i.e. a==b)
    #eps = _const_like(a, 1e-7) if str(getattr(a, "dtype", "")).endswith("32") else _const_like(a, 1e-16)
    out = a + at.log1p(-at.minimum(ed, 1.0 - eps))

    # if b >= a -> set to -inf
    #neg_inf = _const_like(a, -np.inf)
    return at.where(b < a, out, neg_inf)



def logsumexp(x, y):
    """`log(exp(x)+exp(y))` """
    #return x + at.log1p(at.exp(y-x))
    return at.logaddexp(x, y)


def safe_logsumexp(x, axis=None, keepdims=False):
    """
    Numerically stable logsumexp for PyTensor/JAX.
    Uses standard max-shift trick and avoids log(0) with a tiny floor.
    """
    #x = at.as_tensor_variable(x)
    # dtype = getattr(x, "dtype", "float64")

    # if dtype == "float32":
    #     tiny = at.as_tensor_variable(1e-20, dtype=dtype)
    # else:
    #     tiny = at.as_tensor_variable(1e-300, dtype=dtype)

    # max over the axis for stability
    xmax = at.max(x, axis=axis, keepdims=True)

    # subtract max (so at least one element is 0 → exp(0)=1)
    shifted = x - xmax

    # compute sum of exponentials
    sumexp = at.sum(at.exp(shifted), axis=axis, keepdims=keepdims)

    # avoid log(0) just in case (e.g. all -inf, or extreme underflow)
    sumexp_safe = at.maximum(sumexp, tiny)

    out = xmax + at.log(sumexp_safe)
    if not keepdims and axis is not None:
        out = at.squeeze(out, axis=axis)

    return out



def safe_logsumexp3(a, b, c):
    """
    Stable elementwise log(exp(a) + exp(b) + exp(c)).
    No stack/clip. Avoids NaNs from (-inf) - (-inf).
    """


    m = at.maximum(a, at.maximum(b, c))
    #neg_inf = _const_like(m, -np.inf)
    #zero = _const_like(m, 0.0)

    # If m is -inf, use 0 for the shift so (a - m_safe) doesn't do inf-inf.
    m_safe = at.where(at.isneginf(m), 0., m)

    # Compute exp in the shifted space
    z = at.exp(a - m_safe) + at.exp(b - m_safe) + at.exp(c - m_safe)

    # Optional tiny floor to avoid log(0) if everything underflows (rare but possible in float32)
    #tiny = _const_like(z, 1e-30) if str(getattr(z, "dtype", "")) == "float32" else _const_like(z, 1e-300)
    z = at.maximum(z, tiny)

    out = m_safe + at.log(z)

    # Restore exact -inf where appropriate
    return at.where(at.isneginf(m), neg_inf, out)


    
def logitat(p, eps=1e-12):
    #return at.log(p) - at.log(1. - p)
    # Always stay strictly away from 0 and 1
    p_safe = at.clip(p, eps, 1.0 - eps)
    # Use log1p for better stability near the boundaries
    return at.log(p_safe) - at.log1p(-p_safe)


def inv_logitat(p):
    return 1. / (1 + at.exp(-p))

def inv_flogitat(p):
    return (at.exp(p) - 1. ) / (1. + at.exp(p))

 
def flogitat(p, eps=1e-12):
    p_safe = at.clip(p, -1.0 + eps, 1.0 - eps)
    return at.log(1.0 + p_safe) - at.log(1.0 - p_safe)
    #return at.log(1 + p) - at.log(1 - p)


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
    - 0.5 * mu.shape[0] * at.log(2 * PI)
    - at.sum(at.log(at.diagonal(L)))  # log determinant of L
    )
    return sample, logp


def stick_breaking(beta):
    portion_remaining = at.concatenate([[1], at.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining


def frechet_logp_full(value, lambda_ell, d):
    """
    Fréchet-like kernel:
      log f(x) = log(alpha*lambda) - (alpha+1) log x - lambda * x^{-alpha},  x>0
    with alpha = d/2 > 0, lambda>0.
    """
    #x   = at.as_tensor_variable(value)
    #lam = at.as_tensor_variable(lambda_ell)
    #d_  = at.as_tensor_variable(d)
    x = value
    lam = lambda_ell
    d_ = d
    
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

    
#######################
# sigmoids
########################

#######################
# working 


def sigmoid(x, x0, s, eps=1e-12, clip=1e-15):
    """
    Stable sigmoid using tanh, with scale regularization.
    Designed to be friendly to JAX (no dtype churn).
    """
    #eps_c = _const_like(x, eps)
    s_pos = at.maximum(s, eps )

    t = (x - x0) / s_pos
    y = 0.5 * ( at.tanh(0.5 * t) + 1.0) 

    if clip is not None:
        y = at.clip(y, clip, 1.0 - clip)

    return y



def log_sigmoid(x, m, sig):
    return at.log(sigmoid(x, m, sig)) 


def safe_sigmoid(x, x0, eps):
    
    return sigmoid(x, x0, eps, clip=1e-15)


def softplus(x):
    # log(1 + exp(x)) with good numerical stability
    return at.maximum(x, 0) + at.log1p(at.exp(-at.abs(x)))


def safe_log(x, ):
    # Work dtype detection
    eps = at.constant(1e-12, dtype="float32") if x.dtype == "float32" else at.constant(1e-30, dtype="float64")  
    return at.log( at.clip(x, eps, np.inf, ))


##########################
####### Interpolators and integrators ########
##########################

def meshgrid_at(x, y):
    #x = at.as_tensor_variable(x)
    #y = at.as_tensor_variable(y)
    nx = x.shape[0]
    ny = y.shape[0]

    X = at.alloc(x, nx, ny)      # Broadcast x along columns
    Y = at.alloc(y, nx, ny).T    # Broadcast y along rows, then transpose

    return X.T, Y.T



def atinterp(x, xs, ys, eps=1e-12, side="right"):
    # xs, ys tensors; x can be scalar or tensor
    idxs = at.searchsorted(xs, x, side=side)
    idxs = at.clip(idxs, 1, xs.shape[0] - 1)
    # <-- stop grad through the discrete selection -->
    idxs = stop_grad(idxs)

    xl = xs[idxs - 1]; xh = xs[idxs]
    yl = ys[idxs - 1]; yh = ys[idxs]

    #eps_t = at.as_tensor_variable(eps, dtype=xl.dtype)
    denom = at.maximum(xh - xl, eps )
    r = (x - xl) / denom
    return (1 - r) * yl + r * yh

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




def uniform_interp_indices(x, x0, x1, n_pts, eps=1e-30):
    """
    Compute interpolation bin indices j and fractional positions r
    for a uniform grid spanning [x0, x1] with n_pts points.

    Parameters
    ----------
    x : tensor
        Query points (unsorted ok, multidimensional ok).
    x0, x1 : scalar tensors
        Grid boundaries (must match your uniform grid).
    n_pts : int
        Number of grid points (not bins).
        Grid values are implicitly x0 + k * dx  for k=0..n_pts-1.
    eps : float
        Small stabilizer to avoid division by zero.

    Returns
    -------
    j : int tensor
        Clipped, integer indices into the grid, shape like x.
    r : float tensor
        Fractional interpolation weight in [0,1], shape like x.
    """
    # x  = at.as_tensor_variable(x)
    # x0 = at.as_tensor_variable(x0, dtype=x.dtype)
    # x1 = at.as_tensor_variable(x1, dtype=x.dtype)

    #dtype = getattr(x, "dtype", "float64")
    #int_dtype = "int64" if dtype=="float64" else "int32"
    
    # number of intervals
    n_minus_1 = n_pts - 1
    n_minus_1_f = n_minus_1 #at.cast(n_minus_1, dtype)

    dx = (x1 - x0) / at.maximum(n_minus_1_f, eps)

    # fractional location
    t = (x - x0) / at.maximum(dx, eps)

    # bin index
    j = at.floor(t)
    j = at.clip(j, 0, n_pts - 2) #.astype(int_dtype)

    # stop gradient through the index selection
    j = stop_grad(j)

    # fractional offset
    r = t - j #at.cast(j, dtype)
    #r = at.cast(r, dtype)

    return j, r



def uniform_interp_indices_jax(x, x0, x1, n_pts, eps=1e-30):
    dtype = getattr(x, "dtype", "float64")
    int_dtype = "int64" if dtype == "float64" else "int32"

    n_minus_1_f = n_pts - 1 #at.cast(n_pts - 1, dtype)
    #eps_t = at.as_tensor_variable(eps)  # no dtype specified → keeps original promotion tendencies

    dx = (x1 - x0) / at.maximum(n_minus_1_f, eps)
    t  = (x - x0) / at.maximum(dx, eps)

    j = stop_grad(at.clip(at.floor(t), 0, n_pts - 2)) #.astype(int_dtype))
    r = t - j #at.cast(t - at.cast(j, dtype), dtype)
    return j, r

    
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
        out = np.trapezoid(y, x_b, axis=self.axis)  # (M,)
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
                    return jnp.sum(jnp.trapezoid(y_, x_, axis=1))

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





# Precompute n-point Gauss–Legendre nodes/weights on [0,1]
def gauss_legendre_01(n=32, ): #dtype="float64"):
    from numpy.polynomial.legendre import leggauss
    #dtype=pytensor.config.floatX
    x, w = leggauss(n)                 # on [-1, 1]
    x01 = (x + 1.0) * 0.5              # map to [0, 1]
    w01 = w * 0.5
    return x01, w01 #.astype(dtype)

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
        z_nodes = z[..., None] * _x01_np  # shape (..., n)
        integrand = 1.0 / Efun_at(z_nodes, Om, w0)  # shape (..., n)
        I = at.sum(_w01_np * integrand, axis=-1)     # shape (...)
        return (c_light / H0) * z * I * 1e-03

def Xifun_at(z, Xi0, n):
    return Xi0+(1-Xi0)/(1+z)**n


def Xifun_at_polexp(z, Xi0, n):
    r"""
    Ξ(z) = exp(  -(1-Ξ0)[1 - (1+z)^n] / (1+z)^{2n}  )
           * [ Ξ0 + (1-Ξ0)(1+z)^{-n} ]
    """
    exponent = -(1 - Xi0) * (1 - (1 + z)**n) / (1 + z)**(2 * n)
    prefactor = Xi0 + (1 - Xi0) * (1 + z)**(-n)
    return at.exp(exponent) * prefactor


def dLfun_at(z, H0, Om, w0, Xi0, n, interp=False, dc=None, param='vanilla'):
    """Luminosity distance at redshift ``z``."""
    
    if param=='vanilla':
        Xi = Xifun_at(z, Xi0, n)
        #print("In dLfun_at, using vanilla")
    elif param=='polexp':
        Xi = Xifun_at_polexp(z, Xi0, n)
        #print("In dLfun_at, using polexp")
    
    if dc is not None:
        return Xi*(z+1.0)*dc
    else:
        return Xi*(z+1.0)*dcfun_at(z, H0, Om, w0, interp=interp)


def Efun_at(z, Om, w0):
    # E(z) = sqrt( Om (1+z)^3 + (1-Om) (1+z)^{3(1+w0)} )
    a = (1. + z)
    return at.sqrt(Om * a**3 + (1. - Om) * a**(3. * (1. + w0)))


def Efun_num(z, Om, w0):
    # E(z) = sqrt( Om (1+z)^3 + (1-Om) (1+z)^{3(1+w0)} )
    a = 1.0 + z
    return np.sqrt(Om * a**3 + (1.0 - Om) * a**(3.0 * (1.0 + w0)))



def z_from_dL_at( r, H0, Om, w0, Xi0, n , interp=False, param='vanilla'):
    dLGrid_at = dLfun_at( zGridGlobals_at, H0, Om, w0, Xi0, n , interp=interp, param=param)
    z2dL = atinterp( r, dLGrid_at, zGridGlobals_at ) 
    return z2dL 


    
def log_j_at(z, Om, H0=70, dc=None,  interp=False):
    if dc is None:
        dc = dcfun_at(z, H0, Om, interp=interp)
    dc*=H0/c_light*1e03
    return at.log(4*PI)+2*at.log(dc)-at.log(Efun_at(z, Om=Om))


def log_dV_dz_at(z, H0, Om0, w0, dc=None, interp=False):
    
    # c_light_ = at.as_tensor_variable(c_light, dtype=work_dtype)
    # four_pi_ = at.as_tensor_variable(4*PI, dtype=work_dtype)
    # ten_ = at.as_tensor_variable(10., dtype=work_dtype)
    # three_ = at.as_tensor_variable(3., dtype=work_dtype)
    
    if dc is None:
        dc = dcfun_at(z, H0, Om0, w0, interp=interp)    
    
    res =  at.log(4*PI)+at.log(c_light)-at.log(H0)+2*at.log(dc)-at.log(Efun_at(z, Om0, w0))-3.0*at.log(10.0)
    
    return res


def log_ddL_dz(z, H0, Om0,  w0, Xi0, n, dc=None, interp=False, param='vanilla'):
    
    # one_ = at.as_tensor_variable(1., dtype=work_dtype)
    # ten_to_the_three = at.as_tensor_variable(1e03, dtype=work_dtype)
    # two_ = one_+one_

    
    # H0 in Mpc, dLs in Gpc
    if dc is None:
        dc = dcfun_at(z, H0, Om0,  w0, interp=interp) # Gpc

    if param=='vanilla':
        #print("In log_ddL_dz, using vanilla")

        Xi = Xifun_at(z, Xi0, n)
        res = at.log( ( Xi - n*(1.-Xi0)/(1.+z)**n ) * dc + Xi * c_light * (1.+z)/(1e03*H0*Efun_at(z,Om0,  w0)) )  
        
    elif param=='polexp':
        #print("In log_ddL_dz, using polexp")

        # ---- Xi(z) in the polexp parametrization ----
        onepz = 1. + z

        exponent = -(1. - Xi0) * (1. - onepz**n) / (onepz**( two_ * n))
        prefactor = Xi0 + (1. - Xi0) * onepz**(-n)

        Xi = at.exp(exponent) * prefactor

        # ---- dXi/dz for polexp ----
        # exponent = -(1 - Xi0) * C * D, with
        # C = 1 - (1+z)^n, D = (1+z)^(-2n)
        C = 1 - onepz**n
        D = onepz**(-2. * n)
        dC = -n * onepz**(n - 1.)
        dD = -2. * n * onepz**(-2. * n - 1.)

        d_exponent = -(1. - Xi0) * (dC * D + C * dD)

        # prefactor = Xi0 + (1 - Xi0) * (1+z)^(-n)
        d_prefactor = -(1. - Xi0) * n * onepz**(-n - 1.)

        # dXi/dz = exp(exponent) * (d_exponent * prefactor + d_prefactor)
        dXi = at.exp(exponent) * (d_exponent * prefactor + d_prefactor)

        # ---- d d_c / dz ----
        ddc_dz = c_light / (1e03 * H0 * Efun_at(z, Om0, w0))

        # ---- d d_L / dz = d/dz [Xi (1+z) d_c] ----
        dL_dz = (dXi * onepz + Xi) * dc + Xi * onepz * ddc_dz

        # log d(d_L/dz)
        res = at.log(dL_dz)
    
      
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
    # work_dtype = getattr(z, "dtype", "float64")
    # print("work_dtype in log_p_z_MD_unnorm is %s"%work_dtype)
    
    log_psiz = log_psi_z_MD(z, gamma, kappa, zp) #gamma*at.log1p(z)-at.log(1+((1+z)/(1+zp))**(gamma+kappa))
    
    log_dVdz = log_dV_dz_at(z, H0, Om, w0, dc=dc )

    return log_psiz+log_dVdz


def N_per_year( gamma, kappa, zp, H0, Om, w0, R0=1., dc=None, z_max = 100, res=1000):

    zgrid = at.linspace(0, z_max, steps=res) 
    pz = R0*at.exp( log_p_z_MD_unnorm(zgrid, gamma, kappa, zp, H0, Om, w0, dc=dc))
    norm = attrapzvec(pz, zgrid)
    return norm

def log_psi_z_MD(z, gamma, kappa, zp):
    
    # work_dtype = getattr(z, "dtype", "float64")
    # print("work_dtype in log_psi_z_MD is %s"%work_dtype)
    # one_ = at.as_tensor_variable(1., dtype=work_dtype)
    
    lC0 = at.log( 1.+(1.+zp)**(-gamma-kappa))
    log_psiz = lC0+gamma*at.log1p(z)-at.log(1.+((1.+z)/(1.+zp))**(gamma+kappa))
    
    return (log_psiz-at.log1p(z))


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

    logpdf = -0.5 * (2.0 * at.log(2.0 * PI) + logdetC + quad)  # (n,)

    return at.where(where_inf, MIN , logpdf)



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

    return at.where( (m1>=ml) & (m1<=mh) & (m2>=ml) & (m2<=mh) & (m2<=m1), -2*at.log( mh-ml ) , neg_inf = _const_like(m1, -np.inf)  )


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
                    -at.log(scale)-0.5*at.log(2*PI)-at.log(Phibeta-Phialpha) + 0.5*(-(x-loc)**2/(scale**2)) , -np.inf )


def truncGausslowerupper_at_lpdf_nonly(x, loc, scale, xmin=0, xmax=1):    

    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    Phibeta = 0.5*(1.+at.erf((xmax-loc)/(at.sqrt(2.)*scale)))
    
    return -at.log(scale)-0.5*at.log(2*PI)-at.log(Phibeta-Phialpha) + 0.5*(-(x-loc)**2/(scale**2)) 

def truncGausslower_at_lpdf(x, loc, scale, xmin=0):    

    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    #Phibeta = 0.5*(1.+at.erf((xmax-loc)/(at.sqrt(2.)*scale)))
    
    return at.where( x>=xmin, 
                    -at.log(scale)-0.5*at.log(2*PI)-at.log(1.-Phialpha) + 0.5*(-(x-loc)**2/(scale**2)) ,  -np.inf )


def double_gauss_norm(mu, sigma):
    z = -mu / sigma
    C = 0.5 * (1 + at.erf(z / at.sqrt(2)))
    return 0.5 - C + 0.5 * C**2


def truncGausslower_at_logpdf(x, loc, scale, xmin=0):  
    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    return at.where(x>xmin, at.log(1./(at.sqrt(2.*PI)*scale)/(1.-Phialpha)) + -(x-loc)**2/(2*scale**2) , -np.inf )
    #return -at.log(scale)-0.5*at.log(2.*PI) -0.5*(x-loc)**2/(scale**2)

def truncGausslower_at_pdf(x, loc, scale, xmin=0):  
    
    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    return at.where(x>xmin, at.exp( -(x-loc)**2/(2*scale**2))/(at.sqrt(2.*PI)*scale)/(1.-Phialpha) , 0. )
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


def gaussian_logpdf_pair(m1s, m2s, mu, sd, z=None, eps=1e-06):
    """
    Compute per-component 1D Gaussian log-pdfs for (m1, m2) given
    means mu and std-devs sd.
    """
    #work_dtype = getattr(m1s, "dtype", "float64")

    m1 = m1s[None, :]          # (1, N)
    m2 = m2s[None, :]          # (1, N)

    mu1 = mu[0][:, None]       # (K, 1)
    mu2 = mu[1][:, None]       # (K, 1)

    sd1 = sd[0][:, None]       # (K, 1)
    sd2 = sd[1][:, None]       # (K, 1)

    # Typed constants (avoid np.inf upcasting and repeated as_tensor_variable nodes)
    #eps = at.constant(1e-6, dtype=work_dtype)
    eps2 = eps * eps
    inf = np.inf

    var1 = at.clip(sd1 * sd1, eps2, inf)  # (K,1)
    var2 = at.clip(sd2 * sd2, eps2, inf)  # (K,1)

    diff1 = m1 - mu1                      # (K,N)
    diff2 = m2 - mu2                      # (K,N)

    # 1D Gaussian logpdfs (keep same constant expression, but ensure constants don't upcast)
    const = -0.5 * at.log(2.0 * PI)

    logp1 = const - 0.5 * at.log(var1) - 0.5 * (diff1 * diff1 / var1)
    logp2 = const - 0.5 * at.log(var2) - 0.5 * (diff2 * diff2 / var2)

    if z is not None:
        z = z[None, :]
        muz = mu[2][:, None]       # (K, 1)
        sdz = sd[2][:, None]

        varz = at.clip(sdz * sdz, eps2, inf)  # (K,1)
        diffz = z - muz
        logpz = const - 0.5 * at.log(varz) - 0.5 * (diffz * diffz / varz)
    else:
        logpz = at.zeros_like(logp1, dtype=work_dtype)

    return logp1, logp2, logpz





def gaussian_logpdf_pair_from_interp(theta, interp_vals, interp_grids, z=None):
    """
    Interpolate Gaussian-mixture logpdfs on non-uniform grids for:
      - m1  (e.g. log Mc)
      - m2  (e.g. logit q)
      - optionally log(1+z)

    Parameters
    ----------
    theta : tuple (m1, m2)
        Each of shape (N,).
    interp_vals : list
        [lp_m1_grid, lp_m2_grid] or [lp_m1_grid, lp_m2_grid, lp_z_grid],
        each of shape (K, n_grid_dim) with precomputed logpdf values.
    interp_grids : list
        [m1_grid, m2_grid] or [m1_grid, m2_grid, z_grid],
        each 1D non-uniform grid (n_grid_dim,).
    z : TensorVariable or None
        If provided, same length as m1/m2 and in the same transformed
        coord as z_grid (e.g. log(1+z)).

    Returns
    -------
    lpdfm1, lpdfm2, lpdf3 : TensorVariables
        Each of shape (K, N), where N = number of evaluation points.
    """
    m1, m2 = theta

    #work_dtype = getattr(m1, "dtype", "float64")

    # unpack grids and banked logpdfs
    m1_grid = interp_grids[0]               # (n_Mc,)
    m2_grid = interp_grids[1]               # (n_q,)

    lp_m1_grid = interp_vals[0]             # (K, n_Mc)
    lp_m2_grid = interp_vals[1]             # (K, n_q)

    # ==============================
    # 1) M1 / logMc (non-uniform)
    # ==============================
    j1, r1 = _interp_indices_nonuniform(m1, m1_grid)  # (N,)

    # lower/upper neighbor values for all components at once: (K, N)
    yl1 = lp_m1_grid[:, j1 - 1]
    yh1 = lp_m1_grid[:, j1]

    r1b = r1[None, :]                        # (1, N) for broadcasting
    lpdfm1 = (1.0 - r1b) * yl1 + r1b * yh1   # (K, N)

    # ==============================
    # 2) M2 / logit(q) (non-uniform)
    # ==============================
    j2, r2 = _interp_indices_nonuniform(m2, m2_grid)  # (N,)

    yl2 = lp_m2_grid[:, j2 - 1]             # (K, N)
    yh2 = lp_m2_grid[:, j2]                 # (K, N)

    r2b = r2[None, :]
    lpdfm2 = (1.0 - r2b) * yl2 + r2b * yh2   # (K, N)

    # ==============================
    # 3) Optional z / log(1+z) (non-uniform)
    # ==============================
    if z is None:
        # neutral factor: same shape as lpdfm2, all zeros
        lpdf3 = at.zeros_like(lpdfm2)
    else:
        z_grid = interp_grids[2]            # (n_z,)
        lp_z_grid = interp_vals[2]          # (K, n_z)

        j3, r3 = _interp_indices_nonuniform(z, z_grid)  # (N,)

        yl3 = lp_z_grid[:, j3 - 1]          # (K, N)
        yh3 = lp_z_grid[:, j3]              # (K, N)
        r3b = r3[None, :]

        lpdf3 = (1.0 - r3b) * yl3 + r3b * yh3  # (K, N)

    return lpdfm1, lpdfm2, lpdf3


def gaussian_logpdf_pair_from_interp_pymc(theta, interp_vals, interp_grids, z=None):
    """
    Interpolate Gaussian-mixture logpdfs on *non-uniform* grids for:
      - m1  (e.g. log Mc)
      - m2  (e.g. logit q)
      - optionally log(1+z)

    Parameters
    ----------
    theta : tuple (m1, m2)
        m1, m2 are 1D TensorVariables (N,) in the transformed coordinates.
    interp_vals : list
        [lp_m1_grid, lp_m2_grid] or [lp_m1_grid, lp_m2_grid, lp_z_grid]
        where each lp_*_grid has shape (K, n_grid_dim),
        K = number of mixture components.
    interp_grids : list
        [m1_grid, m2_grid] or [m1_grid, m2_grid, z_grid],
        each a 1D non-uniform grid (n_grid_dim,).
    z : TensorVariable or None
        If provided, same length as m1/m2 and in the same transformed coord
        as z_grid (e.g. log(1+z)).

    Returns
    -------
    lpdfm1, lpdfm2, lpdf3 : TensorVariables
        Each of shape (K, N), where N = number of evaluation points.
    """
    m1, m2 = theta
    #work_dtype = getattr(m1, "dtype", "float64")

    # unpack grids and banked logpdfs
    m1_grid = interp_grids[0]               # (n_Mc,)
    m2_grid = interp_grids[1]               # (n_q,)

    lp_m1_grid = interp_vals[0]             # (K, n_Mc)
    lp_m2_grid = interp_vals[1]             # (K, n_q)

    if K is None:
        K = lp_m1_grid.shape[0]                 # number of mixture components

    # convenience: row indices for broadcasting (K,1)
    row_idx = at.arange(K).reshape((K, 1))

    # ==============================
    # 1) M1 / logMc (non-uniform)
    # ==============================
    j1, r1 = _interp_indices_nonuniform(m1, m1_grid)  # j1,r1: (N,)

    # columns for lower / upper grid points, shape (1, N)
    col1_L = (j1 - 1).reshape((1, j1.shape[0]))
    col1_R = j1.reshape((1, j1.shape[0]))

    # pick logpdf values at neighbours: shapes (K, N)
    yl1 = lp_m1_grid[row_idx, col1_L]
    yh1 = lp_m1_grid[row_idx, col1_R]

    r1b = r1.reshape((1, r1.shape[0]))      # (1, N) for broadcasting
    lpdfm1 = (1.0 - r1b) * yl1 + r1b * yh1  # (K, N)

    # ==============================
    # 2) M2 / logit(q) (non-uniform)
    # ==============================
    j2, r2 = _interp_indices_nonuniform(m2, m2_grid)  # (N,)

    col2_L = (j2 - 1).reshape((1, j2.shape[0]))
    col2_R = j2.reshape((1, j2.shape[0]))

    yl2 = lp_m2_grid[row_idx, col2_L]       # (K, N)
    yh2 = lp_m2_grid[row_idx, col2_R]       # (K, N)

    r2b = r2.reshape((1, r2.shape[0]))
    lpdfm2 = (1.0 - r2b) * yl2 + r2b * yh2  # (K, N)

    # ==============================
    # 3) Optional z / log(1+z)
    # ==============================
    if z is None:
        # same shape as lpdfm2, but zero (i.e. neutral factor)
        lpdf3 = at.zeros_like(lpdfm2)
    else:
        z_grid = interp_grids[2]            # (n_z,)
        lp_z_grid = interp_vals[2]          # (K, n_z)

        j3, r3 = _interp_indices_nonuniform(z, z_grid)  # (N,)

        col3_L = (j3 - 1).reshape((1, j3.shape[0]))
        col3_R = j3.reshape((1, j3.shape[0]))

        yl3 = lp_z_grid[row_idx, col3_L]    # (K, N)
        yh3 = lp_z_grid[row_idx, col3_R]    # (K, N)

        r3b = r3.reshape((1, r3.shape[0]))
        lpdf3 = (1.0 - r3b) * yl3 + r3b * yh3  # (K, N)

    return lpdfm1, lpdfm2, lpdf3


    
def gaussian_logpdf_pair_from_interp_lin(theta, interp_vals, interp_grids, z=None):

    m1, m2 = theta
    
    m1_grid = interp_grids[0]         # (n_Mc,)
    m2_grid = interp_grids[1]         # (n_q,)
    lp_m1_grid, lp_m2_grid = interp_vals[0], interp_vals[1]  # (K, n_Mc), (K, n_q)

    # ----- M1 / logMc -----
    x0_1 = m1_grid[0]
    x1_1 = m1_grid[-1]
    nU_1 = m1_grid.shape[0]
    
    j1, r1 = uniform_interp_indices(m1, x0_1, x1_1, nU_1)  # j1,r1: (N,)

    yl1 = lp_m1_grid[:, j1]        # (K, N)
    yh1 = lp_m1_grid[:, j1 + 1]    # (K, N)
    r1b = r1[None, :]              # (1, N)

    lpdfm1 = (1.0 - r1b) * yl1 + r1b * yh1   # (K, N)

    # ----- M2 / logitq -----
    x0_2 = m2_grid[0]
    x1_2 = m2_grid[-1]
    nU_2 = m2_grid.shape[0]
    
    j2, r2 = uniform_interp_indices(m2, x0_2, x1_2, nU_2)  # (N,)

    yl2 = lp_m2_grid[:, j2]        # (K, N)
    yh2 = lp_m2_grid[:, j2 + 1]    # (K, N)
    r2b = r2[None, :]

    lpdfm2 = (1.0 - r2b) * yl2 + r2b * yh2   # (K, N)

    if z is None:
        lpdf3 = at.zeros(lpdfm2.shape)

    else:
        # ----- log(1+z) -----
        
        z_grid = interp_grids[2]  
        lp_z_grid = interp_vals[2]
        
        x0_3 = z_grid[0]
        x1_3 = z_grid[-1]
        nU_3 = z_grid.shape[0]
        
        j3, r3 = uniform_interp_indices(z, x0_3, x1_3, nU_3)  # (N,)
    
        yl3 = lp_z_grid[:, j3]        # (K, N)
        yh3 = lp_z_grid[:, j3 + 1]    # (K, N)
        r3b = r3[None, :]
    
        lpdf3 = (1.0 - r3b) * yl3 + r3b * yh3   # (K, N)
        
    return lpdfm1, lpdfm2, lpdf3


def build_1d_gaussian_mixture_grid_components(
    mu_d, sigma_d,
    x_low, x_high,
    n_total_min=2000,    # interpreted as *total* desired number of points
    frac_uniform = 0.2,
    k_sigma=4.0,         # μ ± kσ window for each component
    sigma_floor=1e-4,    # floor on σ for grid *only* (not for pdf)
    K=30,
    eps=1e-05
):
    """
    Build a 1D non-uniform grid for a Gaussian mixture in x with:

      - A fixed total number of points N_total = n_total_min.
      - A fixed fraction f_uniform of points in a global uniform grid
        over [x_low, x_high] to cover the extrema.
      - The remaining points distributed equally across the K Gaussians,
        each in the window [mu_k - k_sigma*sigma_k, mu_k + k_sigma*sigma_k].
      - The mean mu_k of each component is *always* included explicitly.

    Notes
    -----
    - All geometry is built with stop_grad(), so the grid does not
      create gradient paths.
    - The final grid is sorted with at.sort (no unique). Duplicates
      are harmless for your non-uniform interpolation (which guards
      denominators by eps).
    """

    # ---- number of components (compile-time constant) ----
    if K <= 0:
        raise ValueError("build_1d_gaussian_mixture_grid_components: K = len(mu_d) must be > 0.")

    # ---- total points and uniform fraction ----
    N_total = int(n_total_min)
    if N_total < K:
        # Need at least one point per component
        N_total = K

    N_uniform_raw = int(round(frac_uniform * N_total))
    # ensure we leave at least one point per component
    N_uniform = max(2, min(N_uniform_raw, N_total - K))
    if N_uniform < 0:
        N_uniform = 0

    N_comp_total = max(N_total - N_uniform, K)  # total points to allocate to components

    # base per-component count (at least 1)
    base_per_comp = max(1, N_comp_total // K)
    remainder = N_comp_total % K  # some components get one extra

    # ---- detach parameters for geometry ----
    mu_s      = stop_grad(mu_d)
    sigma_s   = stop_grad(sigma_d)
    x_low_s   = stop_grad(x_low)
    x_high_s  = stop_grad(x_high)

    #dtype = getattr(x_low_s, "dtype", "float64")

    #eps         = at.as_tensor_variable(1e-5, dtype=dtype)
    #sigma_floor = at.as_tensor_variable(sigma_floor, dtype=dtype)
    k_sigma_t   = k_sigma #at.as_tensor_variable(k_sigma, dtype=dtype)

    xmin = x_low_s  + eps
    xmax = x_high_s - eps
    span = at.maximum(xmax - xmin, 1e-6 )

    sigma_eff = at.maximum(at.abs(sigma_s), sigma_floor)

    win_min_raw = mu_s - k_sigma_t * sigma_eff
    win_max_raw = mu_s + k_sigma_t * sigma_eff

    win_min = at.clip(win_min_raw, xmin, xmax)
    win_max = at.clip(win_max_raw, xmin, xmax)

    tiny = 1e-6 * span
    win_width = at.maximum(win_max - win_min, tiny)

    # ---- per-component bands + explicit means ----
    comp_grids = []

    for k in range(K):
        # how many points for this component
        n_k = base_per_comp + (1 if k < remainder else 0)

        mu_k      = mu_s[k]
        win_min_k = win_min[k]
        win_width_k = win_width[k]

        if n_k <= 1:
            # Only the mean
            x_comp_k = mu_k.reshape((1,))
        else:
            # reserve one point for the mean, n_win on the window
            n_win = n_k - 1
            if n_win > 1:
                denom_k = float(n_win - 1)
                t_k = at.arange(n_win) / denom_k  # [0,1]
            else:
                t_k = at.zeros((1,))

            x_win_k = win_min_k + win_width_k * t_k          # n_win points
            x_mean_k = mu_k.reshape((1,))                    # ensure mean included
            x_comp_k = at.concatenate([x_win_k, x_mean_k], axis=0)

        comp_grids.append(x_comp_k)

    x_comp_all = at.concatenate(comp_grids, axis=0)  # (N_comp_total,)

    # ---- global uniform background over [xmin, xmax] ----
    if N_uniform > 0:
        if N_uniform > 1:
            denom_u = float(N_uniform - 1)
            t_u = at.arange(N_uniform) / denom_u
        else:
            t_u = at.zeros((1,))
        x_uniform = xmin + (xmax - xmin) * t_u
    else:
        x_uniform = at.zeros((0,))

    # ---- combine, clip, sort ----
    x_all = at.concatenate([x_uniform, x_comp_all], axis=0)
    x_all = at.clip(x_all, xmin, xmax)

    # sort after stop_grad to remove geometry from gradient path
    x_all_sg = stop_grad(x_all)
    x_grid = at.sort(x_all_sg)

    return x_grid





def redshift_mixture_log_norm(mu, sd, logw, 
                              y_min, y_max,  H0, Om, w0, Ny=512, eps=1e-6):
    """
    log normalization for:
        N = ∫ dy [ Σ_k w_k p_k(y) * dV/dz(z(y)) ]
    where y = log(1+z), z(y)=exp(y)-1.
    """

    # grid in y = log(1+z)
    y_grid = at.linspace(y_min, y_max, Ny)  # (Ny,)
    yg = y_grid[None, :]                                        # (1, Ny)

    # component params for y
    muy = mu[2][:, None]           # (K, 1)
    sdy = sd[2][:, None]           # (K, 1)

    #eps = at.as_tensor_variable(1e-6, dtype=sdy.dtype)
    vary = at.clip(sdy**2, eps**2, np.inf)

    const = -0.5 * at.log(2.0 * PI)
    logpy = const - 0.5 * at.log(vary) - 0.5 * ((yg - muy) ** 2 / vary)  # (K, Ny)
    py = at.exp(logpy)                                                     # (K, Ny)

    # weights
    w = at.exp(logw)
    w = w / at.sum(w)

    mix_y = at.sum(w[:, None] * py, axis=0)  # (Ny,)

    # selection factor dV/dz evaluated at z = exp(y)-1
    z_grid = at.exp(y_grid) - 1.0            # (Ny,)
    dVdz = at.exp(log_dV_dz_at(z_grid, H0, Om, w0))

    integrand = mix_y * dVdz #/(1.+z_grid)                # (Ny,)

    # trapezoid over y
    #dy = (y_max - y_min) / at.as_tensor_variable(Ny - 1, dtype=y_grid.dtype)
    #N_val = at.sum(0.5 * (integrand[1:] + integrand[:-1])) * dy

    N_val = attrapzvec(integrand, y_grid,  )

    return at.log(N_val)





####### Power Law + Peak ########


def truncated_power_law(m, alpha, ml, mh):
        
        where_compute = (ml < m) & (m < mh )

        result = at.where(where_compute, at.log(m)*(-alpha), -np.inf )
        
        return result



def logpdf_PLP(theta, lambdaBBHmass, pairing=True):
    
        m1, m2 = theta
        lambdaPeak, alpha, beta, deltam, ml, mh, muMass, sigmaMass = lambdaBBHmass
                
        where_compute = (m2 <= m1) & (ml <= m2) & (m1 <= mh ) 

        lpdfm1 = at.where(where_compute, logpdfm1_PLP(m1,  lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass ), -np.inf )
        lpdfm2 = at.where(where_compute,logpdfm2_PLP(m2, beta, deltam, ml), -np.inf)
        if pairing:
            lC = at.where(where_compute, logC_PLP(m1, beta, deltam,  ml, ), -np.inf )
        ln = at.where(where_compute, logNorm_PLP( lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass), -np.inf )
        
        return at.where( where_compute, lpdfm1+lpdfm2+lC-ln, -np.inf )
        

def logS_PLP(m, deltam, ml, eps=1e-12):
    """
    Smoothly goes from -inf (log 0) below ml to 0 (log 1) above ml+deltam,
    with a C^1 transition (smoothstep) in between. Numerically robust.
    """

    #eps = at.as_tensor_variable(eps, dtype=m.dtype)
    #one_ = at.as_tensor_variable(1.0, dtype=m.dtype)
    #two_ = at.as_tensor_variable(2.0, dtype=m.dtype)
    #three_ = at.as_tensor_variable(3.0, dtype=m.dtype)
    
    # normalize position in the window and clamp to [0, 1]
    t = (m - ml) / at.maximum(deltam, eps)
    t = at.clip(t, eps, 1.-eps )

    # smoothstep: S(t) = 3t^2 - 2t^3, monotone from 0→1 with zero slope at ends
    S = t * t * (3. - 2. * t)

    # log S, safely (avoid log(0) at the lower edge)
    return at.log(at.clip(S, eps, 1. ))
    
def logS_PLP_LVK(m, deltam, ml,):
        
        maskL = m <= ml 
        maskU = m >= (ml + deltam) 
        
        maskM = ~(maskL | maskU)
        
        s = at.where( maskL,  -np.inf, 0.  )
        
        s1 = at.where( maskM,  at.log(1/(1+ at.exp(deltam/(m-ml) + deltam/(m-ml - deltam) ) ))  , s  )
        
        return s1   



def logpdfm1_PLP(m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass):

    where_compute = (ml <= m) & (m <= mh )

    norm = norm_truncated_pl_num(alpha, ml, mh)
    trunc_component = at.where(where_compute, 1./m**alpha/norm, -np.inf )
    gauss_component = at.where(where_compute, at.exp(-(m-muMass)**2/(2*sigmaMass**2))/(at.sqrt(2*PI)*sigmaMass), -np.inf )

    lS = logS_PLP(m, deltam, ml) 
        
    result =  at.where( where_compute, at.log( (1-lambdaPeak)*trunc_component+lambdaPeak*gauss_component)+lS
                       , -np.inf )
    return result

    

def logpdfm2_PLP(m2, beta, deltam, ml):

    where_compute = (ml<= m2) #& (~where_nan)
    res = at.log(m2)*(beta)+logS_PLP(m2, deltam, ml)
    result = at.where( where_compute, res, -np.inf )
           
    return result

        

def logC_PLP( m, beta, deltam, ml, res=100):
    '''
    Gives inverse log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
    '''
    
    if res!=100:
        _tgrid = at.linspace(0, 1, res)#
    else:
        _tgrid = _get_t_grid_100()
    
    xx = ml + (max_m - ml) * _tgrid

    # Evaluate log-pdf on the fixed grid, then zero-out below ml
    logp2 = logpdfm2_PLP(xx, beta, deltam, ml)          # (NM,)
    p2    = at.exp(logp2)                                # (NM,)

    # CDF via trapezoid from the fixed grid (below-ml bins contribute 0)
    cdf = atcumtrapz(p2, xx)                             # (NM-1,)

    # Interpolate log C at m
    return atinterp(m, xx[1:], at.log(cdf))



def logNorm_PLP(lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, res=500):
    """
    Log integral of p(m1, m2) dm1 dm2 (total normalization of the mass function).
    Uses a cached global grid; ml and mh can be stochastic.
    """

    if res!=500:
        _tgrid = at.linspace(0, 1, res)
    else:
        _tgrid = _get_t_grid()
    
    xx = ml + (mh - ml) * _tgrid

    # Evaluate log-pdf on fixed grid
    logp = logpdfm1_PLP(xx, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass)  # (NM,)
    p    = at.exp(logp)

    Z = attrapzvec(p, xx)                              # (scalar)
    return at.log(at.clip(Z, 1e-300 , -np.inf ))
            
    
            
def norm_truncated_pl_num(alpha, mmin, mmax):

    return 1/(1-alpha)*(mmax**(1-alpha)-mmin**(1-alpha))





def log_norm_truncated_pl_num(alpha, mmin, mmax, eps=1e-12):
    """
    log ∫_{mmin}^{mmax} m^{-alpha} dm
    = log( (mmax^(1-α) - mmin^(1-α)) / (1-α) )

    Uses the stable rewrite:
      t = 1 - alpha
      a = log(mmax), b = log(mmin), Δ = a - b > 0
      log integral = t*b + log(|expm1(t*Δ)|) - log(|t|)

    NOTE: Like your original, this expression is numerically well-behaved near t~0,
    but if t is *exactly* 0 it becomes log(0)-log(0). (Same behavior as your code.)
    """
    import numpy as np
    import pytensor.tensor as at

    dtype = getattr(mmin, "dtype", "float64")

    #epsv   = at.as_tensor_variable(eps, dtype=dtype)
    # one    = at.as_tensor_variable(1.0, dtype=dtype)
    tiny_r = 1e-12 #at.as_tensor_variable(1e-12, dtype=dtype)  # same idea as your (1 + 1e-12)
    INF    = np.inf #at.as_tensor_variable(np.inf, dtype=dtype)

    # sanitize bounds
    mmin_c = at.clip(mmin, eps, INF)
    mmax_c = at.clip(mmax, eps, INF)
    mmax_c = at.maximum(mmax_c, mmin_c * (1. + tiny_r))

    t = 1 - alpha

    # a = log(mmax), b = log(mmin), Δ = a - b
    b = at.log(mmin_c)
    delta = at.log(mmax_c) - b

    # stable closed form
    return (t * b
            + at.log(at.abs(at.expm1(t * delta)))
            - at.log(at.abs(t)))


def log_norm_truncated_pl_num_0(alpha, mmin, mmax, eps=1e-12):
    """
    log ∫_{mmin}^{mmax} m^{-alpha} dm
    = log( (mmax^(1-α) - mmin^(1-α)) / (1-α) ), with a stable α≈1 path, no switch.
    """
    # sanitize bounds
    #dtype = mmin.dtype
    #epsv  = at.as_tensor_variable(eps, dtype=dtype)

    mmin_c = at.clip(mmin, eps, INF)
    mmax_c = at.maximum(at.clip(mmax, eps, INF), mmin_c * (1.0 + 1e-12))

    # t = 1 - alpha
    t   = 1. - alpha

    # Let a = log(mmax), b = log(mmin), Δ = a - b > 0
    a = at.log(mmax_c)
    b = at.log(mmin_c)
    delta = a - b

    # For t≠0: log( (mmax^t - mmin^t) / |t| )
    # = t*b + log(|expm1(t*Δ)|) - log(|t|).
    # This expression is *also* the correct continuous limit at t→0 (α→1):
    # as t→0, log(|expm1(t*Δ)|) - log|t| → log(Δ), giving log(log(mmax/mmin)).
    return (t * b
            + at.log(at.abs(at.expm1(t * delta)))
            - at.log(at.abs(t)))
    

####### Power Law + Peak smooth edges , LVK low-end ########



def logpdf_PLP_reg(theta, lambdaBBHmass,  smoothing='LVK'):
    
        m1, m2 = theta
        lambdaPeak, alpha, beta, deltam, ml, mh, muMass, sigmaMass = lambdaBBHmass
                

        lpdfm1 = logpdfm1_PLP_reg( m1, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing=smoothing )
        
        lpdfm2 = logpdfm2_PLP_reg(m2, beta, deltam, ml, smoothing=smoothing)
        
        ln = logNorm_PLP_reg( lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing=smoothing )
        
        return lpdfm1 +lpdfm2-ln-logC_PLP_reg(m1, beta, deltam,  ml, smoothing=smoothing) 
        


 
def logpdfm1_PLP_reg(m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, sl=0.05, sh=0.05, smoothing='LVK'):

    return logpdfm1_PLP_noreg(m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing=smoothing)  + log_sigmoid(m, ml, sl) + at.log(1.0 -safe_sigmoid(m, mh, sh)) 
    
    # at.log(1-sigmoid(m, mh, sh))  #log1m_sigmoid_stable(m, mh, sh)
    #at.log(1-safe_sigmoid(m, mh, sh)) 
    #+ log1m_sigmoid_stable(m, mh, sh)

def logpdfm1_PLP_noreg(m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing='LVK'):

    #half_ = at.as_tensor_variable(0.5, dtype=m.dtype)
    #two_pi_ = at.as_tensor_variable(2*PI, dtype=m.dtype)
    
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
        left_edge  = 1.0 - safe_sigmoid(m, m_g, sig_g_low )
        right_edge = safe_sigmoid(m, m_g + w_g, sig_g_high )
        
        # Smooth mask transitions from 1 to 0 over the window [m_g, m_g + w_g]
        mask = at.log( left_edge + right_edge )
        
        # Smoothly blend between lpdfval and MIN
        return mask + lpdfval
        




def logC_PLP_reg( m, beta, deltam, ml, res=500, smoothing='LVK'):
    '''
    Gives log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
    '''
      

    if res!=500:
        _tgrid = at.linspace(0, 1, res)
    else:
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
    
    #itr = atinterp( m, xx[1:], at.log(cdf) )
    
    x0 = xx[1]                 # because you used xx[1:] for interpolation
    x1 = xx[-1]
    nU = xx.shape[0] - 1       # length of xx[1:]
    itr = atinterp_uniform(m, x0, x1, nU, at.log(cdf))

    return itr



def logNorm_PLP_reg( lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing='LVK', res=500):
    
    '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

    '''


    if res!=500:
        _tgrid = at.linspace(0, 1, res)
    else:
        _tgrid = _get_t_grid()
        
    ms = ml + (mh - ml) * _tgrid 
    
    ps = at.exp( logpdfm1_PLP_noreg( ms , lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing=smoothing  ))

    return  at.log( attrapzvec(ps, ms) )





def logpdf_PLPreg_from_interp(theta, interp_vals, interp_grids, force_m2_less_than_m1=False):
    """
    Log joint pdf p(m1, m2) for the PLPreg mass model, using precomputed grids.

    Uses the same interpolation machinery as logpdf_DPLDP_from_interp.
    interp_vals  = [lp_m1_grid, lp_m2_grid, lC_of_m1, ln]
    interp_grids = [m1_grid, m2_grid]
    """
    return logpdf_DPLDP_from_interp(
        theta,
        interp_vals,
        interp_grids,
        force_m2_less_than_m1=force_m2_less_than_m1,
    )

    

####### double Power Law + double Peak  LVK low-end ########

def log_broken_power_law_DPLDP_pdf(m1, alpha1, alpha2, mb, m1_low, m_high, sh=0.05, sl=0.05, epsilon=0.01):
    
    # log normalization constant: does NOT depend on m1
    norm1 = (m_high * (m_high / mb) ** (-alpha2) - mb) / (-alpha2 + 1)
    norm2 = (mb - m1_low * (m1_low / mb) ** (-alpha1)) / (-alpha1 + 1)
    log_N = at.log(norm1 + norm2)

    # precompute log(m1/mb) once
    log_m1_over_mb = at.log(m1 / mb)
    log_val1 = -alpha1 * log_m1_over_mb
    log_val2 = -alpha2 * log_m1_over_mb

    # Smooth weight function
    w = safe_sigmoid(-m1, -mb, epsilon)
    log_w     = at.log(w)
    log_1mw   = at.log1p(-w)

    log_mix_val = logsumexp(
        log_w   + log_val1,
        log_1mw + log_val2,
    )

    s1 = at.log1p(-safe_sigmoid(m1, m_high, sh))
    s2 = at.log(safe_sigmoid(m1, m1_low, sl))

    return log_mix_val - log_N + s1 + s2


def logpdfm1_DPLDP(m1, alpha1, alpha2, mb,
    mu1, sigma1, mu2, sigma2,
    m1_low, m_high, delta_m1,
    lambda0, lambda1,
    epsilon,
    smoothing='LVK', simplex_repair=False, eps_w=1e-15):


    #work_dtype = getattr(m1, "dtype", "float64")

    #one = at.as_tensor_variable(1.0, dtype=work_dtype)

    # eps_w = at.as_tensor_variable(
    #     1e-6 if str(work_dtype) == "float32" else 1e-12,
    #     dtype=work_dtype
    # )

    if not simplex_repair:
        #log_lambda0 = at.log(lambda0)
        #log_lambda1 = at.log(lambda1)

        log_lambda0 = at.log(at.clip(lambda0, eps_w, 1.0-eps_w))
        log_lambda1 = at.log(at.clip(lambda1, eps_w, 1.0-eps_w))

        lambda2_raw  = 1. - lambda0 - lambda1
        lambda2_safe = at.clip(lambda2_raw, eps_w, 1.0-eps_w)
        log_lambda2  = at.log(lambda2_safe)

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
    log_pnorm1 = truncGausslowerupper_at_lpdf(m1, mu1, sigma1, xmin=m1_low, xmax=m_high)
    log_pnorm2 = truncGausslowerupper_at_lpdf(m1, mu2, sigma2, xmin=m1_low, xmax=m_high)

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

    return log_S + log_mix



def logpdf_DPLDP_from_interp(theta, interp_vals, interp_grids, force_m2_less_than_m1=False):
    """
    Interpolation-only evaluator for the non-evolving DPLDP model,
    on non-uniform m1 and m2 grids.

    interp_grids = [m1_grid, m2_grid]
    interp_vals  = [lp_m1_grid, lp_m2_grid, lC_of_m1, ln]

    Returns log p(m1,m2) = log p(m1) + log p(m2) - log C(m1) - ln
    """

    m1, m2 = theta

    m1_grid, m2_grid = interp_grids
    lp_m1_grid, lp_m2_grid, lC_of_m1, ln = interp_vals

    # ------------------------------------------------------------
    # 0) HARD SUPPORT MASK  (same philosophy as DPLDP-z)
    # ------------------------------------------------------------
    ok = (
        (m1 >= m1_grid[0]) & (m1 <= m1_grid[-1]) &
        (m2 >= m2_grid[0]) & (m2 <= m2_grid[-1])
    )

    if force_m2_less_than_m1:
        ok = ok & (m2 <= m1)

    # avoid C(m1)=0 zone (logC=-inf -> +inf in joint)
    ok = ok & (m1 > m2_grid[0])

    # ------------------------------------------------------------
    # 1) SAFE indices + weights
    # ------------------------------------------------------------
    j1, r1 = _interp_indices_nonuniform_safe(m1, m1_grid)
    j2, r2 = _interp_indices_nonuniform_safe(m2, m2_grid)

    # ------------------------------------------------------------
    # 2) Interpolate log p(m1)
    # ------------------------------------------------------------
    yl_m1 = lp_m1_grid[j1 - 1]
    yh_m1 = lp_m1_grid[j1]
    lpdfm1 = (1.0 - r1) * yl_m1 + r1 * yh_m1

    # ------------------------------------------------------------
    # 3) Interpolate log C(m1)
    # ------------------------------------------------------------
    yl_C = lC_of_m1[j1 - 1]
    yh_C = lC_of_m1[j1]
    lC = (1.0 - r1) * yl_C + r1 * yh_C

    # ------------------------------------------------------------
    # 4) Interpolate log p(m2)
    # ------------------------------------------------------------
    yl_m2 = lp_m2_grid[j2 - 1]
    yh_m2 = lp_m2_grid[j2]
    lpdfm2 = (1.0 - r2) * yl_m2 + r2 * yh_m2

    # ------------------------------------------------------------
    # 5) Combine
    # ------------------------------------------------------------
    lpdf = lpdfm1 + lpdfm2 - lC - ln

    return at.where(ok, lpdf, -np.inf)

    
def logpdf_DPLDP_from_interp_02(theta, interp_vals, interp_grids, force_m2_less_than_m1=False):

    m1, m2 = theta

    m1_grid = interp_grids[0]
    m2_grid = interp_grids[1]
    lp_m1_grid, lp_m2_grid, lC_of_m1, ln = interp_vals

    # ----- M1 interpolation indices computed once (non-uniform grid) -----
    j1, r1 = _interp_indices_nonuniform(m1, m1_grid)

    # interpolate logpdf(m1)
    yl_m1 = lp_m1_grid[j1 - 1]
    yh_m1 = lp_m1_grid[j1]
    lpdfm1 = (1.0 - r1) * yl_m1 + r1 * yh_m1

    # interpolate C(m1)
    yl_C = lC_of_m1[j1 - 1]
    yh_C = lC_of_m1[j1]
    lC   = (1.0 - r1) * yl_C + r1 * yh_C

    # ----- M2 interpolation indices (non-uniform grid) -----
    j2, r2 = _interp_indices_nonuniform(m2, m2_grid)

    # interpolate logpdf(m2)
    yl_m2 = lp_m2_grid[j2 - 1]
    yh_m2 = lp_m2_grid[j2]
    lpdfm2 = (1.0 - r2) * yl_m2 + r2 * yh_m2

    # ----- combine -----
    lpdf = lpdfm1 + lpdfm2 - lC - ln

    if force_m2_less_than_m1:
        eval = at.and_(at.and_(m2 <= m1, m2 > 0), m1 > 0)
        return at.where(eval, lpdf, -np.inf )
    else:
        return lpdf


def logpdf_DPLDP_from_interp_lin(theta, interp_vals, interp_grids, force_m2_less_than_m1=False):

        m1, m2 = theta
    
        m1_grid = interp_grids[0]
        m2_grid = interp_grids[1]
        lp_m1_grid, lp_m2_grid, lC_of_m1, ln = interp_vals

        # # ----- M1 interpolation indices computed once -----
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


        ########### my v1
            
        # x0_1 = m1_grid[0]
        # x1_1 = m1_grid[-1]
        # nU_1 = m1_grid.shape[0] - 1
    
        # lpdfm1 = atinterp_uniform(m1, x0_1, x1_1, nU_1, lp_m1_grid)


        # x0_2 = m2_grid[0]
        # x1_2 = m2_grid[-1]
        # nU_2 = m2_grid.shape[0] - 1
    
        # lpdfm2 = atinterp_uniform(m2, x0_2, x1_2, nU_2, lp_m2_grid)


        # lC = atinterp_uniform(m1, x0_1, x1_1, nU_1, lC_of_m1)
        
 

        lpdf = lpdfm1 + lpdfm2 -lC -ln

        if force_m2_less_than_m1:
            eval = at.and_(at.and_(m2 <= m1, m2 > 0), m1 > 0)
            return at.where(eval, lpdf, -np.inf )
        else:
            return lpdf
    


def logpdf_DPLDP(theta, lambdaBBHmass, force_m2_less_than_m1=False, has_m2_break=False, smoothing='LVK', resC=100, resN=500, interp_vals=None, interp_grids=None, norm=True, simplex_repair=False):
    
        m1, m2 = theta
        alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, beta, m2_low, delta_m2, epsilon, m_g, w_g, sig_g_low, sig_g_high = lambdaBBHmass
                

        if interp_vals is None:
            
            lpdfm1 = logpdfm1_DPLDP( m1, alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, epsilon, smoothing=smoothing, simplex_repair=simplex_repair)
        
            lpdfm2 = logpdfm2_PLP_reg(m2, beta, delta_m2, m2_low, m_g=m_g, w_g=w_g, sig_g_low = sig_g_low, sig_g_high = sig_g_high, has_m2_break=has_m2_break, smoothing=smoothing)
            
            lC = logC_DPLDP(m1, beta, delta_m2,  m2_low, m_g=m_g, w_g=w_g, sig_g_low = sig_g_low, sig_g_high = sig_g_high, has_m2_break=has_m2_break, smoothing=smoothing, res=resC) 
            if norm:
                ln = logNorm_DPLDP(  alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, epsilon, smoothing=smoothing, res=resN)
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
        
     

def logC_DPLDP( m, beta, deltam, m2_low, m_g=45, w_g=80, sig_g_low=5, sig_g_high = 5, has_m2_break=False, res=500, smoothing='LVK'):
    '''
    Gives log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
    '''

    if res!=500:
        _tgrid = at.linspace(0, 1, res)
    else:
        _tgrid = _get_t_grid()
    
    xx = m2_low + (max_m - m2_low) * _tgrid 
        
    p2 = at.exp( logpdfm2_PLP_reg( xx , beta, deltam, m2_low, m_g=m_g, w_g=w_g, sig_g_low=sig_g_low, sig_g_high = sig_g_high, has_m2_break=has_m2_break, smoothing=smoothing))
    
    cdf = atcumtrapz( p2, xx, )
    cdf = at.clip(cdf, 1e-300, np.inf)

    #itr = atinterp( m, xx[1:], at.log(cdf) )

    # grid endpoints and size (xx is uniform)
    x0 = xx[1]                 # because you used xx[1:] for interpolation
    x1 = xx[-1]
    nU = xx.shape[0] - 1       # length of xx[1:]
    #print("ising uniform interp")
    itr = atinterp_uniform(m, x0, x1, nU, at.log(cdf))
    
    return itr




def logNorm_DPLDP( alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, epsilon, res=500, smoothing='LVK', simplex_repair=False):
    
    '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )
    '''
    
    if res!=500:
        _tgrid = at.linspace(0, 1, res)
    else:
        _tgrid = _get_t_grid()
    
    ms = m1_low + (m_high - m1_low) * _tgrid 
            
    lpdf = logpdfm1_DPLDP( ms , alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, epsilon, smoothing=smoothing, simplex_repair=simplex_repair  )
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
        joint = at.where(eval, lpdfval, -np.inf )
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


####### DPLDP-z ########



def theta_of_z(z, theta_0, theta_inf, z_t, delta_z):
    """
    Generic redshift evolution for a hyperparameter, in the spirit of Eq. (2):
        θ(z) = θ0 + (θ_inf - θ0) * s(z; z_t, Δz),

    where s is a smooth sigmoid between 0 and 1.
    Works with scalar or array z (broadcasts).
    """
    x = (z - z_t) / delta_z
    # tanh-based sigmoid: smoothly goes from 0 to 1 around z_t
    s = 0.5 * (1.0 + at.tanh(x))
    return theta_0 + (theta_inf - theta_0) * s

def logpdfm1_DPLDP_z(
    m1, z,
    # low-z hyperparameters
    alpha1_0, alpha2_0, mb_0,
    mu1_0, sigma1_0, mu2_0, sigma2_0,
    m1_low, m_high, delta_m1,
    lambda0_0, lambda1_0,
    epsilon,
    # evolution hyperparameters for each θ in {alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2}
    alpha1_inf, z_alpha1, dz_alpha1,
    alpha2_inf, z_alpha2, dz_alpha2,
    mb_inf,    z_mb,     dz_mb,
    mu1_inf,   z_mu1,    dz_mu1,
    sigma1_inf,z_sigma1, dz_sigma1,
    mu2_inf,   z_mu2,    dz_mu2,
    sigma2_inf,z_sigma2, dz_sigma2,
    # NEW: mixture evolution specified by endpoints + shared (z_lambda, dz_lambda)
    lambda0_inf, lambda1_inf, z_lambda, dz_lambda,
    smoothing='LVK',
    simplex_repair=False
):
    """
    Redshift-evolving version of logpdfm1_DPLDP with:
      - shape parameters evolved via theta_of_z(...)
      - mixture weights evolved as a convex combination

        lambda(z) = (1 - S_lambda(z)) * lambda_0 + S_lambda(z) * lambda_inf,

      where S_lambda(z) = 0.5 * (1 + tanh((z - z_lambda)/dz_lambda)).
    """

    # --- shape evolution as before ---
    alpha1  = theta_of_z(z, alpha1_0,  alpha1_inf,  z_alpha1,  dz_alpha1)
    alpha2  = theta_of_z(z, alpha2_0,  alpha2_inf,  z_alpha2,  dz_alpha2)
    mb      = theta_of_z(z, mb_0,      mb_inf,      z_mb,      dz_mb)
    mu1     = theta_of_z(z, mu1_0,     mu1_inf,     z_mu1,     dz_mu1)
    sigma1  = theta_of_z(z, sigma1_0,  sigma1_inf,  z_sigma1,  dz_sigma1)
    mu2     = theta_of_z(z, mu2_0,     mu2_inf,     z_mu2,     dz_mu2)
    sigma2  = theta_of_z(z, sigma2_0,  sigma2_inf,  z_sigma2,  dz_sigma2)

    # --- shared S_lambda(z) for the mixture weights ---
    x_l = (z - z_lambda) / dz_lambda
    S_l = 0.5 * (1.0 + at.tanh(x_l))  # same shape as in logpdf_DPLDP_z / logNorm_DPLDP_z

    # low-z and high-z λ2 from simplex
    lambda2_0   = 1.0 - lambda0_0 - lambda1_0
    lambda2_inf = 1.0 - lambda0_inf - lambda1_inf

    # convex combination: λ(z) = (1-S) λ(0) + S λ(∞)
    lambda0 = (1.0 - S_l) * lambda0_0 + S_l * lambda0_inf
    lambda1 = (1.0 - S_l) * lambda1_0 + S_l * lambda1_inf
    lambda2 = (1.0 - S_l) * lambda2_0 + S_l * lambda2_inf  # not passed explicitly, but useful conceptually

    # --- call your original m1 logpdf with z-dependent quantities ---
    return logpdfm1_DPLDP(
        m1,
        alpha1, alpha2, mb,
        mu1, sigma1, mu2, sigma2,
        m1_low, m_high, delta_m1,
        lambda0, lambda1,
        epsilon,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
    )


def logpdfm1_DPLDP_z_0(
    m1, z,
    # low-z hyperparameters (your current λ_BBHmass pieces)
    alpha1_0, alpha2_0, mb_0,
    mu1_0, sigma1_0, mu2_0, sigma2_0,
    m1_low, m_high, delta_m1,
    lambda0_0, lambda1_0,
    epsilon,
    # evolution hyperparameters for each θ in {alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, lambda0, lambda1}
    alpha1_inf, z_alpha1, dz_alpha1,
    alpha2_inf, z_alpha2, dz_alpha2,
    mb_inf,    z_mb,     dz_mb,
    mu1_inf,   z_mu1,    dz_mu1,
    sigma1_inf,z_sigma1, dz_sigma1,
    mu2_inf,   z_mu2,    dz_mu2,
    sigma2_inf,z_sigma2, dz_sigma2,
    lambda0_inf, z_lambda0, dz_lambda0,
    lambda1_inf, z_lambda1, dz_lambda1,
    smoothing='LVK',
    simplex_repair=False
    
):
    """
    Redshift-evolving version of logpdfm1_DPLDP.
    All heavy lifting still done by your original logpdfm1_DPLDP;
    this function only constructs θ(z).
    """

    # --- build z-dependent hyperparameters using Eq.(2)-style evolution ---
    alpha1  = theta_of_z(z, alpha1_0,  alpha1_inf,  z_alpha1,  dz_alpha1)
    alpha2  = theta_of_z(z, alpha2_0,  alpha2_inf,  z_alpha2,  dz_alpha2)
    mb      = theta_of_z(z, mb_0,      mb_inf,      z_mb,      dz_mb)
    mu1     = theta_of_z(z, mu1_0,     mu1_inf,     z_mu1,     dz_mu1)
    sigma1  = theta_of_z(z, sigma1_0,  sigma1_inf,  z_sigma1,  dz_sigma1)
    mu2     = theta_of_z(z, mu2_0,     mu2_inf,     z_mu2,     dz_mu2)
    sigma2  = theta_of_z(z, sigma2_0,  sigma2_inf,  z_sigma2,  dz_sigma2)
    lambda0 = theta_of_z(z, lambda0_0, lambda0_inf, z_lambda0, dz_lambda0)
    lambda1 = theta_of_z(z, lambda1_0, lambda1_inf, z_lambda1, dz_lambda1)


    # --- now call your original m1 logpdf with these z-dependent quantities ---
    return logpdfm1_DPLDP(
        m1,
        alpha1, alpha2, mb,
        mu1, sigma1, mu2, sigma2,
        m1_low, m_high, delta_m1,
        lambda0, lambda1,
        epsilon,
        smoothing=smoothing,
        simplex_repair=simplex_repair
    )


def logpdf_DPLDP_z(
    theta, z,
    lambdaBBHmass_lowz,
    evo_params,
    force_m2_less_than_m1=False,
    has_m2_break=False,
    smoothing='LVK',
    resC=100, resN=500,
    interp_vals=None, interp_grids=None,
    simplex_repair=False
):
    """
    Redshift-evolving wrapper around your original logpdf_DPLDP.

    Parameters
    ----------
    theta : (m1, m2)
    z     : redshift (scalar or array broadcasting with m1, m2)
    lambdaBBHmass_lowz :
        Same vector you currently pass to logpdf_DPLDP, interpreted as
        the z≈0 values of the hyperparameters.
    evo_params :
        Flat tuple/array of the *evolution* hyperparameters, ordered as:
          (alpha1_inf, z_alpha1, dz_alpha1,
           alpha2_inf, z_alpha2, dz_alpha2,
           mb_inf,    z_mb,     dz_mb,
           mu1_inf,   z_mu1,    dz_mu1,
           sigma1_inf,z_sigma1, dz_sigma1,
           mu2_inf,   z_mu2,    dz_mu2,
           sigma2_inf,z_sigma2, dz_sigma2,
           lambda0_inf, lambda1_inf, z_lambda, dz_lambda)
    """

    m1, m2 = theta

    # unpack low-z hyperparameters (exactly your current order)
    (alpha1_0, alpha2_0, mb_0,
     mu1_0, sigma1_0, mu2_0, sigma2_0,
     m1_low, m_high, delta_m1,
     lambda0_0, lambda1_0,
     beta, m2_low, delta_m2,
     epsilon, m_g, w_g, sig_g_low, sig_g_high) = lambdaBBHmass_lowz

    # unpack evolution parameters
    (alpha1_inf,  z_alpha1,  dz_alpha1,
     alpha2_inf,  z_alpha2,  dz_alpha2,
     mb_inf,      z_mb,      dz_mb,
     mu1_inf,     z_mu1,     dz_mu1,
     sigma1_inf,  z_sigma1,  dz_sigma1,
     mu2_inf,     z_mu2,     dz_mu2,
     sigma2_inf,  z_sigma2,  dz_sigma2,
     lambda0_inf, lambda1_inf, z_lambda, dz_lambda) = evo_params

    # --- build z-dependent hyperparameters for the shape parameters ---
    alpha1  = theta_of_z(z, alpha1_0,  alpha1_inf,  z_alpha1,  dz_alpha1)
    alpha2  = theta_of_z(z, alpha2_0,  alpha2_inf,  z_alpha2,  dz_alpha2)
    mb      = theta_of_z(z, mb_0,      mb_inf,      z_mb,      dz_mb)
    mu1     = theta_of_z(z, mu1_0,     mu1_inf,     z_mu1,     dz_mu1)
    sigma1  = theta_of_z(z, sigma1_0,  sigma1_inf,  z_sigma1,  dz_sigma1)
    mu2     = theta_of_z(z, mu2_0,     mu2_inf,     z_mu2,     dz_mu2)
    sigma2  = theta_of_z(z, sigma2_0,  sigma2_inf,  z_sigma2,  dz_sigma2)

    # --- shared S_lambda(z) for the mixture weights ---
    x_l    = (z - z_lambda) / dz_lambda
    S_l    = 0.5 * (1.0 + at.tanh(x_l))

    lambda2_0   = 1.0 - lambda0_0 - lambda1_0
    lambda2_inf = 1.0 - lambda0_inf - lambda1_inf

    lambda0 = (1.0 - S_l) * lambda0_0 + S_l * lambda0_inf
    lambda1 = (1.0 - S_l) * lambda1_0 + S_l * lambda1_inf
    lambda2 = (1.0 - S_l) * lambda2_0 + S_l * lambda2_inf

    # (we only pass lambda0, lambda1 to logpdfm1_DPLDP; lambda2 is implied)

    # rebuild a z-dependent mass-parameter vector for the downstream calls
    lambdaBBHmass_z = (
        alpha1, alpha2, mb,
        mu1, sigma1, mu2, sigma2,
        m1_low, m_high, delta_m1,
        lambda0, lambda1,
        beta, m2_low, delta_m2,
        epsilon, m_g, w_g, sig_g_low, sig_g_high
    )

    # now just call your original logpdf_DPLDP
    lpdf_ =  logpdf_DPLDP(
        theta,
        lambdaBBHmass_z,
        force_m2_less_than_m1=force_m2_less_than_m1,
        has_m2_break=has_m2_break,
        smoothing=smoothing,
        resC=resC, resN=resN,
        interp_vals=interp_vals,
        interp_grids=interp_grids,
        norm=False,
        simplex_repair=simplex_repair
    )

    ln = logNorm_DPLDP_z(
        z,
        alpha1_0, alpha2_0, mb_0, mu1_0, sigma1_0, mu2_0, sigma2_0,
        m1_low, m_high, delta_m1, lambda0_0, lambda1_0, epsilon,
        alpha1_inf, z_alpha1, dz_alpha1,
        alpha2_inf, z_alpha2, dz_alpha2,
        mb_inf,    z_mb,     dz_mb,
        mu1_inf,   z_mu1,    dz_mu1,
        sigma1_inf,z_sigma1, dz_sigma1,
        mu2_inf,   z_mu2,    dz_mu2,
        sigma2_inf,z_sigma2, dz_sigma2,
        lambda0_inf, lambda1_inf, z_lambda, dz_lambda,
        res=resN, smoothing=smoothing,
        simplex_repair=simplex_repair
    )

    return lpdf_ - ln



def logpdf_DPLDP_z_0(
    theta, z,
    lambdaBBHmass_lowz,
    evo_params,
    force_m2_less_than_m1=False,
    has_m2_break=False,
    smoothing='LVK',
    resC=100, resN=500,
    interp_vals=None, interp_grids=None,
    simplex_repair=False
):
    """
    Redshift-evolving wrapper around your original logpdf_DPLDP.

    Parameters
    ----------
    theta : (m1, m2)
    z     : redshift (scalar or array broadcasting with m1, m2)
    lambdaBBHmass_lowz :
        Same vector you currently pass to logpdf_DPLDP, interpreted as
        the z≈0 values of the hyperparameters.
    evo_params :
        A flat tuple/array of the *evolution* hyperparameters, ordered as:
          (alpha1_inf, z_alpha1, dz_alpha1,
           alpha2_inf, z_alpha2, dz_alpha2,
           mb_inf,    z_mb,     dz_mb,
           mu1_inf,   z_mu1,    dz_mu1,
           sigma1_inf,z_sigma1, dz_sigma1,
           mu2_inf,   z_mu2,    dz_mu2,
           sigma2_inf,z_sigma2, dz_sigma2,
           lambda0_inf, z_lambda0, dz_lambda0,
           lambda1_inf, z_lambda1, dz_lambda1)
    Other arguments are passed straight through to your original logpdf_DPLDP.
    """

    m1, m2 = theta

    # unpack low-z hyperparameters (exactly your current order)
    (alpha1_0, alpha2_0, mb_0,
     mu1_0, sigma1_0, mu2_0, sigma2_0,
     m1_low, m_high, delta_m1,
     lambda0_0, lambda1_0,
     beta, m2_low, delta_m2,
     epsilon, m_g, w_g, sig_g_low, sig_g_high) = lambdaBBHmass_lowz

    # unpack evolution parameters
    (alpha1_inf, z_alpha1, dz_alpha1,
     alpha2_inf, z_alpha2, dz_alpha2,
     mb_inf,     z_mb,     dz_mb,
     mu1_inf,    z_mu1,    dz_mu1,
     sigma1_inf, z_sigma1, dz_sigma1,
     mu2_inf,    z_mu2,    dz_mu2,
     sigma2_inf, z_sigma2, dz_sigma2,
     lambda0_inf, z_lambda0, dz_lambda0,
     lambda1_inf, z_lambda1, dz_lambda1) = evo_params

    # --- build z-dependent hyperparameters ---
    alpha1  = theta_of_z(z, alpha1_0,  alpha1_inf,  z_alpha1,  dz_alpha1)
    alpha2  = theta_of_z(z, alpha2_0,  alpha2_inf,  z_alpha2,  dz_alpha2)
    mb      = theta_of_z(z, mb_0,      mb_inf,      z_mb,      dz_mb)
    mu1     = theta_of_z(z, mu1_0,     mu1_inf,     z_mu1,     dz_mu1)
    sigma1  = theta_of_z(z, sigma1_0,  sigma1_inf,  z_sigma1,  dz_sigma1)
    mu2     = theta_of_z(z, mu2_0,     mu2_inf,     z_mu2,     dz_mu2)
    sigma2  = theta_of_z(z, sigma2_0,  sigma2_inf,  z_sigma2,  dz_sigma2)
    lambda0 = theta_of_z(z, lambda0_0, lambda0_inf, z_lambda0, dz_lambda0)
    lambda1 = theta_of_z(z, lambda1_0, lambda1_inf, z_lambda1, dz_lambda1)



    # rebuild a z-dependent mass-parameter vector for the downstream calls
    lambdaBBHmass_z = (
        alpha1, alpha2, mb,
        mu1, sigma1, mu2, sigma2,
        m1_low, m_high, delta_m1,
        lambda0, lambda1,
        beta, m2_low, delta_m2,
        epsilon, m_g, w_g, sig_g_low, sig_g_high
    )

    # now just call your original logpdf_DPLDP
    lpdf_ =  logpdf_DPLDP(
        theta,
        lambdaBBHmass_z,
        force_m2_less_than_m1=force_m2_less_than_m1,
        has_m2_break=has_m2_break,
        smoothing=smoothing,
        resC=resC, resN=resN,
        interp_vals=interp_vals,
        interp_grids=interp_grids,
        norm=False,
        simplex_repair=simplex_repair
    )
    
    ln = logNorm_DPLDP_z(
        z,
        alpha1_0, alpha2_0, mb_0, mu1_0, sigma1_0, mu2_0, sigma2_0,
        m1_low, m_high, delta_m1, lambda0_0, lambda1_0, epsilon,
        alpha1_inf, z_alpha1, dz_alpha1,
        alpha2_inf, z_alpha2, dz_alpha2,
        mb_inf,    z_mb,     dz_mb,
        mu1_inf,   z_mu1,    dz_mu1,
        sigma1_inf,z_sigma1, dz_sigma1,
        mu2_inf,   z_mu2,    dz_mu2,
        sigma2_inf,z_sigma2, dz_sigma2,
        lambda0_inf, z_lambda0, dz_lambda0,
        lambda1_inf, z_lambda1, dz_lambda1,
        res=resN, smoothing=smoothing,
        simplex_repair=simplex_repair
    )

    return lpdf_ - ln



def logNorm_DPLDP_z(
    z, 
    alpha1_0, alpha2_0, mb_0, mu1_0, sigma1_0, mu2_0, sigma2_0,
    m1_low, m_high, delta_m1, lambda0_0, lambda1_0, epsilon,
    alpha1_inf, z_alpha1, dz_alpha1,
    alpha2_inf, z_alpha2, dz_alpha2,
    mb_inf,    z_mb,     dz_mb,
    mu1_inf,   z_mu1,    dz_mu1,
    sigma1_inf,z_sigma1, dz_sigma1,
    mu2_inf,   z_mu2,    dz_mu2,
    sigma2_inf,z_sigma2, dz_sigma2,
    lambda0_inf, lambda1_inf, z_lambda, dz_lambda,
    smoothing="LVK",
    res=500,
    simplex_repair=False
):
    """
    Same semantics as your original logNorm_DPLDP_z, but:
    - we compute theta(z) once per z,
    - including mixture weights via a shared S_lambda(z),
    - then broadcast over m1_grid.

    Returns vector (Nevt,) of log-normalizations for each z.
    """

    # --- grid in m1, same as before ---
    if res != 500:
        _tgrid = at.linspace(0, 1, res)
    else:
        _tgrid = _get_t_grid()

    #work_dtype = getattr(z, "dtype", "float64")
    #_tgrid = at.as_tensor_variable(_tgrid, dtype=work_dtype)

    m1_grid = m1_low + (m_high - m1_low) * _tgrid  # (N1,)

    # --- make z a 1D tensor ---
    z = at.atleast_1d(z)
    K = z.shape[0]          # number of events
    N1 = m1_grid.shape[0]   # grid size in m1

    # --- evolve all shape hyperparameters ONLY over z (shape: (K,)) ---
    alpha1  = theta_of_z(z, alpha1_0,  alpha1_inf,  z_alpha1,  dz_alpha1)
    alpha2  = theta_of_z(z, alpha2_0,  alpha2_inf,  z_alpha2,  dz_alpha2)
    mb      = theta_of_z(z, mb_0,      mb_inf,      z_mb,      dz_mb)
    mu1     = theta_of_z(z, mu1_0,     mu1_inf,     z_mu1,     dz_mu1)
    sigma1  = theta_of_z(z, sigma1_0,  sigma1_inf,  z_sigma1,  dz_sigma1)
    mu2     = theta_of_z(z, mu2_0,     mu2_inf,     z_mu2,     dz_mu2)
    sigma2  = theta_of_z(z, sigma2_0,  sigma2_inf,  z_sigma2,  dz_sigma2)

    # --- shared S_lambda(z) for mixture weights ---
    x_l    = (z - z_lambda) / dz_lambda
    S_l    = 0.5 * (1.0 + at.tanh(x_l))

    lambda2_0   = 1.0 - lambda0_0 - lambda1_0
    lambda2_inf = 1.0 - lambda0_inf - lambda1_inf

    lambda0 = (1.0 - S_l) * lambda0_0 + S_l * lambda0_inf
    lambda1 = (1.0 - S_l) * lambda1_0 + S_l * lambda1_inf
    lambda2 = (1.0 - S_l) * lambda2_0 + S_l * lambda2_inf

    # --- broadcast to (K, N1) and flatten ---
    M_flat = at.tile(m1_grid, K)  # shape: (K * N1,)

    alpha1_flat  = at.repeat(alpha1,  N1)
    alpha2_flat  = at.repeat(alpha2,  N1)
    mb_flat      = at.repeat(mb,      N1)
    mu1_flat     = at.repeat(mu1,     N1)
    sigma1_flat  = at.repeat(sigma1,  N1)
    mu2_flat     = at.repeat(mu2,     N1)
    sigma2_flat  = at.repeat(sigma2,  N1)
    lambda0_flat = at.repeat(lambda0, N1)
    lambda1_flat = at.repeat(lambda1, N1)
    # lambda2_flat = at.repeat(lambda2, N1)  # not passed explicitly

    # --- evaluate m1 logpdf in one big vectorized call ---
    lp_flat = logpdfm1_DPLDP(
        M_flat,
        alpha1_flat, alpha2_flat, mb_flat,
        mu1_flat, sigma1_flat, mu2_flat, sigma2_flat,
        m1_low, m_high, delta_m1,
        lambda0_flat, lambda1_flat,
        epsilon,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
    )

    # reshape back to (K, N1) and integrate over m1
    logp = lp_flat.reshape((K, N1))

    return at.log(attrapzvec(at.exp(logp), m1_grid, axis=1))


def logNorm_DPLDP_z_0(
    z, 
    alpha1_0, alpha2_0, mb_0, mu1_0, sigma1_0, mu2_0, sigma2_0,
    m1_low, m_high, delta_m1, lambda0_0, lambda1_0, epsilon,
    alpha1_inf, z_alpha1, dz_alpha1,
    alpha2_inf, z_alpha2, dz_alpha2,
    mb_inf,    z_mb,     dz_mb,
    mu1_inf,   z_mu1,    dz_mu1,
    sigma1_inf,z_sigma1, dz_sigma1,
    mu2_inf,   z_mu2,    dz_mu2,
    sigma2_inf,z_sigma2, dz_sigma2,
    lambda0_inf, z_lambda0, dz_lambda0,
    lambda1_inf, z_lambda1, dz_lambda1,
    smoothing="LVK",
    res=500,
    simplex_repair=False
):
    """
    Same semantics as your original logNorm_DPLDP_z, but:
    - we compute theta(z) once per z,
    - then broadcast over m1_grid.

    Returns vector (Nevt,) of log-normalizations for each z.
    """

    # --- grid in m1, same as before ---
    if res != 500:
        _tgrid = at.linspace(0, 1, res)
    else:
        _tgrid = _get_t_grid()

    # make sure we don't upcast dtype by mistake
    #work_dtype = getattr(z, "dtype", "float64")
    #_tgrid = at.as_tensor_variable(_tgrid, dtype=work_dtype)

    m1_grid = m1_low + (m_high - m1_low) * _tgrid  # (N1,)

    # --- make z a 1D tensor ---
    z = at.atleast_1d(z)
    K = z.shape[0]          # number of events
    N1 = m1_grid.shape[0]   # grid size in m1

    # --- evolve all hyperparameters ONLY over z (shape: (K,)) ---
    alpha1  = theta_of_z(z, alpha1_0,  alpha1_inf,  z_alpha1,  dz_alpha1)
    alpha2  = theta_of_z(z, alpha2_0,  alpha2_inf,  z_alpha2,  dz_alpha2)
    mb      = theta_of_z(z, mb_0,      mb_inf,      z_mb,      dz_mb)
    mu1     = theta_of_z(z, mu1_0,     mu1_inf,     z_mu1,     dz_mu1)
    sigma1  = theta_of_z(z, sigma1_0,  sigma1_inf,  z_sigma1,  dz_sigma1)
    mu2     = theta_of_z(z, mu2_0,     mu2_inf,     z_mu2,     dz_mu2)
    sigma2  = theta_of_z(z, sigma2_0,  sigma2_inf,  z_sigma2,  dz_sigma2)
    lambda0 = theta_of_z(z, lambda0_0, lambda0_inf, z_lambda0, dz_lambda0)
    lambda1 = theta_of_z(z, lambda1_0, lambda1_inf, z_lambda1, dz_lambda1)

    # --- broadcast to (K, N1) and flatten ---
    # m1_grid is the same for all events: repeat it K times
    M_flat = at.tile(m1_grid, K)  # shape: (K * N1,)

    # each hyperparameter depends only on z, so we repeat each value N1 times
    alpha1_flat  = at.repeat(alpha1,  N1)
    alpha2_flat  = at.repeat(alpha2,  N1)
    mb_flat      = at.repeat(mb,      N1)
    mu1_flat     = at.repeat(mu1,     N1)
    sigma1_flat  = at.repeat(sigma1,  N1)
    mu2_flat     = at.repeat(mu2,     N1)
    sigma2_flat  = at.repeat(sigma2,  N1)
    lambda0_flat = at.repeat(lambda0, N1)
    lambda1_flat = at.repeat(lambda1, N1)

    # --- evaluate m1 logpdf in one big vectorized call ---
    lp_flat = logpdfm1_DPLDP(
        M_flat,
        alpha1_flat, alpha2_flat, mb_flat,
        mu1_flat, sigma1_flat, mu2_flat, sigma2_flat,
        m1_low, m_high, delta_m1,
        lambda0_flat, lambda1_flat,
        epsilon,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
    )

    # reshape back to (K, N1) and integrate over m1
    logp = lp_flat.reshape((K, N1))

    # integrate p(m1 | z) over m1 with the same trapezoidal rule
    # attrapzvec integrates along axis=1, returns shape (K,)
    return at.log(attrapzvec(at.exp(logp), m1_grid, axis=1))




def logpdf_DPLDP_z_from_interp_lin(theta, z, interp_vals, interp_grids, force_m2_less_than_m1=False):
    """
    Interpolation-only evaluator for the redshift-evolving DPLDP model.

    interp_grids = [m1_grid, m2_grid, z_bank]   (all assumed UNIFORM grids)
    interp_vals  = [lp_m1_bank, lp_m2_grid, lC_of_m1, ln_bank]

      lp_m1_bank: shape (K, N1)  with K=len(z_bank), N1=len(m1_grid)
      lp_m2_grid: shape (N2,)
      lC_of_m1:   shape (N1,)     (log C(m1)), z-independent
      ln_bank:    shape (K,)      (logNorm(z_k))

    Returns log p(m1,m2 | z) via:
      - interpolate in z between bank slices k and k+1
      - within each slice, interpolate in m1 and m2 as usual
    """
    m1, m2 = theta

    m1_grid = interp_grids[0]
    m2_grid = interp_grids[1]
    z_bank  = interp_grids[2]

    lp_m1_bank, lp_m2_grid, lC_of_m1, ln_bank = interp_vals  # note: ln_bank is vector now

    # ----- Z interpolation indices (uniform bank) -----
    z0 = z_bank[0]
    z1 = z_bank[-1]
    K  = z_bank.shape[0]
    kz, rz = uniform_interp_indices(z, z0, z1, K)  # kz in [0, K-2]

    # ----- M1 interpolation indices computed once -----
    x0_1 = m1_grid[0]
    x1_1 = m1_grid[-1]
    nU_1 = m1_grid.shape[0]
    j1, r1 = uniform_interp_indices(m1, x0_1, x1_1, nU_1)

    # interpolate logpdf(m1|z_k) and logpdf(m1|z_{k+1})
    lpdfm1_k  = (1 - r1) * lp_m1_bank[kz,     j1] + r1 * lp_m1_bank[kz,     j1 + 1]
    lpdfm1_k1 = (1 - r1) * lp_m1_bank[kz + 1, j1] + r1 * lp_m1_bank[kz + 1, j1 + 1]

    # interpolate ln(z_k) and ln(z_{k+1})
    ln_k  = ln_bank[kz]
    ln_k1 = ln_bank[kz + 1]

    # interpolate logC(m1) (z-independent)
    lC = (1 - r1) * lC_of_m1[j1] + r1 * lC_of_m1[j1 + 1]

    # ----- M2 interpolation indices computed once -----
    x0_2 = m2_grid[0]
    x1_2 = m2_grid[-1]
    nU_2 = m2_grid.shape[0]
    j2, r2 = uniform_interp_indices(m2, x0_2, x1_2, nU_2)

    # interpolate logpdf(m2) (z-independent)
    lpdfm2 = (1 - r2) * lp_m2_grid[j2] + r2 * lp_m2_grid[j2 + 1]

    # assemble slice logpdfs and interpolate in z
    lp_k  = lpdfm1_k  + lpdfm2 - lC - ln_k
    lp_k1 = lpdfm1_k1 + lpdfm2 - lC - ln_k1
    lpdf  = (1 - rz) * lp_k + rz * lp_k1

    if force_m2_less_than_m1:
        ok = at.and_(at.and_(m2 <= m1, m2 > 0), m1 > 0)
        return at.where(ok, lpdf, _const_like(m1, -np.inf))
    else:
        return lpdf



# -------------------------------
# Generic non-uniform 1D indices
# -------------------------------

def interp_logpdf_1d_nonuniform(x, x_grid, y_grid):
    """
    Interpolate y_grid(x_grid) to y(x) using your indexer.
    """
    j, r = _interp_indices_nonuniform(x, x_grid)
    yL = y_grid[j - 1]
    yR = y_grid[j]
    return (1.0 - r) * yL + r * yR


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

    # clip x into grid domain (avoid out-of-bounds indices)
    x_clip = at.clip(x, x_grid[0], x_grid[-1])

    # searchsorted gives insertion index in [0..N]
    j = at.searchsorted(x_grid, x_clip, side="right")

    # clamp to valid interpolation interval [1..N-1]
    j = at.clip(j, 1, N - 1)

    xL = x_grid[j - 1]
    xR = x_grid[j]
    denom = at.maximum(xR - xL, 1e-30)

    r = (x_clip - xL) / denom
    r = at.clip(r, 0.0, 1.0)

    return j, r



def _interp_indices_nonuniform(x, x_grid):
    """
    Robust non-uniform 1D interpolation indices.

    Returns:
      j  : right index, always in [1, N-1]
      r  : fraction in [0,1]
    """
    # ensure x is inside bounds (important!)
    x = at.clip(x, x_grid[0], x_grid[-1])

    # searchsorted with side="right" ensures:
    # if x == x_grid[0] -> j = 1 (NOT 0)
    j = at.searchsorted(x_grid, x, side="right")

    # enforce 1 <= j <= N-1
    j = at.clip(j, 1, x_grid.shape[0] - 1)

    xL = x_grid[j - 1]
    xR = x_grid[j]

    denom = at.maximum(xR - xL, 1e-12)
    r = (x - xL) / denom
    r = at.clip(r, 0.0, 1.0)

    return j, r

    
def _interp_indices_nonuniform_0(x, xs, eps_xs=1e-9):
    """
    Same I/O and math. More JAX-friendly: eps is consistently typed to xs.dtype.
    """
    x0 = xs[0]
    x1 = xs[-1]
    N = xs.shape[0]

    #eps_xs = at.as_tensor_variable(eps, dtype=xs.dtype)

    # clip queries slightly inside the grid to avoid boundary issues
    xq = at.clip(x, x0 + eps_xs, x1 - eps_xs)

    # searchsorted on the non-uniform grid
    idxs = at.searchsorted(xs, xq, side="right")
    idxs = stop_grad(at.clip(idxs, 1, N - 1))

    xl = xs[idxs - 1]
    xh = xs[idxs]

    denom = at.maximum(xh - xl, eps_xs)

    r = (xq - xl) / denom
    #r = at.cast(r, x.dtype)

    return idxs, r

    
def _interp_indices_nonuniform_0(x, xs, eps=1e-9):
    """
    Compute (idx, r) so that interpolation on a non-uniform grid xs
    can be done as:
        yl = ys[idx - 1]
        yh = ys[idx]
        y  = (1 - r) * yl + r * yh
    
    This is the same scheme as atinterp/atinterp_safe, but it returns
    the indices and weights so we can reuse them for multiple ys.
    """
    # x = at.as_tensor_variable(x)
    # xs = at.as_tensor_variable(xs)
    
    x0 = xs[0]
    x1 = xs[-1]
    
    # clip queries slightly inside the grid to avoid boundary issues
    xq = at.clip(x, x0 + eps, x1 - eps)
    #xq = stop_grad(xq)
    
    # searchsorted on the non-uniform grid
    idxs = at.searchsorted(xs, xq, side="right")
    N = xs.shape[0]
    # keep indices in [1, N-1] so (idx-1, idx) is always valid
    idxs = at.clip(idxs, 1, N - 1)
    
    # stop gradient through the discrete index selection
    idxs = stop_grad(idxs)
    
    # compute interpolation fraction r
    xl = xs[idxs - 1]
    xh = xs[idxs]
    eps_t = at.as_tensor_variable(eps, dtype=xl.dtype)
    denom = at.maximum(xh - xl, eps_t)
    
    r = (xq - xl) / denom
    r = at.cast(r, x.dtype)
    #r = stop_grad(r)
    
    return idxs, r


def logpdf_DPLDP_z_from_interp(theta, z, interp_vals, interp_grids, force_m2_less_than_m1=False):
    m1, m2 = theta

    m1_grid, m2_grid, z_bank = interp_grids
    lp_m1_bank, lp_m2_grid, lC_of_m1, ln_bank = interp_vals

    # ------------------------------------------------------------
    # 0) HARD SUPPORT MASK (this is the production fix)
    # ------------------------------------------------------------
    ok = (
        #at.isfinite(m1) & at.isfinite(m2) & at.isfinite(z)
        (m1 >= m1_grid[0]) & (m1 <= m1_grid[-1])
        & (m2 >= m2_grid[0]) & (m2 <= m2_grid[-1])
        & (z  >= z_bank[0])  & (z  <= z_bank[-1])
    )


    # optional physical constraint
    if force_m2_less_than_m1:
        ok = ok & (m2 <= m1)

    # CRITICAL: avoid C(m1)=0 region which would produce +inf
    ok = ok & (m1 > m2_grid[0])

    # ------------------------------------------------------------
    # 1) SAFE indices + weights
    # ------------------------------------------------------------
    kR, rz = _interp_indices_nonuniform_safe(z,  z_bank)
    kL = kR - 1

    j1, r1 = _interp_indices_nonuniform_safe(m1, m1_grid)
    j2, r2 = _interp_indices_nonuniform_safe(m2, m2_grid)

    # ------------------------------------------------------------
    # 2) Interpolate log p(m1 | z)
    # ------------------------------------------------------------
    yl_m1_L = lp_m1_bank[kL, j1 - 1]
    yh_m1_L = lp_m1_bank[kL, j1]
    lpdfm1_L = (1.0 - r1) * yl_m1_L + r1 * yh_m1_L

    yl_m1_R = lp_m1_bank[kR, j1 - 1]
    yh_m1_R = lp_m1_bank[kR, j1]
    lpdfm1_R = (1.0 - r1) * yl_m1_R + r1 * yh_m1_R

    lpdfm1 = (1.0 - rz) * lpdfm1_L + rz * lpdfm1_R

    # ------------------------------------------------------------
    # 3) Interpolate log C(m1)
    # ------------------------------------------------------------
    yl_C = lC_of_m1[j1 - 1]
    yh_C = lC_of_m1[j1]
    lC   = (1.0 - r1) * yl_C + r1 * yh_C

    # If logC is -inf or nan -> reject safely
    #ok = ok & at.isfinite(lC)

    # ------------------------------------------------------------
    # 4) Interpolate log p(m2)
    # ------------------------------------------------------------
    yl_m2 = lp_m2_grid[j2 - 1]
    yh_m2 = lp_m2_grid[j2]
    lpdfm2 = (1.0 - r2) * yl_m2 + r2 * yh_m2

    # ------------------------------------------------------------
    # 5) Interpolate ln_norm(z)
    # ------------------------------------------------------------
    ln_L = ln_bank[kL]
    ln_R = ln_bank[kR]
    ln   = (1.0 - rz) * ln_L + rz * ln_R

    #ok = ok & at.isfinite(ln)

    # ------------------------------------------------------------
    # 6) Assemble joint logpdf
    # ------------------------------------------------------------
    lpdf = lpdfm1 + lpdfm2 - lC - ln

    return at.where(ok, lpdf, -np.inf)



def logpdf_DPLDP_z_from_interp_02(theta, z, interp_vals, interp_grids, force_m2_less_than_m1=False):
    """
    Safe interpolation-only evaluator for the redshift-evolving DPLDP model
    on non-uniform grids in (m1, m2, z).

    Key robustness features:
      - Inputs are clipped to grid bounds BEFORE computing interpolation indices
        (prevents j=0 -> j-1=-1 wraparound and kL=-1 issues).
      - The returned logpdf is masked to -inf OUTSIDE the true support,
        so proposals outside support do not return garbage.

    Returns:
        log p(m1, m2 | z) = log p(m1|z) + log p(m2) - log C(m1) - logNorm(z)
    """

    m1, m2 = theta

    # unpack grids and precomputed tables
    m1_grid, m2_grid, z_bank = interp_grids
    lp_m1_bank, lp_m2_grid, lC_of_m1, ln_bank = interp_vals

    # -------------------------------
    # Define support (true domain)
    # -------------------------------
    m1_min = m1_grid[0]
    m1_max = m1_grid[-1]
    m2_min = m2_grid[0]
    m2_max = m2_grid[-1]
    z_min  = z_bank[0]
    z_max  = z_bank[-1]

    # "True support" mask (mathematically correct)
    m1_ok = at.and_(m1 >= m1_min, m1 <= m1_max)
    m2_ok = at.and_(m2 >= m2_min, m2 <= m2_max)
    z_ok  = at.and_(z  >= z_min,  z  <= z_max)

    ok = at.and_(at.and_(m1_ok, m2_ok), z_ok)

    if force_m2_less_than_m1:
        ok = at.and_(ok, at.and_(m2 <= m1, at.and_(m2 > 0, m1 > 0)))

    # -------------------------------
    # Clip inputs ONLY for safe indexing
    # (so indices are always valid)
    # -------------------------------
    m1_clip = at.clip(m1, m1_min, m1_max)
    m2_clip = at.clip(m2, m2_min, m2_max)
    z_clip  = at.clip(z,  z_min,  z_max)

    # -------------------------------
    # Indices + weights for each axis
    # -------------------------------
    # z: indices into z_bank
    kR, rz = _interp_indices_nonuniform(z_clip, z_bank)
    kL = kR - 1

    # m1: indices into m1_grid
    j1, r1 = _interp_indices_nonuniform(m1_clip, m1_grid)

    # m2: indices into m2_grid
    j2, r2 = _interp_indices_nonuniform(m2_clip, m2_grid)

    # -------------------------------
    # Interpolate p(m1 | z)
    # -------------------------------
    # At z = z_L
    yl_m1_L = lp_m1_bank[kL, j1 - 1]
    yh_m1_L = lp_m1_bank[kL, j1]
    lpdfm1_L = (1.0 - r1) * yl_m1_L + r1 * yh_m1_L

    # At z = z_R
    yl_m1_R = lp_m1_bank[kR, j1 - 1]
    yh_m1_R = lp_m1_bank[kR, j1]
    lpdfm1_R = (1.0 - r1) * yl_m1_R + r1 * yh_m1_R

    # Interpolate in z
    lpdfm1 = (1.0 - rz) * lpdfm1_L + rz * lpdfm1_R

    # -------------------------------
    # Interpolate C(m1) (z-independent)
    # -------------------------------
    yl_C = lC_of_m1[j1 - 1]
    yh_C = lC_of_m1[j1]
    lC   = (1.0 - r1) * yl_C + r1 * yh_C

    # -------------------------------
    # Interpolate p(m2) (z-independent)
    # -------------------------------
    yl_m2 = lp_m2_grid[j2 - 1]
    yh_m2 = lp_m2_grid[j2]
    lpdfm2 = (1.0 - r2) * yl_m2 + r2 * yh_m2

    # -------------------------------
    # Interpolate logNorm(z)
    # -------------------------------
    ln_L = ln_bank[kL]
    ln_R = ln_bank[kR]
    ln   = (1.0 - rz) * ln_L + rz * ln_R

    # -------------------------------
    # Assemble final logpdf
    # -------------------------------
    lpdf = lpdfm1 + lpdfm2 - lC - ln

    # IMPORTANT: enforce correct support
    # (also prevents out-of-support proposals from producing garbage)
    return at.where(ok, lpdf, -np.inf)



def logpdf_DPLDP_z_from_interp_01(theta, z, interp_vals, interp_grids, force_m2_less_than_m1=False):
    """
    Interpolation-only evaluator for the redshift-evolving DPLDP model,
    for *non-uniform* grids in m1, m2, and z.

    interp_grids = [m1_grid, m2_grid, z_bank]
      m1_grid : (N1,) non-uniform m1 grid
      m2_grid : (N2,) non-uniform m2 grid
      z_bank  : (K,)  non-uniform z grid

    interp_vals  = [lp_m1_bank, lp_m2_grid, lC_of_m1, ln_bank]
      lp_m1_bank : (K, N1)  log p(m1 | z_k) on m1_grid, for each z_k in z_bank
      lp_m2_grid : (N2,)    log p(m2) on m2_grid (z-independent)
      lC_of_m1   : (N1,)    log C(m1) on m1_grid   (z-independent)
      ln_bank    : (K,)     logNorm(z_k) for each z_k in z_bank

    Given theta = (m1, m2) and z (all can be vectors), returns
        log p(m1, m2 | z)
    by:
      - interpolating in m1 on each z-slice,
      - interpolating C(m1) in m1,
      - interpolating p(m2) in m2,
      - interpolating logNorm in z,
      - and combining as: log p = log p_m1 + log p_m2 - log C - logNorm.
    """

    m1, m2 = theta

    # unpack grids and precomputed tables
    m1_grid, m2_grid, z_bank = interp_grids
    lp_m1_bank, lp_m2_grid, lC_of_m1, ln_bank = interp_vals

    

    # -------------------------------
    # Indices + weights for each axis
    # -------------------------------
    # z: indices into z_bank
    kR, rz = _interp_indices_nonuniform(z,  z_bank)
    kL = kR - 1  # left neighbour slice in z

    # m1: indices into m1_grid (we will reuse these for lp_m1_bank and lC_of_m1)
    j1, r1 = _interp_indices_nonuniform(m1, m1_grid)

    # m2: indices into m2_grid
    j2, r2 = _interp_indices_nonuniform(m2, m2_grid)

    # -------------------------------
    # Interpolate p(m1 | z)
    # -------------------------------
    # At z = z_L (left slice)
    yl_m1_L = lp_m1_bank[kL, j1 - 1]
    yh_m1_L = lp_m1_bank[kL, j1]
    lpdfm1_L = (1.0 - r1) * yl_m1_L + r1 * yh_m1_L

    # At z = z_R (right slice)
    yl_m1_R = lp_m1_bank[kR, j1 - 1]
    yh_m1_R = lp_m1_bank[kR, j1]
    lpdfm1_R = (1.0 - r1) * yl_m1_R + r1 * yh_m1_R

    # Finally interpolate in z between slices L and R
    lpdfm1 = (1.0 - rz) * lpdfm1_L + rz * lpdfm1_R

    # -------------------------------
    # Interpolate C(m1) (z-independent)
    # -------------------------------
    yl_C = lC_of_m1[j1 - 1]
    yh_C = lC_of_m1[j1]
    lC   = (1.0 - r1) * yl_C + r1 * yh_C

    # -------------------------------
    # Interpolate p(m2) (z-independent)
    # -------------------------------
    yl_m2 = lp_m2_grid[j2 - 1]
    yh_m2 = lp_m2_grid[j2]
    lpdfm2 = (1.0 - r2) * yl_m2 + r2 * yh_m2

    # -------------------------------
    # Interpolate logNorm(z)
    # -------------------------------
    ln_L = ln_bank[kL]
    ln_R = ln_bank[kR]
    ln   = (1.0 - rz) * ln_L + rz * ln_R

    # -------------------------------
    # Assemble final logpdf
    # -------------------------------
    lpdf = lpdfm1 + lpdfm2 - lC - ln

    if force_m2_less_than_m1:
        ok = at.and_(at.and_(m2 <= m1, m2 > 0), m1 > 0)
        return at.where(ok, lpdf,  -np.inf )
    else:
        return lpdf





def build_m1_grid_DPLDP_z(
    z_bank,
    # low-z hyperparameters
    mu1_0, sigma1_0, mu2_0, sigma2_0, mb_0,
    # high-z (asymptotic) hyperparameters
    mu1_inf, sigma1_inf, mu2_inf, sigma2_inf, mb_inf,
    # evolution hyperparameters
    z_mu1, dz_mu1,
    z_sigma1, dz_sigma1,
    z_mu2, dz_mu2,
    z_sigma2, dz_sigma2,
    z_mb, dz_mb,
    # support for m1
    m1_low, m_high,
    delta_m1,
    # grid resolution controls
    n_peak=2500,      # points in the "interesting" band (peaks + break)
    n_tail_low=400,   # points in low-mass tail
    n_tail_high=400,  # points in high-mass tail
    k_sigma=4.0,      # how many sigmas around each Gaussian to cover
    n_taper=10, 
     n_taper_eff = 200
):
    """
    Symbolic non-uniform m1 grid for the DPLDP-z mass model (with redshift evolution).

    Structure:
      - low tail:   [m1_low, band_min)
      - Gaussian 1 window over all z
      - Gaussian 2 window over all z
      - mid band:   [band_min, band_max] envelope over both peaks + break
      - high tail:  [band_max, m_high]

    n_peak is split into:
      ~20% for Gaussian 1, ~20% for Gaussian 2, rest in the mid band.

    All points are:
      - inside (m1_low, m_high),
      - sorted,
      - deduplicated.
    """

    # ---- detach all hyperparameters for grid geometry (no grad through grid) ----
    mu1_0_s      = stop_grad(mu1_0)
    sigma1_0_s   = stop_grad(sigma1_0)
    mu2_0_s      = stop_grad(mu2_0)
    sigma2_0_s   = stop_grad(sigma2_0)
    mb_0_s       = stop_grad(mb_0)

    mu1_inf_s    = stop_grad(mu1_inf)
    sigma1_inf_s = stop_grad(sigma1_inf)
    mu2_inf_s    = stop_grad(mu2_inf)
    sigma2_inf_s = stop_grad(sigma2_inf)
    mb_inf_s     = stop_grad(mb_inf)

    z_mu1_s      = stop_grad(z_mu1)
    dz_mu1_s     = stop_grad(dz_mu1)
    z_sigma1_s   = stop_grad(z_sigma1)
    dz_sigma1_s  = stop_grad(dz_sigma1)
    z_mu2_s      = stop_grad(z_mu2)
    dz_mu2_s     = stop_grad(dz_mu2)
    z_sigma2_s   = stop_grad(z_sigma2)
    dz_sigma2_s  = stop_grad(dz_sigma2)
    z_mb_s       = stop_grad(z_mb)
    dz_mb_s      = stop_grad(dz_mb)

    m1_low_s     = stop_grad(m1_low)
    m_high_s     = stop_grad(m_high)
    delta_m1_s = stop_grad(delta_m1) 

    # dtype & tiny eps near boundaries
    #dtype = getattr(m1_low_s, "dtype", "float64")
    eps   = 1e-4 #at.as_tensor_variable(1e-4, dtype=dtype)

    # ensure z_bank is a tensor (but treated as constant for geometry)
    #z_bank = at.as_tensor_variable(z_bank)

    # global support (slightly shrunken to avoid exact boundaries)
    xmin = m1_low_s + eps
    xmax = m_high_s - eps
    span = at.maximum(xmax - xmin, 1e-06) #at.as_tensor_variable(1e-6, dtype=dtype))

    # -----  explicit taper window grid -----
    # make sure the window has nonzero width and lies in support
    # ----- explicit taper window grid (clustered near xmin) -----
    taper_hi = at.clip(xmin + at.maximum(delta_m1_s, 1e-6), xmin, xmax)
    taper_w  = at.maximum(taper_hi - xmin, 1e-6)
    
    if n_taper > 1:
        # cluster points near xmin using logarithmic spacing
        eps_t = 1e-4  # controls closeness of the first interior point (fraction of taper width)
        u = at.linspace(0.0, 1.0, n_taper)  # [0,1]
        t = at.exp(at.log(eps_t) * (1.0 - u))   # goes from eps_t -> 1
        t = (t - eps_t) / (1.0 - eps_t)         # rescale to [0,1]
        m1_taper = xmin + taper_w * t
    else:
        m1_taper = at.zeros((0,))

    # ---- evolve hyperparameters over z_bank (using detached params) ----
    mu1_z = theta_of_z(z_bank, mu1_0_s,  mu1_inf_s,  z_mu1_s,    dz_mu1_s)
    sigma1_z = theta_of_z(z_bank, sigma1_0_s, sigma1_inf_s, z_sigma1_s, dz_sigma1_s)

    mu2_z = theta_of_z(z_bank, mu2_0_s,  mu2_inf_s,  z_mu2_s,    dz_mu2_s)
    sigma2_z = theta_of_z(z_bank, sigma2_0_s, sigma2_inf_s, z_sigma2_s, dz_sigma2_s)

    mb_z = theta_of_z(z_bank, mb_0_s, mb_inf_s, z_mb_s, dz_mb_s)

    k_sigma_t = k_sigma #at.as_tensor_variable(k_sigma, dtype=dtype)

    # ---- Gaussian windows over all z ----
    # First for each z, then take global min/max over z.
    g1_min_z = mu1_z - k_sigma_t * at.abs(sigma1_z)
    g1_max_z = mu1_z + k_sigma_t * at.abs(sigma1_z)

    g2_min_z = mu2_z - k_sigma_t * at.abs(sigma2_z)
    g2_max_z = mu2_z + k_sigma_t * at.abs(sigma2_z)

    # global windows (clipped to support)
    g1_min = at.clip(at.min(g1_min_z), xmin, xmax)
    g1_max = at.clip(at.max(g1_max_z), xmin, xmax)
    g2_min = at.clip(at.min(g2_min_z), xmin, xmax)
    g2_max = at.clip(at.max(g2_max_z), xmin, xmax)

    tiny = 1e-6 * span
    g1_width = g1_max - g1_min
    g2_width = g2_max - g2_min

    has_g1 = at.gt(g1_width, tiny)
    has_g2 = at.gt(g2_width, tiny)

    # ---- global "interesting" band over all z (both peaks + break) ----
    peak_min_z = at.minimum(g1_min_z, g2_min_z)
    peak_min_z = at.minimum(peak_min_z, mb_z)

    peak_max_z = at.maximum(g1_max_z, g2_max_z)
    peak_max_z = at.maximum(peak_max_z, mb_z)

    band_min = at.clip(at.min(peak_min_z), xmin, xmax)
    band_max = at.clip(at.max(peak_max_z), xmin, xmax)  # min/max then clip

    band_width = at.maximum(band_max - band_min, tiny)

    # ---- split n_peak between Gaussians and the mid band ----
    # use Python ints; n_peak is passed as plain int in your code
    frac_gauss1 = 0.2
    frac_gauss2 = 0.2

    n_g1  = int(n_peak * float(frac_gauss1))
    n_g2  = int(n_peak * float(frac_gauss2))
    if n_g1 < 0: n_g1 = 0
    if n_g2 < 0: n_g2 = 0
    if n_g1 + n_g2 > n_peak:
        scale = float(n_peak) / float(n_g1 + n_g2)
        n_g1 = int(round(n_g1 * scale))
        n_g2 = int(round(n_g2 * scale))
    n_mid = max(n_peak - n_g1 - n_g2, 0)

    # ---- segments ----

    # 1) low tail: ideally [taper_hi, band_min), but keep fixed length (n_tail_low)
    # so it always has shape (n_tail_low,) and compiles.
    
    if n_tail_low > 0:
        denom_low = float(n_tail_low + 1)  # +1 so we avoid including the endpoint
        t_low = (at.arange(n_tail_low) + 1.0) / denom_low   # in (0,1)
    
        low_start = taper_hi
        low_width = band_min - low_start
    
        # fallback width if the segment would be empty or negative.
        # use something comparable to the taper resolution (not microscopic).
        # taper_w ~ delta_m1, so taper_w / n_taper is a sensible spacing scale.
        fallback_w = at.maximum(taper_w / at.maximum(n_taper, 1), 1e-3)  # Msun scale floor
    
        tail_good = low_start + low_width * t_low
        tail_fallback = low_start + fallback_w * t_low
    
        # if low_width > 0 -> use the good tail, else use fallback tail
        m1_low_tail = at.switch(at.gt(low_width, 0), tail_good, tail_fallback)
    
    else:
        m1_low_tail = at.zeros((0,))


    # 2) Gaussian 1: [g1_min, g1_max]
    if n_g1 > 0:
        if n_g1 > 1:
            denom_g1 = float(n_g1 - 1)
        else:
            denom_g1 = 1.0
        t_g1 = at.arange(n_g1) / denom_g1
        m1_g1 = g1_min + g1_width * t_g1
        # if the window is effectively degenerate, kill it

        fallback_width = 1e-08 * span  #  small compared to global support
        #fallback_width = at.maximum(1e-8 * span, 10.0 * ramp_step)
        
        # center the fallback at the midpoint of the proposed window (≈ mu1 band)
        g1_center = 0.5 * (g1_min + g1_max)
        g1_center = at.clip(g1_center, xmin + fallback_width, xmax - fallback_width)
        
        fallback_g1 = g1_center + fallback_width * (t_g1 - 0.5)  # monotone in t_g1

        m1_g1 = at.switch(has_g1, m1_g1, fallback_g1)
    else:
        m1_g1 = at.zeros((0,))

    # 3) Gaussian 2: [g2_min, g2_max]
    if n_g2 > 0:
        if n_g2 > 1:
            denom_g2 = float(n_g2 - 1)
        else:
            denom_g2 = 1.0
        t_g2 = at.arange(n_g2) / denom_g2
        m1_g2 = g2_min + g2_width * t_g2

        g2_center = 0.5 * (g2_min + g2_max)
        g2_center = at.clip(g2_center, xmin + fallback_width, xmax - fallback_width)
        fallback_g2 = g2_center + fallback_width * (t_g2 - 0.5)
        m1_g2 = at.switch(has_g2, m1_g2, fallback_g2)
        
    else:
        m1_g2 = at.zeros((0,))

    # 4) mid band: [band_min, band_max]
    if n_mid > 0:
        if n_mid > 1:
            denom_mid = float(n_mid - 1)
        else:
            denom_mid = 1.0
        t_mid = at.arange(n_mid) / denom_mid
        m1_mid = band_min + band_width * t_mid
    else:
        m1_mid = at.zeros((0,))

    # 5) high tail: [band_max, xmax]
    if n_tail_high > 0:
        denom_high = float(max(n_tail_high, 1))  # NOTE: n_tail_high (not n_tail_high-1)
        t_high = at.arange(n_tail_high) / denom_high  # in [0, 1) never hits 1
        m1_high_tail = band_max + (xmax - band_max) * t_high
    else:
        m1_high_tail = at.zeros((0,))

    # ---- combine, clip, sort, deduplicate ----
    m1_grid_raw = at.concatenate(
        [m1_taper, m1_low_tail, m1_g1, m1_g2, m1_mid, m1_high_tail],
        axis=0,
    )

    # just in case anything nudged outside [xmin, xmax]
    m1_grid_clipped = at.clip(m1_grid_raw, xmin, xmax)

    # enforce monotonicity and remove duplicates
    m1_grid_sorted = at.sort(m1_grid_clipped)
    
    Ntot = m1_grid_sorted.shape[0]

    # pick a ramp small enough that the *total* ramp never exceeds eps/2
    ramp_step = at.minimum(1e-6, 0.5 * eps / at.maximum(Ntot - 1, 1))
    
    ramp = ramp_step * at.arange(Ntot, dtype=m1_grid_sorted.dtype)
    
    m1_grid_strict = at.clip(m1_grid_sorted + ramp, xmin, xmax)
    return m1_grid_strict
    

def build_m1_grid_DPLDP_z_0(
    z_bank,
    # low-z hyperparameters
    mu1_0, sigma1_0, mu2_0, sigma2_0, mb_0,
    # high-z (asymptotic) hyperparameters
    mu1_inf, sigma1_inf, mu2_inf, sigma2_inf, mb_inf,
    # evolution hyperparameters
    z_mu1, dz_mu1,
    z_sigma1, dz_sigma1,
    z_mu2, dz_mu2,
    z_sigma2, dz_sigma2,
    z_mb, dz_mb,
    # support for m1
    m1_low, m_high,
    # grid resolution controls
    n_peak=2500,      # points in the "interesting" band (peaks + break)
    n_tail_low=400,   # points in low-mass tail
    n_tail_high=400,  # points in high-mass tail
    k_sigma=4.0,      # how many sigmas around each Gaussian to cover
):
    """
    Symbolic non-uniform m1 grid for the DPLDP-z mass model (with redshift evolution).

    All parameters are PyTensor scalars or vectors (random variables allowed).
    The grid is constructed *inside* the graph and depends on the evolved
    means/widths over the provided z_bank.

    Parameters
    ----------
    z_bank : 1D TensorVariable
        Redshift grid (e.g. from atools.make_z_grid(...)).
    mu1_0, sigma1_0, mu2_0, sigma2_0, mb_0 :
        Low-z (z~0) hyperparameters.
    mu1_inf, sigma1_inf, mu2_inf, sigma2_inf, mb_inf :
        Asymptotic (z->inf) hyperparameters.
    z_mu1, dz_mu1, z_sigma1, dz_sigma1, z_mu2, dz_mu2,
    z_sigma2, dz_sigma2, z_mb, dz_mb :
        Evolution hyperparameters used in theta_of_z.
    m1_low, m_high :
        Support bounds for m1.
    n_peak, n_tail_low, n_tail_high :
        Number of grid points in the central band and tails.
    k_sigma :
        How many sigmas around the peaks the band should cover.

    Returns
    -------
    m1_grid : 1D TensorVariable
        Non-uniform m1 grid spanning [m1_low, m_high], with:
        - fine spacing where peaks and break live (for all z in z_bank),
        - coarser spacing in low/high tails.
    """

    # --- dtype and constants ---
    #dtype = getattr(m1_low, "dtype", "float64")
    eps = 1e-03 #at.as_tensor_variable(1e-3, dtype=dtype)
    k_sigma_t = k_sigma #at.as_tensor_variable(k_sigma, dtype=dtype)

    # Ensure z_bank is a tensor
    #z_bank = at.as_tensor_variable(z_bank, dtype=dtype)

    # -------------------------------
    # Evolve hyperparameters over z_bank
    # -------------------------------
    # mu1(z), sigma1(z), mu2(z), sigma2(z), mb(z)

    mu1_z = theta_of_z(z_bank, mu1_0,  mu1_inf,  z_mu1,    dz_mu1)
    sigma1_z = theta_of_z(z_bank, sigma1_0, sigma1_inf, z_sigma1, dz_sigma1)

    mu2_z = theta_of_z(z_bank, mu2_0,  mu2_inf,  z_mu2,    dz_mu2)
    sigma2_z = theta_of_z(z_bank, sigma2_0, sigma2_inf, z_sigma2, dz_sigma2)

    mb_z = theta_of_z(z_bank, mb_0, mb_inf, z_mb, dz_mb)

    # -------------------------------
    # Global "interesting" band over all z
    # -------------------------------
    # For each z: cover both Gaussians ± k_sigma * sigma
    peak_min_z = at.minimum(
        mu1_z - k_sigma_t * at.abs(sigma1_z),
        mu2_z - k_sigma_t * at.abs(sigma2_z),
    )
    peak_max_z = at.maximum(
        mu1_z + k_sigma_t * at.abs(sigma1_z),
        mu2_z + k_sigma_t * at.abs(sigma2_z),
    )

    # Include break mb(z)
    band_min_z = at.minimum(peak_min_z, mb_z)
    band_max_z = at.maximum(peak_max_z, mb_z)

    # Now take global min/max over z
    band_min = at.min(band_min_z)
    band_max = at.max(band_max_z)

    # Clip to support and add small margin
    band_min = at.maximum(band_min, m1_low + eps)
    band_max = at.minimum(band_max, m_high - eps)

    # -------------------------------
    # Build three segments:
    #   1) low tail:   [m1_low, band_min)
    #   2) peak band:  [band_min, band_max)
    #   3) high tail:  [band_max, m_high]
    # -------------------------------

    # 1) low tail: [m1_low, band_min), n_tail_low points, endpoint excluded
    if n_tail_low > 0:
        t_low = at.arange(n_tail_low) / n_tail_low
        m1_low_tail = m1_low + (band_min - m1_low) * t_low
    else:
        m1_low_tail = at.zeros((0,))

    # 2) central band: [band_min, band_max), n_peak points, endpoint excluded
    if n_peak > 0:
        t_peak = at.arange(n_peak) / n_peak
        m1_peak_band = band_min + (band_max - band_min) * t_peak
    else:
        m1_peak_band = at.zeros((0,))

    # 3) high tail: [band_max, m_high], n_tail_high points, endpoint included
    if n_tail_high > 1:
        denom_high = n_tail_high - 1
        t_high = at.arange(n_tail_high) / denom_high
    elif n_tail_high == 1:
        t_high = at.zeros((1,))
    else:
        t_high = at.zeros((0,))

    if n_tail_high > 0:
        m1_high_tail = band_max + (m_high - band_max) * t_high
    else:
        m1_high_tail = at.zeros((0,))

    # Concatenate all segments
    m1_grid = at.concatenate([m1_low_tail, m1_peak_band, m1_high_tail], axis=0)

    return m1_grid




def build_m1_grid_DPLDP(
    alpha1, alpha2, mb,
    mu1, sigma1, mu2, sigma2,
    m1_low, m_high,
    delta_m1,
    n_peak=2500,
    n_tail_low=400,
    n_tail_high=400,
    frac_gauss1=0.2,
    frac_gauss2=0.2,
    k_sigma_gauss=3.0,
    k_sigma_band=4.0,
    n_taper=10,
    n_taper_eff=200,
):
    """
    Symbolic non-uniform m1 grid for non-evolving DPLDP.

    Structure:
      - taper:      [m1_low, m1_low+delta_m1] (clustered near m1_low)
      - low tail:   [taper_hi, band_min)
      - Gaussian 1: [mu1 - kσ1, mu1 + kσ1] (with fallback if degenerate)
      - Gaussian 2: [mu2 - kσ2, mu2 + kσ2] (with fallback if degenerate)
      - mid band:   [band_min, band_max] envelope over peaks + mb
      - high tail:  [band_max, m_high)   (endpoint excluded)

    Guarantees:
      - all points inside (m1_low, m_high)
      - strictly increasing (via tiny ramp)
      - avoids repeated xmin/xmax collapse
    """

    # ---- detach hyperparameters for grid geometry (no grad through geometry) ----
    mb_sg       = stop_grad(mb)
    mu1_sg      = stop_grad(mu1)
    sigma1_sg   = stop_grad(sigma1)
    mu2_sg      = stop_grad(mu2)
    sigma2_sg   = stop_grad(sigma2)
    m1_low_sg   = stop_grad(m1_low)
    m_high_sg   = stop_grad(m_high)
    delta_m1_sg = stop_grad(delta_m1)

    # gentle boundary offset (avoid exact endpoints)
    eps = 1e-4
    xmin = m1_low_sg + eps
    xmax = m_high_sg - eps
    span = at.maximum(xmax - xmin, 1e-6)

    # ------------------------------------------------------------
    # 0) Taper grid: clustered near xmin (important for logS_PLP)
    # ------------------------------------------------------------
    taper_hi = at.clip(xmin + at.maximum(delta_m1_sg, 1e-6), xmin, xmax)
    taper_w  = at.maximum(taper_hi - xmin, 1e-6)

    if n_taper > 1:
        eps_t = 1e-4  # smallest fraction of taper width for the first interior point
        u = at.linspace(0.0, 1.0, n_taper)  # [0,1]
        t = at.exp(at.log(eps_t) * (1.0 - u))   # eps_t -> 1
        t = (t - eps_t) / (1.0 - eps_t)         # -> [0,1]
        m1_taper = xmin + taper_w * t
    else:
        m1_taper = at.zeros((0,))

    # ------------------------------------------------------------
    # 1) Gaussian windows (clip to support)
    # ------------------------------------------------------------
    k_g = k_sigma_gauss
    k_b = k_sigma_band

    g1_min_raw = mu1_sg - k_g * at.abs(sigma1_sg)
    g1_max_raw = mu1_sg + k_g * at.abs(sigma1_sg)
    g2_min_raw = mu2_sg - k_g * at.abs(sigma2_sg)
    g2_max_raw = mu2_sg + k_g * at.abs(sigma2_sg)

    g1_min = at.clip(g1_min_raw, xmin, xmax)
    g1_max = at.clip(g1_max_raw, xmin, xmax)
    g2_min = at.clip(g2_min_raw, xmin, xmax)
    g2_max = at.clip(g2_max_raw, xmin, xmax)

    tiny = 1e-6 * span
    g1_width = g1_max - g1_min
    g2_width = g2_max - g2_min

    has_g1 = at.gt(g1_width, tiny)
    has_g2 = at.gt(g2_width, tiny)

    # ------------------------------------------------------------
    # 2) Envelope "interesting band" over peaks + mb
    # ------------------------------------------------------------
    peak_min_raw = at.minimum(g1_min_raw, g2_min_raw)
    peak_min_raw = at.minimum(peak_min_raw, mb_sg)

    peak_max_raw = at.maximum(g1_max_raw, g2_max_raw)
    peak_max_raw = at.maximum(peak_max_raw, mb_sg)

    band_min = at.clip(peak_min_raw, xmin, xmax)
    band_max = at.clip(peak_max_raw, xmin, xmax)

    band_width = at.maximum(band_max - band_min, tiny)

    # ------------------------------------------------------------
    # 3) Split n_peak between Gaussians + mid band
    # ------------------------------------------------------------
    n_g1  = int(n_peak * float(frac_gauss1))
    n_g2  = int(n_peak * float(frac_gauss2))
    if n_g1 < 0: n_g1 = 0
    if n_g2 < 0: n_g2 = 0
    if n_g1 + n_g2 > n_peak:
        scale = float(n_peak) / float(n_g1 + n_g2)
        n_g1 = int(round(n_g1 * scale))
        n_g2 = int(round(n_g2 * scale))
    n_mid = max(n_peak - n_g1 - n_g2, 0)

    # ------------------------------------------------------------
    # 4) Low tail: start AFTER taper, keep fixed length
    # ------------------------------------------------------------
    if n_tail_low > 0:
        denom_low = float(n_tail_low + 1)
        t_low = (at.arange(n_tail_low) + 1.0) / denom_low  # in (0,1)

        low_start = taper_hi
        low_width = band_min - low_start

        fallback_w = at.maximum(taper_w / at.maximum(n_taper, 1), 1e-3)

        tail_good = low_start + low_width * t_low
        tail_fallback = low_start + fallback_w * t_low

        m1_low_tail = at.switch(at.gt(low_width, 0), tail_good, tail_fallback)
    else:
        m1_low_tail = at.zeros((0,))

    # ------------------------------------------------------------
    # 5) Gaussian 1 segment (with fallback window if degenerate)
    # ------------------------------------------------------------
    if n_g1 > 0:
        denom_g1 = float(max(n_g1 - 1, 1))
        t_g1 = at.arange(n_g1) / denom_g1

        m1_g1 = g1_min + g1_width * t_g1

        fallback_width = 1e-8 * span
        g1_center = 0.5 * (g1_min + g1_max)
        g1_center = at.clip(g1_center, xmin + fallback_width, xmax - fallback_width)
        fallback_g1 = g1_center + fallback_width * (t_g1 - 0.5)

        m1_g1 = at.switch(has_g1, m1_g1, fallback_g1)
    else:
        m1_g1 = at.zeros((0,))

    # ------------------------------------------------------------
    # 6) Gaussian 2 segment (with fallback window if degenerate)
    # ------------------------------------------------------------
    if n_g2 > 0:
        denom_g2 = float(max(n_g2 - 1, 1))
        t_g2 = at.arange(n_g2) / denom_g2

        m1_g2 = g2_min + g2_width * t_g2

        fallback_width = 1e-8 * span
        g2_center = 0.5 * (g2_min + g2_max)
        g2_center = at.clip(g2_center, xmin + fallback_width, xmax - fallback_width)
        fallback_g2 = g2_center + fallback_width * (t_g2 - 0.5)

        m1_g2 = at.switch(has_g2, m1_g2, fallback_g2)
    else:
        m1_g2 = at.zeros((0,))

    # ------------------------------------------------------------
    # 7) Mid band segment
    # ------------------------------------------------------------
    if n_mid > 0:
        denom_mid = float(max(n_mid - 1, 1))
        t_mid = at.arange(n_mid) / denom_mid
        m1_mid = band_min + band_width * t_mid
    else:
        m1_mid = at.zeros((0,))

    # ------------------------------------------------------------
    # 8) High tail: endpoint excluded (avoid exact xmax)
    # ------------------------------------------------------------
    if n_tail_high > 0:
        denom_high = float(max(n_tail_high, 1))   # not (n_tail_high-1)
        t_high = at.arange(n_tail_high) / denom_high  # in [0,1)
        m1_high_tail = band_max + (xmax - band_max) * t_high
    else:
        m1_high_tail = at.zeros((0,))

    # ------------------------------------------------------------
    # Combine -> clip -> sort -> tiny ramp for strict monotonicity
    # ------------------------------------------------------------
    m1_grid_raw = at.concatenate(
        [m1_taper, m1_low_tail, m1_g1, m1_g2, m1_mid, m1_high_tail],
        axis=0,
    )

    m1_grid_clipped = at.clip(m1_grid_raw, xmin, xmax)
    m1_grid_sorted = at.sort(m1_grid_clipped)

    # tiny ramp ensures strict increase (does not affect resolution)
    ramp_step = 1e-6
    ramp = ramp_step * at.arange(m1_grid_sorted.shape[0], dtype=m1_grid_sorted.dtype)
    #m1_grid_strict = m1_grid_sorted + ramp
    m1_grid_strict = at.clip(m1_grid_sorted + ramp, xmin, xmax)
    return m1_grid_strict


def build_m1_grid_DPLDP_01(
    alpha1, alpha2, mb,
    mu1, sigma1, mu2, sigma2,
    m1_low, m_high,
    delta_m1,
    n_peak=2500,
    n_tail_low=400,
    n_tail_high=400,
    frac_gauss1=0.2,
    frac_gauss2=0.2,
    k_sigma_gauss=3.0,
    k_sigma_band=4.0,
    n_taper=10,          # NEW: points inside [m1_low, m1_low+delta_m1]
    n_taper_eff=200.0,   # NEW: used for tie-only ramp scale
):
    # ---- detach hyperparameters for grid construction ----
    mb_sg       = stop_grad(mb)
    mu1_sg      = stop_grad(mu1)
    sigma1_sg   = stop_grad(sigma1)
    mu2_sg      = stop_grad(mu2)
    sigma2_sg   = stop_grad(sigma2)
    m1_low_sg   = stop_grad(m1_low)
    m_high_sg   = stop_grad(m_high)
    delta_m1_sg = stop_grad(delta_m1)

    eps = 1e-4
    xmin = m1_low_sg + eps
    xmax = m_high_sg - eps
    span = at.maximum(xmax - xmin, 1e-6)

    # ---------------------------------------------------------
    # (0) Explicit taper grid: [xmin, xmin + delta_m1]
    # ---------------------------------------------------------
    taper_hi = at.clip(xmin + at.maximum(delta_m1_sg, 1e-6), xmin, xmax)
    taper_w  = at.maximum(taper_hi - xmin, 1e-6)

    if n_taper > 0:
        denom_t = float(max(n_taper - 1, 1))
        t_taper = at.arange(n_taper) / denom_t
        m1_taper = xmin + taper_w * t_taper
    else:
        m1_taper = at.zeros((0,))

    # ---- Gaussian windows ----
    k_g = k_sigma_gauss
    k_b = k_sigma_band

    g1_min_raw = mu1_sg - k_g * at.abs(sigma1_sg)
    g1_max_raw = mu1_sg + k_g * at.abs(sigma1_sg)
    g2_min_raw = mu2_sg - k_g * at.abs(sigma2_sg)
    g2_max_raw = mu2_sg + k_g * at.abs(sigma2_sg)

    g1_min = at.clip(g1_min_raw, xmin, xmax)
    g1_max = at.clip(g1_max_raw, xmin, xmax)
    g2_min = at.clip(g2_min_raw, xmin, xmax)
    g2_max = at.clip(g2_max_raw, xmin, xmax)

    tiny = 1e-6 * span
    g1_width = g1_max - g1_min
    g2_width = g2_max - g2_min
    has_g1 = at.gt(g1_width, tiny)
    has_g2 = at.gt(g2_width, tiny)

    # ---- envelope band over both Gaussians + mb ----
    peak_min_raw = at.minimum(at.minimum(g1_min_raw, g2_min_raw), mb_sg)
    peak_max_raw = at.maximum(at.maximum(g1_max_raw, g2_max_raw), mb_sg)

    band_min = at.clip(peak_min_raw, xmin, xmax)
    band_max = at.clip(peak_max_raw, xmin, xmax)
    band_width = at.maximum(band_max - band_min, tiny)

    # ---- split n_peak between Gaussians and mid band ----
    n_g1  = int(n_peak * float(frac_gauss1))
    n_g2  = int(n_peak * float(frac_gauss2))
    if n_g1 < 0: n_g1 = 0
    if n_g2 < 0: n_g2 = 0
    if n_g1 + n_g2 > n_peak:
        scale = float(n_peak) / float(n_g1 + n_g2)
        n_g1 = int(round(n_g1 * scale))
        n_g2 = int(round(n_g2 * scale))
    n_mid = max(n_peak - n_g1 - n_g2, 0)

    # ---------------------------------------------------------
    # (1) Low tail: [xmin, band_min)
    # IMPORTANT: start slightly above xmin so we don't duplicate taper points
    # ---------------------------------------------------------
    if n_tail_low > 0:
        denom_low = float(max(n_tail_low, 1))
        # start at 1/denom_low (exclude exact xmin)
        t_low = (at.arange(n_tail_low) + 1.0) / denom_low
        m1_low_tail = xmin + (band_min - xmin) * t_low
    else:
        m1_low_tail = at.zeros((0,))

    # ---------------------------------------------------------
    # (2) Gaussian 1 segment with safe fallback near its center
    # ---------------------------------------------------------
    if n_g1 > 0:
        denom_g1 = float(max(n_g1 - 1, 1))
        t_g1 = at.arange(n_g1) / denom_g1
        m1_g1 = g1_min + g1_width * t_g1

        fallback_width = 1e-8 * span
        g1_center = 0.5 * (g1_min + g1_max)
        g1_center = at.clip(g1_center, xmin + fallback_width, xmax - fallback_width)
        fallback_g1 = g1_center + fallback_width * (t_g1 - 0.5)

        m1_g1 = at.switch(has_g1, m1_g1, fallback_g1)
    else:
        m1_g1 = at.zeros((0,))

    # ---------------------------------------------------------
    # (3) Gaussian 2 segment with safe fallback near its center
    # ---------------------------------------------------------
    if n_g2 > 0:
        denom_g2 = float(max(n_g2 - 1, 1))
        t_g2 = at.arange(n_g2) / denom_g2
        m1_g2 = g2_min + g2_width * t_g2

        fallback_width = 1e-8 * span
        g2_center = 0.5 * (g2_min + g2_max)
        g2_center = at.clip(g2_center, xmin + fallback_width, xmax - fallback_width)
        fallback_g2 = g2_center + fallback_width * (t_g2 - 0.5)

        m1_g2 = at.switch(has_g2, m1_g2, fallback_g2)
    else:
        m1_g2 = at.zeros((0,))

    # ---------------------------------------------------------
    # (4) Mid band: [band_min, band_max]
    # ---------------------------------------------------------
    if n_mid > 0:
        denom_mid = float(max(n_mid - 1, 1))
        t_mid = at.arange(n_mid) / denom_mid
        m1_mid = band_min + band_width * t_mid
    else:
        m1_mid = at.zeros((0,))

    # ---------------------------------------------------------
    # (5) High tail: [band_max, xmax]
    # ---------------------------------------------------------
    if n_tail_high > 0:
        denom_high = float(max(n_tail_high - 1, 1))
        t_high = at.arange(n_tail_high) / denom_high
        m1_high_tail = band_max + (xmax - band_max) * t_high
    else:
        m1_high_tail = at.zeros((0,))

    # ---- combine and clip ----
    m1_grid_raw = at.concatenate(
        [m1_taper, m1_low_tail, m1_g1, m1_g2, m1_mid, m1_high_tail],
        axis=0,
    )
    m1_grid_clipped = at.clip(m1_grid_raw, xmin, xmax)
    m1_grid_sorted = at.sort(m1_grid_clipped)

    # ---------------------------------------------------------
    # Remove repeats WITHOUT scan: tie-only ramp
    # (this spreads only the duplicates; doesn’t shift the whole grid)
    # ---------------------------------------------------------
    dx = at.diff(m1_grid_sorted)
    ties = at.le(dx, 0)

    min_step = at.maximum(delta_m1_sg / n_taper_eff, 1e-6)

    tie_count = at.concatenate(
        [at.zeros((1,), dtype=m1_grid_sorted.dtype),
         at.cumsum(ties).astype(m1_grid_sorted.dtype)]
    )

    m1_grid_strict = m1_grid_sorted + min_step * tie_count
    m1_grid_strict = at.clip(m1_grid_strict, xmin, xmax)

    return m1_grid_strict



def build_m1_grid_DPLDP_old(
    alpha1, alpha2, mb,
    mu1, sigma1, mu2, sigma2,
    m1_low, m_high,
    delta_m1,
    n_peak=2500,
    n_tail_low=400,
    n_tail_high=400,
    frac_gauss1=0.2,   # fraction of n_peak for Gaussian 1
    frac_gauss2=0.2,   # fraction of n_peak for Gaussian 2
    k_sigma_gauss=3.0, # ±kσ around each Gaussian
    k_sigma_band=4.0,  # envelope band around both Gaussians + mb
):
    """
    Adaptive non-uniform m1 grid for non-evolving DPLDP.

    Structure:
      - low tail:   [m1_low, band_min)
      - Gaussian 1: [mu1 - kσ1, mu1 + kσ1]
      - Gaussian 2: [mu2 - kσ2, mu2 + kσ2]
      - mid band:   [band_min, band_max] (envelope over both peaks + mb)
      - high tail:  [band_max, m_high]

    n_peak is split into:
      n_g1  = frac_gauss1 * n_peak  points for Gaussian 1
      n_g2  = frac_gauss2 * n_peak  points for Gaussian 2
      n_mid = remaining points in the envelope band

    Guarantees:
      - all points in (m1_low, m_high),
      - grid is sorted and deduplicated,
      - no aggressive low cut.
    """

    # ---- detach hyperparameters for grid construction (no gradient through geometry) ----
    mb_sg       = stop_grad(mb)
    mu1_sg      = stop_grad(mu1)
    sigma1_sg   = stop_grad(sigma1)
    mu2_sg      = stop_grad(mu2)
    sigma2_sg   = stop_grad(sigma2)
    m1_low_sg   = stop_grad(m1_low)
    m_high_sg   = stop_grad(m_high)
    delta_m1_sg = stop_grad(delta_m1)

    # dtype
    #dtype = getattr(getattr(m1_low_sg, "dtype", None) or m_high_sg.dtype, "lower", lambda: "float64")()

    # *** MUCH GENTLER EPS ***
    # tiny fixed offset just to avoid exactly hitting the boundaries
    eps = 1e-04 #at.as_tensor_variable(1e-4, dtype=dtype)

    # global support, with minimal safety margins
    xmin = m1_low_sg + eps
    xmax = m_high_sg - eps
    span = at.maximum(xmax - xmin, 1e-06) #at.as_tensor_variable(1e-6, dtype=dtype))

    # ---- Gaussian windows ----
    k_g = k_sigma_gauss #at.as_tensor_variable(k_sigma_gauss, dtype=dtype)
    k_b = k_sigma_band #at.as_tensor_variable(k_sigma_band,  dtype=dtype)

    # raw gaussian windows (before clipping)
    g1_min_raw = mu1_sg - k_g * at.abs(sigma1_sg)
    g1_max_raw = mu1_sg + k_g * at.abs(sigma1_sg)

    g2_min_raw = mu2_sg - k_g * at.abs(sigma2_sg)
    g2_max_raw = mu2_sg + k_g * at.abs(sigma2_sg)

    # clip to [xmin, xmax]
    g1_min = at.clip(g1_min_raw, xmin, xmax)
    g1_max = at.clip(g1_max_raw, xmin, xmax)
    g2_min = at.clip(g2_min_raw, xmin, xmax)
    g2_max = at.clip(g2_max_raw, xmin, xmax)

    tiny = 1e-6 * span
    g1_width = g1_max - g1_min
    g2_width = g2_max - g2_min

    has_g1 = at.gt(g1_width, tiny)
    has_g2 = at.gt(g2_width, tiny)

    # ---- envelope band over both Gaussians + break mb ----
    peak_min_raw = at.minimum(g1_min_raw, g2_min_raw)
    peak_min_raw = at.minimum(peak_min_raw, mb_sg)

    peak_max_raw = at.maximum(g1_max_raw, g2_max_raw)
    peak_max_raw = at.maximum(peak_max_raw, mb_sg)

    band_min = at.clip(peak_min_raw, xmin, xmax)
    band_max = at.clip(peak_max_raw, xmin, xmax)

    band_width = at.maximum(band_max - band_min, tiny)

    # ---- split n_peak between Gaussians + band (Python ints) ----
    n_g1  = int(n_peak * float(frac_gauss1))
    n_g2  = int(n_peak * float(frac_gauss2))
    if n_g1 < 0: n_g1 = 0
    if n_g2 < 0: n_g2 = 0
    if n_g1 + n_g2 > n_peak:
        scale = float(n_peak) / float(n_g1 + n_g2)
        n_g1 = int(round(n_g1 * scale))
        n_g2 = int(round(n_g2 * scale))
    n_mid = max(n_peak - n_g1 - n_g2, 0)

    # 1) low tail: [xmin, band_min)
    if n_tail_low > 0:
        denom_low = float(max(n_tail_low, 1))
        t_low = at.arange(n_tail_low) / denom_low
        m1_low_tail = xmin + (band_min - xmin) * t_low
    else:
        m1_low_tail = at.zeros((0,))

    # 2) Gaussian 1: [g1_min, g1_max]
    if n_g1 > 0:
        if n_g1 > 1:
            denom_g1 = float(n_g1 - 1)
        else:
            denom_g1 = 1.0
        t_g1 = at.arange(n_g1) / denom_g1
        m1_g1 = g1_min + g1_width * t_g1
        m1_g1 = at.switch(has_g1, m1_g1, at.zeros_like(m1_g1))
    else:
        m1_g1 = at.zeros((0,))

    # 3) Gaussian 2: [g2_min, g2_max]
    if n_g2 > 0:
        if n_g2 > 1:
            denom_g2 = float(n_g2 - 1)
        else:
            denom_g2 = 1.0
        t_g2 = at.arange(n_g2) / denom_g2
        m1_g2 = g2_min + g2_width * t_g2
        m1_g2 = at.switch(has_g2, m1_g2, at.zeros_like(m1_g2))
    else:
        m1_g2 = at.zeros((0,))

    # 4) mid band: [band_min, band_max]
    if n_mid > 0:
        if n_mid > 1:
            denom_mid = float(n_mid - 1)
        else:
            denom_mid = 1.0
        t_mid = at.arange(n_mid) / denom_mid
        m1_mid = band_min + band_width * t_mid
    else:
        m1_mid = at.zeros((0,))

    # 5) high tail: [band_max, xmax]
    if n_tail_high > 0:
        if n_tail_high > 1:
            denom_high = float(n_tail_high - 1)
        else:
            denom_high = 1.0
        t_high = at.arange(n_tail_high) / denom_high
        m1_high_tail = band_max + (xmax - band_max) * t_high
    else:
        m1_high_tail = at.zeros((0,))

    # ---- combine, clip, sort, deduplicate ----
    m1_grid_raw = at.concatenate(
        [m1_low_tail, m1_g1, m1_g2, m1_mid, m1_high_tail],
        axis=0,
    )

    # just in case anything slipped slightly out of bounds
    m1_grid_clipped = at.clip(m1_grid_raw, xmin, xmax)

    # sort & remove duplicates
    m1_grid_sorted = at.sort(m1_grid_clipped)
    #m1_grid_unique = stop_grad(at.unique(m1_grid_sorted))

    return m1_grid_sorted



def build_m1_grid_PLPreg(
    ml, mh,
    muMass, sigmaMass,
    deltam,
    n_peak=2500,
    n_tail_low=400,
    n_tail_high=400,
    frac_gauss=0.4,
    k_sigma_gauss=3.0,
    k_sigma_band=4.0,
    n_taper=10,
):
    """
    Symbolic non-uniform m1 grid for PLPreg (power-law + single Gaussian peak).

    Structure:
      - taper:      [ml, ml + deltam] (log-clustered near ml)
      - low tail:   [taper_hi, band_min)  fixed-length, with fallback if empty
      - Gaussian:   [mu - kσ, mu + kσ]    with fallback if degenerate
      - mid band:   [band_min, band_max]
      - high tail:  [band_max, mh)        endpoint excluded
    Guarantees:
      - inside (ml, mh)
      - strictly increasing (tiny ramp)
      - fixed shapes (compiles)
    """

    # detach geometry params (no grad through grid)
    ml_s = stop_grad(ml)
    mh_s = stop_grad(mh)
    mu_s = stop_grad(muMass)
    sig_s = stop_grad(sigmaMass)
    deltam_s = stop_grad(deltam)

    # gentle boundary offset
    eps = 1e-4
    xmin = ml_s + eps
    xmax = mh_s - eps
    span = at.maximum(xmax - xmin, 1e-6)

    # ------------------------------------------------------------
    # 0) Taper grid near xmin (log clustered)
    # ------------------------------------------------------------
    taper_hi = at.clip(xmin + at.maximum(deltam_s, 1e-6), xmin, xmax)
    taper_w  = at.maximum(taper_hi - xmin, 1e-6)

    if n_taper > 1:
        eps_t = 1e-4
        u = at.linspace(0.0, 1.0, n_taper)
        t = at.exp(at.log(eps_t) * (1.0 - u))     # eps_t -> 1
        t = (t - eps_t) / (1.0 - eps_t)           # -> [0,1]
        m1_taper = xmin + taper_w * t
    else:
        m1_taper = at.zeros((0,))

    # ------------------------------------------------------------
    # 1) Gaussian window and band window (clipped)
    # ------------------------------------------------------------
    k_g = k_sigma_gauss
    k_b = k_sigma_band

    g_min_raw = mu_s - k_g * at.abs(sig_s)
    g_max_raw = mu_s + k_g * at.abs(sig_s)

    g_min = at.clip(g_min_raw, xmin, xmax)
    g_max = at.clip(g_max_raw, xmin, xmax)

    band_min_raw = mu_s - k_b * at.abs(sig_s)
    band_max_raw = mu_s + k_b * at.abs(sig_s)

    band_min = at.clip(band_min_raw, xmin, xmax)
    band_max = at.clip(band_max_raw, xmin, xmax)

    tiny = 1e-6 * span

    g_width = g_max - g_min
    has_g = at.gt(g_width, tiny)

    band_width = at.maximum(band_max - band_min, tiny)

    # ------------------------------------------------------------
    # 2) Split peak budget
    # ------------------------------------------------------------
    n_g = int(n_peak * float(frac_gauss))
    n_g = max(0, min(n_g, n_peak))
    n_mid = max(n_peak - n_g, 0)

    # ------------------------------------------------------------
    # 3) Low tail AFTER taper (fixed length, fallback if empty)
    # ------------------------------------------------------------
    if n_tail_low > 0:
        denom_low = float(n_tail_low + 1)
        t_low = (at.arange(n_tail_low) + 1.0) / denom_low  # (0,1)

        low_start = taper_hi
        low_width = band_min - low_start

        # fallback width comparable to taper resolution
        fallback_w = at.maximum(taper_w / at.maximum(n_taper, 1), 1e-3)

        tail_good = low_start + low_width * t_low
        tail_fallback = low_start + fallback_w * t_low

        m1_low_tail = at.switch(at.gt(low_width, 0), tail_good, tail_fallback)
    else:
        m1_low_tail = at.zeros((0,))

    # ------------------------------------------------------------
    # 4) Gaussian segment (fallback if degenerate)
    # ------------------------------------------------------------
    if n_g > 0:
        denom_g = float(max(n_g - 1, 1))
        t_g = at.arange(n_g) / denom_g
        m1_g = g_min + g_width * t_g

        fallback_width = 1e-8 * span
        g_center = 0.5 * (g_min + g_max)
        g_center = at.clip(g_center, xmin + fallback_width, xmax - fallback_width)
        fallback_g = g_center + fallback_width * (t_g - 0.5)

        m1_g = at.switch(has_g, m1_g, fallback_g)
    else:
        m1_g = at.zeros((0,))

    # ------------------------------------------------------------
    # 5) Mid band
    # ------------------------------------------------------------
    if n_mid > 0:
        denom_mid = float(max(n_mid - 1, 1))
        t_mid = at.arange(n_mid) / denom_mid
        m1_mid = band_min + band_width * t_mid
    else:
        m1_mid = at.zeros((0,))

    # ------------------------------------------------------------
    # 6) High tail (exclude xmax)
    # ------------------------------------------------------------
    if n_tail_high > 0:
        denom_high = float(max(n_tail_high, 1))
        t_high = at.arange(n_tail_high) / denom_high  # [0,1)
        m1_high_tail = band_max + (xmax - band_max) * t_high
    else:
        m1_high_tail = at.zeros((0,))

    # ------------------------------------------------------------
    # Combine -> clip -> sort -> tiny ramp
    # ------------------------------------------------------------
    m1_grid_raw = at.concatenate(
        [m1_taper, m1_low_tail, m1_g, m1_mid, m1_high_tail],
        axis=0
    )

    m1_grid_clipped = at.clip(m1_grid_raw, xmin, xmax)
    m1_grid_sorted = at.sort(m1_grid_clipped)

    ramp_step = 1e-6
    ramp = ramp_step * at.arange(m1_grid_sorted.shape[0], dtype=m1_grid_sorted.dtype)
    m1_grid_strict = m1_grid_sorted + ramp

    return m1_grid_strict

    

def build_m1_grid_PLPreg_0(
    ml, mh,
    muMass, sigmaMass,
    n_peak=2500,
    n_tail_low=400,
    n_tail_high=400,
    frac_gauss=0.4,     # fraction of n_peak for the Gaussian window
    k_sigma_gauss=3.0,  # ±kσ around the Gaussian peak
    k_sigma_band=4.0,   # envelope band around the peak
):
    """
    Adaptive non-uniform m1 grid for the PLPreg (power-law + single Gaussian peak) mass model.

    Structure:
      - low tail:   [ml, band_min)
      - Gaussian:   [muMass - kσ, muMass + kσ]
      - mid band:   [band_min, band_max] (envelope over the peak)
      - high tail:  [band_max, mh]

    n_peak is split into:
      n_g   = frac_gauss * n_peak  points for the Gaussian window
      n_mid = remaining points in the envelope band
    """

    # ---- detach hyperparameters for grid construction (no gradient through geometry) ----
    ml_sg    = stop_grad(ml)
    mh_sg    = stop_grad(mh)
    mu_sg    = stop_grad(muMass)
    sigma_sg = stop_grad(sigmaMass)

    # dtype similar to build_m1_grid_DPLDP
    # dtype = getattr(
    #     getattr(ml_sg, "dtype", None) or mh_sg.dtype,
    #     "lower",
    #     lambda: "float64",
    # )()

    # small offset to avoid exactly hitting boundaries
    eps = 1e-04 #at.as_tensor_variable(1e-4, dtype=dtype)

    xmin = ml_sg + eps
    xmax = mh_sg - eps
    span = at.maximum(xmax - xmin, 1e-06) #at.as_tensor_variable(1e-6, dtype=dtype))

    # ---- Gaussian window around the peak ----
    k_g = k_sigma_gauss #at.as_tensor_variable(k_sigma_gauss, dtype=dtype)
    k_b = k_sigma_band #at.as_tensor_variable(k_sigma_band,  dtype=dtype)

    g_min_raw = mu_sg - k_g * at.abs(sigma_sg)
    g_max_raw = mu_sg + k_g * at.abs(sigma_sg)

    g_min = at.clip(g_min_raw, xmin, xmax)
    g_max = at.clip(g_max_raw, xmin, xmax)

    tiny   = 1e-6 * span
    g_width = g_max - g_min
    has_g   = at.gt(g_width, tiny)

    # envelope band for mid region
    band_min_raw = mu_sg - k_b * at.abs(sigma_sg)
    band_max_raw = mu_sg + k_b * at.abs(sigma_sg)

    band_min = at.clip(band_min_raw, xmin, xmax)
    band_max = at.clip(band_max_raw, xmin, xmax)

    band_width = at.maximum(band_max - band_min, tiny)

    # ---- split n_peak between Gaussian window and band ----
    n_g = int(n_peak * float(frac_gauss))
    n_g = max(0, min(n_g, n_peak))
    n_mid = max(n_peak - n_g, 0)

    # 1) low tail: [xmin, band_min)
    if n_tail_low > 0:
        denom_low = float(max(n_tail_low, 1))
        t_low = at.arange(n_tail_low) / denom_low
        m1_low_tail = xmin + (band_min - xmin) * t_low
    else:
        m1_low_tail = at.zeros((0,))

    # 2) Gaussian window: [g_min, g_max]
    if n_g > 0:
        denom_g = float(max(n_g - 1, 1))
        t_g = at.arange(n_g) / denom_g
        m1_g = g_min + g_width * t_g
        m1_g = at.switch(has_g, m1_g, at.zeros_like(m1_g))
    else:
        m1_g = at.zeros((0,))

    # 3) mid band: [band_min, band_max]
    if n_mid > 0:
        denom_mid = float(max(n_mid - 1, 1))
        t_mid = at.arange(n_mid) / denom_mid
        m1_mid = band_min + band_width * t_mid
    else:
        m1_mid = at.zeros((0,))

    # 4) high tail: [band_max, xmax]
    if n_tail_high > 0:
        denom_high = float(max(n_tail_high - 1, 1))
        t_high = at.arange(n_tail_high) / denom_high
        m1_high_tail = band_max + (xmax - band_max) * t_high
    else:
        m1_high_tail = at.zeros((0,))

    # ---- combine and sort ----
    m1_grid_raw = at.concatenate([m1_low_tail, m1_g, m1_mid, m1_high_tail], axis=0)
    m1_grid_clipped = at.clip(m1_grid_raw, xmin, xmax)
    m1_grid_sorted = at.sort(m1_grid_clipped)

    return m1_grid_sorted

    
    
#####################################
class GridInterpolator_at:
    '''
    points :: n x n tensor with points where to evaluate the grid
    grid :: the grid on which we want to interpolate, size m x m
    values :: values of function to interpolate evaluated at grid
    '''
    def __init__(self, grid, values, verbose=False):
        self.grid = grid # tuple of len(2) first element is first variable, second element second variale
        self.values = values
        self.verbose = verbose
    def __call__(self, points, verbose=None):
        indices, norm_distances, out_of_bounds = self._find_indices(points, verbose=self.verbose)
        result = self._evaluate_linear(indices, norm_distances, out_of_bounds, verbose=self.verbose)
        return result
    def _i_nd_from_xi_grid(self, x, g):
        i = at.searchsorted(g, x) - 1
        i = at.where(i < 0, 0, i)
        i = at.where(i > g.size - 2, g.size - 2, i)
        return i, (x - g[i]) / (g[i + 1] - g[i])
    def _find_indices(self, xi, verbose=False):
        indices = []
        norm_distances = []
        out_of_bounds = at.zeros((xi.shape[1]), dtype=bool)
         # iterate through dimensions
        indices=[]
        norm_distances=[]
        for ix in range(len(self.grid)):
            idx_, nd_ = self._i_nd_from_xi_grid( xi[ix], self.grid[ix])
            indices.append(idx_)
            norm_distances.append(nd_)
        return indices, norm_distances, out_of_bounds
    def _evaluate_linear(self, indices, norm_distances, out_of_bounds, verbose=False):
        from itertools import product
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))
        if verbose:
            print('indices in eval_lin')
            print(len(indices))
            print(indices[0].eval())
            print(indices[1].eval())
        # find relevant values
        # each i and i+1 represents a edge
        edges = product(*[[i, i + 1] for i in indices])
        values = at.as_tensor_variable(0.)
        for edge_indices in edges:
            weight = at.as_tensor_variable(1.)
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight = weight*at.switch(at.eq(ei,i), 1 - yi, yi)
            values = values+ self.values[edge_indices[1], edge_indices[0]]* weight[vslice]
        return values

def Pdet( osnr_interp_at, m1det, m2det, dL, Theta, thresh,
         sigma=1., rand_noise=False, rng = None, seed = 42, ref_dist_Gpc_at=1.):
    """
    file: mass grid and related optimal snr values
    m1det: detector-frame mass 1
    m2det: detector-frame mass 2
    dL: luminosity distance in Gpc
    Theta: combination of anterna pattern functions
    osnr_interp: 2D interpolant
    ref_dist_Gpc: the reference distance at which osnr_interp was calculated
    rand_noise: add random N(0,1)
    """
    
    # evaluate the interpolant
    pts_at = at.stack([m1det, at.full_like(m1det, m2det)], axis=0)
    osnr_at = osnr_interp_at(pts_at)
    #print(osnr_at.eval().shape)
    # add noise (eventually)
    if rand_noise:
        rng = np.random.default_rng(seed)
        rs = rng.normal(0.0, 1.0, size = m1det.eval().shape)
    else:
        rs = 0.0
    rs_at = at.as_tensor_variable(rs)
    # compute true snr
    snr_at = osnr_at * ref_dist_Gpc_at / dL * Theta + rs_at
    #result = 0.5 * (1 + at.erf((snr_at - thresh) / sigma))
    result = 0.5 * (1 + at.erf((snr_at - thresh)))
    return result

