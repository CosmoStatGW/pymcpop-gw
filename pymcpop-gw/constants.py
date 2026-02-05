import pytensor.tensor as at
import pade_cosmo as pc
import numpy as np


# ---------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------


c_light = 299792458*1e-03
_PI = np.pi


# ---------------------------------------------------------------------
# Gauss–Legendre nodes for integration
# ---------------------------------------------------------------------


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

_x01_at = x01_at
_w01_at = w01_at

# ---------------------------------------------------------------------
# Coefficients for pade approxiamtion of dc
# ---------------------------------------------------------------------


p, q = pc.flat_wcdm_pade_coefficients(w0=-1.0, zpower=0)  # arrays of floats


# ---------------------------------------------------------------------
# Shared interpolation grids
# ---------------------------------------------------------------------


max_m = 500.0

_tgrid_np = np.linspace(0.0, 1.0, 500).astype("float64")
_tgrid_at = at.as_tensor_variable(_tgrid_np)  # if you really want stop_grad, do it where you build graphs

def _get_t_grid():
    return _tgrid_at