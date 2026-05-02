import pade_cosmo as pc
import numpy as np
import jax.numpy as jnp

# ---------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------


c_light = 299792458*1e-03
_PI = np.pi

PlanckFiducials = {'H0': 67.66, 'Om':0.31, 'w0':-1, 'Xi0': 1, 'nXi0':0}

PLANCK15_H0 = 67.9
PLANCK15_OM = 0.3065



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

_x01_np, _w01_np = gauss_legendre_01(n=64)  # 16–64 usually plenty




# ---------------------------------------------------------------------
# Coefficients for pade approxiamtion of dc
# ---------------------------------------------------------------------


p, q = pc.flat_wcdm_pade_coefficients(w0=-1.0, zpower=0)  # arrays of floats


# ---------------------------------------------------------------------
# Shared interpolation grids
# ---------------------------------------------------------------------


max_m = 500.0

_tgrid_np = np.linspace(0.0, 1.0, 500).astype("float64")

def _get_t_grid():
    return _tgrid_at



z_nodes_jax = jnp.logspace( jnp.log10(1e-05), jnp.log10(100), 1200)

z_nodes_np = np.logspace( np.log10(1e-05), np.log10(100), 1200)

