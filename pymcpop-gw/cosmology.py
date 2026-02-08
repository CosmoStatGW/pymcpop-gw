from __future__ import annotations

from pytensor_utils import attrapzvec
from constants import _PI, c_light

# ---------------------------------------------------------------------
# Hubble
# ---------------------------------------------------------------------

def Efun(bk, z, Om, w0):
    """
    Dimensionless Hubble parameter E(z).

    E(z) = sqrt( Om (1+z)^3 + (1-Om) (1+z)^{3(1+w0)} )
    """
    ainv = 1.0 + z
    return bk.sqrt(Om * ainv**3 + (1.0 - Om) * ainv ** (3.0 * (1.0 + w0)))


# ---------------------------------------------------------------------
# modified GW propagation factor
# ---------------------------------------------------------------------


def Xi_vanilla(bk, z, Xi0, n):
    """Legacy 'vanilla' Xi(z) parameterization."""
    return Xi0 + (1.0 - Xi0) / (1.0 + z) ** n


def Xi_polexp(bk, z, Xi0, n):
    r"""
    Ξ(z) = exp(  -(1-Ξ0)[1 - (1+z)^n] / (1+z)^{2n}  )
           * [ Ξ0 + (1-Ξ0)(1+z)^{-n} ]
    """
    onepz = 1.0 + z
    exponent = -(1.0 - Xi0) * (1.0 - onepz**n) / (onepz ** (2.0 * n))
    pref = Xi0 + (1.0 - Xi0) * onepz ** (-n)
    return bk.exp(exponent) * pref


# ---------------------------------------------------------------------
# comoving distance by integration
# ---------------------------------------------------------------------


def dcfun_quad(bk, z, H0, Om, w0, x01, w01):
    """
    Comoving distance d_c(z) in Gpc, using Gauss–Legendre quadrature on [0,z].

    Inputs:
      z: array-like (...,)
      x01, w01: nodes/weights on [0,1] (shape (n,))
      H0: km/s/Mpc

    Returns:
      d_c(z) in Gpc
    """
    
    z_nodes = z[..., None] * x01
    
    integrand = 1.0 / Efun(bk, z_nodes, Om, w0)
    
    I = bk.sum(w01 * integrand, axis=-1)
    
    return (c_light / H0) * z * I * 1e-03

# ---------------------------------------------------------------------
# luminosity distance
# ---------------------------------------------------------------------

def dLfun(bk, z, H0, Om, w0, Xi0, nXi0, *, x01, w01, dc=None, Xi=None, param="vanilla"):
    """
    Luminosity distance d_L(z) in Gpc.

    Notes:
      - c_light is a module-level constant (imported from constants.py)
      - If dc is None, this computes dc via dcfun_quad (needs x01,w01).
    """
    if Xi is None:
        if param == "vanilla":
            Xi = Xi_vanilla(bk, z, Xi0, nXi0)
        elif param == "polexp":
            Xi = Xi_polexp(bk, z, Xi0, nXi0)
        else:
            raise ValueError(f"Unknown param='{param}' (expected 'vanilla' or 'polexp')")

    if dc is None:
        if x01 is None or w01 is None:
            raise ValueError("dLfun: dc is None, so you must pass x01 and w01 to compute dc via dcfun_quad.")
        dc = dcfun_quad(bk, z, H0, Om, w0, x01, w01)

    return Xi * (1.0 + z) * dc


# ---------------------------------------------------------------------
# log(dV/dz)
# ---------------------------------------------------------------------
def log_dV_dz(bk, z, H0, Om0, w0, *, dc=None, E=None, x01=None, w01=None):
    """
    Backend-agnostic log(dV/dz).

    If dc is not provided, it will be computed via dcfun_quad (needs x01,w01).

    log(dV/dz) =
      log(4π) + log(c) - log(H0) + 2log(dc) - log(E(z)) - 3log(10)
    """
    if dc is None:
        if x01 is None or w01 is None:
            raise ValueError("log_dV_dz: dc is None, so you must pass x01 and w01 to compute dc via dcfun_quad.")
        dc = dcfun_quad(bk, z, H0, Om0, w0, x01, w01)

    if E is None:
        E = Efun(bk, z, Om0, w0)
        
    return (
        bk.log(4.0 * _PI)
        + bk.log(c_light)
        - bk.log(H0)
        + 2.0 * bk.log(dc)
        - bk.log(E)
        - 3.0 * bk.log(10.0)
    )


# ---------------------------------------------------------------------
# log(d dL / dz)
# ---------------------------------------------------------------------
def log_ddL_dz(
    bk,
    z,
    H0,
    Om0,
    w0,
    Xi0,
    n,
    *,
    dc=None,
    E=None,
    Xi=None,
    x01=None,
    w01=None,
    param="vanilla",
):
    """
    Backend-agnostic log(d(dL)/dz).

    - If dc is None, it will be computed via dcfun_quad (needs x01,w01).
    - Uses Xi_vanilla / Xi_polexp from this module.
    """
    if dc is None:
        if x01 is None or w01 is None:
            raise ValueError("log_ddL_dz: dc is None, so you must pass x01 and w01 to compute dc via dcfun_quad.")
        dc = dcfun_quad(bk, z, H0, Om0, w0, x01, w01)
    
    if E is None:
        E = Efun(bk, z, Om0, w0)

        
         
    onepz = 1.0 + z

    if param == "vanilla":
        if Xi is None:
            Xi = Xi_vanilla(bk, z, Xi0, n)

        term1 = (Xi - n * (1.0 - Xi0) / (onepz**n)) * dc
        term2 = Xi * c_light * onepz / (1.0e03 * H0 * E)
        return bk.log(term1 + term2)

    elif param == "polexp":
        if Xi is None:
            Xi = Xi_polexp(bk, z, Xi0, n)

        exponent = -(1.0 - Xi0) * (1.0 - onepz**n) / (onepz ** (2.0 * n))
        prefactor = Xi0 + (1.0 - Xi0) * onepz ** (-n)

        C = 1.0 - onepz**n
        D = onepz ** (-2.0 * n)
        dC = -n * onepz ** (n - 1.0)
        dD = -2.0 * n * onepz ** (-2.0 * n - 1.0)
        d_exponent = -(1.0 - Xi0) * (dC * D + C * dD)

        d_prefactor = -(1.0 - Xi0) * n * onepz ** (-n - 1.0)
        dXi = bk.exp(exponent) * (d_exponent * prefactor + d_prefactor)

        ddc_dz = c_light / (1.0e03 * H0 * E)
        dL_dz = (dXi * onepz + Xi) * dc + Xi * onepz * ddc_dz
        return bk.log(dL_dz)

    else:
        raise ValueError(f"Unknown param='{param}' (expected 'vanilla' or 'polexp').")


# ---------------------------------------------------------------------
# log normalisation for uniform source-frame
# ---------------------------------------------------------------------


def compute_log_norm_UniformSourceFrame(bk, z_min, z_max, H0, Om0, w0, x01, w01):
    """
    Backend-agnostic compute_log_norm_UniformSourceFrame.

    Uses:
      - dcfun_quad for dc(z)
      - log_dV_dz for log(dV/dz)
      - attrapzvec(bk, y, x) from integrate.py
    """
    z = bk.linspace(z_min, z_max, 10000)

    dc = dcfun_quad(bk, z, H0, Om0, w0, x01, w01)
    log_dVdz = log_dV_dz(bk, z, H0, Om0, w0, dc=dc)

    integrand = bk.exp(log_dVdz) / (1.0 + z)
    norm = attrapzvec(bk, integrand, z)
    return bk.log(norm)