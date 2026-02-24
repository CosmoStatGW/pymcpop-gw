from __future__ import annotations

#from pytensor_utils import attrapzvec, atinterp
from constants import _PI, c_light
from pytensor_tools import zGridGlobals

from constants import _x01_np as x01
from constants import _w01_np as w01
import jax.numpy as jnp
import jax

import pade_cosmo as pc
p, q = pc.flat_wcdm_pade_coefficients(w0=-1.0, zpower=0, xp=jnp)


# ---------------------------------------------------------------------
# quick approximations
# ---------------------------------------------------------------------



def Phi(bk, x):
    num = 1 + 1.320*x + 0.4415* bk.power(x,2) + 0.02656*bk.power(x,3)
    den = 1 + 1.392*x + 0.5121* bk.power(x,2) + 0.03944*bk.power(x,3)
    return num/den
    

def Om_of_z(bk, z, Om0):
    return (1.0-Om0)/Om0/bk.power(1.0+z,3)


def comoving_distance_flatLCDM_approx(bk, z, H0, Om0):
    D_H = (c_light/1.0e3)  / H0 #Mpc
    dist = 2.*D_H * (Phi(bk, Om_of_z(bk, 0., Om0)) - Phi(bk, Om_of_z(bk, z, Om0))/bk.sqrt(1.+z))/bk.sqrt(Om0) # in Mpc
    return dist


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


def dcfun_quad(bk, z, H0, Om, w0, integrate_dc='trapz' ):
    """
    Comoving distance d_c(z) in Gpc

    Inputs:
      z: array-like (...,)
      H0: km/s/Mpc

    Returns:
      d_c(z) in Gpc
    """

    if integrate_dc=='gauss_legendre':
        
        z_nodes = z[..., None] * x01
        
        integrand = 1.0 / Efun(bk, z_nodes, Om, w0)
        
        I = bk.sum( w01  * integrand, axis=-1)

        dc_ = (c_light / H0) * z * I * 1e-03
        
    elif integrate_dc=='trapz':

        zz = bk.linspace( 0., z, num=500)
        E = Efun(bk, zz, Om, w0 )
        dc_ = c_light / H0 * bk.trapezoid( 1./E, x=zz, axis=0 )*1e-03

    elif integrate_dc=='pade':
        
        dc_ = pc.comoving_distance_pade(z, H0, Om, w0=-1.0, p=p, q=q, xp=bk) 

    elif integrate_dc=='quick':
        #print("quick pade approx")
        dc_ = comoving_distance_flatLCDM_approx(bk, z, H0, Om, )
        
    else:
        raise ValueError(f"Unknown itegration method: {integrate_dc}")
        
    return dc_

# ---------------------------------------------------------------------
# luminosity distance
# ---------------------------------------------------------------------

def dLfun(bk, z, H0, Om, w0, Xi0, nXi0, *, dc=None, Xi=None, param="vanilla", integrate_dc="trapz"):
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
        dc = dcfun_quad( bk, z, H0, Om, w0, integrate_dc = integrate_dc )

    return Xi * (1.0 + z) * dc


# ---------------------------------------------------------------------
# inversion of dL(z)
# ---------------------------------------------------------------------

def z_from_dL(bk, dL, H0=None, Om=None, w0=None, Xi0=None, nXi0=None, *, z_nodes = None, d_nodes = None, param="vanilla", integrate_dc="trapz", zmin=1e-4, zmax=100):
    
   
    if z_nodes is None:
        #print("warning: recomputing z nodes")
        z_nodes = bk.logspace( bk.log10(zmin), bk.log10(zmax), 1200)    
    
    if d_nodes is None:
        #print("warning: recomputing d nodes")
        d_nodes = dLfun(bk, z_nodes, H0, Om, w0, Xi0, nXi0, dc=None, Xi=None, param=param, integrate_dc=integrate_dc)


    return bk.interp(dL, d_nodes, z_nodes, left = "extrapolate", right = "extrapolate" )
        




# ---------------------------------------------------------------------
# log(dV/dz)
# ---------------------------------------------------------------------
def log_dV_dz(bk, z, H0, Om0, w0, *, dc=None, E=None ):
    """
    Backend-agnostic log(dV/dz).

    If dc is not provided, it will be computed via dcfun_quad (needs x01,w01).

    log(dV/dz) =
      log(4π) + log(c) - log(H0) + 2log(dc) - log(E(z)) - 3log(10)
    """
    if dc is None:
        dc = dcfun_quad(bk, z, H0, Om0, w0 )

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
    param="vanilla",
):
    """
    Backend-agnostic log(d(dL)/dz).

    - If dc is None, it will be computed via dcfun_quad (needs x01,w01).
    - Uses Xi_vanilla / Xi_polexp from this module.
    """
    if dc is None:
        dc = dcfun_quad(bk, z, H0, Om0, w0,  )
    
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


def compute_log_norm_UniformSourceFrame(bk, d_min, d_max, H0, Om0, w0, ):
    """
    Backend-agnostic compute_log_norm_UniformSourceFrame.

    Uses:
      - dcfun_quad for dc(z)
      - log_dV_dz for log(dV/dz)
    """

    z_min, z_max = z_from_dL(bk, bk.asarray([d_min, d_max]), H0=H0, Om=Om0, w0=w0, Xi0=1, nXi0=0, z_nodes = None, d_nodes = None, param="vanilla", integrate_dc="trapz", zmin=1e-5, zmax=100)
    
    z = bk.linspace(z_min, z_max, 10000)

    dc = dcfun_quad(bk, z, H0, Om0, w0,)
    log_dVdz = log_dV_dz(bk, z, H0, Om0, w0, dc=dc)

    integrand = bk.exp(log_dVdz) / (1.0 + z)
    norm = bk.trapezoid(integrand, z)
    return bk.log(norm), z_min, z_max