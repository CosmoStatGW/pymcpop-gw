from __future__ import annotations


# We rely on your backend-agnostic cosmology functions
from cosmology import log_dV_dz, dcfun_quad, Efun


# ---------------------------------------------------------------------
# Madau-Dickinson
# ---------------------------------------------------------------------


def log_psi_z_MD(bk, z, gamma, kappa, zp):
    """
    Madau-Dickinson-like psi(z) piece (your legacy formula),
    returned as log_psi(z) - log(1+z) (source-frame factor).
    """
    lC0 = bk.log(1.0 + (1.0 + zp) ** (-gamma - kappa))
    log_psiz = (
        lC0
        + gamma * bk.log1p(z)
        - bk.log(1.0 + ((1.0 + z) / (1.0 + zp)) ** (gamma + kappa))
    )
    return log_psiz - bk.log1p(z)


def log_p_z_MD_unnorm(bk, z, gamma, kappa, zp, H0, Om, w0, *, dc=None, E=None, x01=None, w01=None):
    """
    log p(z) up to a normalization constant:
      log_psi_MD(z) + log(dV/dz)

    If dc is not provided, it will be computed from (x01, w01).
    """
    if dc is None:
        if x01 is None or w01 is None:
            raise ValueError("log_p_z_MD_unnorm: dc is None, so you must pass x01 and w01.")
        dc = dcfun_quad(bk, z, H0, Om, w0, x01, w01)
    if E is None:
        E = Efun(bk, z, Om, w0)

    log_psiz = log_psi_z_MD(bk, z, gamma, kappa, zp)
    log_dVdz = log_dV_dz(bk, z, H0, Om, w0, dc=dc, E=E)
    return log_psiz + log_dVdz



# ---------------------------------------------------------------------
# Power-law
# ---------------------------------------------------------------------



def log_p_z_PL_unnorm(bk, z, gamma, H0, Om, w0, *, dc=None, x01=None, w01=None):
    """
    log p(z) up to a normalization constant:
      gamma*log(1+z) + log(dV/dz) - log(1+z)

    i.e. (gamma-1)*log(1+z) + log(dV/dz)
    """
    if dc is None:
        if x01 is None or w01 is None:
            raise ValueError("log_p_z_PL_unnorm: dc is None, so you must pass x01 and w01.")
        dc = dcfun_quad(bk, z, H0, Om, w0, x01, w01)

    log_psiz = gamma * bk.log1p(z)
    log_dVdz = log_dV_dz(bk, z, H0, Om, w0, dc=dc)
    return log_psiz + log_dVdz - bk.log1p(z)