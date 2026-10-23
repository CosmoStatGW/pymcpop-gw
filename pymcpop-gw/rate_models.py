from __future__ import annotations


from cosmology import log_dV_dz, dcfun_quad, Efun


# ---------------------------------------------------------------------
# Madau-Dickinson
# ---------------------------------------------------------------------


def log_psi_z_MD(bk, z, gamma, kappa, zp):
    """
    Madau-Dickinson-like psi(z) piece,
    returned as log_psi(z) .
    """
    lC0 = bk.log(1.0 + (1.0 + zp) ** (-gamma - kappa))
    log_psiz = (
        lC0
        + gamma * bk.log1p(z)
        - bk.log(1.0 + ((1.0 + z) / (1.0 + zp)) ** (gamma + kappa))
    )
    return log_psiz


def log_p_z_MD_unnorm(bk, z, gamma, kappa, zp, H0, Om, w0, *, dc=None, E=None, integrate_dc='quick'):
    """
    log p(z) up to a normalization constant:
      log_psi_MD(z) + log(dV/dz)

    If dc is not provided, compute it with the selected distance integrator.
    """
    if dc is None:
        dc = dcfun_quad(bk, z, H0, Om, w0, integrate_dc=integrate_dc) 
    if E is None:
        E = Efun(bk, z, Om, w0)

    log_psiz = log_psi_z_MD(bk, z, gamma, kappa, zp)
    log_dVdz = log_dV_dz(bk, z, H0, Om, w0, dc=dc, E=E)
    return log_psiz + log_dVdz
