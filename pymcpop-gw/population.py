from __future__ import annotations

# backend-agnostic pieces
from cosmology import Xi_vanilla, Xi_polexp, log_ddL_dz as log_ddL_dz_bk
from rate_models import log_p_z_MD_unnorm as log_p_z_MD_unnorm_bk
from spin_models import logpdf_default_spin_gauss as logpdf_default_spin_gauss_bk
from mass_models import logpdf_DPLDP as logpdf_DPLDP_bk
from pytensor_utils import logdiffexp as logdiffexp_bk
from pytensor_utils import logsumexp as _logsumexp
from jax_utils import _searchsorted_bk, _interp_prepare_bk, _interp_apply_bk

import jax.numpy as jnp

try:
    from jax_utils import make_interp_pt_cached_dy as _make_interp_pt_cached_dy
    _JAX_INTERP_PT = _make_interp_pt_cached_dy(eps=1e-30, side="right")
except Exception:
    _JAX_INTERP_PT = None


def log_p_pop(
    bk,
    m1s,
    m2s,
    z,
    dL,
    spins,
    Lambda,
    *,
    rate_model,
    mass_model,
    spin_model,
    smoothing="LVK",
    simplex_repair=False,
    has_m2_break=False,
    norm_gauss="uplow",
    dc=None,
    log_ddL_dz_pre=None,
    param="vanilla",
    interp_vals_mass=None,
    interp_grids_mass=None,
    is_observed=False,
    z_grid=None,
    verbose=False,
):
    """
    Backend-agnostic log_p_pop_at,
    currently supporting ONLY:
      rate_model == "MD"
      spin_model == "default_gauss"
      mass_model == "DPLDP"
    """

    # Cosmology hyper-params
    H0, Om, w0, Xi0, n = Lambda[0], Lambda[1], Lambda[2], Lambda[3], Lambda[4]

    # If dc not provided, infer from dL and Xi(z)
    if dc is None:
        if param == "vanilla":
            Xi = Xi_vanilla(bk, z, Xi0, n)
        elif param == "polexp":
            Xi = Xi_polexp(bk, z, Xi0, n)
        else:
            raise ValueError(f"Unknown param='{param}'")
        dc = dL / (1.0 + z) / Xi

    # -----------------------
    # rate model (MD only)
    # -----------------------
    if rate_model == "MD":
        gamma, kappa, zp = Lambda[5], Lambda[6], Lambda[7]
        lpz = log_p_z_MD_unnorm_bk(bk, z, gamma, kappa, zp, H0, Om, w0, dc=dc)
        istart = 8
        z_dpuc = None
    else:
        raise NotImplementedError("Only rate_model=='MD' is implemented in this rewrite.")

    # -----------------------
    # spin model (default_gauss only)
    # -----------------------
    if spin_model == "default_gauss":
        muChi = Lambda[istart + 0]
        sigmaChi = Lambda[istart + 1]
        zeta = Lambda[istart + 2]
        sigmat = Lambda[istart + 3]

        # expected spins layout: (chi1, chi2, cost1, cost2)
        lpspin = logpdf_default_spin_gauss_bk(bk, spins, (muChi, sigmaChi, zeta, sigmat))
        istart_spin = istart + 4
    else:
        raise NotImplementedError("Only spin_model=='default_gauss' is implemented in this rewrite.")

    # -----------------------
    # mass model (DPLDP only)
    # -----------------------
    if mass_model == "DPLDP":
        # 21 params
        x1  = Lambda[istart_spin +  0]; x2  = Lambda[istart_spin +  1]
        x3  = Lambda[istart_spin +  2]; x4  = Lambda[istart_spin +  3]
        x5  = Lambda[istart_spin +  4]; x6  = Lambda[istart_spin +  5]
        x7  = Lambda[istart_spin +  6]; x8  = Lambda[istart_spin +  7]
        x9  = Lambda[istart_spin +  8]; x10 = Lambda[istart_spin +  9]
        x11 = Lambda[istart_spin + 10]; x12 = Lambda[istart_spin + 11]
        x13 = Lambda[istart_spin + 12]; x14 = Lambda[istart_spin + 13]
        x15 = Lambda[istart_spin + 14]; x16 = Lambda[istart_spin + 15]
        x17 = Lambda[istart_spin + 16]; x18 = Lambda[istart_spin + 17]
        x19 = Lambda[istart_spin + 18]; x20 = Lambda[istart_spin + 19]
        x21 = Lambda[istart_spin + 20]

        lambdaBBHmass = (
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
            x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21
        )
        
        if interp_vals_mass is not None:
            #print("Log p_pop using interp_vals_mass")
            # Expected packing (matches your PyMC construction):
            # interp_vals_mass  = [lp_m1_grid, lp_m2_grid, lC_of_m1_grid, ln_m1]
            # interp_grids_mass = [m1_grid,  m2_grid]
            lp_m1_grid, lp_m2_grid, lC_of_m1_grid, ln_m1 = interp_vals_mass
            m1_grid, m2_grid = interp_grids_mass

            # Use fast JAX custom-VJP interp when running under JAX arrays,
            # otherwise fall back to backend-agnostic interp used in sel_bias.
            use_jax = (
                _JAX_INTERP_PT is not None
                and jnp is not None
                and (type(m1_grid).__module__.startswith("jax") or type(m1s).__module__.startswith("jax"))
            )

            if use_jax:
                lpm1 = _JAX_INTERP_PT(m1s, m1_grid, lp_m1_grid)
                lpm2 = _JAX_INTERP_PT(m2s, m2_grid, lp_m2_grid)
                lC   = _JAX_INTERP_PT(m1s, m1_grid, lC_of_m1_grid)
            else:
                i1, t1 = _interp_prepare_bk(bk, m1s, m1_grid, eps=1e-30, side="right")
                lpm1 = _interp_apply_bk(bk, i1, t1, lp_m1_grid)

                i2, t2 = _interp_prepare_bk(bk, m2s, m2_grid, eps=1e-30, side="right")
                lpm2 = _interp_apply_bk(bk, i2, t2, lp_m2_grid)

                i3, t3 = _interp_prepare_bk(bk, m1s, m1_grid, eps=1e-30, side="right")
                lC = _interp_apply_bk(bk, i3, t3, lC_of_m1_grid)

            lpmass = lpm1 + lpm2 - lC - ln_m1

        else:
            #print("Log p_pop using logpdf_DPLDP_bk")
            lpmass = logpdf_DPLDP_bk(
                bk,
                (m1s, m2s),
                lambdaBBHmass,
                force_m2_less_than_m1=False,
                has_m2_break=has_m2_break,
                smoothing=smoothing,
                interp_vals=None,
                interp_grids=None,
                norm=True,
                simplex_repair=simplex_repair,
                norm_gauss=norm_gauss,
            )
    else:
        raise NotImplementedError("Only mass_model=='DPLDP' is implemented in this rewrite.")

    # -----------------------
    # Jacobian term
    # -----------------------
    if log_ddL_dz_pre is None:
        # IMPORTANT: we already have dc, so cosmology.log_ddL_dz doesn't need x01,w01
        log_dthD_dth = log_ddL_dz_bk(
            bk, z, H0, Om, w0, Xi0, n, dc=dc, param=param
        )
    else:
        log_dthD_dth = log_ddL_dz_pre

    log_dthD_dth = log_dthD_dth + 2.0 * bk.log1p(z)

    # population log density
    lp = lpz - log_dthD_dth + lpmass + lpspin
    return lp






def sel_bias_with_uncertainty(
    bk,
    m1inj,
    m2inj,
    dLinj,
    spinsInj,
    log_p_draw,
    log_p_incl,
    dL_grid,
    dc_grid,
    log_ddL_dz_grid,
    Lambda,
    Ndraw,
    *,
    zgrid,
    rate_model,
    mass_model,
    spin_model,
    smoothing="LVK",
    simplex_repair=False,
    has_m2_break=False,
    norm_gauss="uplow",
    param="vanilla",
    interp_vals_mass=None,
    interp_grids_mass=None,
    is_observed=False,
    z_grid=None,
    verbose=False,
    subtract_log_p_incl=True,
    eps_interp=1e-12,
    side_interp="right",
    # -------- NEW optional precomputed cosmology at injections --------
    zinj=None,
    dcinj=None,
    log_ddL_dz_inj=None,
):
    """
    Single canonical selection-bias function used by both forward and VJP.

    If zinj/dcinj/log_ddL_dz_inj are provided, internal inversions/interps are skipped.
    """

    # ---- if not precomputed, compute zinj, dcinj, log_ddL_dz_inj by interpolation ----
    if zinj is None:
        # z(dL): xp=dL_grid, fp=zgrid
        i_dL, t_dL = _interp_prepare_bk(bk, dLinj, dL_grid, eps=eps_interp, side=side_interp)
        zinj = _interp_apply_bk(bk, i_dL, t_dL, zgrid)

    if (dcinj is None) or (log_ddL_dz_inj is None):
        # dc(z), log_ddL_dz(z): xp=zgrid, fp=grids
        i_z, t_z = _interp_prepare_bk(bk, zinj, zgrid, eps=eps_interp, side=side_interp)
        if dcinj is None:
            dcinj = _interp_apply_bk(bk, i_z, t_z, dc_grid)
        if log_ddL_dz_inj is None:
            log_ddL_dz_inj = _interp_apply_bk(bk, i_z, t_z, log_ddL_dz_grid)

    onepz = 1.0 + zinj
    m1Src = m1inj / onepz
    m2Src = m2inj / onepz

    log_p_pop_vals = log_p_pop(
        bk,
        m1Src, m2Src, zinj, dLinj,
        spinsInj,
        Lambda,
        rate_model=rate_model,
        mass_model=mass_model,
        spin_model=spin_model,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
        has_m2_break=has_m2_break,
        norm_gauss=norm_gauss,
        dc=dcinj,
        log_ddL_dz_pre=log_ddL_dz_inj,
        param=param,
        interp_vals_mass=interp_vals_mass,
        interp_grids_mass=interp_grids_mass,
        is_observed=is_observed,
        z_grid=z_grid,
        verbose=verbose,
    )

    log_sel_b = log_p_pop_vals - log_p_draw
    if subtract_log_p_incl:
        log_sel_b = log_sel_b - log_p_incl

    # fast two-logsumexp reduction (matches your Op)
    x = log_sel_b
    m = bk.max(x)
    u = bk.exp(x - m)
    s1 = bk.sum(u)
    s2 = bk.sum(u * u)
    lse1 = m + bk.log(s1)
    lse2 = 2.0 * m + bk.log(s2)

    logN = bk.log(Ndraw)
    log_mu = lse1 - logN
    logs2  = lse2 - logN

    var_log_lik_u = logdiffexp_bk(bk, logs2 - 2.0 * log_mu, 1.0) - bk.log(Ndraw - 1.0)
    return log_mu, var_log_lik_u