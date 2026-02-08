from __future__ import annotations

# backend-agnostic pieces
from cosmology import Xi_vanilla, Xi_polexp, Efun, log_ddL_dz as log_ddL_dz_bk
from rate_models import log_p_z_MD_unnorm as log_p_z_MD_unnorm_bk
from spin_models import logpdf_default_spin_gauss as logpdf_default_spin_gauss_bk
from mass_models import logpdf_DPLDP as logpdf_DPLDP_bk
from pytensor_utils import logdiffexp as logdiffexp_bk
from pytensor_utils import logsumexp as _logsumexp
from jax_utils import _interp_prepare_bk, _interp_apply_bk, _interp_apply_multi_bk, _interp_prepare_uniform_bk
from pytensor_utils import atinterp, atinterp_uniform

import jax.numpy as jnp

try:
    import jax    
except Exception as e:
    print(e)
    raise ValueError()



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
    Xi=None,
    E=None,
    log_ddL_dz_pre=None,
    param="vanilla",
    interp_vals_mass=None,
    interp_grids_mass=None,
    is_observed=False,
    z_grid=None,
    verbose=False,
    linear_mass=False
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

    if Xi is None:
        if param == "vanilla":
            Xi = Xi_vanilla(bk, z, Xi0, n)
        elif param == "polexp":
            Xi = Xi_polexp(bk, z, Xi0, n)
        else:
            raise ValueError(f"Unknown param='{param}'")
    # If dc not provided, infer from dL and Xi(z)
    if dc is None:
        dc = dL / (1.0 + z) / Xi
        
    if E is None:
        E = Efun(bk, z, Om0, w0)

    # -----------------------
    # rate model (MD only)
    # -----------------------
    if rate_model == "MD":
        gamma, kappa, zp = Lambda[5], Lambda[6], Lambda[7]
        lpz = log_p_z_MD_unnorm_bk(bk, z, gamma, kappa, zp, H0, Om, w0, dc=dc, E=E)
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
            # Expected packing:
            # interp_vals_mass  = [lp_m1_grid, lp_m2_grid, lC_of_m1_grid, ln_m1]
            # interp_grids_mass = [m1_grid,  m2_grid]
            lp_m1_grid, lp_m2_grid, lC_of_m1_grid, ln_m1 = interp_vals_mass
            m1_grid, m2_grid = interp_grids_mass


            # Use fast JAX custom-VJP interp when running under JAX arrays,
            # otherwise fall back to backend-agnostic interp used in sel_bias.
            use_jax = jnp is not None

            ok = (
                    (m1s >= m1_grid[0]) & (m1s <= m1_grid[-1]) &
                    (m2s >= m2_grid[0]) & (m2s <= m2_grid[-1])
                )
            # avoid C(m1)=0 zone (logC=-inf -> +inf in joint)
            ok = ok & (m1s > m2_grid[0])


            if use_jax:

                if linear_mass:
                    #### Uniform mass grids
                    lpm1 = atinterp_uniform(bk, m1s, m1_grid, lp_m1_grid)
                    lpm2 = atinterp_uniform(bk, m2s, m2_grid, lp_m2_grid)
                    lC   = atinterp_uniform(bk, m1s, m1_grid, lC_of_m1_grid)

                else:
                    
                    m1g = m1_grid #jax.lax.stop_gradient(m1_grid)
                    i1, t1 = _interp_prepare_bk(bk, m1s, m1g, eps=1e-12, side="right")
                    # #i1 = bk.stop_grad(i1)
    
                    # stack once
                    m1_tables = jnp.stack([lp_m1_grid, lC_of_m1_grid], axis=0)  # (2, N)
                    vals = _interp_apply_multi_bk(bk, i1, t1, m1_tables)
                    lpm1 = vals[0]
                    lC   = vals[1]
    
                    lpm2 = atinterp(bk, m2s, m2_grid, lp_m2_grid)

    
            else:
                raise ValueError()
                i1, t1 = _interp_prepare_bk(bk, m1s, m1_grid, eps=1e-30, side="right")
                lpm1 = _interp_apply_bk(bk, i1, t1, lp_m1_grid)

                i2, t2 = _interp_prepare_bk(bk, m2s, m2_grid, eps=1e-30, side="right")
                lpm2 = _interp_apply_bk(bk, i2, t2, lp_m2_grid)

                i3, t3 = _interp_prepare_bk(bk, m1s, m1_grid, eps=1e-30, side="right")
                lC = _interp_apply_bk(bk, i3, t3, lC_of_m1_grid)

            lpdf = lpm1 + lpm2 - lC - ln_m1
            lpmass =  bk.where(ok, lpdf, -1e30)

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
            bk, z, H0, Om, w0, Xi0, n, dc=dc, param=param, Xi=Xi, E=E, 
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
    #zgrid,
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
    # -------- optional precomputed cosmology at injections --------
    zinj=None,
    dcinj=None,
    log_ddL_dz_inj=None,
    XiInj=None,
    Einj=None,
    linear_mass=False
):
    """
    Single canonical selection-bias function used by both forward and VJP.

    If zinj/dcinj/log_ddL_dz_inj are provided, internal inversions/interps are skipped.
    """

    # ---- if not precomputed, compute zinj, dcinj, log_ddL_dz_inj by interpolation ----
    if zinj is None:
        raise ValueError()
        # z(dL): xp=dL_grid, fp=zgrid
        i_dL, t_dL = _interp_prepare_bk(bk, dLinj, dL_grid, eps=eps_interp, side=side_interp)
        zinj = _interp_apply_bk(bk, i_dL, t_dL, z_grid)

    if (dcinj is None) or (log_ddL_dz_inj is None):
        raise ValueError()
        # dc(z), log_ddL_dz(z): xp=zgrid, fp=grids
        i_z, t_z = _interp_prepare_bk(bk, zinj, z_grid, eps=eps_interp, side=side_interp)
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
        Xi=XiInj,
        E=Einj,
        param=param,
        interp_vals_mass=interp_vals_mass,
        interp_grids_mass=interp_grids_mass,
        is_observed=is_observed,
        z_grid=z_grid,
        verbose=verbose,
        linear_mass=linear_mass
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


def make_sel_bias_with_uncertainty_cuvjp(
    *,
    bk,
    rate_model,
    mass_model,
    spin_model,
    smoothing="LVK",
    simplex_repair=False,
    has_m2_break=False,
    norm_gauss="uplow",
    param="vanilla",
    is_observed=False,
    z_grid=None,
    verbose=False,
    subtract_log_p_incl=True,
    eps_interp=1e-12,
    side_interp="right",
    linear_mass=False,
    linear_z=False,
):
    """
    Fast selection custom VJP.

    Backward strategy:
      - Use jax.vjp on sel_bias_with_uncertainty treating zinj constant,
        to get gradients wrt (dcinj, log_ddL_dz_inj, XiInj, Einj, Lambda).
      - Compute g_zinj via cheap chain rule:
            g_z = g_dc * d(dc)/dz + g_logdd * d(logdd)/dz + g_Xi * dXi/dz + g_E * dE/dz
        using precomputed (idx_z_inj, t_z_inj) and the grids.
      - Add implicit correction into Lambda[:5] using dz_dtheta_inj.
      - Do NOT propagate gradients to zinj itself (we replace that path).

    Requires caller to pass (idx_z_inj, t_z_inj) from core.
    """

    # --- helpers ---
    def _zeros_like_tree(x):
        # x may be ndarray or list/tuple of ndarrays
        if isinstance(x, (list, tuple)):
            return type(x)([_zeros_like_tree(xx) for xx in x])
        return jnp.zeros_like(x)

    # We assume zgrid is 1D and strictly increasing
    # Use mean dz for slope scaling (safe for near-uniform grids);
    # if not uniform you can replace with local dz from zgrid[idx+1]-zgrid[idx].
    def _local_dz(zgrid, idx):
        # idx is int array in [0, Nz-2]
        return zgrid[idx + 1] - zgrid[idx]

    def _interp_slope_1d(zgrid, fp, idx, t):
        """
        For linear interp:
            f(z) = (1-t) fp[idx] + t fp[idx+1]
        derivative wrt z:
            df/dz = (fp[idx+1]-fp[idx]) / (zgrid[idx+1]-zgrid[idx])
        """
        dz = _local_dz(zgrid, idx)
        dz = jnp.maximum(dz, eps_interp)
        return (fp[idx + 1] - fp[idx]) / dz

    @jax.custom_vjp
    def sel_cuvjp(
        m1inj, m2inj, dLinj, spinsInj,
        log_p_draw, log_p_incl,
        zgrid, dc_grid, log_ddL_dz_grid,
        zinj, dcinj, log_ddL_dz_inj, XiInj, Einj,
        Lambda, Ndraw,
        dz_dtheta_inj,
        idx_z_inj, t_z_inj,   # <-- REQUIRED for fast bwd
    ):
        log_mu, var_u = sel_bias_with_uncertainty(
            bk,
            m1inj, m2inj, dLinj,
            spinsInj,
            log_p_draw,
            log_p_incl,
            dL_grid=None,
            dc_grid=dc_grid,
            log_ddL_dz_grid=log_ddL_dz_grid,
            Lambda=Lambda,
            Ndraw=Ndraw,
            rate_model=rate_model,
            mass_model=mass_model,
            spin_model=spin_model,
            smoothing=smoothing,
            simplex_repair=simplex_repair,
            has_m2_break=has_m2_break,
            norm_gauss=norm_gauss,
            param=param,
            interp_vals_mass=None,
            interp_grids_mass=None,
            is_observed=is_observed,
            z_grid=zgrid,
            verbose=verbose,
            subtract_log_p_incl=subtract_log_p_incl,
            eps_interp=eps_interp,
            side_interp=side_interp,
            zinj=zinj,
            dcinj=dcinj,
            log_ddL_dz_inj=log_ddL_dz_inj,
            XiInj=XiInj,
            Einj=Einj,
            linear_mass=linear_mass,
        )
        return log_mu, var_u

    def sel_fwd(*args):
        out = sel_cuvjp(*args)
        return out, args

    def sel_bwd(saved_args, g_out):
        (g_log_mu, g_var_u) = g_out

        (
            m1inj, m2inj, dLinj, spinsInj,
            log_p_draw, log_p_incl,
            zgrid, dc_grid, log_ddL_dz_grid,
            zinj, dcinj, log_ddL_dz_inj, XiInj, Einj,
            Lambda, Ndraw,
            dz_dtheta_inj,
            idx_z_inj, t_z_inj,
        ) = saved_args

        # ---------- Part 1: autodiff wrt (dcinj, logdd_inj, XiInj, Einj, Lambda) with zinj held constant ----------
        def psi(dcinj_, logdd_inj_, XiInj_, Einj_, Lambda_):
            return sel_bias_with_uncertainty(
                bk,
                m1inj, m2inj, dLinj,
                spinsInj,
                log_p_draw,
                log_p_incl,
                dL_grid=None,
                dc_grid=dc_grid,
                log_ddL_dz_grid=log_ddL_dz_grid,
                Lambda=Lambda_,
                Ndraw=Ndraw,
                rate_model=rate_model,
                mass_model=mass_model,
                spin_model=spin_model,
                smoothing=smoothing,
                simplex_repair=simplex_repair,
                has_m2_break=has_m2_break,
                norm_gauss=norm_gauss,
                param=param,
                interp_vals_mass=None,
                interp_grids_mass=None,
                is_observed=is_observed,
                z_grid=zgrid,
                verbose=False,  # keep bwd lighter
                subtract_log_p_incl=subtract_log_p_incl,
                eps_interp=eps_interp,
                side_interp=side_interp,
                zinj=zinj,  # CONSTANT HERE
                dcinj=dcinj_,
                log_ddL_dz_inj=logdd_inj_,
                XiInj=XiInj_,
                Einj=Einj_,
                linear_mass=linear_mass,
            )

        # psi returns (log_mu, var_u)
        (log_mu0, var_u0), pull = jax.vjp(psi, dcinj, log_ddL_dz_inj, XiInj, Einj, Lambda)
        g_dcinj, g_logdd_inj, g_XiInj, g_Einj, g_Lambda = pull((g_log_mu, g_var_u))

        # ---------- Part 2: cheap g_zinj via chain rule ----------
        # slopes from grids
        idx = idx_z_inj.astype(jnp.int32)

        ddc_dz = _interp_slope_1d(zgrid, dc_grid, idx, t_z_inj)
        dlogdd_dz = _interp_slope_1d(zgrid, log_ddL_dz_grid, idx, t_z_inj)

        # analytic derivatives
        # E(z) = sqrt( Om (1+z)^3 + (1-Om) (1+z)^{3(1+w0)} )
        H0, Om, w0, Xi0, nXi0 = Lambda[0], Lambda[1], Lambda[2], Lambda[3], Lambda[4]
        onepz = 1.0 + zinj
        a3 = onepz ** 3.0
        a3w = onepz ** (3.0 * (1.0 + w0))
        S = Om * a3 + (1.0 - Om) * a3w
        E = jnp.sqrt(S)
        # dS/dz = 3 Om (1+z)^2 + (1-Om) * a3w * 3(1+w0)/(1+z)
        dS_dz = 3.0 * Om * (onepz ** 2.0) + (1.0 - Om) * a3w * (3.0 * (1.0 + w0) / onepz)
        dE_dz = 0.5 * dS_dz / jnp.maximum(E, eps_interp)

        if param == "vanilla":
            # Xi(z)=Xi0 + (1-Xi0)*(1+z)^(-n)
            dXi_dz = (1.0 - Xi0) * (-nXi0) * (onepz ** (-nXi0 - 1.0))
        else:
            # If polexp, implement analytic derivative here.
            # If you don't, you must NOT include g_Xi term (or define Xi derivative properly).
            raise NotImplementedError("Need dXi/dz for param='polexp' to keep gradients exact.")

        # chain rule
        g_zinj = g_dcinj * ddc_dz + g_logdd_inj * dlogdd_dz + g_XiInj * dXi_dz + g_Einj * dE_dz

        # ---------- Part 3: implicit correction into Lambda[:5] ----------
        # dz_dtheta_inj: (5, Ninj)
        g_theta5_from_inv = jnp.sum(g_zinj[None, :] * dz_dtheta_inj, axis=1)  # (5,)
        g_Lambda = g_Lambda.at[:5].add(g_theta5_from_inv)

        # ---------- Return cotangents for ALL inputs ----------
        z_m1inj = jnp.zeros_like(m1inj)
        z_m2inj = jnp.zeros_like(m2inj)
        z_dLinj = jnp.zeros_like(dLinj)
        z_spins = _zeros_like_tree(spinsInj)
        z_lpd   = jnp.zeros_like(log_p_draw)
        z_lpi   = jnp.zeros_like(log_p_incl)

        z_zgrid = jnp.zeros_like(zgrid)
        z_dcgrid = jnp.zeros_like(dc_grid)
        z_logddgrid = jnp.zeros_like(log_ddL_dz_grid)

        z_zinj = jnp.zeros_like(zinj)         # IMPORTANT: we do not backprop to zinj
        z_Ndraw = jnp.zeros_like(Ndraw)
        z_dz_dtheta = jnp.zeros_like(dz_dtheta_inj)
        z_idx = jnp.zeros_like(idx_z_inj)
        z_t   = jnp.zeros_like(t_z_inj)

        return (
            z_m1inj, z_m2inj, z_dLinj, z_spins,
            z_lpd, z_lpi,
            z_zgrid, z_dcgrid, z_logddgrid,
            z_zinj, g_dcinj, g_logdd_inj, g_XiInj, g_Einj,
            g_Lambda, z_Ndraw,
            z_dz_dtheta,
            z_idx, z_t,
        )

    sel_cuvjp.defvjp(sel_fwd, sel_bwd)
    return sel_cuvjp



def _zeros_like_tree(x):
    # works for arrays, scalars, and lists/tuples of arrays
    return jax.tree_util.tree_map(lambda a: jnp.zeros_like(a), x)
