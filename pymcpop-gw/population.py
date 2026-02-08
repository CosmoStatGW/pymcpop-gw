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



def make_dL_to_z_cuvjp(*, bk, eps_interp=1e-12, side_interp="right"):
    """
    Fast custom VJP for z = z(dL) using the *dL-grid bracketing* only.
    Avoids a second z->grid searchsorted/interp in the backward.

    Signature matches your call:
      inv_dL_to_z(dL, dL_grid, zgrid, log_ddL_dz_grid, dL_dtheta_grid, Lambda) -> z

    Gradients:
      - g_dL exact for the chosen linear inversion: dz/ddL = (dz/dk)/(ddL/dk)
      - g_Lambda[:5] via implicit: dz/dtheta = -(dL_dtheta)/(ddL/dz)
      - no grads to grids (return zeros)
    """
    import jax
    import jax.numpy as jnp

    @jax.custom_vjp
    def inv_dL_to_z(dL, dL_grid, zgrid, log_ddL_dz_grid, dL_dtheta_grid, Lambda):
        # Use your standard interpolator in dL space
        return atinterp(bk, dL, dL_grid, zgrid, eps=eps_interp, side=side_interp)

    def fwd(dL, dL_grid, zgrid, log_ddL_dz_grid, dL_dtheta_grid, Lambda):
        # Build dL-space bracketing once
        # i, t satisfy: dL_grid[i] <= dL < dL_grid[i+1] (depending on side)
        i, t = _interp_prepare_bk(bk, dL, dL_grid, eps=eps_interp, side=side_interp)

        # z(dL) by applying same weights to zgrid (1-1 correspondence with dL_grid)
        z = _interp_apply_bk(bk, i, t, zgrid)

        # Build dz/ddL using slopes on the same bracket.
        # dz/ddL = (z[i+1]-z[i]) / (dL[i+1]-dL[i])
        # Need gathers at i and i+1
        i1 = i + 1

        z0  = jnp.take(zgrid,    i,  mode="clip")
        z1  = jnp.take(zgrid,    i1, mode="clip")
        dL0 = jnp.take(dL_grid,  i,  mode="clip")
        dL1 = jnp.take(dL_grid,  i1, mode="clip")

        dz = z1 - z0
        ddL = dL1 - dL0

        # Safe divide (monotonic dL_grid => ddL>0, but keep eps for robustness)
        dz_ddL = dz / (ddL + eps_interp)  # (N,)

        # Interpolate dL_dtheta_grid[:, i] in the SAME bracket using t
        # dL_dtheta_grid shape (5, Nz)
        dLth0 = jnp.take(dL_dtheta_grid, i,  axis=1, mode="clip")  # (5,N)
        dLth1 = jnp.take(dL_dtheta_grid, i1, axis=1, mode="clip")  # (5,N)
        dL_dtheta_at = (1.0 - t)[None, :] * dLth0 + t[None, :] * dLth1

        # dz/dtheta = -(dL/dtheta) / (ddL/dz)  and dz/ddL = 1/(ddL/dz)
        dz_dtheta = -dL_dtheta_at * dz_ddL[None, :]  # (5,N)

        # Save small things + small grids for zeros_like in bwd
        saved = (dz_ddL, dz_dtheta, dL_grid, zgrid, log_ddL_dz_grid, dL_dtheta_grid, Lambda)
        return z, saved

    def bwd(saved, g_z):
        dz_ddL, dz_dtheta, dL_grid, zgrid, log_ddL_dz_grid, dL_dtheta_grid, Lambda = saved

        # g_dL
        g_dL = g_z * dz_ddL

        # Only first 5 Lambda entries affected by inversion path
        g_theta5 = jnp.sum(dz_dtheta * g_z[None, :], axis=1)  # (5,)
        g_Lambda = jnp.zeros_like(Lambda)
        g_Lambda = g_Lambda.at[:5].set(g_theta5)

        # No grads for the grids (treated as constants in this op)
        g_dL_grid = jnp.zeros_like(dL_grid)
        g_zgrid = jnp.zeros_like(zgrid)
        g_logdd = jnp.zeros_like(log_ddL_dz_grid)
        g_dLdth = jnp.zeros_like(dL_dtheta_grid)

        return (g_dL, g_dL_grid, g_zgrid, g_logdd, g_dLdth, g_Lambda)

    inv_dL_to_z.defvjp(fwd, bwd)
    return inv_dL_to_z


def make_dL_to_z_cuvjp_uniform(*, bk, eps_interp=1e-12):
    """
    Custom VJP for z = atinterp_uniform(dL; dL_u, z_u).

    Inputs:
      dL: (N,)
      dL_u: (NdL,)    uniform in dL
      z_u:  (NdL,)
      zgrid: (Nz,)    uniform in z (only used to interpolate dL_dtheta_grid to z)
      dL_dtheta_grid: (5, Nz)
      Lambda: (33,)

    Output:
      z: (N,)

    Gradients:
      - g_dL exact for the chosen linear interpolation on (dL_u, z_u)
      - g_Lambda[:5] via implicit: dz/dtheta = -(dL_dtheta(z)) * dz/ddL
      - no grads to tables/grids
    """
    import jax
    import jax.numpy as jnp

    def _prep_uniform(x, x_u):
        # x_u must be 1D increasing, uniform spacing
        x0 = x_u[0]
        dx = x_u[1] - x_u[0]
        # fractional index
        r = (x - x0) / (dx + eps_interp)
        i = jnp.floor(r).astype(jnp.int32)
        # clip to valid [0, n-2]
        n = x_u.shape[0]
        i = jnp.clip(i, 0, n - 2)
        t = r - i.astype(r.dtype)
        # clip t for safety (can happen due to eps)
        t = jnp.clip(t, 0.0, 1.0)
        return i, t, dx

    def _apply_uniform(i, t, fp):
        fp0 = jnp.take(fp, i, mode="clip")
        fp1 = jnp.take(fp, i + 1, mode="clip")
        return (1.0 - t) * fp0 + t * fp1

    @jax.custom_vjp
    def inv_dL_to_z_uniform(dL, dL_u, z_u, zgrid, dL_dtheta_grid, Lambda):
        # your canonical forward for this branch
        return atinterp_uniform(bk, dL, dL_u, z_u, eps=eps_interp, side="right")

    def fwd(dL, dL_u, z_u, zgrid, dL_dtheta_grid, Lambda):
        # ---- invert on uniform dL table
        i, t, ddL = _prep_uniform(dL, dL_u)
        z = _apply_uniform(i, t, z_u)

        # dz/ddL from local slope in the same bracket
        z0 = jnp.take(z_u, i, mode="clip")
        z1 = jnp.take(z_u, i + 1, mode="clip")
        dz_ddL = (z1 - z0) / (ddL + eps_interp)  # (N,)

        # ---- interpolate dL_dtheta_grid to *z* (zgrid is uniform in this branch)
        # First: build dL_dtheta_u = dL_dtheta_grid evaluated at z_u (NdL ~ 4096 => cheap)
        iz, tz, _dz = _prep_uniform(z_u, zgrid)  # z_u is (NdL,)
        # gather (5, NdL)
        dLth0 = jnp.take(dL_dtheta_grid, iz, axis=1, mode="clip")
        dLth1 = jnp.take(dL_dtheta_grid, iz + 1, axis=1, mode="clip")
        dL_dtheta_u = (1.0 - tz)[None, :] * dLth0 + tz[None, :] * dLth1  # (5,NdL)

        # Then: interpolate dL_dtheta_u along the same dL bracket used for inversion
        dLth_u0 = jnp.take(dL_dtheta_u, i, axis=1, mode="clip")      # (5,N)
        dLth_u1 = jnp.take(dL_dtheta_u, i + 1, axis=1, mode="clip")  # (5,N)
        dL_dtheta_at = (1.0 - t)[None, :] * dLth_u0 + t[None, :] * dLth_u1  # (5,N)

        # Implicit: dz/dtheta = -(dL/dtheta) * dz/ddL
        dz_dtheta = -dL_dtheta_at * dz_ddL[None, :]  # (5,N)

        saved = (dz_ddL, dz_dtheta, dL_u, z_u, zgrid, dL_dtheta_grid, Lambda)
        return z, saved

    def bwd(saved, g_z):
        dz_ddL, dz_dtheta, dL_u, z_u, zgrid, dL_dtheta_grid, Lambda = saved

        g_dL = g_z * dz_ddL  # (N,)

        g_theta5 = jnp.sum(dz_dtheta * g_z[None, :], axis=1)  # (5,)
        g_Lambda = jnp.zeros_like(Lambda)
        g_Lambda = g_Lambda.at[:5].set(g_theta5)

        # No grads to tables/grids
        g_dL_u = jnp.zeros_like(dL_u)
        g_z_u = jnp.zeros_like(z_u)
        g_zgrid = jnp.zeros_like(zgrid)
        g_dL_dtheta_grid = jnp.zeros_like(dL_dtheta_grid)

        return (g_dL, g_dL_u, g_z_u, g_zgrid, g_dL_dtheta_grid, g_Lambda)

    inv_dL_to_z_uniform.defvjp(fwd, bwd)
    return inv_dL_to_z_uniform



def _zeros_like_tree(x):
    # works for arrays, scalars, and lists/tuples of arrays
    return jax.tree_util.tree_map(lambda a: jnp.zeros_like(a), x)
