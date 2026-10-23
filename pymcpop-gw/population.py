from __future__ import annotations

# backend-agnostic pieces
from cosmology import Xi_vanilla, Xi_polexp, Efun, log_ddL_dz as log_ddL_dz_bk, z_from_dL, gp_z_from_dL_and_logdd, dcfun_quad


from rate_models import log_p_z_MD_unnorm
from spin_models import logpdf_default_spin_gauss as logpdf_default_spin_gauss_bk
import mass_models
import spin_models
from numerical_utils import logdiffexp as logdiffexp_bk, Mcq_from_m1m2, logit, safe_logsumexp_jax, atinterp, atinterp_uniform, atcumtrapz, attrapzvec



import jax.numpy as jnp
from jax import lax

import jax




# ---------------------------------------------------------------------
#  utils
# ---------------------------------------------------------------------

def fixed_mass_grid(
    lo_buffer=0.9,
    mmin=3.0,
    mbulk=100.0,
    mmax=5000.0,
    n_buffer=80,
    n_bulk=3500,
    n_tail=700,
):
    g0 = jnp.logspace(jnp.log10(lo_buffer), jnp.log10(mmin), n_buffer, endpoint=False)
    g1 = jnp.logspace(jnp.log10(mmin), jnp.log10(mbulk), n_bulk, endpoint=False)
    g2 = jnp.logspace(jnp.log10(mbulk), jnp.log10(mmax), n_tail)

    return jnp.concatenate([g0, g1, g2])


m1_grid_ = fixed_mass_grid( n_bulk=3500, n_tail=700, n_buffer=50 )
m2_grid_ = fixed_mass_grid( n_bulk=1000, n_tail=400, n_buffer=50 )
# DPUC_TruncM2 uses analytic truncated-normal normalization in log m2.
assert jnp.all(m1_grid_[1:] > m1_grid_[:-1])
assert jnp.all(m2_grid_[1:] > m2_grid_[:-1])
    

def _zeros_like_tree(x):
    # works for arrays, scalars, and lists/tuples of arrays
    return jax.tree_util.tree_map(lambda a: jnp.zeros_like(a), x)

def _add_pytrees(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _dp_m1_log_env(bk, m1, alpha0, alpha1, alpha2, mb1, mb2, delta1, delta2):
    """Smooth 2-break m1 envelope in x=log(m1).

    log E = -a0*x -(a1-a0)*d1*softplus((x-x1)/d1)
                  -(a2-a1)*d2*softplus((x-x2)/d2)
    """
    eps_mass = 1.0e-300
    x = bk.log(bk.maximum(m1, eps_mass))
    x1 = bk.log(bk.maximum(mb1, eps_mass))
    x2 = bk.log(bk.maximum(mb2, eps_mass))
    d1 = bk.maximum(jnp.asarray(delta1, dtype=jnp.float64), 1.0e-6)
    d2 = bk.maximum(jnp.asarray(delta2, dtype=jnp.float64), 1.0e-6)
    return (
        -alpha0 * x
        - (alpha1 - alpha0) * d1 * jax.nn.softplus((x - x1) / d1)
        - (alpha2 - alpha1) * d2 * jax.nn.softplus((x - x2) / d2)
    )

def _dp_m1_env_extra_count(DP_m1_env_form):
    return 4 if str(DP_m1_env_form) == "smooth_broken_pl_2breaks" else 0


def lambda_slices(
    rate_model,
    spin_model,
    mass_model,
    K_dp=30,
    DPUC_TruncM2_shared_weights=False,
    DP_m1_env_form="powerlaw",
):
    """Slices in the existing retained-model Lambda layout."""
    if rate_model != "MD":
        raise ValueError(f"Unknown rate_model: {rate_model}")
    cosmo = slice(0, 5)
    rate = slice(5, 8)
    i = 8
    if spin_model == "default_gauss":
        spin = slice(i, i + 4)
        i += 4
    elif spin_model == "none":
        spin = slice(i, i)
    else:
        raise ValueError(f"Unknown spin_model: {spin_model}")

    if mass_model == "PLPreg":
        n_mass = 8
    elif mass_model == "DPLDP":
        n_mass = 21
    elif mass_model == "DPLDP-z":
        n_mass = 47
    elif mass_model == "DPUC":
        K = int(K_dp)
        n_mass = 5 * K + 8 + _dp_m1_env_extra_count(DP_m1_env_form)
    elif mass_model == "DPUC_Primary":
        K = int(K_dp)
        n_mass = 3 * K + 8 + _dp_m1_env_extra_count(DP_m1_env_form)
    elif mass_model == "DPUC_TruncM2":
        K = int(K_dp)
        W = 1 if bool(DPUC_TruncM2_shared_weights) else 2
        n_mass = (5 + W) * K + 8 + _dp_m1_env_extra_count(DP_m1_env_form)
    else:
        raise ValueError(f"Unknown mass_model: {mass_model}")
    return cosmo, rate, spin, slice(i, i + n_mass)


def split_Lambda(Lambda_, mass_model, rate_model, spin_model):
    """
    Split the flat Lambda_ list into (cosmo, rate, spin, mass) lists,
    following the construction in your model code.

    Returns
    -------
    cosmo_params : list
    rate_params  : list
    spin_params  : list
    mass_params  : list
    """
    i = 0

    # -------------------------
    # Cosmology: [H0, Om, w0, Xi0, nXi0]
    # -------------------------
    cosmo = Lambda_[i:i+5]
    i += 5

    # -------------------------
    # Rate model
    # -------------------------
    if rate_model != "MD":
        raise ValueError(f"Unknown rate_model: {rate_model}")
    rate = Lambda_[i:i+3]
    i += 3

    if spin_model == "default_gauss":
        spin = Lambda_[i:i+4]
        i += 4
    elif spin_model == "none":
        spin = []
    else:
        raise ValueError(f"Unknown spin_model: {spin_model}")

    # -------------------------
    # Mass model
    # -------------------------
    if mass_model == "PLPreg":
        # [lamP, alpha, beta, deltam, ml, mh, muM, sM]
        mass = Lambda_[i:i+8]
        i += 8

    elif mass_model == "DPLDP":
        mass = Lambda_[i:i+21]
        i += 21

    elif mass_model == "DPLDP-z":
        mass = Lambda_[i:i+47]
        i += 47

    else:
        raise ValueError(f"Unknown mass_model: {mass_model}")

    return cosmo, rate, spin, mass


def unpack_mass_DPLDP_z(mass_params):
    n_lowz = 21
    n_evo  = 26   # was 23; add mb_inf, z_mb, dz_mb
    n_tot  = n_lowz + n_evo  # 47

    if len(mass_params) != n_tot:
        raise ValueError(f"DPLDP-z mass_params must have length {n_tot}, got {len(mass_params)}")

    lambdaBBHmass_lowz = mass_params[:n_lowz]
    evo_params         = mass_params[n_lowz:n_tot]
    return lambdaBBHmass_lowz, evo_params
    


# ---------------------------------------------------------------------
#  core function
# ---------------------------------------------------------------------

def _make_pop_and_sel_core(
    *,
    bk,
    #zgrid,
    rate_model,
    mass_model,
    spin_model,
    smoothing,
    simplex_repair,
    has_m2_break,
    norm_gauss,
    param,
    verbose,
    subtract_log_p_incl,
    skip_sel=False,
    chunk_inj=0,
    chunk_evt=0,
    K_dp: int = 30,
    DP_truncate=False,
    DP_m1_env=False,
    DP_m1_env_form="powerlaw",
    DP_m1_env_delta1=0.2,
    DP_m1_env_delta2=0.2,
    DP_q_env=False,
    DPUC_Primary_q_norm="conditional",
    DP_corr_m1m2=False,
    DP_corr_eta_rho = 2.0,
    DP_corr_f_minor = 0.25,
    PL_DPUC_normalize_gmm=False,
    DPUC_kernel="gaussian",
    DPUC_kernel_rcut=4.0,
    DPUC_kernel_dr=0.15,
    DPUC_kernel_xcut=3.0,
    DPUC_kernel_ycut=3.0,
    DPUC_q_coord="logitq",
    DPUC_TruncM2_shared_weights=False,
    DPUC_TruncM2_secondary_coord="logm2",
    interp_mass = 0,
    integrate_dc = 'trapz',
    pop_only = False,
    stop_grad_var_u: bool = True,
    return_var = True,
    z_nodes = None,
    is_single_PE=False,
    taper_kind = "sigmoid",   # or "power"
    taper_p = 12.0           # only used if taper_kind == "power"
):
    """Build the single source of truth JAX core function.

    Returns a pure function:
        (evt arrays..., inj arrays..., Lambda, Ndraw[, PE bookkeeping])
        -> (log_evt, log_mu, log_var_sel_u, var_log_lik_evs)

    Semantics:
      - pop_only=False:
          log_evt is the old per-event log_p_pop_evt, shape (Nevents,)
          var_log_lik_evs = 0
      - pop_only=True and PE bookkeeping is provided:
          log_evt is the per-event MC-marginalized log likelihood, shape (Nevents,)
          var_log_lik_evs is the event-MC contribution already in likelihood-variance scale.
    """

    if str(mass_model) == "DPUC_Primary":
        mass_model = "DPUC_Primary"
        DP_q_env = True

    DPUC_Primary_q_norm = str(DPUC_Primary_q_norm)
    if DPUC_Primary_q_norm not in ("conditional", "joint"):
        raise ValueError(
            "DPUC_Primary_q_norm must be 'conditional' or 'joint', "
            f"got {DPUC_Primary_q_norm!r}."
        )
    if str(mass_model) != "DPUC_Primary" and DPUC_Primary_q_norm != "conditional":
        raise ValueError(
            "DPUC_Primary_q_norm='joint' is only supported for mass_model='DPUC_Primary'."
        )

    def _f(
        m1det, m2det, dLdet, spins_evt,
        m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
        Lambda, Ndraw,
        log_PE_prior_evt=None,
        event_id_evt=None,
        Nsamples_evt=None,
        gp_state=None,
    ):

        theta5 = Lambda[:5]
        H0, Om, w0, Xi0, nXi0 = theta5

        m1inj = lax.stop_gradient(m1inj)
        m2inj = lax.stop_gradient(m2inj)
        dLinj = lax.stop_gradient(dLinj)
        spins_inj = lax.stop_gradient(spins_inj)
        log_p_draw = lax.stop_gradient(log_p_draw)
        log_p_incl = lax.stop_gradient(log_p_incl)
        Ndraw = lax.stop_gradient(Ndraw)

        if pop_only:
            m1det = lax.stop_gradient(m1det)
            m2det = lax.stop_gradient(m2det)
            dLdet = lax.stop_gradient(dLdet)
            spins_evt = lax.stop_gradient(spins_evt)
            if log_PE_prior_evt is not None:
                log_PE_prior_evt = lax.stop_gradient(log_PE_prior_evt)
            if event_id_evt is not None:
                event_id_evt = lax.stop_gradient(event_id_evt)
            if Nsamples_evt is not None:
                Nsamples_evt = lax.stop_gradient(Nsamples_evt)

        d_nodes = None

        ##################################################
        # Optional : pre-compute p(m1, m2 (z)) on grids
        # mandaory for DPLDP-z, not encouraged for other
        
        if interp_mass:
            
            # pre-computing mass function for later interpolation
            if mass_model == "DPLDP":
                
                _, _, _, mass_p = split_Lambda(Lambda, mass_model, rate_model, spin_model)


                alpha1_, alpha2_, mb_, mu1_, sigma1_, mu2_, sigma2_, m1_low_, m_high_, delta_m1_, lambda0_, lambda1_, lambda2_, beta_, m2_low_, delta_m2_, epsilon_, m_g_, w_g_, sig_g_l_, sig_g_h_ = mass_p

                lp_m1_grid = mass_models.logpdfm1_DPLDP( bk, m1_grid_, alpha1_, alpha2_, mb_, mu1_, sigma1_, mu2_, sigma2_, m1_low_, m_high_, delta_m1_, lambda0_, lambda1_, lambda2_, epsilon_,  smoothing=smoothing, norm_gauss=norm_gauss, taper_kind=taper_kind, taper_p=taper_p) 


                lp_m2_grid = mass_models.logpdfm2_PLP_reg( bk, m2_grid_, beta_, delta_m2_, m2_low_, m_g=m_g_, w_g=w_g_, sig_g_low = sig_g_l_, sig_g_high = sig_g_h_, has_m2_break=has_m2_break, smoothing=smoothing, taper_kind=taper_kind, taper_p=taper_p ) 


                # CDF over m2
                cdf_m2 = atcumtrapz(bk, bk.exp(lp_m2_grid), m2_grid_)
                cdf_m2 = bk.clip(cdf_m2, 1e-300, jnp.inf)
                
                # CDF lives on m2_grid_[1:]
                m2_cdf_grid = m2_grid_[1:]
                logcdf_m2   = bk.log(cdf_m2)
                
                # C(m1) = CDF evaluated at m2=m1 (clipped into CDF grid support)
                mcap = bk.clip(m1_grid_, m2_cdf_grid[0], m2_cdf_grid[-1])
                
                # NON-UNIFORM interpolation 
                lC_of_m1 = atinterp( bk, mcap, m2_cdf_grid, logcdf_m2 )
                
                # Normalization for m1

                lp_max = bk.max(lp_m1_grid)
                p_shift = bk.exp(lp_m1_grid - lp_max)
                I = attrapzvec(bk, p_shift, m1_grid_)
                I = bk.clip(I, 1e-300, jnp.inf)
                ln = bk.log(I) + lp_max
                
                # Pack for later use
                interp_vals_mass  = [lp_m1_grid, lp_m2_grid, lC_of_m1, ln]
                interp_grids_mass = [m1_grid_, m2_grid_]
                
            
            elif mass_model=='DPLDP-z':

                _, _, _, mass_p = split_Lambda(Lambda, mass_model, rate_model, spin_model)
                lambdaBBHmass_lowz, evo_params = unpack_mass_DPLDP_z(mass_p)

          
                (alpha1_0, alpha2_0, mb_0,
                 mu1_0, sigma1_0, mu2_0, sigma2_0,
                 m1_low, m_high, delta_m1,
                 lambda0_0, lambda1_0, lambda2_0, 
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
                 lambda0_inf, lambda1_inf, lambda2_inf, z_lambda, dz_lambda) = evo_params

                # ---------
                # 1) m2 grids (depend on m2 params, but NOT on z in your current model)
                # ---------
                lp_m2_grid = mass_models.logpdfm2_PLP_reg( bk,
                    m2_grid_, beta , delta_m2 , m2_low ,
                    m_g=m_g, w_g=w_g,  sig_g_low=sig_g_low , sig_g_high=sig_g_high ,
                    has_m2_break=has_m2_break, smoothing=smoothing, taper_kind=taper_kind, taper_p=taper_p
                )  # shape (N2,)
            
                # lC_grid evaluated on m1_grid (shape (N1,))

                lp_m2_grid = bk.where(bk.isfinite(lp_m2_grid), lp_m2_grid, -1e30)

                p2 = bk.where(
                    bk.isfinite(lp_m2_grid),
                    bk.exp(lp_m2_grid),
                    0.0,
                )
                
                cdf_m2 = atcumtrapz(bk, p2, m2_grid_)
                cdf_m2 = bk.where(bk.isfinite(cdf_m2), cdf_m2, 0.0)
                cdf_m2 = bk.maximum(cdf_m2, 1e-300)

                # CDF lives on m2_grid_[1:]
                m2_cdf_grid = m2_grid_[1:]
                logcdf_m2   = bk.log(cdf_m2)
                
                # C(m1) = CDF evaluated at m2=m1 (clipped into CDF grid support)
                mcap = bk.clip(m1_grid_, m2_cdf_grid[0], m2_cdf_grid[-1])

                logcdf_m2 = bk.where( bk.isfinite(logcdf_m2), logcdf_m2, -708.3964)

                # NON-UNIFORM interpolation
                lC_of_m1 = atinterp( bk, mcap, m2_cdf_grid, logcdf_m2 )
                

                # ---------
                # 2) Bank lp_m1(z_k, m1_grid_) and ln(z_k)
                # ---------
                K  = z_nodes.shape[0]
                N1 = m1_grid_.shape[0]
                
                M = bk.broadcast_to(m1_grid_[None, :], (K, N1))
                Z = bk.broadcast_to(z_nodes[:, None],   (K, N1))
                
                lp_flat = mass_models.logpdfm1_DPLDP_z( bk, 
                    M.reshape((K * N1,)),
                    Z.reshape((K * N1,)),
                    alpha1_0, alpha2_0, mb_0,
                    mu1_0, sigma1_0, mu2_0, sigma2_0,
                    m1_low , m_high , delta_m1 ,
                    lambda0_0, lambda1_0, lambda2_0,
                    epsilon ,
                    *evo_params,
                    smoothing=smoothing,
                    simplex_repair=simplex_repair,
                    norm_gauss=norm_gauss,
                    taper_kind=taper_kind, taper_p=taper_p
                )
                lp_flat = bk.where(bk.isfinite(lp_flat), lp_flat, -1e30)
                lp_m1_bank = lp_flat.reshape((K, N1))


                
                lp_max = bk.max(lp_m1_bank, axis=1, keepdims=True)

                p_shift = bk.exp(lp_m1_bank - lp_max)
                
                I = attrapzvec(bk, p_shift, m1_grid_[None, :], axis=1)
                
                ln_bank = bk.where(
                    (I > 0.0) & bk.isfinite(I),
                    bk.log(I) + lp_max[:, 0],
                    -jnp.inf,
                )
                
                lp_m1_bank = bk.where(bk.isfinite(lp_m1_bank), lp_m1_bank, -1e30)
                lp_m2_grid = bk.where(bk.isfinite(lp_m2_grid), lp_m2_grid, -1e30)
                lC_of_m1   = bk.where(bk.isfinite(lC_of_m1),   lC_of_m1,    1e30)
                ln_bank    = bk.where(bk.isfinite(ln_bank),    ln_bank,     1e30)
                
                interp_vals_mass  = [lp_m1_bank, lp_m2_grid, lC_of_m1, ln_bank]
                interp_grids_mass = [m1_grid_, m2_grid_, z_nodes]

            else:
                raise NotImplementedError()

            interp_mass_vals = ( interp_grids_mass, interp_vals_mass )
        
        else:
            
            interp_mass_vals = None

        ##################################################
        # Event-side population evaluation helpers.
        # These exactly reproduce the old event-side computation when called
        # on the full event arrays. In marginal PE mode they are called on PE chunks.
        ##################################################

        def _event_logp_raw(Lam, interp_vals, m1d, m2d, dLd, spd, gp_state_=None):
            H0_, Om_, w0_, Xi0_, nXi0_ = Lam[:5]

            if gp_state_ is None:
                z_ = z_from_dL(
                    bk, dLd,
                    H0=H0_, Om=Om_, w0=w0_, Xi0=Xi0_, nXi0=nXi0_,
                    z_nodes=z_nodes,
                    d_nodes=d_nodes,
                    integrate_dc=integrate_dc,
                )
                log_ddL_dz_evt = None
                dc_evt = None
            else:
                z_, log_ddL_dz_evt = gp_z_from_dL_and_logdd(bk, dLd, gp_state_)
                dc_evt = dcfun_quad(bk, z_, H0_, Om_, w0_, integrate_dc=integrate_dc)

            onepz_ = 1.0 + z_
            m1s_ = m1d / onepz_
            m2s_ = m2d / onepz_

            return log_p_pop(
                bk,
                m1s_, m2s_, z_, dLd,
                spin_models._spins_as_list(spd, spin_model),
                Lam,
                rate_model=rate_model,
                mass_model=mass_model,
                spin_model=spin_model,
                smoothing=smoothing,
                simplex_repair=simplex_repair,
                has_m2_break=has_m2_break,
                norm_gauss=norm_gauss,
                dc=dc_evt,
                log_ddL_dz_pre=log_ddL_dz_evt,
                Xi=None,
                E=None,
                param=param,
                verbose=verbose,
                K_dp=K_dp,
                DP_truncate=DP_truncate,
                DP_corr_m1m2=DP_corr_m1m2,
                    DP_corr_eta_rho = DP_corr_eta_rho,
                DP_corr_f_minor = DP_corr_f_minor,
                PL_DPUC_normalize_gmm=PL_DPUC_normalize_gmm,
                    DPUC_kernel=DPUC_kernel,
                    DPUC_kernel_rcut=DPUC_kernel_rcut,
                    DPUC_kernel_dr=DPUC_kernel_dr,
        DPUC_kernel_xcut=DPUC_kernel_xcut,
        DPUC_kernel_ycut=DPUC_kernel_ycut,
        DPUC_q_coord=DPUC_q_coord,
                DP_m1_env=DP_m1_env,
                DP_m1_env_form=DP_m1_env_form,
            DP_m1_env_delta1=DP_m1_env_delta1,
            DP_m1_env_delta2=DP_m1_env_delta2,
            DP_q_env=DP_q_env,
            DPUC_Primary_q_norm=DPUC_Primary_q_norm,
                DPUC_TruncM2_shared_weights=bool(DPUC_TruncM2_shared_weights),
                DPUC_TruncM2_secondary_coord=DPUC_TruncM2_secondary_coord,
                interp_mass_vals=interp_vals,
                taper_kind=taper_kind, taper_p=taper_p
            )

        def _segment_logsumexp_evt(x, sid, nseg):
            if gp_state is not None:
                # Finite-safe reduction from the historical GP implementation.
                x_finite = jnp.isfinite(x)
                x_for_max = jnp.where(x_finite, x, -jnp.inf)
                m = jax.ops.segment_max(x_for_max, sid, num_segments=nseg)
                m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
                m_safe = lax.stop_gradient(m_safe)
                shifted = jnp.where(x_finite, x - m_safe[sid], -jnp.inf)
                shifted = jnp.clip(shifted, -745.0, 60.0)
                u = jnp.exp(shifted)
                u = jnp.where(x_finite, u, 0.0)
                total = jax.ops.segment_sum(u, sid, num_segments=nseg)
                total_safe = jnp.clip(total, 1e-300, jnp.inf)
                return jnp.where(total > 0.0, m_safe + jnp.log(total_safe), -jnp.inf)
            # Stable segment logsumexp over flattened PE samples.
            m = jax.ops.segment_max(x, sid, num_segments=nseg)
            m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
            u = jnp.exp(x - m_safe[sid])
            u = jnp.where(jnp.isfinite(u), u, 0.0)
            s = jax.ops.segment_sum(u, sid, num_segments=nseg)
            return jnp.where(s > 0.0, m_safe + jnp.log(s), -jnp.inf)

        def _event_marginal_full(Lam, interp_vals):
            logp_flat = _event_logp_raw(Lam, interp_vals, m1det, m2det, dLdet, spins_evt, gp_state)
            logw_flat = logp_flat - log_PE_prior_evt

            nseg = Nsamples_evt.shape[0]
            log_sum_w = _segment_logsumexp_evt(logw_flat, event_id_evt, nseg)
            log_sum_w2 = _segment_logsumexp_evt(2.0 * logw_flat, event_id_evt, nseg)

            logN = jnp.log(Nsamples_evt)
            log_l_evt = log_sum_w - logN

            logs2 = log_sum_w2 - 2.0 * logN
            logq = log_sum_w2 - 2.0 * log_sum_w + logN
            
            log_var_evt = logdiffexp_bk(
                bk,
                logq,
                0.0,
            ) - jnp.log(Nsamples_evt - 1.0)



            var_log_lik_evs = jnp.sum(jnp.exp(log_var_evt))
            return log_l_evt, lax.stop_gradient(var_log_lik_evs)

        def _event_marginal_streaming(Lam, interp_vals):
            M = m1det.shape[0]
            B = int(chunk_evt) if int(chunk_evt) > 0 else M

            if B >= M:
                return _event_marginal_full(Lam, interp_vals)

            n_chunks = (M + B - 1) // B
            Mpad = n_chunks * B
            nseg = Nsamples_evt.shape[0]

            @jax.custom_vjp
            def _marginal_core(
                Lam_,
                interp_,
                m1_all,
                m2_all,
                dL_all,
                sp_all,
                log_pe_all,
                event_id_all,
                Nsamples_all,
            ):
                return _marginal_core_impl(
                    Lam_,
                    interp_,
                    m1_all,
                    m2_all,
                    dL_all,
                    sp_all,
                    log_pe_all,
                    event_id_all,
                    Nsamples_all,
                )

            def _pad1(x, fill):
                pad = Mpad - x.shape[0]
                return jnp.concatenate(
                    [x, jnp.full((pad,), fill, dtype=x.dtype)],
                    axis=0,
                )

            def _pad2(x, fill):
                pad = Mpad - x.shape[0]
                return jnp.concatenate(
                    [x, jnp.full((pad, x.shape[1]), fill, dtype=x.dtype)],
                    axis=0,
                )

            def _prepare_arrays(m1_all, m2_all, dL_all, sp_all, log_pe_all, event_id_all):
                m1p = _pad1(m1_all, 1.0)
                m2p = _pad1(m2_all, 1.0)
                dLp = _pad1(dL_all, 1.0)
                spp = _pad2(sp_all, 0.0)
                pep = _pad1(log_pe_all, 0.0)
                eip = _pad1(event_id_all, 0).astype(jnp.int32)
                valid = jnp.arange(Mpad) < M
                return m1p, m2p, dLp, spp, pep, eip, valid

            def _chunk_logw(Lam_, interp_, m1p, m2p, dLp, spp, pep, eip, valid, k):
                start = k * B
                z0 = jnp.array(0, dtype=start.dtype)

                m1c = lax.dynamic_slice(m1p, (start,), (B,))
                m2c = lax.dynamic_slice(m2p, (start,), (B,))
                dLc = lax.dynamic_slice(dLp, (start,), (B,))
                spc = lax.dynamic_slice(spp, (start, z0), (B, spp.shape[1]))
                pec = lax.dynamic_slice(pep, (start,), (B,))
                eic = lax.dynamic_slice(eip, (start,), (B,))
                vc = lax.dynamic_slice(valid, (start,), (B,))

                logp = _event_logp_raw(Lam_, interp_, m1c, m2c, dLc, spc)
                logw = jnp.where(vc, logp - pec, -jnp.inf)
                return logw, eic, vc

            def _stream_log_sums(
                Lam_,
                interp_,
                m1_all,
                m2_all,
                dL_all,
                sp_all,
                log_pe_all,
                event_id_all,
                Nsamples_all,
            ):
                m1p, m2p, dLp, spp, pep, eip, valid = _prepare_arrays(
                    m1_all, m2_all, dL_all, sp_all, log_pe_all, event_id_all
                )

                def body(carry, k):
                    lse1, lse2 = carry

                    logw, eic, vc = _chunk_logw(
                        Lam_, interp_, m1p, m2p, dLp, spp, pep, eip, valid, k
                    )

                    lse1_c = _segment_logsumexp_evt(logw, eic, nseg)
                    lse2_c = _segment_logsumexp_evt(2.0 * logw, eic, nseg)

                    return (
                        jnp.logaddexp(lse1, lse1_c),
                        jnp.logaddexp(lse2, lse2_c),
                    ), None

                init = (
                    jnp.full((nseg,), -jnp.inf, dtype=jnp.float64),
                    jnp.full((nseg,), -jnp.inf, dtype=jnp.float64),
                )

                (log_sum_w, log_sum_w2), _ = lax.scan(
                    body,
                    init,
                    jnp.arange(n_chunks, dtype=jnp.int32),
                )
                return log_sum_w, log_sum_w2

            def _marginal_core_impl(
                Lam_,
                interp_,
                m1_all,
                m2_all,
                dL_all,
                sp_all,
                log_pe_all,
                event_id_all,
                Nsamples_all,
            ):
                log_sum_w, log_sum_w2 = _stream_log_sums(
                    Lam_,
                    interp_,
                    m1_all,
                    m2_all,
                    dL_all,
                    sp_all,
                    log_pe_all,
                    event_id_all,
                    Nsamples_all,
                )

                logN = jnp.log(Nsamples_all)
                log_l_evt = log_sum_w - logN

                logs2 = log_sum_w2 - 2.0 * logN
                logq = log_sum_w2 - 2.0 * log_sum_w + logN
                
                log_var_evt = logdiffexp_bk(
                    bk,
                    logq,
                    0.0,
                ) - jnp.log(Nsamples_all - 1.0)


            #     jax.debug.print(
            #     "log_var_evt min={} max={} finite={} var_evs={}",
            #     jnp.nanmin(log_var_evt),
            #     jnp.nanmax(log_var_evt),
            #     jnp.all(jnp.isfinite(log_var_evt)),
            #     jnp.sum(jnp.exp(log_var_evt)),
            # )

                var_evs = jnp.sum(jnp.exp(log_var_evt))
                return log_l_evt, lax.stop_gradient(var_evs)

            def _marginal_fwd(
                Lam_,
                interp_,
                m1_all,
                m2_all,
                dL_all,
                sp_all,
                log_pe_all,
                event_id_all,
                Nsamples_all,
            ):
                out = _marginal_core_impl(
                    Lam_,
                    interp_,
                    m1_all,
                    m2_all,
                    dL_all,
                    sp_all,
                    log_pe_all,
                    event_id_all,
                    Nsamples_all,
                )

                log_sum_w, _ = _stream_log_sums(
                    Lam_,
                    interp_,
                    m1_all,
                    m2_all,
                    dL_all,
                    sp_all,
                    log_pe_all,
                    event_id_all,
                    Nsamples_all,
                )

                residual = (
                    Lam_,
                    interp_,
                    log_sum_w,
                    m1_all,
                    m2_all,
                    dL_all,
                    sp_all,
                    log_pe_all,
                    event_id_all,
                )
                return out, residual

            def _marginal_bwd(res, g):
                (
                    Lam_,
                    interp_,
                    log_sum_w,
                    m1_all,
                    m2_all,
                    dL_all,
                    sp_all,
                    log_pe_all,
                    event_id_all,
                ) = res

                g_log_l_evt, _g_var = g

                m1p, m2p, dLp, spp, pep, eip, valid = _prepare_arrays(
                    m1_all, m2_all, dL_all, sp_all, log_pe_all, event_id_all
                )

                def body(carry, k):
                    dLam_acc, dInterp_acc = carry

                    logw, eic, vc = _chunk_logw(
                        Lam_, interp_, m1p, m2p, dLp, spp, pep, eip, valid, k
                    )

                    start = k * B
                    z0 = jnp.array(0, dtype=start.dtype)

                    m1c = lax.dynamic_slice(m1p, (start,), (B,))
                    m2c = lax.dynamic_slice(m2p, (start,), (B,))
                    dLc = lax.dynamic_slice(dLp, (start,), (B,))
                    spc = lax.dynamic_slice(spp, (start, z0), (B, spp.shape[1]))
                    pec = lax.dynamic_slice(pep, (start,), (B,))

                    cot = g_log_l_evt[eic] * jnp.exp(logw - log_sum_w[eic])
                    cot = jnp.where(vc & jnp.isfinite(cot), cot, 0.0)

                    if interp_ is None:
                        def score_lam(Lam2):
                            logp = _event_logp_raw(Lam2, None, m1c, m2c, dLc, spc)
                            return logp - pec

                        _, pull = jax.vjp(score_lam, Lam_)
                        (dLam_c,) = pull(cot)
                        dInterp_c = None
                    else:
                        def score_lam_interp(Lam2, interp2):
                            logp = _event_logp_raw(Lam2, interp2, m1c, m2c, dLc, spc)
                            return logp - pec

                        _, pull = jax.vjp(score_lam_interp, Lam_, interp_)
                        dLam_c, dInterp_c = pull(cot)

                    if interp_ is None:
                        return (dLam_acc + dLam_c, None), None
                    else:
                        return (
                            dLam_acc + dLam_c,
                            _add_pytrees(dInterp_acc, dInterp_c),
                        ), None

                dLam0 = jnp.zeros_like(Lam_)
                dInterp0 = None if interp_vals is None else _zeros_like_tree(interp_vals)

                (dLam, dInterp), _ = lax.scan(
                    body,
                    (dLam0, dInterp0),
                    jnp.arange(n_chunks, dtype=jnp.int32),
                )

                return (
                    dLam,       # Lam_
                    dInterp,    # interp_
                    None,       # m1_all
                    None,       # m2_all
                    None,       # dL_all
                    None,       # sp_all
                    None,       # log_pe_all
                    None,       # event_id_all
                    None,       # Nsamples_all
                )

            _marginal_core.defvjp(_marginal_fwd, _marginal_bwd)

            return _marginal_core(
                Lam,
                interp_vals,
                m1det,
                m2det,
                dLdet,
                spins_evt,
                log_PE_prior_evt,
                event_id_evt,
                Nsamples_evt,
            )

        ##################################################
        # Event contribution.
        # IMPORTANT: for pop_only/marginal mode this does NOT compute all PE logp
        # before streaming. For pop_only=False this is exactly the old event path.
        ##################################################

        if pop_only and (log_PE_prior_evt is not None):
            if gp_state is None:
                log_evt, var_log_lik_evs = _event_marginal_streaming(Lambda, interp_mass_vals)
            else:
                # The custom VJP has no GP-state cotangent; retain the GP fallback.
                log_evt, var_log_lik_evs = _event_marginal_full(Lambda, interp_mass_vals)
        else:
            log_evt = _event_logp_raw(
                Lambda, interp_mass_vals,
                m1det, m2det, dLdet, spins_evt, gp_state,
            )
            var_log_lik_evs = jnp.asarray(0.0, dtype=jnp.float64)

        if skip_sel:
            return (
                log_evt,
                jnp.asarray(0.0, dtype=jnp.float64).reshape(()),
                jnp.asarray(0.0, dtype=jnp.float64).reshape(()),
                jnp.asarray(var_log_lik_evs, dtype=jnp.float64).reshape(()),
            )

        log_mu, var_u = sel_bias_with_uncertainty(
            bk,
            m1inj, m2inj, dLinj,
            spins_inj,
            log_p_draw, log_p_incl,
            Lambda, Ndraw,
            rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
            smoothing=smoothing, simplex_repair=simplex_repair,
            has_m2_break=has_m2_break, norm_gauss=norm_gauss,
            param=param, 
            z_grid=z_nodes, 
            d_nodes = d_nodes, 
            verbose=verbose,
            subtract_log_p_incl=subtract_log_p_incl,
            use_streaming_vjp=bool(chunk_inj > 0) and gp_state is None,
            sel_chunk_size=chunk_inj,            # <--- tune
            K_dp=K_dp,
            DP_truncate=DP_truncate,
            DP_m1_env=DP_m1_env,
            DP_m1_env_form=DP_m1_env_form,
            DP_m1_env_delta1=DP_m1_env_delta1,
            DP_m1_env_delta2=DP_m1_env_delta2,
            DP_q_env=DP_q_env,
            DPUC_Primary_q_norm=DPUC_Primary_q_norm,
            DP_corr_m1m2 = DP_corr_m1m2,
            DP_corr_eta_rho = DP_corr_eta_rho,
            DP_corr_f_minor = DP_corr_f_minor,
            PL_DPUC_normalize_gmm = PL_DPUC_normalize_gmm,
            DPUC_kernel = DPUC_kernel,
            DPUC_kernel_rcut = DPUC_kernel_rcut,
            DPUC_kernel_dr = DPUC_kernel_dr,
            DPUC_kernel_xcut = DPUC_kernel_xcut,
            DPUC_kernel_ycut = DPUC_kernel_ycut,
            DPUC_q_coord = DPUC_q_coord,
            interp_mass_vals = interp_mass_vals,
            integrate_dc = integrate_dc,
            return_var = return_var,
            interp_mass = interp_mass,
            gp_state=gp_state,
            DPUC_TruncM2_shared_weights=bool(DPUC_TruncM2_shared_weights),
            taper_kind=taper_kind, taper_p=taper_p
            
        )
        if stop_grad_var_u:
            var_u = lax.stop_gradient(var_u)

        if pop_only and is_single_PE:
            var_log_lik_evs = jnp.zeros_like(var_log_lik_evs)

        return (
            log_evt,
            jnp.asarray(log_mu, dtype=jnp.float64).reshape(()),
            jnp.asarray(var_u, dtype=jnp.float64).reshape(()),
            jnp.asarray(var_log_lik_evs, dtype=jnp.float64).reshape(()),
        )

    return _f




# ---------------------------------------------------------------------
#  p_pop
# ---------------------------------------------------------------------

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
    verbose=False,
    K_dp : int = 30,
    DP_truncate = False,
    DP_m1_env = False,
    DP_m1_env_form="powerlaw",
    DP_m1_env_delta1=0.2,
    DP_m1_env_delta2=0.2,
    DP_q_env = False,
    DPUC_Primary_q_norm="conditional",
    DP_corr_m1m2=False,
    DP_corr_eta_rho = 2.0,
    DP_corr_f_minor = 0.25,
    PL_DPUC_normalize_gmm = False,
    DPUC_kernel="gaussian",
    DPUC_kernel_rcut=4.0,
    DPUC_kernel_dr=0.15,
    DPUC_kernel_xcut=3.0,
    DPUC_kernel_ycut=3.0,
    DPUC_q_coord="logitq",
    DPUC_TruncM2_shared_weights = False,
    DPUC_TruncM2_secondary_coord="logm2",
    interp_mass_vals=  None,
    taper_kind = "sigmoid" ,  # or "power"
    taper_p = 12.0           # only used if taper_kind == "power"
):
    """
    Backend-agnostic log_p_pop
    """


    DPUC_TruncM2_secondary_coord = str(DPUC_TruncM2_secondary_coord)
    if DPUC_TruncM2_secondary_coord not in ("logm2", "logitq"):
        raise ValueError(
            "DPUC_TruncM2_secondary_coord must be 'logm2' or 'logitq', "
            f"got {DPUC_TruncM2_secondary_coord!r}."
        )

    DP_m1_env_form = str(DP_m1_env_form)
    if DP_m1_env_form not in ("powerlaw", "smooth_broken_pl_2breaks"):
        raise ValueError(
            "DP_m1_env_form must be 'powerlaw' or 'smooth_broken_pl_2breaks', "
            f"got {DP_m1_env_form!r}."
        )



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
        E = Efun(bk, z, Om, w0)

    # -----------------------
    # rate model (MD only)
    # -----------------------
    log_one_p_z = bk.log1p(z)
    
    if rate_model != "MD":
        raise ValueError(f"Unknown rate_model: {rate_model}")
    gamma, kappa, zp = Lambda[5], Lambda[6], Lambda[7]
    lpz = log_p_z_MD_unnorm(bk, z, gamma, kappa, zp, H0, Om, w0, dc=dc, E=E)
    istart = 8

    # -----------------------
    # spin model 
    # -----------------------
    if spin_model == "default_gauss":
        muChi = Lambda[istart + 0]
        sigmaChi = Lambda[istart + 1]
        zeta = Lambda[istart + 2]
        sigmat = Lambda[istart + 3]

        # expected spins layout: (chi1, chi2, cost1, cost2)
        lpspin = logpdf_default_spin_gauss_bk(bk, spins, (muChi, sigmaChi, zeta, sigmat))
        istart_spin = istart + 4
    elif spin_model == "none":
        lpspin = bk.zeros_like(z)
        istart_spin = istart
    else:
        raise ValueError(f"Unknown spin_model: {spin_model}")

    # -----------------------
    # mass model 
    # -----------------------
    
    # DPLDP
    if mass_model == "DPLDP":

        if interp_mass_vals is not None:
            
            lpmass = mass_models.logpdf_DPLDP_from_interp(
                    bk, (m1s, m2s), interp_mass_vals)

        else:
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
            
            lpmass = mass_models.logpdf_DPLDP(
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
                taper_kind = taper_kind,
                taper_p = taper_p
        )

    # PLPreg
    elif mass_model == "PLPreg":

        lp  = Lambda[istart_spin + 0]
        al   = Lambda[istart_spin + 1]
        bb   = Lambda[istart_spin + 2]
        dm   = Lambda[istart_spin + 3]
        ml   = Lambda[istart_spin + 4]
        mh   = Lambda[istart_spin + 5]
        muM  = Lambda[istart_spin + 6]
        sM   = Lambda[istart_spin + 7]

        lambdaBBHmass = (
            lp, al, bb, dm, ml, mh, muM, sM
        )


        lpmass = mass_models.logpdf_PLP_reg(
                bk,
                (m1s, m2s),
                lambdaBBHmass,
                smoothing=smoothing,
                 taper_kind = taper_kind,
                taper_p = taper_p
            )
    
    # DPLDP-z
    elif mass_model == "DPLDP-z":


        if interp_mass_vals is not None:
            
            lpmass = mass_models.logpdf_DPLDP_z_from_interp(
                    bk, (m1s, m2s), z, interp_mass_vals)
            
        else:
            # ------------------------------------------------------------
            # UNPACK low-z mass hyperparameters (same 21 as non-evolving)
            # ------------------------------------------------------------
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
        
            lambdaBBHmass_lowz = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                                  x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21]
        
            # ------------------------------------------------------------
            # UNPACK evolution hyperparameters (27 scalars):
            #   (theta_inf, z_theta, dz_theta) for:
            #    alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2,
            #    lambda0, lambda1
            # ------------------------------------------------------------
            j = istart_spin + 21
        
            alpha1_inf  = Lambda[j +  0]; z_alpha1  = Lambda[j +  1]; dz_alpha1  = Lambda[j +  2]
            alpha2_inf  = Lambda[j +  3]; z_alpha2  = Lambda[j +  4]; dz_alpha2  = Lambda[j +  5]
            mb_inf      = Lambda[j +  6]; z_mb      = Lambda[j +  7]; dz_mb      = Lambda[j +  8]
            mu1_inf     = Lambda[j +  9]; z_mu1     = Lambda[j + 10]; dz_mu1     = Lambda[j + 11]
            sigma1_inf  = Lambda[j + 12]; z_sigma1  = Lambda[j + 13]; dz_sigma1  = Lambda[j + 14]
            mu2_inf     = Lambda[j + 15]; z_mu2     = Lambda[j + 16]; dz_mu2     = Lambda[j + 17]
            sigma2_inf  = Lambda[j + 18]; z_sigma2  = Lambda[j + 19]; dz_sigma2  = Lambda[j + 20]
            lambda0_inf = Lambda[j + 21]
            lambda1_inf = Lambda[j + 22]
            lambda2_inf = Lambda[j + 23]
            z_lambda    = Lambda[j + 24]
            dz_lambda   = Lambda[j + 25]
            evo_params = [
                    alpha1_inf,  z_alpha1,  dz_alpha1,
                    alpha2_inf,  z_alpha2,  dz_alpha2,
                    mb_inf,      z_mb,      dz_mb,
                    mu1_inf,     z_mu1,     dz_mu1,
                    sigma1_inf,  z_sigma1,  dz_sigma1,
                    mu2_inf,     z_mu2,     dz_mu2,
                    sigma2_inf,  z_sigma2,  dz_sigma2,
                    lambda0_inf, lambda1_inf, lambda2_inf, z_lambda, dz_lambda,
                ]
        
            # ------------------------------------------------------------
            # Call the redshift-evolving mass pdf
            # ------------------------------------------------------------
            lpmass = mass_models.logpdf_DPLDP_z(
                    bk,
                    (m1s, m2s), z,                     
                    lambdaBBHmass_lowz,
                    evo_params,
                    force_m2_less_than_m1=False,
                    has_m2_break=has_m2_break,
                    smoothing=smoothing,
                    simplex_repair=simplex_repair,
                    norm_gauss=norm_gauss,
                     taper_kind = taper_kind,
                    taper_p = taper_p
                )
            


    # DPUC_TruncM2
    elif mass_model == "DPUC_TruncM2":
    
        if rate_model != "MD":
            raise NotImplementedError(
                "DPUC_TruncM2 is currently implemented for rate_model='MD'. "
                f"Got rate_model={rate_model!r}."
            )
    
    
    
        K = int(K_dp)
        D = 2
    
        # ---------------------------------------------------------------
        # Unpack population parameters
        # ---------------------------------------------------------------
    
        j = int(istart_spin)
    
        mu = Lambda[j : j + D * K].reshape((D, K))
        j += D * K
    
        sd = Lambda[j : j + D * K].reshape((D, K))
        j += D * K
    
        logw1 = Lambda[j : j + K]
        j += K
    
        if bool(DPUC_TruncM2_shared_weights):
            logw2 = logw1
        else:
            logw2 = Lambda[j : j + K]
            j += K
    
        rho = Lambda[j : j + K]
        j += K
    
        m1_low = Lambda[j]
        j += 1
    
        m2_low = Lambda[j]
        j += 1
    
        mmax = Lambda[j]
        j += 1
    
        alpha = Lambda[j]
        j += 1
    
        beta = Lambda[j]
        j += 1
    
        if str(DP_m1_env_form) == "smooth_broken_pl_2breaks":
            alpha1_env = Lambda[j]
            j += 1
    
            alpha2_env = Lambda[j]
            j += 1
    
            mb1_env = Lambda[j]
            j += 1
    
            mb2_env = Lambda[j]
            j += 1
        else:
            alpha1_env = alpha
            alpha2_env = alpha
            mb1_env = jnp.asarray(1.0, dtype=jnp.float64)
            mb2_env = jnp.asarray(1.0, dtype=jnp.float64)
    
        deltam_PL_m1 = Lambda[j]
        j += 1
    
        deltam_PL_m2 = Lambda[j]
        j += 1
    
        deltam_PL_high = Lambda[j]
        j += 1
    
        # ---------------------------------------------------------------
        # Safe physical variables
        # ---------------------------------------------------------------
    
        eps_mass = 1.0e-12
        eps_q = 1.0e-10
    
        m1s_safe = bk.maximum(m1s, eps_mass)
        m2s_safe = bk.maximum(m2s, eps_mass)
    
        q = bk.clip(
            m2s_safe / m1s_safe,
            eps_q,
            1.0 - eps_q,
        )
    
        sd = bk.maximum(sd, eps_mass)
    
        m1_low_safe = bk.maximum(m1_low, eps_mass)
        m2_low_safe = bk.maximum(m2_low, eps_mass)
    
        mmax_safe = bk.maximum(
            mmax,
            m1_low_safe * (1.0 + 1.0e-8),
        )
    
        deltam_PL_m1 = bk.maximum(deltam_PL_m1, eps_mass)
        deltam_PL_m2 = bk.maximum(deltam_PL_m2, eps_mass)
        deltam_PL_high = bk.maximum(deltam_PL_high, eps_mass)
    
        # ---------------------------------------------------------------
        # Primary Gaussian mixture in log(m1)
        # ---------------------------------------------------------------
    
        x1 = bk.log(m1s_safe)[None, :]
    
        mu1 = mu[0][:, None]
        sd1 = sd[0][:, None]
    
        logp1_components_x = (
            -0.5 * bk.log(2.0 * jnp.pi)
            - bk.log(sd1)
            - 0.5 * ((x1 - mu1) / sd1) ** 2
        )
    
        # Convert from a density in log(m1) to a density in m1.
        logp1_components_m = (
            logp1_components_x
            - bk.log(m1s_safe)[None, :]
        )
    
        logg1 = safe_logsumexp_jax(
            logw1[:, None] + logp1_components_m,
            axis=0,
        )
    
        # ---------------------------------------------------------------
        # Smooth mass tapers
        # ---------------------------------------------------------------
    
        log_gate_m1 = bk.zeros_like(m1s_safe)
        log_gate_m2 = bk.zeros_like(m2s_safe)
    
        if DP_truncate:
            # Smooth lower and upper transitions for the primary mass.
            log_gate_m1 = (
                mass_models.log_low_gate_PL_DPUC(
                    bk,
                    m1s_safe,
                    m1_low_safe,
                    deltam_PL_m1,
                )
                + mass_models.log_high_gate_PL_DPUC(
                    bk,
                    m1s_safe,
                    mmax_safe,
                    deltam_PL_high,
                )
            )
    
            # Smooth lower transition for the secondary mass.
            log_gate_m2 = mass_models.log_low_gate_PL_DPUC(
                bk,
                m2s_safe,
                m2_low_safe,
                deltam_PL_m2,
            )
    
        # ---------------------------------------------------------------
        # Optional primary-mass envelope
        # ---------------------------------------------------------------
    
        log_env_m1 = bk.zeros_like(m1s_safe)
    
        if DP_m1_env:
            if str(DP_m1_env_form) == "smooth_broken_pl_2breaks":
                log_env_m1 = _dp_m1_log_env(
                    bk,
                    m1s_safe,
                    alpha,
                    alpha1_env,
                    alpha2_env,
                    mb1_env,
                    mb2_env,
                    DP_m1_env_delta1,
                    DP_m1_env_delta2,
                )
            else:
                log_env_m1 = -alpha * bk.log(m1s_safe)
    
        # ---------------------------------------------------------------
        # Conditional secondary Gaussian
        # ---------------------------------------------------------------
    
        x1_flat = bk.log(m1s_safe)[None, :]
    
        mu1_c = mu[0][:, None]
        sd1_c = bk.maximum(sd[0][:, None], eps_mass)
    
        mu2_base = mu[1][:, None]
        sd2_base = bk.maximum(sd[1][:, None], eps_mass)
    
        rho_c = bk.clip(
            rho[:, None],
            -1.0 + 1.0e-6,
            1.0 - 1.0e-6,
        )
    
        cond_mu2 = (
            mu2_base
            + rho_c
            * (sd2_base / sd1_c)
            * (x1_flat - mu1_c)
        )
    
        cond_sd2 = bk.maximum(
            sd2_base
            * bk.sqrt(
                bk.maximum(
                    1.0 - rho_c**2,
                    1.0e-12,
                )
            ),
            eps_mass,
        )
    
        if DPUC_TruncM2_secondary_coord == "logitq":
            y2 = logit(bk, q)[None, :]
        elif DPUC_TruncM2_secondary_coord == "logm2":
            y2 = bk.log(m2s_safe)[None, :]
        else:
            raise ValueError(
                "DPUC_TruncM2_secondary_coord must be 'logm2' or 'logitq'. "
                f"Got {DPUC_TruncM2_secondary_coord!r}."
            )
    
        logp2_components_x = (
            -0.5 * bk.log(2.0 * jnp.pi)
            - bk.log(cond_sd2)
            - 0.5 * ((y2 - cond_mu2) / cond_sd2) ** 2
        )
    
        # Convert the conditional kernel to a density in physical m2.
        # No conditional truncation normalization is applied.
        if DPUC_TruncM2_secondary_coord == "logitq":
            # y = logit(q), q = m2/m1:
            #
            # |dy/dm2| = 1 / [m2 (1-q)]
            #
            # at fixed m1.
            logp2_components_m = (
                logp2_components_x
                - bk.log(m2s_safe)[None, :]
                - bk.log1p(-q)[None, :]
            )
        else:
            # y = log(m2):
            #
            # |dy/dm2| = 1 / m2.
            logp2_components_m = (
                logp2_components_x
                - bk.log(m2s_safe)[None, :]
            )
    
        # ---------------------------------------------------------------
        # Combine primary and secondary mixtures
        # ---------------------------------------------------------------
    
        if bool(DPUC_TruncM2_shared_weights):
            # Correlated K-component mixture:
            #
            # sum_k w_k p_k(m1) p_k(m2 | m1).
            logg12_m = safe_logsumexp_jax(
                logw1[:, None]
                + logp1_components_m
                + logp2_components_m,
                axis=0,
            )
    
            lpmass_core = logg12_m
    
        else:
            # Product of the primary mixture and a separately weighted
            # conditional-secondary mixture.
            logg2_m = safe_logsumexp_jax(
                logw2[:, None]
                + logp2_components_m,
                axis=0,
            )
    
            lpmass_core = logg1 + logg2_m
    
        # ---------------------------------------------------------------
        # Physical ordering and optional q envelope
        # ---------------------------------------------------------------
    
        # The low-mass boundaries are smooth tapers. The only hard support
        # condition is the physical component-mass ordering.
        log_gate_support = bk.where(
            m2s < m1s,
            0.0,
            -1.0e100,
        )
    
        log_env_q = bk.zeros_like(m1s_safe)
    
        if DP_q_env and DPUC_TruncM2_secondary_coord == "logitq":
            log_env_q = beta * bk.log(q)
    
        # ---------------------------------------------------------------
        # Final mass log-density
        # ---------------------------------------------------------------
    
        lpmass = (
            lpmass_core
            + log_env_m1
            + log_env_q
            + log_gate_m1
            + log_gate_m2
            + log_gate_support
        )

    # DPUC_Primary
    elif mass_model == "DPUC_Primary":

        if rate_model != "MD":
            raise NotImplementedError(
                "DPUC_Primary is currently implemented for rate_model='MD'. "
                f"Got rate_model={rate_model!r}."
            )
        if DP_corr_m1m2:
            raise ValueError(
                "DPUC_Primary is a factorized physical-mass model and does not use "
                "DP_corr_m1m2."
            )

        K = int(K_dp)
        D = 1

        j = int(istart_spin)
        mu = Lambda[j : j + D * K].reshape((D, K)); j += D * K
        sd = Lambda[j : j + D * K].reshape((D, K)); j += D * K
        logw1 = Lambda[j : j + K]; j += K
        m1_low = Lambda[j]; j += 1
        m2_low = Lambda[j]; j += 1
        mmax = Lambda[j]; j += 1
        alpha = Lambda[j]; j += 1
        beta = Lambda[j]; j += 1
        if str(DP_m1_env_form) == "smooth_broken_pl_2breaks":
            alpha1_env = Lambda[j]; j += 1
            alpha2_env = Lambda[j]; j += 1
            mb1_env = Lambda[j]; j += 1
            mb2_env = Lambda[j]; j += 1
        else:
            alpha1_env = alpha
            alpha2_env = alpha
            mb1_env = jnp.asarray(1.0, dtype=jnp.float64)
            mb2_env = jnp.asarray(1.0, dtype=jnp.float64)
        deltam_PL_m1 = Lambda[j]; j += 1
        deltam_PL_m2 = Lambda[j]; j += 1
        deltam_PL_high = Lambda[j]; j += 1

        eps_mass = 1.0e-12
        eps_q = 1.0e-10
        m1s_safe = bk.maximum(m1s, eps_mass)
        m2s_safe = bk.maximum(m2s, eps_mass)
        q = bk.clip(m2s_safe / m1s_safe, eps_q, 1.0 - eps_q)

        sd = bk.maximum(sd, eps_mass)

        # DPUC_Primary GMM is in x=log(m).  Convert to physical-mass density
        # with the Jacobian |d log m / dm| = 1/m.
        x1 = bk.log(m1s_safe)[None, :]
        mu1 = mu[0][:, None]
        sd1 = sd[0][:, None]
        logp1_components_x = (
            -0.5 * bk.log(2.0 * jnp.pi)
            - bk.log(sd1)
            - 0.5 * ((x1 - mu1) / sd1) ** 2
        )
        logg1 = safe_logsumexp_jax(logw1[:, None] + logp1_components_x, axis=0) - bk.log(m1s_safe)

        logg2 = bk.zeros_like(logg1)

        m1_low_safe = bk.maximum(m1_low, eps_mass)
        m2_low_safe = bk.maximum(m2_low, eps_mass)
        mmax_safe = bk.maximum(mmax, m1_low_safe * (1.0 + 1.0e-8))
        deltam_PL_m1 = bk.maximum(deltam_PL_m1, eps_mass)
        deltam_PL_m2 = bk.maximum(deltam_PL_m2, eps_mass)
        deltam_PL_high = bk.maximum(deltam_PL_high, eps_mass)

        log_gate = bk.zeros_like(m1s_safe)
        if DP_truncate:
            log_gate = log_gate + (
                mass_models.log_low_gate_PL_DPUC(bk, m1s_safe, m1_low_safe, deltam_PL_m1)
                + mass_models.log_low_gate_PL_DPUC(bk, m2s_safe, m2_low_safe, deltam_PL_m2)
                + mass_models.log_high_gate_PL_DPUC(bk, m1s_safe, mmax_safe, deltam_PL_high)
                + mass_models.log_high_gate_PL_DPUC(bk, m2s_safe, mmax_safe, deltam_PL_high)
            )

        # Strict ordered-support mask.  This is intentionally not a per-component truncation.
        in_order = m2s < m1s
        log_gate = bk.where(in_order, log_gate, -jnp.inf)

        log_env = bk.zeros_like(m1s_safe)
        if DP_m1_env:
            if str(DP_m1_env_form) == "smooth_broken_pl_2breaks":
                # For the smooth all-CBC envelope use the primary mass only.
                log_env = log_env + _dp_m1_log_env(
                    bk, m1s_safe, alpha, alpha1_env, alpha2_env,
                    mb1_env, mb2_env, DP_m1_env_delta1, DP_m1_env_delta2
                )
            else:
                log_env = log_env - alpha * bk.log(m1s_safe)
        if DP_q_env:
            log_env = log_env + beta * bk.log(q)

        DPUC_Primary_q_norm = str(DPUC_Primary_q_norm)
        if DPUC_Primary_q_norm == "conditional":
            # Conditional secondary-mass normalization for DPUC_Primary:
            #   Z_q(m1) = int_0^1 q^beta G_low(m1*q) G_high(m1*q) m1 dq
            #
            # This is equivalent to integrating over m2, with m2=m1*q and dm2=m1*dq,
            # but avoids dependence on the fixed global m2_grid_ lower edge.
            q_grid = bk.linspace(eps_q, 1.0 - eps_q, 512)

            qg = q_grid[None, :]
            m1g = m1s_safe[:, None]
            m2g = m1g * qg

            log_int = bk.log(bk.maximum(m1g, eps_mass))  # Jacobian: dm2 = m1 dq

            if DP_truncate:
                log_int = log_int + (
                    mass_models.log_low_gate_PL_DPUC(bk, m2g, m2_low_safe, deltam_PL_m2)
                    + mass_models.log_high_gate_PL_DPUC(bk, m2g, mmax_safe, deltam_PL_high)
                )

            if DP_q_env:
                log_int = log_int + beta * bk.log(qg)

            integrand = bk.where(bk.isfinite(log_int), bk.exp(log_int), 0.0)
            Zq = attrapzvec(bk, integrand, bk.stop_grad(q_grid), axis=-1)
            logZq = bk.log(bk.maximum(Zq, eps_mass))

        elif DPUC_Primary_q_norm == "joint":
            # Joint-envelope variant: no conditional normalization at fixed m1.
            logZq = bk.zeros_like(m1s_safe)

        else:
            raise ValueError(
                "DPUC_Primary_q_norm must be 'conditional' or 'joint', "
                f"got {DPUC_Primary_q_norm!r}."
            )

        lpmass = logg1 + logg2 + log_env + log_gate - logZq

    # DPUC mass-only mixture
    elif mass_model == "DPUC":
        K = K_dp
        if DPUC_kernel not in ('gaussian', 'local_native_gaussian', 'local_native_product_gaussian'):
            raise ValueError(f"DPUC_kernel must be 'gaussian', 'local_native_gaussian', or 'local_native_product_gaussian'. Got {DPUC_kernel!r}.")
        if DPUC_q_coord not in ('logitq', 'q'):
            raise ValueError(f"DPUC_q_coord must be 'logitq' or 'q'. Got {DPUC_q_coord!r}.")
        use_q_coord = DPUC_q_coord == 'q'
        use_local_native_kernel = DPUC_kernel in ('local_native_gaussian', 'local_native_product_gaussian')
        use_local_native_product_kernel = DPUC_kernel == 'local_native_product_gaussian'
        j = int(istart_spin)
        K = int(K)
        mu = Lambda[j:j + 2 * K].reshape((2, K))
        j += 2 * K
        sd = Lambda[j:j + 2 * K].reshape((2, K))
        j += 2 * K
        logw = Lambda[j:j + K]
        j += K
        m1_low = Lambda[j]
        j += 1
        m2_low = Lambda[j]
        j += 1
        mmax = Lambda[j]
        j += 1
        alpha = Lambda[j]
        j += 1
        beta = Lambda[j]
        j += 1
        if DP_corr_m1m2:
            rho = Lambda[j:j + K]
            j += K
        else:
            rho = None
        eps_mass = 1e-12
        eps_q = 1e-10
        m1_low_safe = bk.maximum(m1_low, eps_mass)
        m2_low_safe = bk.maximum(m2_low, eps_mass)
        mmax_safe = bk.maximum(mmax, m1_low_safe * (1.0 + 1e-08))
        mmin_safe = m1_low_safe
        deltam_PL_m1 = Lambda[j]
        j += 1
        deltam_PL_m2 = Lambda[j]
        j += 1
        deltam_PL_high = Lambda[j]
        j += 1
        deltam_PL_m1 = bk.maximum(deltam_PL_m1, eps_mass)
        deltam_PL_m2 = bk.maximum(deltam_PL_m2, eps_mass)
        deltam_PL_high = bk.maximum(deltam_PL_high, eps_mass)

        def log_low_plp_reg_m1(m):
            return mass_models.log_low_gate_PL_DPUC(bk, m, m1_low_safe, deltam_PL_m1)

        def log_low_plp_reg_m2(m):
            return mass_models.log_low_gate_PL_DPUC(bk, m, m2_low_safe, deltam_PL_m2)
        m1s_safe = bk.maximum(m1s, eps_mass)
        m2s_safe = bk.maximum(m2s, eps_mass)
        m2s_safe = bk.minimum(m2s_safe, m1s_safe * (1.0 - eps_q))
        Mc_src, q = Mcq_from_m1m2(m1s_safe, m2s_safe)
        q = bk.clip(q, eps_q, 1.0 - eps_q)
        logMc_src = bk.log(bk.maximum(Mc_src, eps_mass))
        if use_q_coord:
            coord2_q = q
            if DP_corr_m1m2:
                logp_xy = mass_models.gaussian_logpdf_pair_corr_xy_trunc_q(bk, logMc_src, coord2_q, mu, sd, rho)
            else:
                logp1, logp2 = mass_models.gaussian_logpdf_pair_trunc_q(bk, logMc_src, coord2_q, mu, sd)
                logp_xy = logp1 + logp2
            native_coord2 = coord2_q
        else:
            logit_q = logit(bk, q)
            if DP_corr_m1m2:
                logp_xy = mass_models.gaussian_logpdf_pair_corr_xy(bk, logMc_src, logit_q, mu, sd, rho)
            else:
                logp1, logp2 = mass_models.gaussian_logpdf_pair(bk, logMc_src, logit_q, mu, sd)
                logp_xy = logp1 + logp2
            native_coord2 = logit_q
        log_kernel_window = 0.0
        if use_local_native_kernel:
            x_std = (logMc_src[None, :] - mu[0][:, None]) / bk.maximum(sd[0][:, None], 1e-12)
            y_std = (native_coord2[None, :] - mu[1][:, None]) / bk.maximum(sd[1][:, None], 1e-12)
            dr_kernel = bk.maximum(jnp.asarray(DPUC_kernel_dr, dtype=jnp.float64), 1e-12)
            if use_local_native_product_kernel:
                log_kernel_window = mass_models.log_high_gate_PL_DPUC(bk, bk.abs(x_std), jnp.asarray(DPUC_kernel_xcut, dtype=jnp.float64), dr_kernel) + mass_models.log_high_gate_PL_DPUC(bk, bk.abs(y_std), jnp.asarray(DPUC_kernel_ycut, dtype=jnp.float64), dr_kernel)
            else:
                if DP_corr_m1m2:
                    rho_k = bk.clip(rho[:, None], -0.999999, 0.999999)
                    one_minus_rho2 = bk.maximum(1.0 - rho_k * rho_k, 1e-12)
                    r2_native = (x_std * x_std + y_std * y_std - 2.0 * rho_k * x_std * y_std) / one_minus_rho2
                else:
                    r2_native = x_std * x_std + y_std * y_std
                r_native = bk.sqrt(bk.maximum(r2_native, 0.0))
                log_kernel_window = mass_models.log_high_gate_PL_DPUC(bk, r_native, jnp.asarray(DPUC_kernel_rcut, dtype=jnp.float64), dr_kernel)
        eps_trunc = 1e-08
        mmax_safe = bk.maximum(mmax, mmin_safe * (1.0 + eps_trunc))
        log_modifier = bk.zeros_like(logMc_src)
        if DP_truncate:
            log_modifier = log_modifier + (log_low_plp_reg_m1(m1s_safe) + log_low_plp_reg_m2(m2s_safe) + mass_models.log_high_gate_PL_DPUC(bk, m1s_safe, mmax_safe, deltam_PL_high))
        if DP_m1_env:
            if str(DP_m1_env_form) == 'smooth_broken_pl_2breaks':
                log_modifier = log_modifier + _dp_m1_log_env(bk, m1s_safe, alpha, alpha1_env, alpha2_env, mb1_env, mb2_env, DP_m1_env_delta1, DP_m1_env_delta2)
            else:
                log_modifier = log_modifier - alpha * bk.log(m1s_safe)
        if DP_q_env:
            log_modifier = log_modifier + beta * bk.log(q)
        log_modifier = bk.where(bk.isfinite(log_modifier), log_modifier, -1e+30)
        logp_components = logp_xy
        if use_local_native_kernel:
            logp_components = logp_components + log_kernel_window
        logp_DPUC = log_modifier + safe_logsumexp_jax(logw[:, None] + logp_components, axis=0)
        if use_q_coord:
            logp_DPUC = logp_DPUC - 2.0 * bk.log(m1s_safe)
        else:
            logp_DPUC = logp_DPUC - bk.log(m2s_safe)
            dm_safe_jac = bk.maximum(m1s_safe - m2s_safe, eps_mass)
            logp_DPUC = logp_DPUC - bk.log(dm_safe_jac)
        if PL_DPUC_normalize_gmm and (not use_local_native_kernel):
            logZ_DPUC = mass_models.dpuc_mixture_logZ_xy_pl_gate(bk, mu[0], sd[0], mu[1], sd[1], logw, rho=rho if DP_corr_m1m2 else None, alpha=alpha, beta=beta, m1_low=m1_low_safe, m2_low=m2_low_safe, mmax=mmax_safe, deltam_PL_m1=deltam_PL_m1, deltam_PL_m2=deltam_PL_m2, deltam_PL_high=deltam_PL_high, DP_truncate=DP_truncate, DP_m1_env=DP_m1_env, DP_q_env=DP_q_env, taper_kind=taper_kind, taper_p=taper_p, eps_mass=eps_mass, eps_q=eps_q)
            logp_DPUC = logp_DPUC - logZ_DPUC
        lpmass = logp_DPUC


            
    # -----------------------
    # Jacobian term
    # -----------------------
    if log_ddL_dz_pre is None:
        
        log_dthD_dth = log_ddL_dz_bk(
            bk, z, H0, Om, w0, Xi0, n, dc=dc, param=param, Xi=Xi, E=E, 
        )
    else:
        log_dthD_dth = log_ddL_dz_pre

    log_dthD_dth = log_dthD_dth + 3.0 * log_one_p_z 
    # here there is a (1+z)^2 prior removal from mass conversion and (1+z) from source-->observer time

    # population log density
    lp = lpz - log_dthD_dth + lpmass + lpspin
    return lp



# ---------------------------------------------------------------------
#  standard sel. bias
# ---------------------------------------------------------------------


def sel_bias_with_uncertainty_legacy(
    bk,
    m1inj,
    m2inj,
    dLinj,
    spinsInj,
    log_p_draw,
    log_p_incl,
    Lambda,
    Ndraw,
    *,
    rate_model,
    mass_model,
    spin_model,
    smoothing="LVK",
    simplex_repair=False,
    has_m2_break=False,
    norm_gauss="uplow",
    param="vanilla",
    z_grid=None,
    d_nodes = None, 
    verbose=False,
    subtract_log_p_incl=False,
    K_dp: int  = 30,
    DP_truncate=False,
    DP_m1_env = False,
    DP_m1_env_form="powerlaw",
    DP_m1_env_delta1=0.2,
    DP_m1_env_delta2=0.2,
    DP_q_env = False,
    DPUC_Primary_q_norm="conditional",
    DP_corr_m1m2=False,
    DP_corr_eta_rho = 2.0,
    DP_corr_f_minor = 0.25,
    PL_DPUC_normalize_gmm = False,
    DPUC_kernel="gaussian",
    DPUC_kernel_rcut=4.0,
    DPUC_kernel_dr=0.15,
    DPUC_kernel_xcut=3.0,
    DPUC_kernel_ycut=3.0,
    DPUC_q_coord="logitq",
    DPUC_TruncM2_shared_weights=False,
    DPUC_TruncM2_secondary_coord="logm2",
    interp_mass_vals=None,
    integrate_dc = 'trapz', 
    return_var = True,
    gp_state=None,
    taper_kind = "sigmoid",   # or "power"
    taper_p = 12.0           # only used if taper_kind == "power"

):
    """
    Single canonical selection-bias function used by both forward and VJP.

    If zinj/dcinj/log_ddL_dz_inj are provided, internal inversions/interps are skipped.
    """

    # ---- compute zinj by interpolation ----

    H0, Om, w0, Xi0, nXi0 = Lambda[0], Lambda[1], Lambda[2], Lambda[3], Lambda[4]
    
    if gp_state is None:
        zinj = z_from_dL(
                bk,
                dLinj,
                H0=H0,
                Om=Om,
                w0=w0,
                Xi0=Xi0,
                nXi0=nXi0,
                z_nodes = z_grid,
                d_nodes = d_nodes,
                param = param,
                integrate_dc = integrate_dc

            )
        log_ddL_dz_inj = None
        dc_inj = None
    else:
        zinj, log_ddL_dz_inj = gp_z_from_dL_and_logdd(bk, dLinj, gp_state)
        dc_inj = dcfun_quad(bk, zinj, H0, Om, w0, integrate_dc=integrate_dc)

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
        dc=dc_inj,
        log_ddL_dz_pre=log_ddL_dz_inj,
        Xi=None,
        E=None,
        param=param,
        verbose=verbose,
        K_dp=K_dp, 
        DP_truncate=DP_truncate,
        DP_m1_env = DP_m1_env,
        DP_q_env = DP_q_env,
        DPUC_Primary_q_norm=DPUC_Primary_q_norm,
        DP_corr_m1m2=DP_corr_m1m2,
            DP_corr_eta_rho = DP_corr_eta_rho,
            DP_corr_f_minor = DP_corr_f_minor,
        PL_DPUC_normalize_gmm = PL_DPUC_normalize_gmm,
        DPUC_kernel=DPUC_kernel,
        DPUC_kernel_rcut=DPUC_kernel_rcut,
        DPUC_kernel_dr=DPUC_kernel_dr,
        DPUC_kernel_xcut=DPUC_kernel_xcut,
        DPUC_kernel_ycut=DPUC_kernel_ycut,
        DPUC_q_coord=DPUC_q_coord,
        DPUC_TruncM2_shared_weights=bool(DPUC_TruncM2_shared_weights),
                DPUC_TruncM2_secondary_coord=DPUC_TruncM2_secondary_coord,
        interp_mass_vals=interp_mass_vals,
        taper_kind = taper_kind,
        taper_p = taper_p,
    )


    
    log_sel_b = log_p_pop_vals - log_p_draw
    if subtract_log_p_incl:
        log_sel_b = log_sel_b - log_p_incl

    # fast two-logsumexp reduction
    x = log_sel_b
    m = bk.max(x, axis=0)
    u = bk.exp(x - m)
    s1 = bk.sum(u)

    Ndraw = bk.asarray(Ndraw, dtype=jnp.float64)
    logN = bk.log(Ndraw)
    lse1 = m + bk.log(s1)
    log_mu = lse1 - logN

    if not return_var:
        return log_mu, bk.zeros_like(log_mu)
    
    s2 = bk.sum(u * u)
    lse2 = 2.0 * m + bk.log(s2)

    logs2  = lse2 - logN

    var_log_lik_u = logdiffexp_bk(bk, logs2 - 2.0 * log_mu, 0. ) - bk.log(Ndraw - 1.0)
    return log_mu, var_log_lik_u





# ---------------------------------------------------------------------
#  sel. bias with streaming 
# ---------------------------------------------------------------------


def sel_bias_with_uncertainty_streaming_vjp_clean(
    bk,
    m1inj,
    m2inj,
    dLinj,
    spinsInj,
    log_p_draw,
    log_p_incl,
    Lambda,
    Ndraw,
    *,
    rate_model,
    mass_model,
    spin_model,
    smoothing="LVK",
    simplex_repair=False,
    has_m2_break=False,
    norm_gauss="uplow",
    param="vanilla",
    z_grid=None,                # kept for API compatibility; unused now
    d_nodes = None, 
    verbose=False,
    chunk_size: int = 65536,
    K_dp: int = 30,
    DP_truncate=False,
    DP_m1_env=False,
    DP_m1_env_form="powerlaw",
    DP_m1_env_delta1=0.2,
    DP_m1_env_delta2=0.2,
    DP_q_env=False,
    DPUC_Primary_q_norm="conditional",
    DP_corr_m1m2=False,
    DP_corr_eta_rho = 2.0,
    DP_corr_f_minor = 0.25,
    PL_DPUC_normalize_gmm = False,
    DPUC_kernel="gaussian",
    DPUC_kernel_rcut=4.0,
    DPUC_kernel_dr=0.15,
    DPUC_kernel_xcut=3.0,
    DPUC_kernel_ycut=3.0,
    DPUC_q_coord="logitq",
    DPUC_TruncM2_shared_weights=False,
    DPUC_TruncM2_secondary_coord="logm2",
    interp_mass_vals=None,
    integrate_dc="trapz",
    return_var = True,
    gp_state=None,
    taper_kind = "sigmoid" ,  # or "power"
    taper_p = 12.0           # only used if taper_kind == "power"
):
    """
    Optimized selection term (patched):
      - forward: streaming (one-pass) computation of log_mu and var_u
      - backward: custom VJP that differentiates ONLY log_mu w.r.t. Lambda
      - z(dL) inversion: delegated to z_from_dL (internal handling of nodes)
      - no gradients are ever taken w.r.t. the injection arrays

    Assumes these exist/imported:
      - z_from_dL(bk, dL, H0, Om, w0, Xi0, nXi0, integrate_dc=...)
      - log_p_pop(...)
      - logdiffexp_bk(bk, a, b)
      - jax, jnp, lax
    """
    if gp_state is not None:
        raise NotImplementedError(
            "Streaming selection VJP does not accept GP state; use the legacy GP path."
        )

    B = int(chunk_size) if chunk_size and chunk_size > 0 else 0

    def _pad_to_multiple(x, n_pad, *, mode="edge"):
        n = x.shape[0]
        pad = n_pad - n
        if pad == 0:
            return x
        if x.ndim == 1:
            tail = jnp.repeat(x[-1:], pad, axis=0) if mode == "edge" else jnp.zeros((pad,), dtype=x.dtype)
            return jnp.concatenate([x, tail], axis=0)
        if x.ndim == 2:
            tail = jnp.repeat(x[-1:, :], pad, axis=0) if mode == "edge" else jnp.zeros((pad, x.shape[1]), dtype=x.dtype)
            return jnp.concatenate([x, tail], axis=0)
        raise ValueError("Unexpected ndim for padding")

    def _make_mask(n, n_pad):
        return jnp.arange(n_pad) < n

    def _score_chunk(
        Lambda_,
        m1_c,
        m2_c,
        dL_c,
        spins_c,
        lpd_c,
        lpi_c,
        mask_c,
    ):
        # Cosmology hyper-params (must match your convention)
        H0, Om, w0, Xi0, nXi0 = Lambda_[0], Lambda_[1], Lambda_[2], Lambda_[3], Lambda_[4]

        # Invert z(dL) via your new internal routine (no z_nodes/d_nodes here)
        zc = z_from_dL(
            bk,
            dL_c,
            H0=H0,
            Om=Om,
            w0=w0,
            Xi0=Xi0,
            nXi0=nXi0,
            integrate_dc=integrate_dc,
            z_nodes = z_grid,
            d_nodes = d_nodes,
        )

        onepz = 1.0 + zc
        m1Src = m1_c / onepz
        m2Src = m2_c / onepz

        lp_pop = log_p_pop(
            bk,
            m1Src, m2Src, zc, dL_c,
            spins_c,
            Lambda_,
            rate_model=rate_model,
            mass_model=mass_model,
            spin_model=spin_model,
            smoothing=smoothing,
            simplex_repair=simplex_repair,
            has_m2_break=has_m2_break,
            norm_gauss=norm_gauss,
            # let log_p_pop compute cosmo auxiliaries internally
            dc=None,
            log_ddL_dz_pre=None,
            Xi=None,
            E=None,
            param=param,
            verbose=verbose,
            K_dp=K_dp,
            DP_truncate=DP_truncate,
            DP_m1_env=DP_m1_env,
            DP_m1_env_form=DP_m1_env_form,
            DP_m1_env_delta1=DP_m1_env_delta1,
            DP_m1_env_delta2=DP_m1_env_delta2,
            DP_q_env=DP_q_env,
            DPUC_Primary_q_norm=DPUC_Primary_q_norm,
            DP_corr_m1m2=DP_corr_m1m2,
                DP_corr_eta_rho = DP_corr_eta_rho,
            DP_corr_f_minor = DP_corr_f_minor,
            PL_DPUC_normalize_gmm = PL_DPUC_normalize_gmm,
            DPUC_kernel=DPUC_kernel,
            DPUC_kernel_rcut=DPUC_kernel_rcut,
            DPUC_kernel_dr=DPUC_kernel_dr,
        DPUC_kernel_xcut=DPUC_kernel_xcut,
        DPUC_kernel_ycut=DPUC_kernel_ycut,
        DPUC_q_coord=DPUC_q_coord,
            DPUC_TruncM2_shared_weights=bool(DPUC_TruncM2_shared_weights),
                DPUC_TruncM2_secondary_coord=DPUC_TruncM2_secondary_coord,
            interp_mass_vals=interp_mass_vals,
            taper_kind = taper_kind,
            taper_p = taper_p,
        )

        x = lp_pop - lpd_c - lpi_c
        x = jnp.where(mask_c, x, -jnp.inf)  # padded entries contribute nothing
        return x  # (B,)

    @jax.custom_vjp
    def _sel_core(
        Lambda_,
        # injection constants (we return zero grads for these)
        m1_,
        m2_,
        dL_,
        spins_,
        lpd_,
        lpi_,
        Ndraw_,
    ):
        log_mu_, var_u_ = _sel_fwd_only(Lambda_, m1_, m2_, dL_, spins_, lpd_, lpi_, Ndraw_)
        return log_mu_, var_u_

    def _sel_fwd_only(
        Lambda_,
        m1_,
        m2_,
        dL_,
        spins_,
        lpd_,
        lpi_,
        Ndraw_,
    ):
        n = m1_.shape[0]
        if B == 0:
            n_chunks, n_pad, B_use = 1, n, n
        else:
            n_chunks = (n + B - 1) // B
            n_pad = n_chunks * B
            B_use = B

        mask = _make_mask(n, n_pad)

        m1p = _pad_to_multiple(m1_, n_pad, mode="edge")
        m2p = _pad_to_multiple(m2_, n_pad, mode="edge")
        dLp = _pad_to_multiple(dL_, n_pad, mode="edge")
        spinsp = _pad_to_multiple(spins_, n_pad, mode="edge")
        lpdp = _pad_to_multiple(lpd_, n_pad, mode="edge")
        lpip = _pad_to_multiple(lpi_, n_pad, mode="edge")

        init = (
            jnp.array(-jnp.inf, dtype=jnp.float64),  # m
            jnp.array(0.0, dtype=jnp.float64),       # s1
            jnp.array(0.0, dtype=jnp.float64),       # s2
        )

        def body(carry, k):
            m, s1, s2 = carry
            start = k * B_use
            z0 = jnp.array(0, dtype=start.dtype)

            m1c = lax.dynamic_slice(m1p, (start,), (B_use,))
            m2c = lax.dynamic_slice(m2p, (start,), (B_use,))
            dLc = lax.dynamic_slice(dLp, (start,), (B_use,))
            spc = lax.dynamic_slice(spinsp, (start, z0), (B_use, spinsp.shape[1]))
            lpdc = lax.dynamic_slice(lpdp, (start,), (B_use,))
            lpic = lax.dynamic_slice(lpip, (start,), (B_use,))
            mc = lax.dynamic_slice(mask, (start,), (B_use,))

            x = _score_chunk(Lambda_, m1c, m2c, dLc, spc, lpdc, lpic, mc)

            m_chunk = jnp.max(x)
            m_new = jnp.maximum(m, m_chunk)

            scale1 = jnp.exp(m - m_new)
            u1 = jnp.exp(x - m_new)
            s1_new = s1 * scale1 + jnp.sum(u1)
            
            if not return_var:
                s2_new = jnp.zeros_like(s1_new)
            else:
                scale2 = jnp.exp(2.0 * (m - m_new))
                u2 = jnp.exp(2.0 * (x - m_new))
                s2_new = s2 * scale2 + jnp.sum(u2)

            return (m_new, s1_new, s2_new), None

        (m_fin, s1_fin, s2_fin), _ = lax.scan(body, init, jnp.arange(n_chunks, dtype=jnp.int32))

        lse1 = m_fin + jnp.log(s1_fin)

        logN = jnp.log(Ndraw_)
        log_mu = lse1 - logN

        if not return_var:
            return log_mu, bk.zeros_like(log_mu)
        else:
            lse2 = 2.0 * m_fin + jnp.log(s2_fin)
            logs2 = lse2 - logN
    
            var_u = logdiffexp_bk(bk, logs2 - 2.0 * log_mu, 0. ) - jnp.log(Ndraw_ - 1.0)
            return log_mu, var_u


    def _sel_fwd(
        Lambda_,
        m1_, m2_, dL_, spins_, lpd_, lpi_, Ndraw_,
    ):
        log_mu, var_u = _sel_fwd_only(Lambda_, m1_, m2_, dL_, spins_, lpd_, lpi_, Ndraw_)
        lse1 = log_mu + jnp.log(Ndraw_)
        # save what backward needs; no d_nodes anymore
        res = (lse1, Ndraw_, Lambda_, m1_, m2_, dL_, spins_, lpd_, lpi_)
        return (log_mu, var_u), res

    def _sel_bwd(res, g):
        (lse1, Ndraw_, Lambda_, m1_, m2_, dL_, spins_, lpd_, lpi_) = res
        g_log_mu, g_var_u = g

        # do not differentiate var_u
        g_var_u = jnp.array(0.0, dtype=jnp.float64)

        n = m1_.shape[0]
        # robust: if B <= 0 or B >= n, avoid padding/chunking
        if (B is None) or (B <= 0) or (B >= n):
            n_chunks = 1
            n_pad = n
            B_use = n
        else:
            n_chunks = (n + B - 1) // B
            n_pad = n_chunks * B
            B_use = B

        mask = _make_mask(n, n_pad)

        m1p = _pad_to_multiple(m1_, n_pad, mode="edge")
        m2p = _pad_to_multiple(m2_, n_pad, mode="edge")
        dLp = _pad_to_multiple(dL_, n_pad, mode="edge")
        spinsp = _pad_to_multiple(spins_, n_pad, mode="edge")
        lpdp = _pad_to_multiple(lpd_, n_pad, mode="edge")
        lpip = _pad_to_multiple(lpi_, n_pad, mode="edge")

        dLambda_acc = jnp.zeros_like(Lambda_)

        def body(carry, k):
            dLam_acc = carry
            start = k * B_use
            z0 = jnp.array(0, dtype=start.dtype)

            m1c = lax.dynamic_slice(m1p, (start,), (B_use,))
            m2c = lax.dynamic_slice(m2p, (start,), (B_use,))
            dLc = lax.dynamic_slice(dLp, (start,), (B_use,))
            spc = lax.dynamic_slice(spinsp, (start, z0), (B_use, spinsp.shape[1]))
            lpdc = lax.dynamic_slice(lpdp, (start,), (B_use,))
            lpic = lax.dynamic_slice(lpip, (start,), (B_use,))
            mc = lax.dynamic_slice(mask, (start,), (B_use,))

            def score_wrapped(Lam_):
                return _score_chunk(Lam_, m1c, m2c, dLc, spc, lpdc, lpic, mc)
            
            # Differentiate only w.r.t Lambda; NEVER w.r.t injection arrays
            x, pull = jax.vjp(score_wrapped, Lambda_)

            w = jnp.exp(x - lse1)
            cot = g_log_mu * w

            (dLam_c,) = pull(cot)
            return dLam_acc + dLam_c, None

        dLambda_acc, _ = lax.scan(
            lambda carry, k: body(carry, k),
            dLambda_acc,
            jnp.arange(n_chunks, dtype=jnp.int32),
        )

        dLambda = dLambda_acc

        zeros_m1 = jnp.zeros_like(m1_)
        zeros_m2 = jnp.zeros_like(m2_)
        zeros_dL = jnp.zeros_like(dL_)
        zeros_sp = jnp.zeros_like(spins_)
        zeros_lpd = jnp.zeros_like(lpd_)
        zeros_lpi = jnp.zeros_like(lpi_)
        zeros_N = jnp.zeros_like(jnp.asarray(Ndraw_).reshape(()))

        # grads must match _sel_core arg order:
        # (Lambda, m1, m2, dL, spins, lpd, lpi, Ndraw)
        return (dLambda, zeros_m1, zeros_m2, zeros_dL, zeros_sp, zeros_lpd, zeros_lpi, zeros_N)

    _sel_core.defvjp(_sel_fwd, _sel_bwd)

    log_mu, var_u = _sel_core(
        Lambda,
        m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl,
        jnp.asarray(Ndraw).reshape(()),
    )
    return log_mu, var_u



def sel_bias_with_uncertainty_streaming_vjp(
    bk,
    m1inj,
    m2inj,
    dLinj,
    spinsInj,
    log_p_draw,
    log_p_incl,
    Lambda,
    Ndraw,
    *,
    rate_model,
    mass_model,
    spin_model,
    smoothing="LVK",
    simplex_repair=False,
    has_m2_break=False,
    norm_gauss="uplow",
    param="vanilla",
    z_grid=None,                # kept for API compatibility; used as z_nodes in z_from_dL
    d_nodes=None,
    verbose=False,
    chunk_size: int = 65536,
    K_dp: int = 30,
    DP_truncate=False,
    DP_m1_env=False,
    DP_m1_env_form="powerlaw",
    DP_m1_env_delta1=0.2,
    DP_m1_env_delta2=0.2,
    DP_q_env=False,
    DPUC_Primary_q_norm="conditional",
    PL_DPUC_normalize_gmm = False,
    DPUC_kernel="gaussian",
    DPUC_kernel_rcut=4.0,
    DPUC_kernel_dr=0.15,
    DPUC_kernel_xcut=3.0,
    DPUC_kernel_ycut=3.0,
    DPUC_q_coord="logitq",
    DP_corr_m1m2=False,
    DP_corr_eta_rho=2.0,
    DP_corr_f_minor=0.25,
    DPUC_TruncM2_shared_weights=False,
    DPUC_TruncM2_secondary_coord="logm2",
    interp_mass_vals=None,
    integrate_dc="trapz",
    return_var=True,
    gp_state=None,
    taper_kind = "sigmoid",   # or "power"
    taper_p = 12.0           # only used if taper_kind == "power"
):
    """
    Optimized selection term with correct custom VJP behavior when interp_mass_vals depends on Lambda.

    Key fix:
      - interp_mass_vals is passed as an EXPLICIT argument to the custom_vjp core,
        and the backward rule returns gradients w.r.t. BOTH (Lambda, interp_mass_vals).
      - This prevents "closed-over value" CustomVJPException and allows gradients to
        flow through the precomputed interpolation banks.

    Assumes these exist/imported:
      - z_from_dL(bk, dL, H0, Om, w0, Xi0, nXi0, integrate_dc=..., z_nodes=..., d_nodes=...)
      - log_p_pop(...)
      - logdiffexp_bk(bk, a, b)
      - jax, jnp, lax
    """

    if gp_state is not None:
        raise NotImplementedError(
            "Streaming selection VJP does not accept GP state; use the legacy GP path."
        )

    if interp_mass_vals is None:
        raise ValueError(
            "sel_bias_with_uncertainty_streaming_vjp requires interp_mass_vals != None "
            "for the 'no closed-over tracer' fix. If you truly want no interpolation, "
            "use the non-interp selection routine or pass interp_mass_vals explicitly."
        )

    B = int(chunk_size) if chunk_size and chunk_size > 0 else 0

    def _pad_to_multiple(x, n_pad, *, mode="edge"):
        n = x.shape[0]
        pad = n_pad - n
        if pad == 0:
            return x
        if x.ndim == 1:
            tail = jnp.repeat(x[-1:], pad, axis=0) if mode == "edge" else jnp.zeros((pad,), dtype=x.dtype)
            return jnp.concatenate([x, tail], axis=0)
        if x.ndim == 2:
            tail = jnp.repeat(x[-1:, :], pad, axis=0) if mode == "edge" else jnp.zeros((pad, x.shape[1]), dtype=x.dtype)
            return jnp.concatenate([x, tail], axis=0)
        raise ValueError("Unexpected ndim for padding")

    def _make_mask(n, n_pad):
        return jnp.arange(n_pad) < n

    def _zeros_like_pytree(tree):
        return jax.tree_util.tree_map(lambda a: jnp.zeros_like(a), tree)

    def _add_pytrees(a, b):
        return jax.tree_util.tree_map(lambda x, y: x + y, a, b)

    def _score_chunk(
        Lambda_,
        interp_mass_vals_,
        m1_c,
        m2_c,
        dL_c,
        spins_c,
        lpd_c,
        lpi_c,
        mask_c,
    ):
        # Cosmology hyper-params (must match your convention)
        H0, Om, w0, Xi0, nXi0 = Lambda_[0], Lambda_[1], Lambda_[2], Lambda_[3], Lambda_[4]

        # Invert z(dL)
        zc = z_from_dL(
            bk,
            dL_c,
            H0=H0,
            Om=Om,
            w0=w0,
            Xi0=Xi0,
            nXi0=nXi0,
            integrate_dc=integrate_dc,
            z_nodes=z_grid,
            d_nodes=d_nodes,
        )

        onepz = 1.0 + zc
        m1Src = m1_c / onepz
        m2Src = m2_c / onepz

        lp_pop = log_p_pop(
            bk,
            m1Src, m2Src, zc, dL_c,
            spins_c,
            Lambda_,
            rate_model=rate_model,
            mass_model=mass_model,
            spin_model=spin_model,
            smoothing=smoothing,
            simplex_repair=simplex_repair,
            has_m2_break=has_m2_break,
            norm_gauss=norm_gauss,
            # let log_p_pop compute cosmo auxiliaries internally
            dc=None,
            log_ddL_dz_pre=None,
            Xi=None,
            E=None,
            param=param,
            verbose=verbose,
            K_dp=K_dp,
            DP_truncate=DP_truncate,
            DP_m1_env=DP_m1_env,
            DP_m1_env_form=DP_m1_env_form,
            DP_m1_env_delta1=DP_m1_env_delta1,
            DP_m1_env_delta2=DP_m1_env_delta2,
            DP_q_env=DP_q_env,
            DPUC_Primary_q_norm=DPUC_Primary_q_norm,
            DP_corr_m1m2=DP_corr_m1m2,
            DP_corr_eta_rho = DP_corr_eta_rho,
            DP_corr_f_minor = DP_corr_f_minor,
            PL_DPUC_normalize_gmm = PL_DPUC_normalize_gmm,
            DPUC_kernel=DPUC_kernel,
            DPUC_kernel_rcut=DPUC_kernel_rcut,
            DPUC_kernel_dr=DPUC_kernel_dr,
        DPUC_kernel_xcut=DPUC_kernel_xcut,
        DPUC_kernel_ycut=DPUC_kernel_ycut,
        DPUC_q_coord=DPUC_q_coord,
            DPUC_TruncM2_shared_weights=bool(DPUC_TruncM2_shared_weights),
            interp_mass_vals=interp_mass_vals_,
            taper_kind = taper_kind,
            taper_p = taper_p,
        )

        x = lp_pop - lpd_c - lpi_c
        x = jnp.where(mask_c, x, -jnp.inf)  # padded entries contribute nothing
        return x  # (B,)

    @jax.custom_vjp
    def _sel_core(
        Lambda_,
        interp_mass_vals_,
        # injection constants (we return zero grads for these)
        m1_,
        m2_,
        dL_,
        spins_,
        lpd_,
        lpi_,
        Ndraw_,
    ):
        log_mu_, var_u_ = _sel_fwd_only(
            Lambda_, interp_mass_vals_, m1_, m2_, dL_, spins_, lpd_, lpi_, Ndraw_
        )
        return log_mu_, var_u_

    def _sel_fwd_only(
        Lambda_,
        interp_mass_vals_,
        m1_,
        m2_,
        dL_,
        spins_,
        lpd_,
        lpi_,
        Ndraw_,
    ):
        n = m1_.shape[0]
        if B == 0:
            n_chunks, n_pad, B_use = 1, n, n
        else:
            n_chunks = (n + B - 1) // B
            n_pad = n_chunks * B
            B_use = B

        mask = _make_mask(n, n_pad)

        m1p = _pad_to_multiple(m1_, n_pad, mode="edge")
        m2p = _pad_to_multiple(m2_, n_pad, mode="edge")
        dLp = _pad_to_multiple(dL_, n_pad, mode="edge")
        spinsp = _pad_to_multiple(spins_, n_pad, mode="edge")
        lpdp = _pad_to_multiple(lpd_, n_pad, mode="edge")
        lpip = _pad_to_multiple(lpi_, n_pad, mode="edge")

        init = (
            jnp.array(-jnp.inf, dtype=jnp.float64),  # m
            jnp.array(0.0, dtype=jnp.float64),       # s1
            jnp.array(0.0, dtype=jnp.float64),       # s2
        )

        def body(carry, k):
            m, s1, s2 = carry
            start = k * B_use
            z0 = jnp.array(0, dtype=start.dtype)

            m1c = lax.dynamic_slice(m1p, (start,), (B_use,))
            m2c = lax.dynamic_slice(m2p, (start,), (B_use,))
            dLc = lax.dynamic_slice(dLp, (start,), (B_use,))
            spc = lax.dynamic_slice(spinsp, (start, z0), (B_use, spinsp.shape[1]))
            lpdc = lax.dynamic_slice(lpdp, (start,), (B_use,))
            lpic = lax.dynamic_slice(lpip, (start,), (B_use,))
            mc = lax.dynamic_slice(mask, (start,), (B_use,))

            x = _score_chunk(Lambda_, interp_mass_vals_, m1c, m2c, dLc, spc, lpdc, lpic, mc)

            m_chunk = jnp.max(x)
            m_new = jnp.maximum(m, m_chunk)

            scale1 = jnp.exp(m - m_new)
            u1 = jnp.exp(x - m_new)
            s1_new = s1 * scale1 + jnp.sum(u1)

            if not return_var:
                s2_new = jnp.zeros_like(s1_new)
            else:
                scale2 = jnp.exp(2.0 * (m - m_new))
                u2 = jnp.exp(2.0 * (x - m_new))
                s2_new = s2 * scale2 + jnp.sum(u2)

            return (m_new, s1_new, s2_new), None

        (m_fin, s1_fin, s2_fin), _ = lax.scan(body, init, jnp.arange(n_chunks, dtype=jnp.int32))

        lse1 = m_fin + jnp.log(s1_fin)
        logN = jnp.log(Ndraw_)
        log_mu = lse1 - logN

        if not return_var:
            return log_mu, bk.zeros_like(log_mu)
        else:
            lse2 = 2.0 * m_fin + jnp.log(s2_fin)
            logs2 = lse2 - logN
            var_u = logdiffexp_bk(bk, logs2 - 2.0 * log_mu, 0.0) - jnp.log(Ndraw_ - 1.0)
            return log_mu, var_u

    def _sel_fwd(
        Lambda_,
        interp_mass_vals_,
        m1_, m2_, dL_, spins_, lpd_, lpi_, Ndraw_,
    ):
        log_mu, var_u = _sel_fwd_only(
            Lambda_, interp_mass_vals_, m1_, m2_, dL_, spins_, lpd_, lpi_, Ndraw_
        )
        lse1 = log_mu + jnp.log(Ndraw_)
        # Save what backward needs
        res = (lse1, Ndraw_, Lambda_, interp_mass_vals_, m1_, m2_, dL_, spins_, lpd_, lpi_)
        return (log_mu, var_u), res

    def _sel_bwd(res, g):
        (lse1, Ndraw_, Lambda_, interp_mass_vals_, m1_, m2_, dL_, spins_, lpd_, lpi_) = res
        g_log_mu, g_var_u = g

        # do not differentiate var_u
        g_var_u = jnp.array(0.0, dtype=jnp.float64)

        n = m1_.shape[0]
        # robust: if B <= 0 or B >= n, avoid padding/chunking
        if (B is None) or (B <= 0) or (B >= n):
            n_chunks = 1
            n_pad = n
            B_use = n
        else:
            n_chunks = (n + B - 1) // B
            n_pad = n_chunks * B
            B_use = B

        mask = _make_mask(n, n_pad)

        m1p = _pad_to_multiple(m1_, n_pad, mode="edge")
        m2p = _pad_to_multiple(m2_, n_pad, mode="edge")
        dLp = _pad_to_multiple(dL_, n_pad, mode="edge")
        spinsp = _pad_to_multiple(spins_, n_pad, mode="edge")
        lpdp = _pad_to_multiple(lpd_, n_pad, mode="edge")
        lpip = _pad_to_multiple(lpi_, n_pad, mode="edge")

        dLambda_acc = jnp.zeros_like(Lambda_)
        dInterp_acc = _zeros_like_pytree(interp_mass_vals_)

        def body(carry, k):
            dLam_acc, dInt_acc = carry
            start = k * B_use
            z0 = jnp.array(0, dtype=start.dtype)

            m1c = lax.dynamic_slice(m1p, (start,), (B_use,))
            m2c = lax.dynamic_slice(m2p, (start,), (B_use,))
            dLc = lax.dynamic_slice(dLp, (start,), (B_use,))
            spc = lax.dynamic_slice(spinsp, (start, z0), (B_use, spinsp.shape[1]))
            lpdc = lax.dynamic_slice(lpdp, (start,), (B_use,))
            lpic = lax.dynamic_slice(lpip, (start,), (B_use,))
            mc = lax.dynamic_slice(mask, (start,), (B_use,))

            def score_wrapped(Lam_, interp_):
                return _score_chunk(Lam_, interp_, m1c, m2c, dLc, spc, lpdc, lpic, mc)

            # Differentiate w.r.t BOTH (Lambda, interp_mass_vals)
            x, pull = jax.vjp(score_wrapped, Lambda_, interp_mass_vals_)
            w = jnp.exp(x - lse1)
            cot = g_log_mu * w

            dLam_c, dInterp_c = pull(cot)
            return (dLam_acc + dLam_c, _add_pytrees(dInt_acc, dInterp_c)), None

        (dLambda_acc, dInterp_acc), _ = lax.scan(
            body,
            (dLambda_acc, dInterp_acc),
            jnp.arange(n_chunks, dtype=jnp.int32),
        )

        dLambda = dLambda_acc
        dInterp = dInterp_acc

        zeros_m1 = jnp.zeros_like(m1_)
        zeros_m2 = jnp.zeros_like(m2_)
        zeros_dL = jnp.zeros_like(dL_)
        zeros_sp = jnp.zeros_like(spins_)
        zeros_lpd = jnp.zeros_like(lpd_)
        zeros_lpi = jnp.zeros_like(lpi_)
        zeros_N = jnp.zeros_like(jnp.asarray(Ndraw_).reshape(()))

        # grads must match _sel_core arg order:
        # (Lambda, interp_mass_vals, m1, m2, dL, spins, lpd, lpi, Ndraw)
        return (dLambda, dInterp, zeros_m1, zeros_m2, zeros_dL, zeros_sp, zeros_lpd, zeros_lpi, zeros_N)

    _sel_core.defvjp(_sel_fwd, _sel_bwd)

    log_mu, var_u = _sel_core(
        Lambda,
        interp_mass_vals,
        m1inj, m2inj, dLinj, spinsInj, log_p_draw, log_p_incl,
        jnp.asarray(Ndraw).reshape(()),
    )
    return log_mu, var_u

    

# ---------------------------------------------------------------------
#  sel. bias wrapper
# ---------------------------------------------------------------------

def sel_bias_with_uncertainty(
    bk,
    m1inj,
    m2inj,
    dLinj,
    spinsInj,
    log_p_draw,
    log_p_incl,
    Lambda,
    Ndraw,
    *,
    rate_model,
    mass_model,
    spin_model,
    smoothing="LVK",
    simplex_repair=False,
    has_m2_break=False,
    norm_gauss="uplow",
    param="vanilla",
    z_grid=None,
    d_nodes = None, 
    verbose=False,
    subtract_log_p_incl=False,
    # new flags
    use_streaming_vjp: bool = True,
    sel_chunk_size: int = 65536,
    K_dp: int =30,
    DP_truncate=False,
    DP_m1_env=False,
    DP_m1_env_form="powerlaw",
    DP_m1_env_delta1=0.2,
    DP_m1_env_delta2=0.2,
    DP_q_env=False,
    DPUC_Primary_q_norm="conditional",
    DP_corr_m1m2=False,
    DP_corr_eta_rho = 2.0,
    DP_corr_f_minor = 0.25,
    PL_DPUC_normalize_gmm = False,
    DPUC_kernel="gaussian",
    DPUC_kernel_rcut=4.0,
    DPUC_kernel_dr=0.15,
    DPUC_kernel_xcut=3.0,
    DPUC_kernel_ycut=3.0,
    DPUC_q_coord="logitq",
    DPUC_TruncM2_shared_weights=False,
    DPUC_TruncM2_secondary_coord="logm2",
    interp_mass_vals=None,
    integrate_dc = 'trapz',
    return_var = True,
    interp_mass = False,
    gp_state=None,
    taper_kind = "sigmoid",   # or "power"
    taper_p = 12.0           # only used if taper_kind == "power"
):
    
   
    if not use_streaming_vjp:
        return sel_bias_with_uncertainty_legacy(
            bk,
            m1inj, m2inj, dLinj, spinsInj,
            log_p_draw, log_p_incl,
            Lambda, Ndraw,
            rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
            smoothing=smoothing, simplex_repair=simplex_repair,
            has_m2_break=has_m2_break, norm_gauss=norm_gauss, param=param, 
            z_grid=z_grid, 
            d_nodes = d_nodes, 
            verbose=verbose,
            subtract_log_p_incl=subtract_log_p_incl,
            K_dp=K_dp, 
            DP_truncate=DP_truncate,
            DP_m1_env=DP_m1_env,
            DP_m1_env_form=DP_m1_env_form,
            DP_m1_env_delta1=DP_m1_env_delta1,
            DP_m1_env_delta2=DP_m1_env_delta2,
            DP_q_env=DP_q_env,
            DPUC_Primary_q_norm=DPUC_Primary_q_norm,
            DP_corr_m1m2=DP_corr_m1m2,
            DP_corr_eta_rho = DP_corr_eta_rho,
            DP_corr_f_minor = DP_corr_f_minor,
            PL_DPUC_normalize_gmm = PL_DPUC_normalize_gmm,
            DPUC_kernel=DPUC_kernel,
            DPUC_kernel_rcut=DPUC_kernel_rcut,
            DPUC_kernel_dr=DPUC_kernel_dr,
        DPUC_kernel_xcut=DPUC_kernel_xcut,
        DPUC_kernel_ycut=DPUC_kernel_ycut,
        DPUC_q_coord=DPUC_q_coord,
            DPUC_TruncM2_shared_weights=bool(DPUC_TruncM2_shared_weights),
                DPUC_TruncM2_secondary_coord=DPUC_TruncM2_secondary_coord,
            interp_mass_vals=interp_mass_vals,
            integrate_dc = integrate_dc,
            return_var = return_var,
            gp_state=gp_state,
            taper_kind = taper_kind,
            taper_p = taper_p,
            
        )

    else:
        if interp_mass:
            
            return sel_bias_with_uncertainty_streaming_vjp(
                bk,
                m1inj, m2inj, dLinj, spinsInj,
                log_p_draw, log_p_incl,
                Lambda, Ndraw,
                rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
                smoothing=smoothing, simplex_repair=simplex_repair,
                has_m2_break=has_m2_break, norm_gauss=norm_gauss, param=param, 
                z_grid=z_grid,
                d_nodes = d_nodes, 
                verbose=verbose,
                chunk_size=sel_chunk_size,
                K_dp=K_dp,
                DP_truncate=DP_truncate,
                DP_m1_env=DP_m1_env,
                DP_m1_env_form=DP_m1_env_form,
            DP_m1_env_delta1=DP_m1_env_delta1,
            DP_m1_env_delta2=DP_m1_env_delta2,
            DP_q_env=DP_q_env,
            DPUC_Primary_q_norm=DPUC_Primary_q_norm,
                DP_corr_m1m2=DP_corr_m1m2,
            DP_corr_eta_rho = DP_corr_eta_rho,
            DP_corr_f_minor = DP_corr_f_minor,
                PL_DPUC_normalize_gmm = PL_DPUC_normalize_gmm,
                DPUC_kernel=DPUC_kernel,
                DPUC_kernel_rcut=DPUC_kernel_rcut,
                DPUC_kernel_dr=DPUC_kernel_dr,
        DPUC_kernel_xcut=DPUC_kernel_xcut,
        DPUC_kernel_ycut=DPUC_kernel_ycut,
        DPUC_q_coord=DPUC_q_coord,
                DPUC_TruncM2_shared_weights=bool(DPUC_TruncM2_shared_weights),
                DPUC_TruncM2_secondary_coord=DPUC_TruncM2_secondary_coord,
                interp_mass_vals=interp_mass_vals,
                integrate_dc = integrate_dc,
                return_var = return_var,
            gp_state=gp_state,
                taper_kind = taper_kind,
                   taper_p = taper_p,
            )

            
        else:
            return sel_bias_with_uncertainty_streaming_vjp_clean(
                bk,
                m1inj, m2inj, dLinj, spinsInj,
                log_p_draw, log_p_incl,
                Lambda, Ndraw,
                rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
                smoothing=smoothing, simplex_repair=simplex_repair,
                has_m2_break=has_m2_break, norm_gauss=norm_gauss, param=param, 
                z_grid=z_grid,
                d_nodes = d_nodes, 
                verbose=verbose,
                chunk_size=sel_chunk_size,
                K_dp=K_dp,
                DP_truncate=DP_truncate,
                DP_m1_env=DP_m1_env,
                DP_m1_env_form=DP_m1_env_form,
            DP_m1_env_delta1=DP_m1_env_delta1,
            DP_m1_env_delta2=DP_m1_env_delta2,
            DP_q_env=DP_q_env,
            DPUC_Primary_q_norm=DPUC_Primary_q_norm,
                DP_corr_m1m2=DP_corr_m1m2,
                DP_corr_eta_rho = DP_corr_eta_rho,
                DP_corr_f_minor = DP_corr_f_minor,
                PL_DPUC_normalize_gmm = PL_DPUC_normalize_gmm,
                DPUC_kernel=DPUC_kernel,
                DPUC_kernel_rcut=DPUC_kernel_rcut,
                DPUC_kernel_dr=DPUC_kernel_dr,
        DPUC_kernel_xcut=DPUC_kernel_xcut,
        DPUC_kernel_ycut=DPUC_kernel_ycut,
        DPUC_q_coord=DPUC_q_coord,
                DPUC_TruncM2_shared_weights=bool(DPUC_TruncM2_shared_weights),
                DPUC_TruncM2_secondary_coord=DPUC_TruncM2_secondary_coord,
                interp_mass_vals=interp_mass_vals,
                integrate_dc = integrate_dc,
                return_var = return_var,
            gp_state=gp_state,
                taper_kind = taper_kind,
                taper_p = taper_p,
            )






# ---------------------------------------------------------------------
#  custom vjp for z(dL)
# ---------------------------------------------------------------------



def make_dL_to_z_cuvjp(*, bk, eps_interp=1e-12, side_interp="left"):
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
        return atinterp(bk, dL, dL_grid, zgrid)

    def fwd(dL, dL_grid, zgrid, log_ddL_dz_grid, dL_dtheta_grid, Lambda):
        # Build dL-space bracketing once
        # i, t satisfy: dL_grid[i] <= dL < dL_grid[i+1] (depending on side)
        i, t = _interp_prepare_bk(bk, dL, dL_grid)

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
        return atinterp_uniform(bk, dL, dL_u, z_u, eps=eps_interp, side="left")

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
