from __future__ import annotations

# backend-agnostic pieces
from cosmology import Xi_vanilla, Xi_polexp, Efun, log_ddL_dz as log_ddL_dz_bk, z_from_dL, dLfun, log_dV_dz


from rate_models import log_p_z_MD_unnorm
from spin_models import logpdf_default_spin_gauss as logpdf_default_spin_gauss_bk
import mass_models
import spin_models
from pytensor_utils import logdiffexp as logdiffexp_bk, Mcq_from_m1m2, logit, log_sigmoid
#from pytensor_utils import logsumexp as _logsumexp

#from jax_utils import _interp_prepare_bk, _interp_apply_bk, _interp_apply_multi_bk, _interp_prepare_uniform_bk
from pytensor_utils import atinterp, atinterp_uniform, atcumtrapz, attrapzvec

import jax.numpy as jnp
from jax import lax

try:
    import jax    
except Exception as e:
    print(e)
    raise ValueError()



# ---------------------------------------------------------------------
#  utils
# ---------------------------------------------------------------------


def _zeros_like_tree(x):
    # works for arrays, scalars, and lists/tuples of arrays
    return jax.tree_util.tree_map(lambda a: jnp.zeros_like(a), x)


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
    if rate_model in ("MD", "DPUC-vol-MD"):
        # [gamma, kappa, zp]
        rate = Lambda_[i:i+3]
        i += 3
    elif rate_model == "PL":
        # [gamma]
        rate = Lambda_[i:i+1]
        i += 1
    elif rate_model in ("DPUC", "DPUC-vol"):
        # rate is modeled jointly with mass; nothing appended here
        rate = []
    else:
        raise ValueError(f"Unknown rate_model: {rate_model}")

    # -------------------------
    # Spin model
    # -------------------------
    if spin_model == "chieffchip":
        # [muEff, sigEff, muP, sigP, rho]
        spin = Lambda_[i:i+5]
        i += 5
    elif spin_model == "chieffchip_uc":
        # [muEff, sigEff, muP, sigP]
        spin = Lambda_[i:i+4]
        i += 4
    elif spin_model == "default":
        # [alphaChi, betaChi, zeta, sigmat]
        spin = Lambda_[i:i+4]
        i += 4
    elif spin_model == "default_gauss":
        # [muChi, sigmaChi, zeta, sigmat]
        spin = Lambda_[i:i+4]
        i += 4
    else:
        # "No model of the spin distribution."
        spin = []

    # -------------------------
    # Mass model
    # -------------------------
    if mass_model == "PLPreg":
        # [lamP, alpha, beta, deltam, ml, mh, muM, sM]
        mass = Lambda_[i:i+8]
        i += 8

    elif mass_model == "DPLDP" or mass_model == "PLDP":
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
    K_dp: int = 30, 
    DP_truncate=False,
    DP_m1_env=False,
    interp_mass = 0,
    integrate_dc = 'trapz',
    pop_only = False,
    stop_grad_var_u: bool = True,
    return_var = True,
    z_nodes = None
):
    """Build the single source of truth JAX core function.

    Returns a pure function:
        (evt arrays..., inj arrays..., Lambda, Ndraw) -> (logp_pop_evt, log_mu, var_u)
    """


    def _f(
        m1det, m2det, dLdet, spins_evt,
        m1inj, m2inj, dLinj, spins_inj, log_p_draw, log_p_incl,
        Lambda, Ndraw,
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
            m1det  = lax.stop_gradient(m1det)
            m2det = lax.stop_gradient(m2det)
            dLdet = lax.stop_gradient(dLdet)
            spins_evt = lax.stop_gradient(spins_evt)

  
        ##################################################
        # Obtain zs from distance-redshift inversion

        d_nodes = None
        #dLfun( bk, z_nodes, H0, Om, w0, Xi0, nXi0, dc=None, Xi=None, param=param, integrate_dc=integrate_dc )

        z_evt = z_from_dL(bk, dLdet, H0=H0, Om=Om, w0=w0, Xi0=Xi0, nXi0=nXi0, 
                          z_nodes = z_nodes, 
                          d_nodes = d_nodes, 
                            integrate_dc = integrate_dc) 

        
        onepz = 1.0 + z_evt
        m1src = m1det / onepz
        m2src = m2det / onepz


        ##################################################
        # Optional : pre-compute p(m1, m2 (z)) on grids
        # mandaory for DPLDP-z, not encouraged for other
        
        if interp_mass:
            
            # pre-computing mass function for later interpolation
            if mass_model=='DPLDP' or mass_model == "PLDP":
                
                _, _, _, mass_p = split_Lambda(Lambda, mass_model, rate_model, spin_model)


                alpha1_, alpha2_, mb_, mu1_, sigma1_, mu2_, sigma2_, m1_low_, m_high_, delta_m1_, lambda0_, lambda1_, lambda2_, beta_, m2_low_, delta_m2_, epsilon_, m_g_, w_g_, sig_g_l_, sig_g_h_ = mass_p

                eps_m = 1e-5
                n2 = 500
                n2_taper = 100
                
                m2_lo = m2_low_ + eps_m
                m2_taper_hi = m2_lo + bk.maximum(delta_m2_, 1e-6)
                
                u1 = bk.linspace(0.0, 1.0, n2_taper)
                
                eps_t = 1e-4
                t = bk.exp(bk.log(eps_t) * (1.0 - u1))     # eps_t -> 1
                t = (t - eps_t) / (1.0 - eps_t)            # -> [0,1]
                seg1 = m2_lo + (m2_taper_hi - m2_lo) * t
                
                u2 = bk.linspace(0.0, 1.0, n2 - n2_taper)
                seg2 = m2_taper_hi + (300.0 - m2_taper_hi) * u2
                
                m2_grid_ = bk.concatenate([seg1[:-1], seg2])
                


            
                m1_grid_ = mass_models.build_m1_grid_DPLDP( bk, 
                                            alpha1=alpha1_,
                                            alpha2=alpha2_,
                                            mb=mb_,
                                            mu1=mu1_,
                                            sigma1=sigma1_,
                                            mu2=mu2_,
                                            sigma2=sigma2_,
                                            m1_low=m1_low_,
                                            m_high=m_high_,
                                            delta_m1=delta_m1_,
                                            n_peak=interp_mass,      # or smaller if you want
                                            n_tail_low=interp_mass//5,
                                            n_tail_high=interp_mass//5,
                                            #k_sigma=4.0,
                                            n_taper=interp_mass//5,          # NEW: points inside [m1_low, m1_low+delta_m1]
                                            n_taper_eff=200.0,   # NEW: used for tie-only ramp scale
                                        )
                
                lp_m1_grid = mass_models.logpdfm1_DPLDP( bk, m1_grid_, alpha1_, alpha2_, mb_, mu1_, sigma1_, mu2_, sigma2_, m1_low_, m_high_, delta_m1_, lambda0_, lambda1_, lambda2_, epsilon_,  smoothing=smoothing, norm_gauss=norm_gauss) 


                lp_m2_grid = mass_models.logpdfm2_PLP_reg( bk, m2_grid_, beta_, delta_m2_, m2_low_, m_g=m_g_, w_g=w_g_, sig_g_low = sig_g_l_, sig_g_high = sig_g_h_, has_m2_break=has_m2_break, smoothing=smoothing ) 


                # CDF over m2
                cdf_m2 = atcumtrapz(bk, bk.exp(lp_m2_grid), m2_grid_)
                cdf_m2 = bk.clip(cdf_m2, 1e-300, np.inf)
                
                # CDF lives on m2_grid_[1:]
                m2_cdf_grid = m2_grid_[1:]
                logcdf_m2   = bk.log(cdf_m2)
                
                # C(m1) = CDF evaluated at m2=m1 (clipped into CDF grid support)
                mcap = bk.clip(m1_grid_, m2_cdf_grid[0], m2_cdf_grid[-1])
                
                # NON-UNIFORM interpolation (must match your test)
                lC_of_m1 = bk.interp(  mcap, m2_cdf_grid, logcdf_m2 )
                
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
                
                eps_m = 1e-5 
                n2 = 500
                n2_taper = 100
                
                m2_lo = m2_low + eps_m
                m2_taper_hi = m2_lo + bk.maximum(delta_m2 , 1e-6)
                
                u1 = bk.linspace(0.0, 1.0, n2_taper)
                
                eps_t = 1e-4
                t = bk.exp(bk.log(eps_t) * (1.0 - u1))     # eps_t -> 1
                t = (t - eps_t) / (1.0 - eps_t)            # -> [0,1]
                seg1 = m2_lo + (m2_taper_hi - m2_lo) * t
                
                u2 = bk.linspace(0.0, 1.0, n2 - n2_taper)
                seg2 = m2_taper_hi + (300.0 - m2_taper_hi) * u2
                
                m2_grid_ = bk.concatenate([seg1[:-1], seg2])

                m1_grid_ =  mass_models.build_m1_grid_DPLDP_z( bk, z_nodes,
                # low-z hyperparameters
                mu1_0, sigma1_0, mu2_0, sigma2_0, mb_0,
                # high-z (asymptotic) hyperparameters
                mu1_inf, sigma1_inf, mu2_inf, sigma2_inf, mb_inf,
                # evolution hyperparameters
                z_mu1, dz_mu1,
                z_sigma1, dz_sigma1,
                z_mu2, dz_mu2,
                z_sigma2, dz_sigma2,
                z_mb, dz_mb,
                # support for m1
                m1_low, m_high,
                delta_m1,
                # grid resolution controls
                n_peak=interp_mass,      # points in the "interesting" band (peaks + break)
                n_tail_low=interp_mass//5,   # points in low-mass tail
                n_tail_high=interp_mass//5,  # points in high-mass tail
                k_sigma=4.0,      #
                n_taper=interp_mass//5,  # points in low-mass tapering
                )


                # ---------
                # 1) m2 grids (depend on m2 params, but NOT on z in your current model)
                # ---------
                lp_m2_grid = mass_models.logpdfm2_PLP_reg( bk,
                    m2_grid_, beta , delta_m2 , m2_low ,
                    m_g=m_g, w_g=w_g,  sig_g_low=sig_g_low , sig_g_high=sig_g_high ,
                    has_m2_break=has_m2_break, smoothing=smoothing
                )  # shape (N2,)
            
                # lC_grid evaluated on m1_grid (shape (N1,))
                cdf_m2 = atcumtrapz(bk, bk.exp(lp_m2_grid), m2_grid_)
                cdf_m2 = bk.clip(cdf_m2, 1e-300, jnp.inf)

                # CDF lives on m2_grid_[1:]
                m2_cdf_grid = m2_grid_[1:]
                logcdf_m2   = bk.log(cdf_m2)
                
                # C(m1) = CDF evaluated at m2=m1 (clipped into CDF grid support)
                mcap = bk.clip(m1_grid_, m2_cdf_grid[0], m2_cdf_grid[-1])
                
                # NON-UNIFORM interpolation
                lC_of_m1 = bk.interp( mcap, m2_cdf_grid, logcdf_m2 )
                

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
                    *evo_params
                                                        ,
                    smoothing=smoothing,
                    simplex_repair=simplex_repair,
                    norm_gauss=norm_gauss
                )
                lp_m1_bank = bk.clip( lp_flat, -1e30, 1e030 ).reshape((K, N1)) # (K,N1)


                lp_max = bk.max(lp_m1_bank, axis=1, keepdims=True)          # (K,1)
                p_shift = bk.exp(lp_m1_bank - lp_max)                       # safe exp
                I = attrapzvec(bk, p_shift, m1_grid_[None, :], axis=1)            # (K,)
                I = bk.clip(I, 1e-300, jnp.inf)
                ln_bank = bk.log(I) + lp_max[:, 0]
             
                # Pack for later use (include z_bank)
                interp_vals_mass  = [lp_m1_bank, lp_m2_grid, lC_of_m1, ln_bank, ]
                interp_grids_mass = [m1_grid_, m2_grid_, z_nodes]
                

            
            else:
                raise NotImplementedError()

            interp_mass_vals = ( interp_grids_mass, interp_vals_mass )
        
        else:
            
            interp_mass_vals = None


        ##################################################
        # Compute log_p_pop

        logp_pop_evt = log_p_pop(
            bk,
            m1src, m2src, z_evt, dLdet, spin_models._spins_as_list(spins_evt, spin_model), 
            Lambda,
            rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
                smoothing=smoothing, simplex_repair=simplex_repair,
                has_m2_break=has_m2_break, norm_gauss=norm_gauss,
                dc=None, 
                log_ddL_dz_pre=None,
                Xi=None, E=None, 
                param=param,
                verbose=verbose,
                K_dp=K_dp, 
                DP_truncate=DP_truncate,
                DP_m1_env=DP_m1_env,
                interp_mass_vals = interp_mass_vals
        )

        if skip_sel:
             return (
            logp_pop_evt,
            jnp.asarray(0., dtype=jnp.float64).reshape(()),
            jnp.asarray(0., dtype=jnp.float64).reshape(()),
        )
            
            
        ##################################################
        # Compute sel. bias and its variance
        
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
            use_streaming_vjp= bool(chunk_inj>0),          # <--- enable optimized backward
            sel_chunk_size=chunk_inj,            # <--- tune
            K_dp=K_dp,
            DP_truncate=DP_truncate,
            DP_m1_env=DP_m1_env,
            interp_mass_vals = interp_mass_vals,
            integrate_dc = integrate_dc,
            return_var = return_var
            
        )
        if stop_grad_var_u:
            var_u = lax.stop_gradient(var_u)

        return (
            logp_pop_evt,
            jnp.asarray(log_mu, dtype=jnp.float64).reshape(()),
            jnp.asarray(var_u, dtype=jnp.float64).reshape(()),
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
    interp_mass_vals=  None,
):
    """
    Backend-agnostic log_p_pop
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
        E = Efun(bk, z, Om, w0)

    # -----------------------
    # rate model (MD only)
    # -----------------------
    log_one_p_z = bk.log1p(z)
    
    if rate_model == "MD" or rate_model == "DPUC-vol-MD":
        
        gamma, kappa, zp = Lambda[5], Lambda[6], Lambda[7]
        lpz = log_p_z_MD_unnorm(bk, z, gamma, kappa, zp, H0, Om, w0, dc=dc, E=E)
        istart = 8
        z_dpuc = None
        
    elif rate_model=='DPUC':
        
        lpz = log_one_p_z
        
        istart = 5

    elif rate_model=='DPUC-vol':
        
        lpz = log_dV_dz(bk, z, H0, Om, w0, dc=dc, E=E )
        
        istart = 5

    else:
        raise ValueError(f"Unknown rate_model: {rate_model}")

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
    if mass_model == "DPLDP" or mass_model == "PLDP":

        if interp_mass_vals is not None:
            
            lpmass = mass_models.logpdf_DPLDPfrom_interp(
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
            )
    
    # DPLDP-z
    elif mass_model == "DPLDP-z":


        if interp_mass_vals is not None:
            
            lpmass = mass_models.logpdf_DPLDP_z_from_interp(
                    bk, (m1s, m2s), z, interp_mass_vals)
            
        else:
            # ------------------------------------------------------------
            # UNPACK low-z mass hyperparameters (same 20 as non-evolving)
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
                    norm_gauss=norm_gauss
                )
            


    # DPUC   
    elif mass_model=='DPUC':


        D = 3 if rate_model in ("DPUC", "DPUC-vol", "DPUC-vol-MD") else 2
        K = K_dp
        
        j = int(istart_spin)
        D = int(D)
        K = int(K)

        mu   = Lambda[j : j + D*K].reshape((D, K)); j += int(D*K)
        sd   = Lambda[j : j + D*K].reshape((D, K)); j += int(D*K)
        logw = Lambda[j : j + K];                  j += K
        mmin = Lambda[j];                          j += 1
        mmax = Lambda[j];                          j += 1
        alpha = Lambda[j];                          j += 1


        Mc_src, q = Mcq_from_m1m2( m1s, m2s )
        logMc_src, logit_q = bk.log( bk.maximum(Mc_src, 1e-10)), logit(bk, q)
            
        logp1, logp2, logp3 = mass_models.gaussian_logpdf_pair( bk, logMc_src, logit_q, mu, sd, z=log_one_p_z )

        
        # Mixture over components → (n_obs,)
        lpmass = bk.logsumexp(logp1 + logp2 + logp3 + logw[:, None], axis=0, )


        # remove jacobian m1, m2 --> log(Mc), logit(q)
        lpmass += ( - bk.log(m2s) 
                      - bk.log(bk.maximum( m1s - m2s, 1e-10
                                         ) )
                     )
        if rate_model in ('DPUC','DPUC-vol', 'DPUC-vol-MD'):
                lpmass -= log_one_p_z

        
        if DP_m1_env:
            lpmass += -alpha * bk.log(m1s)

            
        if DP_truncate:
            # truncate low and up and renormalize

            s_logm = 0.003
            log_gate = (
                log_sigmoid(bk, bk.log(m2s), bk.log(mmin), s_logm) +
                log_sigmoid(bk, bk.log(mmax) - bk.log(m1s), 0.0, s_logm)
            )
            lpmass += log_gate
            
            #inside = (m2Src >= mmin) & (m1Src <= mmax) & (m1Src >= mmin) & (m2Src <= mmax)
            #log_p_pop = bk.switch(inside, log_p_pop, -np.inf)

            logZ, Zk = mass_models.mixture_logZ_physical_vectorized( bk,
                    mux=mu[0], sdx=sd[0],
                    muy=mu[1], sdy=sd[1],
                    logw=logw,
                    mmin=mmin, mmax=mmax
                )
            
            lpmass -= logZ



    else:
        raise ValueError(f"Unknown mass_model: {mass_model}")

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
    interp_mass_vals=None,
    integrate_dc = 'trapz', 
    return_var = True

):
    """
    Single canonical selection-bias function used by both forward and VJP.

    If zinj/dcinj/log_ddL_dz_inj are provided, internal inversions/interps are skipped.
    """

    # ---- compute zinj by interpolation ----

    H0, Om, w0, Xi0, nXi0 = Lambda[0], Lambda[1], Lambda[2], Lambda[3], Lambda[4]
    
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
        dc=None,
        log_ddL_dz_pre=None,
        Xi=None,
        E=None,
        param=param,
        verbose=verbose,
        K_dp=K_dp, 
        DP_truncate=DP_truncate,
        DP_m1_env = DP_m1_env,
        interp_mass_vals=interp_mass_vals
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
    z_grid=None,                # kept for API compatibility; unused now
    d_nodes = None, 
    verbose=False,
    chunk_size: int = 65536,
    K_dp: int = 30,
    DP_truncate=False,
    DP_m1_env=False,
    interp_mass_vals=None,
    integrate_dc="trapz",
    return_var = True
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
            interp_mass_vals=interp_mass_vals,
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
    interp_mass_vals=None,
    integrate_dc = 'trapz',
    return_var = True
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
            interp_mass_vals=interp_mass_vals,
            integrate_dc = integrate_dc,
            return_var = return_var
            
        )


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
        interp_mass_vals=interp_mass_vals,
        integrate_dc = integrate_dc,
        return_var = return_var
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


