#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import pytensor_tools as atools
import pytensor_utils as putils
import pytensor.tensor as at
import pytensor
import pymc as pm
import numpy as np
from pytensor.gradient import disconnected_grad as stop_grad
from pytensor.compile.mode import get_default_mode
from pymc.distributions import transforms as tr
#from pymc.pytensorf import collect_default_updates
from pytensor import config
import h5py

PLPeakO3params = {'H0': 67.66, 'Om':0.31, 'w0':-1, 'Xi0': 1, 'nXi0':0}


#####################################################
#####################################################


import functools

# OpFromGraph moved around across versions; try the modern path first
try:
    from pytensor.compile.builders import OpFromGraph
except Exception:  # fallback for older layouts
    from pytensor.compile.builders import op_from_graph as OpFromGraph  # type: ignore


@functools.lru_cache(maxsize=None)
def _make_log_p_pop_ofg_pure(
    dtype: str,
    rate_model: str,
    mass_model: str,
    spin_model: str,
    smoothing: str,
    has_m2_break: bool,
    spin_arity: int,
    with_dc: bool,
    with_logddldz: bool,   # <<< NEW flag
):
    # Eventwise args are 1-D vectors by contract of the OpFromGraph
    m1s    = at.vector('m1s',    dtype=dtype)
    m2s    = at.vector('m2s',    dtype=dtype)
    z      = at.vector('z',      dtype=dtype)
    dL     = at.vector('dL',     dtype=dtype)
    Lambda = at.vector('Lambda', dtype=dtype)
    inputs = [m1s, m2s, z, dL, Lambda]

    spins_syms = []
    if spin_arity == 4:
        s1  = at.vector('s1',  dtype=dtype)
        s2  = at.vector('s2',  dtype=dtype)
        ct1 = at.vector('ct1', dtype=dtype)
        ct2 = at.vector('ct2', dtype=dtype)
        spins_syms = [s1, s2, ct1, ct2]
        inputs += spins_syms

    dc_sym = None
    if with_dc:
        dc_sym = at.vector('dc', dtype=dtype)
        inputs.append(dc_sym)

    logdd_sym = None
    if with_logddldz:
        logdd_sym = at.vector('log_ddL_dz', dtype=dtype)
        inputs.append(logdd_sym)

    out = log_p_pop_at(
        m1s, m2s, z, dL, spins_syms, Lambda,
        rate_model, mass_model, spin_model,
        smoothing=smoothing, has_m2_break=has_m2_break,
        dc=dc_sym, log_ddL_dz_pre=logdd_sym
    )

    ofg = OpFromGraph(
        inputs, [out] if not isinstance(out, (list, tuple)) else list(out),
        inline=False,
        on_unused_input="ignore",
        mode=get_default_mode(),   # usually honors your global fast_run
        # If your PyTensor version supports it, uncomment for tiny runtime gain:
        trust_inputs=True,
    )
    return ofg


def log_p_pop_at_wrap(
    m1s, m2s, z, dL, spins, Lambda,
    rate_model, mass_model, spin_model,
    smoothing='LVK',
    has_m2_break=False,
    dc=None,
    log_ddL_dz_pre=None  # <<< NEW arg
):
    """
    Wrapper that accepts *any* eventwise shape for (m1s, m2s, z, dL, [spins], [dc], [log_ddL_dz_pre]),
    flattens them to vectors to satisfy the OpFromGraph contract, calls it, and reshapes outputs back.
    """

    # --- helpers -------------------------------------------------------------
    def _as_var(x):
        return x if isinstance(x, at.Variable) else at.as_tensor_variable(x)

    # reference eventwise shape: use z if available, else m1s
    z_var  = _as_var(z)
    m1_var = _as_var(m1s)
    ref    = z_var if z_var.ndim >= 1 else m1_var
    ref_shape = at.shape(ref)
    nflat = at.prod(ref_shape)  # total number of eventwise elements

    def _to_vec_like(x, cast_dtype=None):
        x = _as_var(x)
        if cast_dtype is not None and x.dtype != cast_dtype:
            x = x.astype(cast_dtype)
        # flatten to 1-D vector (OpFromGraph expects vectors)
        if x.ndim == 0:
            x = at.full((nflat,), x, dtype=x.dtype)
        elif x.ndim != 1:
            x = at.reshape(x, (nflat,))
        return x

    # robust dtype for eventwise tensors
    work_dtype = getattr(m1_var, "dtype", None) or getattr(z_var, "dtype", None) or "float64"

    # --- figure out arity / flags -------------------------------------------
    spin_arity = 4 if (isinstance(spins, (list, tuple)) and len(spins) == 4) else 0
    with_dc    = dc is not None
    with_logdd = log_ddL_dz_pre is not None

    # --- build / fetch the OpFromGraph --------------------------------------
    ofg = _make_log_p_pop_ofg_pure(
        dtype=work_dtype,
        rate_model=rate_model,
        mass_model=mass_model,
        spin_model=spin_model,
        smoothing=smoothing,
        has_m2_break=bool(has_m2_break),
        spin_arity=spin_arity,
        with_dc=with_dc,
        with_logddldz=with_logdd,
    )

    # --- flatten all eventwise inputs to vectors ----------------------------
    m1v = _to_vec_like(m1_var,    cast_dtype=work_dtype)
    m2v = _to_vec_like(_as_var(m2s), cast_dtype=work_dtype)
    zv  = _to_vec_like(z_var,     cast_dtype=work_dtype)
    dLv = _to_vec_like(_as_var(dL), cast_dtype=work_dtype)

    args = [m1v, m2v, zv, dLv, _as_var(Lambda).astype(work_dtype)]

    if spin_arity == 4:
        s1, s2, ct1, ct2 = spins
        args += [
            _to_vec_like(_as_var(s1),  work_dtype),
            _to_vec_like(_as_var(s2),  work_dtype),
            _to_vec_like(_as_var(ct1), work_dtype),
            _to_vec_like(_as_var(ct2), work_dtype),
        ]

    if with_dc:
        args.append(_to_vec_like(_as_var(dc), work_dtype))

    if with_logdd:
        args.append(_to_vec_like(_as_var(log_ddL_dz_pre), work_dtype))

    # --- call ofg ------------------------------------------------------------
    out = ofg(*args)

    # ofg returns a list of outputs (because we wrapped [out] when constructing)
    if isinstance(out, (list, tuple)):
        reshaped = []
        for item in out:
            # reshape eventwise outputs back; keep scalars as-is
            if isinstance(item, at.Variable) and item.ndim >= 1:
                reshaped.append(at.reshape(item, ref_shape))
            else:
                reshaped.append(item)
        # If it’s a single-output op, keep old behavior of returning a tensor
        return reshaped[0] if len(reshaped) == 1 else tuple(reshaped)
    else:
        # Safety: if somehow a bare tensor was returned, reshape if needed
        return at.reshape(out, ref_shape) if (isinstance(out, at.Variable) and out.ndim >= 1) else out





def log_p_pop_at(m1s, m2s, z, dL, spins,
                 Lambda, 
                 rate_model, mass_model, spin_model, 
                 smoothing='LVK', 
                 simplex_repair=False,
                 has_m2_break=False, 
                 dc=None, 
                 log_ddL_dz_pre=None,
                 param='vanilla',
                 interp_vals_mass = None,
                 interp_grids_mass = None,
                 is_observed = False,
                 z_grid = None,
                 verbose=False
                ):


    ###################################
    # get parameters and compute log p_pop
    ####################################

    if 'BNS' not in mass_model:
        in_support = (m1s >= 3.0) & (m2s >= 3.0) & (m2s <= m1s)
    
    #was: H0, Om, w0, Xi0, n = Lambda[:5] 
    H0, Om, w0, Xi0, n = Lambda[0], Lambda[1], Lambda[2], Lambda[3], Lambda[4]

    if verbose:
        print(" H0, Om, w0, Xi0, n ")
        print( H0.eval(), Om.eval(), w0.eval(), Xi0.eval(), n.eval() )

    if dc is None:
        if param=='vanilla':
            Xi = atools.Xifun_at(z, Xi0, n)
        elif param=='polexp':
            Xi = atools.Xifun_at_polexp(z, Xi0, n)
        dc = dL/(1+z)/Xi #atools.dcfun_at(z, H0, Om, w0, interp=False)


    ##################################
    # redshift 
    
    if rate_model=='MD':
        
        gamma, kappa, zp = Lambda[5], Lambda[6], Lambda[7] #Lambda[5:8]
        lpz = atools.log_p_z_MD_unnorm(z, gamma, kappa, zp, H0, Om, w0, dc=dc )
        z_dpuc = None
        istart = 8
        if verbose:
            print("  gamma, kappa, zp ")
            print(  gamma.eval(), kappa.eval(), zp.eval() )
        
    elif rate_model=='PL':
        
        gamma = Lambda[5]
        lpz = atools.log_p_z_PL_unnorm(z, gamma, H0, Om, w0, dc=dc )
        z_dpuc = None
        istart = 6

    elif rate_model=='DPUC':

        z_dpuc = at.log1p(z)
        
        lpz = at.zeros(z.shape) #atools.log_dV_dz_at(z, H0, Om, w0, dc=dc ) #-z_dpuc
        
        istart = 5

    elif rate_model=='DPUC-vol':

        z_dpuc = at.log1p(z)
        
        lpz = atools.log_dV_dz_at(z, H0, Om, w0, dc=dc ) - z_dpuc
        
        istart = 5
        

    # ##################################
    # spin
    
    if spin_model=='chieffchip':
        
        #muE, sigE, muP, sigP, rho = Lambda[istart],Lambda[istart+1], Lambda[istart+2], Lambda[istart+3], Lambda[istart+4] #was: Lambda[istart:istart+5]
        muE   = Lambda[istart + 0]
        sigE  = Lambda[istart + 1]
        muP   = Lambda[istart + 2]
        sigP  = Lambda[istart + 3]
        rho   = Lambda[istart + 4]
        chieff, chip = spins[0], spins[1]

        lpspin = atools.logpdf_multivariate_trunc_2D(  chieff, chip, muE, muP, sigE, sigP, rho,
                                                     at.as_tensor_variable(-1.), at.as_tensor_variable(1.), 
                                                     at.as_tensor_variable(0.), at.as_tensor_variable(1.) 
                                                    )
        istart_spin = istart + 5

    elif spin_model=='chieffchip_uc':
        
        #muE, sigE, muP, sigP = Lambda[istart],Lambda[istart+1], Lambda[istart+2], Lambda[istart+3] # was: Lambda[istart:istart+4]
        muE   = Lambda[istart + 0]
        sigE  = Lambda[istart + 1]
        muP   = Lambda[istart + 2]
        sigP  = Lambda[istart + 3]
        chieff, chip = spins[0], spins[1]

        lpchie = atools.truncGausslowerupper_at_lpdf(chieff, muE, sigE, xmin=at.as_tensor_variable(-1), xmax=at.as_tensor_variable(1))
        lpchip = atools.truncGausslowerupper_at_lpdf(chip, muP, sigP, xmin=at.as_tensor_variable(0), xmax=at.as_tensor_variable(1))

        lpspin = lpchie+lpchip
        istart_spin = istart+4

    elif spin_model=='default':

        #alphaChi, betaChi, zeta, sigmat = Lambda[istart],Lambda[istart+1], Lambda[istart+2], Lambda[istart+3]#Lambda[istart:istart+4]
        alphaChi = Lambda[istart + 0]
        betaChi  = Lambda[istart + 1]
        zeta     = Lambda[istart + 2]
        sigmat   = Lambda[istart + 3]
        lpspin = atools.logpdf_default_spin(spins, [alphaChi, betaChi, zeta, sigmat])
        istart_spin = istart+4
    
    elif spin_model=='default_gauss':
        #muChi, sigmaChi, zeta, sigmat = Lambda[istart],Lambda[istart+1], Lambda[istart+2], Lambda[istart+3] #Lambda[istart:istart+4]
        muChi    = Lambda[istart + 0]
        sigmaChi = Lambda[istart + 1]
        zeta     = Lambda[istart + 2]
        sigmat   = Lambda[istart + 3]
        lpspin = atools.logpdf_default_spin_gauss(spins, [muChi, sigmaChi, zeta, sigmat])
        istart_spin = istart+4

        if verbose:
            print(" muChi, sigmaChi, zeta, sigmat ")
            print(  muChi.eval(), sigmaChi.eval(), zeta.eval(), sigmat.eval() )
   
    else:
        lpspin = at.zeros( z.shape )
        istart_spin = istart

    
    ###################################
    # mass

    ### BBH
    if mass_model=='PLPreg':
        
        #lp, al, bb, dm, ml, mh, muM, sM = Lambda[istart_spin], Lambda[istart_spin+1], Lambda[istart_spin+2], Lambda[istart_spin+3], Lambda[istart_spin+4], Lambda[istart_spin+5], Lambda[istart_spin+6], Lambda[istart_spin+7] #Lambda[-8:]
        lp  = Lambda[istart_spin + 0]
        al   = Lambda[istart_spin + 1]
        bb   = Lambda[istart_spin + 2]
        dm   = Lambda[istart_spin + 3]
        ml   = Lambda[istart_spin + 4]
        mh   = Lambda[istart_spin + 5]
        muM  = Lambda[istart_spin + 6]
        sM   = Lambda[istart_spin + 7]
        lpmass = atools.logpdf_PLP_reg([m1s, m2s], [lp, al, bb, dm, ml, mh, muM, sM], smoothing=smoothing)

        # print("mass params check: [lp, al, bb, dm, ml, mh, muM, sM]")
        # print( [v.eval() for v in [lp, al, bb, dm, ml, mh, muM, sM]] )

        # print("mass params check from previous: [lp, al, bb, dm, ml, mh, muM, sM]")
        # print( [v.eval() for v in Lambda[-8:] ] )

    elif mass_model=='DPLDP':
        
        #x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20 = Lambda[-20:]
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

        lambdaBBHmass = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20]

        if interp_vals_mass is not None:
            print("Log p pop will use pre-computed mass function grid")
            lpmass = atools.logpdf_DPLDP_from_interp([m1s, m2s], interp_vals_mass, interp_grids_mass)
        else:
            lpmass = atools.logpdf_DPLDP([m1s, m2s], lambdaBBHmass, force_m2_less_than_m1=False, has_m2_break=has_m2_break, smoothing=smoothing, interp_vals=None, interp_grids = None )


        if verbose:
            print("alpha1","alpha2","mb","mu1","sigma1","mu2","sigma2", "m1_low","m_high","delta_m1", "lambda0","lambda1", "beta","m2_low","delta_m2","epsilon","mu_g","w_g", "sig_g_low","sig_g_high",)
            print( [x_.eval() for x_ in lambdaBBHmass] )


    elif mass_model == "DPLDP-z":
    
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
    
        lambdaBBHmass_lowz = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                              x11, x12, x13, x14, x15, x16, x17, x18, x19, x20]
    
        # ------------------------------------------------------------
        # UNPACK evolution hyperparameters (27 scalars):
        #   (theta_inf, z_theta, dz_theta) for:
        #    alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2,
        #    lambda0, lambda1
        # ------------------------------------------------------------
        j = istart_spin + 20
    
        alpha1_inf  = Lambda[j +  0]; z_alpha1  = Lambda[j +  1]; dz_alpha1  = Lambda[j +  2]
        alpha2_inf  = Lambda[j +  3]; z_alpha2  = Lambda[j +  4]; dz_alpha2  = Lambda[j +  5]
        mb_inf      = Lambda[j +  6]; z_mb      = Lambda[j +  7]; dz_mb      = Lambda[j +  8]
        mu1_inf     = Lambda[j +  9]; z_mu1     = Lambda[j + 10]; dz_mu1     = Lambda[j + 11]
        sigma1_inf  = Lambda[j + 12]; z_sigma1  = Lambda[j + 13]; dz_sigma1  = Lambda[j + 14]
        mu2_inf     = Lambda[j + 15]; z_mu2     = Lambda[j + 16]; dz_mu2     = Lambda[j + 17]
        sigma2_inf  = Lambda[j + 18]; z_sigma2  = Lambda[j + 19]; dz_sigma2  = Lambda[j + 20]
        lambda0_inf = Lambda[j + 21]; z_lambda0 = Lambda[j + 22]; dz_lambda0 = Lambda[j + 23]
        lambda1_inf = Lambda[j + 24]; z_lambda1 = Lambda[j + 25]; dz_lambda1 = Lambda[j + 26]
    
        evo_params = [
            alpha1_inf,  z_alpha1,  dz_alpha1,
            alpha2_inf,  z_alpha2,  dz_alpha2,
            mb_inf,      z_mb,      dz_mb,
            mu1_inf,     z_mu1,     dz_mu1,
            sigma1_inf,  z_sigma1,  dz_sigma1,
            mu2_inf,     z_mu2,     dz_mu2,
            sigma2_inf,  z_sigma2,  dz_sigma2,
            lambda0_inf, z_lambda0, dz_lambda0,
            lambda1_inf, z_lambda1, dz_lambda1,
        ]
    
        # ------------------------------------------------------------
        # Call the redshift-evolving mass pdf
        # ------------------------------------------------------------
        if interp_vals_mass is not None:
            lpmass = atools.logpdf_DPLDP_z_from_interp(
                    (m1s, m2s), z,                 
                    interp_vals_mass, interp_grids_mass,
                    force_m2_less_than_m1=False
                )
            print("lpmass from interp")
            print(lpmass.eval())
        else:
            lpmass = atools.logpdf_DPLDP_z(
                (m1s, m2s), z,                     
                lambdaBBHmass_lowz,
                evo_params,
                force_m2_less_than_m1=False,
                has_m2_break=has_m2_break,
                smoothing=smoothing,
                interp_vals=None,
                interp_grids=None,
                simplex_repair=simplex_repair
            )
            
            
        
    ### BNS
    elif mass_model=='BNSgauss':
        muM, sM = Lambda[istart_spin], Lambda[istart_spin+1] #Lambda[-2:]
        lpmass = atools.logpdf_gauss([m1s, m2s], [muM, sM] )
        
    elif mass_model=='BNSgaussCond':
        muM, sM = Lambda[istart_spin], Lambda[istart_spin+1] #Lambda[-2:]
        lpmass = atools.logpdf_gauss_cond([m1s, m2s], [muM, sM] )

    ### Non - parametric
    elif mass_model=='DPUC':

        w, mu, sd, logw  = Lambda[istart_spin], Lambda[istart_spin+1], Lambda[istart_spin+2], Lambda[istart_spin+3] #Lambda[-5:-1]
            
        
        Nmax = Lambda[istart_spin+4]

        if interp_vals_mass is None:
            
            logp1, logp2, logp3 = atools.gaussian_logpdf_pair( m1s, m2s, mu, sd, z=z_dpuc )
                
        else:
            logp1, logp2, logp3 = atools.gaussian_logpdf_pair_from_interp( [m1s, m2s], interp_vals_mass, interp_grids_mass, z = z_dpuc )
    
        
        if rate_model in ('PL', 'MD'):
            logp_components = logp1 + logp2                    # (K,N)
        else:
            logp_components = logp1 + logp2 + logp3                   # (K,N)

        
        # Mixture over components → (n_obs,)
        #lpmass = at.logsumexp(logp_components + logw[:, None], axis=0)

        
        lpmass = atools.safe_logsumexp(logp_components + logw[:, None], axis=0)

        if rate_model=='DPUC-vol' and is_observed:
            print("Normalize GMM x p(z)")
            log_Nz = atools.redshift_mixture_log_norm( mu=mu, sd=sd, logw=logw, y_min=at.log(1.+at.min(z_grid)), y_max=at.log(1.+at.max(z_grid)),  H0=H0, Om=Om, w0=w0, Ny=2000 )
        elif rate_model=='MD' and is_observed:
            log_Nz = atools.N_per_year( gamma, kappa, zp, H0, Om, w0, R0=1., dc=None, z_max = 100, res=1000)
        elif rate_model=='PL' and is_observed:
            raise NotImplementedError()
        else:
             log_Nz = at.zeros(m1s.shape)

        lpmass -= log_Nz 

    
    elif mass_model=='DP':

        alpha, beta, w, mu, fishers, ldets_inv, logw  = Lambda[istart_spin], Lambda[istart_spin+1], Lambda[istart_spin+2], Lambda[istart_spin+3], Lambda[istart_spin+4], Lambda[istart_spin+5] , Lambda[istart_spin+6] #Lambda[-8:-1]
        Nmax=Lambda[istart_spin+7]

        # 1) Pack observations into (N, 2)
        X = at.stack([m1s, m2s], axis=1)          # (N, 2)
        
        # 2) Differences to component means -> (K, N, 2)
        mu_k2 = mu.T                               # (K, 2)
        diff  = X[None, :, :] - mu_k2[:, None, :]  # (K, N, 2)
        
        # 3) Quadratic form (x-μ)^T Σ^{-1} (x-μ) for all (k, n)
        #    Using batched matmul; result tmp is (K, N, 2), then rowwise dot with diff
        tmp  = at.matmul(diff, fishers)            # (K, N, 2)
        quad = at.sum(diff * tmp, axis=2)          # (K, N)
        
        # 4) Component log-densities (MvN with precision)
        nd = 2
        logp_components = (
            -0.5 * quad
            - 0.5 * nd * atools.safe_log(2.0 * np.pi)
            + 0.5 * ldets_inv[:, None]
            + logw[:, None]
        )                                           # (K, N)

        # # 2a) Solve L * y = diff^T  for each component k
        # #    diff.transpose -> (K, 2, N); solve_lower_triangular acts per-k
        # y = at.solve_lower_triangular(L, diff.transpose(0, 2, 1))  # (K, 2, N)
        
        # # 3) Mahalanobis term: ||y||^2  → (K, N)
        # quad = at.sum(y**2, axis=1).T  # sum over the 2 dims, then transpose to (K, N)
        
        # # 3a) log |Σ^{-1}|  from L:  log|Σ| = 2 * sum(log(diag(L)))  ⇒ log|Σ^{-1}| = -2 * ...
        # logdet_prec = -2.0 * at.sum(atools.safe_log(at.diagonal(L, axis1=1, axis2=2)), axis=1)  # (K,)
        
        # # 4) Component log-densities (d=2)
        # logp_components = (
        #     -0.5 * quad
        #     - 0.5 * 2 * atools.safe_log(2.0 * np.pi)
        #     + 0.5 * logdet_prec[:, None]
        #     + logw[:, None]
        # )  # (K, N)

        # 5) Mixture over components -> per-observation log-lik
        lpmass = at.logsumexp(logp_components, axis=0)  # (N,)

    else:
        raise ValueError(f"Unknown mass_model: {mass_model}")
        
    ###################################
    # jacobian  

    #if rate_model in ('MD', 'PL'):
        
    if log_ddL_dz_pre is None:
        log_dthD_dth = atools.log_ddL_dz( z, H0, Om, w0, Xi0, n, dc=dc, param=param )
    else:
        log_dthD_dth = log_ddL_dz_pre
        
    log_dthD_dth += 2*at.log1p(z)
        
    #else:
    #    log_dthD_dth = at.zeros(z.shape)
    
    ###################################
    # return log pdf
    ####################################
    
    lp =  lpz - log_dthD_dth  + lpmass + lpspin

    MIN = at.as_tensor_variable(-1e30,  dtype=z.dtype)
    
    return lp #at.where(in_support, lp, MIN)


#####################################################



def sel_bias_with_uncertainty_at_0_batched_scan(
    m1inj, m2inj, dLinj, spinsInj, log_p_draw,
    Lambda, Ndraw,
    rate_model, mass_model, spin_model,
    smoothing, has_m2_break, interp,
    wrap_logp=False,
    # kept for API compat (ignored in this variant)
    log_ddL_dz_inj=None,
    zinj=None,
    dcinj=None,
    # grids ONLY for dL->z
    dL_grid=None,               # 1-D, increasing in dL
    z_grid=None,                # 1-D, z(dL_grid)
    dc_grid=None,               # UNUSED in this variant
    log_ddL_dz_grid=None,       # UNUSED in this variant
    *,
    chunk_size=4096,
    param='vanilla',
    **kwargs
):
    """Scan version that only interpolates z(dL) from (dL_grid,z_grid) and
    computes dc(z), log_ddL_dz(z) analytically — matching yesterday's behavior.
    """

    def _as_at(x):
        return x if isinstance(x, at.Variable) else at.as_tensor_variable(x)

    def _pad_to_multiple(x, k, pad_value):
        x = _as_at(x)
        if x.ndim != 1:
            x = at.flatten(x, 1)
        N = x.shape[0]
        C = (N + k - 1) // k
        Npad = C * k - N
        pad = at.full((Npad,), at.as_tensor_variable(pad_value, dtype=x.dtype), dtype=x.dtype)
        xK = at.concatenate([x, pad], axis=0).reshape((C, k))
        return xK, C, N

    def _combine_logsumexp(m_s, s_s, m_c, s_c):
        m_new = at.maximum(m_s, m_c)
        s_new = s_s * at.exp(m_s - m_new) + s_c * at.exp(m_c - m_new)
        return m_new, s_new

    # tensors
    m1_all   = _as_at(m1inj)
    m2_all   = _as_at(m2inj)
    dL_all   = _as_at(dLinj)
    lpd_all  = _as_at(log_p_draw)
    Lambda_t = _as_at(Lambda)

    # unpack cosmology
    H0, Om, w0, Xi0, n = Lambda_t[0], Lambda_t[1], Lambda_t[2], Lambda_t[3], Lambda_t[4]
    log_p_pop_fun = log_p_pop_at_wrap if wrap_logp else log_p_pop_at

    # spins
    spin_is_default = (spin_model in ("default", "default_gauss"))
    if spin_is_default:
        s1_all = _as_at(spinsInj[0]); s2_all = _as_at(spinsInj[1])
        ct1_all = _as_at(spinsInj[2]); ct2_all = _as_at(spinsInj[3])

    work_dtype = getattr(m1_all, "dtype", "float64")
    int_dtype  = "int32" if work_dtype in ("float16", "float32") else "int64"
    K = int(chunk_size)

    # pad & mask
    m1K, C, N = _pad_to_multiple(m1_all,   K, 2.0)
    m2K, _, _ = _pad_to_multiple(m2_all,   K, 1.0)
    dLK, _, _ = _pad_to_multiple(dL_all,   K, 1.0)
    lpdK,_, _ = _pad_to_multiple(lpd_all,  K, 0.0)
    if spin_is_default:
        s1K,  _, _ = _pad_to_multiple(s1_all,  K, 0.0)
        s2K,  _, _ = _pad_to_multiple(s2_all,  K, 0.0)
        ct1K, _, _ = _pad_to_multiple(ct1_all, K, 1.0)
        ct2K, _, _ = _pad_to_multiple(ct2_all, K, 1.0)

    idxs = at.arange(C, dtype=int_dtype)
    valid_mask = (at.arange(C*K, dtype=int_dtype) < N).reshape((C, K))
    NEG_BIG = -np.inf
    eps = at.as_tensor_variable(1e-30, dtype=work_dtype)

    # z(dL) via grids (optional)
    have_dLz = (dL_grid is not None) and (z_grid is not None)
    if have_dLz:
        dL_grid_t = _as_at(dL_grid)
        z_grid_t  = _as_at(z_grid)
        # Precompute idx outside; pad idx with 1 so [il,ih] valid
        idx_full = at.searchsorted(dL_grid_t, dL_all, side="right").astype(int_dtype)
        idx_full = at.clip(idx_full, 1, dL_grid_t.shape[0] - 1)
        idx_full = stop_grad(idx_full)
        one_idx = at.as_tensor_variable(1, dtype=int_dtype)
        idxK, _, _ = _pad_to_multiple(idx_full, K, one_idx)

    # ---- scan body ----
    if spin_is_default:
        def step(i, m_state, m2_state, s1_state, s2_state,
                 m1K, m2K, dLK, lpdK, valid_mask, Lambda_t,
                 s1K, s2K, ct1K, ct2K, *maybe):
            m1 = m1K[i]; m2 = m2K[i]; dL = dLK[i]; lpd = lpdK[i]; mask = valid_mask[i]
            s1 = s1K[i];  s2 = s2K[i]; ct1 = ct1K[i]; ct2 = ct2K[i]
            spins_use = [s1, s2, ct1, ct2]

            # z interpolation or cosmology
            if have_dLz:
                dL_grid_t, z_grid_t, idxK = maybe
                idx = idxK[i]
                il, ih = idx - 1, idx
                xl = dL_grid_t[il]; xh = dL_grid_t[ih]
                yl = z_grid_t[il];  yh = z_grid_t[ih]
                denom = at.maximum(xh - xl, eps)
                r = (dL - xl) / denom
                zinj_c = (1 - r) * yl + r * yh
            else:
                zinj_c  = atools.z_from_dL_at(dL, H0, Om, w0, Xi0, n, interp=interp, param=param)

            # analytic dc and log_ddL_dz
            dc_c    = atools.dcfun_at(zinj_c, H0, Om, w0, interp=interp)
            logdd_c = atools.log_ddL_dz(zinj_c, H0, Om, w0, Xi0, n, dc=dc_c, interp=interp, param=param)

            one_p_z = 1.0 + zinj_c
            m1Src = m1 / one_p_z
            m2Src = m2 / one_p_z

            use_dp = (mass_model in ("DP", "DPUC"))
            if use_dp:
                Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
                #mass_1_use = atools.safe_log(Mc_src_inj)
                #eps = at.as_tensor_variable(1e-30, dtype=Mc_src_inj.dtype)
                mass_1_use = atools.safe_log(at.maximum(Mc_src_inj, eps))
                mass_2_use = atools.logitat(q_inj)
            else:
                mass_1_use = m1Src
                mass_2_use = m2Src

            lp = log_p_pop_fun(
                mass_1_use, mass_2_use, zinj_c, dL, spins_use, Lambda_t,
                rate_model, mass_model, spin_model,
                smoothing=smoothing, has_m2_break=has_m2_break,
                log_ddL_dz_pre=logdd_c, dc=dc_c,
            )

            if use_dp:
                lp = (lp
                      - atools.safe_log(at.maximum(m2Src, eps))
                      - atools.safe_log(at.maximum(m1Src - m2Src, eps))
                      - at.log1p(zinj_c))

            x = at.where(mask, lp - lpd, NEG_BIG)

            m  = at.max(x)
            y  = at.exp(x - m)
            s1c = at.sum(y)
            s2c = at.sum(at.sqr(y))

            m_new,  s1_new = _combine_logsumexp(m_state,  s1_state,  m,     s1c)
            m2c = 2.0 * m
            m2_new, s2_new = _combine_logsumexp(m2_state, s2_state, m2c,    s2c)
            return m_new, m2_new, s1_new, s2_new
    else:
        def step(i, m_state, m2_state, s1_state, s2_state,
                 m1K, m2K, dLK, lpdK, valid_mask, Lambda_t, *maybe):
            m1 = m1K[i]; m2 = m2K[i]; dL = dLK[i]; lpd = lpdK[i]; mask = valid_mask[i]
            spins_use = []

            if have_dLz:
                dL_grid_t, z_grid_t, idxK = maybe
                idx = idxK[i]
                il, ih = idx - 1, idx
                xl = dL_grid_t[il]; xh = dL_grid_t[ih]
                yl = z_grid_t[il];  yh = z_grid_t[ih]
                denom = at.maximum(xh - xl, eps)
                r = (dL - xl) / denom
                zinj_c = (1 - r) * yl + r * yh
            else:
                zinj_c  = atools.z_from_dL_at(dL, H0, Om, w0, Xi0, n, interp=interp, param=param)

            dc_c    = atools.dcfun_at(zinj_c, H0, Om, w0, interp=interp)
            logdd_c = atools.log_ddL_dz(zinj_c, H0, Om, w0, Xi0, n, dc=dc_c, interp=interp, param=param)

            one_p_z = 1.0 + zinj_c
            m1Src = m1 / one_p_z
            m2Src = m2 / one_p_z

            use_dp = (mass_model in ("DP", "DPUC"))
            if use_dp:
                Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
                #mass_1_use = atools.safe_log(Mc_src_inj)
                mass_1_use = atools.safe_log(at.maximum(Mc_src_inj, eps))
                mass_2_use = atools.logitat(q_inj)
            else:
                mass_1_use = m1Src
                mass_2_use = m2Src

            lp = log_p_pop_fun(
                mass_1_use, mass_2_use, zinj_c, dL, spins_use, Lambda_t,
                rate_model, mass_model, spin_model,
                smoothing=smoothing, has_m2_break=has_m2_break,
                log_ddL_dz_pre=logdd_c, dc=dc_c,
            )

            if use_dp:
                lp = (lp
                      - atools.safe_log(at.maximum(m2Src, eps))
                      - atools.safe_log(at.maximum(m1Src - m2Src, eps))
                      - at.log1p(zinj_c))

            x = at.where(mask, lp - lpd, NEG_BIG)

            m  = at.max(x)
            y  = at.exp(x - m)
            s1c = at.sum(y)
            s2c = at.sum(at.sqr(y))

            m_new,  s1_new = _combine_logsumexp(m_state,  s1_state,  m,     s1c)
            m2c = 2.0 * m
            m2_new, s2_new = _combine_logsumexp(m2_state, s2_state, m2c,    s2c)
            return m_new, m2_new, s1_new, s2_new

    # non_sequences
    m_init = at.as_tensor_variable(-at.inf, dtype=work_dtype)
    s_init = at.as_tensor_variable(0.0,    dtype=work_dtype)

    nonseq = [m1K, m2K, dLK, lpdK, valid_mask, Lambda_t]
    if spin_is_default:
        nonseq += [s1K, s2K, ct1K, ct2K]
    if have_dLz:
        nonseq += [dL_grid_t, z_grid_t, idxK]

    (m_fin, m2_fin, s1_fin, s2_fin), _ = pytensor.scan(
        fn=step,
        sequences=[idxs],
        outputs_info=[m_init, m_init, s_init, s_init],
        non_sequences=nonseq,
        strict=True
    )

    tinyL = at.as_tensor_variable(1e-300, dtype=work_dtype)
    logsumexp1 = m_fin[-1]  + atools.safe_log(at.maximum(s1_fin[-1], tinyL))
    logsumexp2 = m2_fin[-1] + atools.safe_log(at.maximum(s2_fin[-1], tinyL))

    Ndraw_t = at.as_tensor_variable(Ndraw).astype(work_dtype)
    log_mu  = logsumexp1 - atools.safe_log(Ndraw_t)
    logs2   = logsumexp2 - atools.safe_log(Ndraw_t)
    logNeff = 2.0 * log_mu - logs2 + atools.safe_log(Ndraw_t)
    Neff    = at.exp(logNeff)
    var_log_lik_u = atools.logdiffexp(logs2 - 2.0 * log_mu, 1.0) - atools.safe_log(Ndraw_t - 1.0)

    return log_mu, Neff, var_log_lik_u





def sel_bias_with_uncertainty_at_loop(
    m1inj, m2inj, dLinj, spinsInj, log_p_draw,
    Lambda, Ndraw,
    rate_model, mass_model, spin_model,
    smoothing, has_m2_break, interp,
    wrap_logp=False,
    # kept for API compat (ignored when grids are provided)
    log_ddL_dz_inj=None,
    zinj=None,
    dcinj=None,
    # symbolic grids (may depend on RVs)
    dL_grid=None,           # 1-D, strictly increasing in dL
    z_grid=None,            # 1-D, z(dL_grid)
    dc_grid=None,           # optional: if provided, we can atinterp on z (loop has no padding)
    log_ddL_dz_grid=None,   # optional: same as above
    *,
    chunk_size=4096,
    N_inj_py=None,
    param='vanilla',
    **kwargs
):
    """
    Low-memory Python loop (GPU-stable).
    - Computes z per chunk: from (dL_grid,z_grid) via atinterp if provided, else analytic.
    - Computes dc and log_ddL_dz per chunk: from grids if both (z_grid & dc/logdd grids) exist, else analytic.
    - Passes dc and log_ddL_dz_pre to log_p_pop_* (new API preserved).
    """
    def _as_at(x):
        return x if isinstance(x, at.Variable) else at.as_tensor_variable(x)

    if N_inj_py is None:
        # try best-effort extraction
        tv = getattr(getattr(m1inj, "tag", object()), "test_value", None)
        if tv is None:
            raise ValueError("Pass N_inj_py=<python int> for the loop variant.")
        N_inj_py = int(tv.shape[0])

    m1_all   = _as_at(m1inj)
    m2_all   = _as_at(m2inj)
    dL_all   = _as_at(dLinj)
    lpd_all  = _as_at(log_p_draw)
    Lambda_t = _as_at(Lambda)

    H0, Om, w0, Xi0, n = Lambda_t[0], Lambda_t[1], Lambda_t[2], Lambda_t[3], Lambda_t[4]
    log_p_pop_fun = log_p_pop_at_wrap if wrap_logp else log_p_pop_at

    # spins
    if (spin_model == "default") or (spin_model == "default_gauss"):
        s1_all = _as_at(spinsInj[0]); s2_all = _as_at(spinsInj[1])
        ct1_all = _as_at(spinsInj[2]); ct2_all = _as_at(spinsInj[3])
        use_spins = True
    else:
        use_spins = False

    CH = int(chunk_size)
    N_py = int(N_inj_py)

    work_dtype = getattr(m1_all, "dtype", "float64")
    eps  = at.as_tensor_variable(1e-30,  dtype=work_dtype)
    tinyL = at.as_tensor_variable(1e-300, dtype=work_dtype)

    log_sum  = at.as_tensor_variable(-at.inf, dtype=work_dtype)
    log_sum2 = at.as_tensor_variable(-at.inf, dtype=work_dtype)

    # grids tensors if provided
    have_z_from_grid = (dL_grid is not None) and (z_grid is not None)
    if have_z_from_grid:
        dL_grid_t = _as_at(dL_grid); z_grid_t = _as_at(z_grid)
    have_dc_grid = (dc_grid is not None) and (z_grid is not None)
    if have_dc_grid:
        dc_grid_t = _as_at(dc_grid)
    have_logdd_grid = (log_ddL_dz_grid is not None) and (z_grid is not None)
    if have_logdd_grid:
        logdd_grid_t = _as_at(log_ddL_dz_grid)

    for start in range(0, N_py, CH):
        stop = min(start + CH, N_py)

        m1  = m1_all[start:stop]
        m2  = m2_all[start:stop]
        dL  = dL_all[start:stop]
        lpd = lpd_all[start:stop]
        spins_use = [s1_all[start:stop], s2_all[start:stop], ct1_all[start:stop], ct2_all[start:stop]] if use_spins else []

        # z per chunk
        if have_z_from_grid:
            zinj_c = atools.atinterp(dL, dL_grid_t, z_grid_t)    # safe in loop (no padding)
        else:
            zinj_c = atools.z_from_dL_at(dL, H0, Om, w0, Xi0, n, interp=interp)

        # dc and logdd per chunk (prefer grids if BOTH available; else analytic)
        if have_dc_grid and have_logdd_grid:
            dc_c    = atools.atinterp(zinj_c, z_grid_t, dc_grid_t)
            logdd_c = atools.atinterp(zinj_c, z_grid_t, logdd_grid_t)
        else:
            dc_c    = atools.dcfun_at(zinj_c, H0, Om, w0, interp=interp)
            logdd_c = atools.log_ddL_dz(zinj_c, H0, Om, w0, Xi0, n, dc=dc_c, interp=interp, param=param)

        one_p_z = 1.0 + zinj_c
        m1Src = m1 / one_p_z
        m2Src = m2 / one_p_z

        if mass_model in ("DP", "DPUC"):
            Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
            mass_1_use = atools.safe_log(Mc_src_inj)
            mass_2_use = atools.logitat(q_inj)
        else:
            mass_1_use = m1Src
            mass_2_use = m2Src

        lp = log_p_pop_fun(
            mass_1_use, mass_2_use, zinj_c, dL, spins_use, Lambda_t,
            rate_model, mass_model, spin_model,
            smoothing=smoothing, has_m2_break=has_m2_break,
            log_ddL_dz_pre=logdd_c,
            dc=dc_c,
        )

        if mass_model in ("DP", "DPUC"):
            lp = (lp
                  - atools.safe_log(at.maximum(m2Src, eps))
                  - atools.safe_log(at.maximum(m1Src - m2Src, eps))
                  - at.log1p(zinj_c))

        x  = lp - lpd
        m  = at.max(x)
        y  = at.exp(x - m)
        s1c = at.sum(y)
        s2c = at.sum(at.sqr(y))

        log_sum  = at.logaddexp(log_sum,  m + atools.safe_log(at.maximum(s1c, tinyL)))
        log_sum2 = at.logaddexp(log_sum2, 2.0*m + atools.safe_log(at.maximum(s2c, tinyL)))

    Ndraw_t = at.as_tensor_variable(Ndraw).astype(work_dtype)
    log_mu  = log_sum  - atools.safe_log(Ndraw_t)
    logs2   = log_sum2 - atools.safe_log(Ndraw_t)
    logNeff = 2.0 * log_mu - logs2 + atools.safe_log(Ndraw_t)
    Neff    = at.exp(logNeff)
    var_log_lik_u = atools.logdiffexp(logs2 - 2.0 * log_mu, 1.0) - atools.safe_log(Ndraw_t - 1.0)

    return log_mu, Neff, var_log_lik_u



def sel_bias_with_uncertainty_at_0_batched(
    m1inj, m2inj, dLinj, spinsInj, log_p_draw,
    Lambda, Ndraw,
    rate_model, mass_model, spin_model,
    smoothing, has_m2_break, interp,
    wrap_logp=False,
    # kept for API compat (used only if grids are not provided)
    log_ddL_dz_inj=None,
    zinj=None,
    dcinj=None,
    # grids ONLY for dL->z
    dL_grid=None,           # 1-D, strictly increasing in dL
    z_grid=None,            # 1-D, z(dL_grid)
    dc_grid=None,           # UNUSED in this variant
    log_ddL_dz_grid=None,   # UNUSED in this variant
    *,
    chunk_size=4096,
    param='vanilla',
    **kwargs
):
    """Vectorized + batched reduction, only z(dL) interpolated from grids.
    dc(z) and log_ddL_dz(z) computed analytically.
    """

    def _as_at(x):
        return x if isinstance(x, at.Variable) else at.as_tensor_variable(x)

    def _vec1(x, dtype):
        v = x if isinstance(x, at.Variable) else at.as_tensor_variable(x)
        if v.ndim != 1:
            v = at.flatten(v, 1)
        return v.astype(dtype) if v.dtype != dtype else v

    def _pad_to_K(x, K, pad_value, dtype):
        x = _vec1(x, dtype)
        N = x.shape[0]
        C = (N + K - 1) // K
        Npad = C * K - N
        pad = at.full((Npad,), at.as_tensor_variable(pad_value, dtype=x.dtype), dtype=x.dtype)
        xK  = at.concatenate([x, pad], axis=0).reshape((C, K))
        return xK, C, N

    def _lin_interp_with_indices(x, xg, yg, idx, eps):
        il, ih = idx - 1, idx
        xl, xh = xg[il], xg[ih]
        yl, yh = yg[il], yg[ih]
        denom  = at.maximum(xh - xl, eps)
        r = (x - xl) / denom
        return (1.0 - r) * yl + r * yh

    # ---- config / dtypes ----
    work_dtype = getattr(m1inj, "dtype", "float64")
    int_dtype  = "int32" if work_dtype in ("float16", "float32") else "int64"
    K = int(chunk_size)

    # ---- pad observed arrays to (C,K) ----
    m1K, C, N = _pad_to_K(m1inj,  K, 2.0, work_dtype)
    m2K, _, _ = _pad_to_K(m2inj,  K, 1.0, work_dtype)
    dLK,  _, _ = _pad_to_K(dLinj, K, 1.0, work_dtype)
    lpdK, _, _ = _pad_to_K(log_p_draw, K, 0.0, work_dtype)

    # spins
    spin_is_default = (spin_model in ("default", "default_gauss"))
    if spin_is_default:
        s1K, _, _  = _pad_to_K(spinsInj[0], K, 0.0, work_dtype)
        s2K, _, _  = _pad_to_K(spinsInj[1], K, 0.0, work_dtype)
        ct1K, _, _ = _pad_to_K(spinsInj[2], K, 1.0, work_dtype)
        ct2K, _, _ = _pad_to_K(spinsInj[3], K, 1.0, work_dtype)

    # mask & constants
    mask   = (at.arange(C * K, dtype=int_dtype) < N).reshape((C, K))
    NEG_BG = -np.inf
    tiny   = at.as_tensor_variable(1e-30,  dtype=work_dtype)
    tinyL  = at.as_tensor_variable(1e-300, dtype=work_dtype)

    # cosmology pieces / logp dispatcher
    Lambda_t = _as_at(Lambda)
    H0, Om, w0, Xi0, n = Lambda_t[0], Lambda_t[1], Lambda_t[2], Lambda_t[3], Lambda_t[4]
    log_p_pop_fun = log_p_pop_at_wrap if wrap_logp else log_p_pop_at
    use_dp = (mass_model in ("DP", "DPUC"))

    # ---------- dL -> z (single pass, flattened) ----------
    have_dLz = (dL_grid is not None) and (z_grid is not None)
    if have_dLz:
        dL_grid_t = _as_at(dL_grid)
        z_grid_t  = _as_at(z_grid)

        dL_flat   = dLK.reshape((-1,))
        mask_flat = mask.reshape((-1,))

        # snap padded rows to interior x before binning
        safe_dL = at.where(mask_flat, dL_flat, dL_grid_t[1])
        idx_dL  = at.searchsorted(dL_grid_t, safe_dL, side="right").astype(int_dtype)
        lo, hi  = at.as_tensor_variable(1, dtype=int_dtype), (dL_grid_t.shape[0] - 1).astype(int_dtype)
        idx_dL  = stop_grad(at.clip(idx_dL, lo, hi))

        z_flat = _lin_interp_with_indices(safe_dL, dL_grid_t, z_grid_t, idx_dL, tiny)
        zK     = z_flat.reshape((C, K))
    else:
        if zinj is not None:
            zK, _, _ = _pad_to_K(zinj, K, 0.0, work_dtype)
        else:
            zK = atools.z_from_dL_at(dLK, H0, Om, w0, Xi0, n, interp=interp, param=param)

    # ---------- analytic dc(z) and log_ddL_dz(z) ----------
    if dcinj is not None:
        dcK, _, _ = _pad_to_K(dcinj, K, 0.0, work_dtype)
    else:
        dcK = atools.dcfun_at(zK, H0, Om, w0, interp=interp)

    if log_ddL_dz_inj is not None:
        dK, _, _ = _pad_to_K(log_ddL_dz_inj, K, 0.0, work_dtype)
    else:
        dK = atools.log_ddL_dz(zK, H0, Om, w0, Xi0, n, dc=dcK, interp=interp, param=param)

    # ---- masses in source frame ----
    one_p_z = 1.0 + zK
    m1SrcK  = m1K / one_p_z
    m2SrcK  = m2K / one_p_z

    if use_dp:
        McK, qK = atools.Mcq_from_m1m2_at(m1SrcK, m2SrcK)
        m1useK  = atools.safe_log(at.maximum(McK, tiny) )
        #mass_1_use = atools.safe_log(at.maximum(Mc_src_inj, eps))
        m2useK  = atools.logitat(qK)
    else:
        m1useK, m2useK = m1SrcK, m2SrcK

    spins_arg = [s1K, s2K, ct1K, ct2K] if spin_is_default else []

    # ---- log p_pop ----
    lpK = log_p_pop_fun(
        m1useK, m2useK, zK, dLK, spins_arg, Lambda_t,
        rate_model, mass_model, spin_model,
        smoothing=smoothing, has_m2_break=has_m2_break,
        log_ddL_dz_pre=dK, dc=dcK,
    )

    if use_dp:
        lpK = (lpK
               - atools.safe_log(at.maximum(m2SrcK, tiny))
               - atools.safe_log(at.maximum(m1SrcK - m2SrcK, tiny))
               - at.log1p(zK))

    # ---- stable batched reduction ----
    xK = at.where(mask, lpK - lpdK, NEG_BG)

    m_chunks = at.max(xK, axis=1)                 # (C,)
    y  = at.exp(xK - m_chunks[:, None])           # (C,K)
    s1 = at.sum(y, axis=1)                        # (C,)
    s2 = at.sum(at.sqr(y), axis=1)                # (C,)

    m_global  = at.max(m_chunks)
    S1 = at.sum(s1 * at.exp(m_chunks - m_global))
    m2_global = 2.0 * m_global
    S2 = at.sum(s2 * at.exp(2.0 * m_chunks - m2_global))

    logsumexp1 = m_global  + atools.safe_log(at.maximum(S1, tinyL))
    logsumexp2 = m2_global + atools.safe_log(at.maximum(S2, tinyL))

    # ---- outputs ----
    Ndraw_t = _as_at(Ndraw).astype(work_dtype)
    log_mu  = logsumexp1 - atools.safe_log(Ndraw_t)
    logs2   = logsumexp2 - atools.safe_log(Ndraw_t)

    logNeff = 2.0 * log_mu - logs2 + atools.safe_log(Ndraw_t)
    Neff    = at.exp(logNeff)
    var_log_lik_u = atools.logdiffexp(logs2 - 2.0 * log_mu, 1.0) - atools.safe_log(Ndraw_t - 1.0)

    return log_mu, Neff, var_log_lik_u

######################

def sel_bias_with_uncertainty_at_0_debug(m1inj, m2inj, dLinj, spinsInj, log_p_draw, 
                                    Lambda,  Ndraw, 
                                    rate_model, mass_model, spin_model, 
                                    smoothing, 
                                    has_m2_break, 
                                    interp, 
                                    wrap_logp=False, 
                                    log_ddL_dz_inj = None,
                                    zinj = None,
                                    dcinj = None,
                                    param='vanilla',
                                    **kwargs):
    work_dtype = getattr(m1inj, "dtype", "float64")

    # ignore everything and just return constants
    log_mu = at.as_tensor_variable(0.0, dtype=work_dtype)
    Neff   = at.as_tensor_variable(100000.0, dtype=work_dtype)
    var_log_lik_u = at.as_tensor_variable(0.5, dtype=work_dtype)
    return log_mu, Neff, var_log_lik_u



def sel_bias_with_uncertainty_at_0_test(m1inj, m2inj, dLinj, spinsInj, log_p_draw, 
                                    Lambda,  Ndraw, 
                                    rate_model, mass_model, spin_model, 
                                    smoothing, 
                                    has_m2_break, 
                                    interp, 
                                    wrap_logp=False, 
                                    log_ddL_dz_inj = None,
                                    zinj = None,
                                    dcinj = None,
                                    param='vanilla',
                                    **kwargs):

    work_dtype = getattr(m1inj, "dtype", "float64")

    if work_dtype == "float32":
        eps      = at.as_tensor_variable(1e-20, dtype=work_dtype)
        tinyL    = at.as_tensor_variable(1e-30, dtype=work_dtype)
        big_neg  = at.as_tensor_variable(-1e6,  dtype=work_dtype)
        big_pos  = at.as_tensor_variable(1e6,   dtype=work_dtype)
        tiny_sum = at.as_tensor_variable(1e-20, dtype=work_dtype)
    else:
        eps      = at.as_tensor_variable(1e-30,  dtype=work_dtype)
        tinyL    = at.as_tensor_variable(1e-300, dtype=work_dtype)
        big_neg  = at.as_tensor_variable(-1e12, dtype=work_dtype)
        big_pos  = at.as_tensor_variable(1e12,  dtype=work_dtype)
        tiny_sum = at.as_tensor_variable(1e-300, dtype=work_dtype)

    #H0, Om, w0, Xi0, n  = Lambda[:5]
    H0  = Lambda[0]
    Om  = Lambda[1]
    w0  = Lambda[2]
    Xi0 = Lambda[3]
    n   = Lambda[4]

    if wrap_logp:
        log_p_pop_fun = log_p_pop_at_wrap
        print("Using wrapped p_pop for inj")
    else:
        log_p_pop_fun = log_p_pop_at
        print("Using regular p_pop for inj")

    if (spin_model == 'default') or (spin_model == 'default_gauss'):
        spinsInj_sel = [spinsInj[0], spinsInj[1], spinsInj[2], spinsInj[3]]
    elif spin_model == 'none':
        spinsInj_sel = []

    if zinj is None:
        print("Sel bias is recomputing zinj!")
        zinj = atools.z_from_dL_at(dLinj, H0, Om, w0, Xi0, n, interp=interp, param=param) 
    if dcinj is None:
        print("Sel bias is recomputing dcinj!")
        dcinj = atools.dcfun_at(zinj, H0, Om, w0, interp=interp)        
    if log_ddL_dz_inj is None:
        print("Sel bias is recomputing log_ddL_dz_inj!")
        log_ddL_dz_inj = atools.log_ddL_dz(zinj, H0, Om,  w0, Xi0, n,
                                           dc=dcinj, interp=interp, param=param)
    
    one_p_z = 1.0 + zinj
    m1Src   = m1inj / one_p_z
    m2Src   = m2inj / one_p_z

    if mass_model in ('DP', 'DPUC'):
        Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
        log_Mc_src_inj = atools.safe_log(at.maximum(Mc_src_inj, eps))
        logit_q_inj    = atools.logitat(q_inj)
        mass_1_use     = log_Mc_src_inj
        mass_2_use     = logit_q_inj
    else:
        mass_1_use = m1Src
        mass_2_use = m2Src

    log_p_pop = log_p_pop_fun(
        mass_1_use, mass_2_use, zinj, dLinj, spinsInj_sel,
        Lambda,
        rate_model, mass_model, spin_model,
        smoothing=smoothing,
        has_m2_break=has_m2_break,
        log_ddL_dz_pre=log_ddL_dz_inj,
        dc=dcinj,
    )

    # If you want the Jacobian back, you can re-enable this block later:
    # if mass_model in ('DP', 'DPUC'):
    #     log_p_pop += (
    #         - atools.safe_log(m2Src)
    #         - atools.safe_log(at.maximum(m1Src - m2Src, eps))
    #         - at.log1p(zinj)
    #     )

    # ------------------------------------------------------
    # SANITIZE log_p_pop: NaNs → big_neg, clip to [big_neg, big_pos]
    # NaN detection: x != x is True only for NaN
    # ------------------------------------------------------
    nan_mask = at.neq(log_p_pop, log_p_pop)
    print(at.sum(nan_mask).eval())
    log_p_pop_clean = at.where(nan_mask, big_neg, log_p_pop)
    log_p_pop_clean = at.clip(log_p_pop_clean, big_neg, big_pos)

    # selection-bias log-weights
    log_sel_b = log_p_pop_clean - log_p_draw
    log_sel_b = at.clip(log_sel_b, big_neg, big_pos)

    # very safe logsumexp over a 1D vector
    def safe_logsumexp(x):
        x = at.as_tensor_variable(x)
        m = at.max(x)
        y = at.exp(x - m)
        s = at.sum(y)
        s_safe = at.maximum(s, tiny_sum)   # avoid log(0)
        return m + at.log(s_safe)

    Ndraw_t = at.as_tensor_variable(Ndraw).astype(work_dtype)

    # mean log selection-bias term
    log_mu = safe_logsumexp(log_sel_b) - atools.safe_log(Ndraw_t)

    # keep Neff and var simple/safe to avoid NaNs
    Neff = Ndraw_t
    var_log_lik_u = at.zeros_like(log_mu)

    return log_mu, Neff, var_log_lik_u




def sel_bias_with_uncertainty_at_0_safe(m1inj, m2inj, dLinj, spinsInj, log_p_draw, 
                                    Lambda,  Ndraw, 
                                    rate_model, mass_model, spin_model, 
                                    smoothing, 
                                    has_m2_break, 
                                    interp, 
                                    wrap_logp=False, 
                                    log_ddL_dz_inj = None,
                                    zinj = None,
                                    dcinj = None,
                                    param='vanilla',
                                    **kwargs):

    work_dtype = getattr(m1inj, "dtype", "float64")

    if work_dtype == "float32":
        # reasonable tiny values in float32
        eps   = at.as_tensor_variable(1e-20, dtype=work_dtype)
        tinyL = at.as_tensor_variable(1e-30, dtype=work_dtype)
    else:
        eps   = at.as_tensor_variable(1e-30,  dtype=work_dtype)
        tinyL = at.as_tensor_variable(1e-300, dtype=work_dtype)

    #H0, Om, w0, Xi0, n  = Lambda[:5]
    H0  = Lambda[0]
    Om  = Lambda[1]
    w0  = Lambda[2]
    Xi0 = Lambda[3]
    n   = Lambda[4]

    if wrap_logp:
        log_p_pop_fun = log_p_pop_at_wrap
        print("Using wrapped p_pop for inj")
    else:
        log_p_pop_fun = log_p_pop_at
        print("Using regular p_pop for inj")

    if (spin_model=='default') or (spin_model=='default_gauss'):
        spinsInj_sel = [spinsInj[0], spinsInj[1], spinsInj[2], spinsInj[3]]
    elif spin_model=='none':
        spinsInj_sel = []

    if zinj is None:
        print("Sel bias is recomputing zinj!")
        zinj = atools.z_from_dL_at(dLinj, H0, Om, w0, Xi0, n, interp=interp, param=param) 
    if dcinj is None:
        print("Sel bias is recomputing dcinj!")
        dcinj = atools.dcfun_at(zinj, H0, Om, w0, interp=interp)        
    if log_ddL_dz_inj is None:
        print("Sel bias is recomputing log_ddL_dz_inj!")
        log_ddL_dz_inj = atools.log_ddL_dz(zinj, H0, Om,  w0, Xi0, n, dc=dcinj, interp=interp, param=param)
    
    one_p_z = 1.0 + zinj
    m1Src  = m1inj/one_p_z
    m2Src  = m2inj/one_p_z

    if mass_model in ('DP', 'DPUC'):
        Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
        #log_Mc_src_inj = atools.safe_log(Mc_src_inj)
        log_Mc_src_inj = atools.safe_log(at.maximum(Mc_src_inj, eps))
        logit_q_inj = atools.logitat(q_inj)      
        mass_1_use = log_Mc_src_inj
        mass_2_use = logit_q_inj
    else:
        mass_1_use = m1Src
        mass_2_use = m2Src

    log_p_pop = log_p_pop_fun(mass_1_use, mass_2_use, zinj, dLinj, spinsInj_sel, 
                              Lambda, 
                              rate_model, mass_model, spin_model, 
                              smoothing=smoothing, 
                              has_m2_break=has_m2_break, 
                              log_ddL_dz_pre = log_ddL_dz_inj,
                              dc = dcinj
                             )

    if mass_model in ('DP', 'DPUC'):
        # remove jacobian m1, m2 --> log(Mc), logit(q)
        log_p_pop += (- atools.safe_log(m2Src) 
                      - atools.safe_log(at.maximum(m1Src - m2Src, eps)) #atools.safe_log(m1Src-m2Src) 
                      #- at.log1p(zinj) 
                     )

    log_sel_b = log_p_pop - log_p_draw

    # ----------------------------------------------------------------------
    # SAFETY 1: guard against NaNs in log_sel_b itself
    # Any injection with NaN log weight is treated as having negligible weight
    # ----------------------------------------------------------------------
    if work_dtype == "float32":
        big_neg = at.as_tensor_variable(-1e6).astype(work_dtype)
    else:
        big_neg = at.as_tensor_variable(-1e12).astype(work_dtype)

    log_sel_b = at.where(at.isnan(log_sel_b), big_neg, log_sel_b)

    # Ndraw must be a symbolic tensor with a floating dtype for logs
    Ndraw_t = at.as_tensor_variable(Ndraw).astype(work_dtype)
    log_Ndraw = atools.safe_log(Ndraw_t)

    # raw log-mean and log-second-moment
    log_mu_raw = at.logsumexp(log_sel_b)         - log_Ndraw
    logs2_raw  = at.logsumexp(2.0 * log_sel_b)   - log_Ndraw

    # ----------------------------------------------------------------------
    # SAFETY 2: avoid inf - inf in Neff / variance
    # ----------------------------------------------------------------------
    if work_dtype == "float32":
        tiny_log = at.as_tensor_variable(-1e6).astype(work_dtype)
    else:
        tiny_log = at.as_tensor_variable(-1e12).astype(work_dtype)

    too_tiny = at.or_(log_mu_raw < tiny_log, logs2_raw < tiny_log)

    # safe versions used inside arithmetic
    log_mu_safe = at.where(too_tiny, at.zeros_like(log_mu_raw), log_mu_raw)
    logs2_safe  = at.where(too_tiny, at.zeros_like(logs2_raw),  logs2_raw)

    #####################################
    # N_eff as in Talbot & Golomb (2023)
    #####################################
    logNeff_raw = 2.0 * log_mu_safe - logs2_safe + log_Ndraw

    logNeff = at.where(
        too_tiny,
        at.as_tensor_variable(-np.inf).astype(work_dtype),
        logNeff_raw,
    )

    #####################################
    # Variance of log l per unit obs (Talbot & Golomb 2023)
    #####################################
    delta_safe = logs2_safe - 2.0 * log_mu_safe
    var_finite = atools.logdiffexp(delta_safe, 1.) - atools.safe_log(Ndraw_t - 1)

    var_inf = at.as_tensor_variable(np.inf).astype(work_dtype)
    var_log_lik_u = at.where(too_tiny, var_inf, var_finite)

    Neff   = at.exp(logNeff)
    log_mu = log_mu_raw  # original log_mu as return value

    return log_mu, Neff, var_log_lik_u


    
    
def sel_bias_with_uncertainty_at_0(m1inj, m2inj, dLinj, spinsInj, log_p_draw, 
                                    Lambda,  Ndraw, 
                                    rate_model, mass_model, spin_model, 
                                    smoothing, 
                                   simplex_repair,
                                    has_m2_break, 
                                    interp, 
                                   log_p_incl = None,
                                    wrap_logp=False, 
                                    log_ddL_dz_inj = None,
                                    zinj = None,
                                    dcinj = None,
                                   param='vanilla',
                                   interp_vals_mass = None,
                                    interp_grids_mass = None,
                                   verbose=False,
                                    **kwargs):

    work_dtype = getattr(m1inj, "dtype", "float64")

    if work_dtype == "float32":
        # reasonable tiny values in float32
        eps   = at.as_tensor_variable(1e-20, dtype=work_dtype)
        tinyL = at.as_tensor_variable(1e-30, dtype=work_dtype)
    else:
        eps   = at.as_tensor_variable(1e-30,  dtype=work_dtype)
        tinyL = at.as_tensor_variable(1e-300, dtype=work_dtype)

    #H0, Om, w0, Xi0, n  = Lambda[:5]
    H0  = Lambda[0]
    Om  = Lambda[1]
    w0  = Lambda[2]
    Xi0 = Lambda[3]
    n   = Lambda[4]

    if wrap_logp:
        log_p_pop_fun = log_p_pop_at_wrap
        print("Using wrapped p_pop for inj")
    else:
        log_p_pop_fun = log_p_pop_at
        print("Using regular p_pop for inj")

    if (spin_model=='default') or (spin_model=='default_gauss'):
        spinsInj_sel = [spinsInj[0], spinsInj[1], spinsInj[2], spinsInj[3]]
    elif spin_model=='none':
        spinsInj_sel = []



    if zinj is None:
        print("Sel bias is recomputing zinj!")
        zinj = atools.z_from_dL_at(dLinj, H0, Om, w0, Xi0, n, interp=interp, param=param) 
    if dcinj is None:
        print("Sel bias is recomputing dcinj!")
        dcinj = atools.dcfun_at(zinj, H0, Om, w0, interp=interp)        
    if log_ddL_dz_inj is None:
        print("Sel bias is recomputing log_ddL_dz_inj!")
        log_ddL_dz_inj = atools.log_ddL_dz(zinj, H0, Om,  w0, Xi0, n, dc=dcinj, interp=interp, param=param)
    
    
    one_p_z = 1.0 + zinj
    m1Src  = m1inj/one_p_z
    m2Src  = m2inj/one_p_z

    if mass_model in ('DP', 'DPUC'):
        Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
        #log_Mc_src_inj = atools.safe_log(Mc_src_inj)
        log_Mc_src_inj = atools.safe_log(at.maximum(Mc_src_inj, eps))
        logit_q_inj = atools.logitat(q_inj)      
        mass_1_use = log_Mc_src_inj
        mass_2_use = logit_q_inj
    else:
        mass_1_use = m1Src
        mass_2_use = m2Src

    log_p_pop = log_p_pop_fun(mass_1_use, mass_2_use, zinj, dLinj, spinsInj_sel, 
                              Lambda, 
                              rate_model, mass_model, spin_model, 
                              smoothing=smoothing, 
                              simplex_repair=simplex_repair,
                              has_m2_break=has_m2_break, 
                              log_ddL_dz_pre = log_ddL_dz_inj,
                              dc = dcinj,
                              interp_vals_mass = interp_vals_mass,
                             interp_grids_mass = interp_grids_mass,
                              verbose=verbose
                             )
    


    if mass_model in ('DP', 'DPUC'): #and interp_vals_mass is None:
        print("Sel. bias: removing jacobian m1, m2 --> log(Mc), logit(q) ")
        # remove jacobian m1, m2 --> log(Mc), logit(q)
        log_p_pop += (- atools.safe_log(m2Src) 
                      - atools.safe_log(at.maximum(m1Src - m2Src, eps))) #atools.safe_log(m1Src-m2Src) 
                      #- at.log1p(zinj) )
        if rate_model in ('DPUC','DPUC-vol'):
                log_p_pop -= at.log1p(zinj) 


    
    log_sel_b = log_p_pop - log_p_draw

    if log_p_incl is not None:
        # print("check in selection bias: log_p_incl")
        # print(log_p_incl)
        # print(log_p_incl.shape)
        # print(log_sel_b.shape.eval())
        log_sel_b = log_sel_b - log_p_incl

    # Ndraw must be a symbolic tensor with a floating dtype for logs
    Ndraw_t = at.as_tensor_variable(Ndraw).astype(m1inj.dtype)
    
    log_mu = at.logsumexp(log_sel_b) - atools.safe_log(Ndraw_t)
    
    logs2 = at.logsumexp(2.0*log_sel_b) - atools.safe_log(Ndraw_t)


    #####################################
    # This is N_eff as in Farr 2019
    #####################################
    ## way 1
    #mu = at.exp(log_mu)
    #muSq = mu*mu
    #s2 = at.exp(  logs2 )
    #sigmaSq = s2 - muSq/Ndraw
    #Neff = muSq/sigmaSq

    ## way 2
    #print("sel_bias_at_vec logs2-2*log_mu " )
    #print((logs2-2*log_mu).eval())
    
    #logNeff = -atools.logdiffexp( logs2-2*log_mu, -atools.safe_log(Ndraw) )


    #####################################
    # This is N_eff as in Talbot Golomb 2023
    # Difference between the two is ~1/N_draw , so negligible for large injection sets
    #####################################

    logNeff = 2*log_mu - logs2 + atools.safe_log(Ndraw_t)

    #####################################
    # This is variance of log l per unit obs as in Talbot Golomb 2023
    #####################################

    var_log_lik_u = atools.logdiffexp( logs2-2*log_mu, 1.) - atools.safe_log(Ndraw_t-1)

    Neff = at.exp(logNeff)
    
    
    return log_mu, Neff, var_log_lik_u



#####################################################



#####################################################
#####################################################


def make_model(  priors,
                 GWData,
                 InjData,
                 ivals={},
                 eps_init = 0.01,
                 sampling_GW = 'gmm',
                 rate_model = 'MD',
                 mass_model = 'PLP',
                 smoothing='LVK',
                 simplex_repair=False,
                 interp_mass = 0,
                 interp_z = 0,
                 has_m2_break = False,
                 spin_model = 'none',
                 spin_inj = 'none',
                 marginal_R0 = True,
                 dLprior = 'none',
                 fix_inj_len = False,
                 chunk_inj = -1,
                 chunk_reduce = False,
                 use_float32 = False,
                 use_float32_bias=False,
                 sel_method='Tobs',
                 N_DP_comp_max = 100,
                 alpha_tail = 0.2,
                 alpha_small = 0.01,
                 L_small_1 = 0.05,
                 L_small_2 = 0.1,
                 s_local = 0.5,
                 find_m_bounds = False,
                 alpha_inv_params = (1, 1),
                 fix_H0 = True,
                fix_Om = True,
               fix_w0 = True,
                 fix_Xi0n = True,
               pade=False,
               zres=150,
                zmin_a=1e-05, zmin_b=1e-03, zmid_b=3.0, zmax_c=10.0, hi_boost=0.20,
                 find_z_bounds = False,
               params_fix=None,
                 Neff_min=4,
                Neff_min_lik=1,
               log_lik_var_min=1,
                 use_sel_spin=True,
                 pop_only = False,
               N_successes_l=None,
               Nsamplesuse = -1,
               include_sel_uncertainty=False,
               sel_smoothing='poly',
               alpha_beta_prior='poly',
               dil_factor=1,
               use_log_alpha_beta=False ,
               allTobs=None,
                 use_updates=True,
                 inj_loop='vec',
                 save_thetas=False,
                 wrap_logp=False,
                 interp_inj=True,
                 param='vanilla',
                 DP_prior='SB',
                 sigma_softmax=0.75,
                 gamma_DP_params = (4, 0.8),
                 is_observed = False,
                 sample_from_pop = False,
                 mmin_inj=-1,
                 is_compressed_inj=False
                ):

    ################################################
    # Read in data and set dimensions
    ################################################

    ## GW data
    if not pop_only:
        # gw data are interpolants of single-event posteriors
        if sampling_GW=='gauss':
            # we sample single-event parameters from broad gaussian approximations of the posteriors
            mus_s, cho_s, log_wts_l, mus_l, icovs_l, log_dets_l, cho_covs_l, Tobs, Nevs = GWData
            wts_l = np.exp(log_wts_l)
            
        elif 'gmm' in sampling_GW or sampling_GW=='gumbel':
            # we sample single-event parameters from the actual single-event posteriors
            wts_l, mus_l, cho_covs_l, Tobs, Nevs = GWData
        else:
            raise ValueError('sampling_GW can be gmm, gmm_cat, gumbel,  gauss ')
            
        

    else:
        # gw data are single-event posterior samples
        # shape of each has to be n_events, n_samples
        m1det, m2det, d, spin_samples, Tobs, allNsamples, where_compute = GWData            

        if Nsamplesuse !=-1 :
            if Nsamplesuse>allNsamples:
                raise ValueError("Must use less samples than those available.")
            print("allNsamples availabe is %s, but %s will be used"%(allNsamples, Nsamplesuse))
            allNsamples =  Nsamplesuse   
            allNsamples_np = allNsamples #allNsamples.eval()
        
        if (spin_model=='default') or (spin_model=='default_gauss'):
           chi1, chi2, cost1, cost2 = spin_samples
        else:
            raise NotImplementedError()

    ## Injections data
    if spin_inj == 'none':
        dLinj, m1inj, m2inj, lpdinj, Ndraw, Ndet, lp_incl_inj = InjData
    elif spin_inj == 'chieffchip':
        dLinj, m1inj, m2inj, chiefffInj, chipInj, lpdinj, Ndraw, Ndet, lp_incl_inj = InjData
    elif (spin_inj == 'chi12xyz' or spin_inj == 'default'):
        if (spin_model=='default') or (spin_model=='default_gauss'):
            dLinj, m1inj, m2inj, chi1Inj, chi2Inj, cost1Inj, cost2Inj, lpdinj, Ndraw, Ndet, lp_incl_inj = InjData
        elif spin_model == 'none':
            dLinj, m1inj, m2inj, lpdinj, Ndraw, Ndet, lp_incl_inj = InjData
            
    ndata = m1inj.shape[0] # number of observing runs to combine
    ndata_np = ndata #ndata.eval()
    ninj = m1inj.shape[1] # max number of injections
    ninj_np = ninj #ninj.eval()

    if not use_sel_spin and spin_model!='none':
        raise ValueError("You are using spin_model=%s but not use_sel_spin. "%spin_model)

    if ndata_np==1:
        
        if use_sel_spin:
            spin_model_name = spin_model
            
            if spin_model == 'chieffchip' or spin_model == 'chieffchip_uc' :
                spinsInj = [ chiefffInj[0], chipInj[0] ]
                
            elif (spin_model == 'default') or (spin_model == 'default_gauss'):
                spinsInj = [ chi1Inj[0], chi2Inj[0], cost1Inj[0], cost2Inj[0] ]
                
            else:
                raise ValueError("use_sel_spin is True, but no valid spin model name was given. Use use_sel_spin=False or provide valid spin model.")
                spinsInj = []
    
        else:
            print("Spin distribution will not be used in the sel effect")
            spinsInj = []
            spin_model_name = 'none'


    
    Ndet_np = Ndet #Ndet.eval()
    N_DP_comp_max_np = N_DP_comp_max #N_DP_comp_max.eval()
    Nevs_np = Nevs #Nevs.eval()

    Tobs_np = Tobs #Tobs.eval()

        
    if not pop_only:
        N = mus_l.shape[0] # number of events in total
        N_np = N #N.eval()
        ngmm = mus_l.shape[1]
        ngmm_np = ngmm #ngmm.eval()
        nd = mus_l.shape[2]
        nd_np = nd #nd.eval()
        print('N:%s, max ngmm: %s, nd: %s '%(N_np, ngmm_np, nd_np))
        print('N evs is %s'%Nevs_np)
        print('Tobs is %s'%Tobs_np)
    else:
        N = m1det.shape[0] # number of events in total
        N_np = N #N.eval()
        Nsamples = m1det.shape[1]
        Nsamples_np = Nsamples #Nsamples.eval()
        print("N samples max will be ")
        print(Nsamples_np)
        print('N:%s, n samples: %s '%(N_np, allNsamples_np))



    
    event_index = np.arange(N_np, dtype=int)
    Ttot = np.sum(Tobs)

    
    print('Injections: :%s, '%(ninj_np))

    print('ninj: :%s, %s datasets,'%(Ndet_np, ndata_np))

    coords = {'event_index': event_index}

    

    if mass_model in ('DP', 'DPUC'):
        coords['component'] = np.arange(N_DP_comp_max_np, dtype=int)
        
        if rate_model in ('DPUC','DPUC-vol'):
            ndim_GMM = 3
        else:
            ndim_GMM = 2

        print('GMM dimension is %s'%ndim_GMM)
        
        coords['GMMdimension'] = np.arange(ndim_GMM, dtype=int)
        coords['GMMdimension_1'] = np.arange(ndim_GMM, dtype=int)
        coords['GMMdimension_2'] = np.arange(ndim_GMM, dtype=int)
        p = ndim_GMM*(ndim_GMM+1)//2  # packed length = 3 for n=2
        
        coords["packed_cholesky"] = np.arange(p)

    if pop_only:
        coords['nsamples'] = np.arange( Nsamples_np, dtype=int )
    else:
         coords['GWdimension'] = np.arange(nd_np, dtype=int)


    if params_fix is None:
        print('No values for parameters to fix passed. Default values will be used. If fixing parameters, check that the values are consistent. Values of fixed parameters:')
        print(PLPeakO3params)
        params_fix=PLPeakO3params


    X = np.float32 if use_float32 else np.float64  # model dtype

    X_name = "float32" if use_float32 else "float64"  # model dtype

    
    if use_float32_bias:
        if not use_float32:
            XI = np.float32
        else:
            XI = X
    else:
        XI = X
    print("Model dtype will be %s"%X)
    print("Injections dtype will be %s"%XI)




    
    if ( find_z_bounds or (mass_model in ('DPUC', 'DP') and find_m_bounds) or mmin_inj!=-1 ):

    
        rng = np.random.default_rng()
        
    
        # --- Compile once: z_from_dL and midpoint derivative ---
        z_sym      = at.dvector('z_nodes')    # if you need it
        d_sym      = at.dvector('dL_nodes')
        H0_sym     = at.dscalar('H0')
        Om_sym     = at.dscalar('Om')
        w0_sym     = at.dscalar('w0')
        Xi0_sym     = at.dscalar('Xi0')
        n_sym     = at.dscalar('nXi0')

        
        # your existing functions but returning NODE arrays
        z_from_dL_sym = atools.z_from_dL_at(d_sym, H0_sym, Om_sym, w0_sym, Xi0_sym, n_sym, interp=pade, param=param)
        #dc_nodes_sym  = atools.dcfun_at(z_sym, H0_sym, Om_sym, w0_const, interp=False)
        #d_log_dLEM_dz_sym = atools.ddL_dz_EM(z_sym, H0_sym, Om_sym, w0_const)
        #lb_mid_fn = pytensor.function([z_sym, H0_sym, Om_sym, ], d_log_dLEM_dz_sym)
        z_from_dL_fn = pytensor.function([d_sym, H0_sym, Om_sym, w0_sym, Xi0_sym, n_sym], z_from_dL_sym)

        
        if fix_H0:
            priors['H0'] = ( params_fix['H0'], params_fix['H0'])
        if fix_Om:
            priors['Om'] = ( params_fix['Om'], params_fix['Om'])
        if fix_w0:
            priors['w0'] = ( -1, -1)
        if fix_Xi0n:
            priors['Xi0'] = ( 1, 1)
            priors['nXi0'] = ( 0, 0)


        # if 'gmm' in sampling_GW:
        #     mus_l_ = mus_l
        #     wts_l_ = wts_l
        #     cho_covs_l_ = cho_covs_l
        
        # elif sampling_GW=='gauss':
        #     mus_l_ = mus_s
        #     wts_l_ = None
        #     cho_covs_l_ = cho_s



        if find_z_bounds:
            print("\nFinding optimal points for redshift interpolation...")
            print("min, max redshift search grid: %s, %s"%(atools.zGridGlobals_at.eval().min(), atools.zGridGlobals_at.eval().max()))
        
            min_z, max_z, z_min_data, z_max_data = putils.find_zgrid_bounds(wts_l, mus_l, cho_covs_l,
                                          priors['H0'], priors['Om'], priors['w0'], priors['Xi0'], priors['nXi0'], 
                                          int(N), int(nd),
                                        dLinj,
                                        z_from_dL_fn,
                                          sampling_GW,
                                          trials=1000, 
                                         )
    
            
            
            zmin_b = max(min_z, z_min_data)
    
            zmin_a = min( zmin_a, min(min_z, z_min_data))
            
            zmid_b = z_max_data
            zmax_c = max(z_max_data, max_z)*(1+0.05)
    
            print("Redshift values found, overwriting default:")
            print("zmin_a=%s, zmin_b=%s, zmid_b=%s, zmax_c=%s"%(zmin_a, zmin_b, zmid_b, zmax_c))


        if (mass_model in ('DPUC', 'DP') and find_m_bounds):

            print("\nFinding prior range for DP-GMM. This will overwrite input arguments.")
       
            scales = putils.find_mass_redshift_bounds(wts_l, mus_l, cho_covs_l,
                                          priors['H0'], priors['Om'], priors['w0'], priors['Xi0'], priors['nXi0'], 
                                          int(N), int(nd),
                                        dLinj,
                            m1inj,
                            m2inj,
                              z_from_dL_fn,
                              sampling_GW,
                              trials=1000, 
                            is_observed = False #is_observed
                          #rng=onp.random.default_rng(123)
                             )
    
            lowmu1 = scales['lMc_min_data'].astype(X)
            upmu1 = scales['lMc_max_data'].astype(X)
    
            lowmu2 = scales['lq_min_data'].astype(X)
            upmu2 = scales['lq_max_data'].astype(X)

            lowmu3 = scales['logz_min_data'].astype(X)
            upmu3 = scales['logz_max_data'].astype(X)


            lowmu1_inj = scales['lMc_min_inj'].astype(X)
            upmu1_inj = scales['lMc_max_inj'].astype(X)
    
            lowmu2_inj = scales['lq_min_inj'].astype(X)
            upmu2_inj = scales['lq_max_inj'].astype(X)

            lowmu3_inj = scales['logz_min_inj'].astype(X)
            upmu3_inj = scales['logz_max_inj'].astype(X)
    
            L_small_1 = scales['lMc_diff'].astype(X)
            L_small_2 = scales['lq_diff'].astype(X)

            L_small_m1 = scales['m1_diff'].astype(X)
            L_small_m2 = scales['m2_diff'].astype(X)
    
            L_small_3 = scales['logz_diff'].astype(X)

            print("Mass/redshift DP-GMM prior values found, overwriting default:")
            print("lowmu1=%s, upmu1=%s, lowmu2=%s, upmu2=%s"%(lowmu1, upmu1, lowmu2, upmu2))
            print("L_small_1=%s, L_small_2=%s, L_small_3=%s"%(L_small_1, L_small_2,L_small_3 ))
            print("L_small_m1=%s, L_small_m2=%s"%(L_small_m1, L_small_m2, ))
        
        if mmin_inj!=-1:
            if 'BNS' in mass_model:
                raise ValueError()
            print("Pre-filtering injections to exclude those with mass<%s solar masses."%mmin_inj)

            dL_min, dL_max = dLinj[0].min(), dLinj[0].max()
            
            # 1) build envelope once 
            dL_grid, zmax_grid = putils.build_zmax_envelope_from_corners(
                z_from_dL_fn, dL_min, dL_max, priors, n_grid=4096
            )
            
            # 2) apply safe filter once
            keep = putils.safe_prefilter_injections_detector_frame(
                m1inj[0], m2inj[0], dLinj[0],
                dL_grid, zmax_grid,
                mmin_src=mmin_inj,
            )
            ninj_or = m1inj.shape[1]
            ninj_new = keep.sum()
            print("Will keep %s injections out of %s"%(ninj_new, ninj_or))

            dLinj, m1inj, m2inj, lpdinj = [ d_[keep] for d_ in dLinj ], [ m_[keep] for m_ in m1inj], [ m_[keep] for m_ in m2inj], [l_[keep] for l_ in lpdinj ]
            spinsInj = [sI[keep] for sI in spinsInj ]
            Ndet[0] = ninj_new

            if is_compressed_inj:
                lp_incl_inj = [ l_[keep] for l_ in lp_incl_inj]
            
    
    if interp_mass!=0:

        print("\nPre-computing mass function on grid for later interpolation. Grid resolution: %s"%interp_mass)

        if interp_mass<100:
                raise ValueError("Use finer grid for accurate mass function.")
        
        tgrid_m1 = np.linspace(0.0, 1.0, interp_mass ).astype(X)
        tgrid_m2 = np.linspace(0.0, 1.0, int(interp_mass/2) ).astype(X)

        if interp_z!=0:
            tgrid_z = np.linspace(0.0, 1.0, interp_z ).astype(X)
            print("Pre-computing rate evolution on grid for later interpolation. Grid resolution: %s"%interp_z)


        if mass_model in ('DPLDP', 'DPLDP-z'):
            if mass_model =='DPLDP':
                sigma_min = min(priors["sigma1"][0], priors["sigma2"][0])
            else:
                sigma_min = min(priors["sigma1_0"][0], priors["sigma2_0"][0])
            MMIN_GRID = 2.
            MMIN_GRID_2 = 1.99

            m1_grid_ = (MMIN_GRID + (300.0 - MMIN_GRID) * tgrid_m1).astype(X)
            m2_grid_ = (MMIN_GRID_2 + (300.0 - MMIN_GRID_2) * tgrid_m2).astype(X)
            
            if mass_model =='DPLDP-z':
                if find_z_bounds:
                    z_bank = (zmin_a + (zmax_c - zmin_a) * tgrid_z).astype(X)
                else:
                    z_bank = (1e-05 + (10. - 1e-05) * tgrid_z).astype(X)
                    
            dx_min_test = np.min(np.diff(m1_grid_))

            if dx_min_test >= 0.5*sigma_min:
                raise ValueError(
                f"Spacing on mass interpolation grid ({dx_min_test:.3g}) is larger than or "
                f"comparable to min prior scale for sigma ({sigma_min:.3g}). "
                "Increase interp_mass or change priors."
            )

        
        
        elif mass_model in ('DPUC', 'DP'):  

            if sel_method=='skip':
            
                MMIN_GRID = lowmu1*(1-0.1)
                MMAX_GRID = upmu1*(1+0.1)

                MMIN_GRID_1 = lowmu2 #*(1-0.1)
                MMAX_GRID_1 = upmu2*(1+0.1)

                MMIN_GRID_2 = lowmu3*(1-0.1)
                MMAX_GRID_2 = upmu3*(1+0.1)
                
            else:
                MMIN_GRID = min(lowmu1_inj, lowmu1)*(1-0.1)
                MMAX_GRID = max(upmu1, upmu1_inj)*(1+0.1)

                MMIN_GRID_1 = min(lowmu2_inj, lowmu2) #*(1-0.1)
                MMAX_GRID_1 = max(upmu2, upmu2_inj)*(1+0.1)

                MMIN_GRID_2 = min(lowmu3_inj, lowmu3)*(1-0.1)
                MMAX_GRID_2 = max(upmu3, upmu3_inj)*(1+0.1)

            
            print("Grid in log(Mc) source between %s and %s"%(MMIN_GRID, MMAX_GRID))
            print("Grid in logit(q) source between %s and %s"%(MMIN_GRID_1, MMAX_GRID_1))
            print("Grid in log(1+z) source between %s and %s"%(MMIN_GRID_2, MMAX_GRID_2))

            log_Mc_grid = np.asarray((MMIN_GRID + (MMAX_GRID - MMIN_GRID) * tgrid_m1)).astype(X)
            logit_q_grid = np.asarray((MMIN_GRID_1 + (MMAX_GRID_1 - MMIN_GRID_1) * tgrid_m1)).astype(X)
            log_1pz_grid = np.asarray((MMIN_GRID_2 + (MMAX_GRID_2 - MMIN_GRID_2) * tgrid_z)).astype(X)
                    
            dx1_min_test = np.min(np.diff(log_Mc_grid))
            dx2_min_test = np.min(np.diff(logit_q_grid))
            dx3_min_test = np.min(np.diff(log_1pz_grid))


            if dx1_min_test >= L_small_m1:
                raise ValueError(
                f"Spacing on log_Mc interpolation grid ({dx1_min_test:.3g}) is larger than or "
                f"comparable to min prior scale for sigma ({L_small_m1:.3g}). "
                "Increase interp_mass or change priors."
            )

            if dx2_min_test >= L_small_m2:
                raise ValueError(
                f"Spacing on logit_q interpolation grid ({dx2_min_test:.3g}) is larger than or "
                f"comparable to min prior scale for sigma ({L_small_m2:.3g}). "
                "Increase interp_mass or change priors."
            )

            if dx3_min_test >= L_small_3:
                raise ValueError(
                f"Spacing on log(1+z) interpolation grid ({dx3_min_test:.3g}) is larger than or "
                f"comparable to min prior scale for sigma ({L_small_3:.3g}). "
                "Increase interp_mass or change priors."
            )
        else:
            raise ValueError('Interpolation not available for this mass model.')

    if is_observed:
        print("Building optimal SNR interpolant...")

        # load interpolant
        with h5py.File('../tables/optimal_snr_aplus_design_05.h5','r') as f:
            m_grid_at = at.as_tensor_variable(np.array(f['ms']))
            osnrs_grid_at = at.as_tensor_variable(np.array(f['SNR']))
            ref_dist_Gpc_at = at.as_tensor_variable(np.array(1.))
        grid_at = (m_grid_at, m_grid_at)
        osnr_interp_at = atools.GridInterpolator_at(grid_at, osnrs_grid_at)
                  
    if sample_from_pop:
        print("Finding init vals for individual event params...")

        rng = np.random.default_rng()
        x = rng.standard_normal(size=(N, nd))
        samples_init = putils.sample_from_per_event_gmm(wts_l, mus_l, cho_covs_l, x, rng=None)

        log_Mc_det_init = samples_init[:, 0]
        logit_q_init = samples_init[:, 1].astype(X)
        logd_init = samples_init[:, 2]

        z_init = (atools.z_from_dL_at(at.exp(logd_init), 67.7, 0.31, -1, 1, 0)).eval().astype(X)
        log_onepz_init = np.log1p(z_init).astype(X)
        log_Mc_src_init = (log_Mc_det_init - log_onepz_init).astype(X)
        
    ################################################
    # Build model
    ################################################
    
    with pm.Model(coords=coords) as model:


        if sampling_GW=='gauss':
            
            # we sample single-event parameters from broad gaussian approximations of the posteriors
            mus_s, cho_s, log_wts_l, mus_l, icovs_l, log_dets_l = at.as_tensor_variable(mus_s), at.as_tensor_variable(cho_s), at.as_tensor_variable(log_wts_l), at.as_tensor_variable(mus_l), at.as_tensor_variable(icovs_l), at.as_tensor_variable(log_dets_l)

            
        elif 'gmm' in sampling_GW:
            # we sample single-event parameters from the actual single-event posteriors
            wts_l, mus_l, cho_covs_l = at.as_tensor_variable(wts_l), at.as_tensor_variable(mus_l), at.as_tensor_variable(cho_covs_l)

        ################################################
        # Cosmological parameters
        ################################################

        
        if fix_H0:
            H0_ =  params_fix['H0']
        else:
            H0_ =  pm.Uniform('H0', lower=priors['H0'][0], upper=priors['H0'][1], initval=ivals.get('H0'))
        
        if fix_Om:
            Om_ = params_fix['Om']
        else:
            Om_ = pm.Uniform('Om', lower=priors['Om'][0], upper=priors['Om'][1], initval=ivals.get('Om')) 

        if fix_w0:
            w0_ = -1.
        else:
            if pade:
                raise NotImplementedError("Pade appproximation with varying w0 not implemented yet. Use pade=False")
            w0_ =  pm.Uniform('w0', lower=priors['w0'][0], upper=priors['w0'][1], initval=ivals.get('w0'))
            
        
        if fix_Xi0n:
            Xi0_ =  1.
            nXi0_ = 0.
        else:
            Xi0_ =  pm.Uniform('Xi0', lower=priors['Xi0'][0], upper=priors['Xi0'][1], initval=ivals.get('Xi0'))
            nXi0_ = pm.Uniform('nXi0', lower=priors['nXi0'][0], upper=priors['nXi0'][1], initval=ivals.get('nXi0')) 

            print("For Xi0-n, we use the %s parameterization"%param)


        Lambda_ = [H0_, Om_, w0_, Xi0_, nXi0_]

        ################################################
        # Redshift evolution of merger rate
        ################################################
        
        if rate_model=='MD':
            
            print('Modeling evolution of merger rate with redshift with Madau-Dickinson profile')
            
            gamma_ = pm.Uniform('gamma', lower=priors['gamma'][0], upper=priors['gamma'][1], initval=ivals.get('gamma'))    
            kappa_ = pm.Uniform('kappa', lower=priors['kappa'][0], upper=priors['kappa'][1], initval=ivals.get('kappa'))
            zp_ = pm.Uniform('zp', lower=priors['zp'][0], upper=priors['zp'][1], initval=ivals.get('zp'))

            # gamma_ = atools.uniform_unconstrained("gamma",  priors['gamma'][0], priors['gamma'][1], init=ivals.get("gamma"))
            # kappa_ = atools.uniform_unconstrained("kappa",  priors['kappa'][0], priors['kappa'][1], init=ivals.get("kappa"))
            # zp_ = atools.uniform_unconstrained("zp",  priors['zp'][0], priors['zp'][1], init=ivals.get("zp"))
            
            Lambda_ += [gamma_, kappa_, zp_]

        elif rate_model=='PL':
            print('Modeling evolution of merger rate with a power law')
            gamma_ = pm.Uniform('gamma', lower=priors['gamma'][0], upper=priors['gamma'][1], initval=ivals.get('gamma'))

            Lambda_ += [gamma_]

        elif rate_model in ('DPUC', 'DPUC-vol'):

            assert mass_model in ('DP', 'DPUC')
            print('Modeling evolution of merger rate with a DP-GMM together with mass')

        ################################################
        # Spin
        ################################################

        if spin_model == 'chieffchip':
            print('Modeling spin distribution with a gaussian in chieff-chip')
            muEff_ = pm.Uniform('muEff', lower=priors['muEff'][0], upper=priors['muEff'][1])
            sigEff_ = pm.Uniform('sigEff', lower=priors['sigEff'][0], upper=priors['sigEff'][1])
            muP_ = pm.Uniform('muP', lower=priors['muP'][0], upper=priors['muP'][1])
            sigP_ = pm.Uniform('sigP', lower=priors['sigP'][0], upper=priors['sigP'][1])
            rho_ = pm.Uniform('rho', lower=priors['rho'][0], upper=priors['rho'][1])

            Lambda_ += [muEff_, sigEff_, muP_, sigP_, rho_]

        elif spin_model=='chieffchip_uc':

            print('Modeling spin distribution with uncorrelated gaussians in chieff-chip')
            muEff_ = pm.Uniform('muEff', lower=priors['muEff'][0], upper=priors['muEff'][1])
            sigEff_ = pm.Uniform('sigEff', lower=priors['sigEff'][0], upper=priors['sigEff'][1])
            muP_ = pm.Uniform('muP', lower=priors['muP'][0], upper=priors['muP'][1])
            sigP_ = pm.Uniform('sigP', lower=priors['sigP'][0], upper=priors['sigP'][1])

            Lambda_ += [muEff_, sigEff_, muP_, sigP_]

        elif spin_model=='default':

            print('Modeling spin distribution with default spin model')

            if not use_log_alpha_beta:
                muChi_ = pm.Uniform('muChi', lower=priors['muChi'][0], upper=priors['muChi'][1])
                varChi_ = pm.Uniform('varChi', lower=priors['varChi'][0], upper=priors['varChi'][1])
                zeta_ = pm.Uniform('zeta', lower=priors['zeta'][0], upper=priors['zeta'][1])
                sigmat_ = pm.Uniform('sigmat', lower=priors['sigmat'][0], upper=priors['sigmat'][1])
    
                kappa_ = muChi_*(1-muChi_)/varChi_-1
    
                alphaChi_ = pm.Deterministic('alphaChi',  muChi_*kappa_ )
                betaChi_ = pm.Deterministic('betaChi',  (1-muChi_)*kappa_ )
                stdChi_ = pm.Deterministic('stdChi',  at.sqrt(varChi_) )
    
    
                Lambda_ += [alphaChi_, betaChi_, zeta_, sigmat_]
    
                # Bound alpha, beta > 1    
                
                if alpha_beta_prior=='poly':
                    print("Tapering prior on alpha_chi, beta_chi with polynomial smoothing")
                    _ = pm.Potential('bound_alphaChi', atools.log_f_smooth_poly(alphaChi_, 5e-4,  1 )  )
                    _ = pm.Potential('bound_betaChi', atools.log_f_smooth_poly(betaChi_, 5e-4,  1  ))
                elif alpha_beta_prior=='sigmoid':
                    print("Tapering prior on alpha_chi, beta_chi with sigmoid smoothing")
                    _ = pm.Potential('bound_alphaChi', atools.log_sigmoid(alphaChi_,  1+3e-04, 1e-04)  )
                    _ = pm.Potential('bound_betaChi', atools.log_sigmoid(betaChi_, 1+3e-04, 1e-04)  )
                else:
                    print("Putting prior on alpha_chi, beta_chi with hard cut")
                    _ = pm.Potential('bound_alphaChi', at.switch( at.le(alphaChi_, 1. ), -np.inf, at.as_tensor_variable(0.) ) )
                    _ = pm.Potential('bound_betaChi', at.switch( at.le(betaChi_, 1. ), -np.inf, 0.0 ) )
        
            else:
                # still to be tested. Might improve sampling/divergences
                print("Sampling in log(alpha-1), log(beta-1)")
                raise NotImplementedError()
                
        elif spin_model=='default_gauss':

            print('Modeling spin distribution with default spin model, gaussian distribution for magnitudes')

            muChi_ = pm.Uniform('muChi', lower=priors['muChi'][0], upper=priors['muChi'][1])
            sigmaChi_ = pm.Uniform('sigmaChi', lower=priors['sigmaChi'][0], upper=priors['sigmaChi'][1])
            
            zeta_ = pm.Uniform('zeta', lower=priors['zeta'][0], upper=priors['zeta'][1])
            sigmat_ = pm.Uniform('sigmat', lower=priors['sigmat'][0], upper=priors['sigmat'][1])

            Lambda_ += [muChi_, sigmaChi_, zeta_, sigmat_]
            
        else:
            print('No model of the spin distribution.')
                

            

        ################################################
        # Mass distribution
        ################################################
            
        if mass_model=='PLPreg':

            ### BBH
            
            # Power law + peak
            print('Modeling mass distribution with LVK Power Law + Peak with regularized edge')
            if smoothing=='LVK':
                print('Using LVK smoothing')
            elif smoothing=='poly':
                print('using differentiable polynomial smoothing')
            
            lamP_   = pm.Uniform("lambdaPeak", lower=priors["lambdaPeak"][0], upper=priors["lambdaPeak"][1], initval=ivals.get("lambdaPeak"))        
            alpha_  = pm.Uniform("alpha",      lower=priors["alpha"][0],      upper=priors["alpha"][1],      initval=ivals.get("alpha"))
            beta_   = pm.Uniform("beta",       lower=priors["beta"][0],       upper=priors["beta"][1],       initval=ivals.get("beta"))
            ml_     = pm.Uniform("ml",         lower=priors["ml"][0],         upper=priors["ml"][1],         initval=ivals.get("ml"))
            mh_     = pm.Uniform("mh",         lower=priors["mh"][0],         upper=priors["mh"][1],         initval=ivals.get("mh"))
            deltam_ = pm.Uniform("deltam",     lower=priors["deltam"][0],     upper=priors["deltam"][1],     initval=ivals.get("deltam"))
            muM_    = pm.Uniform("muMass",     lower=priors["muMass"][0],     upper=priors["muMass"][1],     initval=ivals.get("muMass"))
            sM_     = pm.Uniform("sigmaMass",  lower=priors["sigmaMass"][0],  upper=priors["sigmaMass"][1],  initval=ivals.get("sigmaMass"))

             #lamP_ = atools.uniform_unconstrained("lambdaPeak",  priors['lambdaPeak'][0], priors['lambdaPeak'][1], init=ivals.get("lambdaPeak"))
            # alpha_  = atools.uniform_unconstrained("alpha",     priors["alpha"][0],     priors["alpha"][1],     init=ivals.get("alpha"))
            # beta_   = atools.uniform_unconstrained("beta",      priors["beta"][0],      priors["beta"][1],      init=ivals.get("beta"))
            # ml_     = atools.uniform_unconstrained("ml",        priors["ml"][0],        priors["ml"][1],        init=ivals.get("ml"))
            # mh_     = atools.uniform_unconstrained("mh",        priors["mh"][0],        priors["mh"][1],        init=ivals.get("mh"))
            # deltam_ = atools.uniform_unconstrained("deltam",    priors["deltam"][0],    priors["deltam"][1],    init=ivals.get("deltam"))
            # muM_    = atools.uniform_unconstrained("muMass",    priors["muMass"][0],    priors["muMass"][1],    init=ivals.get("muMass"))
            # sM_     = atools.uniform_unconstrained("sigmaMass", priors["sigmaMass"][0], priors["sigmaMass"][1], init=ivals.get("sigmaMass"))

            Lambda_ += [lamP_, alpha_, beta_, deltam_, ml_, mh_, muM_, sM_ ]


        elif mass_model=='DPLDP':

            print('Modeling mass distribution with Double Power Law + Double Peak ')

            alpha1_   = pm.Uniform("alpha1",   lower=priors["alpha1"][0],   upper=priors["alpha1"][1],   initval=ivals.get("alpha1"))
            alpha2_   = pm.Uniform("alpha2",   lower=priors["alpha2"][0],   upper=priors["alpha2"][1],   initval=ivals.get("alpha2"))
            mb_       = pm.Uniform("mb",       lower=priors["mb"][0],       upper=priors["mb"][1],       initval=ivals.get("mb"))
            mu1_      = pm.Uniform("mu1",      lower=priors["mu1"][0],      upper=priors["mu1"][1],      initval=ivals.get("mu1"))
            sigma1_   = pm.Uniform("sigma1",   lower=priors["sigma1"][0],   upper=priors["sigma1"][1],   initval=ivals.get("sigma1"))
            mu2_      = pm.Uniform("mu2",      lower=priors["mu2"][0],      upper=priors["mu2"][1],      initval=ivals.get("mu2"))
            sigma2_   = pm.Uniform("sigma2",   lower=priors["sigma2"][0],   upper=priors["sigma2"][1],   initval=ivals.get("sigma2"))
            u         = pm.Uniform("u", 0, 1, initval=ivals.get("u"))
            m1_low_   = pm.Deterministic("m1_low", 3 + (10 - 3) * at.sqrt(u))
            v         = pm.Uniform("v", 0, 1, initval=ivals.get("v"))
            m2_low_   = pm.Deterministic("m2_low", 3 + v * (m1_low_ - 3))
            m_high_   = pm.Deterministic("m_high", at.as_tensor_variable(300.0).astype(X)  )
            delta_m1_ = pm.Uniform("delta_m1", lower=priors["delta_m1"][0], upper=priors["delta_m1"][1], initval=ivals.get("delta_m1"))
            lambda_vec = pm.Dirichlet("lambda", a=np.asarray([1, 1, 1], dtype=X), initval=np.asarray(ivals.get("lambda"), dtype=X))
            lambda0_  = pm.Deterministic("lambda0", lambda_vec[0])
            lambda1_  = pm.Deterministic("lambda1", lambda_vec[1])
            lambda2_  = pm.Deterministic("lambda2", lambda_vec[2])
            beta_     = pm.Uniform("beta",     lower=priors["beta"][0],     upper=priors["beta"][1],     initval=ivals.get("beta"))
            delta_m2_ = pm.Uniform("delta_m2", lower=priors["delta_m2"][0], upper=priors["delta_m2"][1], initval=ivals.get("delta_m2"))
            epsilon_  = pm.Deterministic("epsilon", at.as_tensor_variable(0.01))
            if has_m2_break:
                print("Including gap for secondary mass")
                m_g_     =  pm.Uniform("m_g", lower=priors["m_g"][0], upper=priors["m_g"][1], initval=ivals.get("m_g")) 
                w_g_     = pm.Uniform("w_g", lower=priors["w_g"][0], upper=priors["w_g"][1], initval=ivals.get("w_g")) 
                sig_g_l_ = at.as_tensor_variable(1e-02).astype(X)
                sig_g_h_ = at.as_tensor_variable(1e-02).astype(X)
            else:
                m_g_     = at.as_tensor_variable(45.).astype(X)
                w_g_     = at.as_tensor_variable(70.).astype(X)
                sig_g_l_ = at.as_tensor_variable(1e-02).astype(X)
                sig_g_h_ = at.as_tensor_variable(1e-02).astype(X)
            
            Lambda_ += [alpha1_, alpha2_, mb_, mu1_, sigma1_, mu2_, sigma2_, m1_low_, m_high_, delta_m1_, lambda0_, lambda1_, beta_, m2_low_, delta_m2_, epsilon_, m_g_, w_g_, sig_g_l_, sig_g_h_]

        
        
        elif mass_model=='DPLDP-z':


            print("Modeling mass distribution with DPLDP + redshift-evolving hyperparameters")

            # -------------------------
            # Low-z (z≈0) hyperparameters (same as before)
            # -------------------------
            alpha1_0  = pm.Uniform("alpha1_0",  lower=priors["alpha1_0"][0],  upper=priors["alpha1_0"][1],  initval=ivals.get("alpha1_0"))
            alpha2_0  = pm.Uniform("alpha2_0",  lower=priors["alpha2_0"][0],  upper=priors["alpha2_0"][1],  initval=ivals.get("alpha2_0"))
            mb_0      = pm.Uniform("mb_0",      lower=priors["mb_0"][0],      upper=priors["mb_0"][1],      initval=ivals.get("mb_0"))
            mu1_0     = pm.Uniform("mu1_0",     lower=priors["mu1_0"][0],     upper=priors["mu1_0"][1],     initval=ivals.get("mu1_0"))
            sigma1_0  = pm.Uniform("sigma1_0",  lower=priors["sigma1_0"][0],  upper=priors["sigma1_0"][1],  initval=ivals.get("sigma1_0"))
            mu2_0     = pm.Uniform("mu2_0",     lower=priors["mu2_0"][0],     upper=priors["mu2_0"][1],     initval=ivals.get("mu2_0"))
            sigma2_0  = pm.Uniform("sigma2_0",  lower=priors["sigma2_0"][0],  upper=priors["sigma2_0"][1],  initval=ivals.get("sigma2_0"))
            delta_m1_ = pm.Uniform("delta_m1",  lower=priors["delta_m1"][0],upper=priors["delta_m1"][1],initval=ivals.get("delta_m1"))
            
            # m1_low, m2_low, m_high as in your original block
            u        = pm.Uniform("u", 0, 1, initval=ivals.get("u"))
            m1_low_  = pm.Deterministic("m1_low", 3 + (10 - 3) * at.sqrt(u))
            v        = pm.Uniform("v", 0, 1, initval=ivals.get("v"))
            m2_low_  = pm.Deterministic("m2_low", 3 + v * (m1_low_ - 3))
            m_high_  = pm.Deterministic("m_high", at.as_tensor_variable(300.0).astype(X))
            

            
            # secondary-mass hyperparams (unchanged unless you also evolve them)
            beta_     = pm.Uniform("beta",     lower=priors["beta"][0],     upper=priors["beta"][1],     initval=ivals.get("beta"))
            delta_m2_ = pm.Uniform("delta_m2", lower=priors["delta_m2"][0], upper=priors["delta_m2"][1], initval=ivals.get("delta_m2"))
            epsilon_  = pm.Deterministic("epsilon", at.as_tensor_variable(0.01).astype(X))
            
            if has_m2_break:
                print("Including gap for secondary mass")
                m_g_     = pm.Uniform("m_g", lower=priors["m_g"][0], upper=priors["m_g"][1], initval=ivals.get("m_g"))
                w_g_     = pm.Uniform("w_g", lower=priors["w_g"][0], upper=priors["w_g"][1], initval=ivals.get("w_g"))
                sig_g_l_ = at.as_tensor_variable(1e-02).astype(X)
                sig_g_h_ = at.as_tensor_variable(1e-02).astype(X)
            else:
                m_g_     = at.as_tensor_variable(45.).astype(X)
                w_g_     = at.as_tensor_variable(70.).astype(X)
                sig_g_l_ = at.as_tensor_variable(1e-02).astype(X)
                sig_g_h_ = at.as_tensor_variable(1e-02).astype(X)



            # # mixture weights at z≈0
            eps_w = at.as_tensor_variable(1e-12).astype(X)
            # endpoints
            lambda_vec0 = pm.Dirichlet(
                "lambda0_vec",
                a=np.asarray([1, 1, 1], dtype=X),
                initval=np.asarray(ivals.get("lambda"), dtype=X)
            )

            lambda0_0 = pm.Deterministic("lambda0_0", lambda_vec0[0])
            lambda1_0 = pm.Deterministic("lambda1_0", lambda_vec0[1])
            lambda2_0 = pm.Deterministic("lambda2_0", lambda_vec0[2])
            
            # -------------------------
            # Redshift evolution hyperparameters for each θ in:
            # {alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, lambda0, lambda1}
            #
            # Each θ has (θ_inf, z_t, Δz) with θ(z) = θ0 + (θ_inf-θ0)*S((z-z_t)/Δz)
            # -------------------------
            
            # helper: choose priors for (z_t, Δz); you can swap these for your own
            z_t_prior = priors.get("z_t", (0.05, 2.5))
            dz_prior  = priors.get("dz",  (0.01, 2.0))   # Δz > 0; adjust as you like


            # pick priors for the high-z asymptotes; by default reuse the low-z prior ranges
            alpha1_inf_,  z_alpha1_,  dz_alpha1_  = putils.evo_triplet("alpha1", ivals, z_t_prior=z_t_prior, dz_prior=dz_prior, theta0_init=ivals.get("alpha1_0"),
                                                                theta_inf_prior=priors["alpha1_0"])
            alpha2_inf_,  z_alpha2_,  dz_alpha2_  = putils.evo_triplet("alpha2", ivals, z_t_prior=z_t_prior, dz_prior=dz_prior, theta0_init=ivals.get("alpha2_0"),
                                                                theta_inf_prior=priors["alpha2_0"])
            mb_inf_,      z_mb_,      dz_mb_      = putils.evo_triplet("mb",  ivals, z_t_prior=z_t_prior, dz_prior=dz_prior,    theta0_init=ivals.get("mb_0"),
                                                                theta_inf_prior=priors["mb_0"])
            mu1_inf_,     z_mu1_,     dz_mu1_     = putils.evo_triplet("mu1",  ivals, z_t_prior=z_t_prior, dz_prior=dz_prior,   theta0_init=ivals.get("mu1_0"),
                                                                theta_inf_prior=priors["mu1_0"])
            sigma1_inf_,  z_sigma1_,  dz_sigma1_  = putils.evo_triplet("sigma1", ivals, z_t_prior=z_t_prior, dz_prior=dz_prior, theta0_init=ivals.get("sigma1_0"),
                                                                theta_inf_prior=priors["sigma1_0"])
            mu2_inf_,     z_mu2_,     dz_mu2_     = putils.evo_triplet("mu2",  ivals,  z_t_prior=z_t_prior, dz_prior=dz_prior,  theta0_init=ivals.get("mu2_0"),
                                                                theta_inf_prior=priors["mu2_0"])
            sigma2_inf_,  z_sigma2_,  dz_sigma2_  = putils.evo_triplet("sigma2", ivals, z_t_prior=z_t_prior, dz_prior=dz_prior, theta0_init=ivals.get("sigma2_0"),
                                                                theta_inf_prior=priors["sigma2_0"])
            
            # Mixture weights at high z: use a Dirichlet, then map to (lambda0_inf, lambda1_inf)
            lambda_vec_inf = pm.Dirichlet("lambda_inf_vec", a=np.asarray([1, 1, 1], dtype=X),
                                          initval=np.asarray(ivals.get("lambda_inf_vec", [1/3, 1/3, 1/3]), dtype=X))
            lambda0_inf_ = pm.Deterministic("lambda0_inf", lambda_vec_inf[0])
            lambda1_inf_ = pm.Deterministic("lambda1_inf", lambda_vec_inf[1])
            lambda2_inf_ = pm.Deterministic("lambda2_inf", lambda_vec_inf[2])


            
            # Allow separate (z_t, Δz) for lambda0 and lambda1 
            z_lambda0_  = pm.Uniform("z_lambda0",  lower=z_t_prior[0], upper=z_t_prior[1],
                                     initval=ivals.get("z_lambda0", None))
            dz_lambda0_ = pm.Uniform("dz_lambda0", lower=dz_prior[0],  upper=dz_prior[1],
                                     initval=ivals.get("dz_lambda0", None))
            z_lambda1_  = pm.Uniform("z_lambda1",  lower=z_t_prior[0], upper=z_t_prior[1],
                                     initval=ivals.get("z_lambda1", None))
            dz_lambda1_ = pm.Uniform("dz_lambda1", lower=dz_prior[0],  upper=dz_prior[1],
                                     initval=ivals.get("dz_lambda1", None))
            
            if simplex_repair:
                print("Will enforce lambda0(z), lambda1(z), lambda2(z) on the simplex")
            
            # -------------------------
            # Pack hyperparameters for your logpdf_DPLDP_z wrapper
            #   - low-z vector: same order you used before, but with *_0 values
            #   - evolution params: (theta_inf, z_theta, dz_theta) for each evolving parameter
            # -------------------------
            lambdaBBHmass_lowz_ = [
                alpha1_0, alpha2_0, mb_0,
                mu1_0, sigma1_0, mu2_0, sigma2_0,
                m1_low_, m_high_, delta_m1_,
                lambda0_0, lambda1_0,
                beta_, m2_low_, delta_m2_,
                epsilon_, m_g_, w_g_, sig_g_l_, sig_g_h_
            ]
            
            evo_params_ = [
                alpha1_inf_,  z_alpha1_,  dz_alpha1_,
                alpha2_inf_,  z_alpha2_,  dz_alpha2_,
                mb_inf_,      z_mb_,      dz_mb_,
                mu1_inf_,     z_mu1_,     dz_mu1_,
                sigma1_inf_,  z_sigma1_,  dz_sigma1_,
                mu2_inf_,     z_mu2_,     dz_mu2_,
                sigma2_inf_,  z_sigma2_,  dz_sigma2_,
                lambda0_inf_, z_lambda0_, dz_lambda0_,
                lambda1_inf_, z_lambda1_, dz_lambda1_,
            ]
            
            # If your code expects a single list Lambda_, append both
            Lambda_ += [*lambdaBBHmass_lowz_, *evo_params_]
            
        
        
        
        ### BNS
        elif 'BNSgauss' in mass_model:

            if mass_model=='BNSgauss':
                # Uncorrelated gaussians
                print('Modeling mass distribution with uncorrelated gaussian distributions')
            elif mass_model=='BNSgaussCond':
                # Conditioned gaussians
                print('Modeling mass distribution with gaussian distributions with p(m1, m2) = p(m1) p(m2|m1) H(m1-m2)')
                
            muM_ = pm.Uniform('muMass', lower=priors['muMass'][0], upper=priors['muMass'][1])
            sM_ = pm.Uniform('sigmaMass', lower=priors['sigmaMass'][0], upper=priors['sigmaMass'][1] )  
            Lambda_ += [muM_, sM_ ]

        ### Non - parametric
        elif mass_model in ('DPUC', 'DP'):

            print("Modeling mass distribution as Dirichelet Process. Max number of components: %s"%N_DP_comp_max)

            if DP_prior=='SB':

                print("Prior for the process is stick-breaking")
                #### Stick Breaking Prior
                alpha_inv_init = alpha_inv_params[0] / alpha_inv_params[1]
                alpha_inv = pm.Gamma("alpha_inv", alpha_inv_params[0], alpha_inv_params[1], initval=alpha_inv_init )
                print("alpha_inv prior has parameters %s"%str(alpha_inv_params))
                alpha = 1/alpha_inv
    
                beta_init = np.full(N_DP_comp_max_np, 1e-02).astype(X)
                #beta_init[0] = 0.99
    
                beta = pm.Beta("beta", 1.0, alpha, dims="component" , initval=beta_init)
                w = pm.Deterministic("w", atools.stick_breaking(beta), dims="component")

            elif DP_prior=='dirichelet':
                print("Prior for the process is dirichelet")

                print("alpha_total prior is Gamma with parameters %s"%str(gamma_DP_params))
                
                ### Dirichelet Prior
                alpha_total = pm.Gamma("alpha_total", alpha=gamma_DP_params[0], beta=gamma_DP_params[1])  # mean ≈ 5
                a = alpha_total / N_DP_comp_max
                w = pm.Dirichlet("w", a=at.ones(N_DP_comp_max) * a, dims="component")

            elif DP_prior=='softmax':
                print("Prior for the process is softmax")
                print("sigma_w sampled from halfnormal with std=%s"%sigma_softmax)
                
                ### Uniform Prior
                sigma_w = pm.HalfNormal("sigma_w", sigma=sigma_softmax)
                raw_w = pm.Normal("raw_w", 0, sigma_w, dims="component")  # small variance
                w = pm.Deterministic("w", pm.math.softmax(raw_w), dims="component")

            else:
                raise ValueError()

            
            logw = atools.safe_log(w)
        

            #### Mean prior 

            # DPLDP 1k
            # lowmu1 = 1.5
            # upmu1 = 5.5
            # lowmu2 =  -1.2
            # upmu2 =  10.

            U1, U2 = (upmu1-lowmu1) , (upmu2-lowmu2)    # "too-wide" typical std per dim 

            mu1_center = (lowmu1 + upmu1) / 2.0  # 3.55
            mu2_center = (lowmu2 + upmu2) / 2.0
            
            
     
            mu1 = pm.Uniform('mulMc', lower=lowmu1, upper=upmu1, dims= ("component" ), initval=np.full(N_DP_comp_max_np, mu1_center).astype(X) )
            mu2 = pm.Uniform('mulq', lower=lowmu2, upper=upmu2, dims= ("component" ), initval=np.full(N_DP_comp_max_np, mu2_center).astype(X))

            if rate_model in ('DPUC','DPUC-vol' ):
                mu3_center = ( lowmu3+ upmu3) / 2.0
                mu3 = pm.Uniform('mulz', lower=lowmu3, upper=upmu3, dims= ("component" ), initval=np.full(N_DP_comp_max_np, mu3_center).astype(X))

                mus = at.stack([mu1, mu2, mu3], axis=0)
                
            else:
                mus = at.stack([mu1, mu2], axis=0)     
                
            

            mu = pm.Deterministic("mu", mus, dims=("GMMdimension", "component") )

            
            #### Sigma prior 
            
            print("L_small_1 = %s "%L_small_1)
            print("L_small_2 = %s "%L_small_2)

            print("U1 = %s "%U1)
            print("U2 = %s "%U2)


            # # Fréchet shape for 1D marginal: alpha = d/2 with d=1 -> 0.5
            # print("P( sigma < L_small ) = %s "%alpha_small)

            # alpha_shape = 0.5
            #lambda_ell_1 = -atools.safe_log(alpha_small) * L_small_1**(alpha_shape) # small scale
            #lambda_ell_2 = -atools.safe_log(alpha_small) * L_small_2**(alpha_shape) # small scale
            
            # tau1 = pm.CustomDist("tau1", lambda_ell_1, 1,
            #               logp=atools.frechet_logp_full,
            #               transform=tr.log, initval=0.2,
            #               random=atools.frechet_random, )

            # tau2 = pm.CustomDist("tau2", lambda_ell_2, 1,
            #               logp=atools.frechet_logp_full,
            #               transform=tr.log, initval=0.2,
            #               random=atools.frechet_random, )

            tau1 = pm.Uniform("tau1", lower=L_small_1, upper=U1, ) #initval= (U1 / 4.0 ).astype(X)  )
            tau2 = pm.Uniform("tau2", lower=L_small_2, upper=U2, ) #initval= (U2 / 4.0 ).astype(X)  )

            print("s_local = %s "%s_local)

            # eps1 = pm.Normal("eps1", 0.0, s_local, dims=("component",))
            # eps2 = pm.Normal("eps2", 0.0, s_local, dims=("component",))

            eps1 = pm.SkewNormal("eps1", mu=0, sigma=s_local, alpha=+2, dims=("component",), initval=np.zeros(N_DP_comp_max_np).astype(X) )
            eps2 = pm.SkewNormal("eps2", mu=0, sigma=s_local, alpha=+2, dims=("component",), initval=np.zeros(N_DP_comp_max_np).astype(X))


            sig1 = pm.Deterministic("sig1", tau1 * at.exp(eps1) , dims="component")   
            sig2 = pm.Deterministic("sig2", tau2 * at.exp(eps2), dims="component")  

            
            if rate_model in ('DPUC', 'DPUC-vol'):

                
                U3 = (upmu3-lowmu3)

                print("L_small_3 = %s "%L_small_3)
                print("U3 = %s "%U3)

                tau3 = pm.Uniform("tau3", lower=L_small_3, upper=U3, )
                eps3 = pm.SkewNormal("eps3", mu=0, sigma=s_local, alpha=+2, dims=("component",), initval=np.zeros(N_DP_comp_max_np).astype(X))
                sig3 = pm.Deterministic("sig3", tau3 * at.exp(eps3), dims="component")  

                sigs = at.stack([sig1, sig2, sig3], axis=0)
                
            else:
                sigs = at.stack([sig1, sig2], axis=0)

            if alpha_tail!=-1:

                # ----- Penalize large sigma -----
                
                
                print("P(tau_1,2 > U_1,2) = %s "%alpha_tail)
                
                lambda_large_1 = -np.log(alpha_tail) / U1   
                lambda_large_2 = -np.log(alpha_tail) / U2   
    
    
                _ = pm.Potential( "pc_large_ell_1", -lambda_large_1 * tau1,  )
                _ = pm.Potential( "pc_large_ell_2", -lambda_large_2 * tau2, )

            
            if mass_model=='DPUC':
                print("No m1-m2 correlation.")
                
                sd = pm.Deterministic("sig", sigs, dims=("GMMdimension", "component"))

                Lambda_ += [ w, mu, sd, logw ]

            elif mass_model=='DP':
                print("Including m1-m2 correlation.")
                # -------- Correlation prior --------

                eta=1.
                print("eta = %s"%eta)
                rho_u = pm.Beta("rho_u", alpha=eta, beta=eta, dims=("component",))

                # #rho_max = 0.9  # cap on |rho|
                # # choose fraction f of L_small you allow for the minor axis
                f = 0.5   # minor axis at least 100xf% of L_small in worst case
                rho_max = np.sqrt(1.0 - f**2)  # ≈ 0.866
                print("rho_max = %s, with f=%s, i.e minor axis is at least %s of L_small in worst case"%(rho_max,f,f))
                rho   = pm.Deterministic("rho", rho_max * (2.0 * rho_u - 1.0), dims="component")

                # rho = pm.Uniform("rho", lower=-rho_max, upper=rho_max, dims="component")
                # pm.Potential(
                #     "lkj_corr_prior",
                #     (eta - 1.0) * atools.safe_log(1.0 - rho**2).sum()
                # )
    
                # # Useful terms
                one_minus_r2 = 1.0 - rho**2
                sqrt1mr2     = at.sqrt(one_minus_r2)
                
                # ----- Cholesky of Σ (for reference / if you need solves) -----
                # Σ = [[s1^2, ρ s1 s2], [ρ s1 s2, s2^2]]
                # Cholesky L = diag([s1, s2]) @ [[1, 0], [ρ, sqrt(1-ρ^2)]]
                row0 = at.stack([sig1,               at.zeros_like(sig1)], axis=1)          # (K,2)
                row1 = at.stack([rho * sig2,         sig2 * sqrt1mr2     ], axis=1)          # (K,2)
                L    = at.stack([row0, row1], axis=1)     
                Cho_cov = pm.Deterministic("Cho_cov", L, dims=("component","GMMdimension","GMMdimension_1"))
                
                # ----- log |Σ^{-1}| (no inverses) -----
                # det Σ = s1^2 * s2^2 * (1 - ρ^2)
                # log |Σ^{-1}| = - log det Σ
                ldets_inv = pm.Deterministic(
                    "ldets_inv",
                    -2.0 * atools.safe_log(sig1) - 2.0 * atools.safe_log(sig2) - atools.safe_log(one_minus_r2),
                    dims="component",
                )
                
                # ----- Precision Σ^{-1} in closed form (Fisher) -----
                # Σ^{-1} = 1 / [ (1-ρ^2) s1^2 s2^2 ] * [[ s2^2, -ρ s1 s2 ], [ -ρ s1 s2, s1^2 ]]
                # variances
                var1 = sig1**2          # (K,)
                var2 = sig2**2          # (K,)
                cov12 = rho * sig1 * sig2
                
                den = one_minus_r2 * (var1) * (var2)
                F11 =  (var2)            / den
                F22 =  (var1)            / den
                F12 = -(cov12)    / den
                
                Fisher = pm.Deterministic( "Fisher", at.stack([
                    at.stack([F11, F12], axis=1),
                    at.stack([F12, F22], axis=1)
                ], axis=1), dims=("component","GMMdimension_1","GMMdimension_2"))  # shape: (K, 2, 2)
    

                
                # trace = var1 + var2                     # (K,)
                # det   = var1 * var2 * (1.0 - rho**2)    # (K,)
                
                # # discriminant of the characteristic polynomial
                # disc = at.sqrt(trace**2 - 4.0 * det)    # (K,)
                
                # # smallest eigenvalue λ_min
                # lam_min = 0.5 * (trace - disc)          # (K,)
                # s_min   = at.sqrt(lam_min)              # minor-axis std per component

                # L_eig = 1      # "too small" minor-axis std (tune)
                # alpha_eig = 0.05
                
                # lambda_eig = -L_eig * np.log(1.0 - alpha_eig)
                
                # _ = pm.Potential(
                #     "pc_small_eig",
                #     -lambda_eig * at.sum(1.0 / s_min)
                # )


                ################################################
    
                Lambda_ += [ alpha, beta, w, mu, Fisher, ldets_inv, logw ]

            Lambda_ += [N_DP_comp_max]


            
        ################################################
        # If including total normalization of the rate, add it here
        ################################################
        
        if not marginal_R0:
            R0 = pm.Uniform('R0', lower=priors['R0'][0], upper=priors['R0'][1])
        else:
            R0 = at.as_tensor_variable(1.)    
        lR0 = atools.safe_log(R0)


        # if zres=='low':
        #     print('Using z grid with 150 points')
        #     zgrid_ = atools.zGridGlobals_at_low
        # elif zres=='high':
        #     print('Using z grid with 1000 points')
        #     zgrid_ = atools.zGridGlobals_at_high

        
        zgrid_ = stop_grad(at.as_tensor_variable( atools.make_z_grid(total=zres, zmin_a=zmin_a, zmin_b=zmin_b, zmid_b=zmid_b, zmax_c=zmax_c, hi_boost=hi_boost) ))
        print("z grid for interpolation built. Resolution: %s"%zres)
        print("z min: %s , z max: %s"%(zmin_a, zmax_c))
        
        # Precompute cosmology pieces 
        # One grid build to interpolate later
        dc_grid = atools.dcfun_at(zgrid_, H0_, Om_, w0_, interp=pade)
        dL_grid = atools.dLfun_at(zgrid_, H0_, Om_, w0_, Xi0_, nXi0_, interp=pade, dc=dc_grid, param=param)
        log_ddL_dz_grid = atools.log_ddL_dz(zgrid_, H0_, Om_, w0_, Xi0_, nXi0_, dc=dc_grid, interp=pade, param=param)


        # Precompute mass function pieces 

        if interp_mass!=0:

            eps_m = at.as_tensor_variable(1e-5)
            
            m1_grid_ = ( m1_low_+eps_m + (300.0 - m1_low_ ) * tgrid_m1).astype(X)
            m2_grid_ = ( m2_low_+eps_m + (300.0 - m2_low_ ) * tgrid_m2).astype(X)
            
                
            if mass_model=='DPLDP':
                
                lp_m1_grid = atools.logpdfm1_DPLDP( m1_grid_, alpha1_, alpha2_, mb_, mu1_, sigma1_, mu2_, sigma2_, m1_low_, m_high_, delta_m1_, lambda0_, lambda1_, epsilon_,  smoothing=smoothing) 


                lp_m2_grid = atools.logpdfm2_PLP_reg( m2_grid_, beta_, delta_m2_, m2_low_, m_g=m_g_, w_g=w_g_, sig_g_low = sig_g_l_, sig_g_high = sig_g_h_, has_m2_break=has_m2_break, smoothing=smoothing ) 


                # CDF over m2
                cdf_m2 = atools.atcumtrapz(at.exp(lp_m2_grid), m2_grid_)
                
                # cdf_m2 has length m2_grid_.shape[0] - 1
                # grid for cdf_m2 is m2_grid_[1:]
                x0 = m2_grid_[1]
                x1 = m2_grid_[-1]
                nU = m2_grid_.shape[0] - 1  # == cdf_m2.shape[0]
                
                lC_of_m1 = atools.atinterp_uniform(
                    m1_grid_,
                    x0,
                    x1,
                    nU,
                    atools.safe_log(cdf_m2),
                )
                
                # Normalization for m1
                p1 = at.exp(lp_m1_grid)
                ln = atools.safe_log(atools.attrapzvec(p1, m1_grid_))
                
                # Pack for later use
                interp_vals_mass  = [lp_m1_grid, lp_m2_grid, lC_of_m1, ln]
                interp_grids_mass = [m1_grid_, m2_grid_]

            elif mass_model=='DPLDP-z':

                # ---------
                # 1) m2 grids (depend on m2 params, but NOT on z in your current model)
                # ---------
                lp_m2_grid = atools.logpdfm2_PLP_reg(
                    m2_grid_, beta_, delta_m2_, m2_low_,
                    m_g=m_g_, w_g=w_g_, sig_g_low=sig_g_l_, sig_g_high=sig_g_h_,
                    has_m2_break=has_m2_break, smoothing=smoothing
                )  # shape (N2,)
            
                # lC_grid evaluated on m1_grid (shape (N1,))
                cdf_m2 = atools.atcumtrapz(at.exp(lp_m2_grid), m2_grid_)

                x0 = m2_grid_[1]
                x1 = m2_grid_[-1]
                nU = m2_grid_.shape[0] - 1  # == cdf_m2.shape[0]
                
                lC_of_m1 = atools.atinterp_uniform(m1_grid_,x0,x1,nU,atools.safe_log(cdf_m2),)

                # ---------
                # 2) Bank lp_m1(z_k, m1_grid_) and ln(z_k)
                # ---------
                K  = z_bank.shape[0]
                N1 = m1_grid_.shape[0]
                
                M = at.broadcast_to(m1_grid_[None, :], (K, N1))
                Z = at.broadcast_to(z_bank[:, None],   (K, N1))
                
                lp_flat = atools.logpdfm1_DPLDP_z(
                    M.reshape((K * N1,)),
                    Z.reshape((K * N1,)),
                    alpha1_0, alpha2_0, mb_0,
                    mu1_0, sigma1_0, mu2_0, sigma2_0,
                    m1_low_, m_high_, delta_m1_,
                    lambda0_0, lambda1_0,
                    epsilon_,
                    *evo_params_,
                    smoothing=smoothing,
                    simplex_repair=simplex_repair
                )
                lp_m1_bank = at.clip( lp_flat, -1e30, 1e030 ).reshape((K, N1)) # (K,N1)

                ln_bank = atools.safe_log( atools.attrapzvec(at.exp(lp_m1_bank), m1_grid_, axis=1))
             
                # Pack for later use (include z_bank)
                interp_vals_mass  = [lp_m1_bank, lp_m2_grid, lC_of_m1, ln_bank, ]
                interp_grids_mass = [m1_grid_, m2_grid_, z_bank]
                
            elif mass_model in ('DPUC', 'DP'):

                if rate_model in ('MD', 'PL'):
                    lp_Mc_grid, lp_q_grid, lp_z_grid = atools.gaussian_logpdf_pair(log_Mc_grid, logit_q_grid, mu, sd)


                    interp_vals_mass  = [lp_Mc_grid, lp_q_grid]
                    interp_grids_mass = [log_Mc_grid, logit_q_grid]

                else:

           
                    lp_Mc_grid, lp_q_grid, lp_z_grid = atools.gaussian_logpdf_pair(log_Mc_grid, logit_q_grid, mu, sd, z=log_1pz_grid)


                    interp_vals_mass  = [lp_Mc_grid, lp_q_grid, lp_z_grid]
                    interp_grids_mass = [log_Mc_grid, logit_q_grid, log_1pz_grid]
                
                
            else:
                raise NotImplementedError()
        
        else:
            interp_vals_mass = None
            interp_grids_mass = None
            

        
        ## Precompute rate function pieces
        # To implement


        
        ## Precompute spin function pieces
        # To implement


        if not sample_from_pop:
            
            if not pop_only:
            ################################################
            # Individual event mass and distance
            ###############################################
    
                x = pm.Normal( 'x', mu=0, sigma=1, dims= ("event_index" , "GWdimension" ), initval = (np.random.randn(N, nd) * eps_init).astype(X) )    
    
                if 'gauss' not in sampling_GW:
                    
                    if 'gmm' in sampling_GW:
            
                        print('Sampling m1d, m2d, dL from GMM')
        
                        if sampling_GW=='gmm_cat':
                            ###################################
                            # categorical way
        
                            ig = pm.Categorical('idx', p=wts_l, dims= "event_index",  initval=at.argmax(wts_l, axis=1).astype(int) )
        
                        elif sampling_GW=='gmm':
                            ###################################
                            # continuous way
            
                            u_gmm = pm.Normal("u_gmm", 0.0, 1.0, dims= "event_index")
                            v_gmm = at.clip( atools.normal_cdf(u_gmm), 1e-9, 1.0 - 1e-9) 
        
                            cdf_w = at.cumsum(wts_l, axis=1)                                          
                            ig = pm.Deterministic('idx', (v_gmm[:, None] < cdf_w).argmax(axis=1), dims= "event_index" )             
    
                        
                        # Select means and Cholesky factors per batch
                        mu_selected = mus_l[ np.arange(N), ig, :]         # shape (N, D)
                        L_selected = cho_covs_l[ np.arange(N), ig, :, :]  # shape (N, D, D)
                         
                        # Batched matrix multiplication: (N, D, D) @ (N, D, 1) → (N, D, 1)
                        Lx = at.sum(L_selected * x[:, None, :], axis=2)  # → shape (N, D)
    
                
                    else:
                        print('Sampling m1d, m2d, dL from gumbel soft assignment, tau=0.5')
                        
                        #tau = pm.MutableData("tau_gmm", 0.5)  # (note: if grads feel weak, raise to ~0.3–0.7)
                        tau=0.5
                        logits = atools.safe_log(at.clip(wts_l, 1e-12, 1.0))               # (N, K)
                        g = pm.Gumbel("gumbel", mu=0.0, beta=1.0, shape=wts_l.shape)  # (N, K)
                        y_soft = pm.math.softmax((logits + g) / tau, axis=1)      # (N, K)
                        
                        # hard label for inspection (unchanged)
                        ig = pm.Deterministic("idx", at.argmax(y_soft, axis=1), dims="event_index")  # (N,)
                        
                        # --- Straight-Through gate (hard forward, soft gradient) ---
                        # get K from your tensors (N, K, D)
                        K = mus_l.shape[1]
                        topk = at.argmax((logits + g) / tau, axis=1)                                     # (N,)
                        one_hot = at.eq(at.arange(K)[None, :], topk[:, None]).astype(y_soft.dtype)       # (N, K)
                        s_soft_hard = stop_grad(one_hot - y_soft) + y_soft                         # (N, K)
    
                        # --- Soft selection, but with ST gating in forward ---
                        # mu_selected: (N, D)
                        mu_selected = at.sum(mus_l * s_soft_hard[:, :, None], axis=1)
                        
                        # L_selected: (N, D, D)
                        L_selected = at.sum(cho_covs_l * s_soft_hard[:, :, None, None], axis=1)
                        
                        # Lx: (N, D)  [ (N,D,D) * (N,1,D) → (N,D,D); sum over last axis → (N,D) ]
                        Lx = at.sum(L_selected * x[:, None, :], axis=2)
                    
                    
                    # Final transformed sample
                    samples = mu_selected + Lx                # shape (N, D)
        
                    
                    log_Mc_det = samples[:,0]/dil_factor
                    logit_q = samples[:,1]
                    logd = samples[:,2]
                    
        
                    if (spin_model == 'chieffchip') or (spin_model == 'chieffchip_uc') :
            
                        chieff = atools.inv_flogitat(samples[:,3])
                        chip = atools.inv_logitat(samples[:,4])
            
                    elif (spin_model == 'default') or (spin_model == 'default_gauss'):
                        # we have chi1, chi2, cost1, cost2
                        if save_thetas:
                            chi1 = pm.Deterministic('chi1', atools.inv_logitat(samples[:,3]))
                            chi2 = pm.Deterministic('chi2', atools.inv_logitat(samples[:,4]))
                
                            cost1 = pm.Deterministic('cost1', atools.inv_flogitat(samples[:,5]))
                            cost2 = pm.Deterministic('cost2', atools.inv_flogitat(samples[:,6]))
                        else:
                            chi1 = atools.inv_logitat(samples[:,3])
                            chi2 = atools.inv_logitat(samples[:,4])
                            cost1 =atools.inv_flogitat(samples[:,5])
                            cost2 =atools.inv_flogitat(samples[:,6])
                            
                    else:
                        print("No spins computed")
                
    
                
                elif sampling_GW=='gauss' : # to be tested with spins
                    
                    print('Sampling log(Mc), logit(q), log(dL) from Gaussian approximant')


                    
                    # sample = mu + L @ x   (batched)
                    samples = mus_s + at.matmul(cho_s, x[..., None])[..., 0]      # (N, d)
                
                    # logp = log p(x) - log|L|
                    # d = x.shape[1]
                    log_px = -0.5 * at.sum(x**2, axis=1) - 0.5 * x.shape[1] * atools.safe_log(2.0 * np.pi)    # (N,)

                    log_det_L = at.sum(atools.safe_log(at.diagonal(cho_s, axis1=1, axis2=2)), axis=1)  # (N,)
                    pilik = log_px - log_det_L                                               # (N,)

                    # unpack coordinates:
                    log_Mc_det = samples[:, 0]
                    logit_q    = samples[:, 1]
                    logd       = samples[:, 2]
                    
    
                    if spin_model == 'none' :
                        
                        X = at.stack([log_Mc_det, logit_q, logd ], axis=1)
                        d_int  = at.as_tensor_variable(3, dtype=int)
    
    
                    elif spin_model == 'default' or spin_model == 'default_gauss':
    
                        chi1 = atools.inv_logitat(samples[:,3])
                        chi2 = atools.inv_logitat(samples[:,4])
            
                        cost1 = atools.inv_flogitat(samples[:,5])
                        cost2 = atools.inv_flogitat(samples[:,6])
    
                        X = at.stack([log_Mc_det, logit_q, logd,  samples[:,3],  samples[:,4],  samples[:,5],  samples[:,6]], axis=1)
                        d_int  = at.as_tensor_variable(7, dtype=int)
    
    
                
    
                    # X as (N, d)
                    #X = vals.T                                   # (N, d)
                    #print("X shape is %s"%(X[:, None, :].shape.eval()))
                    #print("mus_l shape is %s"%(mus_l.shape.eval()))
                    
                    # Broadcast X against component-wise parameters
                    # diff: (N, ngmm, d)
                    diff = X[:, None, :] - mus_l[:, :, :d_int]                  # (N, 1, d) - (N, ngmm, d)
       
                    
                    # Quadratic form using precision F = Σ^{-1}
                    # tmp = F @ diff[..., None]  -> (N, ngmm, d, 1) -> squeeze to (N, ngmm, d)

                    
                    tmp = at.matmul(icovs_l[:, :, :d_int, :d_int], diff[..., None])[..., 0]   # (N, ngmm, d)


                    
                    # r^T F r for each (obs, comp)
                    quad = at.sum(diff * tmp, axis=-1)            # (N, ngmm)
    
                    
                    # Component logpdfs (Multivariate Normal)
                    log_norm = -0.5 * d_int * atools.safe_log(2.0 * np.pi)     # scalar
                    logp_components = (
                        -0.5 * quad
                        + log_norm
                        - 0.5 * log_dets_l
                        + log_wts_l
                    )                                             # (N, ngmm)


                    # Mixture log-likelihood per observation: logsumexp over components
                    gwl = at.logsumexp(logp_components, axis=1)   # (N,)

            
                
                else:
                    raise NotImplementedError()
    
    
                Mc = at.exp(log_Mc_det)            
                q = atools.inv_logitat(logit_q)
                m1det, m2det = atools.m1m2_from_Mcq_at(Mc, q)
                d = at.exp(logd)
    
                # Compute source-frame quantities. One redsfhit, mass1, mass2 for each event
                zs = atools.atinterp(d, dL_grid, zgrid_)
                one_plus_zs = 1+zs
                m1src = m1det/one_plus_zs 
                m2src = m2det/one_plus_zs  
    
                log_ddL_dz = atools.atinterp( zs, zgrid_, log_ddL_dz_grid) 
                dc = atools.atinterp( zs, zgrid_, dc_grid) 
                
                if save_thetas:
                    d = pm.Deterministic('dL', d , dims="event_index")
                    zs = pm.Deterministic('z', zs, dims= "event_index" ) 
                    m1src = pm.Deterministic('m1src', m1src, dims="event_index")
                    m2src = pm.Deterministic('m2src', m2src , dims="event_index")      
             
                    
            else:
                # we are sampling the usual marginalise likelihood, with "only" pop parameters
                print('We are running inference only on population parameters.')
    
    
                # Compute source-frame quantities. One redsfhit, mass1, mass2 for each event
                # AND for each sample! 
                
                d_stacked  = at.flatten(d)
                zs_stacked = atools.atinterp(d_stacked, dL_grid, zgrid_)
    
                
                zs = at.reshape( zs_stacked, (N, Nsamples) )
                m1src = m1det/(1+zs)
                m2src = m2det/(1+zs)
                
                logd = atools.safe_log(d)
            
            
            ################################################
            # Population prior
            ################################################
    
    
            if wrap_logp:
                log_p_pop_fun = log_p_pop_at_wrap
                print("Using wrapped p_pop")
            else:
                log_p_pop_fun = log_p_pop_at
                print("Using regular p_pop")
    
            
            if spin_model == 'chieffchip' or spin_model == 'chieffchip_uc' :
    
                spins = [ chieff, chip  ]
    
            elif (spin_model == 'default') or (spin_model == 'default_gauss'):
    
                spins = [chi1, chi2, cost1, cost2]
    
            elif spin_model == 'none':
                
                spins = []
    
            if mass_model not in ('DP', 'DPUC', 'DPLDP-z'):
                Lambda_ = at.stack(Lambda_, axis=0)
    
    
            # # Compute comoving distance - if gravity is modified, this is NOT d_L / (1+z) ! 
            # Xi_ = atools.Xifun_at(zs, Xi0_, nXi0_)
            # dc = d/(1+zs)/Xi_, 
    
            
            # Population prior of all events, without the term T_obs*R0
            if mass_model in ('DP', 'DPUC'):
    
                # dirichelet processs will be for log(Mc_src), logit(q) ...
                logMc_src =  log_Mc_det - at.log1p(zs)
                
                log_p_pop = log_p_pop_fun( logMc_src, logit_q, zs, d, spins, Lambda_, rate_model, mass_model, spin_model,  dc=dc,  log_ddL_dz_pre=log_ddL_dz, z_grid = zgrid_ )
                
                
                # ... so remove a jacobian : p( m1, m2 ) = p( log(Mc), logit(q) ) * |J|
                # if using interpolation, the jacobian is already included in the grid.
                print("Likelihood: removing jacobian m1, m2 --> log(Mc), logit(q) ")
                
                eps = at.as_tensor_variable(1e-12, dtype=m2src.dtype)
                log_p_pop -=  atools.safe_log(m2src) + atools.safe_log(at.maximum(m1src - m2src, eps))
    
                if rate_model in ('DPUC','DPUC-vol' ):
                    # also remove jacobian for log(1+z)
                    log_p_pop -= at.log1p(zs) 
                    
                
            else:    
            
                log_p_pop = log_p_pop_fun( m1src, 
                                           m2src, 
                                           zs, 
                                           d, 
                                           spins, 
                                           Lambda_, 
                                           rate_model, mass_model, spin_model, 
                                           smoothing=smoothing,
                                           simplex_repair=simplex_repair,
                                           has_m2_break=has_m2_break, 
                                           dc=dc, 
                                           log_ddL_dz_pre=log_ddL_dz,
                                           interp_vals_mass = interp_vals_mass,
                                           interp_grids_mass = interp_grids_mass,
                                           is_observed = is_observed,
                                           z_grid = zgrid_
                                         )

    
            
            
            
        
        
        else:
            # sampling from GMM and then computing GW likelihood in det space
            print("\nWill sample from population then compute GW likelihood.")


            # k = pm.Categorical( "k", p=w, dims="event_index" )
            # logMc  = pm.Normal("logMc",  mu=mu[0, k], sigma=sd[0, k], dims="event_index")
            # logit_q = pm.Normal("logitq", mu=mu[1, k], sigma=sd[1, k], dims="event_index")
            # y      = pm.Normal("y",      mu=mu[2, k], sigma=sd[2, k], dims="event_index")   # y=log(1+z)
            # _ = pm.Potential("vol_weight", atools.log_dV_dz_at(z, H0_, Om_, w0_, dc=dc).sum()) # or add non-summed version to total logp befor summing


                        
            logMc = pm.Uniform( "logMc_src", lowmu1, upmu1,  dims="event_index", initval=log_Mc_src_init)
            logit_q = pm.Uniform( "logit_q", lowmu2, upmu2,  dims="event_index", initval=logit_q_init)
            y = pm.Uniform("log1pz", lowmu3, upmu3, dims="event_index", initval=log_onepz_init )

            q  = atools.inv_logitat(logit_q)
            z = at.exp(y)-1
            Mc = at.exp(logMc)
            m1s, m2s = atools.m1m2_from_Mcq_at(Mc, q)

            m1det = m1s*(1+z)
            m2det = m2s*(1+z)

            dc = atools.atinterp( z, zgrid_, dc_grid) 

            
            # Compute p_pop
            logp1, logp2, logp3 = atools.gaussian_logpdf_pair( m1s, m2s, mu, sd, z=y )        
            logp_components = logp1 + logp2 + logp3                     # (K,N)
            lpmass = atools.safe_logsumexp(logp_components + logw[:, None], axis=0)


            # compute GW likelihood in det. frame

            log_Mc_det = logMc+y
            d = atools.dLfun_at(z, H0_, Om_, w0_, Xi0_, nXi0_, param=param)
            logd = at.log( d )


            X = at.stack([log_Mc_det, logit_q, logd ], axis=1)
            d_int  = at.as_tensor_variable(3, dtype=int)


            diff = X[:, None, :] - mus_l[:, :, :d_int]                  # (N, 1, d) - (N, ngmm, d)
 
            tmp = at.matmul(icovs_l[:, :, :d_int, :d_int], diff[..., None])[..., 0]   # (N, ngmm, d)
            
            quad = at.sum(diff * tmp, axis=-1)            # (N, ngmm)
            
            log_norm = -0.5 * d_int * atools.safe_log(2.0 * np.pi)     # scalar
            logp_components = (
                -0.5 * quad
                + log_norm
                - 0.5 * log_dets_l
                + log_wts_l
            )                                             # (N, ngmm)
            
            gwl = at.logsumexp(logp_components, axis=1)   # (N,)

            # jacobian
            log_jac_q = -at.log(q) - at.log1p(-q)

            # all
            log_p_pop = lpmass + gwl - log_Mc_det - logd - log_jac_q
        



        if is_observed:
    
            print("Fitting for observed population. Removing factor 1/Pdet")

            Theta = at.ones(d.shape)
          
            log_P_det = atools.safe_log( atools.Pdet( osnr_interp_at, m1det, m2det, d, Theta, at.as_tensor_variable(8.) )
                                       )
            log_p_pop -= log_P_det

            
                
        if dLprior=='dLsq':
            # Remove \pi(d)~dL^2 prior on distance 
            log_p_pop -= 2*logd
            print('Removing dL^2 prior')
       
        elif dLprior == 'dVdz':
            print('Removing prior proportional to 1/(1+z)*dV/dz with H0=67.90, Om=0.3065')
            
            #dc_grid_Planck15 = atools.dcfun_at(zgrid_, 67.90, 0.3065, -1., interp=pade)
            #dVdz_grid_Planck15 = atools.log_dV_dz_at(zgrid_, 67.90, 0.3065, -1., dc=dc_grid_Planck15 )-at.log1p(zgrid_)
            #lpi = atools.atinterp( zs, zgrid_, dVdz_grid_Planck15 )
           
            dc_Planck15 = atools.dcfun_at(zs, 67.90, 0.3065, -1., interp=pade)
            lpi_ = atools.log_dV_dz_at(zs, 67.90, 0.3065, -1., dc=dc_Planck15 )-at.log1p(zs)

            #atools.log_dV_dz_at(zs, 67.90, 0.3065, -1., dc=None )-at.log1p(zs)

            # The following is a hack.
            # When using GWTC data, O1-O2 do not have posteriors with dVdz prior, only dL^2
            # So I remove the dL^2 prior by hand on those
            print(
            "⚠️ Warning: I remove the dL^2 prior by hand on the first 10 elements. This is usually done with LVK data for BBHs as O1-O2 do not have posteriors with dVdz prior. Do this with knowledge of the dataset. "
        )
            if not pop_only:
                # 1D case: shape (N,)
                #lpi = at.concatenate([2 * logd[:10], lpi_[10:]], axis=0)
                lpi = at.set_subtensor(lpi_[:10], 2 * logd[:10])
            
            else:
                # 2D case: shape (N, Nsamples)
                #lpi = at.concatenate([2 * logd[:10, :], lpi_[10:, :]], axis=0)
                # start from lpi_ and replace the first 10 elements
                lpi = at.set_subtensor(lpi_[:10, :], 2 * logd[:10, :])
                  
            
            log_p_pop -= lpi


        if not pop_only:
            if sampling_GW=='gauss' and not sample_from_pop:
                # Add gw likelihood and correct for sampling prior pdf
                log_p_pop -= pilik
                log_p_pop += gwl

            
            # just sum log likelihoods
            likelihood_val = at.sum( log_p_pop ) #pm.Deterministic("lik", at.sum( log_p_pop ) ) 

        
        else:
            # marginalise over single events parameters first
            # shape of p_pop is (hopefully) n_evs x n_samples
            # so average over second dimension
            
            # Compute only where there are samples
            log_p_pop_to_marg = log_p_pop[:, :allNsamples[0]]
            
            log_p_pop_marg = at.logsumexp( log_p_pop_to_marg, axis=1 ) - atools.safe_log(allNsamples)
            

            # then sum log likelihoods
            likelihood_val = at.sum( log_p_pop_marg )  

            # Check number of effective samples for computing MC integral 
            logs2 = at.logsumexp(2*log_p_pop_masked, axis=1) -2*atools.safe_log(allNsamples)
            
            Neff_lik =  pm.Deterministic('Neff_l', at.exp( 2.0*log_p_pop_marg - logs2) ) # this has len = n. of observations
            
            if Neff_min_lik>0:
                
                _ = pm.Potential("Neff_l_bound", at.sum( at.where( Neff_lik<Neff_min_lik*N, -np.inf, 0. ) ) )
              
            else:
                print("No bound on effective number of samples for individual event MC integrals")

        
        # add R0*Tobs if needed. 
        if not marginal_R0:
            print("Will not marginalise over R0.")
            # each term p_pop is multiplied by
            # R0*T_obs . So we get a factor (R0*T_obs)**N_i for every
            # observing run. R0 is the same for every run so I just have
            # (R0)**{\sum N_i} . For T_obs I have T_{obs,1}**N_1 * T_{obs,2}**N_2 * ...
            poiss_term = at.sum(Nevs*atools.safe_log(allTobs))+N*lR0
            likelihood_val += poiss_term
        else:
            print("Will marginalise over R0 with flat-in-log prior.")

        
        
        _ = pm.Potential("likelihood", likelihood_val ) 



        ################################################
        # Selection effect
        ################################################
        
        if sel_method=='skip':
            print('No selection bias!')
        else:
            # add sel effects    
            if ndata_np==1:
                # we passed a single injection set corresponding to multiple observing runs,
                # with injections already containing the correct weights
                print("Using selection effects from a single injection campaign")


                if chunk_inj!=-1:
                    print('Using chunked version of sel. bias for memory efficiency.')
                    if inj_loop=='loop':
                        sel_bias_fun = sel_bias_with_uncertainty_at_loop
                        print("Using version with python loop")
                        print('Chunk size is %s'%chunk_inj)
                    elif inj_loop=='vec':
                        sel_bias_fun = sel_bias_with_uncertainty_at_0_batched 
                        #sel_bias_with_uncertainty_at_scan
                        print("Using version with pytensor vectorization in batches")
                        print('Chunk size is %s'%chunk_inj)
                        #print("use_float32 is %s"%use_float32)
                    elif inj_loop=='scan':
                        sel_bias_fun = sel_bias_with_uncertainty_at_0_batched_scan
                        print("Using version with pytensor scan in batches")
                        print('Chunk size is %s'%chunk_inj)
                    else:
                        raise ValueError("inj_loop can be scan, vec, or loop, got %s"%inj_loop)

                    zinj = None
                    dcinj = None 
                    log_ddL_dz_inj = None

                    dL_grid_inj = dL_grid              # 1-D, strictly increasing in dL
                    z_grid_inj = zgrid_               # 1-D, z(dL_grid)
                    dc_grid_inj = dc_grid              # 1-D, dc(z_grid)
                    log_ddL_dz_grid_inj = log_ddL_dz_grid      # 1-D, log(ddL/dz) sampled at z_grid

                
                else:
                    if chunk_reduce:
                        #print("Using chunked version for reduction of logsumexp")
                        #sel_bias_fun = sel_bias_with_uncertainty_at_scan_slow
                        raise ValueError("Not available")
                    else: 
                        print('Computing sel bias in one chunk')
                        sel_bias_fun = sel_bias_with_uncertainty_at_0

                        if interp_inj:
                            # Interpolate on injections from pre-computed grid
                            print("Injections will use interpolation from pre-computed grid to compute d_c, log_ddL_dz")
                            zinj = atools.atinterp(dLinj[0], dL_grid, zgrid_)
                            dcinj = atools.atinterp( zinj, zgrid_, dc_grid) 
                            log_ddL_dz_inj = atools.atinterp( zinj, zgrid_, log_ddL_dz_grid)
                        else:
                            print("Injections will call usual cosmo functions to compute d_c, log_ddL_dz.")
                            zinj = atools.atinterp(dLinj[0], dL_grid, zgrid_) #None
                            dcinj = None 
                            log_ddL_dz_inj = None


                        dL_grid_inj = None              # 1-D, strictly increasing in dL
                        z_grid_inj = None               # 1-D, z(dL_grid)
                        dc_grid_inj = None              # 1-D, dc(z_grid)
                        log_ddL_dz_grid_inj = None      # 1-D, log(ddL/dz) sampled at z_grid



                          
                
                log_mu_, Neff_, var_ll_u_ = sel_bias_fun( m1inj[0], m2inj[0], dLinj[0], spinsInj, lpdinj[0], 
                                                          Lambda_, 
                                                          Ndraw, 
                                                          rate_model, mass_model, spin_model_name, 
                                                          smoothing, 
                                                          simplex_repair,
                                                          has_m2_break, 
                                                          interp=pade, 
                                                          log_p_incl = lp_incl_inj[0],
                                                         dL_grid=dL_grid_inj,             
                                                        z_grid=z_grid_inj, 
                                                        dc_grid=dc_grid_inj, 
                                                        log_ddL_dz_grid=log_ddL_dz_grid_inj, 
                                                          chunk_size = chunk_inj, 
                                                          use_float32=use_float32_bias, 
                                                          N_inj_py=ninj_np, 
                                                          scan_updates=use_updates, 
                                                          wrap_logp = wrap_logp,
                                                          log_ddL_dz_inj = log_ddL_dz_inj,
                                                            zinj = zinj,
                                                            dcinj = dcinj,
                                                          param=param,
                                                          interp_vals_mass = interp_vals_mass,
                                                           interp_grids_mass = interp_grids_mass,
                                                        
                                                        )

                
                # zinj_tmp_ = atools.atinterp(dLinj[0], dL_grid, zgrid_)

                
                # log_mu_1, Neff_1, var_ll_u_1 = sel_bias_with_uncertainty_at_0( m1inj[0], m2inj[0], dLinj[0], spinsInj, lpdinj[0], 
                #                                           Lambda_, 
                #                                           Ndraw, 
                #                                           rate_model, mass_model, spin_model_name, 
                #                                           smoothing, 
                #                                           has_m2_break, 
                #                                           interp=pade, 
                #                                          dL_grid=None,              # 1-D, strictly increasing in dL
                #                                         z_grid=None,               # 1-D, z(dL_grid)
                #                                         dc_grid=None,              # 1-D, dc(z_grid)
                #                                         log_ddL_dz_grid=None,      # 1-D, log(ddL/dz) sampled at z_grid
                #                                           chunk_size = chunk_inj, 
                #                                           use_float32=False, 
                #                                           N_inj_py=ninj_np, 
                #                                           scan_updates=use_updates, 
                #                                           wrap_logp = False,
                #                                         log_ddL_dz_inj = atools.atinterp( zinj_tmp_, zgrid_, log_ddL_dz_grid),
                #                                          zinj = zinj_tmp_,
                #                                          dcinj = atools.atinterp( zinj_tmp_, zgrid_, dc_grid) ,
                #                                         )

                # print("Difference in log_mu_1 :")
                # print((log_mu_1 - log_mu_).eval().max())

                # print("Difference in var_ll_u_1 :")
                # print((var_ll_u_1 - var_ll_u_).eval().max())
                
                if not marginal_R0:
                    # This is really the number of expected events 
                    sel_effect = -R0*Ttot*at.exp(log_mu_)
                else:
                    sel_effect = -N*log_mu_
    
            else:
                # we passed multiple injections set corresponding to multiple observing runs
                # they need to be properly combined
                # This is useful only if using older LVK injection sets,
                # Deprecated after GWTC-3 

                
                print("Combining selection effects from different injections campaigns")

                spin_model_name = spin_model
                if use_sel_spin:

                    if spin_model == 'chieffchip' or spin_model == 'chieffchip_uc':
                        # shapes: chi1Inj, chi2Inj -> (ndata, ninj)
                        # result: spinsInj -> (ndata, 2, ninj)
                        spinsInj = at.stack([chi1Inj, chi2Inj], axis=1)
                    
                    elif (spin_model == 'default') or (spin_model == 'default_gauss'):
                        # shapes: chi1Inj, chi2Inj, cost1Inj, cost2Inj -> (ndata, ninj)
                        # result: spinsInj -> (ndata, 4, ninj)
                        spinsInj = at.stack([chi1Inj, chi2Inj, cost1Inj, cost2Inj], axis=1)

                else:
                    spinsInj = at.ones( (ndata, 2, ninj) )
                    print("Spin distribution will not be used in the sel effect")
                    spin_model_name = 'none'
                    
                    
                
                if not fix_inj_len:
                    print("Loop over injections sets, dynamical slicing")
                    # This should improve efficiency. But it can give problems with pytensor.scan (?)

                    res_i, _ = pytensor.scan( lambda idata, m1inj_, m2inj_, dLinj_, spinsInj_, lpdinj_, L,  Ndraw_, Ndet_ : sel_bias_with_uncertainty_at( m1inj_[idata, : Ndet_[idata]], m2inj_[idata, : Ndet_[idata]], dLinj_[idata, :Ndet_[idata]],  spinsInj_[idata, :, :Ndet_[idata]], lpdinj_[idata, :Ndet_[idata]], L, Ndraw_[idata], rate_model, mass_model, spin_model_name, smoothing, has_m2_break, interp=pade, dL_grid=dL_grid, z_grid=zgrid_ ), 
                                          sequences = [ np.arange( ndata) ], 
                                          non_sequences = [m1inj, m2inj, dLinj, spinsInj, lpdinj, Lambda_,  Ndraw, Ndet],
                                            profile=True
                                            )
                    log_mu_vec = res_i[0]
                    Neff_ = at.sum(res_i[1])

                    
                else:
                    print("Loop over injections sets, no slicing")
                    # makes it jax-compatible (jax does not support dynamical slicing at the moment)
                    # Not true anymore after pymc v5.10 ? Check
                    res_i, _ = pytensor.scan( lambda idata, m1inj_, m2inj_, dLinj_, spinsInj_, lpdinj_, L,  Ndraw_ : sel_bias_with_uncertainty_at( m1inj_[idata ], m2inj_[idata ], dLinj_[idata], spinsInj_[idata],  lpdinj_[idata], L, Ndraw_[idata], rate_model, mass_model, spin_model, smoothing, has_m2_break, interp=pade, dL_grid=dL_grid, z_grid=zgrid_ ), 
                                      sequences = [ np.arange( ndata) ], 
                                      non_sequences = [m1inj, m2inj, dLinj, spinsInj, lpdinj,  Lambda_,  Ndraw] )

            
                    log_mu_ = res_i[0]
                    Neff_ = at.sum(res_i[1])
    

                
    
                if not marginal_R0:
                    # Sum number of expected events in the two observing runs
                    # p_pop does not contain R_0*Tobs . Add it here
                    sel_effect = -at.sum(at.exp(log_mu_+lR0+atools.safe_log(Tobs)))
                else:
                    if sel_method=='Tobs':
                        sel_effect = -N*at.logsumexp( atools.safe_log(Tobs/Ttot)+log_mu_ )
                        print('Using sel function with weighted obs time average. Obs times: %s'%str(Tobs))
                    elif sel_method=='Nevs':
                        # This is technically wrong, but I leave it here
                        # to check how large the error is when using the wrong expression
                        print('Using sel function with number of events')
                        sel_effect = -at.sum(Nevs*log_mu_)

            
            ################################################
            # Sel effect computed. Now exclude high-variance regions in the integral

            
            Neff = pm.Deterministic('Neff', Neff_ )

            if marginal_R0:
                log_lik_var = pm.Deterministic('log_lik_var', at.exp(var_ll_u_+2*atools.safe_log(N)) )
            else:
                log_lik_var = pm.Deterministic('log_lik_var', at.exp(  var_ll_u_+2*atools.safe_log( R0*Ttot ) + 2*log_mu_ ) )
            
     

            if ((Neff_min==0) and (log_lik_var_min==0)):
                print("No condition on number of effective points in MC integral for sel. effect")
                selection_bias =  sel_effect #pm.Deterministic("sel_bias", sel_effect )
            else:
                if log_lik_var_min==0:

                    # Thresholding on N_eff
                    print("MC integral for sel. effect thresholded on N_eff")
                    
                    if sel_smoothing=='sigmoid':
                        # smooth with sigmoid between Neff_min and Neff_min+1 x Nobs
                        # over a scale = Neff_min
                        # i.e. at Neff_min * Nobs the likelihood becomes smoothly -inf
                        selection_bias = atools.log_sigmoid(Neff, Neff_min*(N+1),  Neff_min)+sel_effect  #pm.Deterministic("sel_bias", atools.log_sigmoid(Neff, Neff_min*(N+1),  Neff_min)+sel_effect )
                    elif sel_smoothing=='poly':
                        # Polynomial smoothing
                        selection_bias =  atools.log_f_smooth_poly(Neff, N/2,  Neff_min*N-N/4)+sel_effect #pm.Deterministic("sel_bias", atools.log_f_smooth_poly(Neff, N/2,  Neff_min*N-N/4)+sel_effect ) 
                    else:
                        # Hard cut
                        
                        selection_bias = sel_effect #pm.Deterministic("sel_bias", sel_effect)                   
                        #ind_sw_sel = pm.Deterministic('ind_sel', 1. * (Neff<Neff_min*N ) )
                        #ind_sel = pm.Bernoulli('bound_Neff', ind_sw_sel, observed=np.zeros(1)  )
                        _ = pm.Potential("bound_Neff", at.switch(Neff >= Neff_min * N, 0.0, -np.inf))

                
                elif Neff_min==0:

                    # Thresholding on likelihood variance
                    print("MC integral for sel. effect thresholded on log lik. variance")
                    
                    if sel_smoothing=='sigmoid':
                        # smooth with sigmoid 
                        print("Tapering sel effect with sigmoid smoothing")
                        
                        selection_bias = sel_effect + atools.logdiffexp( atools.safe_log(1), atools.log_sigmoid(log_lik_var, log_lik_var_min*(1+0.002), 0.001 )) 

                    
                    elif sel_smoothing=='poly':
                        print("Tapering sel effect with polynomial smoothing")

                        
                        #selection_bias = sel_effect + atools.logdiffexp( atools.safe_log(1), atools.log_f_smooth_poly(log_lik_var, 0.01,  log_lik_var_min*(1-0.005) ))  

                        selection_bias = sel_effect
                        _ = pm.Potential("bound_log_lik_var", atools.logS_PLP(log_lik_var_min - log_lik_var, deltam=0.01, ml=-0.01))


                        
                    elif sel_smoothing=='softplus':
                        print("Tapering sel effect with softplus")
                        # Slack (how sharp the corner is) and weight (penalty strength)
                        nu = at.as_tensor_variable(0.01)     # smaller = sharper transition
                        lam = at.as_tensor_variable(1.)     # larger = stronger penalty
                        
                        excess  = (log_lik_var - log_lik_var_min) / nu
                        penalty = lam * at.softplus(excess)          # ≥ 0, ~0 if below threshold

                        selection_bias = sel_effect 
                        
                        # If log_lik_var is a vector, sum to get a scalar penalty:
                        pm.Potential("bound_log_lik_var", -at.sum(penalty))
                    else:
                        print("Tapering sel effect with hard cut")

                        selection_bias = sel_effect #pm.Deterministic("sel_bias", sel_effect)
                        # ind_sw_sel = pm.Deterministic('ind_sel', 1. * (log_lik_var>log_lik_var_min ) )
                        # ind_sel = pm.Bernoulli('bound_log_lik_var', ind_sw_sel, observed=np.zeros(1)  )
                        _ = pm.Potential("bound_log_lik_var", at.switch(log_lik_var <= log_lik_var_min, 0.0, -np.inf))

            
            _ = pm.Potential('selection_bias', selection_bias)

            if marginal_R0:
                if include_sel_uncertainty:
                    
                    
                    # from Farr 2019
                    # print("Including selection function uncertainty as in Farr 2019")
                    #sel_uncertainty = (3*N+N**2)/(2*Neff)

                    # from heinzel-Vitale 2025
                    print("Including selection function uncertainty as in Heinzel-Vitale 2025")
                    sel_uncertainty = - N*(N+1)/(2) * var_ll_u_
                    
                    _ = pm.Potential('selection_uncertainty', sel_uncertainty)
            

    return model

