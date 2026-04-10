#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cat.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import pytensor_tools as atools
import pytensor.tensor as at
import pytensor
import pymc as pm
import numpy as np
import os
from pytensor.gradient import grad, DisconnectedInputError
from pytensor.gradient import disconnected_grad
from pymc.distributions.dist_math import check_parameters
from pymc.distributions import transforms as tr  # for positivity
import pytensor_utils as putils
from pytensor.gradient import disconnected_grad as stop_grad

PLPeakO3params = {'H0': 67.66, 'Om':0.31, 'w0':-1, 'Xi0': 1, 'nXi0':0}

from tqdm import tqdm
import copy

eps   = 1e-30
tinyL = 1e-300
NEG_BIG = -np.inf

#####################################################
#####################################################

def log_p_pop_at(m1s, m2s, z, dL, spins, Lambda, rate_model, mass_model, spin_model, 
                 is_GP_dL, 
                 smoothing='LVK', 
                 has_m2_break=False,
                 log_ddL_dz=None,
                 dc=None, 
                 #ddr_dz=None, 
                 is_inj=False, 
                 #monotonicity=False, 
                 invert_dL_GP=True
                ):


    ###################################
    # get parameters and compute log p_pop
    ####################################

    Lambda_c = Lambda[:3] 
    H0, Om, w0 = Lambda_c 

    if is_GP_dL:
        
        iastro = 4
        gp = Lambda[3] 
    
        # older version. now log_ddL_dz is pre-computed
        
        # d_EM = dL^GW/(d_L^GW/d_EM) = dL^GW/(distance_ratio) 
        # d_c = d_EM/(1+z)
        #dL_em = dL/dr_val
        #dc = dL_em/(1+z) 
        # jacobian
        #dL_em = dc*(1+z)
        #ddLem_dz = at.exp( atools.log_ddL_dz( z, H0, Om, w0, 1., 0., dc=dc ) )
        #log_ddL_dz = at.log( at.abs( dL_em*ddr_dz + dr_val*ddLem_dz ) )


    else:

        Xi0, n = Lambda[3:5] 
        iastro = 5

        #dc =  atools.dcfun_at(z, H0, Om, w0)
        # jacobian
        #log_ddL_dz = atools.log_ddL_dz(z, H0, Om, w0, Xi0, n, dc=dc)

    # if dc is None:
    #     Xi = atools.Xifun_at(z, Xi0, n)
    #     dc = dL/(1+z)/Xi #atools.dcfun_at(z, H0, Om, w0, interp=False)

    ##################################
    # redshift 

    if rate_model=='MD':
        
        gamma, kappa, zp = Lambda[iastro:iastro+3]

        if (invert_dL_GP or (not is_GP_dL) or is_inj ):
            
            # This term contains the comoving distance
            # If there is MG, d_c is not d_L/(1+z)!
            lpz = atools.log_p_z_MD_unnorm(z, gamma, kappa, zp, Lambda_c , dc=dc )
        istart = iastro+3
        
    elif rate_model=='PL':
        
        gamma = Lambda[iastro]
        if (invert_dL_GP or (not is_GP_dL) or is_inj ):
            lpz = atools.log_p_z_PL_unnorm(z, gamma, H0, Om, w0, dc=dc)
        
        istart = iastro+1


    # ##################################
    # spin
    
    if spin_model=='chieffchip':
        
        muE, sigE, muP, sigP, rho = Lambda[istart:istart+5]
        chieff, chip = spins[0], spins[1]

        lpspin = atools.logpdf_multivariate_trunc_2D(  chieff, chip, muE, muP, sigE, sigP, rho,
                                                     at.as_tensor_variable(-1.), at.as_tensor_variable(1.), 
                                                     at.as_tensor_variable(0.), at.as_tensor_variable(1.) 
                                                    )
        istart_spin = istart+5

    elif spin_model=='chieffchip_uc':
        
        muE, sigE, muP, sigP = Lambda[istart:istart+4]
        chieff, chip = spins[0], spins[1]

        lpchie = atools.truncGausslowerupper_at_lpdf(chieff, muE, sigE, xmin=at.as_tensor_variable(-1), xmax=at.as_tensor_variable(1))
        lpchip = atools.truncGausslowerupper_at_lpdf(chip, muP, sigP, xmin=at.as_tensor_variable(0), xmax=at.as_tensor_variable(1))

        lpspin = lpchie+lpchip
        istart_spin = istart+4

    elif spin_model=='default':

        alphaChi, betaChi, zeta, sigmat = Lambda[istart:istart+4]
        lpspin = atools.logpdf_default_spin(spins, [alphaChi, betaChi, zeta, sigmat])
        istart_spin = istart+4
    
    elif spin_model=='default_gauss':
        muChi, sigmaChi, zeta, sigmat = Lambda[istart:istart+4]
        lpspin = atools.logpdf_default_spin_gauss(spins, [muChi, sigmaChi, zeta, sigmat])
        istart_spin = istart+4
   
    else:
        lpspin = at.zeros( z.shape )
        istart_spin = istart

    
    ###################################
    # mass

    ### BBH
    if mass_model=='PLPreg':
        
        lp, al, bb, dm, ml, mh, muM, sM = Lambda[-8:]
        lpmass = atools.logpdf_PLP_reg([m1s, m2s], [lp, al, bb, dm, ml, mh, muM, sM], smoothing=smoothing)

    elif mass_model=='DPLDP':
        
        #lambdaBBHmass = Lambda[-20:]
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
        x21 = Lambda[istart_spin + 20]

        lambdaBBHmass = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21]
        
        lpmass = atools.logpdf_DPLDP([m1s, m2s], lambdaBBHmass, force_m2_less_than_m1=False, has_m2_break=has_m2_break, smoothing=smoothing )
        
        
    ### BNS
    elif mass_model=='BNSgauss':
        muM, sM = Lambda[-2:]
        lpmass = atools.logpdf_gauss([m1s, m2s], [muM, sM] )
        
    elif mass_model=='BNSgaussCond':
        muM, sM = Lambda[-2:]
        lpmass = atools.logpdf_gauss_cond([m1s, m2s], [muM, sM] )

    ### Non - parametric
    elif mass_model=='DPUC':

        w, mu, sd, logw  = Lambda[-5:-1]
        Nmax=Lambda[-1]

        
        # Broadcast to (n_comp, n_obs)
        diff1 = m1s[None, :] - mu[0][:, None]
        diff2 = m2s[None, :] - mu[1][:, None]
        
        sd1 = sd[0][:, None]
        sd2 = sd[1][:, None]
        
        # Per-dimension log-Normal pdfs, broadcasted
        logp1 = -0.5 * (diff1**2 / (sd1**2)) - 0.5 * at.log(2 * atools.PI) - at.log(sd1)
        logp2 = -0.5 * (diff2**2 / (sd2**2)) - 0.5 * at.log(2 * atools.PI) - at.log(sd2)
        
        # Sum the two independent dimensions → (n_comp, n_obs)
        logp_components = logp1 + logp2
        
        # Mixture over components → (n_obs,)
        lpmass = at.logsumexp(logp_components + logw[:, None], axis=0)
    
    elif mass_model=='DP':

        alpha, beta, w, mu, fishers, ldets_inv, logw  = Lambda[-8:-1]
        Nmax=Lambda[-1]

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
            - 0.5 * nd * at.log(2.0 * np.pi)
            + 0.5 * ldets_inv[:, None]
            + logw[:, None]
        )                                           # (K, N)
        
        # 5) Mixture over components -> per-observation log-lik
        lpmass = at.logsumexp(logp_components, axis=0)  # (N,)

    
    ###################################
    # return log pdf
    ####################################

    lp = lpmass + lpspin - log_ddL_dz - 2*at.log1p(z) 

    if (invert_dL_GP or (not is_GP_dL) or is_inj ):

        # if we sampled from p(z) directly, we don't need to add p(z) twice
        
        print('Adding p(z) to p_pop')
        lp += lpz 

    if (is_inj and is_GP_dL and (not invert_dL_GP)):
            
            # we sampled from a normalised p(z) at the numerator, so for injections we need to normalize
            print('In the selection bias, normalizing p(z) to its integral to match numerator')
            zgrid_ = at.sort(at.unique(at.concatenate( [at.linspace(0, 1e-05, 10 ), at.linspace(1e-05, 20, 500), at.linspace(20, 100, 20) ] )))
            
            zint_ = at.exp(atools.log_p_z_MD_unnorm(zgrid_, gamma, kappa, zp, Lambda_c, dc=dc))
            znorm = atools.attrapzvec(zint_, zgrid_)

            lp -= at.log(znorm)

    return lp





def sel_bias_with_uncertainty_at(m1inj, m2inj, dLinj, spinsInj, log_p_draw, 
                                 Lambda,  
                                 Ndraw, 
                                 rate_model, mass_model, spin_model, 
                                 is_GP_dL, 
                                 smoothing, has_m2_break, 
                                 #distance_ratio=None, 
                                 #d_distance_ratio_d_z=None, 
                                 log_ddL_dz_inj = None,
                                 zinj = None,
                                 dcinj = None,
                                 **kwargs
				):


    if (spin_model=='default') or (spin_model=='default_gauss'):
        spinsInj_sel = [spinsInj[0], spinsInj[1], spinsInj[2], spinsInj[3]]
    elif spin_model=='none':
        spinsInj_sel = []


    if not is_GP_dL:
        H0, Om, w0, Xi0, n  = Lambda[:5]
        zinj = atools.z_from_dL_at(dLinj, H0, Om, w0, [Xi0, n] , is_GP_dL )
        #distance_ratio , d_distance_ratio_d_z = None, None
        dcinj = atools.dcfun_at( zinj, H0, Om, w0 )
        log_ddL_dz_inj = atools.log_ddL_dz( zinj, H0, Om, w0, Xi0, n, dc=dcinj )
        

    m1Src  = m1inj/(1+zinj)
    m2Src  = m2inj/(1+zinj)

    if mass_model in ('DP', 'DPUC'):
        Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
        log_Mc_src_inj = at.log(Mc_src_inj)
        logit_q_inj = atools.logitat(q_inj)      
        mass_1_use = log_Mc_src_inj
        mass_2_use = logit_q_inj
    else:
        mass_1_use = m1Src
        mass_2_use = m2Src

    log_p_pop = log_p_pop_at(mass_1_use, 
                             mass_2_use, 
                             zinj, 
                             dLinj, 
                             spinsInj_sel, 
                             Lambda, 
                             rate_model, mass_model, spin_model, 
                             is_GP_dL, 
                             smoothing=smoothing, has_m2_break=has_m2_break,
                             #dr_val=distance_ratio, 
                             #ddr_dz=d_distance_ratio_d_z, 
                             dc = dcinj,
                             log_ddL_dz = log_ddL_dz_inj,
                             is_inj=True
                            )


    if mass_model in ('DP', 'DPUC'):
        # remove jacobian m1, m2 --> log(Mc), logit(q)
        log_p_pop += (- at.log(m2Src) - at.log(m1Src-m2Src) - at.log1p(zinj) )
        print("remove jacobian m1, m2 --> log(Mc), logit(q) in sel. bias")
    else:
        print("No jacobian m1, m2 --> log(Mc), logit(q) in sel. bias")
        
    log_sel_b = log_p_pop-log_p_draw
  
    
    log_mu = at.logsumexp(log_sel_b) - at.log(Ndraw)
    
    logs2 = at.logsumexp(2.0*log_sel_b) - at.log(Ndraw )


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
    
    #logNeff = -atools.logdiffexp( logs2-2*log_mu, -at.log(Ndraw) )


    #####################################
    # This is N_eff as in Talbot Golomb 2023
    # Difference between the two is ~1/N_draw , so negligible for large injection sets
    #####################################

    logNeff = 2*log_mu - logs2 + at.log(Ndraw)

    #####################################
    # This is variance of log l per unit obs as in Talbot Golomb 2023
    #####################################

    var_log_lik_u = atools.logdiffexp( logs2-2*log_mu, 0.) - at.log(Ndraw - 1)

    Neff = at.exp(logNeff)
    
    
    return log_mu, Neff, var_log_lik_u
    

#####################################################

def _isfinite(x):
    return ~(at.isnan(x) | at.isinf(x))

        
def sel_bias_with_uncertainty_at_0_batched_scan_GPU(
    m1inj, m2inj, dLinj, spinsInj, log_p_draw,
    Lambda, Ndraw,
    rate_model, mass_model, spin_model,
    is_GP_dL,
    smoothing='LVK',
    has_m2_break=False,
    log_ddL_dz_inj=None,
    zinj=None,
    dcinj=None,
    dL_grid=None,
    z_grid=None,
    dc_grid=None,
    log_ddL_dz_grid=None,
    invert_dL_GP=True,
    chunk_size=4096,
    verbose=False,
    **kwargs
):
    import pytensor
    import pytensor.tensor as at

    def _maybe_tensor(x):
        try:
            return at.as_tensor_variable(x)
        except Exception:
            return x

    def _pad_to_multiple_1d(x, k, pad_value):
        x = _maybe_tensor(x)
        if x.ndim != 1:
            x = at.flatten(x, 1)
        N = x.shape[0]
        C = (N + k - 1) // k
        Npad = C * k - N
        pad = at.full((Npad,), pad_value, dtype=x.dtype)
        xpad = at.concatenate([x, pad], axis=0)
        return xpad.reshape((C, k)), C, N

    def _combine_logsumexp(m_s, s_s, m_c, s_c):
        only_s = at.eq(s_c, 0.0)
        only_c = at.eq(s_s, 0.0)

        m_new_raw = at.maximum(m_s, m_c)
        s_new_raw = s_s * at.exp(m_s - m_new_raw) + s_c * at.exp(m_c - m_new_raw)

        m_new = at.switch(only_s, m_s, at.switch(only_c, m_c, m_new_raw))
        s_new = at.switch(only_s, s_s, at.switch(only_c, s_c, s_new_raw))

        return m_new, s_new

    spin_is_default = (spin_model in ("default", "default_gauss"))
    spin_is_chieffchip = (spin_model in ("chieffchip", "chieffchip_uc"))
    use_dp = (mass_model in ("DP", "DPUC"))

    have_precomputed_z = zinj is not None
    have_precomputed_dc = dcinj is not None
    have_precomputed_logdd = log_ddL_dz_inj is not None

    have_dLz = (dL_grid is not None) and (z_grid is not None)
    have_dc_grid = dc_grid is not None
    have_logdd_grid = log_ddL_dz_grid is not None

    if is_GP_dL and (not invert_dL_GP):
        raise NotImplementedError(
            "scan-GPU currently supports GP distances only with invert_dL_GP=True"
        )

    if is_GP_dL and invert_dL_GP:
        if not (have_dLz or (have_precomputed_z and have_precomputed_logdd)):
            raise ValueError(
                "For is_GP_dL=True and invert_dL_GP=True, pass either "
                "(dL_grid, z_grid, log_ddL_dz_grid) or (zinj, log_ddL_dz_inj)."
            )

    K = int(chunk_size)

    m1_all = _maybe_tensor(m1inj)
    m2_all = _maybe_tensor(m2inj)
    dL_all = _maybe_tensor(dLinj)
    lpd_all = _maybe_tensor(log_p_draw)

    m1K, C, N = _pad_to_multiple_1d(m1_all, K, m1_all[0])
    m2K, _, _ = _pad_to_multiple_1d(m2_all, K, m2_all[0])
    dLK, _, _ = _pad_to_multiple_1d(dL_all, K, dL_all[0])
    lpdK, _, _ = _pad_to_multiple_1d(lpd_all, K, lpd_all[0])

    if spin_is_default:
        s1K, _, _ = _pad_to_multiple_1d(_maybe_tensor(spinsInj[0]), K, _maybe_tensor(spinsInj[0])[0])
        s2K, _, _ = _pad_to_multiple_1d(_maybe_tensor(spinsInj[1]), K, _maybe_tensor(spinsInj[1])[0])
        ct1K, _, _ = _pad_to_multiple_1d(_maybe_tensor(spinsInj[2]), K, _maybe_tensor(spinsInj[2])[0])
        ct2K, _, _ = _pad_to_multiple_1d(_maybe_tensor(spinsInj[3]), K, _maybe_tensor(spinsInj[3])[0])
    elif spin_is_chieffchip:
        chieff_arr = _maybe_tensor(spinsInj[0])
        chip_arr = _maybe_tensor(spinsInj[1])
        chieffK, _, _ = _pad_to_multiple_1d(chieff_arr, K, chieff_arr[0])
        chipK, _, _ = _pad_to_multiple_1d(chip_arr, K, chip_arr[0])

    if have_precomputed_z:
        zinj_t = stop_grad(_maybe_tensor(zinj))
        zK, _, _ = _pad_to_multiple_1d(zinj_t, K, zinj_t[0])
    else:
        zinj_t = None
        zK = None

    if have_precomputed_dc:
        dcinj_t = stop_grad(_maybe_tensor(dcinj))
        dcK, _, _ = _pad_to_multiple_1d(dcinj_t, K, dcinj_t[0])
    else:
        dcinj_t = None
        dcK = None

    if have_precomputed_logdd:
        logdd_t = stop_grad(_maybe_tensor(log_ddL_dz_inj))
        logddK, _, _ = _pad_to_multiple_1d(logdd_t, K, logdd_t[0])
    else:
        logdd_t = None
        logddK = None

    valid_mask = (at.arange(C * K) < N).reshape((C, K))

    if is_GP_dL:
        Lambda_seq = [
            Lambda[0],
            Lambda[1],
            Lambda[2],
            at.as_tensor_variable(0.0),  # placeholder to preserve indexing
        ] + list(Lambda[4:])
    else:
        Lambda_seq = list(Lambda)

    n_Lambda = len(Lambda_seq)

    if have_dLz:
        dL_grid_t = stop_grad(_maybe_tensor(dL_grid))
        z_grid_t = stop_grad(_maybe_tensor(z_grid))
        dc_grid_t = stop_grad(_maybe_tensor(dc_grid)) if have_dc_grid else None
        logdd_grid_t = stop_grad(_maybe_tensor(log_ddL_dz_grid)) if have_logdd_grid else None

        dL_flat = dLK.reshape((-1,))
        idx_flat, r_flat = atools._interp_indices_nonuniform(dL_flat, dL_grid_t)
        idxK = at.clip(idx_flat.reshape((C, K)), 1, dL_grid_t.shape[0] - 1)
        rK = r_flat.reshape((C, K))
    else:
        dL_grid_t = None
        z_grid_t = None
        dc_grid_t = None
        logdd_grid_t = None
        idxK = None
        rK = None

    seqs = [m1K, m2K, dLK, lpdK, valid_mask]

    if spin_is_default:
        seqs += [s1K, s2K, ct1K, ct2K]
    elif spin_is_chieffchip:
        seqs += [chieffK, chipK]

    if have_precomputed_z:
        seqs += [zK]
    if have_precomputed_dc:
        seqs += [dcK]
    if have_precomputed_logdd:
        seqs += [logddK]

    if have_dLz:
        seqs += [idxK, rK]

    nonseq = []
    if have_dLz:
        nonseq += [dL_grid_t, z_grid_t]
        if have_dc_grid:
            nonseq += [dc_grid_t]
        if have_logdd_grid:
            nonseq += [logdd_grid_t]

    nonseq += Lambda_seq

    def step(*args):
        pos = 0

        m1 = args[pos]; pos += 1
        m2 = args[pos]; pos += 1
        dL = args[pos]; pos += 1
        lpd = args[pos]; pos += 1
        mask = args[pos]; pos += 1

        if spin_is_default:
            chi1 = args[pos]; pos += 1
            chi2 = args[pos]; pos += 1
            cost1 = args[pos]; pos += 1
            cost2 = args[pos]; pos += 1
            spins_use = [chi1, chi2, cost1, cost2]
        elif spin_is_chieffchip:
            chieff = args[pos]; pos += 1
            chip = args[pos]; pos += 1
            spins_use = [chieff, chip]
        else:
            spins_use = []

        zinj_c = None
        dcinj_c = None
        logdd_c = None

        if have_precomputed_z:
            zinj_c = args[pos]; pos += 1
        if have_precomputed_dc:
            dcinj_c = args[pos]; pos += 1
        if have_precomputed_logdd:
            logdd_c = args[pos]; pos += 1

        if have_dLz:
            idxs_loc = args[pos]; pos += 1
            r = args[pos]; pos += 1
        else:
            idxs_loc = None
            r = None

        m_state = args[pos]; pos += 1
        m2_state = args[pos]; pos += 1
        s1_state = args[pos]; pos += 1
        s2_state = args[pos]; pos += 1

        if have_dLz:
            dL_grid_local = args[pos]; pos += 1
            z_grid_local = args[pos]; pos += 1
            dc_grid_local = args[pos] if have_dc_grid else None
            if have_dc_grid:
                pos += 1
            logdd_grid_local = args[pos] if have_logdd_grid else None
            if have_logdd_grid:
                pos += 1
        else:
            dL_grid_local = None
            z_grid_local = None
            dc_grid_local = None
            logdd_grid_local = None

        Lambda_local = list(args[pos:pos + n_Lambda])

        H0 = Lambda_local[0]
        Om = Lambda_local[1]
        w0 = Lambda_local[2]

        if zinj_c is None:
            if have_dLz:
                il = idxs_loc - 1
                ih = idxs_loc
                zl = z_grid_local[il]
                zh = z_grid_local[ih]
                zinj_c = (1.0 - r) * zl + r * zh
            else:
                if is_GP_dL:
                    raise ValueError(
                        "GP scan-GPU path needs either zinj or (dL_grid, z_grid)."
                    )
                Xi0 = Lambda_local[3]
                n = Lambda_local[4]
                zinj_c = atools.z_from_dL_at(dL, H0, Om, w0, [Xi0, n], is_GP_dL)

        if dcinj_c is None:
            if have_dLz and have_dc_grid:
                il = idxs_loc - 1
                ih = idxs_loc
                dcl = dc_grid_local[il]
                dch = dc_grid_local[ih]
                dcinj_c = (1.0 - r) * dcl + r * dch
            else:
                dcinj_c = atools.dcfun_at(zinj_c, H0, Om, w0, interp=False)

        if logdd_c is None:
            if have_dLz and have_logdd_grid:
                logdd_c = atools.atinterp(zinj_c, z_grid_local, logdd_grid_local)
            else:
                if is_GP_dL:
                    raise ValueError(
                        "GP scan-GPU path needs either log_ddL_dz_inj or log_ddL_dz_grid."
                    )
                Xi0 = Lambda_local[3]
                n = Lambda_local[4]
                logdd_c = atools.log_ddL_dz(zinj_c, H0, Om, w0, Xi0, n, dc=dcinj_c)

        # Geometry is deterministic lookup structure here; stop gradients through it
        zinj_c = stop_grad(zinj_c)
        dcinj_c = stop_grad(dcinj_c)
        logdd_c = stop_grad(logdd_c)

        one_p_z = 1.0 + zinj_c
        m1Src = m1 / one_p_z
        m2Src = m2 / one_p_z

        if use_dp:
            Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
            mass_1_use = at.log(at.maximum(Mc_src_inj, eps))
            mass_2_use = atools.logitat(q_inj)
        else:
            mass_1_use = m1Src
            mass_2_use = m2Src

        lp = log_p_pop_at(
            mass_1_use,
            mass_2_use,
            zinj_c,
            dL,
            spins_use,
            Lambda_local,
            rate_model,
            mass_model,
            spin_model,
            is_GP_dL,
            smoothing=smoothing,
            has_m2_break=has_m2_break,
            log_ddL_dz=logdd_c,
            dc=dcinj_c,
            is_inj=True,
            invert_dL_GP=invert_dL_GP,
        )

        if use_dp:
            lp += (
                -at.log(at.maximum(m2Src, eps))
                -at.log(at.maximum(m1Src - m2Src, eps))
                -at.log1p(zinj_c)
            )

        x = at.where(mask, lp - lpd, NEG_BIG)

        finite_x = _isfinite(x)
        any_finite = at.any(finite_x)
        x_safe = at.where(finite_x, x, NEG_BIG)

        m = at.switch(any_finite, at.max(x_safe), 0.0)
        y = at.switch(any_finite, at.exp(x_safe - m), at.zeros_like(x_safe))

        s1c = at.sum(y)
        s2c = at.sum(y * y)

        m_new, s1_new = _combine_logsumexp(m_state, s1_state, m, s1c)
        m2c = 2.0 * m
        m2_new, s2_new = _combine_logsumexp(m2_state, s2_state, m2c, s2c)

        return m_new, m2_new, s1_new, s2_new

    m_init = at.as_tensor_variable(-np.inf, dtype="float64")
    s_init = at.as_tensor_variable(0.0, dtype="float64")

    scan_kwargs = dict(
        fn=step,
        sequences=seqs,
        outputs_info=[m_init, m_init, s_init, s_init],
        non_sequences=nonseq,
        strict=True,
        profile=True,
    )

    try:
        (m_out, m2_out, s1_out, s2_out), _ = pytensor.scan(**scan_kwargs, return_steps=1)
        m_last = m_out[-1]
        m2_last = m2_out[-1]
        s1_last = s1_out[-1]
        s2_last = s2_out[-1]
    except TypeError:
        (m_out, m2_out, s1_out, s2_out), _ = pytensor.scan(**scan_kwargs)
        m_last = m_out[-1]
        m2_last = m2_out[-1]
        s1_last = s1_out[-1]
        s2_last = s2_out[-1]

    logsumexp1 = m_last + at.log(at.maximum(s1_last, tinyL))
    logsumexp2 = m2_last + at.log(at.maximum(s2_last, tinyL))

    Ndraw_t = Ndraw
    log_mu = logsumexp1 - at.log(Ndraw_t)
    logs2 = logsumexp2 - at.log(Ndraw_t)

    logNeff = 2.0 * log_mu - logs2 + at.log(Ndraw_t)
    Neff = at.exp(logNeff)

    var_log_lik_u = atools.logdiffexp(logs2 - 2.0 * log_mu, 0.0) - at.log(Ndraw_t - 1.0)

    return log_mu, Neff, var_log_lik_u




#####################################################
#####################################################

# Prior transforms

#####################################################
#####################################################


# 95% central interval for Normal and 95% point for HalfNormal
NORM_Q95 = 1.959963984540054
NORM_Q99 = 2.5758293035489004
# 2.5758293035489004  # 99% point (Phi^{-1}(0.995))
#1.959963984540054  # Phi^{-1}(0.975)

# For bounded-sigmoid params, choose raw sd so that 95% maps to ~[0.05, 0.95]
RAW_SD_95 = 1.502  # since sigmoid(±1.96*RAW_SD_95) ≈ 0.05 / 0.95

def normal_from_bounds_95(name, low, high, initval=None):
    """Interpret [low, high] as central 95% of a Normal."""
    mu = 0.5 * (low + high)
    sigma = (high - low) / (2.0 * NORM_Q95)
    return pm.Normal(name, mu=mu, sigma=sigma, initval=initval)


def floored_lognormal_q95(name, floor, typical_max_total, initval=None, median_frac=0.2):
    """
    sigma = floor + x, with x ~ LogNormal(mu, sigma_ln)

    We set:
      Q95(x) = raw_typ = typical_max_total - floor
      median(x) = median_frac * raw_typ   (default 0.2)

    This avoids mass piling up at the floor (lognormal density -> 0 as x->0).
    """
    raw_typ = max(1e-12, typical_max_total - floor)
    med = max(1e-12, median_frac * raw_typ)

    # For LogNormal: median = exp(mu), and Q95 = exp(mu + z95*sigma_ln)
    mu = np.log(med)
    sigma_ln = (np.log(raw_typ) - mu) / NORM_Q95

    raw_init = None
    if initval is not None:
        raw_init = max(1e-12, initval - floor)

    x = pm.LogNormal(f"{name}_raw", mu=mu, sigma=sigma_ln, initval=raw_init)
    return pm.Deterministic(name, floor + x)

def floored_halfnormal_typmax95(name, floor, typical_max_total, initval=None):
    """
    Parameter = floor + HalfNormal(raw_sigma),
    with typical_max_total interpreted as the ~95% point of the final parameter.
    So raw 95% point = (typical_max_total - floor).
    """
    raw_typ = max(1e-12, typical_max_total - floor)
    raw_sigma = raw_typ / NORM_Q95  # because P(HN <= raw_typ) = 0.95
    raw_init = None
    if initval is not None:
        raw_init = max(0.0, initval - floor)
    raw = pm.HalfNormal(f"{name}_raw", sigma=raw_sigma, initval=raw_init)
    return pm.Deterministic(name, floor + raw)



def unit_interval_sigmoid(name, initval=None, raw_sigma=1.0):
    """Unconstrained raw ~ Normal(0, raw_sigma), mapped to (0,1) via sigmoid."""
    raw_init = None
    if initval is not None:
        x = float(np.clip(initval, 1e-6, 1.0 - 1e-6))
        raw_init = np.log(x / (1.0 - x))
    raw = pm.Normal(f"{name}_raw", mu=0.0, sigma=raw_sigma, initval=raw_init)
    return pm.Deterministic(name, pm.math.sigmoid(raw))


def bounded_sigmoid(name, low, high, initval=None, raw_sigma=RAW_SD_95):
    """low + (high-low)*sigmoid(raw), raw ~ Normal(0, raw_sigma)."""
    raw_init = None
    if initval is not None:
        t = float((initval - low) / (high - low))
        t = np.clip(t, 1e-6, 1.0 - 1e-6)
        raw_init = np.log(t / (1.0 - t))
    raw = pm.Normal(f"{name}_raw", mu=0.0, sigma=raw_sigma, initval=raw_init)
    return pm.Deterministic(name, low + (high - low) * pm.math.sigmoid(raw))


def log_bounded_sigmoid_95(name, low, high, initval=None):
    """Hard-bounded positive scale in [low, high] but uniform-ish in log space."""
    raw_init = None
    if initval is not None:
        t = float((np.log(initval) - np.log(low)) / (np.log(high) - np.log(low)))
        t = np.clip(t, 1e-6, 1 - 1e-6)
        raw_init = np.log(t / (1 - t))
    raw = pm.Normal(f"{name}_raw", mu=0.0, sigma=RAW_SD_95, initval=raw_init)
    logx = np.log(low) + (np.log(high) - np.log(low)) * pm.math.sigmoid(raw)
    return pm.Deterministic(name, pm.math.exp(logx))


def bounded_sigmoid_logpositive(name, low, high, initval=None, raw_sigma=1.5):
    """
    Parameterize x in [low, high] via log-space:
      logx = lowlog + (highlog-lowlog)*sigmoid(raw)
      raw ~ Normal(0, raw_sigma)
      x = exp(logx)
    This is symmetric around 1 when [low, high] is symmetric in log space (i.e., high=1/low).
    """
    lowlog, highlog = np.log(low), np.log(high)

    raw_init = None
    if initval is not None:
        t = float((np.log(initval) - lowlog) / (highlog - lowlog))
        t = np.clip(t, 1e-6, 1.0 - 1e-6)
        raw_init = np.log(t / (1.0 - t))

    raw = pm.Normal(f"{name}_raw", mu=0.0, sigma=raw_sigma, initval=raw_init)
    logx = pm.Deterministic(f"{name}_log", lowlog + (highlog - lowlog) * pm.math.sigmoid(raw))
    return pm.Deterministic(name, pm.math.exp(logx))



#####################################################
#####################################################


def make_model(  priors,
                 GWData,
                 InjData,
                 ivals={},
                 sampling_GW = 'gmm',
                 rate_model = 'MD',
                 mass_model = 'PLP',
                 reparam_mass = True, 
                 reparam_z = True,
                 reparam_cosmo = True,
                 smoothing='LVK',
                 has_m2_break = False,
                 spin_model = 'none',
                 spin_inj = 'none',
                 marginal_R0 = True,
                 dLprior = 'none',
                 fix_inj_len = False,
                 sel_method='Tobs',
                 N_DP_comp_max = 20,
                is_GP_dL = True,
               find_GP_L = True,
               fout=None,
               monotonicity = 'poly',
                 eps_DE = -1,
                 monotonicity_scale = 1. ,
                 zmin_mono = 0, 
                 zres=150,
                zmin_a=1e-05, zmin_b=1e-03, zmid_b=3.5, zmax_c=100.0, hi_boost=0.20,
                 find_z_bounds = False,
                nu = 0.25,
                 lam = 10,
                 clip_low = -500,
                 clip_high=500,
               GP_prior = 'frechet',
                 large_ell_penalty=False,
               GP_zero_point = 'y',
               rescale_GP=False,
               invert_dL_GP = True,
               dense_grad = False,
                 fix_H0 = True,
                fix_Om = True,
               fix_w0 = True,
                 fix_Xi0n = True,
                 fix_rate = False,
               pade=False,
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
                 U = 10.,
                ell_min=0.05,
                 
                 ell_max=3,
                 res_lowz = 0.1,
                 res_highz = 0.1,
                 fine_res = 0.01,
                 fix_mass=0,
                 inj_loop = 'scan-GPU',
                 chunk_inj=4096,
                ):

    ################################################
    # Read in data and set dimensions
    ################################################

    ## GW data
    if not pop_only:
        
        # gw data are interpolants of single-event posteriors
        if sampling_GW=='gauss' :
            # we sample single-event parameters from broad gaussian approximations of the posteriors
            mus_s, cho_s, log_wts_l, mus_l, icovs_l, log_dets_l, cho_covs_l, Tobs, Nevs = GWData
            wts_l = np.exp(log_wts_l)
        
        elif 'gmm' in sampling_GW :

            wts_l, mus_l, cho_covs_l, icovs_l, log_dets_l, mus_l_sub, icovs_l_sub, log_dets_l_sub, Tobs, Nevs = GWData
            nsub = mus_l_sub.shape[2]
            print('nsub is %s'%nsub)
            
            if not invert_dL_GP:
                nsub = mus_l_sub.shape[2]
                print('nsub is %s'%nsub)
            

        else:
            raise ValueError('sampling_GW can be cho or gauss ')
            
        

    else:
        # gw data are single-event posterior samples
        # shape of each has to be n_events, n_samples
        m1det, m2det, d, spins, dL_prior, Tobs, allNsamples, where_compute, Nevs, allnames = GWData           

       
        if (spin_model=='default') or (spin_model=='default_gauss'):
           chi1, chi2, cost1, cost2 = spins
        elif spin_model=='none':
            pass
        else:
            raise NotImplementedError()

    ## Injections data
    if spin_inj == 'none':
        dLinj, m1inj, m2inj, lpdinj, Ndraw, Ndet = InjData
    elif spin_inj == 'chieffchip':
        dLinj, m1inj, m2inj, chiefffInj, chipInj, lpdinj, Ndraw, Ndet = InjData
    elif (spin_inj == 'chi12xyz' or spin_inj == 'default'):
        if (spin_model=='default') or (spin_model=='default_gauss'):
            dLinj, m1inj, m2inj, chi1Inj, chi2Inj, cost1Inj, cost2Inj, lpdinj, Ndraw, Ndet = InjData
        elif spin_model == 'none':
            dLinj, m1inj, m2inj, lpdinj, Ndraw, Ndet = InjData

    
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
        #print("N samples max will be ")
        #print(Nsamples_np)
        #print('N:%s, n samples: %s '%(N_np, allNsamples_np))

        if Nsamplesuse !=-1 :
            if Nsamplesuse>Nsamples_np:
                raise ValueError("Must use less samples than those available.")
            print("Nsamples_np available is %s, but %s will be used"%(Nsamples_np, Nsamplesuse))

            m1det, m2det, d = m1det[:, :Nsamplesuse], m2det[:, :Nsamplesuse], d[:, :Nsamplesuse]
            dL_prior = dL_prior[:, :Nsamplesuse]

            spins = np.asarray([s[:, :Nsamplesuse] for s in spins ])
            if (spin_model=='default') or (spin_model=='default_gauss'):
               chi1, chi2, cost1, cost2 = chi1[:, :Nsamplesuse], chi2[:, :Nsamplesuse], cost1[:, :Nsamplesuse], cost2[:, :Nsamplesuse]

            allNsamples = Nsamplesuse

            Nsamples = m1det.shape[1]
            Nsamples_np = Nsamples #Nsamples.eval()
            allNsamples_np = np.full( N, Nsamplesuse )

        else:
            allNsamples_np = allNsamples 


        assert np.all( allNsamples_np == Nsamples_np )
        print("N samples will be ")
        print(Nsamples_np)
        print('N:%s, n samples: %s '%(N_np, Nsamples_np))


        ### reshape

        if spin_model in ("default", "default_gauss"):
            spins = np.transpose( np.stack([chi1, chi2, cost1, cost2], axis=1) , (0,2,1) ) # (N,4)
            print("spins shape is %s"%str(spins.shape))

             
        logd = np.log(d)
        
        NsamplesTot = N*Nsamples

        print("Reshaping samples to %s"%NsamplesTot)

        if pop_only:
            m1det_matrix = copy.deepcopy(m1det)
            m2det_matrix = copy.deepcopy(m2det)
            d_matrix = copy.deepcopy(d)
        
        m1det = m1det.reshape(NsamplesTot)
        m2det = m2det.reshape(NsamplesTot)
        d = d.reshape(NsamplesTot)
        logd = logd.reshape(NsamplesTot)
        dL_prior = dL_prior.reshape(NsamplesTot)

        dL_log_prior = np.log(dL_prior)

        # print("dL prior start ")
        # print(dL_prior[:5])

        # print("dL^2 start ")
        # print(d[:5]**2)

        # print("log(dL_prior) start ")
        # print( (dL_log_prior[:5]) )
        
        # spins: if you store (Ne, S, nspin) -> flatten first two axes
        spins = spins.reshape((NsamplesTot, spins.shape[-1]))



    logN = np.log(N)
    
    event_index = np.arange(N_np, dtype=int)
    
    ndata = m1inj.shape[0] # number of observing runs to combine
    ndata_np = ndata #ndata.eval()
    ninj = m1inj.shape[1] # max number of injections
    ninj_np = ninj #ninj.eval()

    Ttot = np.sum(Tobs)

    
    print('Injections: :%s, '%(ninj_np))

    print('ninj: :%s, %s datasets,'%(Ndet_np, ndata_np))

    coords = {'event_index': event_index}

    

    if mass_model in ('DP', 'DPUC'):
        coords['component'] = np.arange(N_DP_comp_max_np, dtype=int)
        coords['GMMdimension'] = np.arange(2, dtype=int)
        coords['GMMdimension_1'] = np.arange(2, dtype=int)
        coords['GMMdimension_2'] = np.arange(2, dtype=int)
        p = 2*(2+1)//2  # packed length = 3 for n=2
        
        coords["packed_cholesky"] = np.arange(p)

    if pop_only:
        coords['nsamples'] = np.arange( Nsamples_np, dtype=int )
    else:
         coords['GWdimension'] = np.arange(nd_np, dtype=int)


    if params_fix is None:
        print('No values for parameters to fix passed. Default values will be used. If fixing parameters, check that the values are consistent. Values of fixed parameters:')
        print(PLPeakO3params)
        params_fix=PLPeakO3params

    if is_GP_dL and find_z_bounds:

        
        U = at.as_tensor_variable( U ) #2.5)         # upper bound for σ with high probability
        
        alpha = at.as_tensor_variable(0.01)    # small tail probability
        
        lambda_ = at.log(1 / alpha) / U
        
        alpha_ell = at.as_tensor_variable(0.005)
        alpha_large = at.as_tensor_variable(0.01)
        
        d_GP = at.as_tensor_variable(1)


    
        rng = np.random.default_rng()

        print("\nFinding optimal points for redshift interpolation and min prior lengthscale for GP...")
        
        print("min, max redshift search grid: %s, %s"%(1e-100, 1e05))
        
        print("Compiling functions...")
        # --- Compile once: z_from_dL and midpoint derivative ---
        z_sym      = at.dvector('z_nodes')    # if you need it
        d_sym      = at.dvector('dL_nodes')
        H0_sym     = at.dscalar('H0')
        Om_sym     = at.dscalar('Om')
        w0_sym     = at.dscalar('w0')
        Xi0_sym     = at.dscalar('Xi0')
        n_sym     = at.dscalar('nXi0')

        
        # your existing functions but returning NODE arrays
        # r, H0, Om, w0, Lambda_MG, is_GP_dL, z_grid
        z_from_dL_sym = atools.z_from_dL_at(d_sym, H0_sym, Om_sym, w0_sym, [Xi0_sym, n_sym], False, atools.zGridGlobals_at_long )
        #dc_nodes_sym  = atools.dcfun_at(z_sym, H0_sym, Om_sym, w0_const, interp=False)
        #d_log_dLEM_dz_sym = atools.ddL_dz_EM(z_sym, H0_sym, Om_sym, w0_const)
        #lb_mid_fn = pytensor.function([z_sym, H0_sym, Om_sym, ], d_log_dLEM_dz_sym)
        z_from_dL_fn = pytensor.function([d_sym, H0_sym, Om_sym, w0_sym, Xi0_sym, n_sym], z_from_dL_sym)

        print("Done.")

        print("Priors grid for search:")
        if fix_H0:
            priors['H0'] = ( params_fix['H0'], params_fix['H0'])
        if fix_Om:
            priors['Om'] = ( params_fix['Om'], params_fix['Om'])
        if fix_w0:
            priors['w0'] = ( -1, -1)

        print('H0')
        print(priors['H0'])
        print('Om')
        print(priors['Om'])
        print('w0')
        print(priors['w0'])
        print('Xi0')
        print(priors['Xi0'])
        print('n')
        print(priors['nXi0'])
        
        #priors['Xi0'] = ( 0.05, 10)
        #priors['nXi0'] = ( 0.05, 10)

        


            
        if not pop_only:
    
            min_z, max_z, z_min_data, z_max_data, z_diff, z_span = putils.find_zgrid_bounds(wts_l, mus_l, cho_covs_l,
                                          priors['H0'], priors['Om'], priors['w0'], priors['Xi0'], priors['nXi0'], 
                                          int(N), int(nd),
                                        dLinj,
                                        z_from_dL_fn,
                                          sampling_GW,
                                          trials=1000, 
                                            return_diff=True                                     
                                         )
        else:

            min_z, max_z, z_min_data, z_max_data, z_diff, z_span = putils.find_zgrid_bounds_from_dL_samples(
                    priors['H0'], priors['Om'], priors['w0'], priors['Xi0'], priors['nXi0'],
                    dLinj,
                    d_matrix,   # shape (n_events, n_samples)
                    z_from_dL_fn,
                    trials=1000,
                    s0=0.10,
                    rng=np.random.default_rng(123),
                    return_diff=True,
            )

        
        
        zmin_b = min(min_z, z_min_data)

        zmin_a = min( zmin_a, min(min_z, z_min_data))
        
        zmid_b = min( zmid_b, z_max_data )
        zmax_c = max(zmax_c, max(z_max_data, max_z))*(1+0.1)

        print("Redshift values found, overwriting default:")
        print("zmin_a=%s, zmin_b=%s, zmid_b=%s, zmax_c=%s"%(zmin_a, zmin_b, zmid_b, zmax_c))


        z_max_mono = max(z_max_data, max_z)

        if ell_min>0:
            print("z_diff found to be %s"%z_diff)
            print("Min length scale passed by hand, = %s. Using max(ell_min, z_diff) "%ell_min)
            ell_min = max( z_diff, ell_min )   
        else:
            ell_min = z_diff

        if ell_max>0:
            print("z_span found to be %s"%z_span)
            print("Max length scale passed by hand, = %s. Using max(ell_max, z_span) "%ell_max)
            ell_max = max( z_span, ell_max )   
        else:
            print("z_span found to be %s"%z_span)
            print("Using this z_span as ell_max.")
            ell_max = z_span
            
        
    
        print(f"ell_min:                  {ell_min:.6g}")
        print(f"ell_max:                  {ell_max:.6g}")
        print(f"z_max_mono:                  {z_max_mono:.6g}")



        if invert_dL_GP:
            print()


            # z_nodes_np, z_fine_np = atools.make_z_grids_GP(
            #     zmin=zmin_a, zmid=zmid_b, zmax=zmax_c,
            #     dz_low_nodes=0.05, n_high_nodes=60,
            #     dz_low_fine=0.01, n_high_fine=300,
            # )
            z_nodes_np, z_fine_np = atools.make_z_grids_GP(zmin=zmin_a, zmax=zmax_c,
                                n_nodes=160,
                                n_fine=900,
                                n_ramp_nodes=12,  # extra points in [zmin, z0)
                                n_ramp_fine=20,)

            
            
            zgrid_      = stop_grad(at.as_tensor_variable(z_nodes_np))
            zgrid_fine_ = stop_grad(at.as_tensor_variable(z_fine_np))

            
            #zgrid_ = stop_grad( at.as_tensor_variable(  np.unique( np.sort( np.concatenate( [np.arange(zmin_a, zmid_b, res_lowz ), np.arange(zmid_b, zmax_c, res_highz ) ]))) ) )
            

            print("z grid for interpolation built. ")
            #print("Resolution up to %s: %s"%(zmid_b, res_lowz))
            #print("Resolution between %s and %s: %s"%(zmid_b, zmax_c, res_highz))
            print("Total len: %s"%(zgrid_.shape.eval()))
            

            #zgrid_fine_ = stop_grad( at.as_tensor_variable(  np.arange(zmin_a, zmax_c, fine_res ) ))
            
            
            print("z fine for integration/monotonicity built")
            #print("Resolution: %s"%(fine_res))
            #print("Resolution between %s and %s: %s"%(zmid_b, zmax_c, 0.5))
            print("Total len: %s"%(zgrid_fine_.shape.eval()))
            
            print("z min: %s , z max: %s"%(z_fine_np.min(), z_fine_np.max()))
        
        beta = 0.1 #atools.find_beta(ell_min, 2., p0=0.01)
        al = 0.05 #atools.find_al(ell_min, 10., p0=0.01)

       
        print()
        lambda_ell = -at.log(alpha_ell) * ell_min**(d_GP / 2)
        print('lambda_ell is %s'%lambda_ell.eval())

        lambda_large = -np.log(alpha_large) / ell_max
        print('lambda_large is %s'%lambda_large.eval())

        print('z_max_mono is %s'%z_max_mono)

        # import matplotlib.pyplot as plt
        # from scipy.stats import gamma
        # from scipy.stats import halfnorm
        # from scipy.stats import invgamma
        # ℓ_vals = at.geomspace(1e-05, 100, 1000)
        # logp_vals = atools.frechet_logp_full(ℓ_vals, lambda_ell, d_GP) 
        # pdf_gamma = gamma.pdf(ℓ_vals.eval(), a=2., scale=1/beta)
        # pdf_gamma_inv = invgamma.pdf( ℓ_vals.eval(), a=al, scale=1/10 )
        # pdf_l = halfnorm(scale=1).pdf(ℓ_vals.eval())
        # plt.plot(ℓ_vals.eval(), at.exp(logp_vals).eval(), label='frechet')
        # plt.plot(ℓ_vals.eval(), pdf_gamma, label='gamma')
        # plt.plot(ℓ_vals.eval(), pdf_l, label='halfnorm')
        # plt.plot(ℓ_vals.eval(), pdf_gamma_inv, label='inv gamma')
        # plt.xlabel("ℓ")
        # plt.ylabel("Prior density")
        # plt.title("PC prior on ℓ")
        # plt.yscale("log")
        # plt.xscale("log")
        # plt.ylim(1e-05,10)
        # plt.axvline(ell_min, ls='--', color='k')
        # plt.legend()
        # plt.grid()
        # #plt.show()
        # plt.savefig( os.path.join(fout, 'ell_prior.pdf'), bbox_inches='tight')
        # plt.close()

    if sampling_GW=='gmm_cat' and not pop_only:
        # we sample single-event parameters from the actual single-event posteriors
        # need tensor variables to correctly slice inside model
        wts_l, mus_l, cho_covs_l = at.constant(wts_l), at.constant(mus_l), at.constant(cho_covs_l)



    
    ################################################
    # Build model
    ################################################
    
    with pm.Model(coords=coords) as model:


        # if sampling_GW=='gauss'  and not pop_only:
        #     # we sample single-event parameters from broad gaussian approximations of the posteriors
        #     mus_s, cho_s, log_wts_l, mus_l, icovs_l, log_dets_l = at.as_tensor_variable(mus_s), at.as_tensor_variable(cho_s), at.as_tensor_variable(log_wts_l), at.as_tensor_variable(mus_l), at.as_tensor_variable(icovs_l), at.as_tensor_variable(log_dets_l)
        # elif 'gmm' in sampling_GW  and not pop_only:
        #     # we sample single-event parameters from the actual single-event posteriors
        #     wts_l, mus_l, cho_covs_l = at.as_tensor_variable(wts_l), at.as_tensor_variable(mus_l), at.as_tensor_variable(cho_covs_l)

        if sampling_GW=='gmm_cat' and not pop_only:
            # we sample single-event parameters from the actual single-event posteriors
            # need tensor variables to correctly slice inside model
            wts_l, mus_l, cho_covs_l = at.constant(wts_l), at.constant(mus_l), at.constant(cho_covs_l)


        ################################################
        # Cosmological parameters
        ################################################

        
        if fix_H0:
            H0_ =  params_fix['H0']
        else:
            if not reparam_cosmo:
                H0_ =  pm.Uniform('H0', lower=priors['H0'][0], upper=priors['H0'][1], initval=ivals.get('H0'))
            else:
                print("Reparametrized prior for H0")
                H0_  = bounded_sigmoid("H0", *priors["H0"], initval=ivals.get("H0"), raw_sigma=1 )


        
        if fix_Om:
            Om_ = params_fix['Om']
        else:
            if not reparam_cosmo:
                Om_ = pm.Uniform('Om', lower=priors['Om'][0], upper=priors['Om'][1], initval=ivals.get('Om')) 
            else:
                print("Reparametrized prior for Om")
                Om_ = bounded_sigmoid("Om", priors["Om"][0], priors["Om"][1], raw_sigma=1, initval=ivals.get("Om"))
                

        if fix_w0:
            w0_ = -1.
        else:
            if pade:
                raise NotImplementedError("Pade appproximation with varying w0 not implemented yet. Use pade=False")
            if not reparam_cosmo:
                w0_ =  pm.Uniform('w0', lower=priors['w0'][0], upper=priors['w0'][1], initval=ivals.get('w0'))
            else:
                print("Reparametrized prior for w0")
                w0_ = bounded_sigmoid("w0", priors["w0"][0], priors["w0"][1], raw_sigma=1, initval=ivals.get("w0"))
        

        Lambda_ = [H0_, Om_, w0_]

        
        if not is_GP_dL:
            if fix_Xi0n:
                Xi0_ =  at.as_tensor_variable(1.)
                nXi0_ = at.as_tensor_variable(0.)
            else:
                Xi0_ =  pm.Uniform('Xi0', lower=priors['Xi0'][0], upper=priors['Xi0'][1])
                nXi0_ = pm.Uniform('n', lower=priors['n'][0], upper=priors['n'][1]) 

            Lambda_MG_ = [ Xi0_, nXi0_]
            iastro=5

        else:
            print('Modeling d^GW/d^EM as a Gaussian process')

            # GP hyperparameters
            #ℓ = pm.HalfNormal("ℓ", sigma=1.0)
            #η = pm.HalfNormal("η", sigma=1.0)
            # mu = pm.Normal("mu", 0, 1)


            
            # Actual length scale
            if GP_prior=='frechet':
                ℓ = pm.CustomDist( "ℓ", 
                                  lambda_ell,
                                  d_GP,
                                  logp=atools.frechet_logp_full,
                                   transform=tr.log,    # enforces ℓ > 0 via log-transform
                                   initval=1,
                                  random=atools.frechet_random,
                                 )
                print('ℓ prior is frechet')
                
                if large_ell_penalty:
                
                    print('Add large ℓ penalty')
                    _ = pm.Potential(
                        "pc_large_ell",
                        -lambda_large * ℓ
                                )
                else:
                    print('No large ℓ penalty')
            
            elif GP_prior=='gamma':
                ℓ = pm.Gamma("ℓ", alpha=2., beta=beta)
                print('ℓ prior is Gamma')
            elif GP_prior=='gammainv':
                ℓ = pm.InverseGamma("ℓ", alpha=al, beta=0.1 )
                print('ℓ prior is Inverse Gamma')
            else:
                raise ValueError()
            
            η = pm.Exponential("η", lam = lambda_)
            print('η prior is Exponential with lambda=%s, from scale U=%s'%(lambda_.eval(), U.eval()))

            cov = η**2 * pm.gp.cov.Matern52( input_dim=1, ls=ℓ ) + pm.gp.cov.WhiteNoise(1e-4)
            gp = pm.gp.Latent(cov_func=cov)
                
            Lambda_MG_ = [ gp  ] 
            iastro = 4
        Lambda_ += Lambda_MG_   
        ################################################
        # Redshift evolution of merger rate
        ################################################
        
        if rate_model=='MD':
            
            print('Modeling evolution of merger rate with redshift with Madau-Dickinson profile')

            if fix_rate:
                print("Fixing rate parameters!")
                gamma_ = pm.Deterministic('gamma', at.as_tensor_variable(3.2) )
                kappa_ = pm.Deterministic('kappa', at.as_tensor_variable(3) )
                zp_ = pm.Deterministic('zp', at.as_tensor_variable(2) )

            else:
                if not reparam_z:
                    gamma_ = pm.Uniform('gamma', lower=priors['gamma'][0], upper=priors['gamma'][1], initval=ivals.get('gamma'))    
                    kappa_ = pm.Uniform('kappa', lower=priors['kappa'][0], upper=priors['kappa'][1], initval=ivals.get('kappa'))
                    zp_ = pm.Uniform('zp', lower=priors['zp'][0], upper=priors['zp'][1], initval=ivals.get('zp'))
    
                else:
                    print("Reparametrized prior for z")
    
                    gamma_a, gamma_b = priors["gamma"]
                    kappa_a, kappa_b = priors["kappa"]
                    zp_a, zp_b       = priors["zp"]
                    
                    gamma_ = bounded_sigmoid("gamma", gamma_a, gamma_b, raw_sigma = 1.5, initval=ivals.get('gamma') )
                    kappa_ = bounded_sigmoid("kappa", kappa_a, kappa_b, raw_sigma=1, initval=ivals.get('kappa') )
                    zp_    = bounded_sigmoid("zp",    zp_a,    zp_b,    raw_sigma=1, initval=ivals.get('zp') )

            Lambda_ += [gamma_, kappa_, zp_]

        elif rate_model=='PL':
            print('Modeling evolution of merger rate with a power law')
            gamma_ = pm.Uniform('gamma', lower=priors['gamma'][0], upper=priors['gamma'][1], initval=ivals.get('gamma'))

            Lambda_ += [gamma_]

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
                    bound_alphaChi_val = pm.Potential('bound_alphaChi', atools.log_f_smooth_poly(alphaChi_, 5e-4,  1 )  )
                    bound_betaChi_val = pm.Potential('bound_betaChi', atools.log_f_smooth_poly(betaChi_, 5e-4,  1  ))
                elif alpha_beta_prior=='sigmoid':
                    print("Tapering prior on alpha_chi, beta_chi with sigmoid smoothing")
                    bound_alphaChi_val = pm.Potential('bound_alphaChi', atools.log_sigmoid(alphaChi_,  1+3e-04, 1e-04)  )
                    bound_betaChi_val = pm.Potential('bound_betaChi', atools.log_sigmoid(betaChi_, 1+3e-04, 1e-04)  )
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

            # muChi_ = pm.Uniform('muChi', lower=priors['muChi'][0], upper=priors['muChi'][1])
            # sigmaChi_ = pm.Uniform('sigmaChi', lower=priors['sigmaChi'][0], upper=priors['sigmaChi'][1])
            
            # zeta_ = pm.Uniform('zeta', lower=priors['zeta'][0], upper=priors['zeta'][1])
            # sigmat_ = pm.Uniform('sigmat', lower=priors['sigmat'][0], upper=priors['sigmat'][1])


            
            # --- reparameterized bounded priors (muChi, sigmaChi, zeta) and HalfNormal (sigmat) ---

             
            muChi_a, muChi_b = priors["muChi"]
            muChi_ = bounded_sigmoid("muChi", muChi_a, muChi_b, raw_sigma = 1.5, initval=ivals.get("muChi", 0.024) )
            

            sigmaChi_a, sigmaChi_b = priors["sigmaChi"]
            sigmaChi_ = bounded_sigmoid("sigmaChi", sigmaChi_a, sigmaChi_b, raw_sigma=1.5, initval=ivals.get("sigmaChi", 0.32))
            
            # zeta in [a,b] via sigmoid reparam
            zeta_a, zeta_b = priors["zeta"]
            zeta_ = bounded_sigmoid("zeta", zeta_a, zeta_b, raw_sigma = 1.5, initval=ivals.get("zeta", 0.2))
            
           
            
            # sigmat: HalfNormal with "typical max" = priors['sigmat'][1] at ~95% quantile
            HN_Q95_TO_SIGMA = 1.959963984540054  # Phi^{-1}(0.975)
            sigmat_floor = priors["sigmat"][0]
            sigmat_typmax = priors["sigmat"][1]
            raw_typ = max(1e-12, sigmat_typmax - sigmat_floor)  # interpret typmax as final 95% point
            sigmat_sigma = raw_typ / HN_Q95_TO_SIGMA
            
            sigmat_raw_init = None
            if ivals.get("sigmat") is not None:
                st = ivals["sigmat"]
            else:
                st=3.
            sigmat_raw_init = max(0.0, st - sigmat_floor)
            
            sigmat_raw = pm.HalfNormal("sigmat_raw", sigma=sigmat_sigma, initval=sigmat_raw_init)
            sigmat_ = pm.Deterministic("sigmat", sigmat_floor + sigmat_raw)



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


        elif mass_model=='DPLDP' or mass_model=='PLDP':

            if mass_model=='DPLDP':
                
                print('Modeling mass distribution with Double Power Law + Double Peak ')

            else:
                print('Modeling mass distribution with single Power Law + Double Peak ')

            
            epsilon_  = pm.Deterministic( "epsilon", at.as_tensor_variable( 0.1 ) )

            
            if not reparam_mass:

                alpha1_ = pm.Uniform("alpha1", lower=priors["alpha1"][0], upper=priors["alpha1"][1], initval=ivals.get("alpha1"))

                m_high_   = pm.Uniform("m_high",      lower=priors["m_high"][0],      upper=priors["m_high"][1],      initval=ivals.get("m_high", 150)) 
                #pm.Deterministic("m_high", at.as_tensor_variable(300.0)) #.astype(X)  )

                delta_m1_ = pm.Uniform("delta_m1", lower=priors["delta_m1"][0], upper=priors["delta_m1"][1], initval=ivals.get("delta_m1"))
                

                if mass_model == "DPLDP":

                    
                    alpha2_ = pm.Uniform("alpha2", lower=priors["alpha2"][0], upper=priors["alpha2"][1], initval=ivals.get("alpha2"))
                    
                    mb_     = pm.Uniform("mb", lower=priors["mb"][0], upper=priors["mb"][1], initval=ivals.get("mb"))

                    
                    u         = pm.Uniform("u", 0, 1, initval=ivals.get("u"))
                    m1_low_   = pm.Deterministic("m1_low", 3 + (10 - 3) * at.sqrt(u))
                    
                    v         = pm.Uniform("v", 0, 1, initval=ivals.get("v"))
                    m2_low_   = pm.Deterministic("m2_low", 3 + v * (m1_low_ - 3))
    
                   
                    delta_m2_ = pm.Uniform("delta_m2", lower=priors["delta_m2"][0], upper=priors["delta_m2"][1], initval=ivals.get("delta_m2"))


                    
                else:
                    alpha2_ = pm.Deterministic("alpha2", alpha1_)   # same name: alpha2
                    mb_     = pm.Deterministic("mb", at.as_tensor_variable(35.0))
                    delta_m2_     = pm.Deterministic("delta_m2", delta_m1_ )

                    m1_low_   = pm.Uniform("m1_low", lower=priors["m1_low"][0], upper=priors["m1_low"][1], initval=ivals.get("m1_low") )

                    m2_low_ = pm.Deterministic("m2_low", m1_low_ )
                
                    
                    

                # Gaussian components 
                
                mu1_      = pm.Uniform("mu1",      lower=priors["mu1"][0],      upper=priors["mu1"][1],      initval=ivals.get("mu1"))
                sigma1_   = pm.Uniform("sigma1",   lower=priors["sigma1"][0],   upper=priors["sigma1"][1],   initval=ivals.get("sigma1"))
                
                mu2_      = pm.Uniform("mu2",      lower=priors["mu2"][0],      upper=priors["mu2"][1],      initval=ivals.get("mu2"))
                sigma2_   = pm.Uniform("sigma2",   lower=priors["sigma2"][0],   upper=priors["sigma2"][1],   initval=ivals.get("sigma2"))

                
                # Mixture weights 
                
                lambda_vec = pm.Dirichlet("lambda", a=np.asarray([1, 1, 1]), initval=np.asarray(ivals.get("lambda")))
                lambda0_  = pm.Deterministic("lambda0", lambda_vec[0])
                lambda1_  = pm.Deterministic("lambda1", lambda_vec[1])
                lambda2_  = pm.Deterministic("lambda2", lambda_vec[2])
                
                beta_     = pm.Uniform("beta",     lower=priors["beta"][0],     upper=priors["beta"][1],     initval=ivals.get("beta"))
                
            
            
            else:

                # --- Slopes / locations: Normal with bounds as 95% typical range ---
          
                print("Using reparametrized mass priors")

                # --- Triangle constraint for m1_low, m2_low preserved ---
                u = unit_interval_sigmoid("u", initval=ivals.get("u"), raw_sigma=1)
                m1_low_ = pm.Deterministic("m1_low", 3 + (10 - 3) * u**1.5 )

                
                # delta_m1 + taper end
                d1_floor = priors["delta_m1"][0]
                d1_typ   = priors["delta_m1"][1]
                delta_m1_ = floored_lognormal_q95("delta_m1", d1_floor, d1_typ, initval=ivals.get("delta_m1"), median_frac=0.3)
                m1_taper_end_ = pm.Deterministic("m1_taper_end", m1_low_ + delta_m1_)

                
                
                

                if mass_model=='DPLDP':
                    if priors["alpha1"] != priors["alpha2"]: raise ValueError(f"alpha1/alpha2 priors differ: {priors['alpha1']} vs {priors['alpha2']}")
                        
                    # bounds -> mid and sigma (same as helper)
                    a_low, a_high = priors["alpha1"][0], priors["alpha1"][1]
                    a_mid = 0.5 * (a_low + a_high)
                    a_sig = (a_high - a_low) / (2.0 * NORM_Q95)
                    
                    # reparam
                    a_bar  = pm.Normal("alpha_bar",  mu=a_mid, sigma=a_sig,
                                       initval=ivals.get("alpha_bar", ivals.get("alpha1")))
                    a_diff = pm.Normal("alpha_diff", mu=0.0,   sigma=np.sqrt(2.0) * a_sig,
                                       initval=ivals.get("alpha_diff", 0.0))
                    
                    alpha1_ = pm.Deterministic("alpha1", a_bar - 0.5 * a_diff)
                    alpha2_ = pm.Deterministic("alpha2", a_bar + 0.5 * a_diff)


                    mb_ = bounded_sigmoid("mb", priors["mb"][0], priors["mb"][1], raw_sigma=1, initval=ivals.get("mb") )

                    

                    
                    v = unit_interval_sigmoid("v", initval=ivals.get("v"), raw_sigma=1)
                    m2_low_ = pm.Deterministic("m2_low", 3 + v * (m1_low_ - 3))

                
                    # delta_m2 + taper end
                    d2_floor = priors["delta_m2"][0]
                    d2_typ   = priors["delta_m2"][1]
                    delta_m2_ = floored_lognormal_q95("delta_m2", d2_floor, d2_typ, initval=ivals.get("delta_m2"), median_frac=0.3)
                    m2_taper_end_ = pm.Deterministic("m2_taper_end", m2_low_ + delta_m2_)
    

                else:
                    # alpha_bar prior on same bounds as before (95% within [a_low, a_high])
                    a_low, a_high = priors["alpha1"][0], priors["alpha1"][1]
                    a_mid = 0.5 * (a_low + a_high)
                    a_sig = (a_high - a_low) / (2.0 * NORM_Q95)
                    
                    a_bar = pm.Normal(
                        "alpha_bar",
                        mu=a_mid,
                        sigma=a_sig,
                        initval=ivals.get("alpha_bar", ivals.get("alpha1")),
                    )

                    a_diff = pm.Deterministic("alpha_diff",  at.as_tensor_variable(0.0))
                    
                    # enforce single slope: alpha1 == alpha2 == alpha_bar
                    alpha1_ = pm.Deterministic("alpha1", a_bar)
                    alpha2_ = pm.Deterministic("alpha2", a_bar)
                    
                    # fix break mass (optional but recommended for identifiability)
                    mb_ = pm.Deterministic("mb", at.as_tensor_variable(35.0))


                    delta_m2_     = pm.Deterministic("delta_m2", delta_m1_ )

                    m2_low_ = pm.Deterministic("m2_low", m1_low_ )

                    
    
    
                beta_   = normal_from_bounds_95("beta",   priors["beta"][0],   priors["beta"][1],   initval=ivals.get("beta"))
                
                
               
    
                  
                # --- Widths: floor + HalfNormal, with priors[*][1] treated as 95% typical max ---
                  
                sigma1_ = floored_lognormal_q95("sigma1", priors["sigma1"][0], priors["sigma1"][1], initval=ivals.get("sigma1"), median_frac=0.2)
                sigma2_ = floored_lognormal_q95("sigma2", priors["sigma2"][0], priors["sigma2"][1], initval=ivals.get("sigma2"), median_frac=0.3)
    
   

                mu1_ = bounded_sigmoid("mu1", priors["mu1"][0], priors["mu1"][1], raw_sigma=1.25, initval=ivals.get("mu1") )
                mu2_ = bounded_sigmoid("mu2", priors["mu2"][0], priors["mu2"][1], raw_sigma=1.25, initval=ivals.get("mu2") )


                
                

   
                mhigh_floor = priors["m_high"][0]
                mmax_median = 0.5 * (priors["m_high"][0] + priors["m_high"][1])
                mmax_q95    = priors["m_high"][1]
                
                delta_med = at.maximum(mmax_median - mhigh_floor, 1e-6)
                delta_q95 = at.maximum(mmax_q95    - mhigh_floor, 1e-6)
                
                mu_delta    = at.log(delta_med)
                sigma_delta = (at.log(delta_q95) - mu_delta) / NORM_Q95
                
                delta_mhigh = pm.LogNormal("delta_mhigh", mu=mu_delta, sigma=sigma_delta)
                m_high_     = pm.Deterministic("m_high", mhigh_floor + delta_mhigh)
                
    
                
                
                
                            
                # --- Lambda  ---
                lam_init = ivals.get("lambda")
                if lam_init is None:
                    lam_init = np.array([1/3, 1/3, 1/3])
                lambda_vec = pm.Dirichlet("lambda", a=np.asarray([1, 1, 1]), initval=np.asarray(lam_init))
                lambda0_  = pm.Deterministic("lambda0", lambda_vec[0])
                lambda1_  = pm.Deterministic("lambda1", lambda_vec[1])
                lambda2_  = pm.Deterministic("lambda2", lambda_vec[2])



            
            if has_m2_break:
                print("Including gap for secondary mass")
                m_g_     =  pm.Uniform("m_g", lower=priors["m_g"][0], upper=priors["m_g"][1], initval=ivals.get("m_g")) 
                w_g_     = pm.Uniform("w_g", lower=priors["w_g"][0], upper=priors["w_g"][1], initval=ivals.get("w_g")) 
                sig_g_l_ = at.as_tensor_variable(1e-02)#.astype(X)
                sig_g_h_ = at.as_tensor_variable(1e-02)#.astype(X)
            else:
                m_g_     = at.as_tensor_variable(45.)#.astype(X)
                w_g_     = at.as_tensor_variable(70.)#.astype(X)
                sig_g_l_ = at.as_tensor_variable(1e-02)#.astype(X)
                sig_g_h_ = at.as_tensor_variable(1e-02)#.astype(X)
            
            Lambda_ += [alpha1_, alpha2_, mb_, mu1_, sigma1_, mu2_, sigma2_, m1_low_, m_high_, delta_m1_, lambda0_, lambda1_, lambda2_, beta_, m2_low_, delta_m2_, epsilon_, m_g_, w_g_, sig_g_l_, sig_g_h_]


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
        elif mass_model=='DPUC':
            print("Modeling mass distribution as Dirichelet Process. Max number of components: %s"%N_DP_comp_max)

            
            alpha = pm.Gamma("alpha", 1.0, 1.0)
            beta = pm.Beta("beta", 1.0, alpha, dims="component" )
            w = pm.Deterministic("w", atools.stick_breaking(beta), dims="component")
            logw = at.log(w)


            #### Sigma prior limits: 


            # Option 1: Fixes std from (\tau * \lambda ) parametrization
            # Here check how to choose the parameters of the Gamma priors!

            #---- global–local prior on SDs (uncorrelated) ----
            # controls local variability
            # a_lam1, b_lam1 = 0.6, 0.6
            # a_lam2, b_lam2 =   0.5, 0.5
            
            # # One global precision per dimension (shared across components)
            # a_tau1, b_tau1 =   24, 4  # for z1 = log(Mc)
            # a_tau2, b_tau2 =  4, 2 # for z2 = logit(q)
            # tau1 = pm.Gamma("tau1", a_tau1, b_tau1)
            # tau2 = pm.Gamma("tau2", a_tau2, b_tau2)
            
            # # Local precisions per component, per dimension
            # lam1 = pm.Gamma("lam1", a_lam1, b_lam1, dims="component")
            # lam2 = pm.Gamma("lam2", a_lam2, b_lam2, dims="component")
            
            # # Per-component, per-dimension standard deviations (independent axes)
            # sig1 = pm.Deterministic("sig1", 1.0 / at.sqrt(tau1 * lam1), dims="component")
            # sig2 = pm.Deterministic("sig2", 1.0 / at.sqrt(tau2 * lam2), dims="component")



            # mu_ln1, s_ln1 = -1.163789654413117, 0.36380796502993723
            # mu_ln2, s_ln2 = -1.1074484102105713, 0.48718303162186627
            # sig1_max, sig2_max = 0.7971738948530454, 2.0086525885321533

            # # In your PyMC model (uncorrelated shown; for correlated use as marginals):
            # sig1 = pm.Truncated("sig1", pm.LogNormal.dist(mu=mu_ln1, sigma=s_ln1),
            #                     lower=0.0, upper=sig1_max, dims="component")
            # sig2 = pm.Truncated("sig2", pm.LogNormal.dist(mu=mu_ln2, sigma=s_ln2),
            #                     lower=0.0, upper=sig2_max, dims="component")


            # sig1 = pm.Uniform("sig1", lower=0.01, upper=5, dims="component")
            # sig2 = pm.Uniform("sig2", lower=0.01, upper=10, dims="component")


            sig1 = pm.InverseGamma("sig1", alpha=4.72, beta=1.39, dims="component")
            sig2 = pm.InverseGamma("sig2", alpha=0.5, beta=0.4, dims="component")
            
            

            sd = pm.Deterministic("sig", at.stack([sig1, sig2], axis=0),  # (2,K)
                      dims=("GMMdimension", "component"))


            #### Mean prior limits:  remember that mu is log(Mc), logit(q).

            # Option 1 : sample mean of the gaussians from given prior
            # with this choice, the prior on the mean will be flat in log(Mc), logit(q).
        

            mu1 = pm.Uniform('mulMc', lower=1.13, upper=4.38, dims= ("component" ))
            mu2 = pm.Uniform('mulq', lower=-2.75, upper=9.37, dims= ("component" ))
            

            # z1_lo = 1.13
            # z1_hi = 4.38
            # mu1_mid = (z1_lo+z1_hi)/2
            # mu1_sd = (-z1_lo+z1_hi)/2
            # span1 = -z1_lo+z1_hi

            # z2_lo = -2.75
            # z2_hi = 9.37
            # mu2_mid = (z2_lo+z2_hi)/2
            # mu2_sd = (-z2_lo+z2_hi)/2
            # span2 = -z2_lo+z2_hi

            # mu1 = pm.TruncatedNormal("mu1", mu=mu1_mid, sigma=mu1_sd,
            #              lower=z1_lo-0.5*span1, upper=z1_hi+0.5*span1,
            #              dims="component")
            # mu2 = pm.TruncatedNormal("mu2", mu=mu2_mid, sigma=mu2_sd,
            #              lower=z2_lo-0.5*span2, upper=z2_hi+0.5*span2,
            #              dims="component")


            mu = pm.Deterministic("mu", at.stack([mu1, mu2], axis=0),  # (2,K)
                      dims=("GMMdimension", "component"))

            # Option 2: check ...

            Lambda_ += [ w, mu, sd, logw ]

            Lambda_ += [N_DP_comp_max]

        elif mass_model=='DP':

            alpha = pm.Gamma("alpha", 1.0, 1.0)
            beta = pm.Beta("beta", 1.0, alpha, dims="component" )
            w = pm.Deterministic("w", atools.stick_breaking(beta), dims="component")
            logw = at.log(w)


            mu1 = pm.Uniform('mulMc', lower=1.13, upper=4.38, dims= ("component" ))
            mu2 = pm.Uniform('mulq', lower=-2.75, upper=9.37, dims= ("component" ))

            mu = pm.Deterministic("mu", at.stack([mu1, mu2], axis=0),  # (2,K)
                      dims=("GMMdimension", "component"))


            ################################################
            # cholesky option 1 (slower)

            # ---- Per-component LKJ Cholesky (NO batching) ----
            # packed_list = []
            # L_list = []
            # for k in range(N_DP_comp_max):
            #     pk = pm.LKJCholeskyCov(f"chol_packed_{k}",
            #                            n=2, eta=2.0, sd_dist=pm.HalfNormal.dist(2.5),
            #                            compute_corr=False)     # returns length-3 vector
            #     Lk = pm.expand_packed_triangular(2, pk, lower=True)  # (2,2)
            #     packed_list.append(pk)
            #     L_list.append(Lk)
    
            # # Stack to tensors
            # chol_packed = at.stack(packed_list, axis=0)   # (K, 3)
            # L = at.stack(L_list, axis=0)                  # (K, 2, 2)
    
            # # Optional: save deterministics with coords
            # #pm.Deterministic("chol_packed_all", chol_packed, dims=("component",))
            # #pm.Deterministic("Sigma_chol", L, dims=("component","dim","dim"))
    
            # # Precompute Σ^{-1} and log|Σ|
            # invL = pm.math.matrix_inverse(L)                  # (K,2,2)
            # Fisher = at.matmul(at.swapaxes(invL, -1, -2), invL)
            # ldets_inv = 2.0 * (pm.math.log(L[:,0,0]) + pm.math.log(L[:,1,1]))  # (K,)



            ################################################
            # cholesky option 2

            # # ---- prior on SDs (uncorrelated) ----
            # # Per-component, per-dimension standard deviations (independent axes)

            sig1 = pm.InverseGamma("sig1", alpha=4.72, beta=1.39, dims="component")
            sig2 = pm.InverseGamma("sig2", alpha=0.5, beta=0.4, dims="component")

            #sig1 = pm.Uniform("sig1", lower=0.01, upper=3, dims="component")
            #sig2 = pm.Uniform("sig2", lower=0.01, upper=5, dims="component")

            # # ----- Correlation prior equivalent to LKJ(eta) in 2D -----
            eta = 1.0  # uninformative on correlations
            rho_u = pm.Beta("rho_u", alpha=eta, beta=eta, dims="component")   # (0,1)
            rho   = pm.Deterministic("rho", 2.0 * rho_u - 1.0, dims="component")  # (-1,1)


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
                -2.0 * at.log(sig1) - 2.0 * at.log(sig2) - at.log(one_minus_r2),
                dims="component",
            )
            
            # ----- Precision Σ^{-1} in closed form (Fisher) -----
            # Σ^{-1} = 1 / [ (1-ρ^2) s1^2 s2^2 ] * [[ s2^2, -ρ s1 s2 ], [ -ρ s1 s2, s1^2 ]]
            den = one_minus_r2 * (sig1**2) * (sig2**2)
            F11 =  (sig2**2)            / den
            F22 =  (sig1**2)            / den
            F12 = -(rho * sig2 * sig1)    / den
            
            Fisher = pm.Deterministic( "Fisher", at.stack([
                at.stack([F11, F12], axis=1),
                at.stack([F12, F22], axis=1)
            ], axis=1), dims=("component","GMMdimension_1","GMMdimension_2"))  # shape: (K, 2, 2)

            
            ################################################

            Lambda_ += [ alpha, beta, w, mu, Fisher, ldets_inv, logw ]

            Lambda_+=[N_DP_comp_max]
            
        ################################################
        # If including total normalization of the rate, add it here
        ################################################
        
        if not marginal_R0:
            R0 = pm.Uniform('R0', lower=priors['R0'][0], upper=priors['R0'][1])
        else:
            R0 = at.as_tensor_variable(1.)    
        lR0 = at.log(R0)


        if not pop_only:
            ################################################
            # Individual event mass and distance
            ################################################
    
            x = pm.Normal( 'x', mu=0, sigma=1, dims= ("event_index" , "GWdimension" ) )
                
            if 'gmm' in sampling_GW:
    
                print('Sampling m1d, m2d, dL from GMM')

                if sampling_GW=='gmm_cat':
                    ###################################

                    ig = pm.Categorical('idx', p=wts_l, dims= "event_index" )

                else:
                    ###################################

                    # ---------- 1) shared selector: hom vs cat AND galaxy index ----------
                    u_gmm = pm.Normal("u_gmm", 0.0, 1.0, dims= "event_index")
                    v_gmm = at.clip(atools.normal_cdf(u_gmm), 1e-9, 1.0 - 1e-9) 

                    # inverse-CDF over weights
                    cdf_w = at.cumsum(wts_l, axis=1)                                          
                    ig = pm.Deterministic('idx', (v_gmm[:, None] < cdf_w).argmax(axis=1), dims= "event_index" )             

                
                # old way. leave it here  please
                # samples = mus_l[ at.arange(N), ig, :] + at.batched_dot( cho_covs_l[at.arange(N), ig, :, :], x )
                
                # Select means and Cholesky factors per batch
                mu_selected = mus_l[ np.arange(N), ig, :]         # shape (N, D)
                L_selected = cho_covs_l[ np.arange(N), ig, :, :]  # shape (N, D, D)

                 
                # # Batched matrix multiplication: (N, D, D) @ (N, D, 1) → (N, D, 1)
                Lx = at.sum(L_selected * x[:, None, :], axis=2)  # → shape (N, D)
                
                # # Final transformed sample
                samples = mu_selected + Lx                # shape (N, D)

                

                
                log_Mc_det = samples[:,0]/dil_factor
                logit_q = samples[:,1]
                logd = samples[:,2]
                
    
                if (spin_model == 'chieffchip') or (spin_model == 'chieffchip_uc') :
        
                    chieff = atools.inv_flogitat(samples[:,3])
                    chip = atools.inv_logitat(samples[:,4])
        
                elif (spin_model == 'default') or (spin_model == 'default_gauss'):
                    # we have chi1, chi2, cost1, cost2
        
                    chi1 = pm.Deterministic('chi1', atools.inv_logitat(samples[:,3]))
                    chi2 = pm.Deterministic('chi2', atools.inv_logitat(samples[:,4]))
        
                    cost1 = pm.Deterministic('cost1', atools.inv_flogitat(samples[:,5]))
                    cost2 = pm.Deterministic('cost2', atools.inv_flogitat(samples[:,6]))
                else:
                    print("No spins computed")
              
            

            
            elif sampling_GW=='gauss' : # to be tested with spins
                
                print('Sampling log(Mc), logit(q), log(dL) from Gaussian approximant')

                # sample = mu + L @ x   (batched)
                samples = mus_s + at.matmul(cho_s, x[..., None])[..., 0]      # (N, d)
                
                # logp = log p(x) - log|L|
                # d = x.shape[1]
                log_px = -0.5 * at.sum(x**2, axis=1) - 0.5 * x.shape[1] * at.log(2.0 * np.pi)    # (N,)
                log_det_L = at.sum(at.log(at.diagonal(cho_s, axis1=1, axis2=2)), axis=1)  # (N,)
                pilik = log_px - log_det_L                                               # (N,)
                

                # unpack coordinates:
                log_Mc_det = samples[:, 0]
                logit_q    = samples[:, 1]
                logd       = samples[:, 2]


                if spin_model == 'none' :
                    
                    vals = at.stack([log_Mc_det, logit_q, logd], axis=0)


                elif spin_model == 'default' :

                    chi1 = atools.inv_logitat(samples[:,3])
                    chi2 = atools.inv_logitat(samples[:,4])
        
                    cost1 = atools.inv_flogitat(samples[:,5])
                    cost2 = atools.inv_flogitat(samples[:,6])

                    vals = at.stack([log_Mc_det, logit_q, logd,  samples[:,3],  samples[:,4],  samples[:,5],  samples[:,6]], axis=0)
            

                # X as (N, d)
                X = vals.T                                   # (N, d)
                #d = vals.shape[0]
                
                # Broadcast X against component-wise parameters
                # diff: (N, ngmm, d)
                diff = X[:, None, :] - mus_l                  # (N, 1, d) - (N, ngmm, d)
                
                # Quadratic form using precision F = Σ^{-1}
                # tmp = F @ diff[..., None]  -> (N, ngmm, d, 1) -> squeeze to (N, ngmm, d)
                tmp = at.matmul(icovs_l, diff[..., None])[..., 0]   # (N, ngmm, d)

                # r^T F r for each (obs, comp)
                quad = at.sum(diff * tmp, axis=-1)            # (N, ngmm)
                
                # Component logpdfs (Multivariate Normal)
                log_norm = -0.5 * vals.shape[0] * at.log(2.0 * np.pi)     # scalar
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
            dval = at.exp(logd)
    
            
        else:
            # we are sampling the usual marginalised likelihood, with "only" pop parameters
            print('We are running inference only on population parameters.')

            # Compute source-frame quantities. One redsfhit, mass1, mass2 for each event
            # AND for each sample! 
            
            # d, logd, m1det, m2det are already reshaped.

            dval = d 
        
        
        # Compute source-frame quantities. 

        if not is_GP_dL:
                
                zs = pm.Deterministic('z', atools.z_from_dL_at(dval, H0_, Om_, w0_, Lambda_MG_ , is_GP_dL ), dims= "event_index" )

                dc = pm.Deterministic('dc', atools.dcfun_at( zs , H0_, Om_,  w0_, ), dims= "event_index" )
                 
                #distance_ratio , d_distance_ratio_d_z = None, None
                log_ddL_dz = atools.log_ddL_dz( zs, H0_, Om_, w0_, Xi0_, nXi0_, dc=dc )
            
        else:

                if rescale_GP:
                    data_range=(atools.zGridGlobals_at.min(), atools.zGridGlobals_at.max())
                else:
                    data_range=None

                if invert_dL_GP:


                    
                    # Precompute cosmology pieces (symbolic)
                    dc_grid      = atools.dcfun_at(zgrid_fine_, H0_, Om_, w0_, interp=False)
                    dLem_grid    = (1.0 + zgrid_fine_) * dc_grid
                    ddLem_dz_grid= atools.ddL_dz_EM(zgrid_fine_, H0_, Om_, w0_, dc=dc_grid)
                    
                    b_full       = atools.d_log_dLEM_dz(zgrid_fine_, H0_, Om_, w0_, dc=dc_grid, safe=False)
                    
                    # # GP log-ratio & its derivative on the grid

                    dLGrid_at, log_distance_ratio_grid, grad_log_distance_ratio_grid, \
                        log_distance_ratio_grid_fine, grad_log_distance_ratio_grid_fine = atools.z_from_dL_at(
                            None, H0_, Om_, w0_, Lambda_MG_,
                            is_GP_dL=True,
                            z_grid=zgrid_,
                            z_grid_fine=zgrid_fine_,
                            out_type='fine',
                            gp_mode=("mono_reparam" if monotonicity=="mono_reparam" else "direct"),
                            taper_z0=0.02,  # or None
                        )

                    # after building dLGrid_at and before inversion
                    oob = at.any((dval < dLGrid_at[0]) | (dval > dLGrid_at[-1]))
                    pm.Potential("interp_oob_penalty", at.switch(oob, -1e6, 0.0))

                    print("oob constraint is %s "%oob.eval())

                    if not pop_only:
                    # Event-level z
                        zs = pm.Deterministic("z", atools.atinterp(dval, dLGrid_at, zgrid_fine_), dims="event_index")
                        
                        # Event-level dc
                        dc = pm.Deterministic("dc", atools.dcfun_at(zs, H0_, Om_, w0_, interp=False), dims="event_index")

                    else:

                        zs =  atools.atinterp(dval, dLGrid_at, zgrid_fine_)
                        dc = atools.dcfun_at(zs, H0_, Om_, w0_, interp=False)
                    
                    # Distance ratio on grid (compute once)
                    distance_ratio_grid = at.exp(log_distance_ratio_grid_fine)
                    


                    if monotonicity == "mono_reparam":
                        q_grid = b_full + grad_log_distance_ratio_grid_fine
                        ddL_dz_grid = (dLem_grid * distance_ratio_grid) * q_grid
                        log_ddL_dz_grid = at.log(at.abs(ddL_dz_grid) + 1e-30)
                    else:
                        s_grid = dLem_grid * grad_log_distance_ratio_grid_fine + ddLem_dz_grid
                        ddL_dz_grid = s_grid * distance_ratio_grid
                        log_ddL_dz_grid = at.log( at.abs( ddL_dz_grid)+ 1e-30 )
                        
                    

                    
                    log_ddL_dz = atools.atinterp( zs, zgrid_fine_, log_ddL_dz_grid )
                    
                    # Interpolate what you need at zs
                    log_dratio_at_z  = atools.atinterp(zs, zgrid_fine_, log_distance_ratio_grid_fine)
                    grad_log_dr_at_z = atools.atinterp(zs, zgrid_fine_, grad_log_distance_ratio_grid_fine)



                    log_ddL_dz_at_z  = atools.atinterp(zs, zgrid_fine_, log_ddL_dz_grid)

                    if not pop_only:
                        distance_ratio = pm.Deterministic("d_ratio", at.exp(log_dratio_at_z), dims="event_index")
                        d_ratio_d_z    = pm.Deterministic("d_ratio_d_z", distance_ratio * grad_log_dr_at_z, dims="event_index")
                        log_ddL_dz     = pm.Deterministic("log_ddL_dz", log_ddL_dz_at_z, dims="event_index")

                    else:
                        distance_ratio = at.exp(log_dratio_at_z)
                        d_ratio_d_z = distance_ratio * grad_log_dr_at_z
                        log_ddL_dz = log_ddL_dz_at_z
                    
                    # Monotonicity barrier
                    print("monotonicity is %s"%monotonicity)
                    if monotonicity is not None:
                        
                        if monotonicity=='poly':
                            print('Imposing d(dL)/dz >0 on all the domain')
                            print('Using smooth polynomial, nu=%s, lam=%s'%(nu, lam))


                            if nu==-1:
                                print("sampling nu")
                                nu = pm.HalfNormal("nu", sigma=1., initval=0.5)
                            if lam==-1:
                                print("sampling lam")
                                lam = pm.HalfNormal("lam", sigma=10., initval=10)
                            

                            # GP derivative g(z)
                            g_grid = grad_log_distance_ratio_grid_fine
                            b_full_fine = atools.d_log_dLEM_dz(zgrid_fine_, H0_, Om_, w0_, dc=None, safe=False)
                            
                            
                            # dimensionless monotonicity condition
                            q_grid = g_grid + b_full_fine

                            mask = (zgrid_fine_ <= z_max_mono)  # boolean mask on the grid

                            if zmin_mono!=0:
                                print("Lower lim for monotonicity penalty at z=%s"%zmin_mono)
                                mask &= (zgrid_fine_ >= zmin_mono)

                            if monotonicity_scale==0:
                                q_mono = q_grid[mask]
                            
                                # tolerance
                                eps = 0.
                            
                                # penalise only q < -eps
                                q_tol = q_mono + eps

                                N_mono = q_tol.shape[0]
                            
                                # then in model:
                                pm.Potential("monotonicity",
                                         -lam * at.sum(atools.poly_hinge_neg(q_tol, nu)) / N_mono 
                                            )

                            else:
                                print("Standardize monotonicity with scale %s"%monotonicity_scale) 
                                g_mono = g_grid[mask]
                                b_mono = b_full_fine[mask]

                                # avoid crazy ratios if b is small but still inside mask
                                b_min  = 1e-10
                                b_safe = at.maximum(b_mono, b_min)
                                
                                # r(z) = 1 + g/b  ;  r>0 <=> dL_GW' > 0
                                r_mono = 1.0 + g_mono / b_safe

                                # standardize so the penalty sees O(1) numbers
                                
                                x = r_mono / monotonicity_scale   # dimensionless, O(1) if r is O(1)
                                N_mono = x.shape[0]
                                
                                # soft monotonicity penalty: penalize x<0 (i.e. r<0)
                                pm.Potential(
                                    "monotonicity",
                                    -lam * at.sum(atools.poly_hinge_neg(x, tau=nu)) / N_mono
                                            )


                        elif monotonicity == "mono_reparam":
                            print("Monotonicity enforced by construction (no Potential).")
                        else:
                            print('No monotonicity constraint.')
                            
                    else:
                            print('No monotonicity constraint.')


                    if eps_DE != -1:

                        print("Imposing large-z constraint on alphaM...")

 
                        Ode0 = 1.0 - Om_
                        ratio = (eps_DE / (1.0 - eps_DE)) * (Om_ / Ode0)
                        ratio = at.clip(ratio, 1e-12, 1e12)
                
                        z_eps = pm.Deterministic("z_eps_DE", at.power(ratio, 1.0 / (3.0 * w0_ )) - 1.0)
                
                        delta_z = 0.2
                        w_tail = pm.math.sigmoid((zgrid_fine_ - z_eps) / delta_z)
                        w_tail = w_tail - at.min(w_tail)
                
                        sigma_tail = pm.HalfNormal("sigma_tail", sigma=0.3)  # was 0.1
                
                        w2 = w_tail**2
                        Neff = at.sum(w2) + 1e-12
                        
                        alphaM_grid = (1.0 + zgrid_fine_) * grad_log_distance_ratio_grid_fine
                        pm.Potential(
                            "highz_turnoff",
                            -0.5 * at.sum((w_tail * alphaM_grid / sigma_tail) ** 2) / Neff
                        )

 
                
                 

        if not pop_only:
            # save values of GW distance and source-frame masses
            d = pm.Deterministic('dL', dval , dims="event_index")      

            m1src = pm.Deterministic('m1src', m1det/(1+zs) , dims="event_index")
            m2src = pm.Deterministic('m2src', m2det/(1+zs) , dims="event_index") 
        
        else:

            m1src = m1det/(1+zs)
            m2src = m2det/(1+zs)
        
                
        
        
        
        ################################################
        # Population prior
        ################################################


        if not pop_only:
            if spin_model == 'chieffchip' or spin_model == 'chieffchip_uc' :
    
                spins = [ chieff, chip  ]
    
            elif (spin_model == 'default') or (spin_model == 'default_gauss'):
    
                spins = [chi1, chi2, cost1, cost2]
    
            elif spin_model == 'none':
                
                spins = []

        # else: spins is give by spins = spins.reshape((NsamplesTot, spins.shape[-1])) so has shape NsamplesTot x 4
        else:
            spins = spins.T
            # now 4 x NsamplesTot
            # p_pop will read it as chi1, chi2, cost1, cost2 = spins so this should be ok.


        
        # Population prior of all events, without the term T_obs*R0
        if mass_model in ('DP', 'DPUC'):

            # dirichelet processs will be for log(Mc_src), logit(q) ...
            logMc_src =  log_Mc_det - at.log1p(zs)
            

            log_p_pop = log_p_pop_at( logMc_src, logit_q, zs, d, spins, 
                                     Lambda_, 
                                     rate_model, mass_model, spin_model, 
                                     is_GP_dL, 
                                     #dr_val=distance_ratio, 
                                     #ddr_dz=d_distance_ratio_d_z
                                     log_ddL_dz = log_ddL_dz,
                                     dc = dc, 
                                     invert_dL_GP=invert_dL_GP
                                    )
            

            # ... so remove a jacobian : p( m1, m2 ) = p( log(Mc), logit(q) ) * |J|
            log_p_pop -=  at.log(m2src) + at.log(m1src-m2src) + at.log1p(zs) 
            
        else:    
        
            log_p_pop = log_p_pop_at(m1src, m2src, zs, d, spins, 
                                     Lambda_, 
                                     rate_model, mass_model, spin_model, 
                                     is_GP_dL, 
                                     smoothing=smoothing, has_m2_break=has_m2_break,
                                     #dr_val=distance_ratio, 
                                     #ddr_dz=d_distance_ratio_d_z, 
                                     #monotonicity=monotonicity, 
                                     log_ddL_dz = log_ddL_dz,
                                     dc = dc, 
                                     invert_dL_GP=invert_dL_GP
                                    )
             

        if not pop_only:
        
            if dLprior=='dLsq':
                # Remove \pi(d)~dL^2 prior on distance 
                log_p_pop -= 2*logd
                print('Removing dL^2 prior')
            elif dLprior == 'dVdz':
                print('Removing prior proportional to 1/(1+z)*dV/dz with H0=67.90, Om=0.3065')
                raise NotImplementedError()
                
                log_p_pop -= lpi

        else:
            print("Using dL PE prior loaded from file.")
            log_PE_prior =  dL_log_prior


        if not pop_only:
            if sampling_GW=='gauss' :
                # Add gw likelihood and correct for sampling prior pdf
                log_p_pop -= pilik
                log_p_pop += gwl


        if (is_GP_dL and (not invert_dL_GP)):
            print('Correcting likelihood from factor R coming from sampling redshift from pop prior')
            log_p_pop += logR
        
        
        # Put it all together
        if not pop_only:
            # just sum log likelihoods
            likelihood_val = at.sum( log_p_pop ) 
        else:
            
        
            log_p_pop = (log_p_pop - log_PE_prior).reshape((N, Nsamples))

            # marginalise over single events parameters first
            log_p_pop_marg = at.logsumexp( log_p_pop, axis=1, ) - at.log(allNsamples)
            
            # then sum log likelihoods
            likelihood_val = at.sum( log_p_pop_marg )  


           # Check number of effective samples for computing MC integral 
            logs2 = at.logsumexp(2*log_p_pop, axis=1) -2*at.log(allNsamples)

            
            Neff_lik =  pm.Deterministic('Neff_l', at.exp( 2.0*log_p_pop_marg - logs2) ) 
            # this has len = n. of observations

        
            log_var_log_lik_evs_all = atools.logdiffexp( logs2 - 2.0 * log_p_pop_marg, 0. ) - at.log(allNsamples - 1.0)

            var_log_lik_evs = at.sum( at.exp(log_var_log_lik_evs_all) )
            
            if Neff_min_lik>0:

                print("Bound on effective number of samples for individual event MC integrals. Min requested: %s"%Neff_min_lik)
                _ = pm.Potential("Neff_l_bound", at.sum( at.where( Neff_lik<Neff_min_lik, -np.inf, 0. ) ) )
              
            else:
                
                print("No bound on effective number of samples for individual event MC integrals. Uncertainty will be propagated to total log lik. variance")

        
        
        # add R0*Tobs if needed. 
        if not marginal_R0:
            print("Will not marginalise over R0.")
            # each term p_pop is multiplied by
            # R0*T_obs . So we get a factor (R0*T_obs)**N_i for every
            # observing run. R0 is the same for every run so I just have
            # (R0)**{\sum N_i} . For T_obs I have T_{obs,1}**N_1 * T_{obs,2}**N_2 * ...
            poiss_term = at.sum(Nevs*at.log(allTobs))+N*lR0
            likelihood_val += poiss_term
        else:
            print("Will marginalise over R0 with flat-in-log prior.")

        
        likelihood = pm.Deterministic("lik", likelihood_val ) 
        likelihood_term = pm.Potential("likelihood", likelihood ) 
        
 

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

                if use_sel_spin:
                    spin_model_name = spin_model
                    
                    if spin_model == 'chieffchip' or spin_model == 'chieffchip_uc' :
                        spinsInj = [ chiefffInj[0], chipInj[0] ]
                        
                    elif (spin_model == 'default') or (spin_model == 'default_gauss'):
                        spinsInj = [ chi1Inj[0], chi2Inj[0], cost1Inj[0], cost2Inj[0] ]
                        
                    else:
                        spinsInj = []

                else:
                    print("Spin distribution will not be used in the sel effect")
                    spinsInj = []
                    spin_model_name = 'none'

                if is_GP_dL:
                    if (inj_loop == 'scan-GPU') and (not invert_dL_GP):
                        raise NotImplementedError(
                            "inj_loop='scan-GPU' is only implemented for is_GP_dL=True with invert_dL_GP=True"
                        )

                    # For both vec and scan-GPU, precompute symbolic per-injection quantities.
                    # This avoids fragile searchsorted calls inside the scan body under JAX.
                    zinj = atools.atinterp(dLinj[0], dLGrid_at, zgrid_fine_)
                    dc_inj = atools.dcfun_at(zinj, H0_, Om_, w0_, interp=False)
                    log_ddL_dz_inj = atools.atinterp(zinj, zgrid_fine_, log_ddL_dz_grid)

                    dL_grid_inj = None
                    z_grid_inj = None
                    dc_grid_inj = None
                    log_ddL_dz_grid_inj = None
                else:
                    zinj = None
                    dc_inj = None
                    log_ddL_dz_inj = None
                    dL_grid_inj = None
                    z_grid_inj = None
                    dc_grid_inj = None
                    log_ddL_dz_grid_inj = None                



                if inj_loop == 'scan-GPU':
                    print("Computing sel bias with GPU scan")
                    sel_bias_fun = sel_bias_with_uncertainty_at_0_batched_scan_GPU
                else:
                    print("Computing sel bias in one chunk")
                    sel_bias_fun = sel_bias_with_uncertainty_at


                if is_GP_dL and invert_dL_GP:
                    dbg_fn = pytensor.function(
                        [],
                        [
                            at.min(dLGrid_at), at.max(dLGrid_at),
                            at.min(dLinj[0]), at.max(dLinj[0]),
                            at.any(at.isnan(dLGrid_at)),
                            at.any(at.isnan(log_ddL_dz_grid)),
                            at.any(at.isnan(zgrid_fine_)),
                            at.any(dLinj[0] < dLGrid_at[0]),
                            at.any(dLinj[0] > dLGrid_at[-1]),
                        ],
                        on_unused_input="ignore",
                    )

                    dbg_vals = dbg_fn()
                    print("min(dLGrid_at), max(dLGrid_at) =", dbg_vals[0], dbg_vals[1])
                    print("min(dLinj), max(dLinj)       =", dbg_vals[2], dbg_vals[3])
                    print("any NaN dLGrid_at            =", dbg_vals[4])
                    print("any NaN log_ddL_dz_grid      =", dbg_vals[5])
                    print("any NaN zgrid_fine_          =", dbg_vals[6])
                    print("any inj below dLGrid_at[0]   =", dbg_vals[7])
                    print("any inj above dLGrid_at[-1]  =", dbg_vals[8])


                log_mu_, Neff_, var_ll_u_ = sel_bias_fun(
                    m1inj[0],
                    m2inj[0],
                    dLinj[0],
                    spinsInj,
                    lpdinj[0],
                    Lambda_,
                    Ndraw,
                    rate_model,
                    mass_model,
                    spin_model_name,
                    is_GP_dL,
                    smoothing=smoothing,
                    has_m2_break=has_m2_break,
                    log_ddL_dz_inj=log_ddL_dz_inj,
                    zinj=zinj,
                    dcinj=dc_inj,
                    dL_grid=dL_grid_inj,
                    z_grid=z_grid_inj,
                    dc_grid=dc_grid_inj,
                    log_ddL_dz_grid=log_ddL_dz_grid_inj,
                    invert_dL_GP=invert_dL_GP,
                    chunk_size=chunk_inj,
                )



                if True:
    
                        if not (is_GP_dL and invert_dL_GP):
                            raise NotImplementedError(
                                "debug_sel_batch is only set up here for is_GP_dL=True and invert_dL_GP=True"
                            )
    
                        # Reference per-injection quantities using the same interpolation objects
                        zinj_tmp_ = atools.atinterp(dLinj[0], dLGrid_at, zgrid_fine_)
                        dcinj_tmp_ = atools.dcfun_at(zinj_tmp_, H0_, Om_, w0_, interp=False)
                        log_ddL_dz_inj_tmp_ = atools.atinterp(zinj_tmp_, zgrid_fine_, log_ddL_dz_grid)
    
                        log_mu_1, Neff_1, var_ll_u_1 = sel_bias_with_uncertainty_at(
                            m1inj[0],
                            m2inj[0],
                            dLinj[0],
                            spinsInj,
                            lpdinj[0],
                            Lambda_,
                            Ndraw,
                            rate_model,
                            mass_model,
                            spin_model_name,
                            is_GP_dL,
                            smoothing=smoothing,
                            has_m2_break=has_m2_break,
                            log_ddL_dz_inj=log_ddL_dz_inj_tmp_,
                            zinj=zinj_tmp_,
                            dcinj=dcinj_tmp_,
                        )
    
                        # print("Difference in log_mu_1 :")
                        # print((log_mu_1 - log_mu_).eval())
    
                        # print("Difference in Neff_1 :")
                        # print((Neff_1 - Neff_).eval())
    
                        # print("Difference in var_ll_u_1 :")
                        # print((var_ll_u_1 - var_ll_u_).eval())

                        debug_fn = pytensor.function(
                            [],
                            [log_mu_1 - log_mu_, Neff_1 - Neff_, var_ll_u_1 - var_ll_u_],
                            on_unused_input="ignore",
                        )
                        dlogmu, dNeff, dvar = debug_fn()
    
                        print("Difference in log_mu_1 :")
                        print(dlogmu)
    
                        print("Difference in Neff_1 :")
                        print(dNeff)
    
                        print("Difference in var_ll_u_1 :")
                        print(dvar)


                
                
                if not marginal_R0:
                    # This is really the number of expected events 
                    sel_effect = -R0*Ttot*at.exp(log_mu_)
                else:
                    sel_effect = -N*log_mu_
    
            else:
               raise NotImplementedError()

            
            ################################################
            # Sel effect computed. Now exclude high-variance regions in the integral

            
            #Neff = pm.Deterministic('Neff', Neff_ )

            if marginal_R0:
                log_lik_var_selb_ =  at.exp(var_ll_u_+2*at.log(N)) 
            else:
                log_lik_var_selb_ = 'log_lik_var', at.exp(  var_ll_u_+2*at.log( R0*Ttot ) + 2*log_mu_ ) 


            if pop_only:
                log_lik_var_ = log_lik_var_selb_ + var_log_lik_evs
                print("Log lik. variance will include contribution from individual event integrals")
            else:
                log_lik_var_ = log_lik_var_selb_ 
                print("Log lik. variance will be just from selection effect.")


            log_lik_var_sg = log_lik_var_
            log_lik_var_selb_sg = log_lik_var_selb_
            
            # Track log lik. variance 
            log_lik_var_save = pm.Deterministic('log_lik_var', log_lik_var_sg )
            log_lik_var_selb_save = pm.Deterministic('log_lik_var_selb', log_lik_var_selb_sg )
     

            if ((Neff_min==0) and (log_lik_var_min==0)):
                print("No condition on number of effective points in MC integral for sel. effect")
                selection_bias =  sel_effect #pm.Deterministic("sel_bias", sel_effect )
            else:
                if log_lik_var_min==0:

                    selection_bias =  sel_effect

                    # Thresholding on N_eff
                    print("MC integral for sel. effect thresholded on N_eff")
                    
                    Neff = N**2 / ( log_lik_var_selb_sg + (N**2)/Ndraw )

                    #raise NotImplementedError()
                    _ = pm.Potential("bound_selb_Neff", at.switch(Neff >= Neff_min*N, 0.0, -np.inf ))

                    print("Bound on effective number of samples for selection effect. Min requested: %s x Nobs"%Neff_min)

                
                elif Neff_min==0:

                    # Thresholding on likelihood variance
                    print("MC integral for sel. effect thresholded on log lik. variance. Max requested: %s"%log_lik_var_min)

                    
                    if sel_smoothing=='sigmoid':
                        # smooth with sigmoid 
                        print("Tapering sel effect with sigmoid smoothing")
                        
                        selection_bias = sel_effect + atools.logdiffexp( at.log(1), atools.log_sigmoid(log_lik_var_sg, log_lik_var_min*(1+0.002), 0.001 )) 

                    
                    elif sel_smoothing=='poly':
                        print("Tapering sel effect with polynomial smoothing")

                        selection_bias = sel_effect
                        _ = pm.Potential("bound_log_lik_var", atools.logS_PLP(log_lik_var_min - log_lik_var_sg, deltam=0.01, ml=-0.01))

                    
                    else:
                        print("Tapering sel effect with hard cut")

                        selection_bias = sel_effect
                                                
                        _ = pm.Potential("bound_log_lik_var", at.switch(log_lik_var_sg <= log_lik_var_min, 0.0, -1e30 ))


            _ = pm.Potential('selection_bias', selection_bias)

            
            if marginal_R0:
                if include_sel_uncertainty:
                    print("Including selection function uncertainty as in Farr 2019s")
                    # from Farr 2019
                    sel_uncertainty = (3*N+N**2)/(2*Neff)
                    
                    sel_uncertainty_term = pm.Potential('selection_uncertainty', sel_uncertainty)
            

    return model, zgrid_

