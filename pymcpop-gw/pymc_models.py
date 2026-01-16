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

eps   = 1e-30
tinyL = 1e-300
NEG_BIG = -np.inf


#####################################################
#####################################################

# P_POP

#####################################################
#####################################################

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
                 verbose=False,
                 #K=None
                ):


    # work_dtype = getattr(m1s, "dtype", "float64")
    # print("work_dtype in log_p_pop call input is %s"%work_dtype)
    
    ###################################
    # get parameters and compute log p_pop
    ####################################

    # if 'BNS' not in mass_model:
    #     in_support = (m1s >= 3.0) & (m2s >= 3.0) & (m2s <= m1s)
    
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
                                                     -1., 1., 
                                                     0., 1.
                                                    )
        istart_spin = istart + 5

    elif spin_model=='chieffchip_uc':
        
        #muE, sigE, muP, sigP = Lambda[istart],Lambda[istart+1], Lambda[istart+2], Lambda[istart+3] # was: Lambda[istart:istart+4]
        muE   = Lambda[istart + 0]
        sigE  = Lambda[istart + 1]
        muP   = Lambda[istart + 2]
        sigP  = Lambda[istart + 3]
        chieff, chip = spins[0], spins[1]

        lpchie = atools.truncGausslowerupper_at_lpdf(chieff, muE, sigE, xmin=-1., xmax=1.)
        lpchip = atools.truncGausslowerupper_at_lpdf(chip, muP, sigP, xmin=0., xmax=1.)

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
        
        if interp_vals_mass is not None:
            # Use precomputed (m1, m2) grid for PLPreg
            lpmass = atools.logpdf_PLPreg_from_interp(
                [m1s, m2s],
                interp_vals_mass,
                interp_grids_mass,
            )
        else:
            # Direct evaluation (no interpolation)
            lpmass = atools.logpdf_PLP_reg(
                [m1s, m2s],
                [lp, al, bb, dm, ml, mh, muM, sM],
                smoothing=smoothing,
            )


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
        #lambda0_inf = Lambda[j + 21]; z_lambda0 = Lambda[j + 22]; dz_lambda0 = Lambda[j + 23]
        #lambda1_inf = Lambda[j + 24]; z_lambda1 = Lambda[j + 25]; dz_lambda1 = Lambda[j + 26]
        lambda0_inf = Lambda[j + 21]
        lambda1_inf = Lambda[j + 22]
        z_lambda    = Lambda[j + 23]
        dz_lambda   = Lambda[j + 24]
    
        # evo_params = [
        #     alpha1_inf,  z_alpha1,  dz_alpha1,
        #     alpha2_inf,  z_alpha2,  dz_alpha2,
        #     mb_inf,      z_mb,      dz_mb,
        #     mu1_inf,     z_mu1,     dz_mu1,
        #     sigma1_inf,  z_sigma1,  dz_sigma1,
        #     mu2_inf,     z_mu2,     dz_mu2,
        #     sigma2_inf,  z_sigma2,  dz_sigma2,
        #     lambda0_inf, z_lambda0, dz_lambda0,
        #     lambda1_inf, z_lambda1, dz_lambda1,
        # ]
        evo_params = [
                alpha1_inf,  z_alpha1,  dz_alpha1,
                alpha2_inf,  z_alpha2,  dz_alpha2,
                mb_inf,      z_mb,      dz_mb,
                mu1_inf,     z_mu1,     dz_mu1,
                sigma1_inf,  z_sigma1,  dz_sigma1,
                mu2_inf,     z_mu2,     dz_mu2,
                sigma2_inf,  z_sigma2,  dz_sigma2,
                lambda0_inf, lambda1_inf, z_lambda, dz_lambda,
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
            logp1, logp2, logp3 = atools.gaussian_logpdf_pair_from_interp( [m1s, m2s], interp_vals_mass, 
                                                                           interp_grids_mass, 
                                                                          # K=K, 
                                                                           z = z_dpuc )
    
        
        if rate_model in ('PL', 'MD'):
            logp_components = logp1 + logp2                    # (K,N)
        else:
            logp_components = logp1 + logp2 + logp3                   # (K,N)
            
        
        # Mixture over components → (n_obs,)
        lpmass = at.logsumexp(logp_components + logw[:, None], axis=0, )

        #lpmass = at.logsumexp(logp_components + logw[:, None] , axis=0) # + logw[:, None]

        
        if rate_model=='DPUC-vol' and is_observed:
            print("Normalize GMM x p(z)")
            log_Nz = atools.redshift_mixture_log_norm( mu=mu, sd=sd, logw=logw, y_min = at.log1p(at.min(z_grid)), y_max=at.log1p(at.max(z_grid) ),  H0=H0, Om=Om, w0=w0, Ny=2000 )
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
            - 0.5 * nd * at.log(2.0 * np.pi)
            + 0.5 * ldets_inv[:, None]
            + logw[:, None]
        )                                           # (K, N)

        # # 2a) Solve L * y = diff^T  for each component k
        # #    diff.transpose -> (K, 2, N); solve_lower_triangular acts per-k
        # y = at.solve_lower_triangular(L, diff.transpose(0, 2, 1))  # (K, 2, N)
        
        # # 3) Mahalanobis term: ||y||^2  → (K, N)
        # quad = at.sum(y**2, axis=1).T  # sum over the 2 dims, then transpose to (K, N)
        
        # # 3a) log |Σ^{-1}|  from L:  log|Σ| = 2 * sum(log(diag(L)))  ⇒ log|Σ^{-1}| = -2 * ...
        # logdet_prec = -2.0 * at.sum(at.log(at.diagonal(L, axis1=1, axis2=2)), axis=1)  # (K,)
        
        # # 4) Component log-densities (d=2)
        # logp_components = (
        #     -0.5 * quad
        #     - 0.5 * 2 * at.log(2.0 * np.pi)
        #     + 0.5 * logdet_prec[:, None]
        #     + logw[:, None]
        # )  # (K, N)

        # 5) Mixture over components -> per-observation log-lik
        lpmass = at.logsumexp(logp_components, axis=0, )  # (N,)

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


    return lp 



#####################################################
#####################################################

# SEL BIAS

#####################################################
#####################################################



def sel_bias_with_uncertainty_at_0(m1inj, m2inj, dLinj, spinsInj, log_p_draw, 
                                    Lambda,  Ndraw, 
                                    rate_model, mass_model, spin_model, 
                                    smoothing, 
                                   simplex_repair,
                                    has_m2_break, 
                                    interp, 
                                   log_p_incl = None,
                                    log_ddL_dz_inj = None,
                                    zinj = None,
                                    dcinj = None,
                                   param='vanilla',
                                   interp_vals_mass = None,
                                    interp_grids_mass = None,
                                   verbose=False,
                                    **kwargs):


    #H0, Om, w0, Xi0, n  = Lambda[:5]
    H0  = Lambda[0]
    Om  = Lambda[1]
    w0  = Lambda[2]
    Xi0 = Lambda[3]
    n   = Lambda[4]


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
        #log_Mc_src_inj = at.log(Mc_src_inj)
        log_Mc_src_inj = at.log(at.maximum(Mc_src_inj, eps))
        logit_q_inj = atools.logitat(q_inj)      
        mass_1_use = log_Mc_src_inj
        mass_2_use = logit_q_inj
    else:
        mass_1_use = m1Src
        mass_2_use = m2Src

    log_p_pop = log_p_pop_at(mass_1_use, mass_2_use, zinj, dLinj, spinsInj_sel, 
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
        log_p_pop += (- at.log(m2Src) 
                      - at.log(at.maximum(m1Src - m2Src, eps))) #at.log(m1Src-m2Src) 
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
    Ndraw_t = Ndraw
    
    log_mu = at.logsumexp(log_sel_b, ) - at.log(Ndraw_t)
    
    logs2 = at.logsumexp(2.0*log_sel_b, ) - at.log(Ndraw_t)


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

    logNeff = 2*log_mu - logs2 + at.log(Ndraw_t)

    #####################################
    # This is variance of log l per unit obs as in Talbot Golomb 2023
    #####################################

    var_log_lik_u = atools.logdiffexp( logs2-2*log_mu, 1.) - at.log(Ndraw_t-1.)

    Neff = at.exp(logNeff)
    
    
    return log_mu, Neff, var_log_lik_u



#####################################################
#####################################################

def sel_bias_with_uncertainty_at_0_batched_scan(
    m1inj, m2inj, dLinj, spinsInj, log_p_draw,
    Lambda, Ndraw,
    rate_model, mass_model, spin_model,
    smoothing,
    simplex_repair,
    has_m2_break,
    interp,
    log_p_incl=None,
    # kept for API compat (ignored if dL_grid / z_grid are provided)
    log_ddL_dz_inj=None,
    zinj=None,
    dcinj=None,
    # grids ONLY for dL->z (and optionally dc, log_ddL_dz as functions of dL)
    dL_grid=None,               # 1-D, increasing in dL
    z_grid=None,                # 1-D, z(dL_grid)
    dc_grid=None,               # 1-D, dc(dL_grid) or dc(z_grid)
    log_ddL_dz_grid=None,       # 1-D, log ddL/dz (dL_grid or z_grid)
    *,
    chunk_size=4096,
    param='vanilla',
    interp_vals_mass=None,
    interp_grids_mass=None,
    verbose=False,
    **kwargs
):
    """Scan/batched version of `sel_bias_with_uncertainty_at_0`."""

    #work_dtype = getattr(m1inj, "dtype", "float64")
    #print("work_dtype in sel_bias_with_uncertainty_at_0_batched_scan is %s"%work_dtype)
    
    #def _as_at(x):
    #    return x if isinstance(x, at.Variable) else at.as_tensor_variable(x)

    def _pad_to_multiple(x, k, pad_value):
        #x = _as_at(x)
        if x.ndim != 1:
            x = at.flatten(x, 1)
        N = x.shape[0]
        C = (N + k - 1) // k
        Npad = C * k - N
        pad = at.full(
            (Npad,),
            pad_value, #at.as_tensor_variable(pad_value, dtype=x.dtype),
            #dtype=x.dtype,
        )
        xpad = at.concatenate([x, pad], axis=0)
        return xpad.reshape((C, k)), C, N

    def _combine_logsumexp(m_s, s_s, m_c, s_c):
        """Combine two log-sum-exp accumulators in a numerically stable way."""
        m_new = at.maximum(m_s, m_c)
        s_new = s_s * at.exp(m_s - m_new) + s_c * at.exp(m_c - m_new)
        return m_new, s_new

    # base tensors
    #m1_all   = _as_at(m1inj)
    #m2_all   = _as_at(m2inj)
    #dL_all   = _as_at(dLinj)
    #lpd_all  = _as_at(log_p_draw)

    m1_all   = m1inj
    m2_all   = m2inj
    dL_all   = dLinj
    lpd_all  = log_p_draw

    # Lambda is a *sequence* (can be heterogeneous for DPUC etc.)
    Lambda_seq = list(Lambda)
    n_Lambda   = len(Lambda_seq)

    # cosmology parameters are the first 5 entries
    H0, Om, w0, Xi0, n = (
        Lambda_seq[0],
        Lambda_seq[1],
        Lambda_seq[2],
        Lambda_seq[3],
        Lambda_seq[4],
    )

 
    # spins
    spin_is_default = (spin_model in ("default", "default_gauss"))
    #if spin_is_default:
    #    s1_all  = _as_at(spinsInj[0])
    #    s2_all  = _as_at(spinsInj[1])
    #    ct1_all = _as_at(spinsInj[2])
    #    ct2_all = _as_at(spinsInj[3])

    #work_dtype = getattr(m1_all, "dtype", "float64")
    #int_dtype  = "int32" if work_dtype in ("float16", "float32") else "int64"
    
    K = int(chunk_size)

    # pad & mask
    m1K, C, N = _pad_to_multiple(m1_all,   K, 2.0)
    m2K, _, _ = _pad_to_multiple(m2_all,   K, 1.0)
    dLK, _, _ = _pad_to_multiple(dL_all,   K, 1.0)
    lpdK, _, _ = _pad_to_multiple(lpd_all, K, 0.0)
    if spin_is_default:
        s1K,  _, _ = _pad_to_multiple(s1_all,  K, 0.0)
        s2K,  _, _ = _pad_to_multiple(s2_all,  K, 0.0)
        ct1K, _, _ = _pad_to_multiple(ct1_all, K, 1.0)
        ct2K, _, _ = _pad_to_multiple(ct2_all, K, 1.0)

    # optional per-injection inclination prior
    have_log_p_incl = log_p_incl is not None
    if have_log_p_incl:
        #log_p_incl_all = _as_at(log_p_incl)
        lpicK, _, _ = _pad_to_multiple(log_p_incl_all, K, 0.0)
    else:
        lpicK = None

    idxs = at.arange(C)
    valid_mask = (at.arange(C * K) < N).reshape((C, K))
    
     #at.as_tensor_variable(-np.inf, dtype=work_dtype) 
    
    # if work_dtype == "float32":
    #     eps = at.as_tensor_variable(1e-20, dtype=work_dtype)
    # else:
    #     eps = at.as_tensor_variable(1e-30, dtype=work_dtype)
        
    # z(dL) / dc / log_ddL_dz via grids (optional)
    have_dLz = (dL_grid is not None) and (z_grid is not None)
    have_dc_grid = (dc_grid is not None)
    have_logdd_grid = (log_ddL_dz_grid is not None)

    if have_dLz:
        dL_grid_t = dL_grid #_as_at(dL_grid)
        z_grid_t  = z_grid #_as_at(z_grid)
        #dc_grid_t = _as_at(dc_grid) if have_dc_grid else None
        #logdd_grid_t = _as_at(log_ddL_dz_grid) if have_logdd_grid else None

        dc_grid_t = dc_grid if have_dc_grid else None
        logdd_grid_t = log_ddL_dz_grid if have_logdd_grid else None
        

    # mass interpolation (precomputed grids)
    have_interp_mass = (interp_vals_mass is not None) and (interp_grids_mass is not None)
    if have_interp_mass:
        interp_vals_mass_seq  = list(interp_vals_mass)
        interp_grids_mass_seq = list(interp_grids_mass)
        n_interp_vals_mass    = len(interp_vals_mass_seq)
        n_interp_grids_mass   = len(interp_grids_mass_seq)
    else:
        interp_vals_mass_seq  = []
        interp_grids_mass_seq = []
        n_interp_vals_mass    = 0
        n_interp_grids_mass   = 0

    # ---- scan body ----
    if spin_is_default:
        def step(i, m_state, m2_state, s1_state, s2_state,
                 m1K, m2K, dLK, lpdK, valid_mask, *extra):

            idx_extra = 0

            # 1) Lambda components
            Lambda_flat = extra[idx_extra:idx_extra + n_Lambda]
            Lambda_local = Lambda_flat
            idx_extra += n_Lambda

            # 2) spin grids
            s1K_local  = extra[idx_extra]
            s2K_local  = extra[idx_extra + 1]
            ct1K_local = extra[idx_extra + 2]
            ct2K_local = extra[idx_extra + 3]
            idx_extra += 4

            # 3) optional log_p_incl
            if have_log_p_incl:
                lpicK_local = extra[idx_extra]
                idx_extra += 1
            else:
                lpicK_local = None

            # 4) optional cosmology grids
            dL_grid_t_local = None
            z_grid_t_local = None
            dc_grid_t_local = None
            logdd_grid_t_local = None
            if have_dLz:
                dL_grid_t_local = extra[idx_extra]
                z_grid_t_local  = extra[idx_extra + 1]
                idx_extra += 2
                if have_dc_grid:
                    dc_grid_t_local = extra[idx_extra]
                    idx_extra += 1
                if have_logdd_grid:
                    logdd_grid_t_local = extra[idx_extra]
                    idx_extra += 1

            # 5) mass interpolation tables
            interp_vals_local = None
            interp_grids_local = None
            if have_interp_mass:
                iv_start = idx_extra
                iv_end   = iv_start + n_interp_vals_mass
                interp_vals_local = list(extra[iv_start:iv_end])
                idx_extra = iv_end

                ig_start = idx_extra
                ig_end   = ig_start + n_interp_grids_mass
                interp_grids_local = list(extra[ig_start:ig_end])
                idx_extra = ig_end

            # base per-chunk slices
            m1  = m1K[i]
            m2  = m2K[i]
            dL  = dLK[i]
            lpd = lpdK[i]
            mask = valid_mask[i]

            # spin components for this chunk
            s1  = s1K_local[i]
            s2  = s2K_local[i]
            ct1 = ct1K_local[i]
            ct2 = ct2K_local[i]
            spins_use = [s1, s2, ct1, ct2]

            # z, dc, log_ddL_dz
            if have_dLz:
                idxs_loc, r = atools._interp_indices_nonuniform(dL, dL_grid_t_local)
                il = idxs_loc - 1
                ih = idxs_loc

                zl = z_grid_t_local[il]
                zh = z_grid_t_local[ih]
                zinj_c = (1.0 - r) * zl + r * zh

                if have_dc_grid and (dc_grid_t_local is not None):
                    dcl = dc_grid_t_local[il]
                    dch = dc_grid_t_local[ih]
                    dc_c = (1.0 - r) * dcl + r * dch
                else:
                    dc_c = atools.dcfun_at(zinj_c, H0, Om, w0, interp=interp)

                if have_logdd_grid and (logdd_grid_t_local is not None):
                    ll = logdd_grid_t_local[il]
                    lh = logdd_grid_t_local[ih]
                    logdd_c = (1.0 - r) * ll + r * lh
                else:
                    logdd_c = atools.log_ddL_dz(
                        zinj_c, H0, Om, w0, Xi0, n,
                        dc=dc_c, interp=interp, param=param
                    )
            else:
                zinj_c = atools.z_from_dL_at(
                    dL, H0, Om, w0, Xi0, n, interp=interp, param=param
                )
                dc_c   = atools.dcfun_at(zinj_c, H0, Om, w0, interp=interp)
                logdd_c = atools.log_ddL_dz(
                    zinj_c, H0, Om, w0, Xi0, n,
                    dc=dc_c, interp=interp, param=param
                )

            one_p_z = 1.0 + zinj_c
            m1Src = m1 / one_p_z
            m2Src = m2 / one_p_z

            use_dp = (mass_model in ("DP", "DPUC"))
            if use_dp:
                Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
                mass_1_use = at.log(at.maximum(Mc_src_inj, eps))
                mass_2_use = atools.logitat(q_inj)
            else:
                mass_1_use = m1Src
                mass_2_use = m2Src

            # choose interp tables or None
            if have_interp_mass:
                interp_vals_arg  = interp_vals_local
                interp_grids_arg = interp_grids_local
            else:
                interp_vals_arg  = None
                interp_grids_arg = None

            lp = log_p_pop_at(
                mass_1_use, mass_2_use, zinj_c, dL, spins_use, Lambda_local,
                rate_model, mass_model, spin_model,
                smoothing=smoothing,
                simplex_repair=simplex_repair,
                has_m2_break=has_m2_break,
                log_ddL_dz_pre=logdd_c,
                dc=dc_c,
                interp_vals_mass=interp_vals_arg,
                interp_grids_mass=interp_grids_arg,
                verbose=verbose,
            )

            if use_dp:
                lp = (
                    lp
                    - at.log(at.maximum(m2Src, eps))
                    - at.log(at.maximum(m1Src - m2Src, eps))
                )
                if rate_model in ("DPUC", "DPUC-vol"):
                    lp = lp - at.log1p(zinj_c)

            if have_log_p_incl:
                lpic = lpicK_local[i]
                x = at.where(mask, lp - lpd - lpic, NEG_BIG)
            else:
                x = at.where(mask, lp - lpd,        NEG_BIG)

            m = at.max(x)
            y = at.exp(x - m)
            s1c = at.sum(y)
            s2c = at.sum(at.sqr(y))

            m_new,  s1_new = _combine_logsumexp(m_state,  s1_state,  m,  s1c)
            m2c = 2.0 * m
            m2_new, s2_new = _combine_logsumexp(m2_state, s2_state, m2c, s2c)
            return m_new, m2_new, s1_new, s2_new

    else:
        def step(i, m_state, m2_state, s1_state, s2_state,
                 m1K, m2K, dLK, lpdK, valid_mask, *extra):

            idx_extra = 0

            # 1) Lambda components
            Lambda_flat = extra[idx_extra:idx_extra + n_Lambda]
            Lambda_local = Lambda_flat
            idx_extra += n_Lambda

            # 2) no spins in this branch

            # 3) optional log_p_incl
            if have_log_p_incl:
                lpicK_local = extra[idx_extra]
                idx_extra += 1
            else:
                lpicK_local = None

            # 4) optional cosmology grids
            dL_grid_t_local = None
            z_grid_t_local = None
            dc_grid_t_local = None
            logdd_grid_t_local = None
            if have_dLz:
                dL_grid_t_local = extra[idx_extra]
                z_grid_t_local  = extra[idx_extra + 1]
                idx_extra += 2
                if have_dc_grid:
                    dc_grid_t_local = extra[idx_extra]
                    idx_extra += 1
                if have_logdd_grid:
                    logdd_grid_t_local = extra[idx_extra]
                    idx_extra += 1

            # 5) mass interpolation tables
            interp_vals_local = None
            interp_grids_local = None
            if have_interp_mass:
                iv_start = idx_extra
                iv_end   = iv_start + n_interp_vals_mass
                interp_vals_local = list(extra[iv_start:iv_end])
                idx_extra = iv_end

                ig_start = idx_extra
                ig_end   = ig_start + n_interp_grids_mass
                interp_grids_local = list(extra[ig_start:ig_end])
                idx_extra = ig_end

            # base per-chunk slices
            m1  = m1K[i]
            m2  = m2K[i]
            dL  = dLK[i]
            lpd = lpdK[i]
            mask = valid_mask[i]
            spins_use = []

            # z, dc, log_ddL_dz
            if have_dLz:
                idxs_loc, r = atools._interp_indices_nonuniform(dL, dL_grid_t_local)
                il = idxs_loc - 1
                ih = idxs_loc

                zl = z_grid_t_local[il]
                zh = z_grid_t_local[ih]
                zinj_c = (1.0 - r) * zl + r * zh

                if have_dc_grid and (dc_grid_t_local is not None):
                    dcl = dc_grid_t_local[il]
                    dch = dc_grid_t_local[ih]
                    dc_c = (1.0 - r) * dcl + r * dch
                else:
                    dc_c = atools.dcfun_at(zinj_c, H0, Om, w0, interp=interp)

                if have_logdd_grid and (logdd_grid_t_local is not None):
                    ll = logdd_grid_t_local[il]
                    lh = logdd_grid_t_local[ih]
                    logdd_c = (1.0 - r) * ll + r * lh
                else:
                    logdd_c = atools.log_ddL_dz(
                        zinj_c, H0, Om, w0, Xi0, n,
                        dc=dc_c, interp=interp, param=param
                    )
            else:
                zinj_c = atools.z_from_dL_at(
                    dL, H0, Om, w0, Xi0, n, interp=interp, param=param
                )
                dc_c   = atools.dcfun_at(zinj_c, H0, Om, w0, interp=interp)
                logdd_c = atools.log_ddL_dz(
                    zinj_c, H0, Om, w0, Xi0, n,
                    dc=dc_c, interp=interp, param=param
                )

            one_p_z = 1.0 + zinj_c
            m1Src = m1 / one_p_z
            m2Src = m2 / one_p_z

            use_dp = (mass_model in ("DP", "DPUC"))
            if use_dp:
                Mc_src_inj, q_inj = atools.Mcq_from_m1m2_at(m1Src, m2Src)
                mass_1_use = at.log(at.maximum(Mc_src_inj, eps))
                mass_2_use = atools.logitat(q_inj)
            else:
                mass_1_use = m1Src
                mass_2_use = m2Src

            # choose interp tables or None
            if have_interp_mass:
                interp_vals_arg  = interp_vals_local
                interp_grids_arg = interp_grids_local
            else:
                interp_vals_arg  = None
                interp_grids_arg = None

            lp = log_p_pop_at(
                mass_1_use, mass_2_use, zinj_c, dL, spins_use, Lambda_local,
                rate_model, mass_model, spin_model,
                smoothing=smoothing,
                simplex_repair=simplex_repair,
                has_m2_break=has_m2_break,
                log_ddL_dz_pre=logdd_c,
                dc=dc_c,
                interp_vals_mass=interp_vals_arg,
                interp_grids_mass=interp_grids_arg,
                verbose=verbose,
            )

            if use_dp:
                lp = (
                    lp
                    - at.log(at.maximum(m2Src, eps))
                    - at.log(at.maximum(m1Src - m2Src, eps))
                )
                if rate_model in ("DPUC", "DPUC-vol"):
                    lp = lp - at.log1p(zinj_c)

            if have_log_p_incl:
                lpic = lpicK_local[i]
                x = at.where(mask, lp - lpd - lpic, NEG_BIG)
            else:
                x = at.where(mask, lp - lpd,        NEG_BIG)

            m = at.max(x)
            y = at.exp(x - m)
            s1c = at.sum(y)
            s2c = at.sum(at.sqr(y))

            m_new,  s1_new = _combine_logsumexp(m_state,  s1_state,  m,  s1c)
            m2c = 2.0 * m
            m2_new, s2_new = _combine_logsumexp(m2_state, s2_state, m2c, s2c)
            return m_new, m2_new, s1_new, s2_new

    # non_sequences
    m_init = at.as_tensor_variable(-np.inf, dtype="float64")
    s_init = at.as_tensor_variable(0.0,     dtype="float64")

    nonseq = [m1K, m2K, dLK, lpdK, valid_mask]
    # unpack Lambda components (critical for DPUC)
    nonseq += Lambda_seq

    if spin_is_default:
        nonseq += [s1K, s2K, ct1K, ct2K]
    if have_log_p_incl:
        nonseq += [lpicK]
    if have_dLz:
        nonseq += [dL_grid_t, z_grid_t]
        if have_dc_grid:
            nonseq += [dc_grid_t]
        if have_logdd_grid:
            nonseq += [logdd_grid_t]
    if have_interp_mass:
        nonseq += interp_vals_mass_seq
        nonseq += interp_grids_mass_seq

    (m_fin, m2_fin, s1_fin, s2_fin), _ = pytensor.scan(
        fn=step,
        sequences=[idxs],
        outputs_info=[m_init, m_init, s_init, s_init],
        non_sequences=nonseq,
        strict=True,
        profile=True
    )

    # if work_dtype == "float32":
    #     # reasonable tiny values in float32
    #     eps   = at.as_tensor_variable(1e-20, dtype=work_dtype)
    #     tinyL = at.as_tensor_variable(1e-30, dtype=work_dtype)
    # else:
    #     eps   = at.as_tensor_variable(1e-30,  dtype=work_dtype)
    #     tinyL = at.as_tensor_variable(1e-300, dtype=work_dtype)
        
    logsumexp1 = m_fin[-1]  + at.log(at.maximum(s1_fin[-1], tinyL))
    logsumexp2 = m2_fin[-1] + at.log(at.maximum(s2_fin[-1], tinyL))

    Ndraw_t = Ndraw #at.as_tensor_variable(Ndraw)#.astype(work_dtype)
    log_mu  = logsumexp1 - at.log(Ndraw_t)
    logs2   = logsumexp2 - at.log(Ndraw_t)
    logNeff = 2.0 * log_mu - logs2 + at.log(Ndraw_t)
    Neff    = at.exp(logNeff)
    var_log_lik_u = atools.logdiffexp(logs2 - 2.0 * log_mu, 1.0) - at.log(Ndraw_t - 1.0)

    return log_mu, Neff, var_log_lik_u


#####################################################
# GPU

def sel_bias_with_uncertainty_at_0_batched_scan_GPU(
    m1inj, m2inj, dLinj, spinsInj, log_p_draw,
    Lambda, Ndraw,
    rate_model, mass_model, spin_model,
    smoothing,
    simplex_repair,
    has_m2_break,
    interp,
    log_p_incl=None,
    # kept for API compat (ignored if dL_grid / z_grid are provided)
    log_ddL_dz_inj=None,
    zinj=None,
    dcinj=None,
    # grids ONLY for dL->z (and optionally dc, log_ddL_dz as functions of dL)
    dL_grid=None,               # 1-D, increasing in dL
    z_grid=None,                # 1-D, z(dL_grid)
    dc_grid=None,               # 1-D, dc(dL_grid) or dc(z_grid)
    log_ddL_dz_grid=None,       # 1-D, log ddL/dz (dL_grid or z_grid)
    *,
    chunk_size=4096,
    param='vanilla',
    interp_vals_mass=None,
    interp_grids_mass=None,
    verbose=False,
    # OOM-safe default: do NOT hoist (C,K) float64 arrays outside scan
    hoist_interp_outputs=False,
    **kwargs
):
    import numpy as np
    import pytensor
    import pytensor.tensor as at

    # -----------------------------
    # PyMC-safe "RV -> value var"
    # -----------------------------
    def _replace_rvs_by_values_compat(x):
        try:
            import pymc as pm
            model = pm.modelcontext(None)
            return model.replace_rvs_by_values([x])[0]
        except Exception:
            return x

    # def _as_at(x):
    #     return x if isinstance(x, at.Variable) else at.as_tensor_variable(x)

    def _val(x):
        return _replace_rvs_by_values_compat(x) #_as_at(x))

    # stop_grad compat
    try:
        from pytensor.gradient import grad_not_implemented, disconnected_grad, stop_gradient as stop_grad
    except Exception:
        try:
            from pytensor.gradient import stop_gradient as stop_grad
        except Exception:
            # very old alias
            from pytensor.gradient import disconnected_grad as stop_grad

    def _pad_to_multiple_1d(x, k, pad_value):
        x = _val(x)
        if x.ndim != 1:
            x = at.flatten(x, 1)
        N = x.shape[0]
        C = (N + k - 1) // k
        Npad = C * k - N
        pad = at.full(
            (Npad,),
            pad_value, #at.as_tensor_variable(pad_value, dtype=x.dtype),
            #dtype=x.dtype,
        )
        xpad = at.concatenate([x, pad], axis=0)
        return xpad.reshape((C, k)), C, N

    def _combine_logsumexp(m_s, s_s, m_c, s_c):
        m_new = at.maximum(m_s, m_c)
        s_new = s_s * at.exp(m_s - m_new) + s_c * at.exp(m_c - m_new)
        return m_new, s_new

    # -----------------------------
    # Flags / constants
    # -----------------------------
    spin_is_default  = (spin_model in ("default", "default_gauss"))
    use_dp           = (mass_model in ("DP", "DPUC"))
    have_dLz         = (dL_grid is not None) and (z_grid is not None)
    have_dc_grid     = (dc_grid is not None)
    have_logdd_grid  = (log_ddL_dz_grid is not None)
    have_interp_mass = (interp_vals_mass is not None) and (interp_grids_mass is not None)

    # -----------------------------
    # Base tensors (keep big arrays in their native dtype)
    # -----------------------------
    m1_all  = _val(m1inj)
    m2_all  = _val(m2inj)
    dL_all  = _val(dLinj)
    lpd_all = _val(log_p_draw)

    #work_dtype = getattr(m1_all, "dtype", "float64")

    # IMPORTANT: OOM-safe mixed precision
    # - big arrays stay in work_dtype (often float32)
    # - all chunk-local compute + accumulators run in compute_dtype
    
    # compute_dtype = "float64" if str(work_dtype) == "float32" else work_dtype
    # accum_dtype = compute_dtype

    K = int(chunk_size)

    # eps/tiny in compute dtype (numerical safety for logs)
    # if str(compute_dtype) == "float32":
    #     eps   = at.as_tensor_variable(1e-20, dtype=compute_dtype)
    #     tinyL = at.as_tensor_variable(1e-30, dtype=compute_dtype)
    # else:
    #     eps   = at.as_tensor_variable(1e-30,  dtype=compute_dtype)
    #     tinyL = at.as_tensor_variable(1e-300, dtype=compute_dtype)

    #neg_inf = at.as_tensor_variable(-np.inf, dtype=compute_dtype)


    # -----------------------------
    # Lambda / interp tables as VALUE vars
    # (these are small; dtype will be handled in step)
    # -----------------------------
    Lambda_seq = [_val(v) for v in list(Lambda)]
    n_Lambda = len(Lambda_seq)

    if have_interp_mass:
        interp_vals_arg  = [_val(v) for v in list(interp_vals_mass)]
        interp_grids_arg = [_val(v) for v in list(interp_grids_mass)]
        n_iv = len(interp_vals_arg)
        n_ig = len(interp_grids_arg)
    else:
        interp_vals_arg = None
        interp_grids_arg = None
        n_iv = 0
        n_ig = 0

    # -----------------------------
    # Pad & mask (C chunks of length K) — remains work_dtype, not upcasted
    # -----------------------------
    m1K, C, N  = _pad_to_multiple_1d(m1_all,  K, 2.0)
    m2K, _, _  = _pad_to_multiple_1d(m2_all,  K, 1.0)
    dLK, _, _  = _pad_to_multiple_1d(dL_all,  K, 1.0)
    lpdK, _, _ = _pad_to_multiple_1d(lpd_all, K, 0.0)

    if log_p_incl is not None:
        lpic_all = _val(log_p_incl)
        lpicK, _, _ = _pad_to_multiple_1d(lpic_all, K, 0.0)
    else:
        # keep same dtype as lpdK (work_dtype)
        lpicK = at.zeros_like(lpdK)

    if spin_is_default:
        s1K,  _, _ = _pad_to_multiple_1d(_val(spinsInj[0]), K, 0.0)
        s2K,  _, _ = _pad_to_multiple_1d(_val(spinsInj[1]), K, 0.0)
        ct1K, _, _ = _pad_to_multiple_1d(_val(spinsInj[2]), K, 1.0)
        ct2K, _, _ = _pad_to_multiple_1d(_val(spinsInj[3]), K, 1.0)

    #int_dtype = "int32"
    valid_mask = (at.arange(C * K) < N).reshape((C, K))

    # -----------------------------
    # Grids (VALUE vars)
    # -----------------------------
    if have_dLz:
        dL_grid_t = _val(dL_grid)
        z_grid_t  = _val(z_grid)
        dc_grid_t = _val(dc_grid) if have_dc_grid else None
        logdd_grid_t = _val(log_ddL_dz_grid) if have_logdd_grid else None

        # HOIST 1: indices + r out of scan
        # (NOTE: this creates (C,K) idx/r arrays, but idx is int32 and r keeps work_dtype;
        # we do NOT create (C,K) float64 z/dc/logdd when hoist_interp_outputs=False.)
        dL_flat = dLK.reshape((-1,))
        idx_flat, r_flat = atools._interp_indices_nonuniform(dL_flat, dL_grid_t)

        idx_flat = stop_grad(idx_flat)#.astype(int_dtype)
        r_flat   = stop_grad(r_flat)#.astype(work_dtype)

        idxK = idx_flat.reshape((C, K))
        rK   = r_flat.reshape((C, K))

        idxK = at.clip(idxK, 1, dL_grid_t.shape[0] - 1)

        # OOM-risky path (off by default)
        if hoist_interp_outputs:
            ilK = idxK - 1
            ihK = idxK
            # These become (C,K) arrays; keep them in work_dtype to reduce memory
            zK = ((1.0 - rK) * z_grid_t[ilK] + rK * z_grid_t[ihK])#.astype(work_dtype)

            dcK = None
            logddK = None
            if have_dc_grid:
                dcK = ((1.0 - rK) * dc_grid_t[ilK] + rK * dc_grid_t[ihK])#.astype(work_dtype)
            if have_logdd_grid:
                logddK = ((1.0 - rK) * logdd_grid_t[ilK] + rK * logdd_grid_t[ihK])#.astype(work_dtype)
        else:
            zK = dcK = logddK = None

    # -----------------------------
    # Helpers used in step
    # -----------------------------
    def _unpack_tail(tail):
        # tail layout: [Lambda..., interp_vals..., interp_grids...]
        Lambda_flat = tail[:n_Lambda]
        # Cast cosmology params to compute_dtype INSIDE scan
        H0  = Lambda_flat[0]
        Om  = Lambda_flat[1]
        w0  = Lambda_flat[2]
        Xi0 = Lambda_flat[3]
        n_  = Lambda_flat[4]

        if have_interp_mass:
            iv = list(tail[n_Lambda:n_Lambda + n_iv])
            ig = list(tail[n_Lambda + n_iv:n_Lambda + n_iv + n_ig])
        else:
            iv = None
            ig = None

        # Also cast the Lambda vector passed to log_p_pop to compute_dtype (small)
        Lambda_cast = [ v for v in Lambda_flat]
        return Lambda_cast, H0, Om, w0, Xi0, n_, iv, ig

    def _cosmo_direct(dL_chunk, H0, Om, w0, Xi0, n_):
        z = atools.z_from_dL_at(dL_chunk, H0, Om, w0, Xi0, n_, interp=interp, param=param)
        dc = atools.dcfun_at(z, H0, Om, w0, interp=interp)
        logdd = atools.log_ddL_dz(z, H0, Om, w0, Xi0, n_, dc=dc, interp=interp, param=param)
        return z, dc, logdd

    def _mass_inputs(m1, m2, z):
        one_p_z = 1.0 + z #at.as_tensor_variable(1.0, dtype=compute_dtype) + z
        m1Src = m1 / one_p_z
        m2Src = m2 / one_p_z
        if use_dp:
            Mc, q = atools.Mcq_from_m1m2_at(m1Src, m2Src)
            mass_1_use = at.log(at.maximum(Mc, eps))
            mass_2_use = atools.logitat(q)
        else:
            mass_1_use = m1Src
            mass_2_use = m2Src
        return mass_1_use, mass_2_use, m1Src, m2Src

    def _dp_jacobian_fix(lp, m1Src, m2Src, z):
        lp = (
            lp
            - at.log(at.maximum(m2Src, eps))
            - at.log(at.maximum(m1Src - m2Src, eps))
        )
        if rate_model in ("DPUC", "DPUC-vol"):
            lp = lp - at.log1p(z)
        return lp

    # -----------------------------
    # Build scan sequences (PER-CHUNK)
    # -----------------------------
    seqs = [m1K, m2K, dLK, lpdK, lpicK, valid_mask]

    if spin_is_default:
        seqs += [s1K, s2K, ct1K, ct2K]

    if have_dLz:
        if hoist_interp_outputs:
            seqs += [zK]
            if have_dc_grid:
                seqs += [dcK]
            if have_logdd_grid:
                seqs += [logddK]
            if (not have_dc_grid) or (not have_logdd_grid):
                seqs += [idxK, rK]
        else:
            seqs += [idxK, rK]

    # -----------------------------
    # Non-sequences: grids + tail
    # -----------------------------
    nonseq = []
    if have_dLz:
        nonseq += [z_grid_t]
        if have_dc_grid:
            nonseq += [dc_grid_t]
        if have_logdd_grid:
            nonseq += [logdd_grid_t]

    nonseq += Lambda_seq
    if have_interp_mass:
        nonseq += interp_vals_arg
        nonseq += interp_grids_arg

    # -----------------------------
    # Step function (chunk-local float64 compute + float64 accumulators)
    # -----------------------------
    def step(*args):
        pos = 0

        # ---- sequences (chunk vectors)
        m1 = args[pos]; pos += 1
        m2 = args[pos]; pos += 1
        dL = args[pos]; pos += 1
        lpd = args[pos]; pos += 1
        lpic = args[pos]; pos += 1
        mask = args[pos]; pos += 1

        # Cast ONLY the chunk vectors to compute_dtype (this is the key)
        # m1 = at.cast(m1, compute_dtype)
        # m2 = at.cast(m2, compute_dtype)
        # dL = at.cast(dL, compute_dtype)
        # lpd = at.cast(lpd, compute_dtype)
        # lpic = at.cast(lpic, compute_dtype)

        if spin_is_default:
            s1  = args[pos]#,   compute_dtype)
            s2  = args[pos+1]#, compute_dtype)
            ct1 = args[pos+2]#, compute_dtype)
            ct2 = args[pos+3]#, compute_dtype)
            pos += 4
            spins_use = [s1, s2, ct1, ct2]
        else:
            spins_use = []

        # dL->z inputs
        z = dc = logdd = None
        idxs_loc = r = None

        if have_dLz:
            if hoist_interp_outputs:
                z = args[pos]; pos += 1
                if have_dc_grid:
                    dc = args[pos]; pos += 1
                if have_logdd_grid:
                    logdd = args[pos]; pos += 1
                if (not have_dc_grid) or (not have_logdd_grid):
                    idxs_loc = args[pos]
                    r = args[pos+1]
                    pos += 2
            else:
                idxs_loc = args[pos]
                r = args[pos+1]
                pos += 2

        # ---- rolling states (float64 if compute_dtype=float64)
        m_state  = args[pos]; m2_state = args[pos+1]; s1_state = args[pos+2]; s2_state = args[pos+3]
        pos += 4

        # ---- grids
        if have_dLz:
            z_grid_t_local = args[pos]; pos += 1
            dc_grid_t_local = None
            logdd_grid_t_local = None
            if have_dc_grid:
                dc_grid_t_local = args[pos]; pos += 1
            if have_logdd_grid:
                logdd_grid_t_local = args[pos]; pos += 1
        else:
            z_grid_t_local = dc_grid_t_local = logdd_grid_t_local = None

        # ---- tail (Lambda + optional interp mass tables)
        tail = args[pos:]
        Lambda_local, H0, Om, w0, Xi0, n_, iv, ig = _unpack_tail(tail)

        # -----------------------------
        # Compute z/dc/logdd (all in compute_dtype)
        # -----------------------------
        if have_dLz:
            if z is None:
                il = idxs_loc - 1
                ih = idxs_loc
                z = (1.0 - r) * z_grid_t_local[il] + r * z_grid_t_local[ih]

            if dc is None:
                if have_dc_grid:
                    il = idxs_loc - 1
                    ih = idxs_loc
                    dc = (1.0 - r) * dc_grid_t_local[il] + r * dc_grid_t_local[ih]
                else:
                    dc = atools.dcfun_at(z, H0, Om, w0, interp=interp)

            if logdd is None:
                if have_logdd_grid:
                    il = idxs_loc - 1
                    ih = idxs_loc
                    logdd = (1.0 - r) * logdd_grid_t_local[il] + r * logdd_grid_t_local[ih]
                else:
                    logdd = atools.log_ddL_dz(z, H0, Om, w0, Xi0, n_, dc=dc, interp=interp, param=param)
        else:
            z, dc, logdd = _cosmo_direct(dL, H0, Om, w0, Xi0, n_)

        # -----------------------------
        # Evaluate per-injection logp (vector over K) in compute_dtype
        # -----------------------------
        mass_1_use, mass_2_use, m1Src, m2Src = _mass_inputs(m1, m2, z)

        lp = log_p_pop_at(
            mass_1_use, mass_2_use, z, dL, spins_use, Lambda_local,
            rate_model, mass_model, spin_model,
            smoothing=smoothing,
            simplex_repair=simplex_repair,
            has_m2_break=has_m2_break,
            log_ddL_dz_pre=logdd,
            dc=dc,
            interp_vals_mass=iv,
            interp_grids_mass=ig,
            verbose=verbose,
        )
        #lp = at.cast(lp, compute_dtype)

        if use_dp:
            lp = _dp_jacobian_fix(lp, m1Src, m2Src, z)

        # -----------------------------
        # Chunk reduction (keep it in accum_dtype)
        # -----------------------------
        x = at.where(mask, lp - lpd - lpic, NEG_BIG)  # compute_dtype
        m = at.max(x)                                # compute_dtype
        y = at.exp(x - m)                            # compute_dtype

        # sums in float64 (or compute_dtype if already float64)
        s1c = at.sum(y)
        s2c = at.sum(y * y)

        # promote m to accum dtype for stable accumulation
        #m = at.cast(m, accum_dtype)

        m_new,  s1_new = _combine_logsumexp(m_state,  s1_state,  m,   s1c)
        m2c = 2.0 * m #at.as_tensor_variable(2.0, dtype=accum_dtype) * m
        m2_new, s2_new = _combine_logsumexp(m2_state, s2_state, m2c, s2c)
        return m_new, m2_new, s1_new, s2_new

    # -----------------------------
    # Run scan (scan over chunks C)
    # -----------------------------
    m_init = at.as_tensor_variable(-np.inf, dtype="float64")
    s_init = at.as_tensor_variable(0.0,     dtype="float64")

    scan_kwargs = dict(
        fn=step,
        sequences=seqs,
        outputs_info=[m_init, m_init, s_init, s_init],
        non_sequences=nonseq,
        strict=True,
        profile=True,
    )

    # Keep only last step if supported (saves graph/memory); otherwise fall back.
    try:
        (m_out, m2_out, s1_out, s2_out), _ = pytensor.scan(**scan_kwargs, return_steps=1)
        m_last  = m_out[-1]
        m2_last = m2_out[-1]
        s1_last = s1_out[-1]
        s2_last = s2_out[-1]
    except TypeError:
        (m_out, m2_out, s1_out, s2_out), _ = pytensor.scan(**scan_kwargs)
        m_last  = m_out[-1]
        m2_last = m2_out[-1]
        s1_last = s1_out[-1]
        s2_last = s2_out[-1]

    # -----------------------------
    # Final reductions (float64 if compute_dtype=float64)
    # -----------------------------
    logsumexp1 = m_last  + at.log(at.maximum(s1_last, tinyL))
    logsumexp2 = m2_last + at.log(at.maximum(s2_last, tinyL))

    Ndraw_t = Ndraw #at.cast(at.as_tensor_variable(Ndraw), accum_dtype)
    log_mu  = logsumexp1 - at.log(Ndraw_t)
    logs2   = logsumexp2 - at.log(Ndraw_t)

    logNeff = 2.0 * log_mu - logs2 + at.log(Ndraw_t)
    Neff    = at.exp(logNeff)

    var_log_lik_u = (
        atools.logdiffexp(logs2 - 2.0 * log_mu, 1.0)
        - at.log(Ndraw_t - 1. )
    )

    return log_mu, Neff, var_log_lik_u

    

    
#####################################################
#####################################################

# MODEL

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
                 dLprior = ['none'],
                 fix_inj_len = False,
                 chunk_inj = -1,
                 chunk_reduce = False,
                 use_float32 = False,
                 use_float32_bias=False,
                 sel_method='Tobs',
                 N_DP_comp_max = 100,
                 alpha_tail = 0.2,
                 alpha_small = 0.01,
                 L_small_1 = 0.5,
                 L_small_2 = 0.5,
                 L_small_3 = 0.1,
                 s_local = 0.5,
                 find_m_bounds = False,
                 alpha_inv_params = (1, 1),
                 fix_H0 = True,
                fix_Om = True,
               fix_w0 = True,
                 fix_Xi0n = True,
                 z_pivot=0.5,
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
                 interp_inj=True,
                 param='vanilla',
                 DP_prior='SB',
                 sigma_softmax=0.75,
                 gamma_DP_params = (4, 0.8),
                 is_observed = False,
                 sample_from_pop = False,
                 mmin_inj=-1,
                 is_compressed_inj=False,
                 debug_sel_batch=False,
                 reparam_z = True,
                 reparam_mass = False,
                ):


    
    # X = np.float32 if use_float32 else np.float64  # model dtype

    # X_name = "float32" if use_float32 else "float64"  # model dtype

    # work_dtype = "float32" if use_float32 else "float64"
    # int_dtype = "int32" if use_float32 else "int64"
    # print("work_dtype in model is %s"%work_dtype)
    # print("int_dtype in model is %s"%int_dtype)

    # half_ = at.as_tensor_variable(0.5, dtype=work_dtype)
    # two_pi_ = at.as_tensor_variable(2*np.pi, dtype=work_dtype)


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
    Nevs_np = np.atleast_1d(Nevs) #Nevs.eval()

    Tobs_np = Tobs #Tobs.eval()

        
    if not pop_only:
        N = mus_l.shape[0]#.astype(int_dtype) # number of events in total
        N_np = N#.astype(int_dtype) #N.eval()
        ngmm = mus_l.shape[1]#.astype(int_dtype)
        ngmm_np = ngmm#.astype(int_dtype) #ngmm.eval()
        nd = mus_l.shape[2]#.astype(int_dtype)
        nd_np = nd#.astype(int_dtype) #nd.eval()
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

    logN = np.log(N)


    
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


    
    # if use_float32_bias:
    #     if not use_float32:
    #         XI = np.float32
    #     else:
    #         XI = X
    # else:
    #     XI = X
    # print("Model dtype will be %s"%X)
    # print("Injections dtype will be %s"%XI)




    
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

        
        z_from_dL_sym = atools.z_from_dL_at(d_sym, H0_sym, Om_sym, w0_sym, Xi0_sym, n_sym, interp=pade, param=param)
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

            print("\nFinding prior range for DP-GMM.")
       
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
    
            lowmu1 = scales['lMc_min_data']#.astype(X)
            upmu1 = scales['lMc_max_data']#.astype(X)
    
            lowmu2 = scales['lq_min_data']#.astype(X)
            upmu2 = scales['lq_max_data']#.astype(X)

            lowmu3 = scales['logz_min_data']#.astype(X)
            upmu3 = scales['logz_max_data']#.astype(X)


            lowmu1_inj = scales['lMc_min_inj']#.astype(X)
            upmu1_inj = scales['lMc_max_inj']#.astype(X)
    
            lowmu2_inj = scales['lq_min_inj']#.astype(X)
            upmu2_inj = scales['lq_max_inj']#.astype(X)

            lowmu3_inj = scales['logz_min_inj']#.astype(X)
            upmu3_inj = scales['logz_max_inj']#.astype(X)
    
            L_small_1_data = scales['lMc_diff']#.astype(X)
            L_small_2_data = scales['lq_diff']#.astype(X)

            L_small_m1 = scales['m1_diff']#.astype(X)
            L_small_m2 = scales['m2_diff']#.astype(X)
    
            L_small_3_data = scales['logz_diff']#.astype(X)

            print("Mass/redshift DP-GMM prior values found, overwriting default:")
            print("lowmu1=%s, upmu1=%s, lowmu2=%s, upmu2=%s"%(lowmu1, upmu1, lowmu2, upmu2))
            print("L_small_1_data=%s, L_small_2_data=%s, L_small_3_data=%s"%(L_small_1_data, L_small_2_data,L_small_3_data ))
            print("L_small_m1=%s, L_small_m2=%s"%(L_small_m1, L_small_m2, ))


            

            if L_small_1>0 and L_small_2>0:
                print("Finding min spacing based on requested spacing in m1-m2...")

                m1_min = min(scales['m1_min_inj'], scales['m1_min_data'] )
                m2_min = min(scales['m2_min_inj'], scales['m2_min_data']  )
                m1_max = min(scales['m1_max_inj'], scales['m1_max_data'] )
                m2_max = min(scales['m2_max_inj'], scales['m2_max_data']  )

                dmin = putils.min_sep_logMc_logitq(
                    m1_min=m1_min, m1_max=m1_max,
                    m2_min=m2_min, m2_max=m2_max,
                    dm1=L_small_1, dm2=L_small_2
                )


                L_small_1 = max(L_small_1_data, dmin[0])#.astype(X) 
                L_small_2 = max(L_small_2_data, dmin[1])#.astype(X)
            else:
                L_small_1 = L_small_1_data
                L_small_2 = L_small_2_data

                if L_small_3>0:

                    L_small_3 = max(L_small_3_data, np.log(1+L_small_3/max(z_max_data, max_z)) )#.astype(X)  
                else:
                    L_small_3 = L_small_3_data
            print(" Final L_small_1=%s, L_small_2a=%s, L_small_3=%s"%(L_small_1, L_small_2,L_small_3 ))

            
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
        
        tgrid_m1 = np.linspace(0.0, 1.0, interp_mass )#.astype(X)
        tgrid_m2 = np.linspace(0.0, 1.0, 500 )#.astype(X)

  
        if mass_model in ('DPLDP', 'DPLDP-z', 'PLPreg'):
            
            if mass_model =='DPLDP':
                sigma_min = min(priors["sigma1"][0], priors["sigma2"][0])
            elif mass_model=='PLPreg':
                sigma_min = priors["sigmaMass"][0]
            else:
                sigma_min = min(priors["sigma1_0"][0], priors["sigma2_0"][0])
            

        
        elif mass_model in ('DPUC', 'DP'):  

            if sel_method=='skip':
            
                MMIN_GRID = lowmu1#*(1-0.1)
                MMAX_GRID = upmu1#*(1+0.1)

                MMIN_GRID_1 = lowmu2 #*(1-0.1)
                MMAX_GRID_1 = upmu2#*(1+0.1)

                MMIN_GRID_2 = lowmu3#*(1-0.1)
                MMAX_GRID_2 = upmu3#*(1+0.1)
                
            else:
                MMIN_GRID = min(lowmu1_inj, lowmu1)#*(1-0.1)
                MMAX_GRID = max(upmu1, upmu1_inj)#*(1+0.1)

                MMIN_GRID_1 = min(lowmu2_inj, lowmu2) #*(1-0.1)
                MMAX_GRID_1 = max(upmu2, upmu2_inj)#*(1+0.1)

                MMIN_GRID_2 = min(lowmu3_inj, lowmu3)#*(1-0.1)
                MMAX_GRID_2 = max(upmu3, upmu3_inj)#*(1+0.1)

            
            print("Grid in log(Mc) source between %s and %s"%(MMIN_GRID, MMAX_GRID))
            print("Grid in logit(q) source between %s and %s"%(MMIN_GRID_1, MMAX_GRID_1))
            print("Grid in log(1+z) source between %s and %s"%(MMIN_GRID_2, MMAX_GRID_2))


        else:
            raise ValueError('Interpolation not available for this mass model.')

    if is_observed:
        print("Building optimal SNR interpolant...")

        # load interpolant
        with h5py.File('../tables/optimal_snr_aplus_design_05.h5','r') as f:
            m_grid_at = at.as_tensor_variable(np.array(f['ms']))
            osnrs_grid_at = at.as_tensor_variable(np.array(f['SNR']))
            #ref_dist_Gpc_at = at.as_tensor_variable(np.array(1.), dtype=work_dtype)
        grid_at = (m_grid_at, m_grid_at)
        osnr_interp_at = atools.GridInterpolator_at(grid_at, osnrs_grid_at)
                  
    if sample_from_pop:
        print("Finding init vals for individual event params...")

        rng = np.random.default_rng()
        x = rng.standard_normal(size=(N, nd))
        samples_init = putils.sample_from_per_event_gmm(wts_l, mus_l, cho_covs_l, x, rng=None)

        log_Mc_det_init = samples_init[:, 0]
        logit_q_init = samples_init[:, 1]#.astype(X)
        logd_init = samples_init[:, 2]

        z_init = (atools.z_from_dL_at(at.exp(logd_init), 67.7, 0.31, -1, 1, 0)).eval()#.astype(X)
        log_onepz_init = np.log1p(z_init)#.astype(X)
        log_Mc_src_init = (log_Mc_det_init - log_onepz_init)#.astype(X)


    zgrid_ =  stop_grad(  at.as_tensor_variable(atools.make_z_grid(total=zres, zmin_a=zmin_a, zmin_b=zmin_b, zmid_b=zmid_b, zmax_c=zmax_c, hi_boost=hi_boost) ) )

    zgrid_mass_ =  stop_grad( at.as_tensor_variable( atools.make_z_grid(total=interp_z, zmin_a=zmin_a, zmin_b=zmin_b, zmid_b=zmid_b, zmax_c=zmax_c, hi_boost=hi_boost) ) )
    
    print("z grid for interpolation built. Resolution: %s"%zres)
    print("z min: %s , z max: %s"%(zmin_a, zmax_c))


    if z_pivot!=0:

        H0_min, H0_max = priors['H0']
        Om_min, Om_max = priors['Om']
        w0_min, w0_max = priors['w0']

        # Evaluate E(z_pivot) at the corners of the (Om, w0) prior box
        Ez_corners = [
            atools.Efun_num(z_pivot, Om_min, w0_min), #.astype(X),
            atools.Efun_num(z_pivot, Om_min, w0_max), #.astype(X),
            atools.Efun_num(z_pivot, Om_max, w0_min), #.astype(X),
            atools.Efun_num(z_pivot, Om_max, w0_max), #.astype(X),
        ]
        Ez_min = min(Ez_corners)#.astype(X)
        Ez_max = max(Ez_corners)#.astype(X)

    if reparam_mass:

        if mass_model=='PLPreg':

            # --- prior bounds and initvals ---
            ml_min, ml_max       = priors["ml"]
            mh_min, mh_max       = priors["mh"]
            deltam_min, deltam_max = priors["deltam"]
            muM_min, muM_max     = priors["muMass"]
            sM_min, sM_max       = priors["sigmaMass"]
            lam_min, lam_max     = priors["lambdaPeak"]
            
            ml_init   = ivals.get("ml",        4.0)#.astype(X)
            mh_init   = ivals.get("mh",        100.0)#.astype(X)
            dm_init = ivals.get("deltam",    3.0)#.astype(X)
            muM_init  = ivals.get("muMass",    35.0)#.astype(X)
            sM_init   = ivals.get("sigmaMass", 5.0)#.astype(X)
            lam_init  = ivals.get("lambdaPeak", 0.05)#.astype(X)
            
            # Small helper for stable initvals on (0,1)
            def _clip01(x, eps=1e-6):
                #if dtype=="float64":
                return np.clip(x, eps, 1.0 - eps)
                # else:
                #     return np.float32(np.clip(x, eps, 1.0 - eps))
            
            # =========================
            # 1) ml with EXACT Uniform prior
            # =========================
            # Will sample a fraction u_ml ~ Uniform(0,1), map to ml = ml_min + u_ml*(ml_max-ml_min)
            # Add Jacobian log|dml/du_ml| = log(ml_max-ml_min) so induced density on ml is uniform.
            u_ml_init = _clip01((ml_init - ml_min) / (ml_max - ml_min))

            u_mh_init = _clip01((mh_init - mh_min) / (mh_max - mh_min))

            u_dm_init = _clip01((dm_init - deltam_min) / (deltam_max - deltam_min))

            log_sM_min = np.log(sM_min)
            log_sM_max = np.log(sM_max)
            log_sM_init = np.clip(np.log(sM_init), log_sM_min + 1e-6, log_sM_max - 1e-6)
            



    vol_in_prior = any('UniformSourceFrame' in s or 'UniformComovingVolume' in s for s in dLprior)
    
    all_dLsq_prior = all(s == 'dLsq' for s in dLprior)
    all_no_dL_prior = all(s == 'none' for s in dLprior)

    edges = [0]
    for n in Nevs_np:
        edges.append(edges[-1] + int(n))

    if vol_in_prior:

        zgrid_dLp = stop_grad(  at.as_tensor_variable(atools.make_z_grid(total=zres, zmin_a=zmin_a, zmin_b=zmin_b, zmid_b=zmid_b, zmax_c=zmax_c, hi_boost=hi_boost) ) )

        dc_grid_Planck15 = atools.dcfun_at(zgrid_dLp, 67.90, 0.3065, -1., interp=False)#.astype(work_dtype)
        dL_grid_Planck15 = atools.dLfun_at(zgrid_dLp, 67.90, 0.3065, -1., 1., 0., interp=False, dc=dc_grid_Planck15, param='vanilla')#.astype(work_dtype)
        

    ################################################
    # Build model
    ################################################
    
    with pm.Model(coords=coords) as model:


        # if sampling_GW=='gauss':
            
        #     # we sample single-event parameters from broad gaussian approximations of the posteriors
        #     mus_s, cho_s, log_wts_l, mus_l, icovs_l, log_dets_l = at.as_tensor_variable(mus_s, dtype=work_dtype), at.as_tensor_variable(cho_s, dtype=work_dtype), at.as_tensor_variable(log_wts_l, dtype=work_dtype), at.as_tensor_variable(mus_l, dtype=work_dtype), at.as_tensor_variable(icovs_l, dtype=work_dtype), at.as_tensor_variable(log_dets_l, dtype=work_dtype)

            
        # elif 'gmm' in sampling_GW:
        #     # we sample single-event parameters from the actual single-event posteriors
        #     wts_l, mus_l, cho_covs_l = at.as_tensor_variable(wts_l, dtype=work_dtype), at.as_tensor_variable(mus_l, dtype=work_dtype), at.as_tensor_variable(cho_covs_l, dtype=work_dtype)

        
        ################################################
        # Cosmological parameters
        ################################################

        if fix_Om:
            Om_ = params_fix['Om']
        else:
            Om_ = pm.Uniform('Om', lower=priors['Om'][0], upper=priors['Om'][1], initval=ivals.get('Om')) 

        if fix_w0:
            w0_ = at.as_tensor_variable(-1.)
        else:
            if pade:
                raise NotImplementedError("Pade appproximation with varying w0 not implemented yet. Use pade=False")
            w0_ =  pm.Uniform('w0', lower=priors['w0'][0], upper=priors['w0'][1], initval=ivals.get('w0'))

            
        
        if fix_H0:
            H0_ =  params_fix['H0']
        else:
            if z_pivot!=0:
                print("Sampling in H(z=%s)"%z_pivot)
                # Define a broad prior for Hp = H(z_pivot)
                # Just choose constants that safely cover the H0 prior range once divided by Ez_pivot.
                Hp_min = H0_min * Ez_min
                Hp_max = H0_max * Ez_max

                H0_init = ivals.get('H0', at.as_tensor_variable(67.7) )
                
                Hp_ = pm.Uniform(
                    'Hp',
                    lower=Hp_min,
                    upper=Hp_max,
                    # optional: if you have numeric Om_init, w0_init, you can precompute a good initval
                    initval=H0_init * atools.Efun_at(z_pivot, ivals.get('Om', 0.3), ivals.get('w0', -1)).eval()
                )

                # E(z_pivot; Om, w0) using your helper
                Ez_pivot = atools.Efun_at(z_pivot, Om_, w0_)
                
                # Physical H0 as a deterministic transform of (Hp, Om, w0)
                H0_ = pm.Deterministic('H0', Hp_ / Ez_pivot)
                
                # Jacobian term: prior flat in H0, but sampling in Hp
                _ = pm.Potential('J_H0_from_Hp', -at.log(Ez_pivot))

            else:
                
                H0_ =  pm.Uniform('H0', lower=priors['H0'][0], upper=priors['H0'][1], initval=ivals.get('H0'))
        
        
        
        if fix_Xi0n:
            Xi0_ =  at.as_tensor_variable(1.) #, dtype=work_dtype)
            nXi0_ = at.as_tensor_variable(0.) #, dtype=work_dtype)
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


            if reparam_z:
                print("Using reparametrized variables for easier geometry")

                gamma_min, gamma_max = priors['gamma']
                kappa_min, kappa_max = priors['kappa']
    
                gamma_init = ivals.get('gamma', 3.2)
                kappa_init = ivals.get('kappa', 3.)
                
                # Prior ranges for s and d
                s_min = gamma_min + kappa_min
                s_max = gamma_max + kappa_max
                d_min = gamma_min - kappa_max
                d_max = gamma_max - kappa_min
                
                s_ = pm.Uniform("gamma_plus_kappa", lower=s_min, upper=s_max,
                                initval=gamma_init + kappa_init)
                d_ = pm.Uniform("gamma_minus_kappa", lower=d_min, upper=d_max,
                                initval=gamma_init - kappa_init)
                
                gamma_ = pm.Deterministic("gamma", 0.5 * (s_ + d_))
                kappa_ = pm.Deterministic("kappa", 0.5 * (s_ - d_))
                #_ = pm.Potential("J_gamma_kappa_from_s_d", -np.log(2.0))
    
    
                z_p_min, z_p_max = priors['zp']
    
                z_p_init = np.asarray( ivals.get('zp'), ) #dtype=X )
    
                log1p_zp_ = pm.Uniform(
                    "log1p_zp",
                    lower = np.log1p(z_p_min), #.astype(X),
                    upper = np.log1p(z_p_max), #.astype(X),
                    initval = np.log1p(z_p_init), #.astype(X) ,
                )
                
                zp_ = pm.Deterministic("zp", at.expm1(log1p_zp_))
                # for flat prior in zp
                pm.Potential("J_zp_from_log1p_MD", at.log1p(zp_))

            else:
                gamma_ = pm.Uniform('gamma', lower=priors['gamma'][0], upper=priors['gamma'][1], initval=ivals.get('gamma'))    
                kappa_ = pm.Uniform('kappa', lower=priors['kappa'][0], upper=priors['kappa'][1], initval=ivals.get('kappa'))
                zp_ = pm.Uniform('zp', lower=priors['zp'][0], upper=priors['zp'][1], initval=ivals.get('zp'))


            
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
                    _ = pm.Potential('bound_alphaChi', at.switch( at.le(alphaChi_, 1. ), -np.inf, 0. ) )
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
            
            alpha_  = pm.Uniform("alpha",      lower=priors["alpha"][0],      upper=priors["alpha"][1],      initval=ivals.get("alpha"))
            beta_   = pm.Uniform("beta",       lower=priors["beta"][0],       upper=priors["beta"][1],       initval=ivals.get("beta"))
            muM_    = pm.Uniform("muMass",     lower=priors["muMass"][0],     upper=priors["muMass"][1],     initval=ivals.get("muMass"))
            
            if not reparam_mass:
                
                lamP_   = pm.Uniform("lambdaPeak", lower=priors["lambdaPeak"][0], upper=priors["lambdaPeak"][1], initval=ivals.get("lambdaPeak"))        
                ml_     = pm.Uniform("ml",         lower=priors["ml"][0],         upper=priors["ml"][1],         initval=ivals.get("ml"))
                mh_     = pm.Uniform("mh",         lower=priors["mh"][0],         upper=priors["mh"][1],         initval=ivals.get("mh"))
                deltam_ = pm.Uniform("deltam",     lower=priors["deltam"][0],     upper=priors["deltam"][1],     initval=ivals.get("deltam"))
                sM_     = pm.Uniform("sigmaMass",  lower=priors["sigmaMass"][0],  upper=priors["sigmaMass"][1],  initval=ivals.get("sigmaMass"))
                
            else:
                print("Using reparametrized variables for easier geometry")


                # tiny buffer to avoid exact boundaries
                eps = 1e-9


                u_ml = pm.Uniform(
                    "u_ml",
                    0.0, 1.0,
                    initval=_clip01((ml_init - ml_min) / (ml_max - ml_min)),
                )
                ml_ = pm.Deterministic("ml", ml_min + u_ml * (ml_max - ml_min))
                # (Jacobian is constant -> optional; leaving it out changes nothing)
                # pm.Potential("J_ml", at.log(ml_max - ml_min))


                # -----------------------
                # 2) mh | ml ~ Uniform(max(mh_min, ml+eps), mh_max)
                #    Ensures mh > ml by construction
                # -----------------------
                mh_lower = at.maximum(mh_min, ml_ + eps)
                mh_span  = mh_max - mh_lower
                
                u_mh = pm.Uniform(
                    "u_mh",
                    0.0, 1.0,
                    initval=_clip01((mh_init - max(mh_min, ml_init + eps)) / (mh_max - max(mh_min, ml_init + eps))),
                )
                mh_ = pm.Deterministic("mh", mh_lower + u_mh * mh_span)
                
                # This Jacobian term makes the induced density for mh|ml uniform in physical space.
                # (Because we sample u_mh uniformly but want flat in mh, and mh_span depends on ml.)
                pm.Potential("J_mh_given_ml", at.log(at.maximum(mh_span, eps)))
                
                # -----------------------
                # 3) deltam | (ml,mh) ~ Uniform(dm_min, min(dm_max, (mh-ml)-eps))
                #    Ensures deltam < (mh-ml) by construction
                # -----------------------
                span_ = mh_ - ml_
                dm_upper = at.minimum(deltam_max, span_ - eps)
                dm_span  = dm_upper - deltam_min
                
                u_dm = pm.Uniform(
                    "u_deltam",
                    0.0, 1.0,
                    initval=_clip01((dm_init - deltam_min) / (deltam_max - deltam_min)),
                )
                deltam_ = pm.Deterministic("deltam", deltam_min + u_dm * dm_span)
                
                pm.Potential("J_deltam_given_span", at.log(at.maximum(dm_span, eps)))
                
                # hard rejection when dm_upper <= dm_min (no valid deltam)
                pm.Potential("valid_deltam_domain", at.switch(dm_span > 0, 0.0, -np.inf))
                
                # -----------------------
                # 4) sigmaMass ~ Uniform(sM_min, sM_max)  (keep it simple & truly flat)
                # -----------------------
                u_sM = pm.Uniform(
                    "u_sigmaMass",
                    0.0, 1.0,
                    initval=_clip01((sM_init - sM_min) / (sM_max - sM_min), ) #dtype=X_name),
                )
                sM_ = pm.Deterministic("sigmaMass", sM_min + u_sM * (sM_max - sM_min))
                # (Jacobian constant -> optional)
                # pm.Potential("J_sigmaMass", at.log(sM_max - sM_min))
                


                lamP_ = pm.Uniform("lambdaPeak", lower=lam_min, upper=lam_max, initval=lam_init)

                


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
            m_high_   = pm.Deterministic("m_high", at.as_tensor_variable(300.0)) #.astype(X)  )
            delta_m1_ = pm.Uniform("delta_m1", lower=priors["delta_m1"][0], upper=priors["delta_m1"][1], initval=ivals.get("delta_m1"))
            lambda_vec = pm.Dirichlet("lambda", a=np.asarray([1, 1, 1]), initval=np.asarray(ivals.get("lambda")))
            lambda0_  = pm.Deterministic("lambda0", lambda_vec[0])
            lambda1_  = pm.Deterministic("lambda1", lambda_vec[1])
            lambda2_  = pm.Deterministic("lambda2", lambda_vec[2])
            beta_     = pm.Uniform("beta",     lower=priors["beta"][0],     upper=priors["beta"][1],     initval=ivals.get("beta"))
            delta_m2_ = pm.Uniform("delta_m2", lower=priors["delta_m2"][0], upper=priors["delta_m2"][1], initval=ivals.get("delta_m2"))
            epsilon_  = pm.Deterministic("epsilon", at.as_tensor_variable(0.01) )
            
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
            
            #sigma1_0  = pm.Uniform("sigma1_0",  lower=priors["sigma1_0"][0],  upper=priors["sigma1_0"][1],  initval=ivals.get("sigma1_0"))
            sigma1_0 = pm.Truncated(
                        "sigma1_0",
                        pm.LogNormal.dist(mu=np.log(0.6), sigma=0.9),
                        lower=priors["sigma1_0"][0],
                        upper=priors["sigma1_0"][1],
                        initval=ivals.get("sigma1_0"),
                    )
            
            mu2_0     = pm.Uniform("mu2_0",     lower=priors["mu2_0"][0],     upper=priors["mu2_0"][1],     initval=ivals.get("mu2_0"))
            #sigma2_0  = pm.Uniform("sigma2_0",  lower=priors["sigma2_0"][0],  upper=priors["sigma2_0"][1],  initval=ivals.get("sigma2_0"))
            sigma2_0 = pm.Truncated(
                        "sigma2_0",
                        pm.LogNormal.dist(mu=np.log(4.0), sigma=0.9),
                        lower=priors["sigma2_0"][0],
                        upper=priors["sigma2_0"][1],
                        initval=ivals.get("sigma2_0"),
                    )
            
            
            delta_m1_ = pm.Uniform("delta_m1",  lower=priors["delta_m1"][0],upper=priors["delta_m1"][1],initval=ivals.get("delta_m1"))
            
            # m1_low, m2_low, m_high as in your original block
            u        = pm.Uniform("u", 0, 1, initval=ivals.get("u"))
            m1_low_  = pm.Deterministic("m1_low", (3 + (10 - 3) * at.sqrt(u)) ) #.astype(X) )
            v        = pm.Uniform("v", 0, 1, initval=ivals.get("v"))
            m2_low_  = pm.Deterministic("m2_low", (3 + v * (m1_low_ - 3)) ) #.astype(X))
            m_high_  = pm.Deterministic("m_high", at.as_tensor_variable(300.0)) #.astype(X))
            

            
            # secondary-mass hyperparams (unchanged unless you also evolve them)
            beta_     = pm.Uniform("beta",     lower=priors["beta"][0],     upper=priors["beta"][1],     initval=ivals.get("beta"))
            delta_m2_ = pm.Uniform("delta_m2", lower=priors["delta_m2"][0], upper=priors["delta_m2"][1], initval=ivals.get("delta_m2"))
            epsilon_  = pm.Deterministic("epsilon", at.as_tensor_variable(0.1) ) #.astype(X))
            
            if has_m2_break:
                print("Including gap for secondary mass")
                m_g_     = pm.Uniform("m_g", lower=priors["m_g"][0], upper=priors["m_g"][1], initval=ivals.get("m_g"))
                w_g_     = pm.Uniform("w_g", lower=priors["w_g"][0], upper=priors["w_g"][1], initval=ivals.get("w_g"))
                sig_g_l_ = at.as_tensor_variable(1e-02)#.astype(X)
                sig_g_h_ = at.as_tensor_variable(1e-02)#.astype(X)
            else:
                m_g_     = at.as_tensor_variable(45.)#.astype(X)
                w_g_     = at.as_tensor_variable(70.)#.astype(X)
                sig_g_l_ = at.as_tensor_variable(1e-02)#.astype(X)
                sig_g_h_ = at.as_tensor_variable(1e-02)#.astype(X)



            # # mixture weights at z≈0
            #eps_w = at.as_tensor_variable(1e-12)#.astype(X)

            # read lambda prior from priors file
            lam0_prior = priors.get("lambda0_vec_0", "Dirichlet(1,1,1)")
            
            if isinstance(lam0_prior, str) and lam0_prior.startswith("Dirichlet"):
                # parse "Dirichlet(a,b,c)"
                inside = lam0_prior[len("Dirichlet("):-1]
                alphas = [float(x.strip()) for x in inside.split(",")]
            else:
                # alternatively allow direct numeric lists in future
                alphas = lam0_prior
            
            alphas = np.asarray(alphas)
            lambda_vec0 = pm.Dirichlet(
                                    "lambda0_vec",
                                    a=alphas,
                                    initval=np.asarray(ivals.get("lambda"))
                                )



            lambda0_0 = pm.Deterministic("lambda0_0", lambda_vec0[0])
            lambda1_0 = pm.Deterministic("lambda1_0", lambda_vec0[1])
            lambda2_0 = pm.Deterministic("lambda2_0", lambda_vec0[2])


            alpha1_inf_,  z_alpha1_,  dz_alpha1_ = putils.evo_triplet(
                "alpha1",
                theta0_rv=alpha1_0,
                ivals=ivals,
                priors=priors,
            )
        
            alpha2_inf_,  z_alpha2_,  dz_alpha2_ = putils.evo_triplet(
                "alpha2",
                theta0_rv=alpha2_0,
                ivals=ivals,
                priors=priors,
            )
        
            # mb_inf_,      z_mb_,      dz_mb_     = putils.evo_triplet(
            #     "mb",
            #     theta0_rv=mb_0,
            #     ivals=ivals,
            #     priors=priors,
            # )
            
            mb_inf_ = pm.Deterministic("mb_inf", mb_0) 
            z_mb_   = pm.Deterministic("z_mb", at.as_tensor_variable(0.0) ) #.astype(X)) 
            dz_mb_  = pm.Deterministic("dz_mb", at.as_tensor_variable(1.0)) #.astype(X))  
            
        
            mu1_inf_,     z_mu1_,     dz_mu1_    = putils.evo_triplet(
                "mu1",
                theta0_rv=mu1_0,
                ivals=ivals,
                priors=priors,
            )
        
            sigma1_inf_,  z_sigma1_,  dz_sigma1_ = putils.evo_triplet(
                "sigma1",
                theta0_rv=sigma1_0,
                ivals=ivals,
                priors=priors,
            )
        
            mu2_inf_,     z_mu2_,     dz_mu2_    = putils.evo_triplet(
                "mu2",
                theta0_rv=mu2_0,
                ivals=ivals,
                priors=priors,
            )
        
            sigma2_inf_,  z_sigma2_,  dz_sigma2_ = putils.evo_triplet(
                "sigma2",
                theta0_rv=sigma2_0,
                ivals=ivals,
                priors=priors,
            )


            # -------------------------
            # High-z mixture weights + shared evolution S_lambda(z)
            # -------------------------

            # Global redshift–transition priors for all evolving hyperparameters
            z_t_prior = priors.get("z_t", (0.05, 1.5))   # lower/upper for z_transition
            dz_prior  = priors.get("dz",  (0.05, 2.0))   # lower/upper for Δz

            # High-z endpoint on the simplex
            lambda_vec_inf = pm.Dirichlet(
                "lambda_inf_vec",
                a=np.asarray([1, 1, 1]),
                initval=np.asarray(ivals.get("lambda_inf_vec", [0.10, 0.05, 0.85])),
            )
            lambda0_inf_ = pm.Deterministic("lambda0_inf", lambda_vec_inf[0])
            lambda1_inf_ = pm.Deterministic("lambda1_inf", lambda_vec_inf[1])
            lambda2_inf_ = pm.Deterministic("lambda2_inf", lambda_vec_inf[2])
            
            # Shared transition redshift and width for the mixture weights
            z_lambda_ = pm.Uniform(
                "z_lambda",
                lower=z_t_prior[0],
                upper=z_t_prior[1],
                initval=ivals.get("z_lambda", 1.1),
            )
            log_dz_lambda_ = pm.Uniform(
                "log_dz_lambda",
                lower=np.log(dz_prior[0]), #.astype(X),
                upper=np.log(dz_prior[1]), #.astype(X),
                initval=ivals.get("log_dz_lambda", np.log(0.5) ),
            )
            dz_lambda_ = pm.Deterministic("dz_lambda", at.exp(log_dz_lambda_))

            
            if simplex_repair:
                print("Will enforce lambda0(z), lambda1(z), lambda2(z) on the simplex")
                raise ValueError()
            # -------------------------
            # Pack hyperparameters for logpdf_DPLDP_z wrapper
            #   - low-z vector: same order, but with *_0 values
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
                lambda0_inf_, lambda1_inf_, z_lambda_, dz_lambda_,
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
    
                beta_init = np.full(N_DP_comp_max_np, 1e-02)#.astype(X)
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

            
            logw = at.log(w)

        

            #### Mean prior 

            # DPLDP 1k
            # lowmu1 = 1.5
            # upmu1 = 5.5
            # lowmu2 =  -1.2
            # upmu2 =  10.

            U1, U2 = (upmu1-lowmu1) , (upmu2-lowmu2)    # "too-wide" typical std per dim 

            mu1_center = (lowmu1 + upmu1) / 2.0  # 3.55
            mu2_center = (lowmu2 + upmu2) / 2.0
            
            
     
            mu1 = pm.Uniform('mulMc', lower=lowmu1, upper=upmu1, dims= ("component" ), initval=np.full(N_DP_comp_max_np, mu1_center)) #.astype(X) )
            mu2 = pm.Uniform('mulq', lower=lowmu2, upper=upmu2, dims= ("component" ), initval=np.full(N_DP_comp_max_np, mu2_center)) #.astype(X))

            if rate_model in ('DPUC','DPUC-vol' ):
                mu3_center = ( lowmu3+ upmu3) / 2.0
                mu3 = pm.Uniform('mulz', lower=lowmu3, upper=upmu3, dims= ("component" ), initval=np.full(N_DP_comp_max_np, mu3_center)) #.astype(X))

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
            #lambda_ell_1 = -at.log(alpha_small) * L_small_1**(alpha_shape) # small scale
            #lambda_ell_2 = -at.log(alpha_small) * L_small_2**(alpha_shape) # small scale
            
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

            eps1 = pm.Normal("eps1", 0.0, s_local, dims=("component",), initval=np.zeros(N_DP_comp_max_np)) #.astype(X))
            eps2 = pm.Normal("eps2", 0.0, s_local, dims=("component",), initval=np.zeros(N_DP_comp_max_np)) #.astype(X))

            # eps1 = pm.SkewNormal("eps1", mu=0, sigma=s_local, alpha=+2, dims=("component",), initval=np.zeros(N_DP_comp_max_np).astype(X) )
            # eps2 = pm.SkewNormal("eps2", mu=0, sigma=s_local, alpha=+2, dims=("component",), initval=np.zeros(N_DP_comp_max_np).astype(X))


            sig1 = pm.Deterministic("sig1", tau1 * at.exp(eps1) , dims="component")   
            sig2 = pm.Deterministic("sig2", tau2 * at.exp(eps2), dims="component")  

            
            if rate_model in ('DPUC', 'DPUC-vol'):

                
                U3 = (upmu3-lowmu3)

                print("L_small_3 = %s "%L_small_3)
                print("U3 = %s "%U3)

                tau3 = pm.Uniform("tau3", lower=L_small_3, upper=U3, )
                # eps3 = pm.SkewNormal("eps3", mu=0, sigma=s_local, alpha=+2, dims=("component",), initval=np.zeros(N_DP_comp_max_np).astype(X))
                eps3 = pm.Normal("eps3", 0.0, s_local, dims=("component",), initval=np.zeros(N_DP_comp_max_np)) #.astype(X))
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
                #     (eta - 1.0) * at.log(1.0 - rho**2).sum()
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
                    -2.0 * at.log(sig1) - 2.0 * at.log(sig2) - at.log(one_minus_r2),
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
        lR0 = at.log(R0)


        
        # Precompute cosmology pieces 
        # One grid build to interpolate later
        dc_grid = atools.dcfun_at(zgrid_, H0_, Om_, w0_, interp=pade)
        dL_grid = atools.dLfun_at(zgrid_, H0_, Om_, w0_, Xi0_, nXi0_, interp=pade, dc=dc_grid, param=param)
        log_ddL_dz_grid = atools.log_ddL_dz(zgrid_, H0_, Om_, w0_, Xi0_, nXi0_, dc=dc_grid, interp=pade, param=param)

        

        # Precompute mass function pieces 

        if interp_mass!=0:

            if mass_model == "PLPreg":

                # ---------------------------
                # 1) m2 grid: log-cluster taper
                # ---------------------------
                eps_m = 1e-5
                n2 = 500
                n2_taper = 150
            
                m2_lo = ml_ + eps_m
                m2_taper_hi = m2_lo + at.maximum(deltam_, 1e-6)
            
                u1 = at.linspace(0.0, 1.0, n2_taper)
                eps_t = 1e-4
                t1 = at.exp(at.log(eps_t) * (1.0 - u1))
                t1 = (t1 - eps_t) / (1.0 - eps_t)
                seg1 = m2_lo + (m2_taper_hi - m2_lo) * t1
            
                u2 = at.linspace(0.0, 1.0, n2 - n2_taper)
                seg2 = m2_taper_hi + (mh_ - m2_taper_hi) * u2
            
                m2_grid_ = at.as_tensor_variable(at.concatenate([seg1[:-1], seg2]))
            
                # ---------------------------
                # 2) m1 grid: adaptive PLPreg grid (should include taper+ramp like DPLDP)
                # ---------------------------
                m1_grid_ = atools.build_m1_grid_PLPreg(
                    ml=ml_,
                    mh=mh_,
                    muMass=muM_,
                    sigmaMass=sM_,
                    deltam=deltam_,
                    n_peak=interp_mass,
                    n_tail_low=interp_mass // 5,
                    n_tail_high=interp_mass // 5,
                    n_taper=interp_mass // 5,
                )
            
                # ---------------------------
                # 3) Bank logpdfs
                # ---------------------------
                lp_m1_grid = atools.logpdfm1_PLP_reg(
                    m1_grid_,
                    lamP_, alpha_, deltam_,
                    ml_, mh_,
                    muM_, sM_,
                    smoothing=smoothing,
                )
            
                lp_m2_grid = atools.logpdfm2_PLP_reg(
                    m2_grid_,
                    beta_, deltam_, ml_,
                    smoothing=smoothing,
                )
            
                # ---------------------------
                # 4) logC(m1): stable CDF + nonuniform interp
                # ---------------------------
                lp2_max = at.max(lp_m2_grid)
                p2_shift = at.exp(lp_m2_grid - lp2_max)
            
                cdf_shift = atools.atcumtrapz(p2_shift, m2_grid_)  # (N2-1,)
                cdf_shift = at.clip(cdf_shift, 1e-300, np.inf)
            
                m2_cdf_grid = m2_grid_[1:]
                logcdf_m2 = at.log(cdf_shift) + lp2_max
            
                mcap = at.clip(m1_grid_, m2_cdf_grid[0], m2_cdf_grid[-1])
                lC_of_m1 = atools.interp_logpdf_1d_nonuniform(mcap, m2_cdf_grid, logcdf_m2)
            
                # ---------------------------
                # 5) stable ln(m1)
                # ---------------------------
                lp1_max = at.max(lp_m1_grid)
                p1_shift = at.exp(lp_m1_grid - lp1_max)
                I1 = atools.attrapzvec(p1_shift, m1_grid_)
                I1 = at.clip(I1, 1e-300, np.inf)
                ln = at.log(I1) + lp1_max
            
                # Pack
                interp_vals_mass  = [lp_m1_grid, lp_m2_grid, lC_of_m1, ln]
                interp_grids_mass = [m1_grid_, m2_grid_]

            
            elif mass_model=='DPLDP':

                    
                eps_m = 1e-5
                n2 = 500
                n2_taper = 100
                
                m2_lo = m2_low_ + eps_m
                m2_taper_hi = m2_lo + at.maximum(delta_m2_, 1e-6)
                
                u1 = at.linspace(0.0, 1.0, n2_taper)
                
                eps_t = 1e-4
                t = at.exp(at.log(eps_t) * (1.0 - u1))     # eps_t -> 1
                t = (t - eps_t) / (1.0 - eps_t)            # -> [0,1]
                seg1 = m2_lo + (m2_taper_hi - m2_lo) * t
                
                u2 = at.linspace(0.0, 1.0, n2 - n2_taper)
                seg2 = m2_taper_hi + (300.0 - m2_taper_hi) * u2
                
                m2_grid_ = at.as_tensor_variable(at.concatenate([seg1[:-1], seg2]))


            
                m1_grid_ = atools.build_m1_grid_DPLDP(
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
                
                lp_m1_grid = atools.logpdfm1_DPLDP( m1_grid_, alpha1_, alpha2_, mb_, mu1_, sigma1_, mu2_, sigma2_, m1_low_, m_high_, delta_m1_, lambda0_, lambda1_, epsilon_,  smoothing=smoothing) 


                lp_m2_grid = atools.logpdfm2_PLP_reg( m2_grid_, beta_, delta_m2_, m2_low_, m_g=m_g_, w_g=w_g_, sig_g_low = sig_g_l_, sig_g_high = sig_g_h_, has_m2_break=has_m2_break, smoothing=smoothing ) 


                # CDF over m2
                cdf_m2 = atools.atcumtrapz(at.exp(lp_m2_grid), m2_grid_)
                cdf_m2 = at.clip(cdf_m2, 1e-300, np.inf)
                
                # CDF lives on m2_grid_[1:]
                m2_cdf_grid = m2_grid_[1:]
                logcdf_m2   = at.log(cdf_m2)
                
                # C(m1) = CDF evaluated at m2=m1 (clipped into CDF grid support)
                mcap = at.clip(m1_grid_, m2_cdf_grid[0], m2_cdf_grid[-1])
                
                # NON-UNIFORM interpolation (must match your test)
                lC_of_m1 = atools.interp_logpdf_1d_nonuniform(mcap, m2_cdf_grid, logcdf_m2)
                
                # Normalization for m1
                #p1 = at.exp(lp_m1_grid)
                #ln = at.log(atools.attrapzvec(p1, m1_grid_))
                lp_max = at.max(lp_m1_grid)
                p_shift = at.exp(lp_m1_grid - lp_max)
                I = atools.attrapzvec(p_shift, m1_grid_)
                I = at.clip(I, 1e-300, np.inf)
                ln = at.log(I) + lp_max
                
                # Pack for later use
                interp_vals_mass  = [lp_m1_grid, lp_m2_grid, lC_of_m1, ln]
                interp_grids_mass = [m1_grid_, m2_grid_]

            elif mass_model=='DPLDP-z':

                eps_m = 1e-5 
                n2 = 500
                n2_taper = 100
                
                m2_lo = m2_low_ + eps_m
                m2_taper_hi = m2_lo + at.maximum(delta_m2_, 1e-6)
                
                u1 = at.linspace(0.0, 1.0, n2_taper)
                
                eps_t = 1e-4
                t = at.exp(at.log(eps_t) * (1.0 - u1))     # eps_t -> 1
                t = (t - eps_t) / (1.0 - eps_t)            # -> [0,1]
                seg1 = m2_lo + (m2_taper_hi - m2_lo) * t
                
                u2 = at.linspace(0.0, 1.0, n2 - n2_taper)
                seg2 = m2_taper_hi + (300.0 - m2_taper_hi) * u2
                
                m2_grid_ = at.as_tensor_variable(at.concatenate([seg1[:-1], seg2]))

                m1_grid_ =  atools.build_m1_grid_DPLDP_z( zgrid_mass_,
                                # low-z hyperparameters
                                mu1_0, sigma1_0, mu2_0, sigma2_0, mb_0,
                                # high-z (asymptotic) hyperparameters
                                mu1_inf_, sigma1_inf_, mu2_inf_, sigma2_inf_, mb_inf_,
                                # evolution hyperparameters
                                z_mu1_, dz_mu1_,
                                z_sigma1_, dz_sigma1_,
                                z_mu2_, dz_mu2_,
                                z_sigma2_, dz_sigma2_,
                                z_mb_, dz_mb_,
                                # support for m1
                                m1_low_, m_high_,
                                delta_m1_,
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
                lp_m2_grid = atools.logpdfm2_PLP_reg(
                    m2_grid_, beta_, delta_m2_, m2_low_,
                    m_g=m_g_, w_g=w_g_, sig_g_low=sig_g_l_, sig_g_high=sig_g_h_,
                    has_m2_break=has_m2_break, smoothing=smoothing
                )  # shape (N2,)
            
                # lC_grid evaluated on m1_grid (shape (N1,))
                cdf_m2 = atools.atcumtrapz(at.exp(lp_m2_grid), m2_grid_)
                cdf_m2 = at.clip(cdf_m2, 1e-300, np.inf)

                # CDF lives on m2_grid_[1:]
                m2_cdf_grid = m2_grid_[1:]
                logcdf_m2   = at.log(cdf_m2)
                
                # C(m1) = CDF evaluated at m2=m1 (clipped into CDF grid support)
                mcap = at.clip(m1_grid_, m2_cdf_grid[0], m2_cdf_grid[-1])
                
                # NON-UNIFORM interpolation (must match your test)
                lC_of_m1 = atools.interp_logpdf_1d_nonuniform(mcap, m2_cdf_grid, logcdf_m2)
                

                # ---------
                # 2) Bank lp_m1(z_k, m1_grid_) and ln(z_k)
                # ---------
                K  = zgrid_mass_.shape[0]
                N1 = m1_grid_.shape[0]
                
                M = at.broadcast_to(m1_grid_[None, :], (K, N1))
                Z = at.broadcast_to(zgrid_mass_[:, None],   (K, N1))
                
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
                # at.clip( lp_flat, -1e30, 1e030 )
                lp_m1_bank = at.clip( lp_flat, -1e30, 1e030 ).reshape((K, N1)) # (K,N1)

                #ln_bank = at.log( atools.attrapzvec(at.exp(lp_m1_bank), m1_grid_, axis=1))
                lp_max = at.max(lp_m1_bank, axis=1, keepdims=True)          # (K,1)
                p_shift = at.exp(lp_m1_bank - lp_max)                       # safe exp
                I = atools.attrapzvec(p_shift, m1_grid_, axis=1)            # (K,)
                I = at.clip(I, 1e-300, np.inf)
                ln_bank = at.log(I) + lp_max[:, 0]
             
                # Pack for later use (include z_bank)
                interp_vals_mass  = [lp_m1_bank, lp_m2_grid, lC_of_m1, ln_bank, ]
                interp_grids_mass = [m1_grid_, m2_grid_, zgrid_mass_]
                
            elif mass_model in ('DPUC', 'DP'):

                n_total_min_mass = interp_mass
                n_per_min = 20
                #n_per_max = 50
                #n_boundary_mass = 50
                k_sigma = 4
                frac_uniform=0.1
                sigma_floor=1e-4


                mu1_lo = MMIN_GRID #min(lowmu1, lowmu1_inj)
                mu1_hi = MMAX_GRID #max(upmu1,  upmu1_inj)
                
                mu2_lo = MMIN_GRID_1 #min(lowmu2, lowmu2_inj)
                mu2_hi = MMAX_GRID_1 #max(upmu2,  upmu2_inj)
                
                # Choose n_eps: 6 is extremely safe (Gaussian tail ~ 1e-9 one-sided)
                n_eps = 6.0
                
                # Bounds for logMc grid
                XLOW1, XHIGH1, sig1_max = putils.safe_interp_bounds_from_tau_eps(
                    mu_low=mu1_lo,
                    mu_high=mu1_hi,
                    tau_upper=U1,       # <-- upper bound of tau1 Uniform
                    s_local=s_local,
                    k_sigma=k_sigma,
                    n_eps=n_eps,
                    extra_frac=0.10,
                )
                
                # Bounds for logit(q) grid
                XLOW2, XHIGH2, sig2_max = putils.safe_interp_bounds_from_tau_eps(
                    mu_low=mu2_lo,
                    mu_high=mu2_hi,
                    tau_upper=U2,       # <-- upper bound of tau2 Uniform
                    s_local=s_local,
                    k_sigma=k_sigma,
                    n_eps=n_eps,
                    extra_frac=0.10,
                )
                
                # OPTIONAL: clip to global physical transform limits (recommended)
                # these should be your transform-domain hard limits (constants)
                #XLOW1  = at.maximum(XLOW1,  at.as_tensor_variable(MMIN_GRID))
                #XHIGH1 = at.minimum(XHIGH1, at.as_tensor_variable(MMAX_GRID))
                
                #XLOW2  = at.maximum(XLOW2,  at.as_tensor_variable(MMIN_GRID_1))
                #XHIGH2 = at.minimum(XHIGH2, at.as_tensor_variable(MMAX_GRID_1))

                
    

                log_Mc_grid = atools.build_1d_gaussian_mixture_grid_components(
                                                mu1, sig1,
                                                XLOW1, XHIGH1,
                                                n_total_min=n_total_min_mass,
                                                frac_uniform=frac_uniform,
                                                k_sigma=k_sigma,
                                                sigma_floor=sigma_floor,
                                                n_per_min = n_per_min,
                                                K = N_DP_comp_max_np
                                            )

                logit_q_grid = atools.build_1d_gaussian_mixture_grid_components(
                                                mu2, sig2,
                                               XLOW2, XHIGH2,
                                                n_total_min=n_total_min_mass,
                                                frac_uniform=frac_uniform,
                                                k_sigma=k_sigma,
                                                sigma_floor=sigma_floor,
                                                n_per_min=n_per_min, 
                                                K = N_DP_comp_max_np
                                            )


                print("grid m1 range:", log_Mc_grid[0].eval(), log_Mc_grid[-1].eval())
                print("mu prior range:", lowmu1, upmu1)
                print("sigma upper used:", sig1_max.eval())


                print("grid logit q range:", logit_q_grid[0].eval(), logit_q_grid[-1].eval())
                print("mu prior range:", lowmu2, upmu2)
                print("sigma upper used:", sig2_max.eval())
                

                if rate_model in ('MD', 'PL'):
                    lp_Mc_grid, lp_q_grid, lp_z_grid = atools.gaussian_logpdf_pair(log_Mc_grid, logit_q_grid, mu, sd)


                    interp_vals_mass  = [lp_Mc_grid, lp_q_grid]
                    interp_grids_mass = [log_Mc_grid, logit_q_grid]

                else:


                    mu3_lo = MMIN_GRID_2 #min(lowmu3, lowmu3_inj)
                    mu3_hi = MMAX_GRID_2 #max(upmu3,  upmu3_inj)

                    # Bounds for logit(q) grid
                    XLOW3, XHIGH3, sig3_max = putils.safe_interp_bounds_from_tau_eps(
                        mu_low=mu3_lo,
                        mu_high=mu3_hi,
                        tau_upper=U3,       # <-- upper bound of tau2 Uniform
                        s_local=s_local,
                        k_sigma=k_sigma,
                        n_eps=n_eps,
                        extra_frac=0.10,
                    )
                    
                    # OPTIONAL: clip to global physical transform limits (recommended)
                    # these should be your transform-domain hard limits (constants)
                    #XLOW3  = at.maximum(XLOW3,  at.as_tensor_variable(MMIN_GRID_2))
                    #XHIGH3 = at.minimum(XHIGH3, at.as_tensor_variable(MMAX_GRID_2))

                
                    log_1pz_grid = atools.build_1d_gaussian_mixture_grid_components(
                                                mu3, sig3,
                                                XLOW3, XHIGH3,
                                                n_total_min=n_total_min_mass,                                                                                    frac_uniform=frac_uniform,
                                                k_sigma=k_sigma,
                                                sigma_floor=sigma_floor,
                                                n_per_min=n_per_min, 
                                                K = N_DP_comp_max_np
                                            )

                    print("grid log(1+z) range:", log_1pz_grid[0].eval(), log_1pz_grid[-1].eval())
                    print("mu prior range:", lowmu3, upmu3)
                    print("sigma upper used:", sig3_max.eval())

           
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
    
                x = pm.Normal( 'x', mu=0, sigma=1, dims= ("event_index" , "GWdimension" ), initval = (np.random.randn(N, nd) * eps_init)) #.astype(X) )    
    
                    
                if 'gmm' in sampling_GW:
            
                    print('Sampling m1d, m2d, dL from GMM')
        
                        
                    ###################################
                    # categorical way

                    ig = pm.Categorical('idx', p=wts_l, dims= "event_index",  initval=at.argmax(wts_l, axis=1)) #.astype(int_dtype) )
        
           
                    # Select means and Cholesky factors per batch
                    mu_selected = mus_l[ np.arange(N), ig, :]         # shape (N, D)
                    L_selected = cho_covs_l[ np.arange(N), ig, :, :]  # shape (N, D, D)
                     
                    # Batched matrix multiplication: (N, D, D) @ (N, D, 1) → (N, D, 1)
                    Lx = at.sum(L_selected * x[:, None, :], axis=2)  # → shape (N, D)
    
                              
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
                    #samples = mus_s + at.matmul(cho_s, x[..., None])[..., 0]      # (N, d)
                    samples = mus_s + at.sum(cho_s * x[:, None, :], axis=-1)

                  
                
                    # logp = log p(x) - log|L|
                    # d = x.shape[1]
                    log_px = -0.5 * at.sum(x**2, axis=1) - 0.5 * nd_np * at.log(2*np.pi)    # (N,)


                    log_det_L = at.sum(at.log(at.diagonal(cho_s, axis1=1, axis2=2)), axis=1)  # (N,)

                    
                    pilik = log_px - log_det_L                                               # (N,)

                    # unpack coordinates:
                    log_Mc_det = samples[:, 0]
                    logit_q    = samples[:, 1]
                    logd       = samples[:, 2]
                    
    
                    if spin_model == 'none' :
                        
                        X = at.stack([log_Mc_det, logit_q, logd ], axis=1)
                        d_int  = 3
    
    
                    elif spin_model == 'default' or spin_model == 'default_gauss':
    
                        chi1 = atools.inv_logitat(samples[:,3])
                        chi2 = atools.inv_logitat(samples[:,4])
            
                        cost1 = atools.inv_flogitat(samples[:,5])
                        cost2 = atools.inv_flogitat(samples[:,6])
    
                        X = at.stack([log_Mc_det, logit_q, logd,  samples[:,3],  samples[:,4],  samples[:,5],  samples[:,6]], axis=1)
                        d_int  = 7
    
    
                
    
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
                    log_norm = (-0.5 * d_int * at.log(2*np.pi)) #.astype(work_dtype)     # scalar
     
                    logp_components = (
                        -0.5 * quad
                        + log_norm
                        - 0.5 * log_dets_l
                        + log_wts_l
                    )                                             # (N, ngmm)

                    # Mixture log-likelihood per observation: logsumexp over components
                    gwl = at.logsumexp(logp_components, axis=1, )   # (N,)

                
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
                
                logd = at.log(d)
            
            
            ################################################
            # Population prior
            ################################################

    
            
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
    
            is_DP = mass_model in ('DP', 'DPUC')
            # Population prior of all events, without the term T_obs*R0
            if is_DP:
    
                # dirichelet processs will be for log(Mc_src), logit(q) ...
                logMc_src =  log_Mc_det - at.log1p(zs)
                
                log_p_pop = log_p_pop_at( logMc_src, logit_q, zs, d, spins, Lambda_, rate_model, mass_model, spin_model,  dc=dc,  log_ddL_dz_pre=log_ddL_dz, z_grid = zgrid_ )
                
                
                # ... so remove a jacobian : p( m1, m2 ) = p( log(Mc), logit(q) ) * |J|
                # if using interpolation, the jacobian is already included in the grid.
                print("Likelihood: removing jacobian m1, m2 --> log(Mc), logit(q) ")
                
                eps = at.as_tensor_variable(1e-12, dtype=m2src.dtype)
                log_p_pop -=  at.log(m2src) + at.log(at.maximum(m1src - m2src, eps))#+at.log1p(zs)
    
                if rate_model in ('DPUC','DPUC-vol' ):
                    # also remove jacobian for log(1+z)
                    log_p_pop -= at.log1p(zs) 
                    
                
            else:    
            
                log_p_pop = log_p_pop_at( m1src, 
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
                                           z_grid = zgrid_,
                                           #K=N_DP_comp_max
                                         )
                        
        
        
        else:
            # sample_from_pop=1
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
            lpmass = at.logsumexp(logp_components + logw[:, None], axis=0 )


            # compute GW likelihood in det. frame

            log_Mc_det = logMc+y
            d = atools.dLfun_at(z, H0_, Om_, w0_, Xi0_, nXi0_, param=param)
            logd = at.log( d )


            X = at.stack([log_Mc_det, logit_q, logd ], axis=1)
            d_int  = 3


            diff = X[:, None, :] - mus_l[:, :, :d_int]                  # (N, 1, d) - (N, ngmm, d)
 
            tmp = at.matmul(icovs_l[:, :, :d_int, :d_int], diff[..., None])[..., 0]   # (N, ngmm, d)
            
            quad = at.sum(diff * tmp, axis=-1)            # (N, ngmm)
            
            log_norm = -0.5 * d_int * at.log(2.0 * np.pi)     # scalar
            logp_components = (
                -0.5 * quad
                + log_norm
                - 0.5 * log_dets_l
                + log_wts_l
            )                                             # (N, ngmm)
            
            gwl = at.logsumexp(logp_components, axis=1, )   # (N,)

            # jacobian
            log_jac_q = -at.log(q) - at.log1p(-q)

            # all
            log_p_pop = lpmass + gwl - log_Mc_det - logd - log_jac_q
        



        if is_observed:
    
            print("Fitting for observed population. Removing factor 1/Pdet")

            Theta = at.ones(d.shape)
          
            log_P_det = at.log( atools.Pdet( osnr_interp_at, m1det, m2det, d, Theta, at.as_tensor_variable(8.) )
                                       )
            log_p_pop -= log_P_det

            
                
        if all_dLsq_prior:
            #dLprior=='dLsq':
            # Remove \pi(d)~dL^2 prior on distance 
            log_p_pop -= 2*logd
            print('Removing dL^2 prior for all events.')

        elif all_no_dL_prior:    
            print("No dL prior removed for all events.")
            
        elif vol_in_prior:

            zs_Planck15 = atools.atinterp(d, dL_grid_Planck15, zgrid_dLp)
            dc_Planck15 = atools.dcfun_at(zs_Planck15, 67.90, 0.3065, -1., interp=False)

            lpi = at.zeros_like(log_p_pop )

            # apply chunk-wise prior removal
            for i, lab in enumerate(dLprior):
                sl = slice(edges[i], edges[i+1])

                print('For events between %s and %s, removing prior %s'%(edges[i], edges[i+1], lab))


                if lab == 'dLsq':
                    print('chunk is dLsq')
                    print(sl)
                    chunk = 2 * logd[sl]

                elif lab == 'none':
                    print('chunk is zero')
                    print(sl)
                    chunk = at.zeros_like(log_p_pop[sl])

                else:
                    # base label + whether we apply the -J correction
                    use_J = lab.endswith('-J')
                    base = lab[:-2] if use_J else lab

                    if base == 'UniformComovingVolume':
                        print('chunk is UniformComovingVolume')
                        print(sl)
                        chunk = atools.log_dV_dz_at(
                            zs_Planck15[sl], 67.90, 0.3065, -1., dc=dc_Planck15[sl]
                        )

                    elif base == 'UniformSourceFrame':
                        print('chunk is UniformSourceFrame')
                        print(sl)
                        chunk = (
                            atools.log_dV_dz_at(
                                zs_Planck15[sl], 67.90, 0.3065, -1., dc=dc_Planck15[sl]
                            )
                            - at.log1p(zs_Planck15[sl])
                        )

                    else:
                        raise ValueError(f"Unknown dL prior label: {lab}")

                    if use_J:
                        print('removing log_ddL_dz ')
                        chunk -= atools.log_ddL_dz(
                            zs_Planck15[sl], 67.90, 0.3065, -1., 1., 0.,
                            dc=dc_Planck15[sl], interp=False, param='vanilla'
                        )

                lpi = at.set_subtensor(lpi[sl], chunk)
            
            log_p_pop -= lpi
            

        
        
        else:
            raise ValueError("Check dL prior choices.")
            

        if not pop_only:
            if sampling_GW=='gauss' and not sample_from_pop:
                # Add gw likelihood and correct for sampling prior pdf
                log_p_pop -= pilik
                log_p_pop += gwl

            
            # just sum log likelihoods

            likelihood_val = at.sum( log_p_pop )

        
        else:
            # marginalise over single events parameters first
            # shape of p_pop is (hopefully) n_evs x n_samples
            # so average over second dimension
            
            # Compute only where there are samples
            log_p_pop_to_marg = log_p_pop[:, :allNsamples[0]]
            
            log_p_pop_marg = at.logsumexp( log_p_pop_to_marg, axis=1, ) - at.log(allNsamples)
            

            # then sum log likelihoods
            likelihood_val = at.sum( log_p_pop_marg )  

            # Check number of effective samples for computing MC integral 
            logs2 = at.logsumexp(2*log_p_pop_masked, axis=1,) -2*at.log(allNsamples)
            
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
            poiss_term = at.sum(Nevs*at.log(allTobs))+N*lR0
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
                    if inj_loop=='scan':
                        sel_bias_fun = sel_bias_with_uncertainty_at_0_batched_scan
                        print("Using version with pytensor scan in batches")
                        print('Chunk size is %s'%chunk_inj)
                    elif inj_loop=='scan-GPU':
                        sel_bias_fun = sel_bias_with_uncertainty_at_0_batched_scan_GPU
                        print("Using version with pytensor scan in batches optimized for GPU")
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
                                                          log_ddL_dz_inj = log_ddL_dz_inj,
                                                            zinj = zinj,
                                                            dcinj = dcinj,
                                                          param=param,
                                                          interp_vals_mass = interp_vals_mass,
                                                           interp_grids_mass = interp_grids_mass,
                                                        
                                                        )

                if debug_sel_batch:
                    
                    zinj_tmp_ = atools.atinterp(dLinj[0], dL_grid, zgrid_)
    
                    
                    log_mu_1, Neff_1, var_ll_u_1 = sel_bias_with_uncertainty_at_0( m1inj[0], m2inj[0], dLinj[0], spinsInj, lpdinj[0], 
                                                              Lambda_, 
                                                              Ndraw, 
                                                              rate_model, mass_model, spin_model_name, 
                                                              smoothing, 
                                                            False,
                                                              has_m2_break, 
                                                              interp=pade, 
                                                             log_p_incl = None,
                                                            log_ddL_dz_inj = atools.atinterp( zinj_tmp_, zgrid_, log_ddL_dz_grid),
                                                             zinj = zinj_tmp_,
                                                             dcinj = atools.atinterp( zinj_tmp_, zgrid_, dc_grid) ,
                                                            )
    
                    
                    
                    print("Difference in log_mu_1 :")
                    print((log_mu_1 - log_mu_).eval().max())
    
                    print("Difference in var_ll_u_1 :")
                    print((var_ll_u_1 - var_ll_u_).eval().max())
                    
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
                                          sequences = [ at.arange( ndata) ], 
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
                                      sequences = [ at.arange( ndata) ], 
                                      non_sequences = [m1inj, m2inj, dLinj, spinsInj, lpdinj,  Lambda_,  Ndraw], 
                                            profile=True)

            
                    log_mu_ = res_i[0]
                    Neff_ = at.sum(res_i[1])
    

                
    
                if not marginal_R0:
                    # Sum number of expected events in the two observing runs
                    # p_pop does not contain R_0*Tobs . Add it here
                    sel_effect = -at.sum(at.exp(log_mu_+lR0+at.log(Tobs)))
                else:
                    if sel_method=='Tobs':
                        sel_effect = -N*at.logsumexp( at.log(Tobs/Ttot)+log_mu_ )
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
                log_lik_var = pm.Deterministic('log_lik_var', at.exp(var_ll_u_+2*logN ) )
            else:
                log_lik_var = pm.Deterministic('log_lik_var', at.exp(  var_ll_u_+2*at.log( R0*Ttot ) + 2*log_mu_ ) )
            
     

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
                        
                        selection_bias = sel_effect + atools.logdiffexp( at.log(1), atools.log_sigmoid(log_lik_var, log_lik_var_min*(1+0.002), 0.001 )) 

                    
                    elif sel_smoothing=='poly':
                        print("Tapering sel effect with polynomial smoothing")

                        
                        #selection_bias = sel_effect + atools.logdiffexp( at.log(1), atools.log_f_smooth_poly(log_lik_var, 0.01,  log_lik_var_min*(1-0.005) ))  

                        selection_bias = sel_effect
                        _ = pm.Potential("bound_log_lik_var", atools.logS_PLP(log_lik_var_min - log_lik_var, deltam=0.01, ml=-0.01))


                        
                    elif sel_smoothing=='softplus':
                        print("Tapering sel effect with softplus")
                        # Slack (how sharp the corner is) and weight (penalty strength)
                        nu = 0.01    # smaller = sharper transition
                        lam = 1.    # larger = stronger penalty
                        
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
                        _ = pm.Potential("bound_log_lik_var", at.switch(log_lik_var <= log_lik_var_min, 0.0, -np.inf ))

            
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

