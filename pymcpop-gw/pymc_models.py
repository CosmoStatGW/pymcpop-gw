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


PLPeakO3params = {'H0': 67.66, 'Om':0.31, 'w0':-1, 'Xi0': 1, 'nXi0':0}

from tqdm import tqdm



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

    elif spin_model=='chieffchip_uc':
        
        muE, sigE, muP, sigP = Lambda[istart:istart+4]
        chieff, chip = spins[0], spins[1]

        lpchie = atools.truncGausslowerupper_at_lpdf(chieff, muE, sigE, xmin=at.as_tensor_variable(-1), xmax=at.as_tensor_variable(1))
        lpchip = atools.truncGausslowerupper_at_lpdf(chip, muP, sigP, xmin=at.as_tensor_variable(0), xmax=at.as_tensor_variable(1))

        lpspin = lpchie+lpchip

    elif spin_model=='default':

        alphaChi, betaChi, zeta, sigmat = Lambda[istart:istart+4]
        lpspin = atools.logpdf_default_spin(spins, [alphaChi, betaChi, zeta, sigmat])
    
    elif spin_model=='default_gauss':
        muChi, sigmaChi, zeta, sigmat = Lambda[istart:istart+4]
        lpspin = atools.logpdf_default_spin_gauss(spins, [muChi, sigmaChi, zeta, sigmat])
   
    else:
        lpspin = at.zeros( z.shape )

    
    ###################################
    # mass

    ### BBH
    if mass_model=='PLPreg':
        
        lp, al, bb, dm, ml, mh, muM, sM = Lambda[-8:]
        lpmass = atools.logpdf_PLP_reg([m1s, m2s], [lp, al, bb, dm, ml, mh, muM, sM], smoothing=smoothing)

    elif mass_model=='DPLDP':
        
        lambdaBBHmass = Lambda[-20:]
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
            
            zint_ = at.exp(log_p_z_MD_unnorm(zgrid_, gamma, kappa, zp, Lambda_c, dc=dc))
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


    if 'DP' in mass_model:
        # remove jacobian m1, m2 --> log(Mc), logit(q)
        log_p_pop += (- at.log(m2Src) - at.log(m1Src-m2Src) - at.log1p(zinj) )

    log_sel_b = log_p_pop-log_p_draw
  
    
    log_mu = at.logsumexp(log_sel_b) - at.log(Ndraw)
    
    logs2 = at.logsumexp(2.0*log_sel_b) - at.log(Ndraw)


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

    var_log_lik_u = atools.logdiffexp( logs2-2*log_mu, 1.) - at.log(Ndraw)

    Neff = at.exp(logNeff)
    
    
    return log_mu, Neff, var_log_lik_u
    

#####################################################



#####################################################
#####################################################


def make_model(  priors,
                 GWData,
                 InjData,
                 ivals={},
                 sampling_GW = 'gmm',
                 rate_model = 'MD',
                 mass_model = 'PLP',
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
                 monotonicity_scale = 1. ,
                 zmin_mono = 0, 
                nu = 0.25,
                 lam = 10,
                 clip_low = -500,
                 clip_high=500,
               GP_prior = 'gammainv',
               GP_zero_point = 'y',
               rescale_GP=False,
               invert_dL_GP = True,
               dense_grad = False,
                 fix_H0 = True,
                fix_Om = True,
               fix_w0 = True,
                 fix_Xi0n = True,
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
                 U = 10.
                ):

    ################################################
    # Read in data and set dimensions
    ################################################

    ## GW data
    if not pop_only:
        
        # gw data are interpolants of single-event posteriors
        if sampling_GW=='gauss' :
            # we sample single-event parameters from broad gaussian approximations of the posteriors
            mus_s, cho_s, log_wts_l, mus_l, icovs_l, log_dets_l, Tobs, Nevs = GWData
        
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
        print("N samples max will be ")
        print(Nsamples_np)
        print('N:%s, n samples: %s '%(N_np, allNsamples_np))




    
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

    if is_GP_dL:

        
        U = at.as_tensor_variable( U ) #2.5)         # upper bound for σ with high probability
        
        alpha = at.as_tensor_variable(0.01)    # small tail probability
        lambda_ = at.log(1 / alpha) / U
        
        alpha_ell = at.as_tensor_variable(0.005)
        alpha_large = at.as_tensor_variable(0.01)
        
        d_GP = at.as_tensor_variable(1)

        if find_GP_L:
            rng = np.random.default_rng(123)
            print("Finding min prior lengthscale for GP...")
            allL = []
            allM = []

            # --- Compile once: z_from_dL and midpoint derivative ---
            z_sym      = at.dvector('z_nodes')    # if you need it
            d_sym      = at.dvector('dL_nodes')
            H0_sym     = at.dscalar('H0')
            Om_sym     = at.dscalar('Om')
            #w0_sym     = at.dscalar('w0')
            w0_const = at.as_tensor_variable(-1.0)
            
            # your existing functions but returning NODE arrays
            z_from_dL_sym = atools.z_from_dL_at(d_sym, H0_sym, Om_sym, w0_const, [1, 0.], False, data_range=None)
            dc_nodes_sym  = atools.dcfun_at(z_sym, H0_sym, Om_sym, w0_const, interp=False)
            d_log_dLEM_dz_sym = atools.ddL_dz_EM(z_sym, H0_sym, Om_sym, w0_const)
            #d_log_dLEM_dz(z_sym, H0_sym, Om_sym, w0_const)

            lb_mid_fn = pytensor.function([z_sym, H0_sym, Om_sym, ], d_log_dLEM_dz_sym)
            z_from_dL_fn = pytensor.function([d_sym, H0_sym, Om_sym, ], z_from_dL_sym)

            # --- Helper to draw from the GMM in NumPy only ---
            def sample_from_per_event_gmm(wts, mus, chol_covs, Xwhite):
                """
                wts : (N, K)
                mus : (N, K, D)
                chol_covs : (N, K, D, D) lower-tri
                Xwhite : (N, D) standard normals
                """
                N, K = wts.shape
                u = rng.random((N, 1))
                cdf = np.cumsum(wts, axis=1)
                k = (u < cdf).argmax(axis=1)            # (N,)
                rows = np.arange(N)
                # draw one component per event with provided white noise Xwhite
                return mus[rows, k, :] + (chol_covs[rows, k, :, :] @ Xwhite[..., None]).squeeze(-1)  # (N, D)

            def robust_stat(x, trim=0.05):
                """Trimmed median absolute for stability."""
                x = np.asarray(x, dtype=np.float64).ravel()
                a, b = np.quantile(x, [trim, 1-trim])
                x = x[(x>=a) & (x<=b)]
                return np.median(x)


            def find_init_hyperparams(wts_l_np, mus_l_np, cho_covs_l_np,
                                      H0_range, Om_range,
                                      N, nd, rescale_GP=False, dmin=None, dmax=None,
                                      trials=50, s0=0.10):
                L_list = []
                M_list = []
                ell_list = []
                dz_all = [] 
                ell_maxs = []
                zmaxs = []
            
                for _ in tqdm(range(trials)):
                    Xwhite = rng.standard_normal((N, nd))

                    if 'gmm' in sampling_GW:
                        samples = sample_from_per_event_gmm(wts_l_np, mus_l_np, cho_covs_l_np, Xwhite)
                    elif sampling_GW=='gauss':
                        samples = mus_l_np + (cho_covs_l_np @ Xwhite[..., None]).squeeze(-1)
            
                    d_nodes = np.exp(samples[:, 2])             # your distance column
                    if rescale_GP:
                        # min-max to provided data_range
                        d_nodes = ( (d_nodes - d_nodes.min()) / (d_nodes.max() - d_nodes.min() + 1e-12) ) * (dmax - dmin) + dmin
            
                    H0 = rng.uniform(*H0_range)
                    Om = rng.uniform(*Om_range)
                    w0 = -1.
            
                    # z from dL via compiled function (returns NumPy)
                    z_nodes = z_from_dL_fn(d_nodes.astype(np.float64), float(H0), float(Om), )

                    z_max_mon_ = np.quantile( z_from_dL_fn( np.squeeze(dLinj).astype(np.float64), float(H0), float(Om), ), 0.99 )
                    
                    z_nodes = np.asarray(z_nodes, dtype=np.float64)
                    z_nodes.sort()
                    # enforce strictly increasing
                    z_nodes = z_nodes[np.insert(np.diff(z_nodes) > 0, 0, True)]
            
                    if z_nodes.size < 3:
                        continue
            
                    # characteristic spacing (robust)
                    L_list.append( np.mean(np.diff(z_nodes)))  #robust_stat(np.diff(z_nodes)))
            
                    # midpoint derivative magnitude
                    lb_mid_pos = lb_mid_fn(z_nodes, float(H0), float(Om), )  # (N-1,)
                    lb_mid = np.asarray(lb_mid_pos, dtype=np.float64)
                    M_list.append(robust_stat(np.abs(lb_mid)))

                    z = z_nodes
                    # Drop (near-)duplicates to avoid zero spacings
                    tol = 1e-12
                    z = z[np.insert(np.diff(z) > tol, 0, True)]
                    
                    dz = np.diff(z)
                    dz_pos = dz[dz > tol]
                    dz_all.append(dz_pos)
                    
                    # Pick a conservative floor (1.5–3× min spacing). You can also use a small percentile.
                    c = 2.0
                    ell_min = float(c * np.min(dz_pos))
                    ell_list.append(ell_min)
                    

                    
                    z_span = np.quantile(z_nodes, 0.95) - np.quantile(z_nodes, 0.05)  # from the same mocks; ~2
                    #print(z_span)
                    ell_max = z_span    # or ell_max = 1.5 * z_span if you want a bit of extra room
                    ell_maxs.append(ell_max)

                    zmaxs.append(max(z_max_mon_, np.quantile(z_nodes, 0.99)))
                    
                if not L_list or not M_list:
                    raise RuntimeError("Could not gather stats for ℓ and ν; check data or ranges.")
            
                L = np.max(L_list)                # conservative min-lengthscale driver
                M = np.max(M_list)                # conservative slope scale
                #ell_min = np.max(ell_list)
                
                # ν so that ν * softplus(g) adds ~5% of typical |lb| when softplus(g)≈s0
                nu0 = float(np.clip(0.05 * M / s0, 1e-3, 0.2))
                # reasonable ℓ0 from spacing (use 5× median gap, or 0.2 of span)
                ell0 = float(max(5.0 * robust_stat(L_list), 0.2 * (np.max(z_nodes) - np.min(z_nodes))))
                # amplitude η0 moderate
                eta0 = 0.2

                dz_all = np.concatenate(dz_all)
                q_small = 0.95      # 50% quantile
                ell_min_data = np.quantile(dz_all, q_small)
                ell_min = 2.0 * ell_min_data   # even more conservative
                ell_max = 2*max(ell_maxs)

                z_max_mono = max(zmaxs)                     # or your z_max, e.g. max detected z


                

            
                return dict(L=L, M=M, nu0=nu0, ell0=ell0, eta0=eta0, ell_min=ell_min, ell_max=ell_max, z_max_mono=z_max_mono)


            # ---- call it (convert your shareds to NumPy once) ----
            if 'gmm' in sampling_GW:
                wts_np = np.asarray(wts_l, dtype=np.float64)
                mus_np = np.asarray(mus_l, dtype=np.float64)
                chol_np = np.asarray(cho_covs_l, dtype=np.float64)
            else:
                wts_np = None
                mus_np = np.asarray(mus_s, dtype=np.float64)
                chol_np = np.asarray(cho_s, dtype=np.float64)
                
            
            stats = find_init_hyperparams(wts_np, mus_np, chol_np,
                                          H0_range=priors['H0'], Om_range=priors['Om'], 
                                          N=int(N), nd=int(nd),
                                          rescale_GP=False, dmin=None, dmax=None,
                                          trials=500, s0=0.10)
            
            nu0  = stats["nu0"]
            ell0 = stats["ell0"]
            eta0 = stats["eta0"]
            ell_min = stats["ell_min"]
            ell_max = stats["ell_max"]
            
            #print(f"L (max spacing proxy): {stats['L']:.6g}")
            #print(f"M (max |lb| proxy):    {stats['M']:.6g}")
            #print(f"nu0:                   {nu0:.6g}")
            #print(f"ell0:                  {ell0:.6g}")
            #print(f"eta0:                  {eta0:.6g}")
            print(f"ell_min:                  {ell_min:.6g}")
            print(f"ell_max:                  {ell_max:.6g}")

            #L = stats["L"]
            #M = stats["M"]
            #ell_min = stats["ell_min"]

    
            beta = atools.find_beta(stats["L"], 2., p0=0.01)
            al = atools.find_al(stats["L"], 10., p0=0.01)

            
        else:
            raise ValueError()
        
        
        
        #print('L is %s'%stats["L"].eval())
        #print('M is %s'%stats["M"].eval())
        #print(f"Found beta: {beta:.4f}")
        #print(f"Found alpha: {al:.4f}")
        #cprint(f"Found nu0: {nu0:.4f}")
        #print(f"Mean length scale: {2 / beta:.4f}")
        
        #if True:
        lambda_ell = -at.log(alpha_ell) * ell_min**(d_GP / 2)
        print('lambda_ell is %s'%lambda_ell.eval())

        lambda_large = -np.log(alpha_large) / ell_max
        print('lambda_large is %s'%lambda_large.eval())

        z_max_mono =  stats["z_max_mono"]
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

        
    ################################################
    # Build model
    ################################################
    
    with pm.Model(coords=coords) as model:


        if sampling_GW=='gauss' :
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
            #
            H0_ =  pm.Uniform('H0', lower=priors['H0'][0], upper=priors['H0'][1], initval=ivals.get('H0'))
            #H0_ =  pm.Normal("H0", mu=70.0, sigma=2.0)


        
        if fix_Om:
            Om_ = params_fix['Om']
        else:
            Om_ = pm.Uniform('Om', lower=priors['Om'][0], upper=priors['Om'][1], initval=ivals.get('Om')) 
            #Om_ = pm.TruncatedNormal("Om", mu=0.25, sigma=0.05, lower=0.05, upper=0.6)

        if fix_w0:
            w0_ = -1.
        else:
            if pade:
                raise NotImplementedError("Pade appproximation with varying w0 not implemented yet. Use pade=False")
            w0_ =  pm.Uniform('w0', lower=priors['w0'][0], upper=priors['w0'][1], initval=ivals.get('w0'))
            
        

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
                print('Add large ℓ penalty')
                _ = pm.Potential(
                    "pc_large_ell",
                    -lambda_large * ℓ
                            )
            
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
            m_high_   = pm.Deterministic("m_high", at.as_tensor_variable(300.0).astype('float64')  )
            delta_m1_ = pm.Uniform("delta_m1", lower=priors["delta_m1"][0], upper=priors["delta_m1"][1], initval=ivals.get("delta_m1"))
            lambda_vec = pm.Dirichlet("lambda", a=np.array([1, 1, 1]), initval=ivals.get("lambda"))
            lambda0_  = pm.Deterministic("lambda0", lambda_vec[0])
            lambda1_  = pm.Deterministic("lambda1", lambda_vec[1])
            lambda2_  = pm.Deterministic("lambda2", lambda_vec[2])
            beta_     = pm.Uniform("beta",     lower=priors["beta"][0],     upper=priors["beta"][1],     initval=ivals.get("beta"))
            delta_m2_ = pm.Uniform("delta_m2", lower=priors["delta_m2"][0], upper=priors["delta_m2"][1], initval=ivals.get("delta_m2"))
            epsilon_  = pm.Deterministic("epsilon", at.as_tensor_variable(0.01))
            if has_m2_break:
                print("Including gap for secondary mass")
                m_g_     = at.as_tensor_variable(45.).astype('float64')
                w_g_     = at.as_tensor_variable(70.).astype('float64')
                sig_g_l_ = at.as_tensor_variable(1e-04)
                sig_g_h_ = at.as_tensor_variable(1e-04)
            else:
                m_g_     = at.as_tensor_variable(45.).astype('float64')
                w_g_     = at.as_tensor_variable(70.).astype('float64')
                sig_g_l_ = at.as_tensor_variable(1e-04)
                sig_g_h_ = at.as_tensor_variable(1e-04)
            
            Lambda_ += [alpha1_, alpha2_, mb_, mu1_, sigma1_, mu2_, sigma2_, m1_low_, m_high_, delta_m1_, lambda0_, lambda1_, beta_, m2_low_, delta_m2_, epsilon_, m_g_, w_g_, sig_g_l_, sig_g_h_]

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
    
            
            # Compute source-frame quantities. One redsfhit, mass1, mass2 for each event

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

                    # # we sampled distance from the posterior. need to invert the dL-z relation
                    # dc_grid = atools.dcfun_at(atools.zGridGlobals_at, H0_, Om_,  w0_, interp=False)
                    # b_full = atools.d_log_dLEM_dz(atools.zGridGlobals_at, H0_, Om_,  w0_ , dc=dc_grid, safe=False)
                    
                    # dLGrid_at, log_distance_ratio_grid, grad_log_distance_ratio_grid = atools.z_from_dL_at (None, H0_, Om_, w0_, Lambda_MG_ , is_GP_dL, data_range=data_range, GP_zero_point=GP_zero_point, dense_grad = dense_grad,  eta=η , ell=ℓ, nu=nu, sgn=sgn, b_full=b_full  )
    
                    # zs = pm.Deterministic('z', atools.atinterp( dval, dLGrid_at, atools.zGridGlobals_at ) , dims= "event_index" ) 

                    # dc = pm.Deterministic('dc', atools.dcfun_at(zs, H0_, Om_,  w0_, interp=False) , dims= "event_index" )
                    

                     
                    # distance_ratio = pm.Deterministic( "d_ratio", at.exp(atools.atinterp( zs, atools.zGridGlobals_at, log_distance_ratio_grid )), dims= "event_index")

                                        
                    # d_log_distance_ratio_d_z = atools.atinterp( zs, atools.zGridGlobals_at, grad_log_distance_ratio_grid )  

                    
                    # d_distance_ratio_d_z = pm.Deterministic( "d_ratio_d_z", d_log_distance_ratio_d_z*distance_ratio, dims= "event_index")

                    
                    # dLem_grid = (1+atools.zGridGlobals_at)*dc_grid

                    # ddLem_dz_grid =  atools.ddL_dz_EM( atools.zGridGlobals_at, H0_, Om_, w0_,  dc=dc_grid )
    
                    # distance_ratio_grid = pm.Deterministic( "d_ratio_grid",  at.exp(log_distance_ratio_grid) )

                    # dL_grid = pm.Deterministic( "dL_grid",  dLem_grid*distance_ratio_grid )
                
                    # s_grid = dLem_grid * grad_log_distance_ratio_grid + ddLem_dz_grid
                    # log_ddL_dz_grid = at.log( at.abs( s_grid * distance_ratio_grid ) )
                    # # log_ddL_dz_grid = at.log( at.abs( dLem_grid*grad_log_distance_ratio_grid*distance_ratio_grid + distance_ratio_grid*ddLem_dz_grid ) )

                    # ddL_dz_grid = pm.Deterministic( "ddL_dz_grid",  s_grid * distance_ratio_grid  )

                    
                    # log_ddL_dz = atools.atinterp( zs, atools.zGridGlobals_at, log_ddL_dz_grid )
                

                    # if monotonicity:

                    #     # Bound explicitly, just in case a few points escape 
                        
                    #     print('Imposing d(dL)/dz >0 on all the domain')
                    #     ν = pm.Deterministic("ν", at.as_tensor_variable(1e-05) )       
                    
                    #     ddL_dz_mon = distance_ratio_grid * ( b_full + grad_log_distance_ratio_grid )
                                            
                    #     Φ = pm.Deterministic("Φ", pm.math.invprobit(pm.math.clip( ddL_dz_mon / ν, -10, 10)))
                    #     # Binary likelihood: all 1s (indicating positive slope)
                    #     monotonicity = pm.Bernoulli("monotonicity", p=Φ, observed=at.ones(log_ddL_dz_grid.shape[0]).eval() )
                    

                    # Precompute cosmology pieces (symbolic)
                    dc_grid      = atools.dcfun_at(atools.zGridGlobals_at, H0_, Om_, w0_, interp=False)
                    dLem_grid    = (1.0 + atools.zGridGlobals_at) * dc_grid
                    ddLem_dz_grid= atools.ddL_dz_EM(atools.zGridGlobals_at, H0_, Om_, w0_, dc=dc_grid)
                    b_full       = atools.d_log_dLEM_dz(atools.zGridGlobals_at, H0_, Om_, w0_, dc=dc_grid, safe=False)
                    
                    # GP log-ratio & its derivative on the grid
                    dLGrid_at, log_distance_ratio_grid, grad_log_distance_ratio_grid = atools.z_from_dL_at(
                        None, H0_, Om_, w0_, Lambda_MG_,
                        is_GP_dL=True,
                        # data_range=data_range, GP_zero_point=GP_zero_point,
                        # dense_grad=dense_grad, eta=η, ell=ℓ, b_full=b_full,
                    )
                    
                    # Event-level z
                    zs = pm.Deterministic("z", atools.atinterp(dval, dLGrid_at, atools.zGridGlobals_at), dims="event_index")
                    
                    # Event-level dc
                    dc = pm.Deterministic("dc", atools.dcfun_at(zs, H0_, Om_, w0_, interp=False), dims="event_index")
                    
                    # Distance ratio on grid (compute once)
                    distance_ratio_grid = at.exp(log_distance_ratio_grid)
                    


                    s_grid = dLem_grid * grad_log_distance_ratio_grid + ddLem_dz_grid
                    ddL_dz_grid = s_grid * distance_ratio_grid
                    log_ddL_dz_grid = at.log( at.abs( ddL_dz_grid)+ 1e-30 )
                    

                    
                    log_ddL_dz = atools.atinterp( zs, atools.zGridGlobals_at, log_ddL_dz_grid )
                    
                    # Interpolate what you need at zs
                    log_dratio_at_z  = atools.atinterp(zs, atools.zGridGlobals_at, log_distance_ratio_grid)
                    grad_log_dr_at_z = atools.atinterp(zs, atools.zGridGlobals_at, grad_log_distance_ratio_grid)



                    log_ddL_dz_at_z  = atools.atinterp(zs, atools.zGridGlobals_at, log_ddL_dz_grid)
                    
                    distance_ratio = pm.Deterministic("d_ratio", at.exp(log_dratio_at_z), dims="event_index")
                    d_ratio_d_z    = pm.Deterministic("d_ratio_d_z", distance_ratio * grad_log_dr_at_z, dims="event_index")
                    log_ddL_dz     = pm.Deterministic("log_ddL_dz", log_ddL_dz_at_z, dims="event_index")
                    
                    # Monotonicity soft barrier
                    print("monotonicity is %s"%monotonicity)
                    if monotonicity is not None:
                        
                        
                        if monotonicity=='softplus':
                            print('Imposing d(dL)/dz >0 on all the domain')
                            # Temperature (smaller => harder constraint). Keep as tensor, no Deterministic needed.
                            print('Using softplus with nu=%s'%nu)
                            #nu   = at.as_tensor_variable(1e-15)
                            k = 1.7/nu
                            pm.Potential( "monotonicity", -at.sum(atools.softplus(-k * s_grid)) )
                            

                        elif monotonicity=='softplus_clip':

                            #nu   = at.as_tensor_variable(1e-05)  
                            print('Imposing d(dL)/dz >0 on all the domain')
                            print('Using stable softplus, nu=%s, clipping between %s, %s'%(nu,clip_low, clip_high ))
                            pm.Potential( "monotonicity", -at.sum( at.logaddexp( 0.0, at.clip(-s_grid/nu, clip_low, clip_high) )  ) )
                            
                        elif monotonicity=='poly':
                            print('Imposing d(dL)/dz >0 on all the domain')
                            print('Using smooth polynomial, nu=%s, lam=%s'%(nu, lam))
                            
                            # pm.Potential("monotonicity", -at.sum(lam * atools.poly_hinge_neg(s_grid, nu)))

                            # GP derivative g(z)
                            g_grid = grad_log_distance_ratio_grid
                            
                            # dimensionless monotonicity condition
                            q_grid = g_grid + b_full

                            mask = (atools.zGridGlobals_at <= z_max_mono)  # boolean mask on the grid

                            if zmin_mono!=0:
                                print("Lower lim for monotonicity penalty at z=%s"%zmin_mono)
                                mask &= (atools.zGridGlobals_at >= zmin_mono)

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
                                b_mono = b_full[mask]

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



                        else:
                            print('No monotonicity constraint.')
                            
                    else:
                            print('No monotonicity constraint.')

 
                else:

                    # we sample z from the pop prior. no need to invert the dL-z relation
                    print('Sampling redshift from population prior')

                    # sample redshift from p(z) ~ ψ(z)/(1+z)*dV/dz
                    if rate_model=='MD':
                        zs = pm.CustomDist( 'z', 
											 gamma_, kappa_, zp_, [H0_, Om_, w0_],
											 logp = atools.log_p_z_MD_unnorm,
											 #random = , 
											 size=(N,))
                    else:
                        raise NotImplementedError()

                    # obtain luminosity distance and the derivative, including the GP

                    # this is log(distance ratio) and its derivative computed on the grid
                    # zGridGlobals_at
                    log_distance_ratio, grad_log_distance_ratio = atools.compute_gp_interp_dist_ratio( atools.zGridGlobals_at, gp, data_range=data_range, name="f", GP_zero_point=GP_zero_point, dense_grad = dense_grad )
                    
                    # now compute distance ratio at the actual events redshifts
                    distance_ratio = pm.Deterministic( "d_ratio", at.exp(atools.atinterp( zs, atools.zGridGlobals_at, log_distance_ratio )))

                    # ... and its derivative
                    d_log_distance_ratio_d_z = atools.atinterp( zs, atools.zGridGlobals_at, grad_log_distance_ratio ) 
                    d_distance_ratio_d_z = pm.Deterministic( "d_ratio_d_z", d_log_distance_ratio_d_z*distance_ratio)

                    dc = atools.dcfun_at(zs, H0_, Om_, w0_)
                    d_EM = (zs+1.0)*dc

                    raise NotImplementedError('Here, need to compute log_ddL_dz')

                    # finally, the GW luminosity distance is distance_ratio * dEM
                    dval = d_EM*distance_ratio

                    

            
                    ### correct for the GW likelihood ratio

                    if spin_model == 'chieffchip' or spin_model == 'chieffchip_uc' :
                        spins = [chieff, chip]
                        vals = at.stack([log_Mc_det, logit_q, at.log(dval), chieff, chip, 
                                         ])
                                        
                    elif (spin_model == 'default') or (spin_model == 'default_gauss'):
                        spins = [chieff, chip, cost1, cost2]
                        vals = at.stack([log_Mc_det, logit_q, at.log(dval), chi1, chi2, cost1, cost2, 
                                         ])
                                        
                    elif spin_model == 'none':
                        spins = []
                        vals = at.stack([log_Mc_det, logit_q, at.log(dval),  
                                         ])

                    ## full GW likelihood 
                    logps, _ = pytensor.scan(fn=lambda iobs, X, M, F, logD, logW: # iobs = event index
				
												pytensor.scan(fn=lambda ig, X, M, F, logD, logW: 

												-0.5*at.sum((X[: , iobs] - M[iobs, ig])*(F[iobs, ig] @ (X[: , iobs] - M[iobs, ig])[:, None])[:, 0])
            									-0.5*nd*at.log(2*atools.PI)
												-0.5 * logD[iobs, ig]
												+ logW[iobs, ig],
									
												sequences=[at.arange(ngmm)], non_sequences=[vals, mus_l, icovs_l, log_dets_l, wts_l]),
						
									sequences=[at.arange(N)], non_sequences=[vals, mus_l, icovs_l, log_dets_l, wts_l])

					# sum of the gmm interpolants over all the gmm components
                    gwl = at.logsumexp(logps, axis=1) # shape (nev,)


                    ## subspace likelihood of the variables sampled from the gmm (all except distance)
                    
                    # consider just the masses (and later spins. but for now spins are not supported )
                    vals_sub = vals[:nsub]  # shape (2, N)
                
                    logps_sub, _ = pytensor.scan(fn=lambda iobs, X_sub, M_sub, F_sub, logD_sub, logW: 
            
                                            pytensor.scan(fn=lambda ig, X_sub, M_sub, F_sub, logD_sub, logW: 

                                            -0.5*at.sum((X_sub[: , iobs] - M_sub[iobs, ig])*(F_sub[iobs, ig] @ (X_sub[: , iobs] - M_sub[iobs, ig])[:, None])[:, 0])
                                            -0.5*2.*at.log(2*atools.PI)
                                            -0.5 * logD_sub[iobs, ig]
                                            + logW[iobs, ig],
                                
                                            sequences=[at.arange(ngmm)], non_sequences=[vals_sub, mus_l_sub, icovs_l_sub, log_dets_l_sub, wts_l]),
                    
                                    sequences=[at.arange(N)], non_sequences=[vals_sub, mus_l_sub, icovs_l_sub, log_dets_l_sub, wts_l])

                    gwl_sub = at.logsumexp(logps_sub, axis=1)

                    # add this to the toal likelihood
                    logR = gwl - gwl_sub
                
            
            # save values of GW distance and source-frame masses
            d = pm.Deterministic('dL', dval , dims="event_index")      

            m1src = pm.Deterministic('m1src', m1det/(1+zs) , dims="event_index")
            m2src = pm.Deterministic('m2src', m2det/(1+zs) , dims="event_index") 
            
                
        else:
            # we are sampling the usual marginalise likelihood, with "only" pop parameters
            print('We are running inference only on population parameters.')


            # Compute source-frame quantities. One redsfhit, mass1, mass2 for each event
            # AND for each sample! 
            
            d_stacked  = at.flatten(d)
            if not is_GP_dL:
                zs_stacked = atools.z_from_dL_at(d_stacked, H0_, Om_, w0_, Lambda_MG_ )
            else:
                raise NotImplementedError()

            
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
             


        
        if dLprior=='dLsq':
            # Remove \pi(d)~dL^2 prior on distance 
            log_p_pop -= 2*logd
            print('Removing dL^2 prior')
        elif dLprior == 'dVdz':
            print('Removing prior proportional to 1/(1+z)*dV/dz with H0=67.90, Om=0.3065')
            lpi_ = atools.log_dV_dz_at(zs, 67.90, 0.3065, dc=dc )-at.log1p(zs)

            # The following is a hack.
            # When using GWTC data, O1-O2 do not have posteriors with dVdz prior, only dL^2
            # So I remove the dL^2 prior by hand on those
            if not pop_only:
                # 1D case: shape (N,)
                lpi = at.concatenate([2 * logd[:10], lpi_[10:]], axis=0)
            else:
                # 2D case: shape (N, Nsamples)
                lpi = at.concatenate([2 * logd[:10, :], lpi_[10:, :]], axis=0)
            
            log_p_pop -= lpi


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
            likelihood_val = at.sum( log_p_pop ) #pm.Deterministic("lik", at.sum( log_p_pop ) ) 
        else:
            # marginalise over single events parameters first
            # shape of p_pop is (hopefully) n_evs x n_samples
            # so average over second dimension
            
            # Compute only where there are samples
            log_p_pop_to_marg = log_p_pop[:, :allNsamples[0]]
            
            log_p_pop_marg = at.logsumexp( log_p_pop_to_marg, axis=1 ) - at.log(allNsamples)
            

            # then sum log likelihoods
            likelihood_val = at.sum( log_p_pop_marg ) #pm.Deterministic("lik", at.sum( log_p_pop_marg ) ) 

            # Check number of effective samples for computing MC integral 
            logs2 = at.logsumexp(2*log_p_pop_masked, axis=1) -2*at.log(allNsamples)
            
            Neff_lik =  pm.Deterministic('Neff_l', at.exp( 2.0*log_p_pop_marg - logs2) ) # this has len = n. of observations
            
            if Neff_min_lik>0:
                
                _ = pm.Potential("Neff_l_bound", at.sum( at.where( Neff_lik<Neff_min_lik*N, -np.inf, 0. ) ) )
                
                # see https://discourse.pymc.io/t/conditionally-reject-samples/3107
                # ind_sw_l = pm.Deterministic('ind_l', 1. * (Neff_lik<Neff_min_lik) )
                # ind_l = pm.Bernoulli('Neff_l_bound', ind_sw_l, observed=np.zeros(N_np), testval=np.zeros(N_np) )

            
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

        
        likelihood = pm.Deterministic("lik", likelihood_val ) 
        likelihood_term = pm.Potential("likelihood", likelihood ) 
        
        #value = at.as_tensor_variable(1.0)  # this is a plain tensor
        #lval = pm.Deterministic("lik", value)     # optional: to log the potential
        #_ = pm.Potential("likelihood", value)  # use the tensor directly here

        #try:
        #    grads = grad(at.sum(likelihood), model.free_RVs)
        #    print("Gradients computed. No disconnected inputs.")
        #except DisconnectedInputError as e:
        #    print("DisconnectedInputError:", e)



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
                    
                    zinj = atools.atinterp( dLinj[0], dLGrid_at, atools.zGridGlobals_at )
                   
                    dc_inj = atools.dcfun_at(zinj, H0_, Om_,  w0_, interp=False)
                    

                    log_ddL_dz_inj   = atools.atinterp(zinj, atools.zGridGlobals_at, log_ddL_dz_grid)
          
                else:
                    zinj, log_ddL_dz_inj, dc_inj = None, None, None
                    
                    
                    
                log_mu_, Neff_, var_ll_u_ = sel_bias_with_uncertainty_at( m1inj[0], 
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
                                                                         smoothing, has_m2_break,
                                                                         #distance_ratio=distance_ratio_inj,
                                                                         #d_distance_ratio_d_z=d_distance_ratio_d_z_inj,
                                                                         log_ddL_dz_inj = log_ddL_dz_inj,
                                                                         zinj = zinj ,
                                                                         dcinj = dc_inj,
                                                                        
                                                                        )

                
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
                if is_GP_dL:
                    raise NotImplementedError()
                
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


                    res_i, _ = pytensor.scan( lambda idata, m1inj_, m2inj_, dLinj_, spinsInj_, lpdinj_, L,  Ndraw_, Ndet_ : sel_bias_with_uncertainty_at( m1inj_[idata, : Ndet_[idata]], m2inj_[idata, : Ndet_[idata]], dLinj_[idata, :Ndet_[idata]], spinsInj_[idata, :, :Ndet_[idata]], lpdinj_[idata, :Ndet_[idata]], L, Ndraw_[idata],                                                                                                                                   rate_model, mass_model, spin_model_name, is_GP_dL, smoothing, has_m2_break ), 

                                          sequences = [ at.arange( ndata) ], 

                                          non_sequences = [m1inj, m2inj, dLinj, spinsInj, lpdinj, Lambda_,  Ndraw, Ndet] )
                    log_mu_vec = res_i[0]
                    Neff_ = at.sum(res_i[1])

                    
                else:
                    print("Loop over injections sets, no slicing")
                    # makes it jax-compatible (jax does not support dynamical slicing at the moment)
                    # Not true anymore after pymc v5.10 ? Check


                    res_i, _ = pytensor.scan( lambda idata, m1inj_, m2inj_, dLinj_, spinsInj_, lpdinj_, L,  Ndraw_ : sel_bias_with_uncertainty_at( m1inj_[idata ], m2inj_[idata ], dLinj_[idata], spinsInj_[idata],  lpdinj_[idata], L, Ndraw_[idata], rate_model, mass_model, spin_model, is_GP_dL, smoothing, has_m2_break, zinj=zinj ), 
                                      sequences = [ at.arange( ndata) ], 

                                      non_sequences = [m1inj, m2inj, dLinj, spinsInj, lpdinj,  Lambda_,  Ndraw] )

            
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
                log_lik_var = pm.Deterministic('log_lik_var', at.exp(var_ll_u_+2*at.log(N)) )
            else:
                log_lik_var = pm.Deterministic('log_lik_var', at.exp(  var_ll_u_+2*at.log( R0*Ttot ) + 2*log_mu_ ) )
            
     

            if ((Neff_min==0) and (log_lik_var_min==0)):
                print("No condition on number of effective points in MC integral for sel. effect")
                selection_bias =  pm.Deterministic("sel_bias", sel_effect )
            else:
                if log_lik_var_min==0:

                    # Thresholding on N_eff
                    print("MC integral for sel. effect thresholded on N_eff")
                    
                    if sel_smoothing=='sigmoid':
                        # smooth with sigmoid between Neff_min and Neff_min+1 x Nobs
                        # over a scale = Neff_min
                        # i.e. at Neff_min * Nobs the likelihood becomes smoothly -inf
                        selection_bias = pm.Deterministic("sel_bias", atools.log_sigmoid(Neff, Neff_min*(N+1),  Neff_min)+sel_effect )
                    elif sel_smoothing=='poly':
                        # Polynomial smoothing
                        selection_bias = pm.Deterministic("sel_bias", atools.log_f_smooth_poly(Neff, N/2,  Neff_min*N-N/4)+sel_effect ) 
                    else:
                        # Hard cut
                        
                        selection_bias = pm.Deterministic("sel_bias", sel_effect)                   
                        #ind_sw_sel = pm.Deterministic('ind_sel', 1. * (Neff<Neff_min*N ) )
                        #ind_sel = pm.Bernoulli('bound_Neff', ind_sw_sel, observed=np.zeros(1)  )
                        _ = pm.Potential("bound_Neff", at.switch(Neff >= Neff_min * N, 0.0, -np.inf))

                
                elif Neff_min==0:

                    # Thresholding on likelihood variance
                    print("MC integral for sel. effect thresholded on log lik. variance")
                    
                    if sel_smoothing=='sigmoid':
                        # smooth with sigmoid 
                        print("Tapering sel effect with sigmoid smoothing")
                        selection_bias = pm.Deterministic("sel_bias", sel_effect+atools.logdiffexp( at.log(1), atools.log_sigmoid(log_lik_var, log_lik_var_min*(1+0.002), 0.001 ))
                                                          )
                    elif sel_smoothing=='poly':
                        print("Tapering sel effect with polynomial smoothing")
                        selection_bias = pm.Deterministic("sel_bias", sel_effect+atools.logdiffexp( at.log(1), atools.log_f_smooth_poly(log_lik_var, 0.01,  log_lik_var_min*(1-0.005) ))   
                                                         )      
                    else:
                        print("Tapering sel effect with hard cut")

                        selection_bias = pm.Deterministic("sel_bias", sel_effect)
                        # ind_sw_sel = pm.Deterministic('ind_sel', 1. * (log_lik_var>log_lik_var_min ) )
                        # ind_sel = pm.Bernoulli('bound_log_lik_var', ind_sw_sel, observed=np.zeros(1)  )
                        _ = pm.Potential("bound_log_lik_var", at.switch(log_lik_var <= log_lik_var_min, 0.0, -np.inf))

            
            selection_bias_term = pm.Potential('selection_bias', selection_bias)

            if marginal_R0:
                if include_sel_uncertainty:
                    print("Including selection function uncertainty as in Farr 2019s")
                    # from Farr 2019
                    sel_uncertainty = (3*N+N**2)/(2*Neff)
                    
                    sel_uncertainty_term = pm.Potential('selection_uncertainty', sel_uncertainty)
            

    return model

