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

PLPeakO3params = {'H0': 67.66, 'Om':0.31, 'w0':-1, 'Xi0': 1, 'nXi0':0}




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

    ##################################
    # redshift 

    if rate_model=='MD':
        
        gamma, kappa, zp = Lambda[iastro:iastro+3]

        if (invert_dL_GP or (not is_GP_dL) or is_inj ):
            
            # This term contains the comoving distance
            # If there is MG, d_c is not d_L/(1+z)!
            lpz = atools.log_p_z_MD_unnorm(z, gamma, kappa, zp, Lambda_c , dc=dc )
        
        
        istart = 8
 
        
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

        # This is the pdf of two independent gaussians - one for log(Mc), one for logit(q)
        # (there must be a smarter way of writing it instead of the loop. Do it for performance please)


        logpmass_1, _ = pytensor.scan( lambda ig, X, M, S, logD,   : 
                                                -0.5*( X-M[ig] )**2/(S[ig]**2)-0.5*at.log(2*atools.PI)-logD[ig],
                                            sequences = [  at.arange(Nmax) ],
                                             non_sequences =  [m1s, mu[0], sd[0], at.log(sd[0]), ],
                                         )


        logpmass_2, _ = pytensor.scan( lambda ig, X, M, S, logD,   : 
                                                -0.5*( X-M[ig] )**2/(S[ig]**2)-0.5*at.log(2*atools.PI)-logD[ig],
                                            sequences = [  at.arange(Nmax) ],
                                             non_sequences =  [m2s, mu[1], sd[1], at.log(sd[1]), ],
                                         )

        lpmass = at.logsumexp(logpmass_1+logpmass_2+logw[:, None], axis=0) # sum on gmm components
    
        #lpmass = at.zeros(m1s.shape)

    
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

    if mass_model=='DPUC':
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


    if mass_model=='DPUC':
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

def get_sample_from_cho_lMclqld(x, mu, L):
    
    
    # for cholesky rules see 
    # https://www.cs.helsinki.fi/u/ahonkela/teaching/compstats1/book/multivariate-normal-distributions-and-numerical-linear-algebra.html
    
    # x, mu have shape 3
    # L has shape 3x3
    # nd = mu.shape[0]

    #mvals = mu+at.dot(L,x)  
    
    #mlik = -0.5*at.dot( x.T, x )-0.5*mu.shape[0]*at.log(2*atools.PI)-at.sum( at.log(at.diagonal(L)) )
    
    #return  mu+at.dot(L,x) , -0.5*at.dot( x.T, x )-0.5*mu.shape[0]*at.log(2*atools.PI)-at.sum( at.log(at.diagonal(L)) )

    sample = mu + (L @ x[:, None])[:, 0]   # instead of at.dot(L, x)

    # Log probability of standard normal x
    logp = (
    -0.5 * at.sum(x**2)   # instead of at.dot(x.T, x)
    - 0.5 * mu.shape[0] * at.log(2 * atools.PI)
    - at.sum(at.log(at.diagonal(L)))  # log determinant of L
    )
    return sample, logp



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
                is_GP_dL = False,
               find_GP_L = True,
               fout=None,
               monotonicity = True,
               GP_prior = 'gammainv',
               GP_zero_point = 'y',
               rescale_GP=False,
               invert_dL_GP = True,
               dense_grad = False,
                 fix_H0 = True,
                fix_Om = True,
               fix_w0 = True,
                 fix_Xi0n = True,
               params_fix=None,
                 Neff_min=4,
                Neff_min_lik=1,
               log_lik_var_min=1,
                 use_sel_spin=True,
                 pop_only = False,
               N_successes_l=None,
               Nsamplesuse = -1,
               transform_samples=True,
               include_sel_uncertainty=False,
               sel_smoothing='poly',
               alpha_beta_prior='poly',
               dil_factor=1,
               use_log_alpha_beta=False ,
               allTobs=None
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
        
        elif sampling_GW=='gmm':

            wts_l, mus_l, cho_covs_l, icovs_l, log_dets_l, mus_l_sub, icovs_l_sub, log_dets_l_sub, Tobs, Nevs = GWData
            nsub = mus_l_sub.shape[2]
            print('nsub is %s'%nsub.eval())
            
            if not invert_dL_GP:
                nsub = mus_l_sub.shape[2]
                print('nsub is %s'%nsub.eval())
            
        else:
            raise ValueError('sampling_GW can be cho or gauss ')
            
        

    else:
        # gw data are single-event posterior samples
        # shape of each has to be n_events, n_samples
        m1det, m2det, d, spin_samples, Tobs, allNsamples, where_compute = GWData

        if transform_samples:
            print('Convert to m1 m2 etc.')
            lMc = m1det
            lq = m2det
            ld = d
    
            qs = atools.inv_logitat(lq) 
    
            
            if (spin_model=='default') or (spin_model=='default_gauss'):
                chi1 = atools.inv_logitat(spin_samples[0])
                chi2 = atools.inv_logitat(spin_samples[1])
                cost1 = atools.inv_flogitat(spin_samples[2])
                cost2 = atools.inv_flogitat(spin_samples[3])
                spin_samples = [chi1, chi2, cost1, cost2]

            m1det, m2det = atools.m1m2_from_Mcq_at(at.exp(lMc), qs )
            d = at.exp(ld)
            

        if Nsamplesuse !=-1 :
            if Nsamplesuse>allNsamples:
                raise ValueError("Must use less samples than those available.")
            print("allNsamples availabe is %s, but %s will be used"%(allNsamples, Nsamplesuse))
            allNsamples =  Nsamplesuse        
        
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

        
        
    if not pop_only:
        N = mus_l.shape[0] # number of events in total
        ngmm = mus_l.shape[1]
        nd = mus_l.shape[2]
        print('N:%s, max ngmm: %s, nd: %s '%(N.eval(), ngmm.eval(), nd.eval()))
        print('N evs is %s'%Nevs.eval())
        print('Tobs is %s'%Tobs.eval())
    else:
        N = m1det.shape[0] # number of events in total
        Nsamples = m1det.shape[1]
        print("N samples max will be ")
        print(Nsamples.eval())
        print('N:%s, n samples: %s '%(N.eval(), allNsamples.eval()))
    
    event_index = at.arange(N).eval()

    
    ndata = m1inj.shape[0] # number of observing runs to combine
    ninj = m1inj.shape[1] # max number of injections
    Ttot = at.sum(Tobs)

    
    print('Injections: :%s, '%(ninj.eval()))

    print('ninj: :%s, %s datasets,'%(Ndet.eval(), ndata.eval()))

    coords = {'event_index': event_index}

    if 'DP' in mass_model:
        coords['component'] = at.arange(N_DP_comp_max).eval()
        coords['GMMdimension'] = at.arange(2.).eval()

    if pop_only:
        coords['nsamples'] = at.arange( Nsamples ).eval()
    else:
         coords['GWdimension'] = at.arange(nd).eval()


    if params_fix is None:
        print('No values for parameters to fix passed. Default values will be used. If fixing parameters, check that the values are consistent. Values of fixed parameters:')
        print(PLPeakO3params)
        params_fix=PLPeakO3params

    if is_GP_dL:

        if find_GP_L:
            print("Finding min prior lengthscale for GP...")
            allL = []
            for i in range(50):
                # wts_l, mus_l, cho_covs_l, Tobs, Nevs
                x_ = np.random.randn(N.eval(), nd.eval())

                if sampling_GW=='gmm':
                    u = np.random.rand(N.eval(), 1)  # one uniform sample per event
                    cdf = np.cumsum(wts_l.eval(), axis=1)  # shape (n_events, n_components)            
                    idx_ = (u < cdf).argmax(axis=1)  
                    
                    #idx_ = wts_l.eval().argmax(axis=1)
                    
                    #print(idx_)
                    
                    # samples = mus_l[ at.arange(N), ig, :] + at.batched_dot( cho_covs_l[at.arange(N), ig, :, :], x )
                    samples_ = mus_l[ at.arange(N), idx_, :] + at.batched_dot(cho_covs_l[at.arange(N), idx_, :, :], x_ )    
                elif sampling_GW=='gauss':
                    raise NotImplementedError()
                
                
                #print(samples_.eval().shape)
                d_ = at.exp(samples_[:,2])
                #print(d_.eval().shape)
                if rescale_GP:
                    d_ = min_max_scaler(d_, data_range=(dmin, dmax)) 

                H0 = np.random.uniform(low=priors['H0'][0], high=priors['H0'][1], size=1)
                Om = np.random.uniform(low=priors['Om'][0], high=priors['Om'][1], size=1)
                z_ = atools.z_from_dL_at(d_, H0, Om, -1, [1, 0.], False, data_range=None )
                
                L_ = at.mean(at.diff(at.sort( z_ )))
                #print(L_.eval())
                allL.append(L_.eval())
            allL = at.as_tensor_variable(np.asarray(allL))
            #print(allL.shape.eval())
            L = at.max(allL, axis=0)

            beta = atools.find_beta(L.eval(), 2., p0=0.01)

            al = atools.find_al(L.eval(), 10., p0=0.01)

            
        else:
            L = at.as_tensor_variable(0.02867221802205662)
            beta = 5.1811
            al = 0.5579
            #L = at.mean(at.diff(at.sort( d_ )))
        print('L is %s'%L.eval())
        print(f"Found beta: {beta:.4f}")
        print(f"Found alpha: {al:.4f}")
        print(f"Mean length scale: {2 / beta:.4f}")
        
        #if True:
        lambda_ell = -at.log(atools.alpha_ell) * L**(atools.d_GP / 2)
        print('lambda_ell is %s'%lambda_ell.eval())

        import matplotlib.pyplot as plt
        from scipy.stats import gamma
        from scipy.stats import halfnorm
        from scipy.stats import invgamma
        ℓ_vals = at.geomspace(1e-05, 10, 1000)
        logp_vals = atools.frechet_logp_full(ℓ_vals, lambda_ell, atools.d_GP) 
        pdf_gamma = gamma.pdf(ℓ_vals.eval(), a=2., scale=1/beta)
        pdf_gamma_inv = invgamma.pdf( ℓ_vals.eval(), a=al, scale=1/10 )
        pdf_l = halfnorm(scale=1).pdf(ℓ_vals.eval())
        plt.plot(ℓ_vals.eval(), at.exp(logp_vals).eval(), label='frechet')
        plt.plot(ℓ_vals.eval(), pdf_gamma, label='gamma')
        plt.plot(ℓ_vals.eval(), pdf_l, label='halfnorm')
        plt.plot(ℓ_vals.eval(), pdf_gamma_inv, label='inv gamma')
        plt.xlabel("ℓ")
        plt.ylabel("Prior density")
        plt.title("PC prior on ℓ")
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(1e-05,10)
        plt.axvline(L.eval(), ls='--', color='k')
        plt.legend()
        plt.grid()
        #plt.show()
        plt.savefig( os.path.join(fout, 'ell_prior.pdf'), bbox_inches='tight')
        plt.close()

        
    ################################################
    # Build model
    ################################################
    
    with pm.Model(coords=coords) as model:

        ################################################
        # Cosmological parameters
        ################################################

        
        if fix_H0:
            H0_ =  at.as_tensor_variable(params_fix['H0'])
        else:
            #
            H0_ =  pm.Uniform('H0', lower=priors['H0'][0], upper=priors['H0'][1], initval=ivals.get('H0'))
            #H0_ =  pm.Normal("H0", mu=70.0, sigma=2.0)


        
        if fix_Om:
            Om_ = at.as_tensor_variable(params_fix['Om'])
        else:
            Om_ = pm.Uniform('Om', lower=priors['Om'][0], upper=priors['Om'][1], initval=ivals.get('Om')) 
            #Om_ = pm.TruncatedNormal("Om", mu=0.25, sigma=0.05, lower=0.05, upper=0.6)

        if fix_w0:
            w0_ = at.as_tensor_variable(-1.)
        else:
            raise NotImplementedError()
        
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
                                  atools.d_GP,
                                  logp=atools.frechet_logp_full,
                                    #size=(1,)
                                 )
                print('ℓ prior is frechet')
            
            elif GP_prior=='gamma':
                ℓ = pm.Gamma("ℓ", alpha=2., beta=beta)
                print('ℓ prior is Gamma')
            elif GP_prior=='gammainv':
                ℓ = pm.InverseGamma("ℓ", alpha=al, beta=0.1 )
                print('ℓ prior is Inverse Gamma')
            else:
                raise ValueError()
            
            η = pm.Exponential("η", lam=atools.lambda_)
            print('η prior is Exponential with lambda=%s, from scale U=%s'%(atools.lambda_.eval(), atools.U.eval()))

            cov = η**2 * pm.gp.cov.Matern52( input_dim=1, ls=ℓ ) + pm.gp.cov.WhiteNoise(1e-4)
            gp = pm.gp.Latent(cov_func=cov)

            # for imposing monotonicity
            #eps = at.as_tensor_variable(1e-12)          # avoid div-by-zero
            
            Lambda_MG_ = [ gp  ] 
            iastro = 4
        Lambda_ += Lambda_MG_   
        ################################################
        # Redshift evolution of merger rate
        ################################################
        
        if rate_model=='MD':
            print('Modeling evolution of merger rate with redshift with Madau-Dickinson profile')
            #gamma_ = pm.Uniform('gamma', lower=priors['gamma'][0], upper=priors['gamma'][1], initval=ivals.get('gamma'))    
            #kappa_ = pm.Uniform('kappa', lower=priors['kappa'][0], upper=priors['kappa'][1], initval=ivals.get('kappa'))
            #zp_ = pm.Uniform('zp', lower=priors['zp'][0], upper=priors['zp'][1], initval=ivals.get('zp'))

            gamma_ = atools.uniform_unconstrained("gamma",  priors['gamma'][0], priors['gamma'][1], init=ivals.get("gamma"))
            kappa_ = atools.uniform_unconstrained("kappa",  priors['kappa'][0], priors['kappa'][1], init=ivals.get("kappa"))
            zp_ = atools.uniform_unconstrained("zp",  priors['zp'][0], priors['zp'][1], init=ivals.get("zp"))
            
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
                    ind_sw_al = pm.Deterministic('ind_al', 1. * (alphaChi_<=1. ) )
                    ind_al = pm.Bernoulli('bound_alphaChi', ind_sw_al, observed=0.  )
                    ind_sw_b = pm.Deterministic('ind_b', 1. * (betaChi_<=1. ) )
                    ind_b = pm.Bernoulli('bound_betaChi', ind_sw_b, observed=0.  )
                    
                    # alternative. 
                    # _ = pm.Potential('bound_alphaChi', at.switch( at.le(alphaChi_, at.as_tensor_variable(1.) ), -atools.INF, at.as_tensor_variable(0.) ) )
                # _ = pm.Potential('bound_betaChi', at.switch( at.le(betaChi_, at.as_tensor_variable(1.) ), -atools.INF, at.as_tensor_variable(0.)) )
        
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
            
            #lamP_   = pm.Uniform("lambdaPeak", lower=priors["lambdaPeak"][0], upper=priors["lambdaPeak"][1], initval=ivals.get("lambdaPeak"))
            lamP_ = atools.uniform_unconstrained("lambdaPeak",  priors['lambdaPeak'][0], priors['lambdaPeak'][1], init=ivals.get("lambdaPeak"))
            
            #alpha_  = pm.Uniform("alpha",      lower=priors["alpha"][0],      upper=priors["alpha"][1],      initval=ivals.get("alpha"))
            #beta_   = pm.Uniform("beta",       lower=priors["beta"][0],       upper=priors["beta"][1],       initval=ivals.get("beta"))
            #ml_     = pm.Uniform("ml",         lower=priors["ml"][0],         upper=priors["ml"][1],         initval=ivals.get("ml"))
            #mh_     = pm.Uniform("mh",         lower=priors["mh"][0],         upper=priors["mh"][1],         initval=ivals.get("mh"))
            #deltam_ = pm.Uniform("deltam",     lower=priors["deltam"][0],     upper=priors["deltam"][1],     initval=ivals.get("deltam"))
            #muM_    = pm.Uniform("muMass",     lower=priors["muMass"][0],     upper=priors["muMass"][1],     initval=ivals.get("muMass"))
            #sM_     = pm.Uniform("sigmaMass",  lower=priors["sigmaMass"][0],  upper=priors["sigmaMass"][1],  initval=ivals.get("sigmaMass"))

            alpha_  = atools.uniform_unconstrained("alpha",     priors["alpha"][0],     priors["alpha"][1],     init=ivals.get("alpha"))
            beta_   = atools.uniform_unconstrained("beta",      priors["beta"][0],      priors["beta"][1],      init=ivals.get("beta"))
            ml_     = atools.uniform_unconstrained("ml",        priors["ml"][0],        priors["ml"][1],        init=ivals.get("ml"))
            mh_     = atools.uniform_unconstrained("mh",        priors["mh"][0],        priors["mh"][1],        init=ivals.get("mh"))
            deltam_ = atools.uniform_unconstrained("deltam",    priors["deltam"][0],    priors["deltam"][1],    init=ivals.get("deltam"))
            muM_    = atools.uniform_unconstrained("muMass",    priors["muMass"][0],    priors["muMass"][1],    init=ivals.get("muMass"))
            sM_     = atools.uniform_unconstrained("sigmaMass", priors["sigmaMass"][0], priors["sigmaMass"][1], init=ivals.get("sigmaMass"))

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
            m_high_   = pm.Deterministic("m_high", at.as_tensor_variable(300.0))
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
                m_g_     = at.as_tensor_variable(45)
                w_g_     = at.as_tensor_variable(70)
                sig_g_l_ = at.as_tensor_variable(1e-04)
                sig_g_h_ = at.as_tensor_variable(1e-04)
            else:
                m_g_     = at.as_tensor_variable(45)
                w_g_     = at.as_tensor_variable(70)
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
            tau1 = pm.Gamma("tau1", 10.0, 1.0, dims="component")
            lambda1_ = pm.Gamma("lambda1_", 15.0, 1.0, dims="component" )

            tau2 = pm.Gamma("tau2", 1.0, 1.0, dims="component")
            lambda2_ = pm.Gamma("lambda2_", 0.8, 1.0, dims="component" )
            
            sig1 = pm.Deterministic("sig1", 1./at.sqrt(lambda1_*tau1), dims= "component" )
            sig2 = pm.Deterministic("sig2", 1./at.sqrt(lambda2_*tau2), dims= "component" )

            # Option 2: Fixes std from std on m1, m2
            #sd_ = pm.Uniform( 'sigm1m2', lower=priors['sig'][0], upper=priors['sig'][1], dims=("GMMdimension",  "component") )
            #sig1 = at.sqrt(sd_[0]**2 * (3/(5*mu_[0]) - 1/(5*(mu_[0]-mu_[1])))**2 + sd_[1]**2 * (3/(5*mu_[1]) - 1/(5*(mu_[1]-mu_[0])))**2)
            #sig2 = at.sqrt(sd_[0]**2 * (1/(mu_[0]-mu_[1]))**2 + sd_[1]**2 * (mu_[0]/(mu_[1]*(mu_[0]-mu_[1])))**2)
        
            # Option 3: Fixes std from given prior
            #sig1 = pm.Uniform('siglMc', lower=priors['siglMc'][0], upper=priors['siglMc'][1], dims=("component" ))
            #sig2 = pm.Uniform('siglq', lower=priors['siglq'][0], upper=priors['siglq'][1], dims= ("component")) 

            sigval = at.zeros( (2, N_DP_comp_max) )
            sigval = at.set_subtensor( sigval[0], sig1 )
            sigval = at.set_subtensor(  sigval[1], sig2 )
            sd = pm.Deterministic("sig", sigval, dims=("GMMdimension" , "component" ))


            #### Mean prior limits:  remember that mu is log(Mc), logit(q).

            # Option 1 : sample mean of the gaussians from given prior
            # with this choice, the prior on the mean will be flat in log(Mc), logit(q).
        
            mu1 = pm.Uniform('mulMc', lower=priors['mulMc'][0], upper=priors['mulMc'][1], dims= ("component" ))
            mu2 = pm.Uniform('mulq', lower=priors['mulq'][0], upper=priors['mulq'][1], dims= ("component" )) 

            muval = at.zeros( (2, N_DP_comp_max) )
            muval = at.set_subtensor( muval[0], mu1 )
            muval = at.set_subtensor( muval[1], mu2 )
            mu = pm.Deterministic("mu", muval, dims=("GMMdimension" , "component" ))

            # Option 2: check ...

            Lambda_ += [ w, mu, sd, logw ]

            Lambda_ += [N_DP_comp_max]
        
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
                
            if sampling_GW=='gmm':
    
                print('Sampling m1d, m2d, dL from GMM')
                ig = pm.Categorical('idx', p=wts_l, dims= "event_index" )

                # old way. leave it here  please
                # samples = mus_l[ at.arange(N), ig, :] + at.batched_dot( cho_covs_l[at.arange(N), ig, :, :], x )
                
                # Select means and Cholesky factors per batch
                mu_selected = mus_l[at.arange(N), ig, :]         # shape (N, D)
                L_selected = cho_covs_l[at.arange(N), ig, :, :]  # shape (N, D, D)
                 
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
                
                # unpack coordinates if you like:
                log_Mc_det = samples[:, 0]
                logit_q    = samples[:, 1]
                logd       = samples[:, 2]


                if spin_model == 'none' :
                    
                    vals = at.zeros( (3, N) )
                
                    vals = at.set_subtensor( vals[0], log_Mc_det )
                    vals = at.set_subtensor( vals[1], logit_q )
                    vals = at.set_subtensor( vals[2], logd )


                elif spin_model == 'default' :

                    chi1 = atools.inv_logitat(res[0][:,3])
                    chi2 = atools.inv_logitat(res[0][:,4])
        
                    cost1 = atools.inv_flogitat(res[0][:,5])
                    cost2 = atools.inv_flogitat(res[0][:,6])
            

                    vals = at.zeros( (7, N) )
                
                    vals = at.set_subtensor( vals[0], log_Mc_det )
                    vals = at.set_subtensor( vals[1], logit_q )
                    vals = at.set_subtensor( vals[2], logd )
                    vals = at.set_subtensor( vals[3], res[0][:,3] )
                    vals = at.set_subtensor( vals[4], res[0][:,4] )
                    vals = at.set_subtensor( vals[5], res[0][:,5] )
                    vals = at.set_subtensor( vals[6], res[0][:,6] )
                    
                
                
                # gw likelihood
                
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

                    # we sampled distance from the posterior. need to invert the dL-z relation
                          
                    
                    dLGrid_at, log_distance_ratio_grid, grad_log_distance_ratio_grid = atools.z_from_dL_at (None, H0_, Om_, w0_, Lambda_MG_ , is_GP_dL, data_range=data_range, GP_zero_point=GP_zero_point, dense_grad = dense_grad,  eta=η , ell=ℓ  )
    
                    zs = pm.Deterministic('z', atools.atinterp( dval, dLGrid_at, atools.zGridGlobals_at ) , dims= "event_index" ) 

                    dc = pm.Deterministic('dc', atools.dcfun_at(zs, H0_, Om_,  w0_, interp=False) , dims= "event_index" )
                    

                    # now derivative d(dL)/dz and comoving distance
                    if dense_grad:

                        # log_distance_ratio_grid is on zGridGlobals_at
                        # grad_log_distance_ratio_grid is maps
                        print("Computing derivatives on denser grid")

                        maps = grad_log_distance_ratio_grid

                        T_like, A_like = maps(zs)                 # both (len(X_like), M)
                        
                        log_distance_ratio   = pm.Deterministic("log_d_ratio",   T_like @ log_distance_ratio_grid, dims= "event_index")
                        distance_ratio = pm.Deterministic( "d_ratio", at.exp(log_distance_ratio), dims= "event_index")

                        # needed for monotonicity. sign matters
                        d_log_distance_ratio_d_z  =   A_like @ log_distance_ratio_grid
                        ddLem_dz = atools.ddL_dz_EM( zs, H0_, Om_, w0_, dc=dc ) 
                        dLem = (1+zs)*dc

                        # not needed
                        #d_distance_ratio_d_z = pm.Deterministic( "d_ratio_d_z", d_log_distance_ratio_d_z*distance_ratio, dims= "event_index")

                        
                                   
                        # needed only for jacobian. abs is ok.
                        s = dLem * d_log_distance_ratio_d_z + ddLem_dz
                        log_ddL_dz = at.log( at.abs( s * distance_ratio ) )

                        

                        # derivative on full grid, for monotonicity
                        
                        T_grid, A_grid = maps( atools.zGridGlobals_at)            # both (len(X_like), M)

                        d_log_distance_ratio_d_z_grid  =  A_grid @ log_distance_ratio_grid
                        
                        dc_grid = atools.dcfun_at(atools.zGridGlobals_at, H0_, Om_,  w0_, interp=False)
                        dLem_grid = (1+atools.zGridGlobals_at)*dc_grid
                        
                        distance_ratio_grid = at.exp(log_distance_ratio_grid)
                        ddLem_dz_grid = atools.ddL_dz_EM( atools.zGridGlobals_at, H0_, Om_, w0_, dc=dc_grid ) 
                        
                                             

                        s_grid = dLem_grid * d_log_distance_ratio_d_z_grid + ddLem_dz_grid
                        log_ddL_dz_grid = at.log( at.abs( s_grid * distance_ratio_grid ) )
                        
                    
                    else:

                        print("Computing derivatives by interpolation")
                         
                        distance_ratio = pm.Deterministic( "d_ratio", at.exp(atools.atinterp( zs, atools.zGridGlobals_at, log_distance_ratio_grid )), dims= "event_index")

                                            
                        d_log_distance_ratio_d_z = atools.atinterp( zs, atools.zGridGlobals_at, grad_log_distance_ratio_grid )  

                        d_log_distance_ratio_d_z_grid = grad_log_distance_ratio_grid
                        
                        d_distance_ratio_d_z = pm.Deterministic( "d_ratio_d_z", d_log_distance_ratio_d_z*distance_ratio, dims= "event_index")
    
                        dc_grid = atools.dcfun_at(atools.zGridGlobals_at, H0_, Om_,  w0_, interp=False)
                        dLem_grid = (1+atools.zGridGlobals_at)*dc_grid
    
                        ddLem_dz_grid =  atools.ddL_dz_EM( atools.zGridGlobals_at, H0_, Om_, w0_,  dc=dc_grid )
        
                        distance_ratio_grid = at.exp(log_distance_ratio_grid)
        

                        s_grid = dLem_grid * grad_log_distance_ratio_grid + ddLem_dz_grid
                        log_ddL_dz_grid = at.log( at.abs( s_grid * distance_ratio_grid ) )
                        # log_ddL_dz_grid = at.log( at.abs( dLem_grid*grad_log_distance_ratio_grid*distance_ratio_grid + distance_ratio_grid*ddLem_dz_grid ) )
                                                
        
                        log_ddL_dz = atools.atinterp( zs, atools.zGridGlobals_at, log_ddL_dz_grid )
                
                             
                    
                    print("H0 init:", float(H0_.eval()))
                    print("Om init:", float(Om_.eval()))
                    print("dLem_grid finite?", np.all(np.isfinite(dLem_grid.eval())))
                    print("ddLem_dz_grid finite?", np.all(np.isfinite(ddLem_dz_grid.eval())))

                    #p1 = pm.Potential("guard_dlem", at.switch(at.all(atools.at_isfinite(dLem_grid)), 0.0, -1e12))
                    #p2 = pm.Potential("guard_ddlem", at.switch(at.all(atools.at_isfinite(ddLem_dz_grid)), 0.0, -1e12))

                    if monotonicity:

                        print('Imposing d(dL)/dz >0 on all the domain')
               
                        # Probit model: P(f′ > 0) = Φ(f′ / ν)
                        # scale of transition to zero
                        #ν = 1e-06 #pm.HalfNormal("ν", sigma=0.001)

                        ν = pm.Deterministic("ν", 0.1 * at.sqrt(5.0 * (η**2) / (3.0 * (ℓ**2))) )
                        

                        ddL_dz_mon = distance_ratio_grid * s_grid
                        
                        Φ = pm.Deterministic("Φ", pm.math.invprobit(pm.math.clip( ddL_dz_mon / ν, -10, 10)))
                        # Binary likelihood: all 1s (indicating positive slope)
                        monotonicity = pm.Bernoulli("monotonicity", p=Φ, observed=at.ones(log_ddL_dz_grid.shape[0]).eval() )


                        if False:
                            lb  = - atools.d_log_dLEM_dz( atools.zGridGlobals_at, H0_, Om_, w0_,  dc=dc_grid ) 
    
                            
    
                            # # Residual in the monotonicity term
                            Δ = d_log_distance_ratio_d_z_grid - lb #).astype("float64")
                                                   
                            
                        
    
                            r = Δ / ν #at.maximum(ν, eps)  
                            x = at.clip(r , -30, 30)
                            
    
                            # ---- Smooth one-sided barrier: enforces Δ >= 0 (i.e., d_log_ratio ≥ -lb) ----
                            monotonicity = pm.Potential("monotonicity", -at.sum(atools.softplus_stable(-x)))

                        
                        print(" ν is %s"%ν.eval())
                        print("monotonicity is ")
                        print(monotonicity.eval())
                
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
        if mass_model=='DPUC':

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
            lpi_ = atools.log_dV_dz_at(zs, 67.90, 0.3065, dc=d/(1+zs) )-at.log1p(zs)

            # The following is a hack.
            # When using GWTC data, O1-O2 do not have posteriors with dVdz prior, only dL^2
            # So I remove the dL^2 prior by hand on those
            if not pop_only:
                lpi = at.zeros( N )    
                lpi = at.set_subtensor( lpi[:10], 2*logd[:10] )
                lpi = at.set_subtensor( lpi[10:], lpi_[10:] )
            else:
                lpi = at.zeros( (N, Nsamples) )    
                lpi = at.set_subtensor( lpi[:10, :], 2*logd[:10, :] )
                lpi = at.set_subtensor( lpi[10:, :], lpi_[10:, :] )
            
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
                
                #_ = pm.Potential("Neff_l_bound", at.sum( at.where( Neff_lik<Neff_min_lik*N, -atools.INF, at.as_tensor_variable(0.) ) ) )
                
                # see https://discourse.pymc.io/t/conditionally-reject-samples/3107
                ind_sw_l = pm.Deterministic('ind_l', 1. * (Neff_lik<Neff_min_lik) )
                ind_l = pm.Bernoulli('Neff_l_bound', ind_sw_l, observed=at.zeros(N).eval(), testval=at.zeros(N) )
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
            if ndata.eval()==1:
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
                    #d_log_distance_ratio_d_z_inj = atools.atinterp( zinj, atools.zGridGlobals_at, grad_log_distance_ratio )        
                    #distance_ratio_inj = at.exp(atools.atinterp( zinj, atools.zGridGlobals_at, log_distance_ratio ))
                    #d_distance_ratio_d_z_inj = d_log_distance_ratio_d_z_inj*distance_ratio_inj
                    
                    dc_inj = atools.dcfun_at(zinj, H0_, Om_,  w0_, interp=False)
                    
                    if not dense_grad:
                        log_ddL_dz_inj = atools.atinterp( zinj,  atools.zGridGlobals_at, log_ddL_dz_grid )
                        #dc_inj = atools.atinterp( zinj,  atools.zGridGlobals_at, dc_grid )
                        
                    else:


                        T_inj, A_inj = maps(zinj) 
                        
                        log_distance_ratio_inj    =  T_inj @ log_distance_ratio_grid 
                        distance_ratio_inj = at.exp(log_distance_ratio_inj)
                        d_log_distance_ratio_d_z_inj  =   A_inj @ log_distance_ratio_grid

                        ddLem_dz_inj =  atools.ddL_dz_EM( zinj, H0_, Om_, w0_, dc=dc_inj )  #atools.safe_exp( atools.log_ddL_dz( zinj, H0_, Om_, w0_, 1., 0., dc=None ) )
                        dLem_inj = (1+zinj)*dc_inj

                        #log_ddL_dz_inj = atools.safe_log(  dLem_inj*d_log_distance_ratio_d_z_inj*distance_ratio_inj + distance_ratio_inj*ddLem_dz_inj ) 
                        
                        s_grid_inj = dLem_inj * d_log_distance_ratio_d_z_inj + ddLem_dz_inj
                        log_ddL_dz_inj = at.log( at.abs( s_grid_inj * distance_ratio_inj ) )

           
                    
                
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

                    if spin_model == 'chieffchip' or spin_model == 'chieffchip_uc' :
    
                        spinsInj = at.zeros( (ndata, 2, ninj) )
                        spinsInj = at.set_subtensor( spinsInj[:, 0, :], chi1Inj )
                        spinsInj = at.set_subtensor( spinsInj[:, 1, :], chi2Inj )
                    
                    
                    elif (spin_model == 'default') or (spin_model == 'default_gauss'):
                    
                        spinsInj = at.zeros( (ndata, 4, ninj) )
                        spinsInj = at.set_subtensor( spinsInj[:, 0, :], chi1Inj )
                        spinsInj = at.set_subtensor( spinsInj[:, 1, :], chi2Inj )
                        spinsInj = at.set_subtensor( spinsInj[:, 2, :], cost1Inj )
                        spinsInj = at.set_subtensor( spinsInj[:, 3, :], cost2Inj )

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
                        print('Using sel function with weighted obs time average. Obs times: %s'%str(Tobs.eval()))
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
                        ind_sw_sel = pm.Deterministic('ind_sel', 1. * (Neff<Neff_min*N ) )
                        ind_sel = pm.Bernoulli('bound_Neff', ind_sw_sel, observed=at.zeros(1).eval()  )
                
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
                        ind_sw_sel = pm.Deterministic('ind_sel', 1. * (log_lik_var>log_lik_var_min ) )
                        ind_sel = pm.Bernoulli('bound_log_lik_var', ind_sw_sel, observed=at.zeros(1).eval()  )
            
            selection_bias_term = pm.Potential('selection_bias', selection_bias)

            if marginal_R0:
                if include_sel_uncertainty:
                    print("Including selection function uncertainty as in Farr 2019s")
                    # from Farr 2019
                    sel_uncertainty = (3*N+N**2)/(2*Neff)
                    
                    sel_uncertainty_term = pm.Potential('selection_uncertainty', sel_uncertainty)
            

    return model

