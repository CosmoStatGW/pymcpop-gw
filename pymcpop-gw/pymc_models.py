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

PLPeakO3params = {'H0': 67.66, 'Om':0.31, 'w0':-1, 'Xi0': 1, 'nXi0':0}




def log_p_pop_at(m1s, m2s, z, dL, spins, Lambda, rate_model, mass_model, spin_model, is_GP_dL, pairing=True, dr_val=None, ddr_dz=None, is_inj=False):

    ###################################
    # get parameters and compute log p_pop
    ####################################

    Lambda_c = Lambda[:3] 
    H0, Om, w0 = Lambda_c 
    # Needed for comoving volume. It is always the EM one !
    if not is_inj:
        dc = pm.Deterministic('d_c', atools.dcfun_at(z, H0, Om, w0) ) 
    else:
        dc = atools.dcfun_at(z, H0, Om, w0)
    
    if is_GP_dL:
        
        iastro = 4
        gp = Lambda[3] 
        # jacobian
        #dz_ddL = at.grad(z.sum(), dL)
        # log_ddL_dz = ( at.log(  at.abs ( 1/ dz_ddL)) ) 
        
        dL_em = dc*(1+z)
        ddLem_dz = at.exp( atools.log_ddL_dz( z, H0, Om, w0, 1., 0., dc=dc ) )
        
        log_ddL_dz = at.log( at.abs( dL_em*ddr_dz + dr_val*ddLem_dz ) )

    else:

        Xi0, n = Lambda[3:5] 
        
        iastro = 5

        # jacobian
        log_ddL_dz = atools.log_ddL_dz(z, H0, Om, w0, Xi0, n, dc=dc)

    ##################################
    # redshift 
    
    if rate_model=='MD':
        
        gamma, kappa, zp = Lambda[iastro:iastro+3]

        # This term contains the comoving distance
        # If there is MG, d_c is not d_L/(1+z)!
        lpz = atools.log_p_z_MD_unnorm(z, gamma, kappa, zp, Lambda_c , dc=dc )
        
        
        istart = 8
 
        
    elif rate_model=='PL':
        
        gamma = Lambda[iastro]
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
        lpmass = atools.logpdf_PLP_reg([m1s, m2s], [lp, al, bb, dm, ml, mh, muM, sM], pairing=pairing)

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

    lp =  lpz - log_ddL_dz - 2*at.log1p(z) + lpmass + lpspin

    return lp



def sel_bias_with_uncertainty_at(m1inj, m2inj, dLinj, spinsInj, log_p_draw, Lambda,  Ndraw, rate_model, mass_model, spin_model, is_GP_dL, pairing, distance_ratio=None, d_distance_ratio_d_z=None, zinj=None):


    if (spin_model=='default') or (spin_model=='default_gauss'):
        spinsInj_sel = [spinsInj[0], spinsInj[1], spinsInj[2], spinsInj[3]]
    elif spin_model=='none':
        spinsInj_sel = []


    if not is_GP_dL:
        H0, Om, w0, Xi0, n  = Lambda[:5]
        zinj = atools.z_from_dL_at(dLinj, H0, Om, w0, [Xi0, n] , is_GP_dL )
        distance_ratio , d_distance_ratio_d_z = None, None
        
    
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

    log_p_pop = log_p_pop_at(mass_1_use, mass_2_use, zinj, dLinj, spinsInj_sel, Lambda, rate_model, mass_model, spin_model, is_GP_dL, pairing=pairing, dr_val=distance_ratio, ddr_dz=d_distance_ratio_d_z, is_inj=True)

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
                 sampling_GW = 'gmm',
                 rate_model = 'MD',
                 mass_model = 'PLP',
                 pairing=True,
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
               GP_prior = 'gamma',
               GP_zero_point = False,
               rescale_GP=False,
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
            # we sample single-event parameters from the actual single-event posteriors
            wts_l, mus_l, cho_covs_l, Tobs, Nevs = GWData
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
    
                u = np.random.rand(N.eval(), 1)  # one uniform sample per event
                cdf = np.cumsum(wts_l.eval(), axis=1)  # shape (n_events, n_components)            
                idx_ = (u < cdf).argmax(axis=1)  
                
                #idx_ = wts_l.eval().argmax(axis=1)
                
                #print(idx_)
                
                # samples = mus_l[ at.arange(N), ig, :] + at.batched_dot( cho_covs_l[at.arange(N), ig, :, :], x )
                samples_ = mus_l[ at.arange(N), idx_, :] + at.batched_dot(cho_covs_l[at.arange(N), idx_, :, :], x_ )    
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
            H0_ =  pm.Uniform('H0', lower=priors['H0'][0], upper=priors['H0'][1])
        
        if fix_Om:
            Om_ = at.as_tensor_variable(params_fix['Om'])
        else:
            Om_ = pm.Uniform('Om', lower=priors['Om'][0], upper=priors['Om'][1]) 

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
                ℓ = pm.DensityDist( "ℓ", logp=lambda x: atools.frechet_logp_full(x, lambda_ell, atools.d_GP)  )
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

            cov = η**2 * pm.gp.cov.Matern52(1, ℓ) + pm.gp.cov.WhiteNoise(1e-5)
            gp = pm.gp.Latent(cov_func=cov)
            
            Lambda_MG_ = [ gp  ] 
            iastro = 4
        Lambda_ += Lambda_MG_   
        ################################################
        # Redshift evolution of merger rate
        ################################################
        
        if rate_model=='MD':
            print('Modeling evolution of merger rate with redshift with Madau-Dickinson profile')
            gamma_ = pm.Uniform('gamma', lower=priors['gamma'][0], upper=priors['gamma'][1])    
            kappa_ = pm.Uniform('kappa', lower=priors['kappa'][0], upper=priors['kappa'][1])
            zp_ = pm.Uniform('zp', lower=priors['zp'][0], upper=priors['zp'][1])

            Lambda_ += [gamma_, kappa_, zp_]

        elif rate_model=='PL':
            print('Modeling evolution of merger rate with a power law')
            gamma_ = pm.Uniform('gamma', lower=priors['gamma'][0], upper=priors['gamma'][1])

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
            if not pairing:
                print('No pairing function C(m1)')
            
            lamP_ = pm.Uniform('lambdaPeak', lower=priors['lambdaPeak'][0], upper=priors['lambdaPeak'][1])
            alpha_ = pm.Uniform('alpha', lower=priors['alpha'][0], upper=priors['alpha'][1])
            beta_ = pm.Uniform('beta', lower=priors['beta'][0], upper=priors['beta'][1])
            ml_ = pm.Uniform('ml', lower=priors['ml'][0], upper=priors['ml'][1])
            mh_ = pm.Uniform('mh', lower=priors['mh'][0], upper=priors['mh'][1])
            deltam_ = pm.Uniform('deltam', lower=priors['deltam'][0], upper=priors['deltam'][1])
            muM_ = pm.Uniform('muMass', lower=priors['muMass'][0], upper=priors['muMass'][1])
            sM_ = pm.Uniform('sigmaMass', lower=priors['sigmaMass'][0], upper=priors['sigmaMass'][1] )  

            Lambda_ += [lamP_, alpha_, beta_, deltam_, ml_, mh_, muM_, sM_ ]

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

                
                res, _ = pytensor.scan( lambda iev, X, M, L: get_sample_from_cho_lMclqld( X[iev], M[iev], L[iev] )  ,
                                        sequences = [ at.arange(N)],
                                        non_sequences = [ x, mus_s, cho_s]
                    ) 

                log_Mc_det = res[0][:,0]
                logit_q = res[0][:,1]
                logd = res[0][:,2]
                pilik = res[1]
                

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
                if False:
                    logps, _ = pytensor.scan( lambda iobs, X, M, F, logD, logW   : 
                                       pytensor.scan( lambda ig, X, M, F, logD, logW  :
                                       -0.5*( (X[: , iobs]-M[iobs, ig]).dot( F[iobs, ig].dot( (X[ :, iobs]-M[iobs, ig]).T )) )-0.5*nd*at.log(2*atools.PI)-0.5*logD[iobs, ig]+logW[iobs, ig],  
                                               sequences = [ at.arange( ngmm ) ],
                                         non_sequences =  [vals, mus_l, icovs_l, log_dets_l, log_wts_l,  ],     
                                                      )   ,                      
                             sequences = [ at.arange(N)  ],
                             non_sequences =  [vals,  mus_l, icovs_l, log_dets_l, log_wts_l
                                              ]
                            )

                else:

                    logps, _ = pytensor.scan( lambda iobs, X, M, F, logD, logW   : 
                                       pytensor.scan( lambda ig, X, M, F, logD, logW  :
                                       -0.5 * at.sum((X[: , iobs]-M[iobs, ig]) * (F[iobs, ig] @ (X[: , iobs]-M[iobs, ig])[:, None])[:, 0])- 0.5 * nd * at.log(2 * atools.PI)- 0.5 * logD[iobs, ig]+ logW[iobs, ig],  
                                               sequences = [ at.arange( ngmm ) ],
                                         non_sequences =  [vals, mus_l, icovs_l, log_dets_l, log_wts_l,  ],     
                                                      )   ,                      
                             sequences = [ at.arange(N)  ],
                             non_sequences =  [vals,  mus_l, icovs_l, log_dets_l, log_wts_l
                                              ]
                            )



                gwl = at.logsumexp(logps, axis=1) # sum on gmm components
        
            
            else:
                raise NotImplementedError()


            Mc = at.exp(log_Mc_det)            
            q = atools.inv_logitat(logit_q)
            m1det, m2det = atools.m1m2_from_Mcq_at(Mc, q)
            d = pm.Deterministic('dL', at.exp(logd) , dims="event_index")
    
            
            # Compute source-frame quantities. One redsfhit, mass1, mass2 for each event
            if not is_GP_dL:
                
                zs = pm.Deterministic('z', atools.z_from_dL_at(d, H0_, Om_, w0_, Lambda_MG_ , is_GP_dL ), dims= "event_index" )
                distance_ratio , d_distance_ratio_d_z = None, None
            
            else:
                
                if rescale_GP:
                    data_range=(atools.zGridGlobals_at.min(), atools.zGridGlobals_at.max())
                else:
                    data_range=None
                
                dLGrid_at, log_distance_ratio, grad_log_distance_ratio = atools.z_from_dL_at(None, H0_, Om_, w0_, Lambda_MG_ , is_GP_dL, data_range=data_range, GP_zero_point=GP_zero_point )

                zs = pm.Deterministic('z', atools.atinterp( d, dLGrid_at, atools.zGridGlobals_at ) , dims= "event_index" ) 
                d_log_distance_ratio_d_z = atools.atinterp( zs, atools.zGridGlobals_at, grad_log_distance_ratio )        
                distance_ratio = pm.Deterministic( "d_ratio", at.exp(atools.atinterp( zs, atools.zGridGlobals_at, log_distance_ratio )))
                d_distance_ratio_d_z = pm.Deterministic( "d_ratio_d_z", d_log_distance_ratio_d_z*distance_ratio)

                if monotonicity:
                    # Probit model: P(f′ > 0) = Φ(f′ / ν)
                    # scale of transition to zero
                    ν = pm.HalfNormal("ν", sigma=0.05)
                    Φ = pm.Deterministic("Φ", pm.math.invprobit(pm.math.clip(d_distance_ratio_d_z / ν, -10, 10)))
                    # Binary likelihood: all 1s (indicating positive slope)
                    _ = pm.Bernoulli("monotonicity", p=Φ, observed=at.ones(N).eval() )
                
                     
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
            
            log_p_pop = log_p_pop_at( logMc_src, logit_q, zs, d, spins, Lambda_, rate_model, mass_model, spin_model, is_GP_dL, dr_val=distance_ratio, ddr_dz=d_distance_ratio_d_z)
            # ... so remove a jacobian : p( m1, m2 ) = p( log(Mc), logit(q) ) * |J|
            log_p_pop -=  at.log(m2src) + at.log(m1src-m2src) + at.log1p(zs) 
            
        else:    
        
            log_p_pop = log_p_pop_at(m1src, m2src, zs, d, spins, Lambda_, rate_model, mass_model, spin_model, is_GP_dL, pairing=pairing, dr_val=distance_ratio, ddr_dz=d_distance_ratio_d_z)
             
        
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
                    d_log_distance_ratio_d_z_inj = atools.atinterp( zinj, atools.zGridGlobals_at, grad_log_distance_ratio )        
                    distance_ratio_inj = at.exp(atools.atinterp( zinj, atools.zGridGlobals_at, log_distance_ratio ))
                    d_distance_ratio_d_z_inj = d_log_distance_ratio_d_z_inj*distance_ratio_inj
                    
                log_mu_, Neff_, var_ll_u_ = sel_bias_with_uncertainty_at( m1inj[0], m2inj[0], dLinj[0], spinsInj, lpdinj[0], Lambda_, Ndraw, rate_model, mass_model, spin_model_name, is_GP_dL, pairing, distance_ratio=distance_ratio_inj, d_distance_ratio_d_z=d_distance_ratio_d_z_inj, zinj=zinj )
                
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

                    res_i, _ = pytensor.scan( lambda idata, m1inj_, m2inj_, dLinj_, spinsInj_, lpdinj_, L,  Ndraw_, Ndet_ : sel_bias_with_uncertainty_at( m1inj_[idata, : Ndet_[idata]], m2inj_[idata, : Ndet_[idata]], dLinj_[idata, :Ndet_[idata]],  spinsInj_[idata, :, :Ndet_[idata]], lpdinj_[idata, :Ndet_[idata]], L, Ndraw_[idata], rate_model, mass_model, spin_model_name, is_GP_dL, pairing, ), 
                                          sequences = [ at.arange( ndata) ], 
                                          non_sequences = [m1inj, m2inj, dLinj, spinsInj, lpdinj, Lambda_,  Ndraw, Ndet] )
                    log_mu_vec = res_i[0]
                    Neff_ = at.sum(res_i[1])

                    
                else:
                    print("Loop over injections sets, no slicing")
                    # makes it jax-compatible (jax does not support dynamical slicing at the moment)
                    # Not true anymore after pymc v5.10 ? Check
                    res_i, _ = pytensor.scan( lambda idata, m1inj_, m2inj_, dLinj_, spinsInj_, lpdinj_, L,  Ndraw_ : sel_bias_with_uncertainty_at( m1inj_[idata ], m2inj_[idata ], dLinj_[idata], spinsInj_[idata],  lpdinj_[idata], L, Ndraw_[idata], rate_model, mass_model, spin_model, is_GP_dL, pairing, zinj=zinj ), 
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

