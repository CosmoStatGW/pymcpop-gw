#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import pytensor_tools as atools
import pytensor.tensor as at
import pytensor
import pymc as pm
import numpy as np

PLPeakO3params = {'H0': 67.66, 'Om':0.31, 'w0':-1, 'Xi0': 1, 'nXi0':0}


#####################################################


def log_p_pop_at(m1s, m2s, z, dL, spins, Lambda, rate_model, mass_model, spin_model):

    ###################################
    # get parameters and compute log p_pop
    ####################################
    
    H0, Om, w0, Xi0, n = Lambda[:5] 

    ##################################
    # redshift 
    
    if rate_model=='MD':
        
        gamma, kappa, zp = Lambda[5:8]
        lpz = atools.log_p_z_MD_unnorm(z, gamma, kappa, zp, H0, Om, w0, dc=dL/(1+z))
        istart = 8
        
    elif rate_model=='PL':
        
        gamma = Lambda[5]
        lpz = atools.log_p_z_PL_unnorm(z, gamma, H0, Om, w0, dc=dL/(1+z))
        istart = 6

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
    
    else:
        lpspin = at.zeros( z.shape )

    
    ###################################
    # mass
    
    if mass_model=='PLPreg':
        
        lp, al, bb, dm, ml, mh, muM, sM = Lambda[-8:]
        lpmass = atools.logpdf_PLP_reg([m1s, m2s], [lp, al, bb, dm, ml, mh, muM, sM])
        
    elif mass_model=='BNSgauss':
        muM, sM = Lambda[-2:]
        lpmass = atools.logpdf_gauss([m1s, m2s], [muM, sM] )

    ###################################
    # jacobian
    
    log_ddL_dz = atools.log_ddL_dz(z, H0, Om, w0, Xi0, n, dL=dL)

    
    ###################################
    # return log pdf
    ####################################
    
    lp =  lpz - log_ddL_dz - 2*at.log1p(z) + lpmass + lpspin

    return lp



def sel_bias_with_uncertainty_at(m1inj, m2inj, dLinj, spinsInj, log_p_draw, Lambda,  Ndraw, rate_model, mass_model, spin_model):


    H0, Om, w0, Xi0, n  = Lambda[:5]

    if spin_model=='default':
        spinsInj_sel = [spinsInj[0], spinsInj[1], spinsInj[2], spinsInj[3]]
    elif spin_model=='none':
        spinsInj_sel = []
    
    zinj = atools.z_from_dL_at(dLinj, H0, Om, w0, Xi0, n  )
    m1Src  = m1inj/(1+zinj)
    m2Src  = m2inj/(1+zinj)

    log_p_pop = log_p_pop_at(m1Src, m2Src, zinj, dLinj, spinsInj_sel, Lambda, rate_model, mass_model, spin_model)

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


def make_model(  priors,
                 GWData,
                 InjData,
                 sampling_GW = 'gmm',
                 rate_model = 'MD',
                 mass_model = 'PLP',
                 spin_model = 'none',
                 spin_inj = 'none',
                 marginal_R0 = True,
                 dLprior = 'none',
                 spinprior = False,
                massprior = False,
                 fix_inj_len = False,
                 sel_method='Tobs',
                 N_DP_comp_max = 10,
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
               use_log_alpha_beta=False      
                ):

    ################################################
    # Read in data and set dimensions
    ################################################

    ## GW data
    if not pop_only:
        # gw data are interpolants of single-event posteriors
        wts_l, mus_l, cho_covs_l, Tobs = GWData

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
    
            
            if spin_model=='default':
                chi1 = atools.inv_logitat(spin_samples[0])
                chi2 = atools.inv_logitat(spin_samples[1])
                cost1 = atools.inv_flogitat(spin_samples[2])
                cost2 = atools.inv_flogitat(spin_samples[3])
                spin_samples = [chi1, chi2, cost1, cost2]

            m1det, m2det = atools.m1m2_from_Mcq_at(at.exp(lMc), qs )
            print(m1det.eval())
            d = at.exp(ld)
            

        if Nsamplesuse !=-1 :
            if Nsamplesuse>allNsamples:
                raise ValueError("Must use less samples than those available.")
            print("allNsamples availabe is %s, but %s will be used"%(allNsamples, Nsamplesuse))
            allNsamples =  Nsamplesuse        
        
        if spin_model=='default':
           chi1, chi2, cost1, cost2 = spin_samples
        else:
            raise NotImplementedError()

    ## Injections data
    if spin_inj == 'none':
        dLinj, m1inj, m2inj, lpdinj, Ndraw, Ndet = InjData
    elif spin_inj == 'chieffchip':
        dLinj, m1inj, m2inj, chiefffInj, chipInj, lpdinj, Ndraw, Ndet = InjData
    elif (spin_inj == 'chi12xyz' or spin_inj == 'default'):
        if spin_model=='default':
            dLinj, m1inj, m2inj, chi1Inj, chi2Inj, cost1Inj, cost2Inj, lpdinj, Ndraw, Ndet = InjData
        elif spin_model == 'none':
            dLinj, m1inj, m2inj, lpdinj, Ndraw, Ndet = InjData

        
        
    if not pop_only:
        N = mus_l.shape[0] # number of events in total
        ngmm = mus_l.shape[1]
        nd = mus_l.shape[2]
        print('N:%s, max ngmm: %s, nd: %s '%(N.eval(), ngmm.eval(), nd.eval()))
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

    if pop_only:
        coords['nsamples'] = at.arange( Nsamples ).eval()
    else:
         coords['GWdimension'] = at.arange(nd).eval()


    if params_fix is None:
        print('No values for parameters to fix passed. Default values will be used. If fixing parameters, check that the values are consistent. Values of fixed parameters:')
        print(PLPeakO3params)
        params_fix=PLPeakO3params
        
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
        
        if fix_Xi0n:
            Xi0_ =  at.as_tensor_variable(1.)
            nXi0_ = at.as_tensor_variable(0.)
        else:
            Xi0_ =  pm.Uniform('Xi0', lower=priors['Xi0'][0], upper=priors['Xi0'][1])
            nXi0_ = pm.Uniform('n', lower=priors['n'][0], upper=priors['n'][1]) 

        Lambda_ = [H0_, Om_, w0_, Xi0_, nXi0_]

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
                    _ = pm.Potential('bound_alphaChi', atools.log_f_smooth_poly(alphaChi_, 5e-4,  1 )  )
                    _ = pm.Potential('bound_betaChi', atools.log_f_smooth_poly(betaChi_, 5e-4,  1  ))
                elif alpha_beta_prior=='sigmoid':
                    print("Tapering prior on alpha_chi, beta_chi with sigmoid smoothing")
                    _ = pm.Potential('bound_alphaChi', atools.log_sigmoid(alphaChi_,  1+3e-04, 1e-04)  )
                    _ = pm.Potential('bound_betaChi', atools.log_sigmoid(betaChi_, 1+3e-04, 1e-04)  )
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

        else:
            print('No model of the spin distribution.')
                

            

        ################################################
        # Mass distribution
        ################################################
            
        if mass_model=='PLPreg':

            # Power law + peak
            print('Modeling mass distribution with LVK Power Law + Peak with regularized edge')
            
            lamP_ = pm.Uniform('lambdaPeak', lower=priors['lambdaPeak'][0], upper=priors['lambdaPeak'][1])
            alpha_ = pm.Uniform('alpha', lower=priors['alpha'][0], upper=priors['alpha'][1])
            beta_ = pm.Uniform('beta', lower=priors['beta'][0], upper=priors['beta'][1])
            ml_ = pm.Uniform('ml', lower=priors['ml'][0], upper=priors['ml'][1])
            mh_ = pm.Uniform('mh', lower=priors['mh'][0], upper=priors['mh'][1])
            deltam_ = pm.Uniform('deltam', lower=priors['deltam'][0], upper=priors['deltam'][1])
            muM_ = pm.Uniform('muMass', lower=priors['muMass'][0], upper=priors['muMass'][1])
            sM_ = pm.Uniform('sigmaMass', lower=priors['sigmaMass'][0], upper=priors['sigmaMass'][1] )  

            Lambda_ += [lamP_, alpha_, beta_, deltam_, ml_, mh_, muM_, sM_ ]

        elif mass_model=='BNSgauss':

            # Uncorrelated gaussians
            print('Modeling mass distribution with uncorrelated gaussian distributions')
            
            muM_ = pm.Uniform('muMass', lower=priors['muMass'][0], upper=priors['muMass'][1])
            sM_ = pm.Uniform('sigmaMass', lower=priors['sigmaMass'][0], upper=priors['sigmaMass'][1] )  
            Lambda_ += [muM_, sM_ ]


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
    
                samples = mus_l[ at.arange(N), ig, :] + at.batched_dot( cho_covs_l[at.arange(N), ig, :, :], x )
    
                log_Mc_det = samples[:,0]/dil_factor
                logit_q = samples[:,1]
                
                Mc = at.exp(log_Mc_det)            
                q = atools.inv_logitat(logit_q)
                m1det, m2det = atools.m1m2_from_Mcq_at(Mc, q)
                logd = samples[:,2]
                d = pm.Deterministic('dL', at.exp(logd) , dims="event_index")
    
                if (spin_model == 'chieffchip') or (spin_model == 'chieffchip_uc') :
        
                    chieff = atools.inv_flogitat(samples[:,3])
                    chip = atools.inv_logitat(samples[:,4])
        
                elif (spin_model == 'default'):
                    # we have chi1, chi2, cost1, cost2
        
                    chi1 = pm.Deterministic('chi1', atools.inv_logitat(samples[:,3]))
                    chi2 = pm.Deterministic('chi2', atools.inv_logitat(samples[:,4]))
        
                    cost1 = pm.Deterministic('cost1', atools.inv_flogitat(samples[:,5]))
                    cost2 = pm.Deterministic('cost2', atools.inv_flogitat(samples[:,6]))
                else:
                    print("No spins computed")
            

                # Compute source-frame quantities. One redsfhit, mass1, mass2 for each event
                zs = pm.Deterministic('z', atools.z_from_dL_at(d, H0_, Om_, w0_, Xi0_, nXi0_ ), dims= "event_index" )
                m1src = pm.Deterministic('m1src', m1det/(1+zs) , dims="event_index")
                m2src = pm.Deterministic('m2src', m2det/(1+zs) , dims="event_index") 

            else:
                # we are not using GMM for p(D|theta)
                raise NotImplementedError()
                

        else:
            # we are sampling the usual marginalise likelihood, with "only" pop parameters
            print('We are running inference only on population parameters.')


            # Compute source-frame quantities. One redsfhit, mass1, mass2 for each event
            # AND for each sample! 
            
            d_stacked  = at.flatten(d)
            zs_stacked = atools.z_from_dL_at(d_stacked, H0_, Om_ )
            
            zs = at.reshape( zs_stacked, (N, Nsamples) )
            m1src = m1det/(1+zs)
            m2src = m2det/(1+zs)
            
            logd = at.log(d)
        
        
        ################################################
        # Population prior
        ################################################

        
        if spin_model == 'chieffchip' or spin_model == 'chieffchip_uc' :

            spins = [ chieff, chip  ]

        elif spin_model == 'default':

            spins = [chi1, chi2, cost1, cost2]

        elif spin_model == 'none':
            
            spins = []
            
        
        # Population prior of all events, without the term T_obs*R0
        log_p_pop = log_p_pop_at(m1src, m2src, zs, d, spins, Lambda_, rate_model, mass_model, spin_model)

        
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

          

        # add R0*Tobs if needed. 
        if not marginal_R0:
            print("Will not marginalise over R0.")
            # each term p_pop is multiplied by
            # R0*T_obs . So we get a factor (R0*T_obs)**N_i for every
            # observing run. R0 is the same for every run so I just have
            # (R0)**{\sum N_i} . For T_obs I have T_{obs,1}**N_1 * T_{obs,2}**N_2 * ...
            poiss_term = atools.sum(Nevs*at.log(Tobs))+at.sum(Nevs)*lR0
            log_p_pop += poiss_term
        else:
            print("Will marginalise over R0 with flat-in-log prior.")


        # Put it all together
        if not pop_only:
            # just sum log likelihoods
            likelihood = pm.Deterministic("lik", at.sum( log_p_pop ) ) 
        else:
            # marginalise over single events parameters first
            # shape of p_pop is (hopefully) n_evs x n_samples
            # so average over second dimension
            
            # Compute only where there are samples
            log_p_pop_to_marg = log_p_pop[:, :allNsamples[0]]
            
            log_p_pop_marg = at.logsumexp( log_p_pop_to_marg, axis=1 ) - at.log(allNsamples)
            

            # then sum log likelihoods
            likelihood = pm.Deterministic("lik", at.sum( log_p_pop_marg ) ) 

            # Check number of effective samples for computing MC integral 
            logs2 = at.logsumexp(2*log_p_pop_masked, axis=1) -2*at.log(allNsamples)
            
            Neff_lik =  pm.Deterministic('Neff_l', at.exp( 2.0*log_p_pop_marg - logs2) ) # this has len = n. of observations
            
            if Neff_min_lik>0:
                
                #_ = pm.Potential("Neff_l_bound", at.sum( at.where( Neff_lik<Neff_min_lik*N, -atools.INF, at.as_tensor_variable(0.) ) ) )
                
                # see https://discourse.pymc.io/t/conditionally-reject-samples/3107
                ind_sw_l = pm.Deterministic('ind_l', 1. * (Neff_lik<Neff_min_lik) )
                ind_l = pm.Bernoulli('Neff_l_bound', ind_sw_l, observed=at.zeros(N).eval(), testval=at.zeros(N) )
                print(ind_l.eval())
            else:
                print("No bound on effective number of samples for individual event MC integrals")
            
        
        _ = pm.Potential("likelihood", likelihood ) 



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
                    elif spin_model == 'default':
                        spinsInj = [ chi1Inj[0], chi2Inj[0], cost1Inj[0], cost2Inj[0] ]
                    else:
                        spinsInj = []

                else:
                    print("Spin distribution will not be used in the sel effect")
                    spinsInj = []
                    spin_model_name = 'none'
                
                log_mu_, Neff_, var_ll_u_ = sel_bias_with_uncertainty_at( m1inj[0], m2inj[0], dLinj[0], spinsInj, lpdinj[0], Lambda_, Ndraw, rate_model, mass_model, spin_model_name)
                
                if not marginal_R0:
                    # This is really the number of expected events in the observing run
                    sel_effect = -at.exp(log_mu_+lR0)*Tobs
                else:
                    sel_effect = -N*log_mu_
    
            else:
                # we passed multiple injections set corresponding to multiple observing runs
                # they need to be properly combined
                # This is useful only if using older injection sets,
                # Deprecated after GWTC-3 
                
                print("Combining selection effects from different injections campaigns")

                spin_model_name = spin_model
                if use_sel_spin:

                    if spin_model == 'chieffchip' or spin_model == 'chieffchip_uc' :
    
                        spinsInj = at.zeros( (ndata, 2, ninj) )
                        spinsInj = at.set_subtensor( spinsInj[:, 0, :], chi1Inj )
                        spinsInj = at.set_subtensor( spinsInj[:, 1, :], chi2Inj )
                    
                    
                    elif spin_model == 'default':
                    
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

                    res_i, _ = pytensor.scan( lambda idata, m1inj_, m2inj_, dLinj_, spinsInj_, lpdinj_, L,  Ndraw_, Ndet_ : sel_bias_with_uncertainty_at( m1inj_[idata, : Ndet_[idata]], m2inj_[idata, : Ndet_[idata]], dLinj_[idata, :Ndet_[idata]],  spinsInj_[idata, :, :Ndet_[idata]], lpdinj_[idata, :Ndet_[idata]], L, Ndraw_[idata], rate_model, mass_model, spin_model_name ), 
                                          sequences = [ at.arange( ndata) ], 
                                          non_sequences = [m1inj, m2inj, dLinj, spinsInj, lpdinj, Lambda_,  Ndraw, Ndet] )
                    log_mu_vec = res_i[0]
                    Neff_ = at.sum(res_i[1])

                    
                else:
                    print("Loop over injections sets, no slicing")
                    # makes it jax-compatible (jax does not support dynamical slicing at the moment)
                    # Not true anymore after pymc v5.10 ? Check
                    res_i, _ = pytensor.scan( lambda idata, m1inj_, m2inj_, dLinj_, spinsInj_, lpdinj_, L,  Ndraw_ : sel_bias_with_uncertainty_at( m1inj_[idata ], m2inj_[idata ], dLinj_[idata], spinsInj_[idata],  lpdinj_[idata], L, Ndraw_[idata], rate_model, mass_model, spin_model ), 
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
            log_lik_var = pm.Deterministic('log_lik_var', at.exp(var_ll_u_+2*at.log(N)) )
     

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
            
            _ = pm.Potential('selection_bias', selection_bias)


            

    return model

