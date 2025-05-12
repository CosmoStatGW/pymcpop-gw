#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import os
import pytensor
import pytensor.tensor as at
import pymc as pm
import argparse
import json
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as np
import numpy as onp
import sys
import corner
import arviz as az
import matplotlib.pyplot as plt

print(jax.default_backend())
print(jax.devices())
print(f"Running on PyMC v{pm.__version__}")


import pymc_models as models
import data_tools as dt
import pytensor_tools as atools

os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'
pytensor.config.floatX = "float64"


parser = argparse.ArgumentParser()


parser.add_argument("--fin_data", nargs='+', type=str, required=True)
parser.add_argument("--fin_injections", nargs='+', type=str, required=True)
parser.add_argument("--fin_priors", default='', type=str, required=True)
parser.add_argument("--backend", default='disk', type=str, required=False)

parser.add_argument("--pop_only", default=0, type=int, required=False)


parser.add_argument("--rate_model", default='MD', type=str, required=False)
parser.add_argument("--mass_model", default='PLPreg', type=str, required=False)
parser.add_argument("--spin_model", default='none', type=str, required=False)
parser.add_argument("--N_DP_comp_max", default=10, type=int, required=False)
parser.add_argument("--marginal_R0", default=1, type=int, required=False)
parser.add_argument("--pairing", default=1, type=int, required=False)


parser.add_argument("--dLprior", default='none', type=str, required=False)
parser.add_argument("--spinprior", default=0, type=int, required=False)
parser.add_argument("--massprior", default=0, type=int, required=False)
parser.add_argument("--use_sel_spin", default=1, type=int, required=False)


parser.add_argument("--sampling_gw", default='gmm', type=str, required=False)
parser.add_argument("--cho_dil", default=1., type=float, required=False)
parser.add_argument("--sel", default='Tobs', type=str, required=False)
parser.add_argument("--ivals", default='', type=str, required=False)
parser.add_argument("--params_fix", default='', type=str, required=False)


parser.add_argument("--n_inj_use", nargs='+', type=float, required=False)
parser.add_argument("--fix_inj_len", default=0, type=int, required=False)
parser.add_argument("--min_Neff", default=0, type=int, required=False)
parser.add_argument("--Neff_min_lik", default=0, type=int, required=False)
parser.add_argument("--log_lik_var_min", default=1, type=float, required=False)

parser.add_argument("--nsamplesmax", default=-1, type=int, required=False)
parser.add_argument("--spin_inj", default='none', type=str, required=False)
parser.add_argument("--Nsamplesuse", default=-1, type=int, required=False)
parser.add_argument("--transform_samples", default=1, type=int, required=False)
parser.add_argument("--sel_uncertainty", default=0, type=int, required=False)
parser.add_argument("--sel_smoothing", default='sigmoid', type=str, required=False)
parser.add_argument("--alpha_beta_prior", default='sigmoid', type=str, required=False)
parser.add_argument("--dil_factor", default=1, type=int, required=False)
parser.add_argument("--use_log_alpha_beta", default=0, type=int, required=False)

parser.add_argument("--fout", default='results/', type=str, required=True)

parser.add_argument("--sampler", default='pymc', type=str, required=False)
parser.add_argument("--nsteps", default=100, type=int, required=True)
parser.add_argument("--ntune", default=100, type=int, required=True)
parser.add_argument("--nchains", default=1, type=int, required=False)
parser.add_argument("--ncores", default=1, type=int, required=False)
parser.add_argument("--target_accept", default=0.8, type=float, required=False)
parser.add_argument("--fix_H0", default=1, type=int, required=False)
parser.add_argument("--fix_Om", default=1, type=int, required=False)
parser.add_argument("--fix_w0", default=1, type=int, required=False)
parser.add_argument("--fix_Xi0n", default=1, type=int, required=False)

parser.add_argument("--allTobs", nargs='+', type=float, required=False)



if __name__=='__main__':

    
    FLAGS = parser.parse_args()

    logfile = os.path.join(FLAGS.fout, 'logfile.txt')
    myLog = dt.Logger(logfile)
    sys.stdout = myLog
    sys.stderr = myLog


    with open(FLAGS.fin_priors) as json_file:
        priors = json.load(json_file)
    
    # save input params for memory
    with open(os.path.join(FLAGS.fout, 'input_args.json' ), 'w') as fp:
        json.dump(vars(FLAGS), fp)

    # save priors for memory
    with open(os.path.join(FLAGS.fout, 'priors.json' ), 'w') as fp:
        json.dump(priors, fp)

    if FLAGS.params_fix!='':
        with open(FLAGS.params_fix) as json_file:
            params_fix = json.load(json_file) 
    else:
        params_fix=None

    ################################################
    # Load data
    ################################################
    
    # load sample means and covs
    print()
    print('*'*80)
    print('Loading data...')
    print('*'*80)
    print()



    
    if not FLAGS.pop_only:

        data = dt.load_data_interp(FLAGS.fin_data)

        samples_means_at = at.as_tensor_variable(data['samples_means'])
        samples_cho_covs_at = at.as_tensor_variable(data['samples_cho_covs']*FLAGS.cho_dil)
    
        gmm_log_wts = at.as_tensor_variable(data['gmm_log_wts'])
        gmm_means = at.as_tensor_variable(data['gmm_means'])
        gmm_icovs = at.as_tensor_variable(data['gmm_icovs'])
        gmm_cho_covs = at.as_tensor_variable(data['gmm_cho_covs'])
        gmm_log_dets = at.as_tensor_variable(data['gmm_log_dets'])
        allNgm = at.as_tensor_variable(data['allNgm'])
        Nevents = at.as_tensor_variable(data['Nevents'])

    else:
        print("Using n max samples = %s"%FLAGS.nsamplesmax)
        data = dt.load_data_samples(FLAGS.fin_data, nmax=FLAGS.nsamplesmax)

        m1d_samples = at.as_tensor_variable(data['m1d_samples'])
        m2d_samples = at.as_tensor_variable(data['m2d_samples'])
        dL_samples = at.as_tensor_variable(data['dL_samples'])
        print("dL_samples shape is %s"%(str(dL_samples.shape)))

        allNsamples = at.as_tensor_variable(data['allNsamples'])
        where_compute = at.as_tensor_variable(data['where_compute'])

        if (FLAGS.spin_model=='default') or (FLAGS.spin_model=='default_gauss'):

            chi1_samples = at.as_tensor_variable(data['chi1_samples'])
            chi2_samples = at.as_tensor_variable(data['chi2_samples'])
            cost1_samples = at.as_tensor_variable(data['cost1_samples'])
            cost2_samples = at.as_tensor_variable(data['cost2_samples'])

            spin_samples = [ chi1_samples, chi2_samples, cost1_samples, cost2_samples ]

        elif FLAGS.spin_model=='none':
            spin_samples = [  ]
        else:
            raise NotImplementedError()

    
    
            
    print("Done.")
    

    # load injections

    print()
    print('*'*80)
    print('Loading injections...')
    print('*'*80)
    print()


    injections = dt.load_injections(FLAGS.fin_injections, allPercUse=FLAGS.n_inj_use)


    if FLAGS.spin_model=='none':
        InjData = [ at.as_tensor_variable(injections['dL']), 
                at.as_tensor_variable(injections['m1d']), 
                at.as_tensor_variable(injections['m2d']), 
                at.as_tensor_variable(injections['log_wt']), 
                at.as_tensor_variable(injections['Ngen']), 
                at.as_tensor_variable(injections['Ndet']), 
                  ]
    else:
        
        if FLAGS.spin_inj=='chieffchip':
            InjData = [ at.as_tensor_variable(injections['dL']), 
                at.as_tensor_variable(injections['m1d']), 
                at.as_tensor_variable(injections['m2d']), 
                at.as_tensor_variable(injections['chieff']), 
                at.as_tensor_variable(injections['chip']), 
                at.as_tensor_variable(injections['log_wt']), 
                at.as_tensor_variable(injections['Ngen']), 
                at.as_tensor_variable(injections['Ndet']), 
                  ]
        elif FLAGS.spin_inj=='chi12xyz':

            if (FLAGS.spin_model=='default') or (FLAGS.spin_model=='default_gauss'):

                print("Computing chi1, chi2, cost1, cost2 in injections...")
    
                chi1Inj = onp.sqrt(injections['spin1x']**2+injections['spin1y']**2+injections['spin1z']**2)
                chi2Inj = onp.sqrt(injections['spin2x']**2+injections['spin2y']**2+injections['spin2z']**2)
    
                cost1Inj = injections['spin1z']/chi1Inj
                cost2Inj = injections['spin2z']/chi2Inj
                
                InjData = [ at.as_tensor_variable(injections['dL']), 
                    at.as_tensor_variable(injections['m1d']), 
                    at.as_tensor_variable(injections['m2d']), 
                    at.as_tensor_variable(chi1Inj), 
                    at.as_tensor_variable(chi2Inj),
                    at.as_tensor_variable(cost1Inj),
                    at.as_tensor_variable(cost2Inj),
                    at.as_tensor_variable(injections['log_wt']), 
                    at.as_tensor_variable(injections['Ngen']), 
                    at.as_tensor_variable(injections['Ndet']), 
                      ]

            elif FLAGS.spin_model=='none':

                print("Injections data has spins but those will not be used !")
    
                InjData = [ at.as_tensor_variable(injections['dL']), 
                    at.as_tensor_variable(injections['m1d']), 
                    at.as_tensor_variable(injections['m2d']), 
                    at.as_tensor_variable(injections['log_wt']), 
                    at.as_tensor_variable(injections['Ngen']), 
                    at.as_tensor_variable(injections['Ndet']), 
                      ]
                
        elif FLAGS.spin_inj=='default':

                InjData = [ at.as_tensor_variable(injections['dL']), 
                    at.as_tensor_variable(injections['m1d']), 
                    at.as_tensor_variable(injections['m2d']), 
                    at.as_tensor_variable(injections['chi1']), 
                    at.as_tensor_variable(injections['chi2']),
                    at.as_tensor_variable(injections['cost1']),
                    at.as_tensor_variable(injections['cost2']),
                    at.as_tensor_variable(injections['log_wt']), 
                    at.as_tensor_variable(injections['Ngen']), 
                    at.as_tensor_variable(injections['Ndet']), 
                      ]

    
            
    if not FLAGS.pop_only:  
    
        if FLAGS.sampling_gw=='gmm':
            GWData =  [
                       at.exp(gmm_log_wts), 
                       gmm_means, 
                       gmm_cho_covs, 
                       at.as_tensor_variable(injections['Tobs']),
                        Nevents
                      ]
        elif FLAGS.sampling_gw=='gauss':
            GWData =  [samples_means_at, 
                       samples_cho_covs_at, 
                       gmm_log_wts, 
                       gmm_means, 
                       gmm_icovs, 
                       gmm_log_dets, 
                       at.as_tensor_variable(injections['Tobs']),
                       Nevents, 
                      ]
            

    else:
        GWData = [ m1d_samples, m2d_samples, dL_samples, spin_samples, #Nevents, 
                       at.as_tensor_variable(injections['Tobs']), allNsamples, where_compute ]
        
        
    print("Done.")


    ################################################
    # Build model
    ################################################
    
    print()
    print('*'*80)
    print('Building model...')
    print('*'*80)
    print()

    if FLAGS.pop_only:
        N = m1d_samples.shape[0]
        N_successes_l = np.ones(N.eval())
    else:   
        N_successes_l = None
    
    model = models.make_model(  priors,
                                    GWData,
                                    InjData,
                                    sampling_GW = FLAGS.sampling_gw,
                                    rate_model = FLAGS.rate_model,
                                    mass_model = FLAGS.mass_model,
                                    pairing=FLAGS.pairing,
                                    spin_model = FLAGS.spin_model,
                                    spin_inj = FLAGS.spin_inj,
                                    dLprior = FLAGS.dLprior,
                                    sel_method=FLAGS.sel,
                                    fix_inj_len=FLAGS.fix_inj_len,
                                    marginal_R0 = FLAGS.marginal_R0,
                                    N_DP_comp_max = FLAGS.N_DP_comp_max,
                                    fix_H0 = FLAGS.fix_H0,
                                    fix_Om = FLAGS.fix_Om,
                                    fix_w0 = FLAGS.fix_w0,
                                    fix_Xi0n = FLAGS.fix_Xi0n,
                                    Neff_min=FLAGS.min_Neff,
                                    Neff_min_lik = FLAGS.Neff_min_lik,
                                    log_lik_var_min = FLAGS.log_lik_var_min,
                                    use_sel_spin=FLAGS.use_sel_spin,
                                    pop_only = FLAGS.pop_only,
                                    N_successes_l = N_successes_l,
                                    Nsamplesuse = FLAGS.Nsamplesuse,
                                    transform_samples = FLAGS.transform_samples,
                                    include_sel_uncertainty = FLAGS.sel_uncertainty,
                                    sel_smoothing = FLAGS.sel_smoothing,
                                    alpha_beta_prior = FLAGS.alpha_beta_prior,
                                    dil_factor=FLAGS.dil_factor,
                                    use_log_alpha_beta=FLAGS.use_log_alpha_beta,
                                    params_fix=params_fix,
                                      allTobs=FLAGS.allTobs
                                )

    print('Done.')

    print()
    print('*'*80)
    print('Running inference...')
    print('*'*80)
    print()
    
    if FLAGS.backend=='disk':
        backend=None
    else:
        # for saving see https://discourse.pymc.io/t/saving-intermediate-results-using-mcmc-in-pymc4/9938
        # Not well tested
        import clickhouse_driver
        import mcbackend
        ch_client = clickhouse_driver.Client("localhost")
        backend = mcbackend.ClickHouseBackend(ch_client)

    
    ################################################
    # Find initial point
    ################################################
    
    ivals = None
    if FLAGS.ivals!='':


        with open(FLAGS.ivals) as json_file:
            ivals = json.load(json_file) 
        
        if FLAGS.marginal_R0:
            try:
                _ =  ivals.pop("R0")
            except:
                pass
        
        if FLAGS.rate_model=='PL':
            _ = ivals.pop('kappa')
            _ = ivals.pop('zp')
            

        if FLAGS.mass_model=='PLP' or FLAGS.mass_model=='PLPreg':
            try:
                _ = ivals.pop('sl')
                _ = ivals.pop('sh')
            except:
                pass

            
        if FLAGS.spin_model=='none':
            try:
                _ = ivals.pop('muEff')
                _ = ivals.pop('sigEff')
                _ = ivals.pop('muP')
                _ = ivals.pop('sigP')
                _ = ivals.pop('rho')
            except:
                pass
            try:
                _ = ivals.pop('muChi')
                _ = ivals.pop('varChi')
                _ = ivals.pop('zeta')
                _ = ivals.pop('sigmat')
                _ = ivals.pop('sigmaChi')
            except:
                pass
        elif FLAGS.spin_model=='chieffchip_uc':
            _ = ivals.pop('rho')
            try:
                _ = ivals.pop('muChi')
                _ = ivals.pop('varChi')
                _ = ivals.pop('zeta')
                _ = ivals.pop('sigmat')
                _ = ivals.pop('sigmaChi')
            except:
                pass
        elif FLAGS.spin_model=='default':
            try:
                _ = ivals.pop('sigmaChi')
                _ = ivals.pop('muEff')
                _ = ivals.pop('sigEff')
                _ = ivals.pop('muP')
                _ = ivals.pop('sigP')
                _ = ivals.pop('rho')
            except:
                pass
            if FLAGS.use_log_alpha_beta:
                # need to pass good initial values here 
                muChi_ = ivals.pop('muChi')
                varChi_ = ivals.pop('varChi')
                kappa_ = muChi_*(1-muChi_)/varChi_-1
    
                alphaChi_ =  muChi_*kappa_ 
                betaChi_ =  (1-muChi_)*kappa_ 
                
                ivals["logAlphaMinusOne"] = np.log(alphaChi_-1)
                ivals["logBetaMinusOne"] = np.log(betaChi_-1)
        elif FLAGS.spin_model=='default_gauss':
            try:
                _ = ivals.pop('varChi')
                _ = ivals.pop('muEff')
                _ = ivals.pop('sigEff')
                _ = ivals.pop('muP')
                _ = ivals.pop('sigP')
                _ = ivals.pop('rho')
            except:
                pass
            
        if FLAGS.fix_H0:
            _ = ivals.pop('H0')
        if FLAGS.fix_Om:
            _ = ivals.pop('Om')
        if FLAGS.fix_w0:
            try:
                _ = ivals.pop('w0')
            except:
                pass
        if FLAGS.fix_Xi0n:
            _ = ivals.pop('Xi0')
            _ = ivals.pop('n')
        
        
        print("Parameters names: %s" %str(list(ivals.keys())))
        vplot = list(ivals.keys())
        
        if FLAGS.ivals!='':

            print("Setting user-provided intial values...")
        
            for k in ivals.keys():
                sig_init = ivals[k]/100
                good = False
                iter = 0
                while not good:
                    ivals[k] += onp.random.randn()*sig_init
                    if (ivals[k] < priors[k][0]) | (ivals[k] > priors[k][1]) :
                        good=False
                        sig_init/=2
                        iter += 1
                    else:
                        good=True
                        
                    if iter==100:
                        print("not able to initialize %s. Value: %s. Prior range: %s, %s"%(k,ivals[k] , priors[k][0], priors[k][1]))
                        raise ValueError('Initialization failed! Check your prior ranges and initial values.')
                        
                        
                
            
            print(ivals)
            print()
            
            
            if not FLAGS.pop_only:
                N = gmm_log_wts.shape[0].eval()
                nd = gmm_means.shape[2].eval()
                    
                ivals['x'] = onp.random.randn(len(data['gmm_means']), nd)*0.01
    
                if FLAGS.sampling_gw=='gmm': # this is ok with spins
    
                    is_init_good = False

                    # Initialize each event around the highest weight gaussian component
                    ivals['idx'] = at.exp(gmm_log_wts).eval().argmax(axis=1)
                    
                    ncomp = gmm_log_wts.shape[1].eval()
    
                    idx_init = -1
                    it = 1
                    while not is_init_good:
                    
                        # check that m1, m2, d are inside prior range
                        samples = gmm_means[ at.arange(N), ivals['idx'], :] + at.batched_dot(gmm_cho_covs[at.arange(N), ivals['idx'], :, :], ivals['x']  )
                        
                        Mc = at.exp(samples[:,0]/FLAGS.dil_factor)            
                        q = atools.inv_logitat(samples[:,1])
                        m1det, m2det = atools.m1m2_from_Mcq_at(Mc, q)
                        logd = samples[:,2]
                        d = at.exp(logd)
                        
                        zs = atools.z_from_dL_at(d, models.PLPeakO3params['H0'], models.PLPeakO3params['Om'], models.PLPeakO3params['w0'], models.PLPeakO3params['Xi0'], models.PLPeakO3params['nXi0'] )
                        m1src = m1det/(1+zs)
                        m2src = m2det/(1+zs)
                 
                        c1 = np.any(m1src.eval()>priors['mh'][1])
                        c2 = np.any(m2src.eval()<priors['ml'][0])
                     
                        
                        if c1 | c2:
                            idx_init -=1
                            it+=1
                            if c1:
                                where_out_1 = np.argwhere(m1src.eval()>priors['mh'][1])
                                irep = list(where_out_1[0])
                                #raise ValueError('Initial m1 is larger than max mass at positions %s '%str(where_out))
                                print('Initial m1 is larger than max mass at positions %s '%str(where_out_1))
                                print("Prior value for m_max is %s"%priors['mh'][1])
                                print("Got mass values :")
                                print( str(m1src.eval()[where_out_1]))
                               
                            if c2:
                                where_out_2 = np.argwhere(m2src.eval()<priors['ml'][0])
                                irep += list(where_out_2[0])
                                #raise ValueError('Initial m2 is lower than min mass at positions %s '%str(where_out))
                                print(('Initial m2 is lower than min mass at positions %s . Min mass: %s'%(str(where_out_2),priors['ml'][0])))
                              
                        else:
                            is_init_good = True
                        
                        if not is_init_good:
                            
                            irep = np.squeeze(np.asarray(irep))
                            if np.ndim(irep)==0:
                                irep = np.asarray([irep])
                            print('Replacing init masses at positions %s'%str(irep))
                            for idx in irep:
                        
                                if m1src.eval()[idx]>priors['mh'][1] :
                                    # Find GMM component that gives minimum m1 and continue resampling from that one
                                    
                                    s = gmm_means.eval()[idx, :][~np.isinf(gmm_log_wts.eval()[idx]), :]
                                    Mc_ = at.exp(s[:,0])            
                                    q_ = atools.inv_logitat(s[:,1])
                                    m1det_, m2det_ = models.m1m2_from_Mcq_at(Mc_, q_)
                                    logd_ = s[:,2]
                                    d_ = at.exp(logd_)
                    
                                    
                                    zs_ = atools.z_from_dL_at(d_, models.PLPeakO3params['H0'], models.PLPeakO3params['Om'] )
                                    m1src_ = m1det_/(1+zs_)
                                    m2src_ = m2det_/(1+zs_)
                       
                    
                                    idx_init_tmp = m1src_.eval().argmin() 
                                    idx_init = np.squeeze(np.argwhere( gmm_means.eval()[idx][:, 0]== s[idx_init_tmp,0] ))
    
                                    ivals['idx'][idx] = idx_init
                        
                                    ivals['x'][idx] = onp.random.randn(1,nd)*0.1
                        
                        if idx_init<=-ncomp:
                            raise ValueError('Initialization of masses failed. Check prior range.')
                
                elif FLAGS.sampling_gw=='gauss': # this might not work with spins


                    
                    N = gmm_means.shape[0] # number of events in total
                    ngmm = gmm_means.shape[1]
                    nd = gmm_means.shape[2]


                    res, _ = pytensor.scan( lambda iev, X, M, L: models.get_sample_from_cho_lMclqld( X[iev], M[iev], L[iev]),
                                        sequences = [at.arange(N)],
                                        non_sequences = [ ivals['x'], samples_means_at, samples_cho_covs_at]
                        ) 

                    
                    log_Mc_det = res[0][:,0]
                    logit_q = res[0][:,1]
                    logd = res[0][:,2]
                    pilik = res[1]

                    #print('Initial sample ok.')

                    if FLAGS.spin_model == 'default' :

                        chi1 = atools.inv_logitat(res[0][:,3])
                        chi2 = atools.inv_logitat(res[0][:,4])
            
                        cost1 = atools.inv_flogitat(res[0][:,5])
                        cost2 = atools.inv_flogitat(res[0][:,6])
                    
                    Mc = at.exp(log_Mc_det)            
                    q = atools.inv_logitat(logit_q)
                    m1det, m2det = atools.m1m2_from_Mcq_at(Mc, q)
                    d = at.exp(logd)

                    #print('Mc')
                    #print(Mc.eval())
                
                
                
                    zs = atools.z_from_dL_at(d, models.PLPeakO3params['H0'], models.PLPeakO3params['Om'], models.PLPeakO3params['w0'], models.PLPeakO3params['Xi0'], models.PLPeakO3params['nXi0'] )
                    m1src = m1det/(1+zs)
                    m2src = m2det/(1+zs)

                    #print('zs')
                    #print(zs.eval())
    
                    c1 = np.any(m1src.eval()>priors['mh'][1])
                    c2 = np.any(m2src.eval()<priors['ml'][0])
    
                    if c1 | c2:
                        #idx_init -=1
                        #it+=1
                        print("Initial conditions not good ")
                        if c1:
                            print("m1 source outside mh prior range for:")
                            print( np.where(m1src.eval()>priors['mh'][1]) )
                            print(m1src.eval()[m1src.eval()>priors['mh'][1]])
                        if c2:
                            print("m2 source outside ml prior range for:")
                            print( np.where(m2src.eval()<priors['ml'][0]) )
                            print(m2src.eval()[m2src.eval()>priors['ml'][0]])
                            
    
                    if FLAGS.spin_model == 'default' :
                        vals = at.zeros( (7, N) )
                
                        vals = at.set_subtensor( vals[0], log_Mc_det )
                        vals = at.set_subtensor( vals[1], logit_q )
                        vals = at.set_subtensor( vals[2], logd )
                        vals = at.set_subtensor( vals[3], res[0][:,3] )
                        vals = at.set_subtensor( vals[4], res[0][:,4] )
                        vals = at.set_subtensor( vals[5], res[0][:,5] )
                        vals = at.set_subtensor( vals[6], res[0][:,6] )

                    else:
                        vals = at.zeros( (3, N) )
                
                        vals = at.set_subtensor( vals[0], log_Mc_det )
                        vals = at.set_subtensor( vals[1], logit_q )
                        vals = at.set_subtensor( vals[2], logd )
                    
                       
                    logps, _ = pytensor.scan( lambda iobs, X, M, F, logD, logW   : 
                                           pytensor.scan( lambda ig, X, M, F, logD, logW  :
                                           -0.5*( (X[: , iobs]-M[iobs, ig]).dot( F[iobs, ig].dot( (X[ :, iobs]-M[iobs, ig]).T )) )-0.5*nd*at.log(2*atools.PI)-0.5*logD[iobs, ig]+logW[iobs, ig],  
                                                   sequences = [ at.arange( ngmm ) ],
                                             non_sequences =  [vals,  gmm_means, gmm_icovs, gmm_log_dets, gmm_log_wts ],     
                                                          )   ,                      
                                 sequences = [ at.arange(N)  ],
                                 non_sequences =  [vals,  gmm_means, gmm_icovs, gmm_log_dets, gmm_log_wts
                                                  ]
                                )
                    gwl = at.logsumexp(logps, axis=1)

                    
                    

                else:
                    raise ValueError()    
            
            else:
                # sampling only pop hyperparamters
                pass
            
            print('Init done. ')
         


    ################################################
    # Run sampler
    ################################################

    if pymc.__version__.split('.')[1]>22: # recent versions of pymc
        sampler_kwargs = {
                    "draws": FLAGS.nsteps,
                    "tune":FLAGS.ntune,
                    "target_accept": FLAGS.target_accept,
                    "chains": FLAGS.nchains,
                    "random_seed": 42,
                    "initvals": ivals,
                    "cores": FLAGS.ncores
                }
        
        if FLAGS.sampler=='pymc':
            
            #sampler_kwargs['cores'] = FLAGS.ncores
            sampler_kwargs['trace'] = backend
            sampler_kwargs['progressbar'] = 'split+stats'
            #sampler_kwargs['step'] = pm.NUTS( target_accept=sampler_kwargs.pop("target_accept"))
        
        elif FLAGS.sampler=='numpyro':
            
            sampler_kwargs['progressbar'] = 'split+stats'

        print('Sampling with %s...' %FLAGS.sampler)
        with model:
            trace = pm.sample( nuts_sampler=FLAGS.sampler, **sampler_kwargs )
        
    
    else:  # works with older versions (but also with newer)

        if FLAGS.sampler=='pymc' :
            with model:          
                trace = pm.sample(  draws=FLAGS.nsteps, 
                                    tune=FLAGS.ntune, 
                                    chains=FLAGS.nchains,
                                    cores=FLAGS.ncores, 
                                    initvals=ivals,
                                  #init='jitter+adapt_diag_grad',
                                    step = pm.NUTS( target_accept=FLAGS.target_accept),
                                    trace=backend,
                                    progressbar=True
                                 )
        
            
        elif FLAGS.sampler=='blackjax':
            try:
                import pymc.sampling_jax
                with model:
                    trace = pymc.sampling_jax.sample_blackjax_nuts(draws=FLAGS.nsteps, 
                                               tune=FLAGS.ntune, 
                                               chains=FLAGS.nchains, 
                                               target_accept=FLAGS.target_accept, 
                                               #random_seed=None, 
                                               initvals=ivals, 
                                               #model=None, 
                                               #var_names=None, 
                                               #keep_untransformed=False, 
                                               #chain_method='parallel', 
                                               #postprocessing_backend=None, 
                                               #postprocessing_vectorize='scan', 
                                               #idata_kwargs=None, 
                                               #trace=backend,
                                              )
            except:
                import pymc.sampling.jax as pmjax
                with model:
                    trace = pmjax.sample_blackjax_nuts(draws=FLAGS.nsteps, 
                                               tune=FLAGS.ntune, 
                                               chains=FLAGS.nchains, 
                                               target_accept=FLAGS.target_accept, 
                                               #random_seed=None, 
                                               initvals=ivals, 
                                               #model=None, 
                                               #var_names=None, 
                                               #keep_untransformed=False, 
                                               #chain_method='parallel', 
                                               #postprocessing_backend=None, 
                                               #postprocessing_vectorize='scan', 
                                               #idata_kwargs=None, 
                                               #trace=backend,
                                              )
        
        elif FLAGS.sampler=='numpyro':
            try:
                import pymc.sampling_jax
                with model:
                    trace = pymc.sampling_jax.sample_numpyro_nuts(draws=FLAGS.nsteps, 
                                               tune=FLAGS.ntune, 
                                               chains=FLAGS.nchains, 
                                               target_accept=FLAGS.target_accept, 
                                               #random_seed=None, 
                                               initvals=ivals, 
                                               #model=None, 
                                               #var_names=None, 
                                               #keep_untransformed=False, 
                                               #chain_method='parallel', # 'vectorized'
                                               #postprocessing_backend=None, 
                                               #postprocessing_vectorize='scan', 
                                               #idata_kwargs=None, 
                                                progressbar=True
                                                                 )
            except:
                import pymc.sampling.jax as pmjax
                with model:
                    trace = pmjax.sample_numpyro_nuts(draws=FLAGS.nsteps, 
                                               tune=FLAGS.ntune, 
                                               chains=FLAGS.nchains, 
                                               target_accept=FLAGS.target_accept, 
                                               #random_seed=None, 
                                               initvals=ivals, 
                                               #model=None, 
                                               #var_names=None, 
                                               #keep_untransformed=False, 
                                               #chain_method='parallel', # 'vectorized'
                                               #postprocessing_backend=None, 
                                               #postprocessing_vectorize='scan', 
                                               #idata_kwargs=None, 
                                                progressbar=True
                                                                 )

        else:
            raise ValueError('sampler argument can be pymc, blackjax or numpyro')


        



    ################################################
    # Save and exit
    ################################################
    
    if FLAGS.backend=='disk':
        trace.to_netcdf( os.path.join(FLAGS.fout, "trace.nc"))
    else:
        # Fetch the run from the database (downloads just metadata)
        run = backend.get_run(trace.run_id)
        idata = run.to_inferencedata()

        az.to_netcdf(idata, os.path.join(FLAGS.fout, "trace.nc"))
        

    #########

    print("\nMaking summary plots...")

    try:
        az.plot_trace(trace, var_names = vplot, );
        plt.savefig( os.path.join(FLAGS.fout, 'trace.pdf'), bbox_inches='tight')
        plt.close()
    except:
        print('No trace plot produced')

    try:
        _ = corner.corner(
            trace,
            var_names=vplot,
            labels=vplot,  
            color='darkred',
            plot_points=False,
            levels=[0.68, 0.90],
            show_titles=True, 
            title_kwargs={"fontsize": 15, }, label_kwargs={"fontsize": 15},
            density=True,
            smooth=0.9, 
            fill_contours=True,
             bins=20, 
            title_fmt='.2f', 
            hist_bin_factor=1,
            quantiles=[0.05, 0.5, 0.95],
    )
    
        plt.savefig( os.path.join(FLAGS.fout, 'corner_all.pdf'), bbox_inches='tight')
        plt.close()
    except:
        print('No corner plot produced')

    print("\nDone.")
    #########

    
    print()
    print('*'*80)
    print('END. Results are saved in: %s'%FLAGS.fout)
    print('*'*80)
    print()

    
    myLog.close()



