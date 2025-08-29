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

#os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'
pytensor.config.floatX = "float64"
#pytensor.config.floatX = "float32"


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
parser.add_argument("--has_m2_break", default=0, type=int, required=False)



parser.add_argument("--dLprior", default='none', type=str, required=False)
parser.add_argument("--spinprior", default=0, type=int, required=False)
parser.add_argument("--massprior", default=0, type=int, required=False)
parser.add_argument("--use_sel_spin", default=1, type=int, required=False)


parser.add_argument("--sampling_gw", default='gmm', type=str, required=False)
parser.add_argument("--cho_dil", default=1., type=float, required=False)
parser.add_argument("--sel", default='Tobs', type=str, required=False)
parser.add_argument("--ivals", default='', type=str, required=False)
parser.add_argument("--eps_init", default=0.01, type=float, required=False)
parser.add_argument("--params_fix", default='', type=str, required=False)
parser.add_argument("--check_init", default=1, type=int, required=False)
parser.add_argument("--debug", default=1, type=int, required=False)


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

    if FLAGS.ivals!='':
        with open(FLAGS.ivals) as json_file:
                ivals = json.load(json_file)
        print('Initial values:')
        print(ivals)
    else:
        print('No initial values passed.')
    
    model = models.make_model(  priors,
                                    GWData,
                                    InjData,
                                  ivals=ivals,
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
    print('Initializing inference...')
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
    # Run sampler
    ################################################

    if int(pm.__version__.split('.')[1])>20: # recent versions of pymc
        
        sampler_kwargs = {
                    "draws": FLAGS.nsteps,
                    "tune":FLAGS.ntune,
                    "target_accept": FLAGS.target_accept,
                    "chains": FLAGS.nchains,
                    "random_seed": 42,
                    "initvals": ivals,
                    "cores": FLAGS.ncores,
                    "progressbar": True,
                    "trace": backend
                }

        
        with model:
            if FLAGS.sampler=='pymc' and ivals is not None:
                
                #sampler_kwargs['cores'] = FLAGS.ncores
                #sampler_kwargs['trace'] = backend
                #sampler_kwargs['progressbar'] = 'split+stats'
                #sampler_kwargs['progressbar_theme'] = pm.util.default_progress_theme
                #sampler_kwargs["initvals"] = ivals
                
                sampler_kwargs['step'] = pm.NUTS( target_accept=sampler_kwargs.pop("target_accept"))
                
            
            #else: #if FLAGS.sampler=='numpyro':
                
                #sampler_kwargs['progressbar'] = True #'split+stats'
                #sampler_kwargs['nuts_sampler_kwargs'] = {'initvals' : ivals}
                #sampler_kwargs['jitter'] = True
                #sampler_kwargs["initvals"] = ivals

            if FLAGS.debug:
                print()
                print('*'*80)
                print('Debugging...')
                print('*'*80)
                print()
        
                model.debug()

                print('\nDone. ')

            if FLAGS.check_init:
                print()
                print('*'*80)
                print('Check initial point...')
                print('*'*80)
                print()
                # initial points
                init_point = model.initial_point()
                for key, val in init_point.items():
                        print(f"{key}: {val}")
                print('\nDone. ')
               
            print()
            print('*'*80)
            print('Sampling...')
            print('*'*80)
            print()
            
            print('Sampling with %s...' %FLAGS.sampler)
            trace = pm.sample( nuts_sampler=FLAGS.sampler, **sampler_kwargs )
        
    
    else:  # works with older versions (but also with newer)

        if FLAGS.sampler=='pymc' :

            
            with model:   
                
                ip = model.initial_point()
                print('Initial point names:')
                print(ip.keys())
                print('Check init vals')
                #for s in initvals if isinstance(ivals, list) else [ivals]:
                #    model.check_start_vals(s)
                
                try:
                    model.check_start_vals(ip)
                    print("Start is finite ✅")
                except Exception as e:
                    print("Start invalid ❌:", e)
                    # Inspect per-term logp
                    f = model.compile_logp(sum=False)
                    parts = f(ip)
                    print("Per-term logps:", parts)
                
                trace = pm.sample(  draws=FLAGS.nsteps, 
                                    tune=FLAGS.ntune, 
                                    chains=FLAGS.nchains,
                                    cores=FLAGS.ncores, 
                                    #initvals=ivals,
                                  #init='jitter+adapt_diag_grad',
                                    step = pm.NUTS( target_accept=FLAGS.target_accept),
                                    trace=backend,
                                    progressbar=True,
                                    #jitter_max_retries=1000, 
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



