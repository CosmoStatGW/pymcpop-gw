#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

# --- set env vars BEFORE importing jax (propagates to spawned workers) ---
import os
#os.environ.setdefault("JAX_ENABLE_X64", "True")   # enables float64 in all processes. done later


#os.environ.setdefault("PYTENSOR_FLAGS", "optimizer_excluding=fusion")
#os.environ.setdefault("PYTENSOR_FLAGS", "gcc__cxxflags=-fbracket-depth=2048")

#os.environ["PYTENSOR_FLAGS"] = "optimizer=fast_run,gcc__cxxflags=-fbracket-depth=2048"


import argparse
import json
import sys
import warnings
import psutil

from tqdm import tqdm 
from tqdm.auto import tqdm
import time
import resource

import arviz as az
import matplotlib.pyplot as plt
import corner


_process = psutil.Process(os.getpid())
def mem_gb():
    return _process.memory_info().rss / (1024**3)  # Resident Set Size in GB

def log_mem(tag):
    print(f"[MEM] {tag}: {mem_gb():.2f} GB RSS")


def main():

    
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("--fin_data", nargs='+', type=str, required=True)
    parser.add_argument("--fin_injections", nargs='+', type=str, required=True)
    parser.add_argument("--fin_priors", default='', type=str, required=True)
    parser.add_argument("--backend", default='disk', type=str, required=False)
    
    parser.add_argument("--pop_only", default=0, type=int, required=False)
    
    
    parser.add_argument("--rate_model", default='MD', type=str, required=False)
    parser.add_argument("--mass_model", default='PLPreg', type=str, required=False)
    parser.add_argument("--spin_model", default='none', type=str, required=False)
    parser.add_argument("--N_DP_comp_max", default=50, type=int, required=False)
    parser.add_argument("--marginal_R0", default=1, type=int, required=False)
    parser.add_argument("--smoothing", default='LVK', type=str, required=False)
    parser.add_argument("--has_m2_break", default=0, type=int, required=False)
    
    
    
    parser.add_argument("--dLprior", default='none', type=str, required=False)
    parser.add_argument("--use_sel_spin", default=1, type=int, required=False)
    
    
    parser.add_argument("--sampling_gw", default='gmm_cat', type=str, required=False)
    parser.add_argument("--cho_dil", default=1., type=float, required=False)
    parser.add_argument("--sel", default='Tobs', type=str, required=False)
    parser.add_argument("--ivals", default='', type=str, required=False)
    parser.add_argument("--eps_init", default=0.01, type=float, required=False)
    parser.add_argument("--params_fix", default='', type=str, required=False)
    parser.add_argument("--check_init", default=1, type=int, required=False)
    parser.add_argument("--debug", default=0, type=int, required=False)
    parser.add_argument("--profile", default=0, type=int, required=False)

    parser.add_argument("--save_thetas", default=0, type=int, required=False)
    
    
    
    parser.add_argument("--n_inj_use", nargs='+', type=float, required=False)
    parser.add_argument("--fix_inj_len", default=0, type=int, required=False)
    parser.add_argument("--min_Neff", default=0, type=int, required=False)
    parser.add_argument("--Neff_min_lik", default=0, type=int, required=False)
    parser.add_argument("--log_lik_var_min", default=1, type=float, required=False)
    parser.add_argument("--chunk_inj", default=-1, type=int, required=False)
    parser.add_argument("--chunk_reduce", default=0, type=int, required=False)
    parser.add_argument("--use_float32", default=0, type=int, required=False)
    parser.add_argument("--use_float32_bias", default=0, type=int, required=False)
    parser.add_argument("--inj_loop", default=0, type=int, required=False)

        
    
    parser.add_argument("--nsamplesmax", default=-1, type=int, required=False)
    parser.add_argument("--spin_inj", default='none', type=str, required=False)
    parser.add_argument("--Nsamplesuse", default=-1, type=int, required=False)
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
    parser.add_argument("--chain_method", default='parallel', type=str, required=False)
    
    
    parser.add_argument("--fix_H0", default=1, type=int, required=False)
    parser.add_argument("--fix_Om", default=1, type=int, required=False)
    parser.add_argument("--fix_w0", default=1, type=int, required=False)
    parser.add_argument("--fix_Xi0n", default=1, type=int, required=False)
    parser.add_argument("--pade", default=0, type=int, required=False)
    parser.add_argument("--zres", default='low', type=str, required=False)
    
    parser.add_argument("--allTobs", nargs='+', type=float, required=False)


    FLAGS = parser.parse_args()

    if FLAGS.chain_method == "vectorized" and FLAGS.ncores > 1:
        raise ValueError(
            "For chain_method='vectorized', set ncores=1. "
            "Vectorized mode runs all chains in one JAX process and "
            "does not use multiprocessing."
        )

    if FLAGS.chain_method == "parallel" and FLAGS.ncores < FLAGS.nchains:
        print(
            f"⚠️ Warning: ncores ({FLAGS.ncores}) < nchains ({FLAGS.nchains}). "
            "This may limit parallel performance."
        )

    device_count = FLAGS.ncores if hasattr(FLAGS, "ncores") else 1

    # ----------------------------------------------------
    # 1️⃣ Environment setup BEFORE importing JAX / NumPyro / PyMC
    # ----------------------------------------------------
    if FLAGS.chain_method == "parallel":
        # Must set before importing numpyro/jax/pymc
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={FLAGS.ncores}"


    
    if FLAGS.use_float32:
        try:
            os.environ["PYTENSOR_FLAGS"] = "floatX=float32,optimizer=fast_run,gcc__cxxflags=-fbracket-depth=2048"
        except:
            os.environ["PYTENSOR_FLAGS"] = "floatX=float32,optimizer=fast_run"
        os.environ.setdefault("JAX_ENABLE_X64", "False")
    
    else:
        try:
            os.environ["PYTENSOR_FLAGS"] = "optimizer=fast_run,gcc__cxxflags=-fbracket-depth=2048"
        except:
            os.environ["PYTENSOR_FLAGS"] = "optimizer=fast_run"
        os.environ.setdefault("JAX_ENABLE_X64", "True")
    

    os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")


    # ----------------------------------------------------
    # 2️⃣ Import libraries (now they see the environment)
    # ----------------------------------------------------
    import numpyro
    
    import jax
    import jax.numpy as np
    
    if FLAGS.use_float32:
        jax.config.update("jax_enable_x64", False)
    else:
        jax.config.update("jax_enable_x64", True)
        
    jax.config.update("jax_debug_nans", True)   # crash at the first NaN/Inf during warmup
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")

    from scipy.special import ndtr, ndtri, erfinv
    
    # Ensure correct device setup
    device_count = FLAGS.ncores if FLAGS.chain_method == "parallel" else FLAGS.ncores
    if FLAGS.chain_method == "parallel":
        numpyro.set_host_device_count(device_count)
    

    # ----------------------------------------------------
    # 3️⃣ Now safe to import PyMC and others
    # ----------------------------------------------------
    import pymc as pm
    import pytensor
    #import pytensor.tensor as at
    import arviz as az
    import numpy as onp
    import matplotlib.pyplot as plt
    import corner
    
    # Custom modules
    import pymc_models as models
    import data_tools as dt
    import pytensor_tools as atools
    import pytensor_utils as autils

    if FLAGS.use_float32:
        pytensor.config.floatX = "float32"
    else:
        pytensor.config.floatX = "float64"
    
    X = np.float32 if pytensor.config.floatX == "float32" else np.float64  # model dtype

    # ----------------------------------------------------
    # 4️⃣ Multiprocessing setup (only for parallel chains)
    # ----------------------------------------------------
    if FLAGS.chain_method == "parallel":
        import multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass


    

    logfile = os.path.join(FLAGS.fout, 'logfile.txt')
    myLog = autils.Logger(logfile)
    sys.stdout = myLog
    sys.stderr = myLog


    print("Available devices:", jax.devices())
    print("Local device count:", jax.local_device_count())
    print("Backend:", jax.default_backend())

    print(f"Running on PyMC v{pm.__version__}")
    print("JAX:", jax.__version__, "NumPyro:", numpyro.__version__)
    if FLAGS.use_float32:
        print("dtype test:", np.array(0., dtype=np.float32).dtype)
    else:
        print("dtype test:", np.array(0., dtype=np.float64).dtype)
    

    print(f"[PID] {os.getpid()}")
    

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

        # samples_means_at = at.as_tensor_variable(data['samples_means'])
        # samples_cho_covs_at = at.as_tensor_variable(data['samples_cho_covs']*FLAGS.cho_dil)
    
        # gmm_log_wts = at.as_tensor_variable(data['gmm_log_wts'])
        # gmm_means = at.as_tensor_variable(data['gmm_means'])
        # gmm_icovs = at.as_tensor_variable(data['gmm_icovs'])
        # gmm_cho_covs = at.as_tensor_variable(data['gmm_cho_covs'])
        # gmm_log_dets = at.as_tensor_variable(data['gmm_log_dets'])
        # allNgm = at.as_tensor_variable(data['allNgm'])
        # Nevents = at.as_tensor_variable(data['Nevents'])

        samples_means_at = data['samples_means']
        samples_cho_covs_at = data['samples_cho_covs']*FLAGS.cho_dil
    
        gmm_log_wts = data['gmm_log_wts']
        gmm_means = data['gmm_means']
        gmm_icovs =  data['gmm_icovs']
        gmm_cho_covs =  data['gmm_cho_covs']
        gmm_log_dets =  data['gmm_log_dets']
        allNgm =  data['allNgm']
        Nevents =  data['Nevents']

    

    else:
        print("Using n max samples = %s"%FLAGS.nsamplesmax)
        data = dt.load_data_samples(FLAGS.fin_data, nmax=FLAGS.nsamplesmax)

        # m1d_samples = at.as_tensor_variable(data['m1d_samples'])
        # m2d_samples = at.as_tensor_variable(data['m2d_samples'])
        # dL_samples = at.as_tensor_variable(data['dL_samples'])
        # print("dL_samples shape is %s"%(str(dL_samples.shape)))

        # allNsamples = at.as_tensor_variable(data['allNsamples'])
        # where_compute = at.as_tensor_variable(data['where_compute'])

        m1d_samples = data['m1d_samples']
        m2d_samples =  data['m2d_samples']
        dL_samples =  data['dL_samples']
        print("dL_samples shape is %s"%(str(dL_samples.shape)))

        allNsamples =  data['allNsamples']
        where_compute = data['where_compute']

        if (FLAGS.spin_model=='default') or (FLAGS.spin_model=='default_gauss'):

            # chi1_samples = at.as_tensor_variable(data['chi1_samples'])
            # chi2_samples = at.as_tensor_variable(data['chi2_samples'])
            # cost1_samples = at.as_tensor_variable(data['cost1_samples'])
            # cost2_samples = at.as_tensor_variable(data['cost2_samples'])
            chi1_samples =  data['chi1_samples']
            chi2_samples =  data['chi2_samples']
            cost1_samples =  data['cost1_samples']
            cost2_samples =  data['cost2_samples']

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
        # InjData = [ at.as_tensor_variable(injections['dL']), 
        #         at.as_tensor_variable(injections['m1d']), 
        #         at.as_tensor_variable(injections['m2d']), 
        #         at.as_tensor_variable(injections['log_wt']), 
        #         at.as_tensor_variable(injections['Ngen']), 
        #         at.as_tensor_variable(injections['Ndet']), 
        #           ]
        InjData = [ injections['dL'], 
                injections['m1d'], 
                injections['m2d'], 
                 injections['log_wt'], 
                 injections['Ngen'], 
                 injections['Ndet'], 
                  ]
    else:
        
        if FLAGS.spin_inj=='chieffchip':
            # InjData = [ at.as_tensor_variable(injections['dL']), 
            #     at.as_tensor_variable(injections['m1d']), 
            #     at.as_tensor_variable(injections['m2d']), 
            #     at.as_tensor_variable(injections['chieff']), 
            #     at.as_tensor_variable(injections['chip']), 
            #     at.as_tensor_variable(injections['log_wt']), 
            #     at.as_tensor_variable(injections['Ngen']), 
            #     at.as_tensor_variable(injections['Ndet']), 
            #       ]
            InjData = [ injections['dL'], 
                 injections['m1d'], 
                 injections['m2d'], 
                 injections['chieff'], 
                 injections['chip'], 
                 injections['log_wt'], 
                 injections['Ngen'], 
                injections['Ndet'], 
                  ]
        elif FLAGS.spin_inj=='chi12xyz':

            if (FLAGS.spin_model=='default') or (FLAGS.spin_model=='default_gauss'):

                print("Computing chi1, chi2, cost1, cost2 in injections...")
    
                chi1Inj = onp.sqrt(injections['spin1x']**2+injections['spin1y']**2+injections['spin1z']**2)
                chi2Inj = onp.sqrt(injections['spin2x']**2+injections['spin2y']**2+injections['spin2z']**2)
    
                cost1Inj = injections['spin1z']/chi1Inj
                cost2Inj = injections['spin2z']/chi2Inj
                
                # InjData = [ at.as_tensor_variable(injections['dL']), 
                #     at.as_tensor_variable(injections['m1d']), 
                #     at.as_tensor_variable(injections['m2d']), 
                #     at.as_tensor_variable(chi1Inj), 
                #     at.as_tensor_variable(chi2Inj),
                #     at.as_tensor_variable(cost1Inj),
                #     at.as_tensor_variable(cost2Inj),
                #     at.as_tensor_variable(injections['log_wt']), 
                #     at.as_tensor_variable(injections['Ngen']), 
                #     at.as_tensor_variable(injections['Ndet']), 
                #       ]
                InjData = [ injections['dL'], 
                     injections['m1d'], 
                     injections['m2d'], 
                     chi1Inj, 
                     chi2Inj,
                     cost1Inj,
                     cost2Inj,
                     injections['log_wt'], 
                     injections['Ngen'], 
                     injections['Ndet'], 
                      ]

            elif FLAGS.spin_model=='none':

                print("Injections data has spins but those will not be used !")
    
                # InjData = [ at.as_tensor_variable(injections['dL']), 
                #     at.as_tensor_variable(injections['m1d']), 
                #     at.as_tensor_variable(injections['m2d']), 
                #     at.as_tensor_variable(injections['log_wt']), 
                #     at.as_tensor_variable(injections['Ngen']), 
                #     at.as_tensor_variable(injections['Ndet']), 
                #       ]
                InjData = [ injections['dL'], 
                    injections['m1d'], 
                    injections['m2d'], 
                    injections['log_wt'], 
                    injections['Ngen'], 
                    injections['Ndet'], 
                      ]
                
        elif FLAGS.spin_inj=='default' or FLAGS.spin_inj=='default_gauss':

                # InjData = [ at.as_tensor_variable(injections['dL']), 
                #     at.as_tensor_variable(injections['m1d']), 
                #     at.as_tensor_variable(injections['m2d']), 
                #     at.as_tensor_variable(injections['chi1']), 
                #     at.as_tensor_variable(injections['chi2']),
                #     at.as_tensor_variable(injections['cost1']),
                #     at.as_tensor_variable(injections['cost2']),
                #     at.as_tensor_variable(injections['log_wt']), 
                #     at.as_tensor_variable(injections['Ngen']), 
                #     at.as_tensor_variable(injections['Ndet']), 
                #       ]
                InjData = [ injections['dL'], 
                     injections['m1d'], 
                     injections['m2d'], 
                     injections['chi1'], 
                     injections['chi2'],
                     injections['cost1'],
                     injections['cost2'],
                    injections['log_wt'], 
                     injections['Ngen'], 
                     injections['Ndet'], 
                      ]
        else:
            raise ValueError('Enter valid spin model.')

    
            
    if not FLAGS.pop_only:  
    
        if 'gmm' in FLAGS.sampling_gw or 'gumbel' in FLAGS.sampling_gw:
            GWData =  [
                       onp.exp(gmm_log_wts), 
                       gmm_means, 
                       gmm_cho_covs, 
                       injections['Tobs'],
                        Nevents
                      ]
        elif FLAGS.sampling_gw=='gauss':
            GWData =  [samples_means_at, 
                       samples_cho_covs_at, 
                       gmm_log_wts, 
                       gmm_means, 
                       gmm_icovs, 
                       gmm_log_dets, 
                       injections['Tobs'],
                       Nevents, 
                      ]
            

    else:
        GWData = [ m1d_samples, m2d_samples, dL_samples, spin_samples, #Nevents, 
                     injections['Tobs'], allNsamples, where_compute ]
        
        
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
        ivals={}

    use_updates =  ('pymc' in FLAGS.sampler)
    # right before building the model
    log_mem("before make_model")
    t0 = time.time()
    model = models.make_model(  priors,
                                    GWData,
                                    InjData,
                                    ivals=ivals,
                                    sampling_GW = FLAGS.sampling_gw,
                                    rate_model = FLAGS.rate_model,
                                    mass_model = FLAGS.mass_model,
                                    smoothing=FLAGS.smoothing,
                                    spin_model = FLAGS.spin_model,
                                    spin_inj = FLAGS.spin_inj,
                                    dLprior = FLAGS.dLprior,
                                    sel_method=FLAGS.sel,
                                    fix_inj_len=FLAGS.fix_inj_len,
                                    use_float32 = FLAGS.use_float32,
                                    use_float32_bias = FLAGS.use_float32_bias,
                                    chunk_inj=FLAGS.chunk_inj,
                                    chunk_reduce = FLAGS.chunk_reduce,
                                    marginal_R0 = FLAGS.marginal_R0,
                                    N_DP_comp_max = FLAGS.N_DP_comp_max,
                                    fix_H0 = FLAGS.fix_H0,
                                    fix_Om = FLAGS.fix_Om,
                                    fix_w0 = FLAGS.fix_w0,
                                    fix_Xi0n = FLAGS.fix_Xi0n,
                                    zres = FLAGS.zres,
                                    pade=FLAGS.pade,
                                    Neff_min=FLAGS.min_Neff,
                                    Neff_min_lik = FLAGS.Neff_min_lik,
                                    log_lik_var_min = FLAGS.log_lik_var_min,
                                    use_sel_spin=FLAGS.use_sel_spin,
                                    pop_only = FLAGS.pop_only,
                                    N_successes_l = N_successes_l,
                                    Nsamplesuse = FLAGS.Nsamplesuse,
                                    include_sel_uncertainty = FLAGS.sel_uncertainty,
                                    sel_smoothing = FLAGS.sel_smoothing,
                                    alpha_beta_prior = FLAGS.alpha_beta_prior,
                                    dil_factor=FLAGS.dil_factor,
                                    use_log_alpha_beta=FLAGS.use_log_alpha_beta,
                                    params_fix=params_fix,
                                      allTobs=FLAGS.allTobs,
                                use_updates = use_updates,
                                inj_loop = FLAGS.inj_loop,
                                save_thetas = FLAGS.save_thetas
                                )
    print(f"[TIMER] make_model took {time.time()-t0:.1f}s")
    log_mem("after make_model")
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
                    #"initvals": ivals,
                    "cores": FLAGS.ncores,
                    "progressbar": True,
                    "trace": backend,
                    #"chain_method":'parallel'
                }

        
        with model:

            print("Setting initial point...")
            ip = model.initial_point()
            
            N = gmm_means.shape[0]
            nd = gmm_means.shape[2]

            ip['x'] = onp.random.randn(N, nd) * FLAGS.eps_init


            wts = onp.exp(gmm_log_wts)
            idx = onp.argmax(wts, axis=1)
            
            if FLAGS.sampling_gw=='gmm_cat':
                ip['idx'] = idx.astype(int)

            elif FLAGS.sampling_gw=='gmm':
                cdf = onp.cumsum(wts, axis=1)
                
                # pick v in the open interval [CDF_{i-1}, CDF_i)
                lo = onp.where(idx == 0, 0.0, cdf[onp.arange(len(idx)), idx - 1])
                hi = cdf[onp.arange(len(idx)), idx]
                v  = onp.clip(0.5 * (lo + hi), 1e-9, 1 - 1e-9)
                
                # invert Phi: u = Phi^{-1}(v) = sqrt(2) * erfinv(2v-1)
                u_init = onp.sqrt(2.0) * erfinv(2.0 * v - 1.0)
                ip['u_gmm'] =  u_init

            elif FLAGS.sampling_gw=='gumbel':
                # inputs
                w = wts                     # (N, K)
                tau = 1e-05                                 # same tau you use in the model
                eps = 1e-6                                # desired spillover mass
                N, K = w.shape
                
                # logits and target index per row
                logits = onp.log(onp.clip(w, 1e-12, 1.0))   # (N, K)
                idx = onp.argmax(logits, axis=1)           # (N,)
                
                # required margin Δ so top prob ≥ 1 - eps:
                # Δ >= tau * log((K-1)/eps)
                Delta = tau * (onp.log(max(K-1, 1)) - onp.log(eps))
                
                # build g_init so (logits + g) gives the target argmax with margin Δ
                g_init = onp.zeros_like(logits)
                for n in range(N):
                    k = idx[n]
                    # best competing logit (exclude the winner)
                    max_other = logits[n, np.arange(K) != k].max() if K > 1 else -onp.inf
                    # ensure: logits[n,k] + g_init[n,k] >= max_other + Δ
                    need = (max_other + Delta) - logits[n, k]
                    g_init[n, k] = max(0.0, need)

                ip['gumbel'] = g_init

            print("Done.")
            
            if FLAGS.debug:
                #print()
                #print('*'*40)
                print('Debugging...')
                #print('*'*40)
                #print()
        
                model.debug()

                print('Done. ')

            if FLAGS.check_init:
                #print()
                #print('*'*40)
                print('Checking initial point...')
                #print('*'*40)
                #print()
   
                
                try:
                    
                    
                    model.check_start_vals(ip)
                
                    # f  = model.compile_logp(sum=True)
                    # g  = model.compile_dlogp()                      # gradient
                    # assert np.isfinite(f(ip))
                    # for gi in g(ip): assert np.all(np.isfinite(gi)) # every block finite
                    

                    # log_lik
                    print("Computing initial log-likelihood...")
                    f = model.compile_logp(sum=True, profile=True)
                    assert np.isfinite(f(ip))
                    
                    # check time
                    start_time = time.time()
                    _ = f(ip)
                    elapsed = time.time() - start_time
                    print(f"Log-likelihood evaluation time: {elapsed:.3f} s")
                    
                    # gradient
                    print("Computing initial log-likelihood's gradient...")
                    g  = model.compile_dlogp(profile=True)                      
                    grads = g(ip)
                    
                    # check time
                    start_time = time.time()
                    _ = g(ip)
                    elapsed = time.time() - start_time
                    print(f"Gradient evaluation time: {elapsed:.3f} s")
                    
                    offset = 0
                    for name, val in ip.items():
                        size = np.size(val)
                        grad_block = grads[offset:offset + size]
                        #print(f"{name}: {grad_block}")
                        offset += size
                        
                        if size == 0:
                            continue
                                            
                        # scalar
                        if np.isscalar(val) or np.ndim(val) == 0:
                            if not np.isfinite(grad_block):
                                print(f"Non finite gradient for '{name}': {grad_block}")
                            continue
                        
                        # tensor
                        try:
                            grad_block = grad_block.reshape(val.shape)
                        except Exception as e:
                            print(f"Impossible reshape for'{name}', grad_block.size={grad_block.size}, val.shape={val.shape}")
                            continue
                        
                        mask = ~np.isfinite(grad_block)
                        if np.any(mask):
                            bad_idx = np.argwhere(mask)
                            print(f"Non finite gradient for '{name}' ({len(bad_idx)} elements):")
                            
                            for idx in bad_idx:
                                idx_tuple = tuple(idx)
                                grad_val = grad_block[idx_tuple]
                                try:
                                    init_val = val[idx_tuple]
                                except Exception:
                                    init_val = "N/A" 
                                event_idx = idx[0] if len(idx) > 0 else None
                                print(f"Event {event_idx}: grad={grad_val}, ival={init_val}, idx={idx_tuple}, log-likelihood={f(ip)}")
                            
                    for gi in grads: assert np.all(np.isfinite(gi)) # every block finite
                    
                    print("Start is finite ✅")

                    
                except Exception as e:
                    print("Start invalid ❌:", e)
                    
                    print('Initial values:')
                    print(ip)

                    f  = model.compile_logp(sum=True)
                    g  = model.compile_dlogp()   
                    
                    total_logp = float(f(ip))
                    grad_norms = [float((gi**2).sum()**0.5) for gi in g(ip)]
                    print("logp(ip) =", total_logp)
                    #print("||grad|| per block:", grad_norms)
    
                    blocks = g(ip)  # list of gradient blocks matching PyMC's internal parameter blocks
    
                    def try_step(ip, block_i, eps=1e-4):
                        ip2 = {k: (v.copy() if hasattr(v, "copy") else onp.array(v)) for k, v in ip.items()}
                        # nudge along block_i in gradient direction
                        bkeys = list(ip2.keys())[block_i:block_i+1]
                        # If you know which keys map to block_i use those; this heuristic just tries each key separately:
                        for k in bkeys:
                            v = ip2[k]
                            ip2[k] = v + eps * onp.sign(1.0)  # small +epsilon
                            val = f(ip2)
                            return float(val)
                        return onp.nan
                    
                    bad = []
                    for i in range(len(blocks)):
                        try:
                            val = try_step(ip, i, eps=1e-4)
                            if not onp.isfinite(val):
                                bad.append(i)
                        except Exception:
                            bad.append(i)
    
                    #print("tiny-step bad blocks:", bad)
    
    
                    # Map value var -> RV
                    v2r = {vv: rv for rv, vv in model.rvs_to_values.items()}
                    
                    # Only check free parameters (the ones HMC moves)
                    free_vvs = [model.rvs_to_values[rv] for rv in model.free_RVs]
                    
                    bad = []
                    eps = 1e-6
                    
                    for i, vv in enumerate(free_vvs):
                        key = vv.name  # e.g. "alpha_interval__", "sigma_log__", etc.
                    
                        # make a fresh copy of the transformed start dict
                        test = {k: (onp.array(v, copy=True) if hasattr(v, "shape") else onp.array(v))
                                for k, v in ip.items()}
                        try:
                            step = eps if onp.ndim(test[key]) == 0 else eps * onp.ones_like(test[key])
                            test[key] = test[key] + step
                            val = f(test)
                            if not onp.isfinite(val):
                                rvname = v2r.get(vv, None).name if v2r.get(vv, None) is not None else None
                                bad.append((i, key, rvname))
                        except Exception as e:
                            rvname = v2r.get(vv, None).name if v2r.get(vv, None) is not None else None
                            bad.append((i, key, rvname, str(e)))
                    
                    print("Problematic value_vars on tiny step:")
                    for row in bad[:50]:
                        if len(row) == 3:
                            i, key, rvname = row
                            print(f"{i:4d} {key:>25}   (RV: {rvname})")
                        else:
                            i, key, rvname, msg = row
                            print(f"{i:4d} {key:>25}   (RV: {rvname}) -> {msg}")
                    print(f"... total bad: {len(bad)}")
    
    
    
                    # build a dict of tiny step along gradient for each value var
                    gblocks = g(ip)
                    eps = 1e-4
                    ip_plus = {k: (onp.array(v, copy=True) if hasattr(v, "shape") else onp.array(v)) for k, v in ip.items()}
                    for vv, grad in zip([model.rvs_to_values[rv] for rv in model.free_RVs], gblocks):
                        key = vv.name
                        step = eps * (grad / (onp.linalg.norm(grad.ravel()) + 1e-12))
                        ip_plus[key] = ip_plus[key] + step
                    
                    print("f(ip)      =", float(f(ip)))
                    print("f(ip_plus) =", float(f(ip_plus)))
    
    
                    import random
                    rng = onp.random.default_rng(0)
                    for trial in range(5):
                        ip_rand = {k: (onp.array(v, copy=True) if hasattr(v, "shape") else onp.array(v)) for k, v in ip.items()}
                        for key, val in ip_rand.items():
                            noise = rng.standard_normal(size=onp.shape(val)) * 1e-4
                            ip_rand[key] = val + noise
                        val = f(ip_rand)
                        print(f"random step {trial}: finite? {onp.isfinite(val)}")
    
                    f_parts = model.compile_logp(sum=False)
                    vals = f_parts(ip_rand)  # or ip_plus
                    # 1) Indices of terms that contain ANY non-finite entries
                    bad_idxs = [i for i, v in enumerate(vals) if not onp.isfinite(onp.asarray(v)).all()]
                    print("bad term indices:", bad_idxs)
    
    
                    rng = onp.random.default_rng(0)
                    keys = list(ip.keys())
                    
                    def try_noise(std):
                        trial = {k: onp.array(v, copy=True) for k, v in ip.items()}
                        for k in keys:
                            trial[k] = trial[k] + rng.standard_normal(size=onp.shape(trial[k])) * std
                        val = f(trial)
                        return float(val), onp.isfinite(val)
                    
                    for std in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
                        v, ok = try_noise(std)
                        print(f"std={std:g}  finite? {ok}  f={v}")
    
                    f_parts = model.compile_logp(sum=False)
                    trial_val, _ = try_noise(1e-3)  # use the smallest std that failed above
                    trial = {k: onp.array(v, copy=True) for k, v in ip.items()}
                    for k in keys:
                        trial[k] = trial[k] + rng.standard_normal(size=onp.shape(trial[k])) * 1e-3
                    vals = f_parts(trial)
                    bad = [i for i, v in enumerate(vals) if not onp.isfinite(onp.asarray(v)).all()]
                    print("bad term indices:", bad)
                    
                    raise ValueError()
                
                print('Done. ')

            
            if FLAGS.profile:
                    print('\nProfiling...')
                    pytensor.config.profile = True
                    pytensor.config.profile_memory = False
                    sys.exit(0)
                    
            # ----- sampler-specific kwargs -----
           
            sampler_kwargs = {
                    "draws": FLAGS.nsteps,
                    "tune":FLAGS.ntune,
                    "target_accept": FLAGS.target_accept,
                    "chains": FLAGS.nchains,
                    #"random_seed": 42,
                    "cores": FLAGS.ncores,
                    "progressbar": True,
                    "trace": backend,
                }
    
            if FLAGS.sampler == "numpyro":
                sampler = "numpyro"
                sampler_kwargs.update({
                                # "cores": 1,                         # JAX: single OS process
                                "target_accept": FLAGS.target_accept,  
                                "nuts_sampler_kwargs": {
                                    "chain_method": FLAGS.chain_method,   # fast on single device
                                    "nuts_kwargs": {
                                        # Choose one:
                                        "dense_mass": False,   # set True if dim ≤ ~50 and strong correlations
                                        "adapt_step_size": True,
                                        "adapt_mass_matrix": True,
                                        "regularize_mass_matrix": 1e-3,
                                        "find_heuristic_step_size": True,  # let NumPyro pick a good initial step
                                        "max_tree_depth": 10,
                                        "forward_mode_differentiation": False,
                                    },
                                },
                            })
            elif FLAGS.sampler == "blackjax":
                sampler = "blackjax"
                sampler_kwargs.update({
                    #"cores": 1,                        # avoid fork
                    "target_accept": FLAGS.target_accept,
                    "nuts_sampler_kwargs": {
                        "chain_method": FLAGS.chain_method   # BlackJAX has no 'nuts_kwargs' block
                    },
                })
            else:
                ta = sampler_kwargs.pop("target_accept", FLAGS.target_accept)
                sampler_kwargs["step"] = pm.NUTS(target_accept=ta)

            print("\nModel variables:")
            # Print only the names of variables that are sampled
            print([v.name for v in model.free_RVs])

            print()
            print('*'*80)
            print('Sampling with %s with %s method...' %(FLAGS.sampler, FLAGS.chain_method))
            print('*'*80)
            print()

            if FLAGS.sampler == 'pymc_bar':
                pytensor.config.exception_verbosity = 'high'

          
                
                # progress bar
                #with tqdm(total=(FLAGS.nsteps + FLAGS.ntune)* FLAGS.nchains) as pbar:
                with tqdm(
                            total=(FLAGS.nsteps + FLAGS.ntune)* FLAGS.nchains,
                            desc="Sampling",
                            dynamic_ncols=True,      # auto width like PyMC
                            smoothing=0.3,           # smoother it/s
                            mininterval=0.1,         # refresh rate
                            leave=True,
                            bar_format=(
                                "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
                                "{rate_fmt}{postfix}]"
                            ),
                        ) as pbar:
                    
                    
                    #def callback(trace, draw):
                    #    pbar.update(1)

                    # --- warm-up (forces compile) ---
                    print()
                    # print("Warm up for compilation...")
                    # _ = pm.sample(draws=2, tune=0, chains=1, cores=1, progressbar=False)
                    # print("Done.")
                    t0 = time.time()
                    log_mem("before pm.sample main")
                    cb = autils.make_tqdm_callback(pbar)

                    
                    trace = pm.sample(nuts_sampler='pymc', idata_kwargs={"log_likelihood": False},
                                      **sampler_kwargs,
                                      callback=cb,
                                      
                                     )

                    
                    print(f"[TIMER] pm.sample (main) took {time.time()-t0:.1f}s")
                    log_mem("after pm.sample main")
                    # Print peak resident memory (max RSS) used by this process.
                    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    # macOS returns bytes; Linux returns kilobytes:
                    if sys.platform == "darwin":
                        peak_gb = peak / (1024**3)
                    else:
                        peak_gb = peak / (1024**2)
                    print(f"[MEM] peak RSS: {peak_gb:.2f} GB")

            else:
                t0 = time.time()
                log_mem("before pm.sample main")
                
                
                
                trace = pm.sample(nuts_sampler=FLAGS.sampler, idata_kwargs={"log_likelihood": False}, pytensor_kwargs={"allow_gc": True}, **sampler_kwargs)
                
                
                
                print(f"[TIMER] pm.sample (main) took {time.time()-t0:.1f}s")
                log_mem("after pm.sample main")
                # Print peak resident memory (max RSS) used by this process.
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # macOS returns bytes; Linux returns kilobytes:
                if sys.platform == "darwin":
                    peak_gb = peak / (1024**3)
                else:
                    peak_gb = peak / (1024**2)
                print(f"[MEM] peak RSS: {peak_gb:.2f} GB")

            print('\nDone.')
        
        
    
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
            var_names = vplot,
            labels = vplot,  
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



if __name__=='__main__':
        

    main()
    
    



