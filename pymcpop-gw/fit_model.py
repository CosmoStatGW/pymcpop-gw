#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

# --- set env vars BEFORE importing jax (propagates to spawned workers) ---
import os
os.environ.setdefault("JAX_ENABLE_X64", "True")   # enables float64 in all processes


import argparse
import json
import sys
import warnings


from tqdm import tqdm 
from tqdm.auto import tqdm
import time


import arviz as az
import matplotlib.pyplot as plt
import corner


from pytensor.tensor.sharedvar import SharedVariable, TensorSharedVariable

def set_pytensor_flag(key: str, value: str):
    """Add/override one key in PYTENSOR_FLAGS without nuking the rest."""
    cur = os.environ.get("PYTENSOR_FLAGS", "").strip()
    items = {}
    if cur:
        for part in cur.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                items[k.strip()] = v.strip()
            else:
                # handle bare flags (rare)
                items[part] = "True"
    items[key] = str(value)
    os.environ["PYTENSOR_FLAGS"] = ",".join(f"{k}={v}" for k, v in items.items())
    


def main():

    
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("--fin_data", nargs='+', type=str, required=True)
    parser.add_argument("--fin_injections", nargs='+', type=str, required=True)
    parser.add_argument("--fin_priors", default='', type=str, required=True)
    parser.add_argument("--backend", default='ztrace', type=str, required=False)
    
    parser.add_argument("--pop_only", default=0, type=int, required=False)
    parser.add_argument("--recompile", default=0, type=int, required=False)
    
    
    
    parser.add_argument("--rate_model", default='MD', type=str, required=False)
    parser.add_argument("--mass_model", default='PLPreg', type=str, required=False)
    parser.add_argument("--spin_model", default='none', type=str, required=False)
    parser.add_argument("--reparam_mass", default=1, type=int, required=False)
    parser.add_argument("--reparam_z", default=1, type=int, required=False)
    parser.add_argument("--reparam_cosmo", default=1, type=int, required=False)

    
    
    parser.add_argument("--N_DP_comp_max", default=50, type=int, required=False)
    parser.add_argument("--marginal_R0", default=1, type=int, required=False)
    parser.add_argument("--smoothing", default='LVK', type=str, required=False)
    parser.add_argument("--has_m2_break", default=0, type=int, required=False)
    
    parser.add_argument("--nev_min", default=0, type=int, required=False)
    parser.add_argument("--nev_max", default=-1, type=int, required=False)
    
    parser.add_argument("--dLprior", default='none', type=str, required=False)
    parser.add_argument("--use_sel_spin", default=1, type=int, required=False)


    parser.add_argument("--sampling_gw", default='gmm', type=str, required=False)
    parser.add_argument("--cho_dil", default=1., type=float, required=False)
    parser.add_argument("--sel", default='Tobs', type=str, required=False)
    parser.add_argument("--ivals", default='', type=str, required=False)
    parser.add_argument("--eps_init", default=0.1, type=float, required=False)
    parser.add_argument("--params_fix", default='', type=str, required=False)
    parser.add_argument("--check_init", default=0, type=int, required=False)
    parser.add_argument("--debug", default=0, type=int, required=False)
    parser.add_argument("--MAP_init", default=0, type=int, required=False)
    
    parser.add_argument("--n_inj_use", nargs='+', type=float, required=False)
    parser.add_argument("--fix_inj_len", default=0, type=int, required=False)
    parser.add_argument("--min_Neff", default=0, type=int, required=False)
    parser.add_argument("--Neff_min_lik", default=0, type=int, required=False)
    parser.add_argument("--log_lik_var_min", default=1, type=float, required=False)
    
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
    parser.add_argument("--dense_mass", default=0, type=int, required=False)

     
    parser.add_argument("--is_GP_dL", default=1, type=int, required=False)
    parser.add_argument("--find_GP_L", default=1, type=int, required=False)
    parser.add_argument("--monotonicity", default='poly', type=str, required=False)
    parser.add_argument("--eps_DE", default=-1, type=float, required=False)
    parser.add_argument("--ell_min", default=0.1, type=float, required=False)
    parser.add_argument("--ell_max", default=3, type=float, required=False)
    parser.add_argument("--fine_res", default=0.05, type=float, required=False)
    parser.add_argument("--res_highz", default=0.1, type=float, required=False)
    parser.add_argument("--res_lowz", default=0.05, type=float, required=False)

    parser.add_argument("--init_GP", default='zeros', type=str, required=False)

    

    
    
    parser.add_argument("--monotonicity_scale", default=0., type=float, required=False)
    parser.add_argument("--zmin_mono", default=0., type=float, required=False)
    parser.add_argument("--find_z_bounds", default=0, type=int, required=False)
    parser.add_argument("--zres", default=150, type=int, required=False)
    parser.add_argument("--zmin_a", default=1e-05, type=float, required=False)
    parser.add_argument("--zmin_b", default=1e-03, type=float, required=False)
    parser.add_argument("--zmid_b", default=3., type=float, required=False)
    parser.add_argument("--zmax_c", default=100., type=float, required=False)
    parser.add_argument("--hi_boost", default=.2, type=float, required=False)
    
    parser.add_argument("--nu", default=0.5, type=float, required=False)
    parser.add_argument("--lam", default=1, type=float, required=False)
    parser.add_argument("--clip_high", default=500, type=float, required=False)
    parser.add_argument("--clip_low", default=-500, type=float, required=False)
    parser.add_argument("--GP_prior", default='frechet', type=str, required=False)
    parser.add_argument("--large_ell_penalty", default=1, type=int, required=False)
    
    parser.add_argument("--GP_zero_point", default='y', type=str, required=False)
    parser.add_argument("--invert_dL_GP", default=1, type=int, required=False)
    parser.add_argument("--dense_grad", default=0, type=int, required=False)
    parser.add_argument("--U", default=0, type=float, required=False)
    
    parser.add_argument("--fix_H0", default=1, type=int, required=False)
    parser.add_argument("--fix_Om", default=1, type=int, required=False)
    parser.add_argument("--fix_w0", default=1, type=int, required=False)
    parser.add_argument("--fix_Xi0n", default=1, type=int, required=False)
    parser.add_argument("--pade", default=0, type=int, required=False)

    parser.add_argument("--fix_mass", default=0, type=int, required=False)
    
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
    
    # Optional but recommended: enable float64 early
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")


    # ----------------------------------------------------
    # 2️⃣ Import libraries (now they see the environment)
    # ----------------------------------------------------
    import numpyro
    
    import jax
    import jax.numpy as np
    import numpy as onp
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)   # crash at the first NaN/Inf during warmup


    # Ensure correct device setup
    device_count = FLAGS.ncores if FLAGS.chain_method == "parallel" else FLAGS.ncores
    if FLAGS.chain_method == "parallel":
        numpyro.set_host_device_count(device_count)
    
    print("Available devices:", jax.devices())
    print("Local device count:", jax.local_device_count())
    print("Backend:", jax.default_backend())


    if FLAGS.recompile:
        import tempfile
    
        # Unique-ish per-process / per-run compiledir
        scratch = os.path.join(
            tempfile.gettempdir(), f"pytensor_{os.getuid()}_{os.getpid()}"
        )
        
        flags = [
            f"compiledir={scratch}",
            #"optimizer=fast_run",
            "compile__timeout=600",  # wait up to 10 min
            "compile__wait=10",      # retry every ~10s
            #"jax__enable_x64=True", 
        ]
        

        for f in flags:
            if "=" in f:
                k, v = f.split("=", 1)
                set_pytensor_flag(k.strip(), v.strip())
            else:
                set_pytensor_flag(f.strip(), "True")
        

        print("\nPYTENSOR_FLAGS for recompile =", os.environ.get("PYTENSOR_FLAGS"))

    
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
    from scipy.special import erfinv
    
    # Custom modules
    import pymc_models as models
    import data_tools as dt
    import pytensor_tools as atools
    import pytensor_utils as autils
    
    pytensor.config.floatX = "float64"
    
    print(f"Running on PyMC v{pm.__version__}")
    print("JAX:", jax.__version__, "NumPyro:", numpyro.__version__)
    print("dtype test:", np.array(0., dtype=np.float64).dtype)
    

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

    with open(FLAGS.fin_priors) as json_file:
        priors = json.load(json_file)

        if FLAGS.is_GP_dL:
            try:
                Uprior = priors['U']
                if FLAGS.U != 0:
                    print("Prior for U is present in priors, but input U=%s was given. This will override the prior"%FLAGS.U)
                    Uprior = FLAGS.U
                else:
                    print("Prior for from priors. U=%s"%Uprior)
            except:
                if FLAGS.U == 0:
                    raise ValueError("Insert U in prior for Gaussian Process or pass U as input arg")
                else:
                    print("Prior for U from input U=%s"%FLAGS.U)
                    Uprior = FLAGS.U

                
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


    if not FLAGS.invert_dL_GP:
        raise ValueError('Gaussian Process for distance ratio requires inversion at least for selection effects! This option at the moment is impossible.')

    
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

        data = dt.load_data_interp(FLAGS.fin_data,)

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

    

        gmm_means_sub = data['gmm_means_sub']
        gmm_icovs_sub = data['gmm_icovs_sub']
        gmm_log_dets_sub = data['gmm_log_dets_sub']


        if FLAGS.nev_min != 0 or FLAGS.nev_max != -1:

            N_or = Nevents

            if FLAGS.nev_max == -1 :
                print("Starting from event %s"%FLAGS.nev_min)
                mask_0D = (slice(FLAGS.nev_min, None ))
                mask_1D = (slice(FLAGS.nev_min, None ), slice(None))
                mask_2D = (slice(FLAGS.nev_min, None), slice(None), slice(None))
                mask_3D = (slice(FLAGS.nev_min , None), slice(None), slice(None), slice(None))
                Nev_exp = Nevents - FLAGS.nev_min
            elif FLAGS.nev_min == 0 :
                print("Ending at event %s"%FLAGS.nev_max)
                mask_0D = (slice(None, FLAGS.nev_max ))
                mask_1D = (slice(None, FLAGS.nev_max ), slice(None))
                mask_2D = (slice(None, FLAGS.nev_max), slice(None), slice(None))
                mask_3D = (slice(None, FLAGS.nev_max), slice(None), slice(None), slice(None))
                Nev_exp = FLAGS.nev_max
            else:
                print("Using events between %s and %s"%(FLAGS.nev_min,FLAGS.nev_max))
                Nev_exp = FLAGS.nev_max - FLAGS.nev_min
                mask_0D = (slice(FLAGS.nev_min, FLAGS.nev_max ))
                mask_1D = (slice(FLAGS.nev_min, FLAGS.nev_max ), slice(None))
                mask_2D = (slice(FLAGS.nev_min, FLAGS.nev_max), slice(None), slice(None))
                mask_3D = (slice(FLAGS.nev_min, FLAGS.nev_max), slice(None), slice(None), slice(None))

            samples_means_at = samples_means_at[mask_1D]
            samples_cho_covs_at = samples_cho_covs_at[mask_2D]
        
            gmm_log_wts = gmm_log_wts[mask_1D]
            gmm_means = gmm_means[mask_2D]
            gmm_icovs =  gmm_icovs[mask_3D]
            gmm_cho_covs =  gmm_cho_covs[mask_3D]
            gmm_log_dets =  gmm_log_dets[mask_1D]
            allNgm =  allNgm[mask_0D]
            Nevents =  len(allNgm)


            gmm_means_sub = gmm_means_sub[mask_2D]
            gmm_icovs_sub = gmm_icovs_sub[mask_3D]
            gmm_log_dets_sub = gmm_log_dets_sub[mask_1D]

            assert Nevents == Nev_exp

            print("Number of events used: %s. Original events were %s."%(Nevents, N_or))

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
        dL_prior =  data['dL_PE_prior']
        print("dL_samples shape is %s"%(str(dL_samples.shape)))

        allNsamples =  data['allNsamples']
        where_compute = data['where_compute']

        allnames =  data['allnames']

        Nevents =  m1d_samples.shape[0]

        if (FLAGS.spin_model=='default') or (FLAGS.spin_model=='default_gauss'):

            chi1_samples =  data['chi1_samples']
            chi2_samples =  data['chi2_samples']
            cost1_samples =  data['cost1_samples']
            cost2_samples =  data['cost2_samples']

            spin_samples = onp.asarray([ chi1_samples, chi2_samples, cost1_samples, cost2_samples ])

        elif FLAGS.spin_model=='none':
            spin_samples = onp.asarray([  ])
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
                
        elif FLAGS.spin_inj=='default':

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

    
            
    if not FLAGS.pop_only:  
    
        if 'gmm' in FLAGS.sampling_gw:
            
            #GWData =  [
            #           at.exp(gmm_log_wts), 
            #           gmm_means, 
            #           gmm_cho_covs, 
            #           at.as_tensor_variable(injections['Tobs']),
            #            Nevents
            #          ]

            GWData =  [ onp.exp(gmm_log_wts), 
    					   gmm_means, 
    					   gmm_cho_covs,
                           gmm_icovs,
                           gmm_log_dets,
                           gmm_means_sub, 
                           gmm_icovs_sub,
                           gmm_log_dets_sub,
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
                       gmm_cho_covs,
                       injections['Tobs'],
                       Nevents, 
                      ]
            

    else:
        GWData = [ m1d_samples, m2d_samples, dL_samples, spin_samples, #Nevents, 
                       dL_prior, 
                     injections['Tobs'], allNsamples, where_compute, Nevents, allnames ]
        
        
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
        N_successes_l = np.ones(N)
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
    
    model, z_grid = models.make_model(  priors,
                                    GWData,
                                    InjData,
                                    ivals=ivals,
                                    sampling_GW = FLAGS.sampling_gw,
                                    rate_model = FLAGS.rate_model,
                                    mass_model = FLAGS.mass_model,
                                    reparam_mass = FLAGS.reparam_mass,
                                    reparam_z = FLAGS.reparam_z,
                                    reparam_cosmo = FLAGS.reparam_cosmo,
                                    smoothing=FLAGS.smoothing,
                                    spin_model = FLAGS.spin_model,
                                    spin_inj = FLAGS.spin_inj,
                                    dLprior = FLAGS.dLprior,
                                    sel_method=FLAGS.sel,
                                    fix_inj_len=FLAGS.fix_inj_len,
                                    marginal_R0 = FLAGS.marginal_R0,
                                    N_DP_comp_max = FLAGS.N_DP_comp_max,
                                    is_GP_dL = FLAGS.is_GP_dL,
                                    find_GP_L = FLAGS.find_GP_L,
                                    monotonicity=FLAGS.monotonicity,
                                    eps_DE = FLAGS.eps_DE,
                                    monotonicity_scale=FLAGS.monotonicity_scale,
                                    zmin_mono=FLAGS.zmin_mono,
                                find_z_bounds = FLAGS.find_z_bounds,
                                zres = FLAGS.zres,
                                    zmin_a=FLAGS.zmin_a, 
                                    zmin_b=FLAGS.zmin_b, 
                                    zmid_b=FLAGS.zmid_b, 
                                    zmax_c=FLAGS.zmax_c, 
                                    hi_boost=FLAGS.hi_boost,
                                    nu = FLAGS.nu,
                                     lam = FLAGS.lam,
                                     clip_low = FLAGS.clip_low,
                                     clip_high=FLAGS.clip_high,
                                    GP_prior=FLAGS.GP_prior,
                                    large_ell_penalty=FLAGS.large_ell_penalty,
                                    GP_zero_point=FLAGS.GP_zero_point,
                                    invert_dL_GP=FLAGS.invert_dL_GP,
                                    dense_grad = FLAGS.dense_grad,
                                    fout=FLAGS.fout,
                                    fix_H0 = FLAGS.fix_H0,
                                    fix_Om = FLAGS.fix_Om,
                                    fix_w0 = FLAGS.fix_w0,
                                    fix_Xi0n = FLAGS.fix_Xi0n,
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
                                    U = Uprior,
                                ell_min=FLAGS.ell_min,
                                        ell_max=FLAGS.ell_max,
                                        res_lowz=FLAGS.res_lowz,
                                        res_highz=FLAGS.res_highz,
                                    fine_res=FLAGS.fine_res,
                                        fix_mass=FLAGS.fix_mass,
                                )

    print('Done.')

    print()
    print('*'*80)
    print('Initializing inference...')
    print('*'*80)
    print()
    
    if FLAGS.backend=='disk':
        backend=None
    elif FLAGS.backend=='ztrace':
        import zarr, numcodecs
        
        from pymc.backends.zarr import ZarrTrace
        
        spath=os.path.join(FLAGS.fout, "trace_backup.zarr")
        backend = ZarrTrace(store=spath,  draws_per_chunk=100)
        print("Intermediate trace will be stored at %s"%spath)
        print("zarr:", zarr.__version__, "| numcodecs:", numcodecs.__version__)
    else:
        raise ValueError("backend can be disk or ztrace, got %s"%FLAGS.backend)

         


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

            #ip = model.initial_point()

            vnames = [v.name for v in model.free_RVs]

            
            if FLAGS.ivals == "":
                if FLAGS.MAP_init:
                    print("No ivals provided; using MAP estimate as init..")
                    MAP = pm.find_MAP()   # CPU backend, no JAX involved
                    print("Initial point with MAP:")
                    ip_tmp_ = model.initial_point()
                    ip = { k:MAP[k] for k in ip_tmp_.keys()}
                    ip_vals = {k:MAP[k] for k in vnames if k not in ('beta', 'mulMc', 'mulq', 'eps1', 'eps2', 'x', 'idx')}
                    print(ip_vals)
                    MAP_init=True
                else:
                    print("No ivals provided; MAP init is False. Using default init.")
                    MAP_init=False
                    ip = model.initial_point()
            else:                    
                #print("Using model init point as init")
                MAP_init=False
                ip = model.initial_point()
                

                if not FLAGS.pop_only:
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
        
                        # print("init gaussian pdf so that idx is argmax(w)")
                        # print("True argmax:")
                        # print(idx)
                        # print("init argmax:")
                        # from scipy.special import ndtr  # standard normal CDF Φ
                        # v = np.clip(ndtr(u_init), 1e-9, 1 - 1e-9)
        
                        # # same logic as (v[:, None] < cdf).argmax(axis=1)
                        # idx_from_u = (v[:, None] < cdf).argmax(axis=1)
                        # print(idx_from_u)
    
                if FLAGS.is_GP_dL:
    
                    if ivals != {}:
                        if 'f_rotated_' not in ivals.keys():
        
                            import scipy.linalg as la
                            if True:
                                print("Initializing f_rotated_ to ....")
                                print("Initialization will be to %s"%FLAGS.init_GP)
                    
                               
                                #z_grid = atools.zGridGlobals   # (N,)
                                z_grid = onp.asarray(z_grid.eval())
                                N = z_grid.size
                                z_mid = 0.5 * (z_grid[1:] + z_grid[:-1])                   # (N-1,)
                                
                                Xn = z_grid[:, None]
                                Xm = z_mid[:,  None]
                                
                                # 2) Hyperparams (same you used for the node init)
                                try:
                                    ell0 = ivals['ℓ']
                                    eta0 = ivals['η']
                                    print("Found ℓ, η in prior")
                                except Exception:
                                    eta0 = 0.2
                                    ell0 = 0.3
                                    print("ℓ, η not found, using η=0.2, ℓ=0.3")
                                
                                jitter = 1e-4
                                
                                # 3) Kernels
                                Knn = atools.matern52_1d(Xn, Xn, eta0, ell0) + jitter * np.eye(N)
                                Kmm = atools.matern52_1d(Xm, Xm, eta0, ell0) + jitter * np.eye(N - 1)
                                Kmn = atools.matern52_1d(Xm, Xn, eta0, ell0)              # (N-1, N)
                                
                                # 4) The node function you used to build f_rotated_ (same as your snippet)
                                z0 = z_grid[0]
                                s0 = 0.0001

                                if FLAGS.init_GP=='zeros':
                                    f_nodes_init = np.zeros(z_grid.shape) #0.5 * s0 * (z_grid - z0)                   # (N,)
                                elif FLAGS.init_GP=='polexp':
                                    
                                    def delta(z, n, Xi0):
                                        """
                                        δ(z) = n(1-Ξ0)/(1-Ξ0 + Ξ0(1+z)^n)
                                               + n(1-Ξ0)/(1+z)^n
                                               - 2n(1-Ξ0)/(1+z)^{2n}
                                        """
                                        z = np.asarray(z)
                                        term1 = n * (1 - Xi0) / (1 - Xi0 + Xi0 * (1 + z)**n)
                                        term2 = n * (1 - Xi0) / (1 + z)**n
                                        term3 = 2 * n * (1 - Xi0) / (1 + z)**(2 * n)
                                        return term1 + term2 - term3

                                    n =  3.
                                    Xi0 = 2.5
                                    
                                    f_nodes_init  = -delta(z_grid, n, Xi0)/(1+z_grid)
                                                                        
            
                                
                                # 5) GP interpolation to midpoints: f_mid = K_mn K_nn^{-1} f_nodes
                                #    Use Cholesky solves for stability; reuse Knn factorization
                                c_nn = la.cho_factor(Knn, lower=True, check_finite=False)
                                f_mid_init = Kmn @ la.cho_solve(c_nn, f_nodes_init)       # (N-1,)
                                
                                # 6) Whiten both with their own Cholesky factors:
                                Ln = la.cholesky(Knn, lower=True, check_finite=False)
                                Lm = la.cholesky(Kmm, lower=True, check_finite=False)
                                
                                u_nodes_init = la.solve_triangular(Ln, f_nodes_init, lower=True, check_finite=False)  # (N,)
                                u_mid_init   = la.solve_triangular(Lm, f_mid_init,   lower=True, check_finite=False)  # (N-1,)
                                
            
                                ip["f_rotated_"]     = u_nodes_init + 1e-4 * onp.random.randn(u_nodes_init.size)
                                #ip["f_mid_rotated_"] = u_mid_init   + 1e-3 * np.random.randn(u_mid_init.size)
                            
                            else:
                                def inv_softplus_stable(y):
                                            # stable inverse softplus
                                            return np.where(
                                                y > 20,
                                                y + np.log1p(-np.exp(-y)),
                                                np.log(np.expm1(y))
                                            )
            
                                # 1) grids
                                z_grid = atools.zGridGlobals
                                N = z_grid.size
                                
                                # 2) hyperparameters for init
                                ell0 = ivals.get("ℓ", 0.3)
                                eta0 = ivals.get("η", 0.2)
                                
                                jitter = 1e-4
                                
                                # 3) kernels
                                Xn = z_grid[:, None]
                                Knn = atools.matern52_1d(Xn, Xn, eta0, ell0) + jitter * np.eye(N)
                                
                                # 4) compute b_em
                                dc = atools.dcfun_at(z_grid, 67.7, 0.31, -1).eval()
                                b_em = atools.d_log_dLEM_dz(z_grid, 67.7, 0.31, -1, dc=dc).eval()
                                
                                # 5) target q = b_em - eps
                                eps = 1e-3
                                q_target = np.clip(b_em - eps, 1e-6, None)
                                
                                # 6) invert softplus
                                f_nodes_init = inv_softplus_stable(q_target)
                                
                                # 7) small noise
                                f_nodes_init += onp.random.normal(scale=0.03, size=f_nodes_init.shape)
                                
                                # 8) whiten
                                Ln = la.cholesky(Knn, lower=True)
                                u_nodes_init = la.solve_triangular(Ln, f_nodes_init, lower=True)
                                
                                # 9) normalize to match N(0,1)
                                u_nodes_init = (u_nodes_init - u_nodes_init.mean()) / u_nodes_init.std()
                                
                                # 10) add tiny noise
                                u_nodes_init += 1e-3 * onp.random.randn(N)
                                
                                ip["f_rotated_"] = u_nodes_init
            
                        else:
                            print("Initializing f_rotated_ from file....")
                            ip["f_rotated_"] = ivals["f_rotated_"]
                            #ip["f_mid_rotated_"] = ivals["f_mid_rotated_"]

            
            if FLAGS.debug:
                #print()
                #print('*'*40)
                print('Debugging...')
                #print('*'*40)
                #print()
        
                model.debug()

                print('Done. ')

            if FLAGS.check_init:

                print('Checking initial point...')
                #print('*'*40)
                #print()

                ip = model.initial_point()
     
                try:
                    model.check_start_vals(ip)
                    f  = model.compile_logp(sum=True)
                    g  = model.compile_dlogp()                      # gradient
                    assert np.isfinite(f(ip))
                    for gi in g(ip): assert np.all(np.isfinite(gi)) # every block finite
                    print("Start is finite ✅")
                except Exception as e:
                    print("Start invalid ❌:", e)
                    
                    print('Initial values:')
                    print(ip)
                    
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

    

            # ----- sampler-specific kwargs -----
           
            sampler_kwargs = {
                    "draws": FLAGS.nsteps,
                    "tune": FLAGS.ntune,
                    "target_accept": FLAGS.target_accept,
                    "chains": FLAGS.nchains,
                    #"random_seed": 42,
                    "cores": FLAGS.ncores,
                    "progressbar": True,
                    "trace": backend,
                    "initvals":ip
                }
    
            if FLAGS.sampler == "numpyro":
                sampler = "numpyro"
                sampler_kwargs.update({
                                "cores": FLAGS.ncores,                         # JAX: single OS process
                               "target_accept": FLAGS.target_accept,  
                                "nuts_sampler_kwargs": {
                                    "jitter": False, 
                                    "chain_method": FLAGS.chain_method,   # fast on single device
                                    "nuts_kwargs": {
                                        # Choose one:
                                        "dense_mass":  FLAGS.dense_mass, 
                                        "adapt_step_size": True,
                                        "adapt_mass_matrix": True,
                                        "regularize_mass_matrix": 1e-3 , #5e-4,
                                        "find_heuristic_step_size": False,  # let NumPyro pick a good initial step
                                        "max_tree_depth": 10,
                                        "forward_mode_differentiation": False,
                                        "step_size":1e-2,
                                    },
                                },
                            })
            elif FLAGS.sampler == "blackjax":
                sampler = "blackjax"
                sampler_kwargs.update({
                    "cores": FLAGS.ncores,                        # avoid fork
                    "target_accept": FLAGS.target_accept,
                    "nuts_sampler_kwargs": {
                        "chain_method": FLAGS.chain_method #"vectorized",  # BlackJAX has no 
                    },
                })
            else:

                if FLAGS.dense_mass:
                    sampler_kwargs["init"] = "adapt_full"
                else:
                    ta = sampler_kwargs.pop("target_accept", FLAGS.target_accept)
                    sampler_kwargs["step"] = pm.NUTS(target_accept=ta, max_treedepth=FLAGS.max_tree_depth)



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
                                
                    cb = autils.make_tqdm_callback(pbar)
                    trace = pm.sample(nuts_sampler='pymc', **sampler_kwargs,
                                      callback=cb,
                                      
                                     )

            else:
                
                trace = pm.sample(nuts_sampler=FLAGS.sampler, **sampler_kwargs)

            print('\nDone.')
        
        
    
    else:  # works with older versions (but also with newer)
        
        if FLAGS.check_init:
                print()
                print('*'*80)
                print('Check initial point...')
                print('*'*80)
                print()
                # initial points
                
                
                # Compile logp function
                logp_func = model.logp

                init_point = model.initial_point()
                print("Initial point:")
                for key, val in init_point.items():
                        print(f"{key}: {val}")
                # Evaluate logp at initial point
                initial_logp = logp_func(**init_point)

                
                print("Initial logp:", initial_logp) 
        
                print('\nDone. ')

        if FLAGS.debug:
                print()
                print('*'*80)
                print('Debugging...')
                print('*'*80)
                print()

    

                print("Potentials:", model.potentials)
                print("Potentials:")
                for p in model.potentials:
                    print("  ", repr(p), "-> type:", type(p))
                        
                model.debug()

                print('\nDone. ')
        
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
        # Fetch the run 
        try:
            idata = autils.load_pymc_zarr_trace_robust(spath)  # your working loader
            print( "idata loaded." )
            #idata_clean = autils.drop_object_vars(idata)
            #print( "idata cleaned." )
            az.to_netcdf(idata, os.path.join(FLAGS.fout, "trace.nc"))
            print( "trace saved." )

        except Exception as e:
            print(e)
            print( "No final trace saved." )
        

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
    
    



