#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

# --- set env vars BEFORE importing jax (propagates to spawned workers) ---
import os, argparse, sys


def early_parse(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--nth", type=int, default=None)
    # ignore everything else
    args, _ = p.parse_known_args(argv)
    return args


early = early_parse(sys.argv[1:])


NTH = early.nth if early.nth is not None else 1
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "64"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

#os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")
#os.environ.setdefault("JAX_LOG_COMPILES", "1")


os.environ["OMP_NUM_THREADS"]      = str(NTH)
os.environ["OPENBLAS_NUM_THREADS"] = str(NTH)
os.environ["MKL_NUM_THREADS"]      = str(NTH)
os.environ["NUMEXPR_NUM_THREADS"]  = str(NTH)
os.environ["BLIS_NUM_THREADS"]     = str(NTH)
os.environ["OMP_DYNAMIC"]          = "FALSE"
os.environ["OMP_PROC_BIND"]        = "FALSE"
os.environ["KMP_AFFINITY"]         = "disabled"


print()

import json
import warnings
import time
import resource


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
    parser.add_argument("--priors_for_mmin", default='', type=str, required=False)
    parser.add_argument("--events_use", nargs='+', default=[], type=str, required=False)
    parser.add_argument("--backend", default='ztrace', type=str, required=False)
    parser.add_argument("--draws_per_chunk", default=100, type=int, required=False)
    

    parser.add_argument("--nev_min", default=0, type=int, required=False)
    parser.add_argument("--nev_max", default=-1, type=int, required=False)
    
    parser.add_argument("--pop_only", default=0, type=int, required=False)
    
    parser.add_argument("--rate_model", default='MD', type=str, required=False)
    parser.add_argument("--mass_model", default='DPLDP', type=str, required=False)
    parser.add_argument("--spin_model", default='none', type=str, required=False)
    parser.add_argument("--interp_mass", default=0, type=int, required=False)
    parser.add_argument("--interp_z", default=0, type=int, required=False)
    parser.add_argument("--linear_mass", default=0, type=int, required=False)
    #
    parser.add_argument("--linear_z", default=0, type=int, required=False)

    
    parser.add_argument("--N_DP_comp_max", default=30, type=int, required=False)
    parser.add_argument("--alpha_tail", default=-1, type=float, required=False)
    parser.add_argument("--alpha_small", default=0.01, type=float, required=False)
    parser.add_argument("--L_small_1", default=1., type=float, required=False)
    parser.add_argument("--L_small_2", default=1., type=float, required=False)
    parser.add_argument("--L_small_3", default=0.5, type=float, required=False)
    parser.add_argument("--s_local", default=0.5, type=float, required=False)
    parser.add_argument("--find_m_bounds", default=0, type=float, required=False)
    parser.add_argument("--q_mbound", default=0.05, type=float, required=False)
    parser.add_argument("--alpha_inv_params", nargs='+', type=float, default=[1., 1.], required=False)
    parser.add_argument("--DP_prior", default='SB', type=str, required=False) # SB, dirichelet, softmax
    parser.add_argument("--sigma_softmax", default=0.75, type=float, required=False)
    parser.add_argument("--gamma_DP_params", nargs='+', type=float, default=[1., 1.], required=False)
    parser.add_argument("--DP_truncate_up", default=0, type=int, required=False)
    parser.add_argument("--DP_truncate_low", default=0, type=int, required=False)
    parser.add_argument("--DP_m1_env", default=0, type=int, required=False)
    parser.add_argument("--M_active", default=5, type=int, required=False)
    parser.add_argument("--tau_prior", default='flat', type=str, required=False)
    
    
    
    

    
    
    
    parser.add_argument("--marginal_R0", default=1, type=int, required=False)
    parser.add_argument("--smoothing", default='LVK', type=str, required=False)
    parser.add_argument("--simplex_repair", default=0, type=int, required=False)

    parser.add_argument("--has_m2_break", default=0, type=int, required=False)
    parser.add_argument("--norm_gauss", default='uplow', type=str, required=False)
    
    
    
    parser.add_argument("--dLprior", nargs='+', default=['none'], type=str, required=False)
    #parser.add_argument("--normalize_PE_prior",  default=1, type=int, required=False)
    parser.add_argument("--penorm_lims",  nargs='+', default=[], type=str, required=False)
    parser.add_argument("--use_sel_spin", default=0, type=int, required=False)
    parser.add_argument("--remove_spin_prior", default=0, type=int, required=False)
    
    
    
    parser.add_argument("--sampling_gw", default='gauss', type=str, required=False)
    parser.add_argument("--cho_dil", default=1., type=float, required=False)
    parser.add_argument("--sel", default='Tobs', type=str, required=False)
    parser.add_argument("--ivals", default='', type=str, required=False)
    parser.add_argument("--MAP_init", default=0, type=int, required=False)
    parser.add_argument("--eps_init", default=0.01, type=float, required=False)
    parser.add_argument("--params_fix", default='', type=str, required=False)
    parser.add_argument("--check_init", default=0, type=int, required=False)
    parser.add_argument("--debug", default=0, type=int, required=False)
    parser.add_argument("--debug_sel_batch", default=0, type=int, required=False)
    parser.add_argument("--profile", default=0, type=int, required=False)
    parser.add_argument("--recompile", default=0, type=int, required=False)

    parser.add_argument("--save_thetas", default=1, type=int, required=False)
    
    
    
    parser.add_argument("--n_inj_use", nargs='+', type=float, required=False)
    parser.add_argument("--fix_inj_len", default=0, type=int, required=False)
    parser.add_argument("--min_Neff", default=0, type=int, required=False)
    parser.add_argument("--Neff_min_lik", default=0, type=int, required=False)
    parser.add_argument("--log_lik_var_min", default=1, type=float, required=False)
    parser.add_argument("--chunk_inj", default=0, type=int, required=False)
    parser.add_argument("--chunk_reduce", default=0, type=int, required=False)
    parser.add_argument("--use_float32", default=0, type=int, required=False)
    parser.add_argument("--use_float32_bias", default=0, type=int, required=False)
    parser.add_argument("--inj_loop", default='scan-GPU', type=str, required=False)
    parser.add_argument("--interp_inj", default=0, type=int, required=False)
    parser.add_argument("--detach_var", default=0, type=int, required=False)
    
    parser.add_argument("--nsamplesmax", default=-1, type=int, required=False)
    parser.add_argument("--spin_inj", default='none', type=str, required=False)
    parser.add_argument("--Nsamplesuse", default=-1, type=int, required=False)
    parser.add_argument("--sel_uncertainty", default=0, type=int, required=False)
    parser.add_argument("--sel_smoothing", default='sigmoid', type=str, required=False)
    parser.add_argument("--alpha_beta_prior", default='sigmoid', type=str, required=False)
    parser.add_argument("--dil_factor", default=1, type=int, required=False)
    parser.add_argument("--use_log_alpha_beta", default=0, type=int, required=False)
    
    parser.add_argument("--fout", default='results/', type=str, required=True)
    
    parser.add_argument("--sampler", default='pymc_bar', type=str, required=False)
    parser.add_argument("--nsteps", default=100, type=int, required=True)
    parser.add_argument("--ntune", default=100, type=int, required=True)
    parser.add_argument("--nchains", default=1, type=int, required=False)
    parser.add_argument("--ncores", default=1, type=int, required=False)
    parser.add_argument("--target_accept", default=0.8, type=float, required=False)
    parser.add_argument("--chain_method", default='parallel', type=str, required=False)
    parser.add_argument("--jax_debug_nans", default=0, type=int, required=False)
    parser.add_argument("--dense_mass", default=0, type=int, required=False)
    parser.add_argument("--max_tree_depth", default=10, type=int, required=False)
    
    
    
    parser.add_argument("--fix_H0", default=1, type=int, required=False)
    parser.add_argument("--fix_Om", default=1, type=int, required=False)
    parser.add_argument("--fix_w0", default=1, type=int, required=False)
    parser.add_argument("--fix_Xi0n", default=1, type=int, required=False)
    parser.add_argument("--z_pivot", default=0, type=float, required=False)
    parser.add_argument("--integrate_dc", default='trapz', type=str, required=False)
    
    
    parser.add_argument("--param", default='vanilla', type=str, required=False)
    parser.add_argument("--pade", default=0, type=int, required=False)
    parser.add_argument("--zres", default=1000, type=int, required=False)
    parser.add_argument("--z_grid_mode", default='std', type=str, required=False)
    parser.add_argument("--rebuild_z", default=1, type=int, required=False)
    
    parser.add_argument("--zmin_a", default=1e-05, type=float, required=False)
    parser.add_argument("--zmin_b", default=1e-03, type=float, required=False)
    parser.add_argument("--zmid_b", default=3., type=float, required=False)
    parser.add_argument("--zmax_c", default=10., type=float, required=False)
    parser.add_argument("--hi_boost", default=.2, type=float, required=False)
    parser.add_argument("--find_z_bounds", default=0, type=int, required=False)
    parser.add_argument("--is_observed", default=0, type=int, required=False)
    parser.add_argument("--sample_from_pop", default=0, type=int, required=False)

    parser.add_argument("--mmin_inj", default=-1, type=float, required=False)
    parser.add_argument("--is_compressed_inj", default=0, type=int, required=False)
    
    parser.add_argument("--allTobs", nargs='+', type=float, required=False)

    parser.add_argument("--reparam_mass", default=0, type=int, required=False)
    parser.add_argument("--reparam_z", default=0, type=int, required=False)
    parser.add_argument("--reparam_cosmo", default=0, type=int, required=False)

    parser.add_argument("--xla_cpu_multi_thread_eigen", default='true', type=str, required=False)

    parser.add_argument("--nth", type=int, default=1)



    FLAGS = parser.parse_args()


    # after FLAGS = parser.parse_args():
    if FLAGS.nth is not None and FLAGS.nth != NTH:
        raise ValueError(f"--nth mismatch: early {NTH} vs parsed {FLAGS.nth}")


    from tqdm import tqdm 
    from tqdm.auto import tqdm

    import psutil
    _process = psutil.Process(os.getpid())
    def mem_gb():
        return _process.memory_info().rss / (1024**3)  # Resident Set Size in GB
    
    def log_mem(tag):
        print(f"[MEM] {tag}: {mem_gb():.2f} GB RSS")

    if FLAGS.sampler in ('numpyro', 'blackjax') and FLAGS.backend=='ztrace':

        print(
            f"⚠️ Warning: backend ({FLAGS.backend}) asked, but sampler is {FLAGS.sampler}. "
            "This sampler does not support ztrace. Setting backend to standard."
        )
        FLAGS.backend = 'disk'

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


    extra = [ "optimizer=fast_run" ]
    

    for f in extra:
        if "=" in f:
            k, v = f.split("=", 1)
            set_pytensor_flag(k.strip(), v.strip())
        else:
            set_pytensor_flag(f.strip(), "True")


    print("\nInitial PYTENSOR_FLAGS =", os.environ.get("PYTENSOR_FLAGS"))


    uses_numpyro = FLAGS.sampler in ("numpyro", "blackjax")
    
    # Are we doing PyMC multiprocessing chains?
    using_pymc_multiproc = (FLAGS.sampler not in ("numpyro","blackjax")) and (FLAGS.ncores > 1)
    
    if using_pymc_multiproc:
        # one JAX device per OS process (chain)
        if FLAGS.xla_cpu_multi_thread_eigen=='true':
            print(f"⚠️ Warning: xla_cpu_multi_thread_eigen ({FLAGS.xla_cpu_multi_thread_eigen}) asked, but sampler is using_pymc_multiproc. "
            "Do this if you have good handling of your memory load.")
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={FLAGS.ncores} --xla_cpu_multi_thread_eigen={FLAGS.xla_cpu_multi_thread_eigen} --xla_cpu_enable_fast_math=true"
    else:
        # single-process JAX multi-device (numpyro/blackjax parallel)
        if FLAGS.xla_cpu_multi_thread_eigen=='false':
            print(f"⚠️ Warning: xla_cpu_multi_thread_eigen ({FLAGS.xla_cpu_multi_thread_eigen}) asked, but sampler not using_pymc_multiproc. "
            "Setting --xla_cpu_multi_thread_eigen=true")
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={FLAGS.ncores} --xla_cpu_multi_thread_eigen=true --xla_cpu_enable_fast_math=true"
    

    print("XLA_FLAGS (final) =", os.environ.get("XLA_FLAGS", ""))


        
    
    # ----------------------------------------------------
    # 2️⃣ Import libraries (now they see the environment)
    # ----------------------------------------------------
    
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")

    print("XLA_FLAGS =", os.environ.get("XLA_FLAGS"))
    print("VECLIB_MAXIMUM_THREADS =", os.environ.get("VECLIB_MAXIMUM_THREADS"))



    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir("/tmp/jax_cache")
    
    
    import jaxify_ops
    
    from pytensor.link.jax.dispatch.basic import jax_funcify
    from pytensor_ops import PopAndSelJAXOp  # same import path as in jaxify_ops
    
    assert PopAndSelJAXOp in jax_funcify.registry, "jax_funcify registration did not stick"
    #print("jax_funcify registered for:", jax_funcify.registry[PopAndSelJAXOp])

    
    if uses_numpyro:
        import numpyro
          
    
    if FLAGS.jax_debug_nans:
        jax.config.update("jax_debug_nans", True)   # crash at the first NaN/Inf during warmup
    else:
        jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_default_matmul_precision", "highest")


    
    if FLAGS.chain_method == "parallel" and uses_numpyro:
        numpyro.set_host_device_count(device_count)

    print("Available devices:", jax.devices())
    print("Local device count:", jax.local_device_count())
    print("Backend:", jax.default_backend())

    print("JAX:", jax.__version__)
    if uses_numpyro:
        print("NumPyro:", numpyro.__version__)

    import jax.numpy as np

    print("jax_enable_x64:", jax.config.jax_enable_x64)
    print("devices:", jax.devices())
    print("float64 test dtype:", np.array([1.0], dtype=np.float64).dtype)
    print()

    
    

    # ----------------------------------------------------
    # 3️⃣ Now safe to import PyMC and others
    # ----------------------------------------------------
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

    
    import pymc as pm
    import pytensor
    pytensor.config.openmp = False
    import numpy as onp

    set_pytensor_flag("optimizer_excluding", "inplace")
    
    print("JAX x64 after importing pymc/pytensor:", jax.config.read("jax_enable_x64"))
    
    

    
    # Custom modules
    import pymc_models as models
    import data_tools as dt
    import pytensor_tools as atools
    import pytensor_utils_old as autils



    

    logfile = os.path.join(FLAGS.fout, 'logfile.txt')
    myLog = autils.Logger(logfile)
    sys.stdout = myLog
    sys.stderr = myLog



    print(f"Running on PyMC v{pm.__version__}")

    
    
    print("dtype test:", np.array(0., dtype=np.float64).dtype)
    
    
    
    print(f"[PID] {os.getpid()}")
    

    with open(FLAGS.fin_priors) as json_file:
        priors = json.load(json_file)

    if FLAGS.priors_for_mmin!='':
        with open(FLAGS.priors_for_mmin) as json_file:
            priors_for_mmin = json.load(json_file)

        with open(os.path.join(FLAGS.fout, 'priors_for_mmin.json' ), 'w') as fp:
            json.dump(priors_for_mmin, fp)
    else:
        priors_for_mmin=''
    
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

        data = dt.load_data_interp(FLAGS.fin_data, events_use=FLAGS.events_use)

        # samples_means_at = at.as_tensor_variable(data['samples_means'])
        # samples_cho_covs_at = at.as_tensor_variable(data['samples_cho_covs']*FLAGS.cho_dil)
    
        # gmm_log_wts = at.as_tensor_variable(data['gmm_log_wts'])
        # gmm_means = at.as_tensor_variable(data['gmm_means'])
        # gmm_icovs = at.as_tensor_variable(data['gmm_icovs'])
        # gmm_cho_covs = at.as_tensor_variable(data['gmm_cho_covs'])
        # gmm_log_dets = at.as_tensor_variable(data['gmm_log_dets'])
        # allNgm = at.as_tensor_variable(data['allNgm'])
        # Nevents = at.as_tensor_variable(data['Nevents'])

        samples_means_at = data['samples_means']#.astype(X)
        samples_cho_covs_at = (data['samples_cho_covs']*FLAGS.cho_dil)#.astype(X)
    
        gmm_log_wts = data['gmm_log_wts']#.astype(X)
        gmm_means = data['gmm_means']#.astype(X)
        gmm_icovs =  data['gmm_icovs']#.astype(X)
        gmm_cho_covs =  data['gmm_cho_covs']#.astype(X)
        gmm_log_dets =  data['gmm_log_dets']#.astype(X)
        allNgm =  data['allNgm']#.astype(X)
        Nevents =  data['Nevents']#.astype(X)
        allnames =  data['allnames']

        if FLAGS.nev_min != 0 or FLAGS.nev_max != -1:

            if FLAGS.events_use!=[]:
                raise ValueError("Cannot select by index and name at the same time")
                
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

    
    if FLAGS.is_compressed_inj:
        print("Injections obtained with compression.")
        log_p_incl = injections['log_p_incl']#.astype(XI)
    else:
        log_p_incl = []#np.full(len(injections['dL']), None)
        for _ in range(len(injections['dL'])):
            log_p_incl.append(None)
        


    if FLAGS.spin_model=='none':
        # InjData = [ at.as_tensor_variable(injections['dL']), 
        #         at.as_tensor_variable(injections['m1d']), 
        #         at.as_tensor_variable(injections['m2d']), 
        #         at.as_tensor_variable(injections['log_wt']), 
        #         at.as_tensor_variable(injections['Ngen']), 
        #         at.as_tensor_variable(injections['Ndet']), 
        #           ]
        InjData = [ injections['dL'],#.astype(XI), 
                injections['m1d'],#.astype(XI), 
                injections['m2d'],#.astype(XI), 
                 injections['log_wt'],#.astype(XI), 
                 injections['Ngen'],#.astype(XI), 
                 injections['Ndet'],#.astype(XI), 
                    log_p_incl
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
            InjData = [ injections['dL'],#.astype(XI), 
                 injections['m1d'],#.astype(XI), 
                 injections['m2d'],#.astype(XI), 
                 injections['chieff'],#.astype(XI), 
                 injections['chip'],#.astype(XI), 
                 injections['log_wt'],#.astype(XI), 
                 injections['Ngen'], ##.astype(XI), 
                injections['Ndet'],#.astype(XI), 
                        log_p_incl
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
                InjData = [ injections['dL'], #.astype(XI), 
                     injections['m1d'], #.astype(XI), 
                     injections['m2d'],#.astype(XI), 
                     chi1Inj, #, .astype(XI), 
                     chi2Inj, #.astype(XI),
                     cost1Inj, #.astype(XI),
                     cost2Inj, #.astype(XI),
                     injections['log_wt'], #.astype(XI), 
                     injections['Ngen'], #.astype(XI), 
                     injections['Ndet'], #.astype(XI), 
                            log_p_incl
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
                InjData = [ injections['dL'], #m .astype(XI), 
                    injections['m1d'], #, .astype(XI), 
                    injections['m2d'], #.astype(XI), 
                    injections['log_wt'], #.astype(XI), 
                    injections['Ngen'], #.astype(XI), 
                    injections['Ndet'], #.astype(XI), 
                            log_p_incl
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
                InjData = [ injections['dL'], #.astype(XI), 
                     injections['m1d'], # .astype(XI), 
                     injections['m2d'], #.astype(XI), 
                     injections['chi1'], #.astype(XI), 
                     injections['chi2'], #.astype(XI),
                     injections['cost1'], #.astype(XI),
                     injections['cost2'], #.astype(XI),
                    injections['log_wt'], #.astype(XI), 
                     injections['Ngen'], #.astype(XI), 
                     injections['Ndet'], #.astype(XI), 
                            log_p_incl
                      ]
        else:
            raise ValueError('Enter valid spin model.')

    
            
    if not FLAGS.pop_only:  
    
        if FLAGS.sampling_gw=='gmm_cat':
            GWData =  [
                       onp.exp(gmm_log_wts), #.astype(X), 
                       gmm_means, #.astype(X), 
                       gmm_cho_covs, #.astype(X), 
                       injections['Tobs'], #.astype(X),
                        Nevents,
                        allnames
                      ]
        elif FLAGS.sampling_gw=='gauss' or FLAGS.sampling_gw=='gmm_marg':
            GWData =  [samples_means_at, #.astype(X), 
                       samples_cho_covs_at, #astype(X), 
                       gmm_log_wts, #.astype(X), 
                       gmm_means, #.astype(X), 
                       gmm_icovs, #.astype(X), 
                       gmm_log_dets, #.astype(X), 
                       gmm_cho_covs, #.astype(X),
                       injections['Tobs'], #.astype(X),
                       Nevents, 
                       allnames
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

    use_updates =  ('pymc' in FLAGS.sampler)
    # right before building the model
    log_mem("before make_model")
    t0 = time.time()
    model = models.make_model(  priors,
                                    GWData,
                                    InjData,
                                    ivals=ivals,
                                    eps_init = FLAGS.eps_init,
                                    sampling_GW = FLAGS.sampling_gw,
                                    rate_model = FLAGS.rate_model,
                                    mass_model = FLAGS.mass_model,
                                    smoothing=FLAGS.smoothing,
                                    simplex_repair=FLAGS.simplex_repair,
                                    has_m2_break=FLAGS.has_m2_break,
                                    norm_gauss=FLAGS.norm_gauss,
                                    interp_mass = FLAGS.interp_mass,
                                    linear_mass=FLAGS.linear_mass,
                                    #linear_z=FLAGS.linear_z,
                                    interp_z = FLAGS.interp_z,
                                    rebuild_z = FLAGS.rebuild_z,
                                    spin_model = FLAGS.spin_model,
                                    spin_inj = FLAGS.spin_inj,
                                    dLprior = FLAGS.dLprior,
                                    #normalize_PE_prior=FLAGS.normalize_PE_prior,
                                    penorm_lims = FLAGS.penorm_lims,
                                    sel_method=FLAGS.sel,
                                    fix_inj_len=FLAGS.fix_inj_len,
                                    use_float32 = FLAGS.use_float32,
                                    use_float32_bias = FLAGS.use_float32_bias,
                                    chunk_inj=FLAGS.chunk_inj,
                                    chunk_reduce = FLAGS.chunk_reduce,
                                    marginal_R0 = FLAGS.marginal_R0,
                                    N_DP_comp_max = FLAGS.N_DP_comp_max,
                                    DP_truncate_up = FLAGS.DP_truncate_up,
                                    DP_truncate_low = FLAGS.DP_truncate_low,
                                    DP_m1_env = FLAGS.DP_m1_env,
                                    alpha_tail = FLAGS.alpha_tail,
                                    alpha_small = FLAGS.alpha_small,
                                    L_small_1 = FLAGS.L_small_1,
                                    L_small_2 = FLAGS.L_small_2,
                                    L_small_3 = FLAGS.L_small_3,
                                    tau_prior = FLAGS.tau_prior,
                                    alpha_inv_params = FLAGS.alpha_inv_params,
                                    M_active = FLAGS.M_active,
                                    s_local = FLAGS.s_local,
                                    find_m_bounds = FLAGS.find_m_bounds,
                                    q_mbound = FLAGS.q_mbound,
                                    fix_H0 = FLAGS.fix_H0,
                                    fix_Om = FLAGS.fix_Om,
                                    fix_w0 = FLAGS.fix_w0,
                                    fix_Xi0n = FLAGS.fix_Xi0n,
                                    z_pivot=FLAGS.z_pivot,
                                    integrate_dc = FLAGS.integrate_dc,
                                    zres = FLAGS.zres,
                                    z_grid_mode=FLAGS.z_grid_mode,
                                    zmin_a=FLAGS.zmin_a, 
                                    zmin_b=FLAGS.zmin_b, 
                                    zmid_b=FLAGS.zmid_b, 
                                    zmax_c=FLAGS.zmax_c, 
                                    hi_boost=FLAGS.hi_boost,
                                    find_z_bounds = FLAGS.find_z_bounds,
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
                                save_thetas = FLAGS.save_thetas,
                                interp_inj=FLAGS.interp_inj,
                                param=FLAGS.param,
                                DP_prior=FLAGS.DP_prior,
                                sigma_softmax=FLAGS.sigma_softmax,
                                gamma_DP_params=FLAGS.gamma_DP_params,
                                is_observed = FLAGS.is_observed,
                                sample_from_pop = FLAGS.sample_from_pop,
                                mmin_inj=FLAGS.mmin_inj,
                                is_compressed_inj=FLAGS.is_compressed_inj,
                                debug_sel_batch=FLAGS.debug_sel_batch,
                                reparam_z = FLAGS.reparam_z,
                                 reparam_mass = FLAGS.reparam_mass,
                                reparam_cosmo = FLAGS.reparam_cosmo,
                                priors_for_mmin=priors_for_mmin,
                                detach_var = FLAGS.detach_var,
                                remove_spin_prior = FLAGS.remove_spin_prior
                                )
    print(f"[TIMER] make_model took {time.time()-t0:.1f}s")
    log_mem("after make_model")
    print('Done.')


    # print()
    # print('*'*80)
    # print('Timing grad...')
    # print('*'*80)

    # with model:
    #     f_logp = model.compile_logp(sum=True)     # returns a callable(point_dict)->float
    #     f_grad = model.compile_dlogp()            # callable(point_dict)->1D array (in var order)
    
    # pt0 = model.initial_point()
    
    # # warmup (compile + first run)
    # _ = f_logp(pt0)
    # _ = f_grad(pt0)
    
    # # logp only
    # t0 = time.perf_counter()
    # for _ in range(5):
    #     _ = f_logp(pt0)
    # t1 = time.perf_counter()
    # print("PyMC logp avg (s):", (t1 - t0)/5)
    
    # # grad only
    # t0 = time.perf_counter()
    # for _ in range(5):
    #     _ = f_grad(pt0)
    # t1 = time.perf_counter()
    # print("PyMC grad avg (s):", (t1 - t0)/5)

    
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
        backend = ZarrTrace(store=spath, draws_per_chunk=FLAGS.draws_per_chunk)
        print("Intermediate trace will be stored at %s"%spath)
        print("Saving every %s steps"%FLAGS.draws_per_chunk)
        print("zarr:", zarr.__version__, "| numcodecs:", numcodecs.__version__)
    else:
        raise ValueError("backend can be disk or ztrace, got %s"%FLAGS.backend)

         
    if FLAGS.debug_sel_batch:
        print("\nMODEL RV CHECKS>>>>>>")
        import pytensor
        import pytensor.tensor as at
        from pytensor.graph.basic import graph_inputs
        from pytensor.tensor.sharedvar import SharedVariable
        from pytensor.tensor.random.type import RandomType
        
        with model:
            logp_var = model.logp()   # scalar log-prob of the entire model
        
        ins = list(graph_inputs([logp_var]))
        bad = [
            v for v in ins
            if isinstance(v, SharedVariable) and isinstance(v.type, RandomType)
        ]
        
        print("Number of Shared[RandomType] vars in model.logp graph:", len(bad))
        for v in bad:
            print("  name:", v.name, "| type:", v.type)
            if v.owner is not None:
                print("    owner op:", type(v.owner.op))
                print("    inputs:", [getattr(inp, "name", None) for inp in v.owner.inputs])

    ################################################
    # Run sampler
    ################################################

    if int(pm.__version__.split('.')[1])>20: # recent versions of pymc

        
        with model:

            print("Setting initial point...")
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

                if FLAGS.use_float32:
                    import traceback
                    from pymc.initial_point import make_initial_point_expression
                    
                    free_rvs = list(model.free_RVs)
                    
                    bad = []
                    for rv in free_rvs:
                        try:
                            _ = make_initial_point_expression(
                                free_rvs=[rv],
                                rvs_to_transforms=model.rvs_to_transforms,
                                initval_strategies={},          # required in PyMC 5.1
                                jitter_rvs=None,
                                default_strategy="support_point",
                                return_transformed=True,
                            )
                        except TypeError as e:
                            bad.append(rv)
                            print("\n=== FAIL ===")
                            print("RV:", rv.name, "dtype:", getattr(rv, "dtype", None))
                            print("Error:", e)
                            # optional: print short traceback
                            # traceback.print_exc(limit=2)
                    
                    print("\nBad RVs:", [rv.name for rv in bad])
                    
                ip = model.initial_point()
                # N = gmm_means.shape[0]
                # nd = gmm_means.shape[2]
    
                # ip['x'] = onp.random.randn(N, nd) * FLAGS.eps_init
    
    
                # wts = onp.exp(gmm_log_wts)
                # idx = onp.argmax(wts, axis=1)
                
                # if FLAGS.sampling_gw=='gmm_cat':
                #     ip['idx'] = idx.astype(int)
    

                
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
                    print("\nComputing initial log-likelihood...")
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
                    


            print("\nModel variables:")
            mvars = model.value_vars
            # Print only the names of variables that are sampled
            print(vnames)

            # NOT SUPPORTED
            # if uses_numpyro and FLAGS.dense_mass:
            #     lambda_sites = [i for i,n in enumerate(vnames) if n != "x"]
        
            #     # Safety: if your model ever has other huge latents, exclude them here too:
            #     # lambda_sites = [n for n in lambda_sites if n not in ("x", "something_else_big")]
                
            #     # NumPyro expects a list of tuples (each tuple is one dense block)
            #     dense_blocks = [tuple(lambda_sites)] if len(lambda_sites) > 1 else False
            #     print("[INFO] dense_mass blocks:", dense_blocks)
            # else:
            #     dense_blocks = 0
            
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

            if MAP_init:
                
                #print(mvars)
                #print("Initial point with MAP:")
                #print(ip)

                #print("mvars:")
                #print(mvars)

                #print("vnames:")
                #print(vnames)
                
                #print("ip keys:")
                #print( list(ip.keys()))
                
                #print("Initial values from MAP:")
                
                ivals = {k: ip[k] for k in ip.keys() if k in vnames}
                #print(ivals)

                # Make JSON-serializable
                ivals_json = {}
                for k, v in ivals.items():
                    arr = np.asarray(v)
                    if arr.shape == ():      # scalar
                        ivals_json[k] = float(arr)
                    else:
                        ivals_json[k] = arr.tolist()
                
                with open( os.path.join( FLAGS.fout,"ivals_MAP.json"), "w") as f:
                    json.dump(ivals_json, f, indent=2)

                sampler_kwargs['initvals'] = MAP
    
            if FLAGS.sampler == "numpyro":

                if FLAGS.check_init:
                    from pymc.sampling.jax import get_jaxified_logp
                    from pymc.initial_point import make_initial_point_fn
                    import jax.numpy as jnp
                
                    # internal initial point in *value_var* space
                    if MAP_init:
                        # Build an internal initial-point function that uses MAP as initvals
                        ip_fn = make_initial_point_fn(model=model, overrides=MAP)
                        
                        # One concrete internal point (value_var space)
                        ip_internal = ip_fn(42)  # seed
                        
                        #print("value_vars:", [vv.name for vv in model.value_vars])
                        print("ip_internal keys:", ip_internal.keys())

                    else:
                        ip_internal = ip
                        
                    value_vars = model.value_vars
                    x0 = [jnp.asarray(ip_internal[vv.name]) for vv in value_vars]
                        
            
                    jax_logp_fn = get_jaxified_logp(model)
                
                    print("Testing JAX logp at PyMC initial point...")
                    print(jax_logp_fn(x0))   

                
                sampler = "numpyro"
                sampler_kwargs.update({
                                # "cores": 1,                         # JAX: single OS process
                                "target_accept": FLAGS.target_accept,  
                                "nuts_sampler_kwargs": {
                                    "jitter": False, 
                                    "chain_method": FLAGS.chain_method,   # fast on single device
                                    "nuts_kwargs": {
                                        # Choose one:
                                        "dense_mass": FLAGS.dense_mass, 
                                        "adapt_step_size": True,
                                        "adapt_mass_matrix": True,
                                        "regularize_mass_matrix": 1e-3,
                                        "find_heuristic_step_size": False,# let NumPyro pick a good initial step
                                        "step_size":0.01,
                                        "max_tree_depth": FLAGS.max_tree_depth,
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

                
                

                if FLAGS.dense_mass:
                    sampler_kwargs["init"] = "adapt_full"
                else:
                    ta = sampler_kwargs.pop("target_accept", FLAGS.target_accept)
                    sampler_kwargs["step"] = pm.NUTS(target_accept=ta, max_treedepth=FLAGS.max_tree_depth)

    



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
                    
                    
                    #draws = sampler_kwargs.get("draws", 1000)
                    #tune = sampler_kwargs.get("tune", 1000)
                    #total_steps = FLAGS.nchains * (tune + draws)  # 4 * 40 = 160
                    #cb=autils.TqdmGlobalCallback(draws=draws, tune=tune, chains=FLAGS.nchains,)
                    #if uses_jax:
                    print("PID", os.getpid(), "jax_enable_x64", jax.config.read("jax_enable_x64"))
                    print("PID", os.getpid(), "JAX_ENABLE_X64 env", os.environ.get("JAX_ENABLE_X64"))
                    
                    trace = pm.sample(nuts_sampler='pymc', 
                                      idata_kwargs={"log_likelihood": False},
                                      callback=cb,
                                      #max_treedepth=FLAGS.max_tree_depth, 
                                      **sampler_kwargs,    
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

                # if FLAGS.sampler=='blaxkjax':
                #     sampler_kwargs["progressbar"] =False
                
                
                trace = pm.sample(nuts_sampler=FLAGS.sampler, 
                                  idata_kwargs={"log_likelihood": False}, 
                                  pytensor_kwargs={"allow_gc": True}, 
                                  **sampler_kwargs)
                
                
                
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
    

    import arviz as az
    
    if FLAGS.backend=='disk':
        trace.to_netcdf( os.path.join(FLAGS.fout, "trace.nc"))
    else:
        # Fetch the run 
        try:
            idata = autils.load_pymc_zarr_trace_robust(spath)  # your working loader
            print( "idata loaded." )
            idata.sample_stats = idata.sample_stats.drop_vars(["sampler_0__warning"], errors="ignore")
            #idata_clean = autils.drop_object_vars(idata)
            print( "idata cleaned." )
            #az.to_netcdf(idata, os.path.join(FLAGS.fout, "trace.nc"))
            idata.to_netcdf(os.path.join(FLAGS.fout, "trace.nc"))
            print( "trace saved." )

        except Exception as e:
            print(e)
            print( "No final trace saved." )

        

    #########

    print("\nMaking summary plots...")

    
    import matplotlib.pyplot as plt

    vplot = [v for v in vnames if v not in ('beta', 'mulMc', 'mulq', 'eps1', 'eps2', 'x', 'idx', 'lambda')]

    try:
        print("Plotting trace...")
        az.plot_trace(trace, var_names = vplot, );
        plt.savefig( os.path.join(FLAGS.fout, 'trace.pdf'), bbox_inches='tight')
        plt.close()
    except:
        print('No trace plot produced')

    try:
        import corner
        print("Plotting corner...")
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

    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
        print("Spawn set")
    except RuntimeError:
        pass  # already set

    
    main()
    
    



