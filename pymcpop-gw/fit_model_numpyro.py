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
#os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

# BLAS/OpenMP caps
os.environ["OMP_NUM_THREADS"]      = str(NTH)
os.environ["OPENBLAS_NUM_THREADS"] = str(NTH)
os.environ["MKL_NUM_THREADS"]      = str(NTH)
os.environ["NUMEXPR_NUM_THREADS"]  = str(NTH)
os.environ["BLIS_NUM_THREADS"]     = str(NTH)
os.environ["OMP_DYNAMIC"]          = "FALSE"
os.environ["KMP_AFFINITY"]         = "disabled"
os.environ["KMP_BLOCKTIME"]        = "0"

# also cap common threadpool names
os.environ["TF_NUM_INTRAOP_THREADS"] = str(NTH)
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

os.environ["ACCELERATE_MAX_THREADS"] = str(NTH)

os.environ["VECLIB_MAXIMUM_THREADS"] = str(NTH)   # macOS Accelerate


# JAX uses a threadpool; this often works better than intra/inter flags on macOS
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_DISABLE_MOST_OPTIMIZATIONS"] = "0"
os.environ["JAX_NUM_THREADS"] = str(NTH)   # try: 1


# IMPORTANT: cap XLA’s own CPU threadpool
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_enable_fast_math=true"
)


print()
#print("XLA_FLAGS (final) =", os.environ["XLA_FLAGS"])


import json
import warnings
import time
import resource

    


# Writes output both on std output and on log file
class Logger(object):
    
    def __init__(self, fname):
        self.terminal = sys.__stdout__
        self.log = open(fname, "w+")
        self.log.write('--------- LOG FILE ---------\n')
        print('Logger created log file: %s' %fname)
        #self.write('Logger')
       
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.close
        sys.stdout = sys.__stdout__
        
    def isatty(self):
        return False



def jitter_positive(v, key, eps):
    v = jnp.asarray(v)
    noise = jax.random.normal(key, shape=v.shape)
    return v * jnp.exp(eps * noise)   # strictly > 0, relative


def jitter_uniform(v, a, b, key, eps, margin=1e-6):
    v = jnp.asarray(v)
    a = jnp.asarray(a, dtype=v.dtype)
    b = jnp.asarray(b, dtype=v.dtype)

    t = (v - a) / (b - a)
    t = jnp.clip(t, margin, 1 - margin)

    z = jnp.log(t) - jnp.log1p(-t)              # logit
    z = z + eps * jax.random.normal(key, v.shape)
    t2 = jax.nn.sigmoid(z)

    return a + (b - a) * t2


def jitter_simplex(v, key, eps):
    v = jnp.asarray(v)
    v = jnp.clip(v, 1e-12, 1.0)      # avoid log(0)
    y = jnp.log(v)
    y = y + eps * jax.random.normal(key, shape=v.shape)
    return jax.nn.softmax(y)



def main():

    
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("--fin_data", nargs='+', type=str, required=True)
    parser.add_argument("--fin_injections", nargs='+', type=str, required=True)
    parser.add_argument("--fin_priors", default='', type=str, required=True)
    parser.add_argument("--priors_for_mmin", default='', type=str, required=False)
    parser.add_argument("--events_use", nargs='+', default=[], type=str, required=False)
    parser.add_argument("--backend", default='ztrace', type=str, required=False)
    parser.add_argument("--seed", default=0, type=int, required=False)

    
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
    parser.add_argument("--linear_z", default=0, type=int, required=False)

    parser.add_argument("--vary_mb", default=1, type=int, required=False)

    
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
    
    

    
    
    
    parser.add_argument("--marginal_R0", default=1, type=int, required=False)
    parser.add_argument("--smoothing", default='LVK', type=str, required=False)
    parser.add_argument("--simplex_repair", default=0, type=int, required=False)

    parser.add_argument("--has_m2_break", default=0, type=int, required=False)
    parser.add_argument("--norm_gauss", default='uplow', type=str, required=False)
    
    
    
    parser.add_argument("--dLprior", nargs='+', default=['none'], type=str, required=False)
    #parser.add_argument("--normalize_PE_prior",  default=1, type=int, required=False)
    parser.add_argument("--penorm_lims",  nargs='+', default=[], type=str, required=False)
    parser.add_argument("--use_sel_spin", default=0, type=int, required=False)
    
    
    parser.add_argument("--sampling_gw", default='gmm_cat', type=str, required=False)
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
    
    #parser.add_argument("--sampler", default='pymc_bar', type=str, required=False)
    parser.add_argument("--nsteps", default=100, type=int, required=True)
    parser.add_argument("--ntune", default=100, type=int, required=True)
    parser.add_argument("--nchains", default=1, type=int, required=False)
    parser.add_argument("--ncores", default=1, type=int, required=False)
    parser.add_argument("--target_accept", default=0.9, type=float, required=False)
    parser.add_argument("--chain_method", default='sequential', type=str, required=False)
    parser.add_argument("--jax_debug_nans", default=0, type=int, required=False)
    parser.add_argument("--dense_mass", default=0, type=int, required=False)
    parser.add_argument("--max_tree_depth", default=10, type=int, required=False)
    parser.add_argument("--find_heuristic_step_size", default=0, type=int, required=False)
    parser.add_argument("--regularize_mass_matrix", default=1e-04, type=float, required=False)

    parser.add_argument("--check_zres", nargs="+", type=int, default=[], required=False)
    
    
    parser.add_argument("--fix_H0", default=1, type=int, required=False)
    parser.add_argument("--fix_Om", default=1, type=int, required=False)
    parser.add_argument("--fix_w0", default=1, type=int, required=False)
    parser.add_argument("--fix_Xi0n", default=1, type=int, required=False)
    parser.add_argument("--z_pivot", default=0, type=float, required=False)
    parser.add_argument("--integrate_dc", default='pade', type=str, required=False)
    
    
    parser.add_argument("--param", default='vanilla', type=str, required=False)
    parser.add_argument("--pade", default=0, type=int, required=False)
    parser.add_argument("--zres", default=1000, type=int, required=False)
    parser.add_argument("--z_grid_mode", default='man', type=str, required=False)
    
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
    
    parser.add_argument("--r",  default=0, type=float, required=False)
    parser.add_argument("--allTobs", nargs='+', type=float, required=False)

    parser.add_argument("--reparam_mass", default=0, type=int, required=False)
    parser.add_argument("--reparam_z", default=0, type=int, required=False)

    parser.add_argument("--remove_spin_prior", default=0, type=int, required=False)

    

    parser.add_argument("--xla_cpu_multi_thread_eigen", default='true', type=str, required=False)

    parser.add_argument("--nth", type=int, default=None)



    FLAGS = parser.parse_args()




    #from tqdm import tqdm 
    from tqdm.auto import tqdm

    import psutil
    _process = psutil.Process(os.getpid())
    def mem_gb():
        return _process.memory_info().rss / (1024**3)  # Resident Set Size in GB
    
    def log_mem(tag):
        print(f"[MEM] {tag}: {mem_gb():.2f} GB RSS")

 
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
    # Sampler / device configuration (NumPyro only)
    # ----------------------------------------------------
    
    uses_numpyro = True
    if not uses_numpyro:
        raise ValueError("This script is configured for NumPyro; set uses_numpyro=True.")
    
    # Decide how many JAX "host devices" to expose in this process.
    # For CPU this controls how many virtual devices XLA creates.
    # For GPU/TPU it should generally be left alone (but your script is CPU-oriented).
    if FLAGS.chain_method == "vectorized":
        # One process, chains are vmapped -> 1 device is enough.
        device_count = 1
    elif FLAGS.chain_method == "parallel":
        # One process, multiple devices, one chain per device.
        # Must be >= nchains to truly run all chains in parallel.
        device_count = int(FLAGS.ncores)
    elif FLAGS.chain_method == "sequential":
        # One process, chains run one after the other.
        device_count = 1
    else:
        raise ValueError("chain_method must be one of: vectorized, parallel, sequential")
    
    # Safety checks
    if FLAGS.chain_method == "vectorized" and FLAGS.ncores > 1:
        raise ValueError(
            "For chain_method='vectorized', set ncores=1. "
            "Vectorized mode runs all chains in one JAX process and does not use multiprocessing."
        )
    
    if FLAGS.chain_method == "parallel" and device_count < int(FLAGS.nchains):
        print(
            f"⚠️ Warning: device_count ({device_count}) < nchains ({FLAGS.nchains}). "
            "Chains will not all run concurrently; consider increasing --ncores."
        )
    
    # XLA flags: expose exactly `device_count` CPU devices in this single process.
    # IMPORTANT: this must be set BEFORE importing jax.
    # If user requested xla_cpu_multi_thread_eigen='false' but we are in single-process mode,
    # we force it to true because it tends to behave better for multi-device CPU runs.
    xla_eigen = FLAGS.xla_cpu_multi_thread_eigen
    if xla_eigen == "false" and FLAGS.chain_method in ("parallel", "vectorized"):
        print("⚠️ Overriding --xla_cpu_multi_thread_eigen=false -> true for NumPyro single-process chains.")
        xla_eigen = "true"
    
    xla_flags = [
        f"--xla_force_host_platform_device_count={device_count}",
        f"--xla_cpu_multi_thread_eigen={xla_eigen}",
        "--xla_cpu_enable_fast_math=true",
    ]
    
    os.environ["XLA_FLAGS"] = " ".join(xla_flags)
    print("XLA_FLAGS (final) =", os.environ["XLA_FLAGS"])
    
    
    # ----------------------------------------------------
    # 2️⃣ Import libraries (now they see the environment)
    # ----------------------------------------------------
    
    import jax
    jax.config.update("jax_enable_x64", True)
    #jax.config.update("jax_default_matmul_precision", "highest")
    
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir("/tmp/jax_cache")
    
    import numpyro
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.initialization import init_to_value, init_to_feasible, init_to_median
    from numpyro.infer.util import initialize_model
    import jax.random as random
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.util import initialize_model

    import arviz as az

    
    if FLAGS.jax_debug_nans:
        jax.config.update("jax_debug_nans", True)
    else:
        jax.config.update("jax_debug_nans", False)
    
    # Tell NumPyro how many devices to use for parallel chains.
    # This MUST match the XLA virtual device count above for CPU.
    if FLAGS.chain_method == "parallel":
        numpyro.set_host_device_count(device_count)
    
    print("Available devices:", jax.devices())
    print("Local device count:", jax.local_device_count())
    print("Backend:", jax.default_backend())
    print("JAX:", jax.__version__)
    print("NumPyro:", numpyro.__version__)
    print("jax_enable_x64:", jax.config.jax_enable_x64)
    


    import jax.numpy as jnp
    import numpy as np


    def _as_jax(v):
        # keep None
        if v is None:
            return None
        # convert lists/tuples to jnp arrays
        if isinstance(v, (list, tuple)):
            return jnp.asarray(v)
        # convert numpy arrays to jnp
        if isinstance(v, np.ndarray):
            return jnp.asarray(v)
        # python scalars -> float
        if isinstance(v, (int, float)):
            return float(v)
        return v

    print("float64 test dtype:", np.array([1.0], dtype=np.float64).dtype)
    print()

    
    

    # ----------------------------------------------------
    # 3️⃣ Now safe to import others
    # ----------------------------------------------------
          
    

    
    # Custom modules import here
    import data_tools as dt
    from jax_models import make_model_jax
    import jax_models as jm
    

    

    logfile = os.path.join(FLAGS.fout, 'logfile.txt')
    myLog = Logger(logfile)
    sys.stdout = myLog
    sys.stderr = myLog
  
    
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

            spin_samples = np.asarray([ chi1_samples, chi2_samples, cost1_samples, cost2_samples ])

        elif FLAGS.spin_model=='none':
            spin_samples = np.asarray([  ])
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
        log_p_incl = injections['log_p_incl']
    else:
        log_p_incl = []
        for _ in range(len(injections['dL'])):
            log_p_incl.append( np.squeeze(np.zeros_like(injections['dL'])) )
        


    if FLAGS.spin_model=='none':

        spinsInj = []
    else:
        
        if FLAGS.spin_inj=='chieffchip':
            spinsInj = [injections['chieff'], injections['chip']]
            
        elif FLAGS.spin_inj=='chi12xyz':

            if (FLAGS.spin_model=='default') or (FLAGS.spin_model=='default_gauss'):

                print("Computing chi1, chi2, cost1, cost2 in injections...")
    
                chi1Inj = np.sqrt(injections['spin1x']**2+injections['spin1y']**2+injections['spin1z']**2)
                chi2Inj = np.sqrt(injections['spin2x']**2+injections['spin2y']**2+injections['spin2z']**2)
    
                cost1Inj = injections['spin1z']/chi1Inj
                cost2Inj = injections['spin2z']/chi2Inj
                
                spinsInj = [chi1Inj, chi2Inj, cost1Inj, cost2Inj]


            elif FLAGS.spin_model=='none':

                print("Injections data has spins but those will not be used !")
    
                spinsInj = []
            
                
        elif FLAGS.spin_inj=='default' or FLAGS.spin_inj=='default_gauss':

                spinsInj = [injections['chi1'], injections['chi2'], injections['cost1'], injections['cost2']]
                
        else:
            raise ValueError('Enter valid spin model.')

    
    InjData = [ injections['dL'], #.astype(XI), 
                     injections['m1d'], # .astype(XI), 
                     injections['m2d'], #.astype(XI), 
                     spinsInj, 
                    injections['log_wt'], #.astype(XI), 
                     injections['Ngen'], #.astype(XI), 
                     injections['Ndet'], #.astype(XI), 
                            log_p_incl
                      ]
        
    if not FLAGS.pop_only:  
    
        if 'gmm' in FLAGS.sampling_gw:
            raise NotImplementedError()
            
            GWData =  [
                       np.exp(gmm_log_wts), #.astype(X), 
                       gmm_means, #.astype(X), 
                       gmm_cho_covs, #.astype(X), 
                       injections['Tobs'], #.astype(X),
                        Nevents,
                        allnames
                      ]
        elif FLAGS.sampling_gw=='gauss':
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
    print()

    if FLAGS.check_zres ==[]:

        model_numpyro, lik_data, core, loglik = make_model_jax(
            priors=priors,
            GWData=GWData,
            InjData=InjData,
            ivals=ivals,
            pop_only = bool(FLAGS.pop_only),
            eps_init=FLAGS.eps_init,
            sampling_GW=FLAGS.sampling_gw,
            rate_model=FLAGS.rate_model,
            mass_model=FLAGS.mass_model,
            smoothing=FLAGS.smoothing,
            simplex_repair=bool(FLAGS.simplex_repair),
            interp_mass=FLAGS.interp_mass,
            interp_z=FLAGS.interp_z,
            has_m2_break=bool(FLAGS.has_m2_break),
            norm_gauss=FLAGS.norm_gauss,
            spin_model=FLAGS.spin_model,
            spin_inj=FLAGS.spin_inj,
            marginal_R0=bool(FLAGS.marginal_R0),
            dLprior=FLAGS.dLprior,
            chunk_inj=FLAGS.chunk_inj,
            chunk_reduce= int(FLAGS.chunk_reduce),
            use_float32=bool(FLAGS.use_float32),
            use_float32_bias=bool(FLAGS.use_float32_bias),
            sel_method=FLAGS.sel,
            sel_smoothing = FLAGS.sel_smoothing,
            N_DP_comp_max=FLAGS.N_DP_comp_max,
            DP_m1_env=bool(FLAGS.DP_m1_env),
            integrate_dc=FLAGS.integrate_dc,
            param=FLAGS.param,
            pade=bool(FLAGS.pade),
            zres=FLAGS.zres,
            z_grid_mode=FLAGS.z_grid_mode,
            zmin_a=FLAGS.zmin_a, zmin_b=FLAGS.zmin_b, zmid_b=FLAGS.zmid_b, zmax_c=FLAGS.zmax_c, hi_boost=FLAGS.hi_boost,
            find_z_bounds=bool(FLAGS.find_z_bounds),
            sample_from_pop=bool(FLAGS.sample_from_pop),
            penorm_lims=FLAGS.penorm_lims,
            use_sel_spin=bool(FLAGS.use_sel_spin),
            allTobs=(np.asarray(FLAGS.allTobs, dtype=np.float64) if FLAGS.allTobs is not None else None),
            params_fix=params_fix,
            fix_H0=bool(FLAGS.fix_H0),
            fix_Om=bool(FLAGS.fix_Om),
            fix_w0=bool(FLAGS.fix_w0),
            fix_Xi0n=bool(FLAGS.fix_Xi0n),
            reparam_mass = bool(FLAGS.reparam_mass),
            remove_spin_prior =  bool(FLAGS.remove_spin_prior),
            r = FLAGS.r,
            vary_mb = FLAGS.vary_mb
        )


    else:
        
        def build_model_for_zres(zres_value):
            return make_model_jax(
            priors=priors,
            GWData=GWData,
            InjData=InjData,
            ivals=ivals,
            pop_only=bool(FLAGS.pop_only),
            eps_init=FLAGS.eps_init,
            sampling_GW=FLAGS.sampling_gw,
            rate_model=FLAGS.rate_model,
            mass_model=FLAGS.mass_model,
            smoothing=FLAGS.smoothing,
            simplex_repair=bool(FLAGS.simplex_repair),
            interp_mass=FLAGS.interp_mass,
            interp_z=FLAGS.interp_z,
            has_m2_break=bool(FLAGS.has_m2_break),
            norm_gauss=FLAGS.norm_gauss,
            spin_model=FLAGS.spin_model,
            spin_inj=FLAGS.spin_inj,
            marginal_R0=bool(FLAGS.marginal_R0),
            dLprior=FLAGS.dLprior,
            chunk_inj=FLAGS.chunk_inj,
            chunk_reduce=int(FLAGS.chunk_reduce),
            use_float32=bool(FLAGS.use_float32),
            use_float32_bias=bool(FLAGS.use_float32_bias),
            sel_method=FLAGS.sel,
            sel_smoothing=FLAGS.sel_smoothing,
            N_DP_comp_max=FLAGS.N_DP_comp_max,
            DP_m1_env=bool(FLAGS.DP_m1_env),
            integrate_dc=FLAGS.integrate_dc,
            param=FLAGS.param,
            pade=bool(FLAGS.pade),
            zres=int(zres_value),
            z_grid_mode=FLAGS.z_grid_mode,
            zmin_a=FLAGS.zmin_a,
            zmin_b=FLAGS.zmin_b,
            zmid_b=FLAGS.zmid_b,
            zmax_c=FLAGS.zmax_c,
            hi_boost=FLAGS.hi_boost,
            find_z_bounds=bool(FLAGS.find_z_bounds),
            sample_from_pop=bool(FLAGS.sample_from_pop),
            penorm_lims=FLAGS.penorm_lims,
            use_sel_spin=bool(FLAGS.use_sel_spin),
            allTobs=(np.asarray(FLAGS.allTobs, dtype=np.float64) if FLAGS.allTobs is not None else None),
            params_fix=params_fix,
            fix_H0=bool(FLAGS.fix_H0),
            fix_Om=bool(FLAGS.fix_Om),
            fix_w0=bool(FLAGS.fix_w0),
            fix_Xi0n=bool(FLAGS.fix_Xi0n),
            reparam_mass=bool(FLAGS.reparam_mass),
            remove_spin_prior=bool(FLAGS.remove_spin_prior),
            r=FLAGS.r,
            vary_mb=FLAGS.vary_mb,
        )
    
    
        model_numpyro, lik_data, core, loglik = build_model_for_zres(FLAGS.zres)

        print("\n" + "=" * 80)
        print("CHECKING REDSHIFT GRID RESOLUTION")
        print("=" * 80)
    
        Lambda_test = jnp.asarray([
            67.9, 0.3065, -1.0, 1.0, 0.0,
            3.2, 3.0, 2.0,
            0.024333031991381315,
            0.31873272890864474,
            0.2123453198667594,
            3.0206244922342362,
    
            1.7, 4.5, 35.0,
            8.5, 0.4, 30.0, 3.5,
            3.189, 150.0, 4.3,
            0.36, 0.59, 0.05,
            1.2, 3.054, 4.9,
            0.1, 45.0, 70.0, 1e-4, 1e-4,
    
            1.7, 1.1, 0.5,
            3.5, 1.1, 0.5,
            35.0, 1.1, 0.5,
            15.0, 1.1, 0.5,
            2.5, 1.1, 0.5,
            60.0, 1.1, 0.5,
            15.0, 1.1, 0.5,
            0.20, 0.20, 0.60,
            1.1, 0.5,
        ], dtype=jnp.float64)
    
        results = []
    
        for zr in FLAGS.check_zres:
            print("\n" + "-" * 80)
            print("zres =", zr)
            print("-" * 80)
    
            _, lik_data_z, core_z, loglik_z = build_model_for_zres(zr)
    
            if FLAGS.pop_only:
                ll, var = loglik_z(Lambda_test, lR0=0.0)
            else:
                x0 = jnp.zeros(
                    (int(lik_data_z.Nobs), int(lik_data_z.mus_s.shape[1])),
                    dtype=jnp.float64,
                )
                ll, var = loglik_z(Lambda_test, x0, lR0=0.0)
    
            ll_f = float(ll)
            var_f = float(var)
            print("ll =", ll_f)
            print("var =", var_f)
    
            results.append((int(zr), ll_f, var_f))
    
        ref_z, ref_ll, ref_var = results[-1]
    
        print("\n" + "=" * 80)
        print("ZRES SUMMARY")
        print("reference zres:", ref_z)
        print("=" * 80)
    
        for zr, ll, var in results:
            print(
                f"zres={zr:5d}  ll={ll: .8f}  "
                f"dll_vs_ref={ll - ref_ll: .8e}  var={var: .8f}"
            )
    
        print("=" * 80)
        return

    ##########################################################################
    ##########################################################################
    # DEBUG 
    ##########################################################################
    ##########################################################################

    if FLAGS.debug:
        print("\n" + "="*80)
        print("Fixed-Lambda likelihood test")
        print("="*80)

        print("pop only is %s"%FLAGS.pop_only)
        print("PE prior is ")
        print(lik_data.log_PE_prior_pe[:10])
        
        # Build Lambda from initial values / priors using the model trace machinery is annoying,
        # so easiest is to use a known Lambda vector saved from PyMC or from your ivals.
        # Replace this with the exact Lambda vector you want to compare.
        
        
        # Lambda_test = jnp.asarray([
        
        #     # --- cosmology (5)
        #     67.9,          # H0
        #     0.3065,        # Om
        #     -1.0,          # w0 (default)
        #     1.0,           # Xi0 (default)
        #     0.0,           # nXi0 (default)
        
        #     # --- rate (3)
        #     3.2,           # gamma
        #     3.0,           # kappa
        #     2.0,           # zp
        
        #     # --- spin (4)
        #     0.024333031991381315,   # muChi
        #     0.31873272890864474,    # sigmaChi
        #     0.2123453198667594,     # zeta
        #     3.0206244922342362,     # sigmat
        
        #     # --- mass (21)
        
        #     # power-law slopes
        #     1.7,           # alpha1
        #     4.5,           # alpha2
        #     36.0,          # mb
        
        #     # Gaussians
        #     9.8,           # mu1
        #     0.65,          # sigma1
        #     33.0,          # mu2
        #     3.9,           # sigma2
        
        #     # low-mass cutoff
        #     2.1,           # m1_low  (default guess)
        #     300.0,         # m_high
        #     4.3,           # delta_m1
        
        #     # mixture weights (lambda vector)
        #     0.36,
        #     0.59,
        #     0.05,
        
        #     # secondary mass
        #     1.2,           # beta
        #     2.0,           # m2_low (default guess)
        #     4.9,           # delta_m2
        
        #     # high-mass Gaussian tail (not provided → safe defaults)
        #     0.01,           # epsilon
        #     60.0,          # m_g
        #     10.0,          # w_g
        #     5.0,           # sig_g_l
        #     5.0,           # sig_g_h
        
        # ], dtype=jnp.float64)

        Lambda_test = jnp.asarray([


            # --- cosmology (5)
            67.9,          # H0
            0.3065,        # Om
            -1.0,          # w0 (default)
            1.0,           # Xi0 (default)
            0.0,           # nXi0 (default)
        
            # --- rate (3)
            3.2,           # gamma
            3.0,           # kappa
            2.0,           # zp
        
            # --- spin (4)
            0.024333031991381315,   # muChi
            0.31873272890864474,    # sigmaChi
            0.2123453198667594,     # zeta
            3.0206244922342362,     # sigmat

            
            # low-z: 21
            1.7, 4.5, 35.0,
            8.5, 0.4, 30.0, 3.5,
            3.189, 150.0, 4.3,
            0.36, 0.59, 0.05,
            1.2, 3.054, 4.9,
            0.1, 45.0, 70.0, 1e-4, 1e-4,
        
            # evo: 26
            1.7, 1.1, 0.5,
            3.5, 1.1, 0.5,
            35.0, 1.1, 0.5,
            15.0, 1.1, 0.5,
            2.5, 1.1, 0.5,
            60.0, 1.1, 0.5,
            15.0, 1.1, 0.5,
            0.10, 0.05, 0.85,
            1.1, 0.5,
        ], dtype=jnp.float64)
        
        print(Lambda_test.shape)  # must be (47,)
    
        ll, var = loglik(Lambda_test)
    
        print("ll =", float(ll))
        print("var_total =", float(var))
    
        if FLAGS.pop_only:
            log_evt, log_mu, log_var_sel_u, var_evs = core(
                lik_data.m1det_pe, lik_data.m2det_pe, lik_data.dL_pe, lik_data.spins_pe,
                lik_data.m1inj, lik_data.m2inj, lik_data.dLinj, lik_data.spins_inj,
                lik_data.log_p_draw, lik_data.log_p_incl,
                Lambda_test, lik_data.Ndraw,
                lik_data.log_PE_prior_pe,
                lik_data.event_id_pe,
                lik_data.Nsamples_evt,
            )
    
            var_sel = jnp.exp(log_var_sel_u + 2.0 * lik_data.logNobs)

            print("log_evt finite:", bool(jnp.all(jnp.isfinite(log_evt))))
            print("log_evt shape:", log_evt.shape)
            print("var_evs raw:", var_evs)
    
            print("sum log_evt =", float(jnp.sum(log_evt)))
            print("log_mu =", float(log_mu))
            print("var_sel =", float(var_sel))
            print("var_evs =", float(var_evs))
            print("var_total reconstructed =", float(var_sel + var_evs))
            print("log_evt first 10 =", np.asarray(log_evt[:10]))
    
        print("="*80)
        return


    ##########################################################################
    ##########################################################################
    # INITIALIZE 
    ##########################################################################
    ##########################################################################
    
    init_vals = {k: v for k, v in ivals.items()}
    rng_key = random.PRNGKey(int(FLAGS.seed))

    if FLAGS.mass_model=='DPLDP' and FLAGS.reparam_mass:

        for drop in ["mb","u","v","mu1","mu2","sigma1","sigma2","delta_m1","delta_m2","alpha1","alpha2", "gamma", "kappa", "zp", "H0", "Om"]:
            init_vals.pop(drop, None)
        
        mb_a, mb_b = priors["mb"][0], priors["mb"][1]
        mb0 = jm.bounded_sigmoid_raw_init(ivals.get("mb"), mb_a, mb_b)
        if mb0 is not None:
            init_vals["mb_raw"] = mb0

        u0 = jm.unit_interval_sigmoid_raw_init(ivals.get("u"))
        if u0 is not None:
            init_vals["u_raw"] = u0
        v0 = jm.unit_interval_sigmoid_raw_init(ivals.get("v"))
        if v0 is not None:
            init_vals["v_raw"] = v0

        for k in ("mu1", "mu2", "gamma", "kappa", "zp", "H0", "Om"):
            pa, pb = priors[k][0], priors[k][1]
            p0 = jm.bounded_sigmoid_raw_init(ivals.get(k), pa, pb)
            if p0 is not None:
                init_vals[k+"_raw"] = p0
        
    


        if "alpha_bar" not in init_vals:
            # prefer ivals["alpha_bar"], else derive from alpha1/alpha2 if present
            if ivals.get("alpha_bar") is not None:
                init_vals["alpha_bar"] = float(ivals["alpha_bar"])
            elif (ivals.get("alpha1") is not None) and (ivals.get("alpha2") is not None):
                init_vals["alpha_bar"] = 0.5 * (float(ivals["alpha1"]) + float(ivals["alpha2"]))
            elif ivals.get("alpha1") is not None:
                init_vals["alpha_bar"] = float(ivals["alpha1"])
            else:
                # fallback to midpoint of prior
                a_low, a_high = priors["alpha1"][0], priors["alpha1"][1]
                init_vals["alpha_bar"] = 0.5 * (a_low + a_high)
    
        if "alpha_diff" not in init_vals:
            # prefer ivals["alpha_diff"], else derive from alpha2-alpha1 if present
            if ivals.get("alpha_diff") is not None:
                init_vals["alpha_diff"] = float(ivals["alpha_diff"])
            elif (ivals.get("alpha1") is not None) and (ivals.get("alpha2") is not None):
                init_vals["alpha_diff"] = float(ivals["alpha2"]) - float(ivals["alpha1"])
            else:
                init_vals["alpha_diff"] = 0.0


        for nm in ("delta_m1", "delta_m2", "sigma1", "sigma2"):
            raw = jm.floored_lognormal_raw_init(ivals, nm, priors)
            if raw is not None:
                init_vals[f"{nm}_raw"] = raw


        # if ivals.get("m_high") is not None:
        #     m_high_ = ivals.get("m_high")
        # else:
        #     m_high_ = (priors["m_high"][1]+priors["m_high"][0])*0.5
        #print("m_high init: %s"%m_high_)

        mhigh_floor = float(priors["m_high"][0])
        m_high0 = float(ivals.get("m_high", 0.5 * (priors["m_high"][0] + priors["m_high"][1])))
        init_vals["delta_mhigh"] = max(1e-6, m_high0 - mhigh_floor)

        # # need an init for m1_low (deterministic from u); use ivals if present, else reconstruct 
        # if ivals.get("m1_low") is not None:
        #     m1_low0 = float(ivals["m1_low"])
        # elif ivals.get("u") is not None:
        #     m1_low0 = 3.0 + (10.0 - 3.0) * float(ivals["u"])**1.5 #np.sqrt(float(ivals["u"]))
        # elif init_vals.get("u_raw") is not None:
        #     u0 = 1.0 / (1.0 + np.exp(-float(init_vals["u_raw"])))
        #     m1_low0 = 3.0 + (10.0 - 3.0) * u0**1.5 #np.sqrt(u0)
        # else:
        #     m1_low0 = 0.5 * (3.0 + 10.0)  # fallback (rough)
        
        # init_vals["delta_mmax"] = max(1e-6, float(m_high_) - m1_low0)

        # IMPORTANT: if you provide init for Dirichlet variable "lambda"
        # it MUST be a JAX array, not a list:
        if "lambda" in init_vals and not isinstance(init_vals["lambda"], jnp.ndarray):
            init_vals["lambda"] = jnp.asarray(init_vals["lambda"], dtype=jnp.float64)


        # --- add spins reparam init values (same style as mass block) ---

        for drop in ["muChi", "sigmaChi", "zeta", "sigmat"]:
            init_vals.pop(drop, None)
        
        # muChi in [a,b] via bounded sigmoid  -> init site is "muChi_raw"
        muChi_a, muChi_b = priors["muChi"][0], priors["muChi"][1]
        muChi0 = jm.bounded_sigmoid_raw_init(ivals.get("muChi"), muChi_a, muChi_b)
        if muChi0 is not None:
            init_vals["muChi_raw"] = muChi0
        
        # zeta in [a,b] via bounded sigmoid  -> init site is "zeta_raw"
        zeta_a, zeta_b = priors["zeta"][0], priors["zeta"][1]
        zeta0 = jm.bounded_sigmoid_raw_init(ivals.get("zeta"), zeta_a, zeta_b)
        if zeta0 is not None:
            init_vals["zeta_raw"] = zeta0
        
        # sigmaChi in [a,b] but sigmoid in log-space -> raw init from log-space fraction
        # if ivals.get("sigmaChi") is not None:
        #     sigmaChi_a, sigmaChi_b = priors["sigmaChi"][0], priors["sigmaChi"][1]
        #     ls_a = np.log(float(sigmaChi_a))
        #     ls_b = np.log(float(sigmaChi_b))
        #     ls0 = np.log(float(ivals["sigmaChi"]))
        #     t = (ls0 - ls_a) / (ls_b - ls_a)
        #     t = np.clip(t, 1e-6, 1.0 - 1e-6)
        #     init_vals["sigmaChi_raw"] = np.log(t / (1.0 - t))

        sigmaChi_a, sigmaChi_b = priors["sigmaChi"][0], priors["sigmaChi"][1]
        sigmaChi0 = jm.bounded_sigmoid_raw_init(ivals.get("sigmaChi"), sigmaChi_a, sigmaChi_b)
        if sigmaChi0 is not None:
            init_vals["sigmaChi_raw"] = sigmaChi0
        
        # sigmat = floor + HalfNormal(raw) -> init site is "sigmat_raw" (nonnegative)
        if ivals.get("sigmat") is not None:
            sigmat_floor = float(priors["sigmat"][0])
            init_vals["sigmat_raw"] = max(0.0, float(ivals["sigmat"]) - sigmat_floor)


    if FLAGS.mass_model == "DPLDP-z" and FLAGS.reparam_mass:

        # Drop deterministic/physical names that are produced from raw/reparam sites
        for drop in [
            "alpha1_0", "alpha2_0", "mb_0",
            "mu1_0", "mu2_0", "sigma1_0", "sigma2_0",
            "u", "v", "m1_low", "m2_low",
            "m_high", "delta_m1", "delta_m2",
            "gamma", "kappa", "zp", "H0", "Om",
            "muChi", "sigmaChi", "zeta", "sigmat",
            "lambda", "lambda0_0", "lambda1_0", "lambda2_0",
            "lambda0_inf", "lambda1_inf", "lambda2_inf",
            "dz_lambda",
        ]:
            init_vals.pop(drop, None)
    
        # -------------------------
        # Cosmology / rate raw inits
        # -------------------------
        for k in ("gamma", "kappa", "zp", "H0", "Om"):
            if k in priors:
                pa, pb = priors[k][0], priors[k][1]
                p0 = jm.bounded_sigmoid_raw_init(ivals.get(k), pa, pb)
                if p0 is not None:
                    init_vals[f"{k}_raw"] = p0
    
        # -------------------------
        # Spin default_gauss raw inits
        # -------------------------
        for k in ("muChi", "sigmaChi", "zeta"):
            if k in priors:
                pa, pb = priors[k][0], priors[k][1]
                p0 = jm.bounded_sigmoid_raw_init(ivals.get(k), pa, pb)
                if p0 is not None:
                    init_vals[f"{k}_raw"] = p0
    
        if ivals.get("sigmat") is not None:
            sigmat_floor = float(priors["sigmat"][0])
            init_vals["sigmat_raw"] = max(0.0, float(ivals["sigmat"]) - sigmat_floor)
    
        # -------------------------
        # Low-z alpha_bar / alpha_diff
        # -------------------------
        if "alpha_bar" not in init_vals:
            if ivals.get("alpha_bar") is not None:
                init_vals["alpha_bar"] = float(ivals["alpha_bar"])
            elif (ivals.get("alpha1_0") is not None) and (ivals.get("alpha2_0") is not None):
                init_vals["alpha_bar"] = 0.5 * (float(ivals["alpha1_0"]) + float(ivals["alpha2_0"]))
            elif ivals.get("alpha1_0") is not None:
                init_vals["alpha_bar"] = float(ivals["alpha1_0"])
            else:
                a_low, a_high = priors["alpha1_0"][0], priors["alpha1_0"][1]
                init_vals["alpha_bar"] = 0.5 * (a_low + a_high)
    
        if "alpha_diff" not in init_vals:
            if ivals.get("alpha_diff") is not None:
                init_vals["alpha_diff"] = float(ivals["alpha_diff"])
            elif (ivals.get("alpha1_0") is not None) and (ivals.get("alpha2_0") is not None):
                init_vals["alpha_diff"] = float(ivals["alpha2_0"]) - float(ivals["alpha1_0"])
            else:
                init_vals["alpha_diff"] = 0.0
    
        # -------------------------
        # Low-z bounded sigmoid raw inits
        # -------------------------
        for k in ("mb_0", "mu1_0", "mu2_0"):
            if k in priors:
                pa, pb = priors[k][0], priors[k][1]
                p0 = jm.bounded_sigmoid_raw_init(ivals.get(k), pa, pb)
                if p0 is not None:
                    init_vals[f"{k}_raw"] = p0
    
        # -------------------------
        # u, v triangle raw inits
        # -------------------------
        u0 = jm.unit_interval_sigmoid_raw_init(ivals.get("u"))
        if u0 is not None:
            init_vals["u_raw"] = u0
    
        v0 = jm.unit_interval_sigmoid_raw_init(ivals.get("v"))
        if v0 is not None:
            init_vals["v_raw"] = v0
    
        # -------------------------
        # Floored lognormal raw inits
        # -------------------------
        # These names must match the NumPyro sample sites:
        # sigma1_0_raw, sigma2_0_raw, delta_m1_raw, delta_m2_raw
        for nm in ("sigma1_0", "sigma2_0", "delta_m1", "delta_m2"):
            if ivals.get(nm) is not None:
                floor = float(priors[nm][0])
                init_vals[f"{nm}_raw"] = max(1e-12, float(ivals[nm]) - floor)
    
        # -------------------------
        # m_high shifted LogNormal init
        # -------------------------
        mhigh_floor = float(priors["m_high"][0])
        m_high0 = float(ivals.get("m_high", 0.5 * (priors["m_high"][0] + priors["m_high"][1])))
        init_vals["delta_mhigh"] = max(1e-6, m_high0 - mhigh_floor)
    
        # -------------------------
        # Low-z lambda Dirichlet init
        # -------------------------
        if ivals.get("lambda") is not None:
            init_vals["lambda0_vec"] = jnp.asarray(ivals["lambda"], dtype=jnp.float64)
    
        # -------------------------
        # High-z lambda Dirichlet init
        # -------------------------
        if ivals.get("lambda_inf_vec") is not None:
            init_vals["lambda_inf_vec"] = jnp.asarray(ivals["lambda_inf_vec"], dtype=jnp.float64)
        else:
            init_vals["lambda_inf_vec"] = jnp.asarray([0.10, 0.05, 0.85], dtype=jnp.float64)
    
        # -------------------------
        # Evolution parameters
        # -------------------------
        # For evo_triplet_numpyro as I wrote it:
        # theta_inf = theta0 + delta_name
        # so initialize delta_name = theta_inf_init - theta0_init when available.
        def _init_delta(name, theta0_key, positive=False, eps_pos=1e-6):
            inf_key = f"{name}_inf"
            delta_key = f"delta_{name}"
        
            if ivals.get(inf_key) is not None and ivals.get(theta0_key) is not None:
                delta = float(ivals[inf_key]) - float(ivals[theta0_key])
            elif ivals.get(delta_key) is not None:
                delta = float(ivals[delta_key])
            else:
                return
        
            if positive:
                theta0 = float(ivals[theta0_key])
                lower = -theta0 + eps_pos
                delta = max(delta, lower + 1e-3)
        
            init_vals[delta_key] = delta   

            
        # --- delta (theta_inf - theta0) init ---
        _init_delta("alpha1", "alpha1_0")
        _init_delta("alpha2", "alpha2_0")
        
        if FLAGS.vary_mb:
            _init_delta("mb", "mb_0")
        
        _init_delta("mu1", "mu1_0")
        _init_delta("sigma1", "sigma1_0")
        _init_delta("mu2", "mu2_0")
        _init_delta("sigma2", "sigma2_0")
        
        
        # --- z transition init ---
        for nm in ("alpha1", "alpha2", "mu1", "sigma1", "mu2", "sigma2"):
            zkey = f"z_{nm}"
            if ivals.get(zkey) is not None:
                init_vals[zkey] = float(ivals[zkey])
        
        if FLAGS.vary_mb:
            if ivals.get("z_mb") is not None:
                init_vals["z_mb"] = float(ivals["z_mb"])
        
        
        # --- log_dz init ---
        for nm in ("alpha1", "alpha2", "mu1", "sigma1", "mu2", "sigma2"):
            logkey = f"log_dz_{nm}"
            dzkey = f"dz_{nm}"
            if ivals.get(logkey) is not None:
                init_vals[logkey] = float(ivals[logkey])
            elif ivals.get(dzkey) is not None:
                init_vals[logkey] = float(np.log(ivals[dzkey]))
        
        if FLAGS.vary_mb:
            if ivals.get("log_dz_mb") is not None:
                init_vals["log_dz_mb"] = float(ivals["log_dz_mb"])
            elif ivals.get("dz_mb") is not None:
                init_vals["log_dz_mb"] = float(np.log(ivals["dz_mb"]))
    
        # log dz sites for evo_triplet_numpyro
        for nm in ("alpha1", "alpha2", "mb", "mu1", "sigma1", "mu2", "sigma2"):
            logkey = f"log_dz_{nm}"
            dzkey = f"dz_{nm}"
            if ivals.get(logkey) is not None:
                init_vals[logkey] = float(ivals[logkey])
            elif ivals.get(dzkey) is not None:
                init_vals[logkey] = float(np.log(ivals[dzkey]))
    
        # Lambda-mixture transition
        if ivals.get("z_lambda") is not None:
            init_vals["z_lambda"] = float(ivals["z_lambda"])
    
        if ivals.get("log_dz_lambda") is not None:
            init_vals["log_dz_lambda"] = float(ivals["log_dz_lambda"])
        elif ivals.get("dz_lambda") is not None:
            init_vals["log_dz_lambda"] = float(np.log(ivals["dz_lambda"]))


    # convert everything to jnp arrays (no python lists)
    # init_vals = { k: jnp.asarray(v) + FLAGS.eps_init * jnp.asarray(np.random.normal())
    # for k, v in init_vals.items()
    #             }
    keys = jax.random.split(rng_key, len(init_vals))
    init_vals = {
    k: jnp.asarray(v)    for (k, v), kk in zip(init_vals.items(), keys)
        }

    init_vals["delta_mhigh"] = 80.0   # not 150
    init_vals["lambda_inf_vec"] = jnp.asarray([0.2, 0.2, 0.6], dtype=jnp.float64)

    if not FLAGS.pop_only:
        # ensure x init exists and is small-ish (with eps_init )
        N  = int(lik_data.Nobs)
        nd = int(lik_data.mus_s.shape[1])
        if "x" not in init_vals:
            # deterministic init is fine; or draw with a fixed seed once outside
            init_vals["x"] = jnp.asarray( FLAGS.eps_init*np.random.normal(loc=0.0, scale=1.0, size=(N, nd)) , dtype=jnp.float64)
            


    # print("\n" + "="*80)
    # print("INIT VALS PASSED TO NUMPYRO")
    # print("="*80)
    # for k in sorted(init_vals.keys()):
    #     v = np.asarray(init_vals[k])
    #     print(f"{k:25s} shape={v.shape} value={v}")
    # print("="*80)
    
    # --- Initialize (lets you plug in mass matrix if you have one) ---
    # Try init_to_value first; if it fails, fallback to feasible.
    
    init_strategy = init_to_value(values=init_vals)
    fallback_init_strategy = init_to_feasible()

    try:
        init_key, rng_key = random.split(rng_key)
        res = initialize_model(
            init_key,
            model_numpyro,
            init_strategy=init_strategy,
            dynamic_args=False,
            #validate_grad=False,   # key
        )
        # z = res.param_info.z
        # pe = res.potential_fn(z)
        # g = jax.grad(res.potential_fn)(z)

        # print("PE:", pe, "finite:", jnp.isfinite(pe))

        # for k, v in g.items():
        #     print(k, "grad finite:", bool(jnp.all(jnp.isfinite(v))), "grad:", v)

   
        print("✅ init_to_value; initial values used.")
        
    except Exception as e:
        
        print(e)
        print("⚠️ init_to_value failed; falling back to init_to_feasible.")
        print("   error:", repr(e))

        print("init_to_value failed:", repr(e))
    
        # Run model trace at init values to inspect sites
        from numpyro.handlers import seed, trace, substitute
        from numpyro.util import format_shapes
    
        tr = trace(
            seed(
                substitute(model_numpyro, data=init_vals),
                init_key,
            )
        ).get_trace()
    
        print(format_shapes(tr))
    
        for name, site in tr.items():
            if site["type"] == "sample":
                value = site["value"]
                fn = site["fn"]
                lp = fn.log_prob(value)
    
                print("\nSITE:", name)
                print("value:", value)
                print("support:", fn.support)
                print("log_prob:", lp)
    
                if not jnp.all(jnp.isfinite(lp)):
                    print("❌ BAD SITE:", name)
    
        raise

        
        init_key, rng_key = random.split(rng_key)
        res = initialize_model(
            init_key,
            model_numpyro,
            init_strategy=fallback_init_strategy,
            dynamic_args=False,
        )
        

    # --- Robust unpack of initialize_model across numpyro versions ---
    potential_fn = None
    postprocess_fn = None
    model_trace = None
    init_state = None
    
    if isinstance(res, tuple):
        if len(res) == 4:
            # numpyro >= ~0.13: (init_params, potential_fn, postprocess_fn, model_trace)
            init_params, potential_fn, postprocess_fn, model_trace = res
        elif len(res) == 3:
            # some older / alternate paths: try to infer meaning
            # common patterns seen: (init_params, potential_fn, model_trace)
            a, b, c = res
            # Heuristic: potential_fn is callable
            if callable(b):
                init_params, potential_fn, model_trace = a, b, c
            else:
                # if we cannot identify potential_fn, fail loudly
                raise RuntimeError(
                    f"initialize_model returned 3-tuple but second element is not callable. "
                    f"Types: {[type(x) for x in res]}"
                )
        else:
            raise RuntimeError(f"Unexpected initialize_model return length: {len(res)}")
    else:
        raise RuntimeError(f"Unexpected initialize_model return type: {type(res)}")
    
    if potential_fn is None or (not callable(potential_fn)):
        raise RuntimeError("potential_fn was not produced by initialize_model; cannot time grad/potential.")
  





    # ---- what the model is sampling (from trace) ----
    sample_sites = [
        name for name, site in model_trace.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    ]
    det_sites = [
        name for name, site in model_trace.items()
        if site["type"] == "deterministic"
    ]
    
    print("[SAMPLED sites]")
    for n in sample_sites:
        fn = model_trace[n]["fn"]
        # best-effort shape reporting
        try:
            sh = tuple(fn.batch_shape) + tuple(fn.event_shape)
        except Exception:
            sh = None
        print(f"  - {n:20s}  dist={type(fn).__name__:<20s}  shape={sh}")
    
    print("[DETERMINISTIC sites]")
    for n in det_sites:
        val = model_trace[n]["value"]
        print(f"  - {n:20s}  shape={tuple(np.shape(val))}")


    # Lambda sites = all sampled sites except the big latent x
    lambda_sites = [n for n in sample_sites if n != "x"]
    
    # Safety: if your model ever has other huge latents, exclude them here too:
    # lambda_sites = [n for n in lambda_sites if n not in ("x", "something_else_big")]
    
    # NumPyro expects a list of tuples (each tuple is one dense block)
    dense_blocks = [tuple(lambda_sites)] if len(lambda_sites) > 1 else False
    print("[INFO] dense_mass blocks:", dense_blocks)

    print()

    if FLAGS.check_init:
        
        print()
        print('Checking initial point...')
        print()

        #raise NotImplementedError()

        print("⚠️ Not yet available.")

    ##########################################################################
    ##########################################################################
    # Profile if requested 
    ##########################################################################
    ##########################################################################

    def _lambda_slices_for_benchmark(rate_model, spin_model, mass_model):
        i = 0
        cosmo = slice(i, i + 5); i += 5
    
        if rate_model in ("MD", "DPUC-vol-MD"):
            rate = slice(i, i + 3); i += 3
        elif rate_model == "PL":
            rate = slice(i, i + 1); i += 1
        elif rate_model in ("DPUC", "DPUC-vol"):
            rate = slice(i, i)
        else:
            raise ValueError(f"Unknown rate_model={rate_model}")
    
        if spin_model == "chieffchip":
            spin = slice(i, i + 5); i += 5
        elif spin_model == "chieffchip_uc":
            spin = slice(i, i + 4); i += 4
        elif spin_model in ("default", "default_gauss"):
            spin = slice(i, i + 4); i += 4
        else:
            spin = slice(i, i)
    
        if mass_model == "PLPreg":
            mass = slice(i, i + 8); i += 8
        elif mass_model in ("DPLDP", "PLDP"):
            mass = slice(i, i + 21); i += 21
        elif mass_model == "DPLDP-z":
            mass = slice(i, i + 47); i += 47
        else:
            raise ValueError(f"Unknown mass_model={mass_model}")
    
        return {"cosmo": cosmo, "rate": rate, "spin": spin, "mass": mass, "npar": i}
    
    
    def _trace_value(model_trace, name, default=None):
        if name in model_trace:
            return model_trace[name]["value"]
        if default is not None:
            return default
        raise KeyError(f"Could not find {name!r} in model_trace and no default was given")
    
    
    def _build_lambda0_from_trace_for_benchmark(FLAGS, priors, params_fix, model_trace):
        import jax.numpy as jnp
    
        vals = []
    
        # Cosmology: [H0, Om, w0, Xi0, nXi0]
        vals.append(jnp.asarray(params_fix["H0"] if FLAGS.fix_H0 else _trace_value(model_trace, "H0"), dtype=jnp.float64))
        vals.append(jnp.asarray(params_fix["Om"] if FLAGS.fix_Om else _trace_value(model_trace, "Om"), dtype=jnp.float64))
        vals.append(jnp.asarray(-1.0 if FLAGS.fix_w0 else _trace_value(model_trace, "w0"), dtype=jnp.float64))
        if FLAGS.fix_Xi0n:
            vals += [jnp.asarray(1.0, dtype=jnp.float64), jnp.asarray(0.0, dtype=jnp.float64)]
        else:
            vals += [jnp.asarray(_trace_value(model_trace, "Xi0"), dtype=jnp.float64),
                     jnp.asarray(_trace_value(model_trace, "nXi0"), dtype=jnp.float64)]
    
        # Rate
        if FLAGS.rate_model in ("MD", "DPUC-vol-MD"):
            vals += [jnp.asarray(_trace_value(model_trace, "gamma"), dtype=jnp.float64),
                     jnp.asarray(_trace_value(model_trace, "kappa"), dtype=jnp.float64),
                     jnp.asarray(_trace_value(model_trace, "zp"), dtype=jnp.float64)]
        elif FLAGS.rate_model == "PL":
            vals += [jnp.asarray(_trace_value(model_trace, "gamma"), dtype=jnp.float64)]
        elif FLAGS.rate_model in ("DPUC", "DPUC-vol"):
            pass
        else:
            raise ValueError(f"Unknown rate_model={FLAGS.rate_model}")
    
        # Spin
        if FLAGS.spin_model == "default_gauss":
            vals += [jnp.asarray(_trace_value(model_trace, "muChi"), dtype=jnp.float64),
                     jnp.asarray(_trace_value(model_trace, "sigmaChi"), dtype=jnp.float64),
                     jnp.asarray(_trace_value(model_trace, "zeta"), dtype=jnp.float64),
                     jnp.asarray(_trace_value(model_trace, "sigmat"), dtype=jnp.float64)]
        elif FLAGS.spin_model == "none":
            pass
        else:
            raise NotImplementedError(f"Add spin packing for spin_model={FLAGS.spin_model}")
    
        # Mass. This mirrors the DPLDP/PLDP packing in jax_models.py.
        if FLAGS.mass_model in ("DPLDP", "PLDP"):
            vals += [
                _trace_value(model_trace, "alpha1"),
                _trace_value(model_trace, "alpha2"),
                _trace_value(model_trace, "mb"),
                _trace_value(model_trace, "mu1"),
                _trace_value(model_trace, "sigma1"),
                _trace_value(model_trace, "mu2"),
                _trace_value(model_trace, "sigma2"),
                _trace_value(model_trace, "m1_low"),
                _trace_value(model_trace, "m_high"),
                _trace_value(model_trace, "delta_m1"),
                _trace_value(model_trace, "lambda0"),
                _trace_value(model_trace, "lambda1"),
                _trace_value(model_trace, "lambda2"),
                _trace_value(model_trace, "beta"),
                _trace_value(model_trace, "m2_low"),
                _trace_value(model_trace, "delta_m2"),
                _trace_value(model_trace, "epsilon", jnp.asarray(0.1, dtype=jnp.float64)),
                _trace_value(model_trace, "m_g", jnp.asarray(45.0, dtype=jnp.float64)),
                _trace_value(model_trace, "w_g", jnp.asarray(70.0, dtype=jnp.float64)),
                _trace_value(model_trace, "sig_g_l", jnp.asarray(1e-2, dtype=jnp.float64)),
                _trace_value(model_trace, "sig_g_h", jnp.asarray(1e-2, dtype=jnp.float64)),
            ]
        elif FLAGS.mass_model == "PLPreg":
            vals += [
                _trace_value(model_trace, "lambdaPeak"), _trace_value(model_trace, "alpha"),
                _trace_value(model_trace, "beta"), _trace_value(model_trace, "deltam"),
                _trace_value(model_trace, "ml"), _trace_value(model_trace, "mh"),
                _trace_value(model_trace, "muMass"), _trace_value(model_trace, "sigmaMass"),
            ]
        elif FLAGS.mass_model == "DPLDP-z":
            raise NotImplementedError(
                "Benchmark Lambda packing for DPLDP-z needs the exact 47-name order from your active jax_models.py. "
                "Do not guess it; add that list explicitly before using this benchmark."
            )
        else:
            raise ValueError(f"Unknown mass_model={FLAGS.mass_model}")
    
        return jnp.stack([jnp.asarray(v, dtype=jnp.float64).reshape(()) for v in vals])
    
    
    def run_selection_block_benchmark(FLAGS, priors, params_fix, model_trace, lik_data, core, *, repeats=10):
        import time
        import jax
        import jax.numpy as jnp
        import numpy as np
        from likelihood import _gw_terms_from_x
    
        Lambda0 = _build_lambda0_from_trace_for_benchmark(FLAGS, priors, params_fix, model_trace)
        slices = _lambda_slices_for_benchmark(FLAGS.rate_model, FLAGS.spin_model, FLAGS.mass_model)
    
        if Lambda0.shape[0] != slices["npar"]:
            raise RuntimeError(f"Lambda length mismatch: built {Lambda0.shape[0]}, expected {slices['npar']}")
    
        x0 = _trace_value(model_trace, "x", None)
        if x0 is None:
            x0 = init_vals.get("x", None)
        if x0 is None:
            raise RuntimeError("Cannot find initial x in model_trace or init_vals")
        x0 = jnp.asarray(x0, dtype=jnp.float64)
    
        # Precompute event coordinates once. The selection term does not depend on x, but core() needs event args.
        m1det, m2det, dLdet, spins_evt, _, _ = _gw_terms_from_x(x0, lik_data)
    
        def selection_only(Lam):
            _, log_mu, _ = core(
                m1det, m2det, dLdet, spins_evt,
                lik_data.m1inj, lik_data.m2inj, lik_data.dLinj,
                lik_data.spins_inj, lik_data.log_p_draw, lik_data.log_p_incl,
                Lam, lik_data.Ndraw,
            )
            return log_mu
    
        full_grad = jax.jit(jax.value_and_grad(selection_only))
    
        def make_block_grad(sl):
            def f(block):
                Lam = Lambda0.at[sl].set(block)
                return selection_only(Lam)
            return jax.jit(jax.value_and_grad(f))
    
        block_grads = {}
        for name in ("cosmo", "rate", "spin", "mass"):
            sl = slices[name]
            if sl.start != sl.stop:
                block_grads[name] = (sl, make_block_grad(sl))
    
        def time_call(label, fn, arg):
            # compile/warmup
            y, g = fn(arg)
            jax.block_until_ready(y)
            jax.block_until_ready(g)
    
            ts = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                y, g = fn(arg)
                jax.block_until_ready(y)
                jax.block_until_ready(g)
                ts.append(time.perf_counter() - t0)
            arr = np.asarray(ts)
            print(f"{label:>12s}: mean={arr.mean():.6f}s  min={arr.min():.6f}s  max={arr.max():.6f}s  grad_shape={tuple(g.shape)}  value={float(y):.6g}")
            return float(arr.mean())
    
        print("\n" + "=" * 80)
        print("Selection-gradient block benchmark")
        print("This is a proxy benchmark: block functions still call the existing full core.")
        print("If block timings are not clearly below full timing, a structural refactor is unlikely to pay off.")
        print("=" * 80)
        print("Lambda length:", Lambda0.shape[0])
        print("Slices:", {k: (v.start, v.stop) for k, v in slices.items() if k != "npar"})
        print("Ninj:", int(lik_data.m1inj.shape[0]), "chunk_inj:", int(FLAGS.chunk_inj))
        print("repeats:", repeats)
        print("-" * 80)
    
        times = {}
        times["full"] = time_call("full", full_grad, Lambda0)
        for name, (sl, fn) in block_grads.items():
            times[name] = time_call(name, fn, Lambda0[sl])
    
        sum_blocks = sum(v for k, v in times.items() if k != "full")
        print("-" * 80)
        print(f"sum(blocks) / full = {sum_blocks / times['full']:.3f}")
        print("Interpretation:")
        print("  <~0.7 : structural decomposition may be worth it")
        print("  ~1.0  : likely little speed gain")
        print("  >1.0  : multiple VJPs probably worse unless memory improves")
        print("=" * 80 + "\n")
        return times

    
    if FLAGS.profile:
        run_selection_block_benchmark(
            FLAGS, priors, params_fix, model_trace, lik_data, core,
            repeats=max(3, int(FLAGS.profile)),
        )
        return

    ##########################################################################
    ##########################################################################
    # SAMPLE 
    ##########################################################################
    ##########################################################################

    
    # --- NUTS kernel config ---
    nuts = NUTS(
        model_numpyro,
        target_accept_prob=float(FLAGS.target_accept),
        max_tree_depth=int(FLAGS.max_tree_depth),
        dense_mass = dense_blocks,  # dense within each chain
        step_size = 1e-2,              # <-- seed a not-too-small epsilon
        adapt_step_size=True,        # keep adaptation ON
        adapt_mass_matrix=True,
        find_heuristic_step_size = bool(FLAGS.find_heuristic_step_size),
        regularize_mass_matrix = float(FLAGS.regularize_mass_matrix),
        forward_mode_differentiation = False,
        
    )
    


    print()
    print('*'*80)
    print('Sampling with numpyro with %s method...' %(FLAGS.chain_method))
    print('*'*80)
    print()


    
    num_warmup = int(FLAGS.ntune)
    num_samples = int(FLAGS.nsteps)
    num_chains = int(FLAGS.nchains)
    
    # Chain execution mode
    # - parallel: uses multiple devices (requires numpyro.set_host_device_count earlier)
    # - sequential: same process, chains one-by-one
    # - vectorized: vmap chains in one JAX program (often fastest if memory allows)
    if FLAGS.chain_method == "parallel":
        chain_method = "parallel"
    elif FLAGS.chain_method == "vectorized":
        chain_method = "vectorized"
    elif FLAGS.chain_method == "sequential":
        chain_method = "sequential"
    else:
        raise ValueError("chain_method must be parallel, vectorized, or sequential")




    mcmc = MCMC(
        nuts,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=True,
    )
    
    run_key, rng_key = random.split(rng_key)
    mcmc.run(run_key,
                extra_fields=(
        "diverging",
        "num_steps",
        "accept_prob",
        "adapt_state.step_size",
    ),
            )
    
    #mcmc.print_summary()
    samples = mcmc.get_samples(group_by_chain=True)

    ################################################
    # Print diagnostics
    ################################################

    extra = mcmc.get_extra_fields()
    
    print("Available extra fields:", list(extra.keys()))
    
    num_steps = extra["num_steps"]
    acc = extra["accept_prob"]
    div = extra["diverging"]
    
    print("\n" + "="*80)
    print("NUTS diagnostics")
    print("="*80)
    
    print(f"num_steps: mean={num_steps.mean():.2f}  max={num_steps.max()}  min={num_steps.min()}")
    print(f"accept_prob: mean={acc.mean():.3f}  min={acc.min():.3f}")
    print(f"divergences: {div.sum()} / {div.size}")
    if "adapt_state.step_size" in extra:
        eps = extra["adapt_state.step_size"]
        print(f"step_size: final={np.ravel(np.asarray(eps))[-1]:.4g}")
    
    # approximate tree depth from number of leapfrog steps
    tree_depth_approx = np.ceil(np.log2(np.asarray(num_steps) + 1)).astype(int)
    
    unique, counts = np.unique(tree_depth_approx, return_counts=True)
    print("\nApprox tree depth histogram:")
    for u, c in zip(unique, counts):
        print(f"  depth {int(u):2d}: {int(c)}")

    
    
    print("="*80)    

    ################################################
    # Save and exit
    ################################################

    
    print( "\nDone." )
    print("\nSaving trace...")
    try:
        idata = az.from_numpyro(mcmc)
        
        # find next available trace_i.nc
        existing = []
        for fn in os.listdir(FLAGS.fout):
            if fn.startswith("trace_") and fn.endswith(".nc"):
                try:
                    i = int(fn[len("trace_"):-len(".nc")])
                    existing.append(i)
                except ValueError:
                    pass
        
        chain_id = 0 if len(existing) == 0 else max(existing) + 1
        
        tout_i = os.path.join(FLAGS.fout, f"trace_{chain_id}.nc")
        az.to_netcdf(idata, tout_i)
        
        np.savez(
            os.path.join(FLAGS.fout, f"trace_{chain_id}.npz"),
            **{k: np.asarray(v) for k, v in samples.items()}
        )
        
        print(f"✅ Single-chain trace saved in {tout_i}")
        
        # concatenate all trace_i.nc into trace.nc
        trace_files = []
        for fn in os.listdir(FLAGS.fout):
            if fn.startswith("trace_") and fn.endswith(".nc"):
                try:
                    i = int(fn[len("trace_"):-len(".nc")])
                    trace_files.append((i, os.path.join(FLAGS.fout, fn)))
                except ValueError:
                    pass
        
        trace_files = [p for _, p in sorted(trace_files)]
        
        if len(trace_files) > 0:
            idatas = [az.from_netcdf(p) for p in trace_files]
            idata = az.concat(idatas, dim="chain")
        
            tout = os.path.join(FLAGS.fout, "trace.nc")
            az.to_netcdf(idata, tout)
        
            print(f"✅ Concatenated trace saved in {tout}")
            print(f"✅ Number of chains concatenated: {len(trace_files)}")
    except Exception as e:
        print(e)
        print("⚠️ Saving failed !")


    
    ################################################
    # Plot
    ################################################


    print("\nMaking summary plots...")

    
    import matplotlib.pyplot as plt

    vplot = dense_blocks[0]

    try:
        print("Plotting trace...")
        az.plot_trace(idata, var_names = vplot, );
        plt.savefig( os.path.join(FLAGS.fout, 'trace.pdf'), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(e)
        print('No trace plot produced')

    try:
        import corner
        print("Plotting corner...")
        _ = corner.corner(
            idata,
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
    except Exception as e:
        print(e)
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
    
    
