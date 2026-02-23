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
    parser.add_argument("--sel_smoothing", default='none', type=str, required=False)
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
    
    
    
    parser.add_argument("--fix_H0", default=1, type=int, required=False)
    parser.add_argument("--fix_Om", default=1, type=int, required=False)
    parser.add_argument("--fix_w0", default=1, type=int, required=False)
    parser.add_argument("--fix_Xi0n", default=1, type=int, required=False)
    parser.add_argument("--z_pivot", default=0, type=float, required=False)
    parser.add_argument("--integrate_dc", default='pade', type=str, required=False)
    
    
    parser.add_argument("--param", default='vanilla', type=str, required=False)
    parser.add_argument("--pade", default=0, type=int, required=False)
    parser.add_argument("--zres", default=1000, type=int, required=False)
    parser.add_argument("--z_grid_mode", default='piecewise_linear', type=str, required=False)
    
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


    model_numpyro, lik_data, core, loglik = make_model_jax(
        priors=priors,
        GWData=GWData,
        InjData=InjData,
        ivals=ivals,
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
        chunk_reduce=bool(FLAGS.chunk_reduce),
        use_float32=bool(FLAGS.use_float32),
        use_float32_bias=bool(FLAGS.use_float32_bias),
        sel_method=FLAGS.sel,
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
        reparam_mass = bool(FLAGS.reparam_mass)
    )


    ##########################################################################
    ##########################################################################
    # INITIALIZE 
    ##########################################################################
    ##########################################################################
    
    init_vals = {k: v for k, v in ivals.items()}
    rng_key = random.PRNGKey(0)

    if FLAGS.mass_model=='DPLDP' and FLAGS.reparam_mass:

        for drop in ["mb","u","v","mu1","mu2","sigma1","sigma2","delta_m1","delta_m2","alpha1","alpha2"]:
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

        mu1a, mu1b = priors["mu1"][0], priors["mu1"][1]
        mu10 = jm.bounded_sigmoid_raw_init(ivals.get("mu1"), mu1a, mu1b)
        if mu10 is not None:
            init_vals["mu1_raw"] = mu10
        
        mu2a, mu2b = priors["mu2"][0], priors["mu2"][1]
        mu20 = jm.bounded_sigmoid_raw_init(ivals.get("mu2"), mu2a, mu2b)
        if mu20 is not None:
            init_vals["mu2_raw"] = mu20


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

        # IMPORTANT: if you provide init for Dirichlet variable "lambda"
        # it MUST be a JAX array, not a list:
        if "lambda" in init_vals and not isinstance(init_vals["lambda"], jnp.ndarray):
            init_vals["lambda"] = jnp.asarray(init_vals["lambda"], dtype=jnp.float64)



    # convert everything to jnp arrays (no python lists)
    # init_vals = { k: jnp.asarray(v) + FLAGS.eps_init * jnp.asarray(np.random.normal())
    # for k, v in init_vals.items()
    #             }
    keys = jax.random.split(rng_key, len(init_vals))
    init_vals = {
    k: jnp.asarray(v)    for (k, v), kk in zip(init_vals.items(), keys)
        }
    
    # ensure x init exists and is small-ish (with eps_init )
    N  = int(lik_data.Nobs)
    nd = int(lik_data.mus_s.shape[1])
    if "x" not in init_vals:
        # deterministic init is fine; or draw with a fixed seed once outside
        init_vals["x"] = jnp.asarray( FLAGS.eps_init*np.random.normal(loc=0.0, scale=1.0, size=(N, nd)) , dtype=jnp.float64)
        

    
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
            dynamic_args=False
        )
        print("✅ init_to_value; initial values used.")
    except Exception as e:
        print("⚠️ init_to_value failed; falling back to init_to_feasible.")
        print("   error:", repr(e))
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
    # SAMPLE 
    ##########################################################################
    ##########################################################################

    
    # --- NUTS kernel config ---
    nuts = NUTS(
        model_numpyro,
        target_accept_prob=float(FLAGS.target_accept),
        max_tree_depth=int(FLAGS.max_tree_depth),
        dense_mass = dense_blocks,  # dense within each chain
        step_size=1e-2,              # <-- seed a not-too-small epsilon
        adapt_step_size=True,        # keep adaptation ON
        adapt_mass_matrix=True,
        find_heuristic_step_size = False,
        regularize_mass_matrix = 1e-03,
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
    mcmc.run(run_key)
    
    #mcmc.print_summary()
    samples = mcmc.get_samples(group_by_chain=True)

    

    ################################################
    # Save and exit
    ################################################

    
    print( "\nDone." )
    print("\nSaving trace...")
    try:
        idata = az.from_numpyro(mcmc)
        tout = os.path.join(FLAGS.fout, "trace.nc")
        az.to_netcdf(idata, tout)
    
        # also save raw samples in npz
        np.savez(os.path.join(FLAGS.fout, "trace.npz"), **{k: np.asarray(v) for k, v in samples.items()})

        print("✅ Trace saved in %s"%tout)
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
    
    
