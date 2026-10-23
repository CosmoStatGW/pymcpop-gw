#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

# --- set env vars BEFORE importing jax (propagates to spawned workers) ---
import os, argparse, sys
import logging

_logger = logging.getLogger(__name__)
from settings_io import build_parser, load_settings_file, parse_args_with_optional_settings, settings_to_argv



def _lambda_slices_for_benchmark(FLAGS):
    rate_model, spin_model, mass_model = FLAGS.rate_model, FLAGS.spin_model, FLAGS.mass_model
    i = 0
    cosmo = slice(i, i + 5); i += 5

    rate = slice(i, i + 3); i += 3

    if spin_model in ('default_gauss',):
        spin = slice(i, i + 4); i += 4
    else:
        spin = slice(i, i)

    if mass_model == "PLPreg":
        mass = slice(i, i + 8)
        i += 8
    elif mass_model in ('DPLDP',):
        mass = slice(i, i + 21)
        i += 21
    elif mass_model == "DPLDP-z":
        mass = slice(i, i + 47)
        i += 47
    elif mass_model == "DPUC_Primary":
        K = int(FLAGS.N_DP_comp_max)
        D = 1
        W = 1
        n_scalar_after_gmm = 8 + (4 if str(getattr(FLAGS, "DP_m1_env_form", "powerlaw")) == "smooth_broken_pl_2breaks" else 0)
        mass = slice(i, i + 2 * D * K + W * K + n_scalar_after_gmm)
        i += 2 * D * K + W * K + n_scalar_after_gmm

    elif mass_model == "DPUC_TruncM2":
        K = int(FLAGS.N_DP_comp_max)
        D = 2
        W = 1 if bool(getattr(FLAGS, "DPUC_TruncM2_shared_weights", 0)) else 2
        n_scalar_after_gmm = 8 + (4 if str(getattr(FLAGS, "DP_m1_env_form", "powerlaw")) == "smooth_broken_pl_2breaks" else 0)
        mass = slice(i, i + 2 * D * K + W * K + K + n_scalar_after_gmm)
        i += 2 * D * K + W * K + K + n_scalar_after_gmm


    elif mass_model == "DPUC":
        K = int(FLAGS.N_DP_comp_max)
        n_scalar_after_gmm = 8 + (4 if FLAGS.DP_m1_env_form == "smooth_broken_pl_2breaks" else 0)
        mass = slice(i, i + 5 * K + n_scalar_after_gmm)
        i += 5 * K + n_scalar_after_gmm
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
    """Use the model's canonical packing rather than reconstructing a second layout."""
    return _trace_value(model_trace, "Lambda")


def run_selection_block_benchmark(FLAGS, priors, params_fix, model_trace, lik_data, core, *, repeats=10, gp_state_factory=None):
    import time
    import jax
    import jax.numpy as jnp
    import numpy as np
    from likelihood import _gw_terms_from_x

    Lambda0 = _build_lambda0_from_trace_for_benchmark(FLAGS, priors, params_fix, model_trace)
    slices = _lambda_slices_for_benchmark(FLAGS)

    if Lambda0.shape[0] != slices["npar"]:
        raise RuntimeError(f"Lambda length mismatch: built {Lambda0.shape[0]}, expected {slices['npar']}")

    if FLAGS.pop_only:
        m1det, m2det, dLdet, spins_evt = (lik_data.m1det_pe, lik_data.m2det_pe,
                                        lik_data.dL_pe, lik_data.spins_pe)
    else:
        x0 = jnp.asarray(_trace_value(model_trace, "x"), dtype=jnp.float64)
        m1det, m2det, dLdet, spins_evt, _, _ = _gw_terms_from_x(x0, lik_data)

    def selection_only(Lam):
        _, log_mu, _ = core(
            m1det, m2det, dLdet, spins_evt,
            lik_data.m1inj, lik_data.m2inj, lik_data.dLinj,
            lik_data.spins_inj, lik_data.log_p_draw, lik_data.log_p_incl,
            Lam, lik_data.Ndraw,
            gp_state=(gp_state_factory(Lam) if gp_state_factory else None),
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






def early_parse(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--nth", type=int, default=None)
    p.add_argument("--settings", type=str, default=None)
    # ignore everything else
    args, _ = p.parse_known_args(argv)

    # --nth controls environment variables before imports, so it must also be
    # readable from the settings file during this early parse.  A command-line
    # --nth, when provided, takes precedence.
    if args.nth is None and args.settings is not None:
        settings = load_settings_file(args.settings)
        nth = settings.get("nth")
        if nth is not None:
            args.nth = int(str(nth).strip())

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
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
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

    
    parser = build_parser()
    FLAGS = parse_args_with_optional_settings(parser)

    if FLAGS.mass_model == "DPUC_Primary":
        FLAGS.DP_q_env = 1
        if FLAGS.DP_corr_m1m2:
            raise ValueError("DPUC_Primary does not use DP_corr_m1m2.")

    if str(FLAGS.mass_model) == "DPUC_Primary" and not bool(FLAGS.DP_q_env):
        raise ValueError(
            "mass_model='DPUC_Primary' implements the logm1 + q^beta branch and "
            "therefore requires DP_q_env=1. Got DP_q_env=0."
        )

    if FLAGS.mass_model != "DPUC_Primary" and FLAGS.DPUC_Primary_q_norm != "conditional":
        raise ValueError(
            "--DPUC_Primary_q_norm is only meaningful for --mass_model DPUC_Primary."
        )

    if FLAGS.DPUC_kernel in ("local_native_gaussian", "local_native_product_gaussian"):
        if FLAGS.mass_model != "DPUC":
            raise ValueError(
                "DPUC_kernel local-native branches are only supported for "
                f"mass_model='DPUC'. Got mass_model={FLAGS.mass_model!r}."
            )

    os.makedirs(FLAGS.fout, exist_ok=True)

    if hasattr(FLAGS, "settings") and FLAGS.settings: import shutil; shutil.copy(FLAGS.settings, os.path.join(FLAGS.fout, os.path.basename(FLAGS.settings)))

    if bool(FLAGS.bounded_gmm_transform):
        if FLAGS.pop_only:
            raise NotImplementedError("--bounded_gmm_transform is only implemented for pop_only=0 surrogate fits.")
        if FLAGS.sampling_gw != 'gauss':
            raise NotImplementedError("--bounded_gmm_transform is only implemented for sampling_gw='gauss'.")




    #from tqdm import tqdm 
    from tqdm.auto import tqdm

    import psutil
    try:
        _process = psutil.Process(os.getpid())
    except (psutil.Error, OSError) as exc:
        _process = None
        print(f"[MEM] Process memory monitoring unavailable: {exc}")

    def mem_gb():
        if _process is None:
            return None
        try:
            return _process.memory_info().rss / (1024**3)  # Resident Set Size in GB
        except (psutil.Error, OSError):
            return None
    
    def log_mem(tag):
        memory = mem_gb()
        if memory is None:
            print(f"[MEM] {tag}: unavailable")
        else:
            print(f"[MEM] {tag}: {memory:.2f} GB RSS")

 
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
    from numpyro.infer.initialization import init_to_value
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
    print(f"Models: mass={FLAGS.mass_model}, rate={FLAGS.rate_model}, spin={FLAGS.spin_model}, "
          f"cosmology={'GP' if FLAGS.is_GP_dL else 'standard'}; seed={FLAGS.seed}")
  
    
    print("dtype test:", np.array(0., dtype=np.float64).dtype)
    
    
    
    print(f"[PID] {os.getpid()}")
    

    with open(FLAGS.fin_priors) as json_file:
        priors = json.load(json_file)

    ########### GP
    if FLAGS.is_GP_dL:
        try:
            Uprior = priors["U"]
            if FLAGS.U != 0:
                print(f"Prior for U is present in priors, but input U={FLAGS.U} was given. This will override the prior")
                Uprior = FLAGS.U
            else:
                print(f"Prior for U from priors. U={Uprior}")
        except KeyError:
            if FLAGS.U == 0:
                raise ValueError("Insert U in prior for Gaussian Process or pass U as input arg")
            else:
                print(f"Prior for U from input U={FLAGS.U}")
                Uprior = FLAGS.U

        if not FLAGS.invert_dL_GP:
            raise ValueError("Gaussian Process for distance ratio requires inversion.")
    else:
        Uprior = None
    ########### END GP

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
        params_fix={"H0":67.7, "Om":0.31, "w0":-1, "Xi0":1., "nXi0":0.}

    ################################################
    # Load data
    ################################################
    
    # load sample means and covs
    print()
    print('*'*80)
    print('Loading data...')
    print('*'*80)
    print()


    if ( FLAGS.events_exclude != [] or FLAGS.events_use != []) and (FLAGS.nev_min != 0 or FLAGS.nev_max != -1):
        raise ValueError("Passed events_exclude=%s, events_use=%s, nev_min=%s, nev_max=%s. If events_exclude is not empty, nev_min must be 0 and nev_max=-1  "%(str(FLAGS.events_exclude), str(FLAGS.events_use), FLAGS.nev_min, FLAGS.nev_max ))
    
    
    if not FLAGS.pop_only:

        #data = dt.load_data_interp(FLAGS.fin_data, events_use=FLAGS.events_use)
        data = dt.load_data_interp(
            FLAGS.fin_data,
            events_use=FLAGS.events_use,
            events_exclude=FLAGS.events_exclude,
            bounded_transform=bool(FLAGS.bounded_gmm_transform),
        )

 
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

        if bool(FLAGS.bounded_gmm_transform):
            gmm_fit_coord_bounds = data['gmm_fit_coord_bounds']
            gmm_fit_coord_names = data['gmm_fit_coord_names']
            gmm_fit_transform_info = data['gmm_fit_transform_info']

            if gmm_fit_coord_bounds.shape[0] != samples_means_at.shape[0]:
                raise ValueError(
                    "gmm_fit_coord_bounds event dimension does not match samples_means. "
                    f"Got {gmm_fit_coord_bounds.shape[0]} and {samples_means_at.shape[0]}."
                )
            if gmm_fit_coord_bounds.shape[1] != samples_means_at.shape[1]:
                raise ValueError(
                    "gmm_fit_coord_bounds coordinate dimension does not match samples_means. "
                    f"Got {gmm_fit_coord_bounds.shape[1]} and {samples_means_at.shape[1]}."
                )
            if gmm_fit_coord_bounds.shape[2] != 2:
                raise ValueError(
                    "gmm_fit_coord_bounds must have shape (Nevents, ndim, 2). "
                    f"Got {gmm_fit_coord_bounds.shape}."
                )
            if len(gmm_fit_coord_names) != samples_means_at.shape[1]:
                raise ValueError(
                    "gmm_fit_coord_names length does not match samples_means dimension. "
                    f"Got {len(gmm_fit_coord_names)} and {samples_means_at.shape[1]}."
                )
        else:
            gmm_fit_coord_bounds = None
            gmm_fit_coord_names = None
            gmm_fit_transform_info = None

        if FLAGS.nev_min != 0 or FLAGS.nev_max != -1:
            if FLAGS.events_use or FLAGS.events_exclude:
                raise ValueError("Cannot select by index and name at the same time")
            N_original = int(np.sum(Nevents))
            first = int(FLAGS.nev_min)
            last = N_original if FLAGS.nev_max == -1 else int(FLAGS.nev_max)
            if not 0 <= first < last <= N_original:
                raise ValueError(f"Event range [{first}, {last}) is outside [0, {N_original})")
            selection = slice(first, last)
            samples_means_at = samples_means_at[selection]
            samples_cho_covs_at = samples_cho_covs_at[selection]
            gmm_log_wts = gmm_log_wts[selection]
            gmm_means = gmm_means[selection]
            gmm_icovs = gmm_icovs[selection]
            gmm_cho_covs = gmm_cho_covs[selection]
            gmm_log_dets = gmm_log_dets[selection]
            if bool(FLAGS.bounded_gmm_transform):
                gmm_fit_coord_bounds = gmm_fit_coord_bounds[selection]
            allNgm = allNgm[selection]
            selected_groups = []
            offset = 0
            for names in allnames:
                group_first = min(len(names), max(0, first - offset))
                group_last = min(len(names), max(0, last - offset))
                selected_groups.append(names[group_first:group_last])
                offset += len(names)
            allnames = selected_groups
            Nevents = np.asarray([len(names) for names in allnames], dtype=np.int64)
            if int(Nevents.sum()) != len(allNgm):
                raise ValueError("Selected event names and GMM arrays are not aligned")
            print(f"Using {last - first} of {N_original} events (indices [{first}, {last}))")


    

    else:
        print("Using n max samples = %s"%FLAGS.nsamplesmax)
        #data = dt.load_data_samples(FLAGS.fin_data, nmax=FLAGS.nsamplesmax)
        data = dt.load_data_samples(FLAGS.fin_data, nmax=FLAGS.nsamplesmax, events_exclude=FLAGS.events_exclude, events_use=FLAGS.events_use)

        m1d_samples = data['m1d_samples']
        m2d_samples =  data['m2d_samples']
        dL_samples =  data['dL_samples']
        dL_prior =  data['dL_PE_prior']
        print("dL_samples shape is %s"%(str(dL_samples.shape)))

        allNsamples =  data['allNsamples']
        where_compute = data['where_compute']

        allnames =  data['allnames']

        Nevents =  m1d_samples.shape[0]

        if FLAGS.nev_min != 0 or FLAGS.nev_max != -1:
            raise NotImplementedError()

        if FLAGS.spin_model == 'default_gauss':

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
        
        if FLAGS.spin_inj=='chi12xyz':

            if FLAGS.spin_model == 'default_gauss':

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
            raise NotImplementedError("sampling_gw containing 'gmm' has no implemented launcher path")
        elif FLAGS.sampling_gw=='gauss':
            if bool(FLAGS.bounded_gmm_transform):
                GWData =  [samples_means_at, #.astype(X),
                           samples_cho_covs_at, #astype(X),
                           gmm_log_wts, #.astype(X),
                           gmm_means, #.astype(X),
                           gmm_icovs, #.astype(X),
                           gmm_log_dets, #.astype(X),
                           gmm_cho_covs, #.astype(X),
                           gmm_fit_coord_bounds,
                           gmm_fit_coord_names,
                           gmm_fit_transform_info,
                           injections['Tobs'], #.astype(X),
                           Nevents,
                           allnames
                          ]
            else:
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
        print(f'Loaded {len(ivals)} initialization entries from {FLAGS.ivals}')
        _logger.debug('Initial values: %s', ivals)
    else:
        print('No initial values passed.')
        ivals={}
    print()

    if FLAGS.check_zres ==[]:

        model_numpyro, lik_data, core, loglik, loglik_components, model_init_info = make_model_jax(
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
            taper_kind = FLAGS.taper_kind,
            taper_p = FLAGS.taper_p,
            
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
            log_lik_var_min = FLAGS.log_lik_var_min,
            skip_sel = FLAGS.skip_sel,

            
            N_DP_comp_max=FLAGS.N_DP_comp_max,
            alpha_tail=FLAGS.alpha_tail,
            alpha_small=FLAGS.alpha_small,
            L_small_1=FLAGS.L_small_1,
            L_small_2=FLAGS.L_small_2,
            s_local=FLAGS.s_local,
            find_m_bounds=bool(FLAGS.find_m_bounds),
            q_mbound_mu_low=FLAGS.q_mbound_mu_low,
            q_mbound_mu_up=FLAGS.q_mbound_mu_up,
            q_mbound_sig_low=FLAGS.q_mbound_sig_low,
            q_mbound_sig_up=FLAGS.q_mbound_sig_up,
            alpha_inv_params=FLAGS.alpha_inv_params,
            DP_mu_bounds_source=FLAGS.DP_mu_bounds_source,
            DP_sig_bounds_source=FLAGS.DP_sig_bounds_source,
            M_active=FLAGS.M_active,
            tau_prior=FLAGS.tau_prior,
            DP_prior=FLAGS.DP_prior,
            sigma_softmax=FLAGS.sigma_softmax,
            gamma_DP_params=FLAGS.gamma_DP_params,
            DP_truncate_up=bool(FLAGS.DP_truncate_up),
            DPUC_mmax_fixed=FLAGS.DPUC_mmax_fixed,
            DP_truncate_low=bool(FLAGS.DP_truncate_low),
            DP_m1_env=bool(FLAGS.DP_m1_env),
            DP_m1_env_form=FLAGS.DP_m1_env_form,
            DP_m1_env_delta1=FLAGS.DP_m1_env_delta1,
            DP_m1_env_delta2=FLAGS.DP_m1_env_delta2,
            DP_q_env=bool(FLAGS.DP_q_env),
            DPUC_Primary_q_norm=FLAGS.DPUC_Primary_q_norm,
            DP_corr_m1m2=bool(FLAGS.DP_corr_m1m2),
            DP_corr_eta_rho = FLAGS.DP_corr_eta_rho,
            DP_corr_f_minor = FLAGS.DP_corr_f_minor,
            PL_DPUC_normalize_gmm = bool(FLAGS.PL_DPUC_normalize_gmm),
            DPUC_kernel = FLAGS.DPUC_kernel,
            DPUC_kernel_rcut = FLAGS.DPUC_kernel_rcut,
            DPUC_kernel_dr = FLAGS.DPUC_kernel_dr,
            DPUC_kernel_xcut = FLAGS.DPUC_kernel_xcut,
            DPUC_kernel_ycut = FLAGS.DPUC_kernel_ycut,
            DPUC_q_coord = FLAGS.DPUC_q_coord,
            DPUC_TruncM2_shared_weights=bool(FLAGS.DPUC_TruncM2_shared_weights),
            DPUC_TruncM2_mu2_mode=FLAGS.DPUC_TruncM2_mu2_mode,
            DPUC_TruncM2_secondary_coord=FLAGS.DPUC_TruncM2_secondary_coord,
            DPUC_TruncM2_area_reg=bool(FLAGS.DPUC_TruncM2_area_reg),
            DPUC_TruncM2_log_area_min=FLAGS.DPUC_TruncM2_log_area_min,
            DPUC_TruncM2_log_area_width=FLAGS.DPUC_TruncM2_log_area_width,
            DPUC_TruncM2_log_area_lambda=FLAGS.DPUC_TruncM2_log_area_lambda,
            DPUC_TruncM2_aspect_reg=bool(FLAGS.DPUC_TruncM2_aspect_reg),
            DPUC_TruncM2_aspect_max=FLAGS.DPUC_TruncM2_aspect_max,
            DPUC_TruncM2_aspect_width=FLAGS.DPUC_TruncM2_aspect_width,
            DPUC_TruncM2_aspect_lambda=FLAGS.DPUC_TruncM2_aspect_lambda,
            DPUC_TruncM2_low_center_weight_reg=bool(FLAGS.DPUC_TruncM2_low_center_weight_reg),
            DPUC_TruncM2_low_center_mcut=FLAGS.DPUC_TruncM2_low_center_mcut,
            DPUC_TruncM2_low_center_log_width=FLAGS.DPUC_TruncM2_low_center_log_width,
            DPUC_TruncM2_low_center_weight_cap=FLAGS.DPUC_TruncM2_low_center_weight_cap,
            DPUC_TruncM2_low_center_weight_width=FLAGS.DPUC_TruncM2_low_center_weight_width,
            DPUC_TruncM2_low_center_weight_lambda=FLAGS.DPUC_TruncM2_low_center_weight_lambda,
            DPUC_TruncM2_weight_cap_reg=bool(FLAGS.DPUC_TruncM2_weight_cap_reg),
            DPUC_TruncM2_weight_cap=FLAGS.DPUC_TruncM2_weight_cap,
            DPUC_physical_Lsmall_shared_mode = FLAGS.DPUC_physical_Lsmall_shared_mode,
            DPUC_q_mu_bounds_source = FLAGS.DPUC_q_mu_bounds_source,
            DPUC_low_taper_deltas=FLAGS.DPUC_low_taper_deltas,
            DPUC_low_end = FLAGS.DPUC_low_end,
            DPUC_low_end_share = FLAGS.DPUC_low_end_share,
            DPUC_high_taper_delta=FLAGS.DPUC_high_taper_delta,
            center_eps_widths=bool(FLAGS.center_eps_widths),
            L_small_source=FLAGS.L_small_source,
            L_small_res_percentile=FLAGS.L_small_res_percentile,
            L_small_res_factor=FLAGS.L_small_res_factor,
            mu_log_alpha_inv = FLAGS.mu_log_alpha_inv,
            sigma_log_alpha_inv = FLAGS.sigma_log_alpha_inv,
    
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
            reparam_cosmo = bool(FLAGS.reparam_cosmo),
            reparam_z = bool(FLAGS.reparam_z),
            remove_spin_prior =  bool(FLAGS.remove_spin_prior),
            r = FLAGS.r,
            vary_mb = FLAGS.vary_mb,
            is_GP_dL=bool(FLAGS.is_GP_dL),
            find_GP_L=bool(FLAGS.find_GP_L),
            monotonicity=FLAGS.monotonicity,
            GP_kernel=FLAGS.GP_kernel,
            U=Uprior,
            GP_prior=FLAGS.GP_prior,
            GP_jitter=FLAGS.GP_jitter,
            large_ell_penalty=bool(FLAGS.large_ell_penalty),
            ell_min=FLAGS.ell_min,
            ell_max=FLAGS.ell_max,
            ell0_min=FLAGS.ell0_min,
            ell0_max=FLAGS.ell0_max,
            nu=FLAGS.nu,
            lam=FLAGS.lam,
            res_nbins=FLAGS.res_nbins,
            res_q=FLAGS.res_q,
            smooth_GP_resolution_loglinear=FLAGS.smooth_GP_resolution_loglinear,
            fine_res=FLAGS.fine_res,
            res_highz=FLAGS.res_highz,
            res_lowz=FLAGS.res_lowz,
            eps_DE=FLAGS.eps_DE,
            invert_dL_GP=bool(FLAGS.invert_dL_GP),
            n_gp_nodes=FLAGS.n_gp_nodes,
            n_gp_fine=FLAGS.n_gp_fine,
            fout=FLAGS.fout,
            GP_oob_guard=bool(FLAGS.GP_oob_guard),
            alpha_eta=FLAGS.alpha_eta,
            alpha_ell=FLAGS.alpha_ell,
            alpha_large=FLAGS.alpha_large,
            logXi_guard=bool(FLAGS.logXi_guard),
            logXi_abs_max_allowed=FLAGS.logXi_abs_max_allowed,
            logXi_guard_strength=FLAGS.logXi_guard_strength,
        )

        # print("---- bounded GMM diagnostic ----")
        # print("lik_data.gmm_fit_transform_mode =", getattr(lik_data, "gmm_fit_transform_mode", None))
        # print("lik_data.gmm_fit_coord_bounds shape =", None if getattr(lik_data, "gmm_fit_coord_bounds", None) is None else lik_data.gmm_fit_coord_bounds.shape)
        # print("lik_data.gmm_fit_coord_names =", getattr(lik_data, "gmm_fit_coord_names", None))
        # print("--------------------------------")


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
            taper_kind = FLAGS.taper_kind,
            taper_p = FLAGS.taper_p,
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
            log_lik_var_min = FLAGS.log_lik_var_min,
            skip_sel = FLAGS.skip_sel,
                         
            N_DP_comp_max=FLAGS.N_DP_comp_max,
            alpha_tail=FLAGS.alpha_tail,
            alpha_small=FLAGS.alpha_small,
            L_small_1=FLAGS.L_small_1,
            L_small_2=FLAGS.L_small_2,
            s_local=FLAGS.s_local,
            find_m_bounds=bool(FLAGS.find_m_bounds),
            q_mbound_mu_low=FLAGS.q_mbound_mu_low,
            q_mbound_mu_up=FLAGS.q_mbound_mu_up,
            q_mbound_sig_low=FLAGS.q_mbound_sig_low,
            q_mbound_sig_up=FLAGS.q_mbound_sig_up,
            alpha_inv_params=FLAGS.alpha_inv_params,
                DP_mu_bounds_source=FLAGS.DP_mu_bounds_source,
                DP_sig_bounds_source=FLAGS.DP_sig_bounds_source,
            M_active=FLAGS.M_active,
            tau_prior=FLAGS.tau_prior,
            DP_prior=FLAGS.DP_prior,
            sigma_softmax=FLAGS.sigma_softmax,
            gamma_DP_params=FLAGS.gamma_DP_params,
            DP_truncate_up=bool(FLAGS.DP_truncate_up),
                DPUC_mmax_fixed=FLAGS.DPUC_mmax_fixed,
            DP_truncate_low=bool(FLAGS.DP_truncate_low),
            DP_m1_env=bool(FLAGS.DP_m1_env),
            DP_m1_env_form=FLAGS.DP_m1_env_form,
            DP_m1_env_delta1=FLAGS.DP_m1_env_delta1,
            DP_m1_env_delta2=FLAGS.DP_m1_env_delta2,
            DP_q_env=bool(FLAGS.DP_q_env),
            DPUC_Primary_q_norm=FLAGS.DPUC_Primary_q_norm,
            DP_corr_eta_rho = FLAGS.DP_corr_eta_rho,
            DP_corr_f_minor = FLAGS.DP_corr_f_minor,
            PL_DPUC_normalize_gmm = bool(FLAGS.PL_DPUC_normalize_gmm),
            DPUC_kernel = FLAGS.DPUC_kernel,
            DPUC_kernel_rcut = FLAGS.DPUC_kernel_rcut,
            DPUC_kernel_dr = FLAGS.DPUC_kernel_dr,
            DPUC_kernel_xcut = FLAGS.DPUC_kernel_xcut,
            DPUC_kernel_ycut = FLAGS.DPUC_kernel_ycut,
            DPUC_q_coord = FLAGS.DPUC_q_coord,
            DPUC_TruncM2_shared_weights=bool(FLAGS.DPUC_TruncM2_shared_weights),
            DPUC_TruncM2_mu2_mode=FLAGS.DPUC_TruncM2_mu2_mode,
            DPUC_TruncM2_secondary_coord=FLAGS.DPUC_TruncM2_secondary_coord,
            DPUC_TruncM2_area_reg=bool(FLAGS.DPUC_TruncM2_area_reg),
            DPUC_TruncM2_log_area_min=FLAGS.DPUC_TruncM2_log_area_min,
            DPUC_TruncM2_log_area_width=FLAGS.DPUC_TruncM2_log_area_width,
            DPUC_TruncM2_log_area_lambda=FLAGS.DPUC_TruncM2_log_area_lambda,
            DPUC_TruncM2_aspect_reg=bool(FLAGS.DPUC_TruncM2_aspect_reg),
            DPUC_TruncM2_aspect_max=FLAGS.DPUC_TruncM2_aspect_max,
            DPUC_TruncM2_aspect_width=FLAGS.DPUC_TruncM2_aspect_width,
            DPUC_TruncM2_aspect_lambda=FLAGS.DPUC_TruncM2_aspect_lambda,
            DPUC_TruncM2_low_center_weight_reg=bool(FLAGS.DPUC_TruncM2_low_center_weight_reg),
            DPUC_TruncM2_low_center_mcut=FLAGS.DPUC_TruncM2_low_center_mcut,
            DPUC_TruncM2_low_center_log_width=FLAGS.DPUC_TruncM2_low_center_log_width,
            DPUC_TruncM2_low_center_weight_cap=FLAGS.DPUC_TruncM2_low_center_weight_cap,
            DPUC_TruncM2_low_center_weight_width=FLAGS.DPUC_TruncM2_low_center_weight_width,
            DPUC_TruncM2_low_center_weight_lambda=FLAGS.DPUC_TruncM2_low_center_weight_lambda,
            DPUC_TruncM2_weight_cap_reg=bool(FLAGS.DPUC_TruncM2_weight_cap_reg),
            DPUC_TruncM2_weight_cap=FLAGS.DPUC_TruncM2_weight_cap,
            DPUC_physical_Lsmall_shared_mode = FLAGS.DPUC_physical_Lsmall_shared_mode,
            DPUC_q_mu_bounds_source = FLAGS.DPUC_q_mu_bounds_source,
            DPUC_low_taper_deltas=FLAGS.DPUC_low_taper_deltas,
                DPUC_low_end = FLAGS.DPUC_low_end,
            DPUC_low_end_share = FLAGS.DPUC_low_end_share,
                DPUC_high_taper_delta=FLAGS.DPUC_high_taper_delta,
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
            reparam_cosmo = bool(FLAGS.reparam_cosmo),
            reparam_z = bool(FLAGS.reparam_z),
            remove_spin_prior=bool(FLAGS.remove_spin_prior),
            r=FLAGS.r,
            vary_mb=FLAGS.vary_mb,
            is_GP_dL=bool(FLAGS.is_GP_dL),
            find_GP_L=bool(FLAGS.find_GP_L),
            monotonicity=FLAGS.monotonicity,
            GP_kernel=FLAGS.GP_kernel,
            U=Uprior,
            GP_prior=FLAGS.GP_prior,
            GP_jitter=FLAGS.GP_jitter,
            large_ell_penalty=bool(FLAGS.large_ell_penalty),
            ell_min=FLAGS.ell_min,
            ell_max=FLAGS.ell_max,
            ell0_min=FLAGS.ell0_min,
            ell0_max=FLAGS.ell0_max,
            nu=FLAGS.nu,
            lam=FLAGS.lam,
            res_nbins=FLAGS.res_nbins,
            res_q=FLAGS.res_q,
            smooth_GP_resolution_loglinear=FLAGS.smooth_GP_resolution_loglinear,
            fine_res=FLAGS.fine_res,
            res_highz=FLAGS.res_highz,
            res_lowz=FLAGS.res_lowz,
            eps_DE=FLAGS.eps_DE,
            invert_dL_GP=bool(FLAGS.invert_dL_GP),
            n_gp_nodes=FLAGS.n_gp_nodes,
            n_gp_fine=FLAGS.n_gp_fine,
            fout=FLAGS.fout,
            GP_oob_guard=bool(FLAGS.GP_oob_guard),
            alpha_eta=FLAGS.alpha_eta,
            alpha_ell=FLAGS.alpha_ell,
            alpha_large=FLAGS.alpha_large,
            logXi_guard=bool(FLAGS.logXi_guard),
            logXi_abs_max_allowed=FLAGS.logXi_abs_max_allowed,
            logXi_guard_strength=FLAGS.logXi_guard_strength,
            DP_corr_m1m2=bool(FLAGS.DP_corr_m1m2),
            center_eps_widths=bool(FLAGS.center_eps_widths),
            L_small_source=FLAGS.L_small_source,
            L_small_res_percentile=FLAGS.L_small_res_percentile,
            L_small_res_factor=FLAGS.L_small_res_factor,
            mu_log_alpha_inv=FLAGS.mu_log_alpha_inv,
            sigma_log_alpha_inv=FLAGS.sigma_log_alpha_inv,
        )
    
    
        model_numpyro, lik_data, core, loglik, loglik_components, model_init_info = build_model_for_zres(FLAGS.zres)

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
    
            _, lik_data_z, core_z, loglik_z, _, _ = build_model_for_zres(zr)
    
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
    
    init_vals = {k: v for k, v in ivals.items() if not str(k).startswith("_")}
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
        # For evo_triplet_numpyro :
        # theta_inf = theta0 + delta_name
        # so initialize delta_name = theta_inf_init - theta0_init when available.
        def _softplus_inv(y):
            y = float(y)
            y = max(y, 1e-12)
            return float(np.log(np.expm1(y)))
        
        
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
                sigma = float(priors.get(f"delta_{name}_sigma", 1.0))
                lower = -theta0 + eps_pos
        
                # Model uses:
                # delta = lower + sigma * softplus(raw)
                # so raw init must satisfy:
                # softplus(raw) = (delta - lower) / sigma
                y = (delta - lower) / sigma
        
                if y <= 0.0:
                    raise ValueError(
                        f"Invalid init for delta_{name}: delta={delta}, "
                        f"lower={lower}, sigma={sigma}, y={y}. "
                        "Need delta > lower."
                    )
        
                init_vals[f"delta_{name}_raw"] = _softplus_inv(y)
            else:
                init_vals[delta_key] = delta
        
        
        # --- delta (theta_inf - theta0) init ---
        _init_delta("alpha1", "alpha1_0")
        _init_delta("alpha2", "alpha2_0")
        
        if FLAGS.vary_mb:
            _init_delta("mb", "mb_0", positive=True, eps_pos=5.0)
        
        _init_delta("mu1", "mu1_0", positive=True, eps_pos=5.0)
        _init_delta("sigma1", "sigma1_0", positive=True, eps_pos=0.1)
        _init_delta("mu2", "mu2_0", positive=True, eps_pos=5.0)
        _init_delta("sigma2", "sigma2_0", positive=True, eps_pos=0.1)
        
        
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



    if FLAGS.mass_model in ('DPUC', 'DPUC_Primary', 'DPUC_TruncM2'):
    
        K = int(model_init_info["N_DP_comp_max"])
        M_active = min(int(model_init_info["M_active"]), K)
    
        # ------------------------------------------------------------------
        # CLEAN init_vals from anything that is NOT a NumPyro sample site
        # ------------------------------------------------------------------
        for drop in [
            'beta',
            'beta_q',
            'mulMc',
            'mulq',
            'tau1',
            'tau2',
            'tau1_raw',
            'tau2_raw',
            'eps1',
            'eps2',
            'alpha_inv',
            'alpha_SB',
            'log_alpha_inv',
            'mmax_DP',
            'mmax_DP_raw',
            'delta_mmax_DP',
            'u',
            'u_raw',
            'v',
            'v_raw',
            'm1_low',
            'm2_low',
            'alpha',
            'alpha_raw',
            'DP_m1_env_alpha0',
            'DP_m1_env_alpha0_raw',
            'DP_m1_env_alpha1',
            'DP_m1_env_alpha1_raw',
            'DP_m1_env_alpha2',
            'DP_m1_env_alpha2_raw',
            'DP_m1_env_mb1',
            'DP_m1_env_mb1_raw',
            'DP_m1_env_mb2',
            'DP_m1_env_mb2_raw',
            'DP_m1_env_order_mb',
            'deltam_PL_DPUC_m1',
            'deltam_PL_DPUC_m2',
            'deltam_PL_DPUC_m1_raw',
            'deltam_PL_DPUC_m2_raw',
            'deltam_PL_DPUC_m1_log',
            'deltam_PL_DPUC_m2_log',
            'eps1_raw',
            'eps2_raw',
            'alpha1',
            'alpha2',
            'alpha_bar',
            'alpha_diff',
            'mb',
            'mb_raw',
            'mu1',
            'mu2',
            'sigma1',
            'sigma2',
            'delta_m1',
            'delta_m2',
            'lambda',
            'H0',
            'H0_raw',
            'Om',
            'Om_raw',
            'w0',
            'w0_raw',
            'gamma',
            'gamma_raw',
            'kappa',
            'kappa_raw',
            'zp',
            'zp_raw',
            'muChi',
            'muChi_raw',
            'sigmaChi',
            'sigmaChi_raw',
            'zeta',
            'zeta_raw',
            'sigmat',
            'sigmat_raw',
        ]:
        
            init_vals.pop(drop, None)

        # Helpers for using an exported posterior draw as a NumPyro initial point.
        # They are deliberately local to the DPUC-family block to avoid changing
        # behavior for other mass models.
        def _ivals_array(name, shape=None):
            if name not in ivals or ivals.get(name) is None:
                return None
            arr = np.asarray(ivals.get(name), dtype=np.float64)
            if shape is not None and tuple(arr.shape) != tuple(shape):
                return None
            if not np.all(np.isfinite(arr)):
                return None
            return arr

        def _logit_unit(t):
            t = np.clip(np.asarray(t, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
            return np.log(t / (1.0 - t))

        def _bounded_raw_from_value(x, lo, hi):
            lo = float(lo); hi = float(hi)
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, lo + 1.0e-6 * (hi - lo), hi - 1.0e-6 * (hi - lo))
            return _logit_unit((x - lo) / (hi - lo))

        def _stick_breaking_beta_from_logw(logw, K_use=None):
            K_use = K if K_use is None else int(K_use)
            logw = np.asarray(logw, dtype=np.float64).reshape(-1)
            if logw.shape[0] != K_use or not np.all(np.isfinite(logw)):
                return None
            w = np.exp(logw - np.max(logw))
            w = w / np.sum(w)
            beta = np.zeros(K_use, dtype=np.float64)
            remaining = 1.0
            for kk in range(K_use):
                if kk == K_use - 1:
                    beta[kk] = 0.5
                else:
                    frac = w[kk] / max(remaining, 1.0e-12)
                    beta[kk] = np.clip(frac, 1.0e-6, 1.0 - 1.0e-6)
                    remaining -= w[kk]
            return beta
    
        # ------------------------------------------------------------------
        # COSMOLOGY / RATE (MATCH MODEL EXACTLY)
        # ------------------------------------------------------------------
        if not FLAGS.fix_H0:
            if FLAGS.reparam_cosmo:
                init_vals["H0_raw"] = float(_bounded_raw_from_value(ivals.get("H0", 70.0), priors["H0"][0], priors["H0"][1]))
            else:
                init_vals["H0"] = float(ivals.get("H0", 70.0))
    
        if not FLAGS.fix_Om:
            if FLAGS.reparam_cosmo:
                init_vals["Om_raw"] = float(_bounded_raw_from_value(ivals.get("Om", 0.3), priors["Om"][0], priors["Om"][1]))
            else:
                init_vals["Om"] = float(ivals.get("Om", 0.3))
    
        if not FLAGS.fix_w0:
            if FLAGS.reparam_cosmo:
                init_vals["w0_raw"] = float(_bounded_raw_from_value(ivals.get("w0", -1.0), priors["w0"][0], priors["w0"][1]))
            else:
                init_vals["w0"] = float(ivals.get("w0", -1.0))
    
        for k in ("gamma", "kappa", "zp"):
            if FLAGS.reparam_z:
                init_vals[f"{k}_raw"] = float(_bounded_raw_from_value(ivals.get(k, 0.5 * (float(priors[k][0]) + float(priors[k][1]))), priors[k][0], priors[k][1]))
            else:
                init_vals[k] = float(ivals.get(k, 0.0))
    
        # ------------------------------------------------------------------
        # SPINS
        # ------------------------------------------------------------------
        if FLAGS.spin_model == "default_gauss":
            for k in ("muChi", "sigmaChi", "zeta"):
                init_vals[f"{k}_raw"] = 0.0
            init_vals["sigmat_raw"] = 1.0
    
        # ------------------------------------------------------------------
        # DP WEIGHTS (STICK BREAKING)
        # Match the model-side M_active initialization logic.
        # ------------------------------------------------------------------
        init_vals["alpha_inv"] = float(ivals.get("alpha_inv", 10.0))
        
        M_active_init = min(int(FLAGS.M_active), K)
        
        if M_active_init == 0:
            beta_init = np.full(K, 0.02, dtype=np.float64)
            beta_init[:5] = [0.80, 0.60, 0.35, 0.20, 0.10]
        else:
            tail_mass_frac = 0.02
            active_mass = 1.0 - tail_mass_frac
            n_tail = K - M_active_init
        
            active_profile = np.linspace(1.0, 0.9, M_active_init)
            active_profile = active_profile / active_profile.sum()
            w_active = active_mass * active_profile
        
            if n_tail > 0:
                w_tail = np.full(n_tail, tail_mass_frac / n_tail)
                w_init = np.concatenate([w_active, w_tail])
            else:
                w_init = w_active.copy()
        
            beta_init = np.zeros(K, dtype=np.float64)
            remaining = 1.0
            eps_beta = 1e-6
        
            for k in range(K):
                if k == K - 1:
                    beta_init[k] = 0.5
                else:
                    frac = w_init[k] / max(remaining, 1e-12)
                    beta_init[k] = np.clip(frac, eps_beta, 1.0 - eps_beta)
                    remaining -= w_init[k]
        
        beta_from_trace = None
        beta_arr = _ivals_array("beta", (K,))
        if beta_arr is not None:
            beta_from_trace = np.clip(beta_arr, 1.0e-6, 1.0 - 1.0e-6)
        else:
            beta_from_trace = _stick_breaking_beta_from_logw(_ivals_array("logw"))
        init_vals["beta"] = beta_from_trace if beta_from_trace is not None else beta_init

        if FLAGS.mass_model == 'DPUC_TruncM2' and (not bool(getattr(FLAGS, 'DPUC_TruncM2_shared_weights', 0))):
            if "SB" in FLAGS.DP_prior:
                beta_w2_from_trace = _stick_breaking_beta_from_logw(_ivals_array("logw2"))
                init_vals["beta_w2"] = beta_w2_from_trace if beta_w2_from_trace is not None else beta_init.copy()
            elif FLAGS.DP_prior == "dirichelet":
                init_vals["w2"] = np.ones(K, dtype=np.float64) / float(K)
            elif FLAGS.DP_prior == "softmax":
                init_vals["raw_w2"] = np.zeros(K, dtype=np.float64)

        if FLAGS.mass_model in ("DPUC_TruncM2", ) and bool(getattr(FLAGS, "DP_corr_m1m2", 0)):
            # rho_u ~ Beta(eta, eta), rho = rho_max * (2*rho_u - 1).
            rho_u_arr = _ivals_array("rho_u", (K,))
            if rho_u_arr is not None:
                init_vals["rho_u"] = np.clip(rho_u_arr, 1.0e-6, 1.0 - 1.0e-6)
            else:
                rho_arr = _ivals_array("rho", (K,))
                if rho_arr is not None:
                    rho_max0 = np.sqrt(max(0.0, 1.0 - float(FLAGS.DP_corr_f_minor) ** 2))
                    init_vals["rho_u"] = np.clip(0.5 * (rho_arr / rho_max0 + 1.0), 1.0e-6, 1.0 - 1.0e-6)
                else:
                    init_vals["rho_u"] = 0.5 * np.ones(K, dtype=np.float64)


        # ------------------------------------------------------------------
        # DPUC low-taper width init.
        #
        # If:
        #     --DPUC_low_taper_deltas -1
        # then model samples:
        #     deltam_PL_DPUC_m1_raw
        #     deltam_PL_DPUC_m2_raw
        #
        # Deterministic physical outputs remain:
        #     deltam_PL_DPUC_m1
        #     deltam_PL_DPUC_m2
        # ------------------------------------------------------------------
        DPUC_low_taper_deltas_init = list(FLAGS.DPUC_low_taper_deltas)

        if len(DPUC_low_taper_deltas_init) == 1 and float(DPUC_low_taper_deltas_init[0]) < 0.0:

            def _bounded_logpositive_raw_init(initval, low, high):
                low = float(low)
                high = float(high)
                initval = float(initval)

                if not (0.0 < low < high):
                    raise ValueError(f"Need 0 < low < high, got low={low}, high={high}")

                initval = np.clip(initval, low * (1.0 + 1.0e-6), high * (1.0 - 1.0e-6))

                t = (np.log(initval) - np.log(low)) / (np.log(high) - np.log(low))
                t = np.clip(t, 1.0e-6, 1.0 - 1.0e-6)
                return np.log(t / (1.0 - t))

            dm1_low, dm1_high = float(priors["delta_m1"][0]), float(priors["delta_m1"][1])
            dm2_low, dm2_high = float(priors["delta_m2"][0]), float(priors["delta_m2"][1])

            dm1_init = float(
                ivals.get(
                    "deltam_PL_DPUC_m1",
                    ivals.get("delta_m1", 2.0),
                )
            )
            dm2_init = float(
                ivals.get(
                    "deltam_PL_DPUC_m2",
                    ivals.get("delta_m2", 2.0),
                )
            )

            init_vals["deltam_PL_DPUC_m1_raw"] = _bounded_logpositive_raw_init(
                dm1_init,
                dm1_low,
                dm1_high,
            )
            init_vals["deltam_PL_DPUC_m2_raw"] = _bounded_logpositive_raw_init(
                dm2_init,
                dm2_low,
                dm2_high,
            )

        elif len(DPUC_low_taper_deltas_init) == 2:
            pass

        else:
            raise ValueError(
                "DPUC_low_taper_deltas must be either two positive fixed values "
                "or single negative value -1 to sample them. "
                f"Got {DPUC_low_taper_deltas_init}."
            )

        # ------------------------------------------------------------------
        # MEANS (STRICTLY INSIDE SUPPORT)
        # ------------------------------------------------------------------
        low1, up1 = float(model_init_info["lowmu1"]), float(model_init_info["upmu1"])
        low2, up2 = float(model_init_info["lowmu2"]), float(model_init_info["upmu2"])
    
        eps = 1e-2
    
        mu1 = np.linspace(low1 + eps, up1 - eps, K)
        mu2 = np.linspace(low2 + eps, up2 - eps, K)
    
        # init_vals["mulMc"] = mu1
        # init_vals["mulq"] = mu2

        t1 = (mu1 - low1) / (up1 - low1)
        t2 = (mu2 - low2) / (up2 - low2)
        
        t1 = np.clip(t1, 1.0e-6, 1.0 - 1.0e-6)
        t2 = np.clip(t2, 1.0e-6, 1.0 - 1.0e-6)
        
        mu_arr = _ivals_array("mu")
        mulMc_raw_arr = _ivals_array("mulMc_raw", (K,))
        mulq_raw_arr = _ivals_array("mulq_raw", (K,))
        q_mu_raw_arr = _ivals_array("q_mu_raw", (K,))

        if mulMc_raw_arr is not None:
            init_vals["mulMc_raw"] = mulMc_raw_arr
        elif mu_arr is not None and mu_arr.shape[0] >= 1 and mu_arr.shape[1] == K:
            init_vals["mulMc_raw"] = _bounded_raw_from_value(mu_arr[0], low1, up1)
        else:
            init_vals["mulMc_raw"] = np.log(t1 / (1.0 - t1))

        if not FLAGS.mass_model == 'DPUC_Primary':
            if FLAGS.mass_model in ("DPUC_TruncM2") and FLAGS.DPUC_TruncM2_mu2_mode == "ordered":
                if mulq_raw_arr is not None:
                    init_vals["mulq_raw"] = mulq_raw_arr
                else:
                    init_vals["mulq_raw"] = np.zeros(K, dtype=np.float64)
            elif FLAGS.mass_model in ("DPUC_TruncM2",  ) and FLAGS.DPUC_TruncM2_mu2_mode == "q_ratio":
                if q_mu_raw_arr is not None:
                    init_vals["q_mu_raw"] = q_mu_raw_arr
                elif mu_arr is not None and mu_arr.shape[0] >= 2 and mu_arr.shape[1] == K:
                    qlo = model_init_info.get("q_mu_low")
                    qhi = model_init_info.get("q_mu_up")
                    if qlo is None or qhi is None:
                        init_vals["q_mu_raw"] = np.zeros(K, dtype=np.float64)
                    else:
                        if str(getattr(FLAGS, "DPUC_TruncM2_secondary_coord", "logm2")) == "logitq":
                            q_mu0 = 1.0 / (1.0 + np.exp(-mu_arr[1]))
                        else:
                            q_mu0 = np.exp(mu_arr[1] - np.asarray(mu_arr[0]))
                        init_vals["q_mu_raw"] = _bounded_raw_from_value(q_mu0, float(qlo), float(qhi))
                else:
                    init_vals["q_mu_raw"] = np.zeros(K, dtype=np.float64)
            else:
                if mulq_raw_arr is not None:
                    init_vals["mulq_raw"] = mulq_raw_arr
                elif mu_arr is not None and mu_arr.shape[0] >= 2 and mu_arr.shape[1] == K:
                    init_vals["mulq_raw"] = _bounded_raw_from_value(mu_arr[1], low2, up2)
                else:
                    init_vals["mulq_raw"] = np.log(t2 / (1.0 - t2))
    

        
        # ------------------------------------------------------------------
        # WIDTHS (CRITICAL FOR NAN FIX)
        # ------------------------------------------------------------------
        U1 = up1 - low1
        U2 = up2 - low2
    
        tau_floor_eps = 1.0e-6
        L1_used = float(model_init_info.get("L_small_1", 0.0))
        L2_used = float(model_init_info.get("L_small_2", 0.0))

        def _set_tau_init(name, default, U, Lsmall):
            val = ivals.get(name, default)
            if FLAGS.tau_prior == "bounded_sigmoid":
                init_vals[f"{name}_raw"] = float(_bounded_raw_from_value(val, tau_floor_eps, max(tau_floor_eps * 10.0, float(U - Lsmall))))
            else:
                init_vals[name] = float(np.clip(float(val), tau_floor_eps, max(tau_floor_eps * 10.0, float(U - Lsmall))))

        _set_tau_init("tau1", 0.3 * U1, U1, L1_used)
        if not FLAGS.mass_model == 'DPUC_Primary':
            _set_tau_init("tau2", 0.4 * U2, U2, L2_used)

    
    
        # ------------------------------------------------------------------
        # NOISE TERMS (SMALL!)
        # ------------------------------------------------------------------
        if FLAGS.center_eps_widths:
            e1 = _ivals_array("eps1_raw", (K,))
            if e1 is None:
                e1 = 1e-3 * np.random.randn(K)
            init_vals["eps1_raw"] = e1 - np.mean(e1)
            if not FLAGS.mass_model == 'DPUC_Primary':
                e2 = _ivals_array("eps2_raw", (K,))
                if e2 is None:
                    e2 = 1e-3 * np.random.randn(K)
                init_vals["eps2_raw"] = e2 - np.mean(e2)
        
        else:
            eps1_arr = _ivals_array("eps1", (K,))
            init_vals["eps1"] = eps1_arr if eps1_arr is not None else 1e-3 * np.random.randn(K)
            if not FLAGS.mass_model == 'DPUC_Primary':
                eps2_arr = _ivals_array("eps2", (K,))
                init_vals["eps2"] = eps2_arr if eps2_arr is not None else 1e-3 * np.random.randn(K)
        
    
        # ------------------------------------------------------------------
        # TRUNCATION
        #
        # Match jax_models.py exactly:
        #
        #   mmax_DP = shifted_lognormal_q95("mmax_DP", floor, q95, ...)
        #             sampled site: delta_mmax_DP
        # ------------------------------------------------------------------
        if FLAGS.DP_truncate_low:

            DPUC_low_end_init = list(FLAGS.DPUC_low_end)

            if len(DPUC_low_end_init) == 1 and float(DPUC_low_end_init[0]) < 0.0:
                # sample low end 
            
                m1_lo = float(priors["m1_low"][0])
                m1_hi = float(priors["m1_low"][1])
                m2_lo = float(priors["m2_low"][0])

                if m1_hi <= m1_lo:
                    raise ValueError(f"Invalid m1_low prior bounds: {priors['m1_low']}")

                if m2_lo >= m1_lo:
                    raise ValueError(
                        "Need priors['m2_low'][0] < priors['m1_low'][0] for the constrained DPUC low-mass prior."
                    )

                # Your model:
                #   u = sigmoid(u_raw)
                #   m1_low = m1_lo + (m1_hi - m1_lo) * u**1.5
                m1_low0 = float(ivals.get("m1_low", 0.5 * (m1_lo + m1_hi)))
                m1_low0 = np.clip(m1_low0, m1_lo + 1.0e-6, m1_hi - 1.0e-6)

                u0 = ((m1_low0 - m1_lo) / (m1_hi - m1_lo)) ** (1.0 / 1.5)
                u0 = np.clip(u0, 1.0e-6, 1.0 - 1.0e-6)

                init_vals["u_raw"] = float(np.log(u0 / (1.0 - u0)))

                # Your model:
                #   v = sigmoid(v_raw)
                #   m2_low = m2_lo + v * (m1_low - m2_lo)
                m2_low0 = float(ivals.get("m2_low", 0.5 * (m2_lo + m1_low0)))
                m2_low0 = np.clip(m2_low0, m2_lo + 1.0e-6, m1_low0 - 1.0e-6)

                v0 = (m2_low0 - m2_lo) / (m1_low0 - m2_lo)
                v0 = np.clip(v0, 1.0e-6, 1.0 - 1.0e-6)

                init_vals["v_raw"] = float(np.log(v0 / (1.0 - v0)))
            else:
                # low end is fixed
                pass
        
        if FLAGS.DP_truncate_up and float(getattr(FLAGS, "DPUC_mmax_fixed", -1.0)) <= 0.0:
            mmax_floor = float(priors["mmax_DP"][0])
            mmax_q95 = float(priors["mmax_DP"][1])
        
            mmax0 = float(
                ivals.get(
                    "mmax_DP",
                    0.5 * (mmax_floor + mmax_q95),
                )
            )
        
            # Keep strictly above floor.
            init_vals["delta_mmax_DP"] = max(1e-6, mmax0 - mmax_floor)
    
        if FLAGS.DP_m1_env:
            if str(getattr(FLAGS, "DP_m1_env_form", "powerlaw")) == "smooth_broken_pl_2breaks":
                for key in (
                    "DP_m1_env_alpha0",
                    "DP_m1_env_alpha1",
                    "DP_m1_env_alpha2",
                    "DP_m1_env_mb1",
                    "DP_m1_env_mb2",
                ):
                    if key not in priors:
                        raise KeyError(
                            "DP_m1_env_form='smooth_broken_pl_2breaks' requires "
                            f"priors[{key!r}]."
                        )

                def _bounded_raw_for_prior(name, default):
                    lo, hi = float(priors[name][0]), float(priors[name][1])
                    x = float(ivals.get(name, default))
                    x = np.clip(x, lo + 1.0e-6 * (hi - lo), hi - 1.0e-6 * (hi - lo))
                    t = (x - lo) / (hi - lo)
                    t = np.clip(t, 1.0e-6, 1.0 - 1.0e-6)
                    return float(np.log(t / (1.0 - t)))

                init_vals["DP_m1_env_alpha0_raw"] = _bounded_raw_for_prior("DP_m1_env_alpha0", 0.0)
                init_vals["DP_m1_env_alpha1_raw"] = _bounded_raw_for_prior("DP_m1_env_alpha1", 2.0)
                init_vals["DP_m1_env_alpha2_raw"] = _bounded_raw_for_prior("DP_m1_env_alpha2", 5.0)
                init_vals["DP_m1_env_mb1_raw"] = _bounded_raw_for_prior("DP_m1_env_mb1", 4.0)
                init_vals["DP_m1_env_mb2_raw"] = _bounded_raw_for_prior("DP_m1_env_mb2", 40.0)
            else:
                # Initialize the physical Normal sample site; accept the legacy key.
                init_vals["alpha"] = float(ivals.get("alpha", ivals.get("alpha1", 2.0)))

        if FLAGS.DP_q_env:
            def _bounded_raw_for_beta_q(x0=1.0):
                lo, hi = priors["beta"][0], priors["beta"][1]
                x = float(ivals.get("beta_q", ivals.get("beta", x0)))
                x = np.clip(x, lo + 1.0e-6 * (hi - lo), hi - 1.0e-6 * (hi - lo))
                t = (x - lo) / (hi - lo)
                t = np.clip(t, 1.0e-6, 1.0 - 1.0e-6)
                return float(np.log(t / (1.0 - t)))
            init_vals["beta_q_raw"] = _bounded_raw_for_beta_q()


        


    # ----------------------------------------------------------------------
    if FLAGS.is_GP_dL:

        if ivals != {}:
            if "f_rotated" not in ivals:

                print("Initializing GP...")

                #import numpy as onp
                #import jax.numpy as jnp
                #import jax
                z_grid = jnp.asarray(model_init_info["z_gp_nodes"], dtype=jnp.float64)
                x_grid = jnp.log1p(z_grid)
                N = x_grid.shape[0]

                if FLAGS.GP_kernel == "gibbs_resolution":
                    ell = ivals.get("ell", 1.0)
                    eta = ivals.get("eta", 0.2)
                else:
                    ell = ivals.get("ell", 0.3)
                    eta = ivals.get("eta", 0.2)

                print(f"Initial ell={ell}, eta={eta}")

                if FLAGS.GP_kernel == "gibbs_resolution":
                    x_centers = jnp.asarray(model_init_info["x_res_centers"], dtype=jnp.float64)
                    r_centers = jnp.asarray(model_init_info["r_res_centers"], dtype=jnp.float64)
                    from numerical_utils import gibbs_kernel_matrix
                    from backends import JAXBackend

                    K = gibbs_kernel_matrix(
                        JAXBackend(), x_grid,
                        ell,
                        eta,
                        x_centers,
                        r_centers,
                        jitter=0.0,
                    )

                else:
                    from numerical_utils import matern52_kernel_matrix
                    from backends import JAXBackend

                    K = matern52_kernel_matrix(
                        JAXBackend(), x_grid,
                        ell,
                        eta,
                        jitter=0.0,
                    )

                K = K + 1e-6 * jnp.eye(N)

                if FLAGS.init_GP == "zeros":
                    f_nodes_init = jnp.zeros(N)

                elif FLAGS.init_GP == "polexp":
                    raise NotImplementedError(
                        "init_GP='polexp' needs an exact JAX port from the PyMC init block; "
                        "do not approximate it here."
                    )

                else:
                    raise ValueError(f"Unknown init_GP: {FLAGS.init_GP}")

                L = jnp.linalg.cholesky(K)

                u_nodes = jax.scipy.linalg.solve_triangular(
                    L,
                    f_nodes_init,
                    lower=True,
                )

                key = jax.random.PRNGKey(FLAGS.seed)
                noise = 1e-6 * jax.random.normal(key, shape=u_nodes.shape)

                init_vals["f_rotated"] = u_nodes + noise
                init_vals["ell"] = float(ell)
                init_vals["eta"] = float(eta)

                print("GP init done.")

    # Remove None before converting. Then force all init values to real float64.
    init_vals = {k: v for k, v in init_vals.items() if v is not None}

    init_vals = {
        k: jnp.asarray(v, dtype=jnp.float64)
        for k, v in init_vals.items()
    }

 
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
    
    if FLAGS.is_GP_dL and "f_rotated" in init_vals:
        expected_gp = (len(model_init_info["z_gp_nodes"]),)
        if tuple(init_vals["f_rotated"].shape) != expected_gp:
            raise ValueError(f"f_rotated initialization has shape {init_vals['f_rotated'].shape}; "
                             f"the selected GP grid requires {expected_gp}")
    if not FLAGS.pop_only and "x" in init_vals:
        expected_x = (int(lik_data.Nobs), int(lik_data.mus_s.shape[1]))
        if tuple(init_vals["x"].shape) != expected_x:
            raise ValueError(f"Event initialization x has shape {init_vals['x'].shape}; "
                             f"selected events and coordinates require {expected_x}")

    # --- Initialize (lets you plug in mass matrix if you have one) ---
    # Use the supplied initialization; report failures without changing the point.
    
    init_strategy = init_to_value(values=init_vals)

    try:
        init_key, rng_key = random.split(rng_key)
        res = initialize_model(
            init_key,
            model_numpyro,
            init_strategy=init_strategy,
            dynamic_args=False,
            #validate_grad=False,   # key
        )
        z = res.param_info.z
        # pe = res.potential_fn(z)
        # g = jax.grad(res.potential_fn)(z)

        # print("PE:", pe, "finite:", jnp.isfinite(pe))

        # for k, v in g.items():
        #     print(k, "grad finite:", bool(jnp.all(jnp.isfinite(v))), "grad:", v)

   
        print("✅ init_to_value; initial values used.")
        
    except Exception as e:
        
        print(e)
        print("Initialization failed; inspect supplied initial values and model support.")
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
  





    def _unpack_initialize_model_result(res_local):
        """Return (param_info_or_params, potential_fn, postprocess_fn, model_trace)."""
        if isinstance(res_local, tuple):
            if len(res_local) == 4:
                return res_local
            if len(res_local) == 3:
                a, b, c = res_local
                if callable(b):
                    return a, b, None, c
                raise RuntimeError(
                    f"initialize_model returned 3-tuple but second element is not callable. "
                    f"Types: {[type(x) for x in res_local]}"
                )
        raise RuntimeError(f"Unexpected initialize_model result: {type(res_local)}")

    def _param_info_to_z(param_info_or_params):
        """NumPyro versions may return ParamInfo(z, pe, grad) or raw params."""
        return param_info_or_params.z if hasattr(param_info_or_params, "z") else param_info_or_params

    def _support_kind(site):
        s = site["fn"].support
        s_str = str(s).lower()
        if "simplex" in s_str:
            return "simplex"
        if "positive" in s_str or "greaterthan(lower_bound=0" in s_str:
            return "positive"
        if hasattr(s, "lower_bound") and hasattr(s, "upper_bound"):
            lb = np.asarray(s.lower_bound)
            ub = np.asarray(s.upper_bound)
            if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
                return "interval"
        return "real"

    def _jitter_init_value_for_site(name, value, key, eps):
        """Small constrained-space perturbation for a single sample-site init value."""
        v = jnp.asarray(value)
        if eps <= 0:
            return v

        site = model_trace.get(name, None)
        kind = _support_kind(site) if site is not None else "real"

        if kind == "simplex":
            vv = jnp.clip(v, 1e-12, 1.0)
            y = jnp.log(vv)
            y = y + eps * jax.random.normal(key, shape=vv.shape, dtype=vv.dtype)
            return jax.nn.softmax(y)

        if kind == "positive":
            return jnp.clip(v, 1e-300) * jnp.exp(eps * jax.random.normal(key, shape=v.shape, dtype=v.dtype))

        if kind == "interval":
            s = site["fn"].support
            lb = jnp.asarray(s.lower_bound, dtype=v.dtype)
            ub = jnp.asarray(s.upper_bound, dtype=v.dtype)
            margin = jnp.asarray(1e-8, dtype=v.dtype)
            t = jnp.clip((v - lb) / (ub - lb), margin, 1.0 - margin)
            y = jnp.log(t) - jnp.log1p(-t)
            y = y + eps * jax.random.normal(key, shape=v.shape, dtype=v.dtype)
            return lb + (ub - lb) * jax.nn.sigmoid(y)

        # real-valued unconstrained or normal sites: additive perturbation
        return v + eps * jax.random.normal(key, shape=v.shape, dtype=v.dtype)

    def _jitter_init_values(init_vals_base, key, eps):
        """Return one independently jittered init_vals dictionary."""
        keys_local = jax.random.split(key, max(1, len(init_vals_base)))
        out = {}
        for kk, (name, value) in zip(keys_local, init_vals_base.items()):
            # Only perturb actual sampled sites. Extra deterministic/user entries are passed through.
            if name in sample_sites:
                out[name] = _jitter_init_value_for_site(name, value, kk, eps)
            else:
                out[name] = jnp.asarray(value)
        return out

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


    # Lambda sites = all sampled sites except the big latents
    big_latents = {'x', 'f_rotated', 'beta', 'eps1', 'eps2', 'eps1_raw', 'eps2_raw', 'mulMc', 'mulq', 'mulMc_raw', 'mulq_raw', 'q_mu', 'q_mu_raw', 'beta_w2', 'raw_w2', 'w2', 'rho_u'}



    lambda_sites = [
        n for n in sample_sites
        if n not in big_latents
    ]


    
    if FLAGS.dense_mass:
        # NumPyro expects a list of tuples (each tuple is one dense block)
        dense_blocks = [tuple(lambda_sites)] if len(lambda_sites) > 1 else False
        print("[INFO] dense_mass blocks:", dense_blocks)
        dense_blocks_plot = dense_blocks
    else:
        dense_blocks = []
        dense_blocks_plot = [tuple(lambda_sites)] if len(lambda_sites) > 1 else False

    print()

    def _gp_state_at_lambda(Lam):
        """Rebuild distances for Lambda diagnostics at the initialized GP function."""
        if not FLAGS.is_GP_dL:
            return None
        from backends import JAXBackend
        from cosmology import z_from_dL_at_monotone
        grid = z_from_dL_at_monotone(
            JAXBackend(), jnp.asarray([1.0], dtype=jnp.float64),
            z_nodes=jnp.asarray(model_init_info["z_gp_nodes"], dtype=jnp.float64),
            f_nodes=_trace_value(model_trace, "f_nodes"),
            H0=Lam[0], Om=Lam[1], w0=Lam[2],
            z_fine=jnp.asarray(model_init_info["z_gp_fine"], dtype=jnp.float64),
            eps_q=1e-12 if FLAGS.eps_DE < 0 else FLAGS.eps_DE,
            integrate_dc=FLAGS.integrate_dc,
        )
        return {key: grid[key] for key in ("z_fine", "dL_gw_fine", "log_ddL_dz_fine")}

    if FLAGS.check_init:
    
        print()
        print("=" * 100)
        print("CHECK_INIT: likelihood decomposition at initialized point")
        print("=" * 100)
    
        Lambda0 = _trace_value(model_trace, "Lambda")

        if not FLAGS.pop_only:
            x0 = _trace_value(model_trace, "x")
    
        try:
            lR0_0 = _trace_value(model_trace, "lR0")
        except KeyError:
            lR0_0 = jnp.asarray(0.0, dtype=jnp.float64)



        print("\nCHECK_INIT: constrained sample/deterministic sites")
        print("-" * 100)
        
        for name, site in model_trace.items():
            if site.get("type") not in ("sample", "deterministic"):
                continue
        
            val = np.asarray(jax.device_get(site["value"]))
        
            # avoid dumping huge event latent x fully
            if name == "x":
                print(
                    f"{name:25s}: shape={val.shape} "
                    f"min={np.nanmin(val): .6g} max={np.nanmax(val): .6g} "
                    f"mean={np.nanmean(val): .6g}"
                )
            else:
                print(f"{name:25s}: shape={val.shape} value={val}")
        
        print("-" * 100)

        try:
            log_lik_var_min_used = _trace_value(model_trace, "log_lik_var_min_used")
            log_lik_var_sg_used = _trace_value(model_trace, "log_lik_var_sg_used")
            passes_gate_used = _trace_value(model_trace, "passes_log_lik_var_gate_used")
        
            print("\nCHECK_INIT: variance gate values actually used in model")
            print("-" * 100)
            print("log_lik_var_min_used       :", np.asarray(jax.device_get(log_lik_var_min_used)))
            print("log_lik_var_sg_used        :", np.asarray(jax.device_get(log_lik_var_sg_used)))
            print("passes_log_lik_var_gate    :", np.asarray(jax.device_get(passes_gate_used)))
            print("-" * 100)
        except KeyError:
            pass
        
        Lambda0 = _trace_value(model_trace, "Lambda")
        Lambda0_np = np.asarray(jax.device_get(Lambda0))
        
        print("\nCHECK_INIT: packed Lambda")
        print("-" * 100)
        print(f"Lambda shape: {Lambda0_np.shape}")
        print(Lambda0_np)
        print("-" * 100)

        if not FLAGS.pop_only:
            comp = loglik_components(Lambda0, x0, lR0=lR0_0, gp_state=_gp_state_at_lambda(Lambda0))
        else:
            comp = loglik_components(Lambda0, lR0=lR0_0, gp_state=_gp_state_at_lambda(Lambda0))

        dump_path = os.path.join(FLAGS.fout, "numpyro_check_init_likelihood_dump.npz")
        np.savez(
            dump_path,
            **{k: np.asarray(jax.device_get(v)) for k, v in comp.items()}
        )
        print(f"Saved NumPyro check-init likelihood dump to: {dump_path}")
        
        point_path = os.path.join(FLAGS.fout, "numpyro_check_init_point.npz")
        if not FLAGS.pop_only:
            np.savez(
                point_path,
                Lambda=np.asarray(jax.device_get(Lambda0)),
                x=np.asarray(jax.device_get(x0)),
                lR0=np.asarray(jax.device_get(lR0_0)),
            )
        else:
            np.savez(
                point_path,
                Lambda=np.asarray(jax.device_get(Lambda0)),
                lR0=np.asarray(jax.device_get(lR0_0)),
            )
        print(f"Saved NumPyro check-init point to: {point_path}")

    
        keys_main = [
            "ll",
            "ll_evt",
            "log_mu",
            "sel_term",
            "Nobs_sel_term",
            "log_var_sel_u",
            "var_log_lik_sel",
            "var_log_lik_evs",
            "var_log_lik",
            "n_bad_logp_pop_evt",
            "n_bad_log_PE_prior_evt",
            "n_bad_log_jac_evt",
            "min_logp_pop_evt",
            "max_logp_pop_evt",
            "min_log_PE_prior_evt",
            "max_log_PE_prior_evt",
            "min_log_jac_evt",
            "max_log_jac_evt",
        ]
    
        for k in keys_main:
            if k in comp:
                v = np.asarray(jax.device_get(comp[k]))
                print(f"{k:25s}: shape={v.shape} value={v}")
    
        print("-" * 100)
    
        #ll0, var0 = loglik(Lambda0, x0, lR0=lR0_0)

        if not FLAGS.pop_only:
            ll0, var0 = loglik(Lambda0, x0, lR0=lR0_0, gp_state=_gp_state_at_lambda(Lambda0))
        else:
            ll0, var0 = loglik(Lambda0, lR0=lR0_0, gp_state=_gp_state_at_lambda(Lambda0))

    
        ll0 = np.asarray(jax.device_get(ll0))
        var0 = np.asarray(jax.device_get(var0))
        comp_ll = np.asarray(jax.device_get(comp["ll"]))
        comp_var = np.asarray(jax.device_get(comp["var_log_lik"]))
    
        print("consistency ll          :", ll0, comp_ll, "diff=", ll0 - comp_ll)
        print("consistency var_log_lik :", var0, comp_var, "diff=", var0 - comp_var)
    
        print("-" * 100)
        print("finite ll              :", np.isfinite(ll0))
        print("finite var_log_lik     :", np.isfinite(var0))
        print("passes var gate        :", bool(var0 <= FLAGS.log_lik_var_min))
        print("log_lik_var_min        :", FLAGS.log_lik_var_min)
        print("=" * 100)
        print()

        print("\nCHECK_INIT: gradient diagnostics at initialized point")
        print("-" * 100)
        
        def scalar_ll_from_Lambda(Lam):
            if not FLAGS.pop_only:
                c = loglik_components(Lam, x0, lR0=lR0_0, gp_state=_gp_state_at_lambda(Lam))
            else:
                c = loglik_components(Lam,lR0=lR0_0, gp_state=_gp_state_at_lambda(Lam))
            return c["ll"]
        
        def scalar_sel_from_Lambda(Lam):
            if not FLAGS.pop_only:
                c = loglik_components(Lam, x0, lR0=lR0_0, gp_state=_gp_state_at_lambda(Lam))
            else:
                c = loglik_components(Lam, lR0=lR0_0, gp_state=_gp_state_at_lambda(Lam))
            return -c["Nobs_sel_term"]
        
        def scalar_evt_from_Lambda(Lam):
            if not FLAGS.pop_only:
                c = loglik_components(Lam, x0, lR0=lR0_0, gp_state=_gp_state_at_lambda(Lam))
            else:
                c = loglik_components(Lam, lR0=lR0_0, gp_state=_gp_state_at_lambda(Lam))
            return c["ll_evt"]
        
        g_ll = jax.grad(scalar_ll_from_Lambda)(Lambda0)
        g_sel = jax.grad(scalar_sel_from_Lambda)(Lambda0)
        g_evt = jax.grad(scalar_evt_from_Lambda)(Lambda0)
        
        for name, g in [
            ("grad total ll", g_ll),
            ("grad event ll", g_evt),
            ("grad selection ll", g_sel),
        ]:
            gg = np.asarray(jax.device_get(g))
            print(
                f"{name:20s}: "
                f"finite={np.all(np.isfinite(gg))} "
                f"norm={np.linalg.norm(gg):.6g} "
                f"maxabs={np.max(np.abs(gg)):.6g} "
                f"argmax={np.argmax(np.abs(gg))}"
            )
        
        print("-" * 100)
        
        slices = {
            name: sl for name, sl in _lambda_slices_for_benchmark(FLAGS).items()
            if isinstance(sl, slice) and sl.stop > sl.start
        }

        if not FLAGS.pop_only:
            for block, sl in slices.items():
                for name, g in [
                    ("total", g_ll),
                    ("event", g_evt),
                    ("selection", g_sel),
                ]:
                    gg = np.asarray(jax.device_get(g[sl]))
                    print(
                        f"{block:14s} {name:10s}: "
                        f"norm={np.linalg.norm(gg):.6g} "
                        f"maxabs={np.max(np.abs(gg)):.6g}"
                    )
        else:
            for block, sl in slices.items():
                for name, g in [
                    ("total", g_ll),
                    ("event", g_evt),
                    ("selection", g_sel),
                ]:
                    gg = np.asarray(jax.device_get(g[sl]))
                    print(
                        f"{block:14s} {name:10s}: "
                        f"norm={np.linalg.norm(gg):.6g} "
                        #f"maxabs={np.max(np.abs(gg)):.6g}"
                    )
        
        print("=" * 100)

        print("\nCHECK_INIT: NumPyro unconstrained potential gradient")
        print("-" * 100)
        
        # Correct source of unconstrained init params.
        # Do NOT use model_trace here.
        z0 = res.param_info.z
        
        pe0, gz0 = jax.value_and_grad(res.potential_fn)(z0)
        
        print("potential_energy:", np.asarray(jax.device_get(pe0)))
        print("finite PE       :", np.isfinite(np.asarray(jax.device_get(pe0))))
        
        for k, g in gz0.items():
            gg = np.asarray(jax.device_get(g))
            print(
                f"{k:25s}: shape={gg.shape} "
                f"finite={np.all(np.isfinite(gg))} "
                f"norm={np.linalg.norm(gg):.6g} "
                f"maxabs={np.max(np.abs(gg)):.6g}"
            )
        
        print("-" * 100)

        print("\nCHECK_INIT: factor log_probs")
        print("-" * 100)
        
        factor_total = 0.0
        
        for name, site in model_trace.items():
            if site.get("type") != "sample":
                continue
        
            is_observed = site.get("is_observed", False)
            fn = site.get("fn", None)
            val = site.get("value", None)
        
            if fn is None or val is None:
                continue
        
            lp = fn.log_prob(val)
            lp_sum = jnp.sum(lp)
        
            lp_np = np.asarray(jax.device_get(lp))
            lp_sum_np = np.asarray(jax.device_get(lp_sum))
        
            print(
                f"{name:25s}: observed={is_observed} "
                f"shape={lp_np.shape} "
                f"sum={lp_sum_np} "
                f"finite={np.all(np.isfinite(lp_np))}"
            )
        
            factor_total = factor_total + lp_sum
        
        print("-" * 100)
        print("sum sample/factor log_probs:", np.asarray(jax.device_get(factor_total)))
        print("minus sum log_probs        :", np.asarray(jax.device_get(-factor_total)))
        print("=" * 100)

        if FLAGS.mass_model=='DPUC':

            print("\nCHECK_INIT: finite-difference sensitivity of sig2 block")
            print("-" * 100)
            
            Lambda0 = _trace_value(model_trace, "Lambda")
            if not FLAGS.pop_only:
                x0 = _trace_value(model_trace, "x")
            lR0_0 = _trace_value(model_trace, "lR0")
            
            mass_start = slices["mass"].start
            K = int(FLAGS.N_DP_comp_max)
            sig1_slice = slice(mass_start + 2 * K, mass_start + 3 * K)
            sig2_slice = slice(mass_start + 3 * K, mass_start + 4 * K)
            
            def comp_at(Lam):
                if not FLAGS.pop_only:
                    c = loglik_components(Lam, x0, lR0=lR0_0, gp_state=_gp_state_at_lambda(Lam))
                else:
                    c = loglik_components(Lam, lR0=lR0_0, gp_state=_gp_state_at_lambda(Lam))
                return {
                    "ll": c["ll"],
                    "ll_evt": c["ll_evt"],
                    "log_mu": c["log_mu"],
                    "Nobs_sel_term": c["Nobs_sel_term"],
                    "var_log_lik": c["var_log_lik"],
                }
            
            c0 = comp_at(Lambda0)
            
            for frac in [1e-3, 3e-3, 1e-2, 3e-2, 5e-2]:
                for sign in [-1.0, +1.0]:
                    Lam = Lambda0.at[sig2_slice].multiply(1.0 + sign * frac)
                    c = comp_at(Lam)
            
                    print(
                        f"sig2 scale {1.0 + sign * frac: .6f} | "
                        f"dll={float(c['ll'] - c0['ll']): .6g} | "
                        f"dll_evt={float(c['ll_evt'] - c0['ll_evt']): .6g} | "
                        f"dlog_mu={float(c['log_mu'] - c0['log_mu']): .6g} | "
                        f"dNobs_sel={float(c['Nobs_sel_term'] - c0['Nobs_sel_term']): .6g} | "
                        f"var={float(c['var_log_lik']): .6g}"
                    )
            
            print("-" * 100)
    
            for label, sl in [("sig1", sig1_slice), ("sig2", sig2_slice)]:
                print(f"\n{label} scale sensitivity")
                c0 = comp_at(Lambda0)
            
                for frac in [1e-3, 3e-3, 1e-2, 3e-2, 5e-2]:
                    for sign in [-1.0, +1.0]:
                        Lam = Lambda0.at[sl].multiply(1.0 + sign * frac)
                        c = comp_at(Lam)
            
                        print(
                            f"{label} scale {1.0 + sign * frac: .6f} | "
                            f"dll={float(c['ll'] - c0['ll']): .6g} | "
                            f"dll_evt={float(c['ll_evt'] - c0['ll_evt']): .6g} | "
                            f"dlog_mu={float(c['log_mu'] - c0['log_mu']): .6g} | "
                            f"dNobs_sel={float(c['Nobs_sel_term'] - c0['Nobs_sel_term']): .6g} | "
                            f"var={float(c['var_log_lik']): .6g}"
                        )
            print("=" * 100)
    
    ##########################################################################
    ##########################################################################
    # Profile if requested 
    ##########################################################################
    ##########################################################################

    if FLAGS.profile:
    
    
        run_selection_block_benchmark(
            FLAGS, priors, params_fix, model_trace, lik_data, core,
            repeats=max(3, int(FLAGS.profile)),
            gp_state_factory=_gp_state_at_lambda,
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
        target_accept_prob = float(FLAGS.target_accept),
        init_strategy = init_strategy,
        max_tree_depth = int(FLAGS.max_tree_depth),
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






    init_params_for_run = None
    jitter_eps = float(FLAGS.init_per_chain_jitter)

    if jitter_eps > 0.0 and num_chains > 1:
        print()
        print("[INFO] Building per-chain initial parameters from --ivals/init_vals")
        print(f"[INFO] init_per_chain_jitter = {jitter_eps:.6g}")
        print("[INFO] Perturbations are applied in constrained space, then converted by initialize_model.")

        chain_keys = random.split(rng_key, num_chains + 1)
        rng_key = chain_keys[0]
        init_params_list = []

        for c in range(num_chains):
            vals_c = _jitter_init_values(init_vals, chain_keys[c + 1], jitter_eps)
            init_strategy_c = init_to_value(values=vals_c)
            res_c = initialize_model(
                chain_keys[c + 1],
                model_numpyro,
                init_strategy=init_strategy_c,
                dynamic_args=False,
            )
            init_params_c, _, _, _ = _unpack_initialize_model_result(res_c)
            init_params_list.append(_param_info_to_z(init_params_c))

        init_params_for_run = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0),
            *init_params_list,
        )
        print("[INFO] Per-chain init_params built and will be passed to mcmc.run(init_params=...).")

    elif jitter_eps > 0.0 and num_chains == 1:
        print()
        print("[INFO] Building jittered single-chain initial parameters from --ivals/init_vals")
        print(f"[INFO] init_per_chain_jitter = {jitter_eps:.6g}")
        print("[INFO] Perturbations are applied in constrained space, then converted by initialize_model.")

        jitter_key, rng_key = random.split(rng_key)
        vals_0 = _jitter_init_values(init_vals, jitter_key, jitter_eps)
        init_strategy_0 = init_to_value(values=vals_0)
        res_0 = initialize_model(
            jitter_key,
            model_numpyro,
            init_strategy=init_strategy_0,
            dynamic_args=False,
        )
        init_params_0, _, _, _ = _unpack_initialize_model_result(res_0)
        init_params_for_run = _param_info_to_z(init_params_0)
        print("[INFO] Jittered single-chain init_params built and will be passed to mcmc.run(init_params=...).")

    elif num_chains > 1:
        print("[INFO] init_per_chain_jitter <= 0: chains use the same init_to_value strategy values where provided.")
    else:
        print("[INFO] init_per_chain_jitter <= 0: chain uses init_to_value strategy values where provided.")


    mcmc = MCMC(
        nuts,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=True,
    )
    
    run_key, rng_key = random.split(rng_key)
    run_kwargs = dict(extra_fields=(
        "diverging", "num_steps", "accept_prob", "adapt_state.step_size",
    ))
    if init_params_for_run is not None:
        run_kwargs["init_params"] = init_params_for_run
    elif not FLAGS.is_GP_dL:
        initial_z = _param_info_to_z(init_params)
        if num_chains > 1:
            initial_z = jax.tree_util.tree_map(
                lambda value: jnp.broadcast_to(value, (num_chains,) + value.shape), initial_z)
        run_kwargs["init_params"] = initial_z
    mcmc.run(run_key, **run_kwargs)

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

    vplot = dense_blocks_plot[0]

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
    
    
