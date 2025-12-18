#!/usr/bin/env python
import os
import argparse
import numpy as np
import json

# ----------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------
# Example:
from utils_train import compile_sel_bias_fn
from train_emulator import sample_lambda_mixture, eval_sel_batch, sample_Lambda_lhs, build_initial_design
import data_tools as dt
#
# ----------------------------------------------------------------------

def make_frozen_testset(
    priors,
    ordered_keys,
    rate_model,
    mass_model,
    spin_model,
    m1inj, m2inj, dLinj,
    chi1Inj, chi2Inj, cost1Inj, cost2Inj,
    lpdinj, Ngen,
    test_path,
    n_test=10000,
    seed=9999,
    overwrite=False,
    test_strategy="mixture",   # "mixture" or "spacefill"
    frac_lhs=0.6, frac_edge=0.25, frac_stress=0.15,  # only for spacefill
):
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    if os.path.exists(test_path) and not overwrite:
        print(f"[make-test] exists, skipping (use --overwrite): {test_path}")
        return

    print("[make-test] compiling selection function...")
    sel_fn = compile_sel_bias_fn(
        rate_model=rate_model,
        mass_model=mass_model,
        spin_model=spin_model,
        smoothing="poly",
        simplex_repair=False,
        has_m2_break=False,
        interp=False,
        param="vanilla",
        use_float32=False,
    )
    print("[make-test] sel_fn compiled")

    print(f"[make-test] sampling frozen test set: n_test={n_test} seed={seed} strategy={test_strategy}")
    if test_strategy == "mixture":
        Lambda0 = sample_lambda_mixture(priors, ordered_keys, n_test, seed=seed)
    elif test_strategy == "spacefill":
        Lambda0 = build_initial_design(
            priors, ordered_keys,
            n_total=n_test, seed=seed,
            frac_lhs=frac_lhs, frac_edge=frac_edge, frac_stress=frac_stress
        )
    else:
        raise ValueError("test_strategy must be 'mixture' or 'spacefill'")

    lm, nf, lv, ok = eval_sel_batch(
        sel_fn, Lambda0,
        m1inj, m2inj, dLinj,
        chi1Inj, chi2Inj, cost1Inj, cost2Inj,
        lpdinj, Ngen
    )

    Lambda_test = Lambda0[ok]
    np.savez(
        test_path,
        Lambda_test=Lambda_test,
        log_mu_true_test=lm,
        log_var_true_test=lv,
        neff_true_test=nf,
        strategy=test_strategy,
        seed=seed,
        n_test_requested=n_test,
        frac_lhs=frac_lhs, frac_edge=frac_edge, frac_stress=frac_stress,
    )
    print(f"[make-test] saved: N={Lambda_test.shape[0]} -> {test_path}")

def generate_dataset_shards(
    priors,
    ordered_keys,
    rate_model,
    mass_model,
    spin_model,
    m1inj, m2inj, dLinj,
    chi1Inj, chi2Inj, cost1Inj, cost2Inj,
    lpdinj, Ngen,
    out_dir,
    total_points,
    shard_size,
    seed,
):
    os.makedirs(out_dir, exist_ok=True)

    print("[generate] compiling selection function...")
    sel_fn = compile_sel_bias_fn(
        rate_model=rate_model,
        mass_model=mass_model,
        spin_model=spin_model,
        smoothing="poly",
        simplex_repair=False,
        has_m2_break=False,
        interp=False,
        param="vanilla",
        use_float32=False,
    )
    print("[generate] sel_fn compiled")

    n_shards = int(np.ceil(total_points / shard_size))

    for s in range(n_shards):
        shard_path = os.path.join(out_dir, f"dataset_shard_{s:04d}.npz")
        if os.path.exists(shard_path):
            print(f"[generate] shard exists, skipping: {shard_path}")
            continue

        n = shard_size if s < n_shards - 1 else total_points - s * shard_size
        shard_seed = seed + 10000 + s

        print(f"[generate] shard {s+1}/{n_shards}: sampling {n} Lambdas")
        #X0 = sample_lambda_mixture(priors, ordered_keys, n, seed=shard_seed)
        X0 = build_initial_design(priors, ordered_keys, n, seed=shard_seed, frac_lhs=0.6,
                                frac_edge=0.25,
                                frac_stress=0.15,)


        lm, nf, lv, ok = eval_sel_batch(
            sel_fn, X0,
            m1inj, m2inj, dLinj,
            chi1Inj, chi2Inj, cost1Inj, cost2Inj,
            lpdinj, Ngen
        )

        X = X0[ok]

        np.savez(
            shard_path,
            X=X,
            log_mu=lm,
            log_var=lv,
            neff=nf,
        )

        print(f"[generate] saved {X.shape[0]} points -> {shard_path}")

    print("[generate] done")


    

def merge_dataset_shards(shard_dir, merged_path):
    files = sorted(
        f for f in os.listdir(shard_dir)
        if f.startswith("dataset_shard_") and f.endswith(".npz")
    )
    if not files:
        raise RuntimeError(f"No shard files found in {shard_dir}")

    Xs, mus, vars_, neffs = [], [], [], []

    for f in files:
        path = os.path.join(shard_dir, f)
        z = np.load(path)
        Xs.append(z["X"])
        mus.append(z["log_mu"])
        vars_.append(z["log_var"])
        neffs.append(z["neff"])

    X = np.concatenate(Xs, axis=0)
    log_mu = np.concatenate(mus, axis=0)
    log_var = np.concatenate(vars_, axis=0)
    neff = np.concatenate(neffs, axis=0)

    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    np.savez(merged_path, X=X, log_mu=log_mu, log_var=log_var, neff=neff)
    print(f"[merge] merged {X.shape[0]} points -> {merged_path}")

# ----------------------------------------------------------------------
# MAIN (CLI ENTRY POINT)
# ----------------------------------------------------------------------

def main(priors, ordered_keys, rate_model, mass_model, spin_model, m1inj, m2inj, dLinj, chi1Inj, chi2Inj, cost1Inj,cost2Inj, lpdinj, Ngen ):
    
    p = argparse.ArgumentParser(description="Generate and/or merge selection-emulator dataset shards.")
    p.add_argument("--mode", choices=["generate", "merge", "both", "make-test"], default="both",
               help="What to do. 'merge' will NOT generate new shards. 'make-test' only creates frozen test set.")
    p.add_argument("--out-dir", required=True, help="Directory containing dataset_shard_*.npz")
    p.add_argument("--merged-path", default=None, help="Output path for merged dataset.npz")

    # generation args
    p.add_argument("--total-points", type=int, default=20000)
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-path", default=None, help="Output NPZ path for frozen test set (make-test mode).")
    p.add_argument("--n-test", type=int, default=10000, help="Number of test points for frozen test set.")
    p.add_argument("--test-seed", type=int, default=9999, help="Seed for frozen test set sampling.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs (test set).")


    args = p.parse_args()

    # MERGE only
    if args.mode == "merge":
        if args.merged_path is None:
            raise ValueError("--merged-path is required for --mode merge")
        merge_dataset_shards(args.out_dir, args.merged_path)
        return


    # MAKE TESTSET only
    if args.mode == "make-test":
        # if args.config_module is None:
        #     raise ValueError("--config-module is required for --mode make-test")
        if args.test_path is None:
            raise ValueError("--test-path is required for --mode make-test")

        # print(f"[config] loading {args.config_module}")
        # cfg = __import__(args.config_module, fromlist=["*"])

        make_frozen_testset(
            priors=priors,
            ordered_keys=ordered_keys,
            rate_model=rate_model,
            mass_model=mass_model,
            spin_model=spin_model,
            m1inj=m1inj,
            m2inj=m2inj,
            dLinj=dLinj,
            chi1Inj=chi1Inj,
            chi2Inj=chi2Inj,
            cost1Inj=cost1Inj,
            cost2Inj=cost2Inj,
            lpdinj=lpdinj,
            Ngen=Ngen,
            test_path=args.test_path,
            n_test=args.n_test,
            seed=args.test_seed,
            overwrite=args.overwrite,
        )
        return

    # ------------------------------------------------------------------
    # Load user configuration
    # ------------------------------------------------------------------
    #print(f"[config] loading {args.config_module}")
    #cfg = __import__(args.config_module, fromlist=["*"])

    generate_dataset_shards(
        priors=priors,
        ordered_keys=ordered_keys,
        rate_model=rate_model,
        mass_model=mass_model,
        spin_model=spin_model,
        m1inj=m1inj,
        m2inj=m2inj,
        dLinj=dLinj,
        chi1Inj=chi1Inj,
        chi2Inj=chi2Inj,
        cost1Inj=cost1Inj,
        cost2Inj=cost2Inj,
        lpdinj=lpdinj,
        Ngen=Ngen,
        out_dir=args.out_dir,
        total_points=args.total_points,
        shard_size=args.shard_size,
        seed=args.seed,
    )

    if args.mode in ("both",) and args.merged_path is not None:
        merge_dataset_shards(args.out_dir, args.merged_path)

if __name__ == "__main__":



    # define priors, ordered_keys, models, injections
    print("Loading prior files...")
    
    fin_priors='../pymcpop-gw/priors_files/priors_GWTC4_DPLDP_SS.json'
    
    with open(fin_priors) as json_file:
            priors = json.load(json_file)
    
    priors['u']  = (0,1)
    priors['v']  = (0,1)
    
    mass_model='DPLDP'
    spin_model='default_gauss'
    rate_model='MD'
    
    
    priors['w0']  = (-1.5, -0.5)
    
    priors["epsilon"]  = (1e-02, )
    priors["sig_g_low"]  = (1e-02, )
    priors["sig_g_high"]  = (1e-02, )


    print("Priors:")
    print(priors)

    
    
    ordered_keys = ordered_keys = [
      "H0","Om", "w0","Xi0","nXi0",
      "gamma","kappa","zp",
      "muChi", "sigmaChi", "zeta", "sigmat",
      # DPLDP mass params (20):
      "alpha1","alpha2","mb","mu1","sigma1","mu2","sigma2","m1_low","m_high","delta_m1",
      "lambda0","lambda1",
      "beta","m2_low","delta_m2","epsilon","mu_g","w_g","sig_g_low","sig_g_high"
    ]

    
    rate_model = 'MD'
    mass_model = 'DPLDP'
    spin_model = 'default_gauss'


    
    fin_injections = [ '/Users/Michi/Library/CloudStorage/Dropbox/Local/Physics_projects/GWOSC/O4a/injections_mixture-semi_o1_o2-real_o3_o4a-cartesian_spins_20250503134659UTC_' ]


    print("Loading injection files from %s"%fin_injections[0])
    
    
    injections = dt.load_injections(fin_injections, allPercUse=[1.])
    
    
    dLinj = injections['dL'][0]
    m1inj = injections['m1d'][0]
    m2inj = injections['m2d'][0]
    lpdinj = injections['log_wt'][0]
    Ngen = injections['Ngen'][0]
    Ndet = injections['Ndet'][0]
    
    if (spin_model=='default') or (spin_model=='default_gauss'):
    
        print("Computing chi1, chi2, cost1, cost2 in injections...")
        
        chi1Inj = np.sqrt(injections['spin1x']**2+injections['spin1y']**2+injections['spin1z']**2)
        chi2Inj = np.sqrt(injections['spin2x']**2+injections['spin2y']**2+injections['spin2z']**2)
    
        cost1Inj = injections['spin1z']/chi1Inj
        cost2Inj = injections['spin2z']/chi2Inj

        chi1Inj = chi1Inj[0]
        chi2Inj = chi2Inj[0]
    
        cost1Inj = cost1Inj[0]
        cost2Inj = cost2Inj[0]


    print("Injections loaded.")



    
    main(priors, ordered_keys, rate_model, mass_model, spin_model, m1inj, m2inj, dLinj, chi1Inj, chi2Inj, cost1Inj,cost2Inj, lpdinj, Ngen)



