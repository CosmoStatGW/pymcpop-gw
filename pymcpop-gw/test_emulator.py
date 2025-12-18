import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils_train import (
    sample_Lambda_from_priors,
    sample_edge_points,
    compile_sel_bias_fn,
)

# ----------------------------
# Model definition must match training
# ----------------------------
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden=256, n_hidden=4):
        super().__init__()
        layers = []
        dim = d_in
        for _ in range(n_hidden):
            layers += [nn.Linear(dim, d_hidden), nn.SiLU()]
            dim = d_hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(dim, 2)  # [log_mu, log_var]

    def forward(self, x):
        return self.head(self.net(x))


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def sample_lambda_mixture(priors, ordered_keys, n, seed=0,
                         frac_prior=0.7, frac_edges=0.2, frac_stress=0.1):
    """
    Independent, frozen test-mixture sampler.
    Uses your prior sampler + edge sampler.
    """
    rng = np.random.default_rng(seed)
    n_prior = int(round(n * frac_prior))
    n_edges = int(round(n * frac_edges))
    n_stress = n - n_prior - n_edges

    Ls = []

    # prior
    Ls.append(sample_Lambda_from_priors(priors, ordered_keys, rng=rng, n=n_prior))

    # edges
    edge_block = sample_edge_points(
        priors, ordered_keys,
        n_random=max(1, n_edges // 10),
        include_edges=True,
        baseline="mid",
        seed=seed + 11
    )
    if edge_block.shape[0] >= n_edges:
        Ls.append(edge_block[:n_edges])
    else:
        Ls.append(edge_block)
        Ls.append(sample_Lambda_from_priors(priors, ordered_keys, rng=rng, n=(n_edges - edge_block.shape[0])))

    # stress: more edges + random baselines
    stress_edges = sample_edge_points(
        priors, ordered_keys,
        n_random=max(1, (n_stress // 2) // 10),
        include_edges=True,
        baseline="random",
        seed=seed + 29
    )
    n_se = min(stress_edges.shape[0], n_stress // 2)
    Ls.append(stress_edges[:n_se])
    Ls.append(sample_Lambda_from_priors(priors, ordered_keys, rng=rng, n=(n_stress - n_se)))

    out = np.concatenate(Ls, axis=0)
    rng.shuffle(out, axis=0)
    return out


def eval_sel_batch(sel_fn, Lambdas,
                   m1inj, m2inj, dLinj,
                   chi1Inj, chi2Inj, cost1Inj, cost2Inj,
                   lpdinj, Ngen):
    """
    Exact evaluation of sel_fn on a batch of Lambdas.
    Returns arrays of shape (B,) after filtering non-finite rows.
    """
    B = Lambdas.shape[0]
    log_mu = np.empty(B, dtype=np.float64)
    neff   = np.empty(B, dtype=np.float64)
    log_var= np.empty(B, dtype=np.float64)
    zeros = np.zeros(lpdinj.shape, dtype=lpdinj.dtype)

    for i in range(B):
        lm, n, lv = sel_fn(
            m1inj, m2inj, dLinj,
            chi1Inj, chi2Inj, cost1Inj, cost2Inj,
            lpdinj, np.squeeze(Lambdas[i]),
            Ngen, zeros
        )
        log_mu[i] = float(lm)
        neff[i]   = float(n)
        log_var[i]= float(lv)

    ok = np.isfinite(log_mu) & np.isfinite(log_var) & np.isfinite(neff)
    return log_mu[ok], neff[ok], log_var[ok], ok


@torch.no_grad()
def ensemble_predict(models, Xn, device="cpu", batch_size=8192):
    """
    Xn: standardized numpy array (N,d)
    Returns:
      mean_pred: (N,2)
      disagreement: (N,) = sum var across outputs
      per_model_preds: (M,N,2)
    """
    X = torch.from_numpy(Xn).float()
    preds = []
    for m in models:
        m.eval()
        out = []
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i+batch_size].to(device)
            out.append(m(xb).cpu().numpy())
        preds.append(np.vstack(out))
    P = np.stack(preds, axis=0)          # (M,N,2)
    mean = P.mean(axis=0)                # (N,2)
    var = P.var(axis=0, ddof=0)          # (N,2)
    disagreement = var.sum(axis=1)       # (N,)
    return mean, disagreement, P


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))

def mad(x):
    return float(np.median(np.abs(x)))

def pct(x, q):
    return float(np.percentile(np.abs(x), q))


def binned_calibration(x_unc, err, nbins=10):
    """
    Bin by uncertainty proxy x_unc, return per-bin:
      center, count, rms(err), median(abs(err))
    """
    x_unc = np.asarray(x_unc)
    err = np.asarray(err)

    qs = np.quantile(x_unc, np.linspace(0, 1, nbins+1))
    # avoid duplicates
    qs = np.unique(qs)
    if qs.size < 3:
        return []

    out = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (x_unc >= lo) & (x_unc <= hi)
        if m.sum() < 50:
            continue
        center = 0.5 * (lo + hi)
        out.append((center, int(m.sum()), float(np.sqrt(np.mean(err[m]**2))), float(np.median(np.abs(err[m])))))
    return out


def save_scatter_true_pred(y_true, y_pred, title, path, alpha=0.4):
    plt.figure()
    plt.scatter(y_true, y_pred, s=10, alpha=alpha)
    lo = np.nanmin([y_true.min(), y_pred.min()])
    hi = np.nanmax([y_true.max(), y_pred.max()])
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Exact (sel_fn)")
    plt.ylabel("Emulator (ensemble mean)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_residual_hist(resid, title, path, bins=60):
    plt.figure()
    plt.hist(resid, bins=bins)
    plt.xlabel("Residual (pred - true)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_uncertainty_vs_error(x_unc, abs_err, title, path, alpha=0.35):
    plt.figure()
    plt.scatter(x_unc, abs_err, s=10, alpha=alpha)
    plt.xlabel("Uncertainty proxy")
    plt.ylabel("|error in log_mu|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def save_binned_curve(x_unc, err, title, path, nbins=10, min_count=30):
    """
    Plots binned RMS(error) and median(|error|) vs uncertainty proxy.
    """
    x_unc = np.asarray(x_unc)
    err = np.asarray(err)

    qs = np.quantile(x_unc, np.linspace(0, 1, nbins+1))
    qs = np.unique(qs)
    if qs.size < 3:
        return False

    centers, rms_vals, med_vals, counts = [], [], [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (x_unc >= lo) & (x_unc <= hi)
        if m.sum() < min_count:
            continue
        centers.append(0.5 * (lo + hi))
        rms_vals.append(float(np.sqrt(np.mean(err[m]**2))))
        med_vals.append(float(np.median(np.abs(err[m]))))
        counts.append(int(m.sum()))

    if len(centers) < 2:
        return False

    plt.figure()
    plt.plot(centers, rms_vals, marker="o", label="RMS(error in log_mu)")
    plt.plot(centers, med_vals, marker="s", label="Median(|error|)")
    plt.xlabel("Uncertainty proxy (binned)")
    plt.ylabel("Error scale")
    plt.title(title + f"  (min_count={min_count})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return True


def gate_metrics(pass_gate, large_err):
    """
    pass_gate: bool array, True means accepted by gate
    large_err: bool array, True means actually bad (|err|>threshold)
    """
    pass_gate = np.asarray(pass_gate, dtype=bool)
    large_err = np.asarray(large_err, dtype=bool)

    false_accept = (pass_gate & large_err).sum()
    false_reject = ((~pass_gate) & (~large_err)).sum()
    accept = pass_gate.sum()
    reject = (~pass_gate).sum()

    return {
        "accept_rate": float(accept / pass_gate.size),
        "reject_rate": float(reject / pass_gate.size),
        "false_accept_rate": float(false_accept / max(1, accept)),
        "false_reject_rate": float(false_reject / max(1, reject)),
        "false_accept_count": int(false_accept),
        "false_reject_count": int(false_reject),
    }


def main(args):
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    # ----------------------------
    # USER CONFIG (fill this part)
    # ----------------------------
    # 1) priors and ordered_keys
    priors = json.load(open(args.priors_json, "r"))
    ordered_keys = json.load(open(args.ordered_keys_json, "r"))  # list[str]

    # 2) your model choices, must match training compilation
    rate_model = args.rate_model
    mass_model = args.mass_model
    spin_model = args.spin_model

    # 3) injections: load from NPZ (recommended)
    inj = np.load(args.injections_npz)
    m1inj = inj["m1inj"]
    m2inj = inj["m2inj"]
    dLinj = inj["dLinj"]
    chi1Inj = inj["chi1Inj"]
    chi2Inj = inj["chi2Inj"]
    cost1Inj = inj["cost1Inj"]
    cost2Inj = inj["cost2Inj"]
    lpdinj = inj["lpdinj"]
    Ngen = float(inj["Ngen"])

    # compile sel_fn (exact reference)
    sel_fn = compile_sel_bias_fn(
        rate_model=rate_model,
        mass_model=mass_model,
        spin_model=spin_model,
        smoothing=args.smoothing,
        simplex_repair=args.simplex_repair,
        has_m2_break=args.has_m2_break,
        interp=args.interp,
        param=args.param,
        use_float32=args.use_float32,
    )

    # load scaler
    scaler = np.load(os.path.join(args.out_dir, "scaler.npz"))
    x_mu = scaler["x_mu"]
    x_sd = scaler["x_sd"]

    # load ensemble checkpoints (supports glob)
    ckpts = sorted(glob.glob(os.path.join(args.out_dir, args.ckpt_glob)))
    if len(ckpts) == 0:
        raise RuntimeError(f"No checkpoints found with glob: {args.ckpt_glob}")

    d_in = len(ordered_keys)
    models = []
    for p in ckpts:
        m = MLP(d_in=d_in, d_hidden=args.hidden, n_hidden=args.layers).to(device)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        models.append(m)
    print(f"[load] loaded {len(models)} ensemble members")

    # -----------------------------------------
    # Build / load frozen test set
    # -----------------------------------------
    test_path = os.path.join(args.out_dir, "testset.npz")
    if os.path.exists(test_path) and not args.regen_test:
        z = np.load(test_path)
        Lambda_test = z["Lambda_test"]
        log_mu_true = z["log_mu_true"]
        log_var_true = z["log_var_true"]
        neff_true = z["neff_true"]
        print(f"[test] loaded frozen test set: N={Lambda_test.shape[0]}")
    else:
        Lambda_test = sample_lambda_mixture(
            priors, ordered_keys, n=args.n_test, seed=args.test_seed,
            frac_prior=args.frac_prior, frac_edges=args.frac_edges, frac_stress=args.frac_stress
        )
        log_mu_true, neff_true, log_var_true, ok = eval_sel_batch(
            sel_fn, Lambda_test,
            m1inj, m2inj, dLinj,
            chi1Inj, chi2Inj, cost1Inj, cost2Inj,
            lpdinj, Ngen
        )
        Lambda_test = Lambda_test[ok]
        np.savez(test_path,
                 Lambda_test=Lambda_test,
                 log_mu_true=log_mu_true,
                 log_var_true=log_var_true,
                 neff_true=neff_true)
        print(f"[test] created frozen test set: N={Lambda_test.shape[0]} (saved to {test_path})")

    # -----------------------------------------
    # Emulator predictions
    # -----------------------------------------
    Xn = standardize_apply(Lambda_test, x_mu, x_sd)
    mean_pred, disagreement, per_model_preds = ensemble_predict(models, Xn, device=device)

    log_mu_pred = mean_pred[:, 0]
    log_var_pred = mean_pred[:, 1]

    err_mu = log_mu_pred - log_mu_true
    err_lv = log_var_pred - log_var_true

    # -----------------------------------------
    # Save plots
    # -----------------------------------------
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Scatter: exact vs predicted
    save_scatter_true_pred(
        log_mu_true, log_mu_pred,
        title="Selection integral: log_mu (exact vs emulator)",
        path=os.path.join(plots_dir, "log_mu_true_vs_pred.png"),
        alpha=0.4
    )
    save_scatter_true_pred(
        log_var_true, log_var_pred,
        title="Selection variance: log_var (exact vs emulator)",
        path=os.path.join(plots_dir, "log_var_true_vs_pred.png"),
        alpha=0.4
    )

    # Residual histograms
    save_residual_hist(
        err_mu,
        title="Residuals: log_mu_pred - log_mu_true",
        path=os.path.join(plots_dir, "resid_log_mu_hist.png"),
        bins=60
    )
    save_residual_hist(
        err_lv,
        title="Residuals: log_var_pred - log_var_true",
        path=os.path.join(plots_dir, "resid_log_var_hist.png"),
        bins=60
    )

    # Uncertainty proxy vs actual error
    save_uncertainty_vs_error(
        disagreement, np.abs(err_mu),
        title="Ensemble disagreement vs |error in log_mu|",
        path=os.path.join(plots_dir, "disagreement_vs_abs_err_log_mu.png"),
        alpha=0.35
    )
    save_uncertainty_vs_error(
        log_var_pred, np.abs(err_mu),
        title="Predicted log_var vs |error in log_mu|",
        path=os.path.join(plots_dir, "logvarpred_vs_abs_err_log_mu.png"),
        alpha=0.35
    )

    # Binned calibration curves
    ok1 = save_binned_curve(
        disagreement, err_mu,
        title="Calibration: ensemble disagreement",
        path=os.path.join(plots_dir, "calibration_disagreement.png"),
        nbins=args.nbins,
        min_count=args.min_bin_count
    )
    ok2 = save_binned_curve(
        log_var_pred, err_mu,
        title="Calibration: predicted log_var",
        path=os.path.join(plots_dir, "calibration_pred_logvar.png"),
        nbins=args.nbins,
        min_count=args.min_bin_count
    )

    print(f"\n[plots] saved to {plots_dir}")
    if not ok1 or not ok2:
        print("[plots] calibration curves may be empty: increase --n_test or lower --min_bin_count.")

    # -----------------------------------------
    # Report pointwise metrics
    # -----------------------------------------
    print("\n=== Pointwise error metrics (test set) ===")
    print(f"log_mu:  RMSE={rmse(err_mu):.6g}  MAD={mad(err_mu):.6g}  |err|p90={pct(err_mu,90):.6g}  |err|p99={pct(err_mu,99):.6g}")
    print(f"log_var: RMSE={rmse(err_lv):.6g}  MAD={mad(err_lv):.6g}  |err|p90={pct(err_lv,90):.6g}  |err|p99={pct(err_lv,99):.6g}")

    # -----------------------------------------
    # Calibration: bin by disagreement
    # -----------------------------------------
    print("\n=== Calibration by ensemble disagreement u(Λ) ===")
    cal = binned_calibration(disagreement, err_mu, nbins=args.nbins)
    for center, n, rms_e, medae in cal:
        print(f"u~{center:.3g}  N={n:6d}  RMS(err_log_mu)={rms_e:.6g}  Med(|err|)={medae:.6g}")

    # Calibration: bin by predicted log_var
    print("\n=== Calibration by predicted log_var ===")
    cal2 = binned_calibration(log_var_pred, err_mu, nbins=args.nbins)
    for center, n, rms_e, medae in cal2:
        print(f"log_var~{center:.3g}  N={n:6d}  RMS(err_log_mu)={rms_e:.6g}  Med(|err|)={medae:.6g}")

    # -----------------------------------------
    # Gate evaluation (optional)
    # -----------------------------------------
    if args.gate_logvar is not None:
        pass_gate = (log_var_pred <= args.gate_logvar)
        large_err = (np.abs(err_mu) >= args.err_thresh)
        gm = gate_metrics(pass_gate, large_err)

        print("\n=== Gate performance ===")
        print(f"Gate: log_var_pred <= {args.gate_logvar}")
        print(f"Badness definition: |err_log_mu| >= {args.err_thresh}")
        for k, v in gm.items():
            print(f"{k}: {v}")

    # save summary
    summary_path = os.path.join(args.out_dir, "test_summary.json")
    summary = dict(
        n_test=int(Lambda_test.shape[0]),
        log_mu=dict(rmse=rmse(err_mu), mad=mad(err_mu), p90=pct(err_mu, 90), p99=pct(err_mu, 99)),
        log_var=dict(rmse=rmse(err_lv), mad=mad(err_lv), p90=pct(err_lv, 90), p99=pct(err_lv, 99)),
        gate=dict(gate_logvar=args.gate_logvar, err_thresh=args.err_thresh) if args.gate_logvar is not None else None,
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", type=str, required=True,
                    help="Training output directory containing scaler.npz and model checkpoints.")
    ap.add_argument("--ckpt_glob", type=str, default="model_*.pt",
                    help="Glob pattern for ensemble checkpoints inside out_dir.")

    ap.add_argument("--priors_json", type=str, required=True,
                    help="Path to priors.json used by your samplers.")
    ap.add_argument("--ordered_keys_json", type=str, required=True,
                    help="JSON file containing ordered_keys list.")
    ap.add_argument("--injections_npz", type=str, required=True,
                    help="NPZ file containing injection arrays and Ngen.")

    ap.add_argument("--rate_model", type=str, required=True)
    ap.add_argument("--mass_model", type=str, required=True)
    ap.add_argument("--spin_model", type=str, required=True)

    ap.add_argument("--smoothing", type=str, default="poly")
    ap.add_argument("--simplex_repair", action="store_true")
    ap.add_argument("--has_m2_break", action="store_true")
    ap.add_argument("--interp", action="store_true")
    ap.add_argument("--param", type=str, default="vanilla")
    ap.add_argument("--use_float32", action="store_true")

    ap.add_argument("--device", type=str, default="cuda", choices=["cpu","cuda"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)

    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--test_seed", type=int, default=9999)
    ap.add_argument("--regen_test", action="store_true")

    ap.add_argument("--frac_prior", type=float, default=0.7)
    ap.add_argument("--frac_edges", type=float, default=0.2)
    ap.add_argument("--frac_stress", type=float, default=0.1)

    ap.add_argument("--nbins", type=int, default=10)

    # gating eval
    ap.add_argument("--gate_logvar", type=float, default=None,
                    help="If set, compute gate stats for log_var_pred <= gate_logvar.")
    ap.add_argument("--err_thresh", type=float, default=0.25,
                    help="Defines 'bad' as |err_log_mu| >= err_thresh for gate evaluation.")

    ap.add_argument("--min_bin_count", type=int, default=30,
                help="Minimum points per bin for calibration curves.")

    args = ap.parse_args()
    main(args)