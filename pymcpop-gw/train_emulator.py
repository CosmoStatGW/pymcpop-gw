import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import qmc

# ---- your module
from utils_train import (
    sample_Lambda_from_priors,
    sample_edge_points,
    compile_sel_bias_fn,
)




def relerr_from_logdiff(dlog: np.ndarray) -> np.ndarray:
    """
    For positive quantities z = exp(logz),
    relative error from log-difference dlog = logz_pred - logz_true is:
      |exp(dlog) - 1|
    """
    return np.abs(np.exp(dlog) - 1.0)

def eval_relerr_metrics(log_mu_pred, log_mu_true, var_u_pred, var_u_true, q=0.9):
    """
    Returns q-quantile relative errors for mu and lik_var (equivalently exp(var_u)).
    """
    d_mu = np.asarray(log_mu_pred) - np.asarray(log_mu_true)
    d_v  = np.asarray(var_u_pred)  - np.asarray(var_u_true)

    re_mu = relerr_from_logdiff(d_mu)
    re_lv = relerr_from_logdiff(d_v)

    return {
        "q": q,
        "relerr_mu_q": float(np.quantile(re_mu, q)),
        "relerr_likvar_q": float(np.quantile(re_lv, q)),
        "relerr_mu_mean": float(np.mean(re_mu)),
        "relerr_likvar_mean": float(np.mean(re_lv)),
    }


# ----------------------------
# Model
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


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def make_loaders(Xn, Y, batch_size=2048, val_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    N = Xn.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    nval = int(val_frac * N)
    ival = idx[:nval]
    itrn = idx[nval:]

    Xt = torch.from_numpy(Xn[itrn]).float()
    Yt = torch.from_numpy(Y[itrn]).float()
    Xv = torch.from_numpy(Xn[ival]).float()
    Yv = torch.from_numpy(Y[ival]).float()

    tr = DataLoader(TensorDataset(Xt, Yt), batch_size=batch_size, shuffle=True)
    va = DataLoader(TensorDataset(Xv, Yv), batch_size=batch_size, shuffle=False)
    return tr, va



@torch.no_grad()
def predict_on_loader(model, loader, device):
    preds = []
    trues = []
    for xb, yb in loader:
        xb = xb.to(device)
        preds.append(model(xb).cpu().numpy())
        trues.append(yb.cpu().numpy())
    P = np.vstack(preds)  # (N,2)
    T = np.vstack(trues)  # (N,2)
    return P, T



def train_model_earlystop(
    model,
    train_loader,
    val_loader,
    device,
    epochs=200,
    lr=2e-3,
    weight_decay=1e-6,
    huber_beta=0.5,
    grad_clip=5.0,
    patience=15,
    min_delta=1e-4,
    # relative-error stopping targets:
    relerr_target=0.05,
    relerr_quantile=0.90,
    patience_pass=5,
    check_every=1,   # evaluate val every N epochs
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=huber_beta)

    best_val = float("inf")
    best_state = None
    no_improve = 0
    pass_streak = 0

    for ep in range(1, epochs + 1):
        # ---- train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        # ---- validate occasionally
        if ep % check_every != 0:
            continue

        model.eval()
        # validation loss
        vloss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                l = loss_fn(pred, yb)
                vloss += float(l.item()) * xb.shape[0]
                n += xb.shape[0]
        vloss /= max(1, n)

        # keep best weights
        if vloss < best_val - min_delta:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # relative-error criteria on validation set
        P, T = predict_on_loader(model, val_loader, device)
        metrics = eval_relerr_metrics(
            log_mu_pred=P[:, 0], log_mu_true=T[:, 0],
            var_u_pred=P[:, 1],  var_u_true=T[:, 1],
            q=relerr_quantile
        )
        passed = (metrics["relerr_mu_q"] <= relerr_target) and (metrics["relerr_likvar_q"] <= relerr_target)
        pass_streak = (pass_streak + 1) if passed else 0

        # stopping decisions
        if pass_streak >= patience_pass:
            # reached accuracy target consistently
            break
        if no_improve >= patience:
            # plateaued
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final report (optional to log)
    return {
        "best_val_loss": float(best_val),
        "stopped_epoch": int(ep),
        "no_improve": int(no_improve),
        "pass_streak": int(pass_streak),
    }




@torch.no_grad()
def ensemble_predict_numpy(models, Xn, device="cpu", batch_size=8192):
    """
    Xn: standardized numpy array (N,d)
    returns mean predictions (N,2) and disagreement (N,)
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
    P = np.stack(preds, axis=0)        # (M,N,2)
    mean = P.mean(axis=0)              # (N,2)
    dis = P.var(axis=0, ddof=0).sum(axis=1)
    return mean, dis

def should_stop_active_learning(
    history,  # list of dicts with metrics per round
    relerr_target=0.05,
    relerr_quantile=0.90,
    stall_rounds=2,
    min_improve=0.01,  # 1% relative improvement threshold
):
    """
    Stop if:
      - both q-quantile relative errors are <= target, OR
      - improvement < min_improve for stall_rounds consecutive rounds
    """
    if len(history) == 0:
        return False, "no_history"

    last = history[-1]
    # success condition
    if (last["relerr_mu_q"] <= relerr_target) and (last["relerr_likvar_q"] <= relerr_target):
        return True, "met_relerr_targets"

    # stall condition needs enough rounds
    if len(history) < stall_rounds + 1:
        return False, "need_more_rounds"

    # compute relative improvement over previous rounds
    # use a scalar summary = max of the two q-errors (worst of the two)
    def score(m): 
        return max(m["relerr_mu_q"], m["relerr_likvar_q"])

    stalled = True
    for k in range(1, stall_rounds + 1):
        prev = history[-(k+1)]
        cur = history[-k]
        prev_s = score(prev)
        cur_s  = score(cur)
        # relative improvement
        imp = (prev_s - cur_s) / max(prev_s, 1e-12)
        if imp >= min_improve:
            stalled = False
            break

    if stalled:
        return True, "stalled_improvement"

    return False, "continue"


def train_model(model, train_loader, val_loader, device, epochs=60, lr=2e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    best_val = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        vloss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                vloss += float(loss.item()) * xb.shape[0]
                n += xb.shape[0]
        vloss /= max(1, n)

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return best_val


@torch.no_grad()
def committee_disagreement(models, X, device):
    # X: torch.FloatTensor (N,d)
    preds = []
    for m in models:
        m.eval()
        preds.append(m(X.to(device)).cpu())  # (N,2)
    P = torch.stack(preds, dim=0)           # (E,N,2)
    var = P.var(dim=0, unbiased=False)      # (N,2)
    score = var.sum(dim=1)                  # (N,)
    return score




def sample_Lambda_lhs(priors, ordered_keys, n, seed=0):
    """
    Latin-hypercube sampling in the unit cube, then apply your prior transforms.

    Supports:
      - Uniform(a,b) for entries like [a,b] or (a,b)
      - Fixed constants for entries like [300.0] or (0.01,)
      - u, v ~ U(0,1) used to build m1_low, m2_low deterministically
      - Dirichlet(1,1,1) for 'lambda' -> lambda0, lambda1, lambda2 via 2 uniforms

    Returns:
      X: (n, d) array aligned with ordered_keys
    """

    # --- Determine which stochastic base dimensions we need in [0,1]
    # We'll build a latent vector z of dimension K using LHS.
    # z will provide uniforms for each "free" Uniform(a,b) parameter,
    # plus u and v, plus 2 uniforms for the Dirichlet(1,1,1).
    free_uniform_keys = []
    fixed_keys = {}

    # Identify fixed vs uniform from priors dict
    # (ignore derived keys: m1_low, m2_low, lambda0/1/2, m_high, epsilon etc)
    derived = {"m1_low", "m2_low", "lambda0", "lambda1", "lambda2"}

    # We will always include u and v if present in priors, since they define m1_low/m2_low.
    # We will include 2 uniforms for Dirichlet if priors["lambda"] is Dirichlet(1,1,1).
    need_u = ("u" in priors)
    need_v = ("v" in priors)
    need_dirichlet = ("lambda" in priors and isinstance(priors["lambda"], str) and "Dirichlet(1,1,1)" in priors["lambda"])

    for k, spec in priors.items():
        if k in derived:
            continue
        if k == "lambda":
            continue  # handled via need_dirichlet
        if k in ("lambda0", "lambda1", "lambda2"):
            continue  # derived
        if k in ("m1_low", "m2_low"):
            continue  # derived

        # fixed if list/tuple length 1
        if isinstance(spec, (list, tuple)) and len(spec) == 1:
            fixed_keys[k] = float(spec[0])
            continue

        # uniform if list/tuple length 2
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            free_uniform_keys.append(k)
            continue

        # strings that reference other values
        if isinstance(spec, str):
            # e.g. 'lambda_vec[0]' -> derived; ignore
            continue

        # If you hit here, it's a prior spec we didn't parse
        raise ValueError(f"Unsupported prior spec for key={k}: {spec}")

    # We will draw LHS in K dimensions:
    # - one per free_uniform_keys
    # - plus u and v (if used)
    # - plus 2 uniforms for Dirichlet(1,1,1) (if used)
    K = len(free_uniform_keys) + (1 if need_u else 0) + (1 if need_v else 0) + (2 if need_dirichlet else 0)

    engine = qmc.LatinHypercube(d=K, seed=seed)
    Z = engine.random(n)  # (n, K) in (0,1)

    # --- Map Z into actual parameters
    out = {}

    col = 0
    # uniforms for simple parameters
    for k in free_uniform_keys:
        lo, hi = priors[k]
        out[k] = lo + (hi - lo) * Z[:, col]
        col += 1

    # u, v
    if need_u:
        u = Z[:, col]
        col += 1
    else:
        u = None

    if need_v:
        v = Z[:, col]
        col += 1
    else:
        v = None

    # Dirichlet(1,1,1): use 2 uniforms r1,r2
    if need_dirichlet:
        r1 = Z[:, col]
        r2 = Z[:, col + 1]
        col += 2
        s1 = np.minimum(r1, r2)
        s2 = np.maximum(r1, r2)
        lam0 = s1
        lam1 = s2 - s1
        lam2 = 1.0 - s2
        out["lambda0"] = lam0
        out["lambda1"] = lam1
        out["lambda2"] = lam2

    # fixed constants
    for k, val in fixed_keys.items():
        out[k] = np.full(n, val, dtype=float)

    # Derived m1_low and m2_low from u,v, matching your PyMC
    if u is not None:
        m1_low = 2.0 + (10.0 - 2.0) * np.sqrt(u)
        out["m1_low"] = m1_low
    if (u is not None) and (v is not None):
        m2_low = 2.0 + v * (m1_low - 2.0)
        out["m2_low"] = m2_low

    # If you want m_high fixed at 300.0 but it's not already in fixed_keys:
    # (your priors has 'm_high': [300.0])
    if "m_high" in priors and isinstance(priors["m_high"], (list, tuple)) and len(priors["m_high"]) == 1:
        out["m_high"] = np.full(n, float(priors["m_high"][0]))

    # Ensure all ordered_keys are present (either sampled, fixed, or derived)
    X = np.zeros((n, len(ordered_keys)), dtype=float)
    for j, k in enumerate(ordered_keys):
        if k in out:
            X[:, j] = out[k]
        else:
            # Allow derived keys that are not in priors dict but are part of ordered_keys
            # If you hit this, it means ordered_keys includes something not handled.
            raise KeyError(f"ordered_keys contains '{k}' but it was not generated by LHS sampler.")

    return X


def boost_low_mlow_points(priors, ordered_keys, n, seed=0, a_u=0.2):
    """
    Generate n points with boosted coverage near (m1_low, m2_low) ~ (3,3),
    while respecting m2_low < m1_low.

    Uses:
      u ~ Beta(a_u, 1) concentrated near 0 for m1_low = 3 + 7*sqrt(u)
      v ~ Uniform(0,1) for m2_low = 3 + v*(m1_low-3)
    Other parameters are sampled from the usual prior sampler (mixture).
    """
    rng = np.random.default_rng(seed)

    # start from a normal prior-like sample for all params
    X = sample_lambda_mixture(priors, ordered_keys, n, seed=seed)

    # find indices
    key_to_idx = {k:i for i,k in enumerate(ordered_keys)}
    i_m1 = key_to_idx["m1_low"]
    i_m2 = key_to_idx["m2_low"]

    # draw u near 0
    u = rng.beta(a_u, 1.0, size=n)
    m1_low = 2.0 + (10.0 - 2.0) * np.sqrt(u)

    # draw m2_low conditional on m1_low
    v = rng.random(n)
    m2_low = 2.0 + v * (m1_low - 2.0)

    X[:, i_m1] = m1_low
    X[:, i_m2] = m2_low
    return X


def build_initial_design(
    priors, ordered_keys, n_total, seed=0,
    frac_lhs=0.6, frac_edge=0.25, frac_stress=0.12,
    frac_low_mlow=0.05,   # NEW
    low_mlow_a_u=0.2      # NEW (smaller => more concentrated near 3)
):
    rng = np.random.default_rng(seed)

    # allocate counts
    n_low   = int(round(n_total * frac_low_mlow))
    n_lhs   = int(round(n_total * frac_lhs))
    n_edge  = int(round(n_total * frac_edge))
    n_stress = n_total - n_low - n_lhs - n_edge

    # main components
    X_lhs = sample_Lambda_lhs(priors, ordered_keys, n_lhs, seed=seed)
    X_edge = sample_edge_points(priors, ordered_keys, n_edge, seed=seed+11)
    X_stress = sample_edge_points(priors, ordered_keys, n_stress, seed=seed+29)

    # NEW: targeted boost for (m1_low,m2_low) lower-left corner
    # X_low = boost_low_mlow_points(
    #     priors, ordered_keys, n_low, seed=seed+77, a_u=low_mlow_a_u
    # )

    X0 = np.concatenate([X_lhs, X_edge, X_stress, ], axis=0) # X_low
    rng.shuffle(X0, axis=0)
    return X0


def build_initial_design_0(priors, ordered_keys, n_total, seed=0,
                         frac_lhs=0.7, frac_edge=0.2, frac_stress=0.1):
    rng = np.random.default_rng(seed)
    n_lhs = int(round(n_total * frac_lhs))
    n_edge = int(round(n_total * frac_edge))
    n_stress = n_total - n_lhs - n_edge

    # 1) LHS in prior-measure space (your new function)
    X_lhs = sample_Lambda_lhs(priors, ordered_keys, n_lhs, seed=seed)

    # 2) Edge points (your existing function already knows constraints)
    X_edge = sample_edge_points(
        priors, ordered_keys,
        n_random=max(1, n_edge // 10),
        include_edges=True,
        baseline="mid",
        seed=seed + 11
    )
    if X_edge.shape[0] > n_edge:
        X_edge = X_edge[:n_edge]

    # 3) Stress points (more edges + random baselines)
    X_stress = sample_edge_points(
        priors, ordered_keys,
        n_random=max(1, n_stress // 10),
        include_edges=True,
        baseline="random",
        seed=seed + 29
    )
    if X_stress.shape[0] > n_stress:
        X_stress = X_stress[:n_stress]

    X0 = np.concatenate([X_lhs, X_edge, X_stress], axis=0)
    rng.shuffle(X0, axis=0)
    return X0


def sample_lambda_mixture(priors, ordered_keys, n, seed=0,
                         frac_prior=0.6, frac_edges=0.25, frac_stress=0.15):
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
    Ls.append(edge_block[:n_edges] if edge_block.shape[0] >= n_edges else edge_block)

    # stress: more edges + priors
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



def main(
    priors, ordered_keys,
    rate_model, mass_model, spin_model,
    # injections:
    m1inj, m2inj, dLinj,
    chi1Inj, chi2Inj, cost1Inj, cost2Inj,
    lpdinj, Ngen,
    out_dir="./emulator_out",
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=0,
    init_points=5000,
    pool_size=50000,
    add_per_round=5000,
    rounds=8,
    ensemble_size=3,
    epochs=200,          # allow early stopping to decide
    batch_size=2048,
    # ---- early stopping (within-round)
    patience=15,
    min_delta=1e-4,
    relerr_target=0.05,
    relerr_quantile=0.90,
    patience_pass=5,
    check_every=1,
    # ---- active learning stopping (across rounds)
    stall_rounds=2,
    min_improve=0.01,
    # ---- frozen test set for active-learning stopping
    n_test=2000,
    test_seed=9999,
):
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, "dataset.npz")
    scaler_path = os.path.join(out_dir, "scaler.npz")
    test_path = os.path.join(out_dir, "testset_for_training_stop.npz")

    print("Compiling selection function...")
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
    print("Done.")

    # -------------------------
    # Load or init training dataset
    # -------------------------
    if os.path.exists(data_path):
        print("Loading init dataset...")
        z = np.load(data_path)
        X = z["X"]
        log_mu = z["log_mu"]
        log_var = z["log_var"]
        neff = z["neff"]
        print(f"[load] N={X.shape[0]}")
    else:
        print("Generating init dataset from prior...")
        X0 = sample_lambda_mixture(priors, ordered_keys, init_points, seed=seed)
        lm, nf, lv, ok = eval_sel_batch(
            sel_fn, X0,
            m1inj, m2inj, dLinj,
            chi1Inj, chi2Inj, cost1Inj, cost2Inj,
            lpdinj, Ngen
        )
        X = X0[ok]
        log_mu, log_var, neff = lm, lv, nf
        np.savez(data_path, X=X, log_mu=log_mu, log_var=log_var, neff=neff)
        print(f"[init] N={X.shape[0]}")

    # -------------------------
    # Create/load a frozen test set (used ONLY for stopping decisions)
    # -------------------------
    if os.path.exists(test_path):
        zt = np.load(test_path)
        Lambda_test = zt["Lambda_test"]
        log_mu_true_test = zt["log_mu_true_test"]
        log_var_true_test = zt["log_var_true_test"]
        print(f"[test] loaded frozen test set: N={Lambda_test.shape[0]}")
    else:
        print(f"[test] creating frozen test set: n_test={n_test}")
        Lambda_test = sample_lambda_mixture(
            priors, ordered_keys, n_test, seed=test_seed
        )
        lm_t, nf_t, lv_t, ok_t = eval_sel_batch(
            sel_fn, Lambda_test,
            m1inj, m2inj, dLinj,
            chi1Inj, chi2Inj, cost1Inj, cost2Inj,
            lpdinj, Ngen
        )
        Lambda_test = Lambda_test[ok_t]
        log_mu_true_test = lm_t
        log_var_true_test = lv_t
        np.savez(
            test_path,
            Lambda_test=Lambda_test,
            log_mu_true_test=log_mu_true_test,
            log_var_true_test=log_var_true_test
        )
        print(f"[test] saved frozen test set: N={Lambda_test.shape[0]} -> {test_path}")

    # -------------------------
    # Active learning loop with stopping
    # -------------------------
    history = []  # stores per-round test metrics dicts

    for r in range(rounds):
        print(f"\n=== round {r+1}/{rounds} ===")

        # fit scaler on current training set
        x_mu, x_sd = standardize_fit(X)
        np.savez(scaler_path, x_mu=x_mu, x_sd=x_sd)
        Xn = standardize_apply(X, x_mu, x_sd)

        Y = np.stack([log_mu, log_var], axis=1)
        tr, va = make_loaders(Xn, Y, batch_size=batch_size, val_frac=0.1, seed=seed + r)

        # ---- train ensemble with early stopping
        models = []
        for e in range(ensemble_size):
            m = MLP(d_in=X.shape[1], d_hidden=256, n_hidden=4).to(device)
            stats = train_model_earlystop(
                m, tr, va, device=device,
                epochs=epochs,
                lr=2e-3,
                patience=patience,
                min_delta=min_delta,
                relerr_target=relerr_target,
                relerr_quantile=relerr_quantile,
                patience_pass=patience_pass,
                check_every=check_every,
            )
            print(
                f"  ens{e}: best_val_loss={stats['best_val_loss']:.6g} "
                f"stopped_epoch={stats['stopped_epoch']}"
            )
            models.append(m)
            torch.save(m.state_dict(), os.path.join(out_dir, f"model_r{r}_e{e}.pt"))

        # ---- evaluate on frozen test set to decide whether to stop active learning
        Xn_test = (Lambda_test - x_mu) / x_sd
        mean_pred, disagreement = ensemble_predict_numpy(models, Xn_test, device=device)

        mtest = eval_relerr_metrics(
            log_mu_pred=mean_pred[:, 0],
            log_mu_true=log_mu_true_test,
            var_u_pred=mean_pred[:, 1],
            var_u_true=log_var_true_test,
            q=relerr_quantile,
        )
        history.append(mtest)

        stop, reason = should_stop_active_learning(
            history,
            relerr_target=relerr_target,
            relerr_quantile=relerr_quantile,
            stall_rounds=stall_rounds,
            min_improve=min_improve,
        )

        print(
            f"[round {r}] test q{int(relerr_quantile*100)} relerr(mu)={mtest['relerr_mu_q']:.4f}  "
            f"q{int(relerr_quantile*100)} relerr(lik_var)={mtest['relerr_likvar_q']:.4f}  "
            f"stop={stop} ({reason})"
        )

        if stop:
            print("[active learning] stopping criterion met; not acquiring more points.")
            break

        # -------------------------
        # Acquisition step (only if continuing)
        # -------------------------
        pool = sample_lambda_mixture(priors, ordered_keys, pool_size, seed=seed + 1000 + r)
        pooln = standardize_apply(pool, x_mu, x_sd)
        poolt = torch.from_numpy(pooln).float()

        score = committee_disagreement(models, poolt, device=device).numpy()

        top = np.argsort(score)[-add_per_round:][::-1]
        X_new = pool[top]

        lm, nf, lv, ok = eval_sel_batch(
            sel_fn, X_new,
            m1inj, m2inj, dLinj,
            chi1Inj, chi2Inj, cost1Inj, cost2Inj,
            lpdinj, Ngen
        )
        X_new = X_new[ok]

        X = np.concatenate([X, X_new], axis=0)
        log_mu = np.concatenate([log_mu, lm], axis=0)
        log_var = np.concatenate([log_var, lv], axis=0)
        neff = np.concatenate([neff, nf], axis=0)

        np.savez(data_path, X=X, log_mu=log_mu, log_var=log_var, neff=neff)
        print(f"[data] N={X.shape[0]}")

    # save stopping history for inspection
    hist_path = os.path.join(out_dir, "active_learning_history.json")
    try:
        import json
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"[saved] {hist_path}")
    except Exception as e:
        print(f"[warn] could not save history json: {e}")

    print("\nDone.")


def main_no_earlystop(
    priors, ordered_keys,
    rate_model, mass_model, spin_model,
    # injections:
    m1inj, m2inj, dLinj,
    chi1Inj, chi2Inj, cost1Inj, cost2Inj,
    lpdinj, Ngen,
    out_dir="./emulator_out",
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=0,
    init_points=5000,
    pool_size=50000,
    add_per_round=5000,
    rounds=8,
    ensemble_size=3,
    epochs=60,
    batch_size=2048,
):
    
    data_path = os.path.join(out_dir, "dataset.npz")
    scaler_path = os.path.join(out_dir, "scaler.npz")

    print("Compiling selection function...")
    # compile sel_fn using your function
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
    print("Done.")

    # load or init dataset
    if os.path.exists(data_path):
        print("Loading init dataset...")
        z = np.load(data_path)
        X = z["X"]
        log_mu = z["log_mu"]
        log_var = z["log_var"]
        neff = z["neff"]
        print(f"[load] N={X.shape[0]}")
    else:
        print("Generating init dataset from prior...")
        X0 = sample_lambda_mixture(priors, ordered_keys, init_points, seed=seed)
        lm, nf, lv, ok = eval_sel_batch(
            sel_fn, X0,
            m1inj, m2inj, dLinj,
            chi1Inj, chi2Inj, cost1Inj, cost2Inj,
            lpdinj, Ngen
        )
        X = X0[ok]
        log_mu, log_var, neff = lm, lv, nf
        np.savez(data_path, X=X, log_mu=log_mu, log_var=log_var, neff=neff)
        print(f"[init] N={X.shape[0]}")
        

    rng = np.random.default_rng(seed)

    for r in range(rounds):
        print(f"\n=== round {r+1}/{rounds} ===")

        # fit scaler
        x_mu, x_sd = standardize_fit(X)
        np.savez(scaler_path, x_mu=x_mu, x_sd=x_sd)
        Xn = standardize_apply(X, x_mu, x_sd)

        Y = np.stack([log_mu, log_var], axis=1)
        tr, va = make_loaders(Xn, Y, batch_size=batch_size, val_frac=0.1, seed=seed + r)

        # train ensemble
        models = []
        for e in range(ensemble_size):
            m = MLP(d_in=X.shape[1], d_hidden=256, n_hidden=4).to(device)
            best_val = train_model(m, tr, va, device=device, epochs=epochs, lr=2e-3)
            print(f"  ens{e}: best_val={best_val:.6g}")
            models.append(m)
            torch.save(m.state_dict(), os.path.join(out_dir, f"model_r{r}_e{e}.pt"))

        # candidate pool
        pool = sample_lambda_mixture(priors, ordered_keys, pool_size, seed=seed + 1000 + r)
        pooln = standardize_apply(pool, x_mu, x_sd)
        poolt = torch.from_numpy(pooln).float()

        # acquisition score = disagreement
        score = committee_disagreement(models, poolt, device=device).numpy()

        # pick top points
        top = np.argsort(score)[-add_per_round:][::-1]
        X_new = pool[top]

        # evaluate sel_fn on selected points
        lm, nf, lv, ok = eval_sel_batch(
            sel_fn, X_new,
            m1inj, m2inj, dLinj,
            chi1Inj, chi2Inj, cost1Inj, cost2Inj,
            lpdinj, Ngen
        )
        X_new = X_new[ok]

        # append
        X = np.concatenate([X, X_new], axis=0)
        log_mu = np.concatenate([log_mu, lm], axis=0)
        log_var = np.concatenate([log_var, lv], axis=0)
        neff = np.concatenate([neff, nf], axis=0)

        np.savez(data_path, X=X, log_mu=log_mu, log_var=log_var, neff=neff)
        print(f"[data] N={X.shape[0]}")

    print("\nDone.")


if __name__ == "__main__":

    
    #raise SystemExit("Fill inputs then call main(...).")

    # ----------------------------
    # User: provide these
    # ----------------------------
    # priors: dict loaded from your JSON
    # ordered_keys: list[str]
    # injections: arrays (1D) + Ngen scalar
    #
    # m1inj, m2inj, dLinj, lpdinj : (Ninj,)
    # chi1Inj, chi2Inj, cost1Inj, cost2Inj : (Ninj,)  (if spin_model default)
    # Ngen : scalar

    import sys
    sys.path.append('../pymcpop-gw/')
    
    import pytensor_tools as atools
    import pymc_models as models
    import data_tools as dt
    
    import importlib 
    
    import re
    
    
    importlib.reload(atools)
    importlib.reload(models)
    importlib.reload(dt)

    out_dir="../results/offline_shards_lh"
    os.makedirs(out_dir, exist_ok=True)
    print("Created out dir %s"%out_dir)
    
    
    
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

    with open(os.path.join(out_dir, "priors.json"), "w") as f:
        json.dump(priors, f, indent=2)

    
    
    
    
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
    np.savez(os.path.join(out_dir,  "injections.npz"),
         m1inj=m1inj, m2inj=m2inj, dLinj=dLinj,
         chi1Inj=chi1Inj, chi2Inj=chi2Inj,
         cost1Inj=cost1Inj, cost2Inj=cost2Inj,
         lpdinj=lpdinj, Ngen=Ngen)
    print("Injections saved as .npz")
        
    ordered_keys = [
      "H0","Om", "w0","Xi0","nXi0",
      "gamma","kappa","zp",
      "muChi", "sigmaChi", "zeta", "sigmat",
      # DPLDP mass params (20):
      "alpha1","alpha2","mb","mu1","sigma1","mu2","sigma2","m1_low","m_high","delta_m1",
      "lambda0","lambda1",
      "beta","m2_low","delta_m2","epsilon","mu_g","w_g","sig_g_low","sig_g_high"
    ]


    with open(os.path.join(out_dir, "ordered_keys.json"), "w") as f:
        json.dump(ordered_keys, f, indent=2)


    print("ordered_keys: ")
    print(ordered_keys)


    print("Starting training...")

    main(
        priors, ordered_keys,
        rate_model, mass_model, spin_model,
        # injections:
        m1inj, m2inj, dLinj,
        chi1Inj, chi2Inj, cost1Inj, cost2Inj,
        lpdinj, Ngen,
        out_dir=out_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=0,
        # ↓↓↓ laptop-safe but informative ↓↓↓
        init_points=300,
        pool_size=2000,
        add_per_round=200,
        rounds=2,
        ensemble_size=2,
        epochs=15,
        batch_size=128,
    )
    

    