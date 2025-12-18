import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import numpy as np
import json
import pymc as pm
import pytensor.tensor as at
import pytensor
from tqdm import tqdm
import re

from pytensor.tensor import TensorVariable
import sys
    
import pytensor_tools as atools
import pymc_models as models
import data_tools as dt



def sample_Lambda_from_priors(priors, ordered_keys, rng=None, n=1, need_m1_low=True, need_m2_low=True):
    """
    Sample Lambda vectors from a priors dict.

    Supports:
      - [low, high] uniform
      - [fixed] fixed
      - "Dirichlet(a,b,c)"
      - "lambda_vec[i]" references
      - special derived (if requested in ordered_keys):
          m1_low = 3 + (10-3)*sqrt(u), with u ~ U(0,1)
          m2_low = 3 + v*(m1_low-3),   with v ~ U(0,1)

    If 'm1_low' and/or 'm2_low' are in ordered_keys and not explicitly in priors,
    they will be generated via (u, v). If you also include 'u'/'v' in ordered_keys,
    those will be returned too; otherwise they are sampled internally.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Parse any Dirichlet specs
    dirichlet_specs = {}
    for k, v in priors.items():
        if isinstance(v, str) and v.strip().startswith("Dirichlet("):
            m = re.match(r"Dirichlet\((.*)\)", v.strip())
            if m is None:
                raise ValueError(f"Could not parse {k}: {v}")
            alpha = [float(x.strip()) for x in m.group(1).split(",")]
            dirichlet_specs[k] = np.array(alpha, dtype=float)

    Lambdas = np.zeros((n, len(ordered_keys)), dtype=float)

    for s in range(n):
        ctx = {}

        # 1) sample Dirichlet vectors (e.g. "lambda")
        for name, alpha in dirichlet_specs.items():
            vec = rng.dirichlet(alpha)
            ctx[name] = vec
            if name == "lambda":
                ctx["lambda_vec"] = vec

        # 2) Decide whether we need triangle masses
        #need_m1_low = ("m1_low" in ordered_keys) and ("m1_low" not in priors)
        #need_m2_low = ("m2_low" in ordered_keys) and ("m2_low" not in priors)

        # sample u, v if needed (or if explicitly in ordered_keys and not in priors)
        if need_m1_low or ("u" in ordered_keys and "u" not in priors) or need_m2_low or ("v" in ordered_keys and "v" not in priors):
            if "u" in priors:
                # allow overriding u from priors if provided
                spec_u = priors["u"]
                if isinstance(spec_u, (list, tuple)) and len(spec_u) == 2:
                    u = rng.uniform(float(spec_u[0]), float(spec_u[1]))
                elif isinstance(spec_u, (list, tuple)) and len(spec_u) == 1:
                    u = float(spec_u[0])
                else:
                    raise ValueError(f"Unsupported prior spec for 'u': {spec_u}")
            else:
                u = rng.uniform(0.0, 1.0)

            if "v" in priors:
                spec_v = priors["v"]
                if isinstance(spec_v, (list, tuple)) and len(spec_v) == 2:
                    v = rng.uniform(float(spec_v[0]), float(spec_v[1]))
                elif isinstance(spec_v, (list, tuple)) and len(spec_v) == 1:
                    v = float(spec_v[0])
                else:
                    raise ValueError(f"Unsupported prior spec for 'v': {spec_v}")
            else:
                v = rng.uniform(0.0, 1.0)

            ctx["u"] = float(u)
            ctx["v"] = float(v)

            # Derived triangle mapping
            if need_m1_low or need_m2_low:
                m1_low = 3.0 + (10.0 - 3.0) * np.sqrt(u)
                m2_low = 3.0 + v * (m1_low - 3.0)
                ctx["m1_low"] = float(m1_low)
                ctx["m2_low"] = float(m2_low)
            else:
                ctx["u"] = float(u)
                ctx["v"] = float(v)

        # 3) Fill Lambda in required order
        for j, key in enumerate(ordered_keys):
            # already computed (u, v, m1_low, m2_low, etc.)
            if key in ctx:
                Lambdas[s, j] = float(ctx[key])
                continue

            spec = priors.get(key, None)
            if spec is None:
                if key=='epsilon':
                    val=1e-02
                    spec=1e-02
                elif key=='sig_g_low':
                    val=1e-02
                    spec=1e-02
                elif key=='sig_g_high':
                    val=1e-02
                    spec=1e-02
                else:
                    raise KeyError(f"Prior missing for key '{key}' (and not derived)")

            if isinstance(spec, str):
                expr = spec.strip()
                mref = re.match(r"lambda_vec\[(\d+)\]", expr)
                if mref:
                    idx = int(mref.group(1))
                    if "lambda_vec" not in ctx:
                        raise ValueError("Found lambda_vec[...] prior but no 'lambda' Dirichlet prior was sampled.")
                    val = float(ctx["lambda_vec"][idx])
                else:
                    raise ValueError(f"Unknown string prior expression for '{key}': {spec}")

            elif isinstance(spec, (list, tuple)) and len(spec) == 1:
                val = float(spec[0])

            elif isinstance(spec, (list, tuple)) and len(spec) == 2:
                lo, hi = float(spec[0]), float(spec[1])
                val = rng.uniform(lo, hi)
            elif isinstance(spec, (float)):
                pass
                
            else:
                raise ValueError(f"Unsupported prior spec for '{key}': {spec}")

            ctx[key] = float(val)
            Lambdas[s, j] = float(val)

    return Lambdas



def _resolve_range(priors, key):
    """Return (lo, hi, is_fixed). Supports [lo,hi] or [val]."""
    spec = priors.get(key, None)
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)) and len(spec) == 1:
        v = float(spec[0])
        return (v, v, True)
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return (float(spec[0]), float(spec[1]), False)
    return None

def _sample_dirichlet_if_present(priors, rng):
    """Return lambda_vec if 'lambda' is Dirichlet(...) else None."""
    spec = priors.get("lambda", None)
    if isinstance(spec, str) and spec.strip().startswith("Dirichlet("):
        inside = spec.strip()[len("Dirichlet("):-1]
        alpha = np.array([float(x.strip()) for x in inside.split(",")], dtype=float)
        return rng.dirichlet(alpha)
    return None

def _triangle_m1m2_low(rng, u=None, v=None):
    """Your exact mapping."""
    if u is None:
        u = rng.uniform(0.0, 1.0)
    if v is None:
        v = rng.uniform(0.0, 1.0)
    m1_low = 3.0 + (10.0 - 3.0) * np.sqrt(u)
    m2_low = 3.0 + v * (m1_low - 3.0)
    return float(m1_low), float(m2_low)

def sample_edge_points(
    priors,
    ordered_keys,
    n_random=20,
    include_edges=True,
    edge_mode="minmax",   # "minmax" only for now
    baseline="mid",       # "mid" or "random"
    seed=0,
):
    """
    Build Lambda samples:
      - n_random random prior draws
      - plus 2*D edge points (each param at min and max, others at baseline),
        with Dirichlet handled consistently, and triangle mapping for m1_low/m2_low.

    Returns:
      Lambdas: (N, D) float64
    """
    rng = np.random.default_rng(seed)

    # --- helper to make one full random draw consistent with priors ---
    def draw_one():
        ctx = {}
        # Dirichlet lambda vector if present
        lamvec = _sample_dirichlet_if_present(priors, rng)
        if lamvec is not None:
            ctx["lambda_vec"] = lamvec
            # resolve lambda0/lambda1/lambda2 if referenced
            ctx["lambda0"] = float(lamvec[0])
            ctx["lambda1"] = float(lamvec[1])
            ctx["lambda2"] = float(lamvec[2])

        # triangle lows (only if needed in ordered_keys)
        if ("m1_low" in ordered_keys) or ("m2_low" in ordered_keys):
            m1l, m2l = _triangle_m1m2_low(rng)
            ctx["m1_low"] = m1l
            ctx["m2_low"] = m2l

        # generic scalar sampling
        for k in ordered_keys:
            if k in ctx:
                continue
            spec = priors.get(k, None)
            if spec is None:
                raise KeyError(f"Missing prior for '{k}' and it is not derived.")
            if isinstance(spec, str):
                # support lambda_vec[i] references
                if spec.strip().startswith("lambda_vec["):
                    if "lambda_vec" not in ctx:
                        raise ValueError("lambda_vec reference found but 'lambda' Dirichlet prior not present.")
                    i = int(spec.strip()[len("lambda_vec["):-1])
                    ctx[k] = float(ctx["lambda_vec"][i])
                else:
                    raise ValueError(f"Unsupported string prior for '{k}': {spec}")
            elif isinstance(spec, (list, tuple)) and len(spec) == 1:
                ctx[k] = float(spec[0])
            elif isinstance(spec, (list, tuple)) and len(spec) == 2:
                lo, hi = float(spec[0]), float(spec[1])
                ctx[k] = float(rng.uniform(lo, hi))
            else:
                raise ValueError(f"Unsupported prior spec for '{k}': {spec}")

        return np.array([ctx[k] for k in ordered_keys], dtype=float)

    # --- baseline point ---
    def baseline_point():
        if baseline == "random":
            return draw_one()

        # baseline == "mid"
        ctx = {}

        # Dirichlet baseline: symmetric mean if available, else draw one
        lamvec = _sample_dirichlet_if_present(priors, rng)
        if lamvec is not None:
            # mean of Dirichlet(alpha) is alpha/sum(alpha)
            spec = priors["lambda"].strip()
            inside = spec[len("Dirichlet("):-1]
            alpha = np.array([float(x.strip()) for x in inside.split(",")], dtype=float)
            lamvec = alpha / alpha.sum()
            ctx["lambda_vec"] = lamvec
            ctx["lambda0"] = float(lamvec[0])
            ctx["lambda1"] = float(lamvec[1])
            ctx["lambda2"] = float(lamvec[2])

        # triangle baseline: choose u=v=0.25 (arbitrary but stable)
        if ("m1_low" in ordered_keys) or ("m2_low" in ordered_keys):
            m1l, m2l = _triangle_m1m2_low(rng, u=0.25, v=0.25)
            ctx["m1_low"] = m1l
            ctx["m2_low"] = m2l

        for k in ordered_keys:
            if k in ctx:
                continue
            rr = _resolve_range(priors, k)
            if rr is None:
                spec = priors.get(k, None)
                if isinstance(spec, str) and spec.strip().startswith("lambda_vec["):
                    i = int(spec.strip()[len("lambda_vec["):-1])
                    ctx[k] = float(ctx["lambda_vec"][i])
                else:
                    raise KeyError(f"Missing/unsupported prior for '{k}' in baseline.")
            else:
                lo, hi, is_fixed = rr
                ctx[k] = float(lo) if is_fixed else float(0.5 * (lo + hi))

        return np.array([ctx[k] for k in ordered_keys], dtype=float)

    base = baseline_point()

    # --- random draws ---
    samples = [draw_one() for _ in range(int(n_random))]

    # --- edge points ---
    if include_edges:
        D = len(ordered_keys)
        for j, key in enumerate(ordered_keys):
            # Skip derived triangle vars at edge stage; handle via u,v instead
            if key in ("m1_low", "m2_low"):
                continue
            # Skip lambda0/lambda1 if they are derived from Dirichlet;
            # we will create edges by changing the *Dirichlet* via corner cases instead.
            if isinstance(priors.get(key, None), str) and priors[key].strip().startswith("lambda_vec["):
                continue

            rr = _resolve_range(priors, key)
            if rr is None:
                # cannot edge a non-numeric/derived key
                continue
            lo, hi, is_fixed = rr
            if is_fixed or lo == hi:
                continue

            # min edge
            x = base.copy()
            x[j] = lo
            samples.append(x)

            # max edge
            x = base.copy()
            x[j] = hi
            samples.append(x)

        # Special edges for Dirichlet lambda: push to corners
        if isinstance(priors.get("lambda", None), str) and priors["lambda"].strip().startswith("Dirichlet("):
            # corners (almost all weight in one component)
            corners = [
                np.array([0.999, 0.0005, 0.0005]),
                np.array([0.0005, 0.999, 0.0005]),
                np.array([0.0005, 0.0005, 0.999]),
            ]
            for lamvec in corners:
                x = base.copy()
                # fill lambda0/lambda1/lambda2 wherever they appear
                for j, key in enumerate(ordered_keys):
                    if priors.get(key, "") == "lambda_vec[0]":
                        x[j] = float(lamvec[0])
                    elif priors.get(key, "") == "lambda_vec[1]":
                        x[j] = float(lamvec[1])
                    elif priors.get(key, "") == "lambda_vec[2]":
                        x[j] = float(lamvec[2])
                samples.append(x)

        # Special edges for triangle lows: edge via u,v
        if ("m1_low" in ordered_keys) or ("m2_low" in ordered_keys):
            # u near 0 -> m1_low near 3; u near 1 -> m1_low near 10
            # v near 0 -> m2_low near 3; v near 1 -> m2_low near m1_low
            uv_edges = [
                (1e-6, 1e-6),
                (1e-6, 1.0 - 1e-6),
                (1.0 - 1e-6, 1e-6),
                (1.0 - 1e-6, 1.0 - 1e-6),
            ]
            for u, v in uv_edges:
                m1l, m2l = _triangle_m1m2_low(rng, u=u, v=v)
                x = base.copy()
                for j, key in enumerate(ordered_keys):
                    if key == "m1_low":
                        x[j] = m1l
                    elif key == "m2_low":
                        x[j] = m2l
                samples.append(x)

    return np.stack(samples, axis=0)





def compile_sel_bias_fn(
    rate_model, mass_model, spin_model,
    smoothing="poly",
    simplex_repair=False,
    has_m2_break=False,
    interp=False,
    param="vanilla",
    use_float32=False,
):
    """
    Returns a compiled pytensor function:

      f(m1inj, m2inj, dLinj, spinsInj, log_p_draw, Lambda, Ndraw)
        -> (log_mu, Neff, var_log_lik_u)

    spinsInj convention:
      - if spin_model in ('default','default_gauss'): list/tuple of 4 arrays [chi1, chi2, cost1, cost2]
      - if spin_model == 'none': pass an empty list [] or None
    """
    # choose dtype
    fX = "float32" if use_float32 else "float64"

    # inputs
    m1inj = at.vector("m1inj", dtype=fX)
    m2inj = at.vector("m2inj", dtype=fX)
    dLinj = at.vector("dLinj", dtype=fX)
    log_p_draw = at.vector("log_p_draw", dtype=fX)
    log_p_incl = at.vector("log_p_incl", dtype=fX)

    # spins as 4 vectors (optional)
    if spin_model in ("default", "default_gauss"):
        chi1 = at.vector("chi1Inj", dtype=fX)
        chi2 = at.vector("chi2Inj", dtype=fX)
        cost1 = at.vector("cost1Inj", dtype=fX)
        cost2 = at.vector("cost2Inj", dtype=fX)
        spinsInj = [chi1, chi2, cost1, cost2]
        spin_inputs = [chi1, chi2, cost1, cost2]
    else:
        spinsInj = []
        spin_inputs = []

    # Lambda (unknown length at compile time → vector)
    Lambda = at.vector("Lambda", dtype=fX)

    # Ndraw (scalar)
    Ndraw = at.scalar("Ndraw", dtype=fX)

    # build graph
    log_mu, Neff, var_log_lik_u = models.sel_bias_with_uncertainty_at_0(
        m1inj=m1inj, m2inj=m2inj, dLinj=dLinj,
        spinsInj=spinsInj,
        log_p_draw=log_p_draw,
        Lambda=Lambda, Ndraw=Ndraw,
        rate_model=rate_model, mass_model=mass_model, spin_model=spin_model,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
        has_m2_break=has_m2_break,
        interp=interp,
        log_p_incl=log_p_incl,
        wrap_logp=False,
        log_ddL_dz_inj=None,
        zinj=None,
        dcinj=None,
        param=param,
        interp_vals_mass=None,
        interp_grids_mass=None,
        verbose=False
    )

    f = pytensor.function(
        inputs=[m1inj, m2inj, dLinj] + spin_inputs + [log_p_draw, Lambda, Ndraw, log_p_incl],
        outputs=[log_mu, Neff, var_log_lik_u],
        on_unused_input="ignore",
        mode="FAST_RUN",
    )
    return f


def compile_log_p_pop_fn(
    rate_model, mass_model, spin_model,
    smoothing="poly",
    simplex_repair=False,
    has_m2_break=False,
    param="vanilla",
    use_float32=False,
):
    fX = "float32" if use_float32 else "float64"

    m1s = at.vector("m1s", dtype=fX)
    m2s = at.vector("m2s", dtype=fX)
    z   = at.vector("z", dtype=fX)
    dL  = at.vector("dL", dtype=fX)

    if spin_model in ("default", "default_gauss"):
        chi1 = at.vector("chi1", dtype=fX)
        chi2 = at.vector("chi2", dtype=fX)
        cost1 = at.vector("cost1", dtype=fX)
        cost2 = at.vector("cost2", dtype=fX)
        spins = [chi1, chi2, cost1, cost2]
        spin_inputs = [chi1, chi2, cost1, cost2]
    else:
        spins = []
        spin_inputs = []

    Lambda = at.vector("Lambda", dtype=fX)

    lp = models.log_p_pop_at(
        m1s, m2s, z, dL, spins,
        Lambda,
        rate_model=rate_model,
        mass_model=mass_model,
        spin_model=spin_model,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
        has_m2_break=has_m2_break,
        dc=None,
        log_ddL_dz_pre=None,
        param=param,
        interp_vals_mass=None,
        interp_grids_mass=None,
        is_observed=False,
        z_grid=None,
    )

    f = pytensor.function(
        inputs=[m1s, m2s, z, dL] + spin_inputs + [Lambda],
        outputs=lp,
        on_unused_input="ignore",
        mode="FAST_RUN",
    )
    return f


# z_sym      = at.dvector('z_nodes')    
# d_sym      = at.dvector('dL_nodes')
# H0_sym     = at.dscalar('H0')
# Om_sym     = at.dscalar('Om')
# w0_sym     = at.dscalar('w0')
# Xi0_sym     = at.dscalar('Xi0')
# n_sym     = at.dscalar('nXi0')


# z_from_dL_sym = atools.z_from_dL_at(d_sym, H0_sym, Om_sym, w0_sym, Xi0_sym, n_sym, interp=False, param='vanilla')
# z_from_dL_fn = pytensor.function([d_sym, H0_sym, Om_sym, w0_sym, Xi0_sym, n_sym], z_from_dL_sym)


# sel_fn = compile_sel_bias_fn(
#     rate_model=rate_model,
#     mass_model=mass_model,
#     spin_model=spin_model, 
#     smoothing="poly",
#     simplex_repair=False,
#     has_m2_break=False,
#     interp=False,          
#     param="vanilla",
#     use_float32=False,    
# )


# lp_fn = compile_log_p_pop_fn(
#     rate_model, mass_model, spin_model,
#     smoothing="poly",
#     simplex_repair=False,
#     has_m2_break=False,
#     param="vanilla",
#     use_float32=False,
# )





