# likelihood.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

import cosmology as cosmo
from backends import JAXBackend

bk = JAXBackend()

# -----------------------------
# small helpers (pure JAX)
# -----------------------------

def inv_logitat(x):
    # maps R -> (0, 1)
    return jax.nn.sigmoid(x)

def inv_flogitat(x):
    # maps R -> (-1, 1)
    return 2.0 * jax.nn.sigmoid(x) - 1.0

def m1m2_from_Mcq(Mc, q):
    # Standard chirp mass relation: Mc = (m1 m2)^(3/5) / (m1+m2)^(1/5)
    # with q = m2/m1 <= 1
    # => m1 = Mc * (1+q)^(1/5) / q^(3/5)
    m1 = Mc * jnp.power(1.0 + q, 1.0 / 5.0) / jnp.power(q, 3.0 / 5.0)
    m2 = q * m1
    return m1, m2

def logclip(x, tiny=1e-300):
    return jnp.log(jnp.clip(x, tiny, jnp.inf))


# -----------------------------
# dL prior coding
# -----------------------------

DLPRIOR_CODES = {
    "none": 0,
    "dLsq": 1,
    "UniformComovingVolume": 2,
    "UniformComovingVolume-J": 3,
    "UniformSourceFrame": 4,
    "UniformSourceFrame-J": 5,
    "UniformSourceFrame-bilby": 6,
    # If you ever add "UniformComovingVolume-bilby", give it a new code.
}

def encode_dLprior_list(dLprior_list):
    codes = []
    for s in dLprior_list:
        if s not in DLPRIOR_CODES:
            raise ValueError(f"Unknown dLprior string: {s}")
        codes.append(DLPRIOR_CODES[s])
    return np.asarray(codes, dtype=np.int32)


# -----------------------------
# Likelihood data container
# -----------------------------

@dataclass(frozen=True)
class LikDataGauss:
    # event / GW surrogate data
    mus_s: jnp.ndarray        # (N, nd)
    cho_s: jnp.ndarray        # (N, nd, nd)
    mus_l: jnp.ndarray        # (N, ngmm, nd)
    icovs_l: jnp.ndarray      # (N, ngmm, nd, nd)
    log_dets_l: jnp.ndarray   # (N, ngmm)
    log_wts_l: jnp.ndarray    # (N, ngmm)

    # injections / selection
    m1inj: jnp.ndarray        # (ninj,)
    m2inj: jnp.ndarray        # (ninj,)
    dLinj: jnp.ndarray        # (ninj,)
    spins_inj: jnp.ndarray    # (ninj, nspin_inj) possibly (ninj,0)
    log_p_draw: jnp.ndarray   # (ninj,)
    log_p_incl: jnp.ndarray   # (ninj,)
    Ndraw: jnp.ndarray        # scalar float64

    # PE-prior bookkeeping
    labels_evt: jnp.ndarray            # (N,) int32 in [0, nchunks-1]
    prior_code_per_chunk: jnp.ndarray  # (nchunks,) int32
    all_PE_log_norms: jnp.ndarray      # (N,) float64

    # Planck15 grids for volume priors (all in Gpc, as you stated)
    zgrid_dLp: jnp.ndarray       # (K,)
    dL_grid_Planck15: jnp.ndarray  # (K,)
    dc_grid_Planck15: jnp.ndarray  # (K,)

    # bilby prior grid (in Gpc)
    dLgrid_bilby_gpc: Optional[jnp.ndarray] = None
    PE_prior_bilby_grid: Optional[jnp.ndarray] = None

    # misc constants for optional Poisson term
    Nevs_per_chunk: Optional[jnp.ndarray] = None   # (nchunks,)
    allTobs: Optional[jnp.ndarray] = None          # (nchunks,)
    # number of events
    Nobs: int = 0

    # model meta
    spin_model: str = "none"
    rate_model: str = "MD"
    mass_model: str = "PLP"
    smoothing: str = "LVK"
    simplex_repair: bool = False
    has_m2_break: bool = False
    norm_gauss: str = "uplow"
    param: str = "vanilla"
    integrate_dc: str = "trapz"
    subtract_log_p_incl: bool = False
    sample_from_pop: bool = False
    marginal_R0: bool = True


# -----------------------------
# PE prior computation (eventwise)
# -----------------------------

def _interp1d_monotonic(x, xp, fp):
    # xp strictly increasing
    return jnp.interp(x, xp, fp)

def _log_PE_prior_evt(
    dL_evt_gpc: jnp.ndarray,   # (N,)
    logd_evt: jnp.ndarray,     # (N,)
    data: LikDataGauss,
) -> jnp.ndarray:
    """
    Returns log_PE_prior_evt (N,), matching logic:
    log_PE_prior = where(mask_chunk, chunk_prior(d), log_PE_prior) - all_PE_log_norms
    """
    N = dL_evt_gpc.shape[0]
    labels = data.labels_evt
    codes = data.prior_code_per_chunk

    # Precompute Planck15 z(d) and dc(z) once (used only by volume priors)
    # In your PyMC you did: zs_Planck15 = z_from_dL(d, z_nodes, d_nodes=dL_grid_Planck15)
    # Here we invert by interpolation on the precomputed monotonic (dL(z), z) relation.
    z_planck15 = _interp1d_monotonic(dL_evt_gpc, data.dL_grid_Planck15, data.zgrid_dLp)
    dc_planck15 = _interp1d_monotonic(z_planck15, data.zgrid_dLp, data.dc_grid_Planck15)

    

    def chunk_prior_from_code(code: jnp.ndarray) -> jnp.ndarray:
        # returns vector (N,)
        # Use lax.switch for JIT-friendly branching
        def case_none(_):
            return jnp.zeros((N,), dtype=jnp.float64)

        def case_dLsq(_):
            return 2.0 * logd_evt

        def case_ucv(_):
            # log_dV/dz with Planck15 cosmology, with dc provided
            return cosmo.log_dV_dz(
                bk,
                z_planck15,
                67.9, 0.3065, -1.0,
                dc=dc_planck15,
                E=None,
            )

        def case_ucv_J(_):
            out = cosmo.log_dV_dz(
                bk, z_planck15, 67.9, 0.3065, -1.0,
                dc=dc_planck15, E=None
            )

            out = out - cosmo.log_ddL_dz(
                bk,
                z_planck15,
                67.9, 0.3065, -1.0,
                Xi0=1.0,
                n=0.0,
                dc=dc_planck15,
                param="vanilla",
            )
            return out

        def case_usf(_):
            out = cosmo.log_dV_dz(
                bk, z_planck15, 67.9, 0.3065, -1.0,
                dc=dc_planck15, E=None
            )
            return out - jnp.log1p(z_planck15)

        def case_usf_J(_):
            out = cosmo.log_dV_dz(
                bk, z_planck15, 67.9, 0.3065, -1.0,
                dc=dc_planck15, E=None
            )
            out = out - jnp.log1p(z_planck15)
            out = out - cosmo.log_ddL_dz(
                bk,
                z_planck15,
                67.9, 0.3065, -1.0,
                Xi0=1.0,
                n=0.0,
                dc=dc_planck15,
                param="vanilla",
            )
            return out

        def case_bilby(_):
            if data.dLgrid_bilby_gpc is None or data.PE_prior_bilby_grid is None:
                # hard fail inside jit is annoying; return -inf so it breaks clearly
                return jnp.full((N,), -jnp.inf, dtype=jnp.float64)
            p = _interp1d_monotonic(dL_evt_gpc, data.dLgrid_bilby_gpc, data.PE_prior_bilby_grid)
            return logclip(p)

        return lax.switch(
            code,
            [case_none, case_dLsq, case_ucv, case_ucv_J, case_usf, case_usf_J, case_bilby],
            operand=None
        )

    # Assemble per-event prior using chunk masks
    log_pe = jnp.zeros((N,), dtype=jnp.float64)

    nchunks = codes.shape[0]

    def body(i, acc):
        code_i = codes[i]
        chunk = chunk_prior_from_code(code_i)  # (N,)
        mask = (labels == i)
        acc = jnp.where(mask, chunk, acc)
        return acc

    log_pe = lax.fori_loop(0, nchunks, body, log_pe)

    # subtract per-event normalizations (fixed, same as your PyMC)
    log_pe = log_pe - data.all_PE_log_norms
    return log_pe


# -----------------------------
# GW surrogate terms (gauss proposal + GMM likelihood)
# -----------------------------

def _gw_terms_from_x(
    x: jnp.ndarray,           # (N, nd)
    data: LikDataGauss,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns:
      m1det, m2det, dLdet (all (N,))
      spins_evt (N, nspin_evt)
      log_jac_evt = gwl - pilik (N,)
      plus logd (N,) for PE prior building.
    """
    mus_s = data.mus_s
    cho_s = data.cho_s

    # samples = mus_s + sum(cho_s * x[:,None,:], axis=-1)
    samples = mus_s + jnp.einsum("nij,nj->ni", cho_s, x)

    N, nd = samples.shape

    # proposal logpdf for x and cholesky det
    log_px = -0.5 * jnp.sum(x * x, axis=1) - 0.5 * nd * jnp.log(2.0 * jnp.pi)
    log_det_L = jnp.sum(jnp.log(jnp.diagonal(cho_s, axis1=1, axis2=2)), axis=1)
    pilik = log_px - log_det_L  # (N,)

    log_Mc_det = samples[:, 0]
    logit_q = samples[:, 1]
    logd = samples[:, 2]

    if data.spin_model in ("default", "default_gauss"):
        # mixture coordinates use the raw transformed spin dims (same as your PyMC)
        X = jnp.stack(
            [log_Mc_det, logit_q, logd, samples[:, 3], samples[:, 4], samples[:, 5], samples[:, 6]],
            axis=1
        )
        d_int = 7

        # physical spins for population
        chi1 = inv_logitat(samples[:, 3])
        chi2 = inv_logitat(samples[:, 4])
        cost1 = inv_flogitat(samples[:, 5])
        cost2 = inv_flogitat(samples[:, 6])
        spins_evt = jnp.stack([chi1, chi2, cost1, cost2], axis=1)  # (N,4)

    elif data.spin_model == "none":
        X = jnp.stack([log_Mc_det, logit_q, logd], axis=1)
        d_int = 3
        spins_evt = jnp.zeros((N, 0), dtype=jnp.float64)

    else:
        raise NotImplementedError(f"spin_model={data.spin_model} not yet supported in gauss branch")

    # GMM likelihood gwl
    diff = X[:, None, :] - data.mus_l[:, :, :d_int]                 # (N,ngmm,d)
    tmp = jnp.matmul(data.icovs_l[:, :, :d_int, :d_int], diff[..., None])[..., 0]  # (N,ngmm,d)
    quad = jnp.sum(diff * tmp, axis=-1)                              # (N,ngmm)

    log_norm = -0.5 * d_int * jnp.log(2.0 * jnp.pi)
    logp_components = (
        -0.5 * quad
        + log_norm
        - 0.5 * data.log_dets_l
        + data.log_wts_l
    )
    gwl = jax.scipy.special.logsumexp(logp_components, axis=1)       # (N,)

    # detector-frame masses and distance
    Mc = jnp.exp(log_Mc_det)
    q = inv_logitat(logit_q)
    m1det, m2det = m1m2_from_Mcq(Mc, q)
    dLdet = jnp.exp(logd)  # Gpc (as per your convention)

    log_jac_evt = jnp.zeros((N,), dtype=jnp.float64)
    if (not data.sample_from_pop):
        # same as PyMC: log_jacobian -= pilik; += gwl
        log_jac_evt = gwl - pilik

    return m1det, m2det, dLdet, spins_evt, log_jac_evt, logd


# -----------------------------
# factory: build core + loglik
# -----------------------------

def make_loglik_gauss(
    core_fn: Callable,   #  _make_pop_and_sel_core(...) result 
    data: LikDataGauss,
) -> Callable:
    """
    Returns a jitted function:
      loglik(Lambda, x) -> scalar
    """
    Nobs = int(data.Nobs)

    @jax.jit
    def loglik(Lambda: jnp.ndarray, x: jnp.ndarray, lR0=0.) -> jnp.ndarray:
        # GW terms from x
        m1det, m2det, dLdet, spins_evt, log_jac_evt, logd = _gw_terms_from_x(x, data)

        # population + selection (log_mu scalar)
        logp_pop_evt, log_mu, log_var = core_fn(
            m1det, m2det, dLdet, spins_evt,
            data.m1inj, data.m2inj, data.dLinj, data.spins_inj, data.log_p_draw, data.log_p_incl,
            Lambda, data.Ndraw
        )

        # PE prior correction (eventwise)
        log_PE_prior_evt = _log_PE_prior_evt(dLdet, logd, data)

        # sum event contributions
        ll_evt = jnp.sum(logp_pop_evt + log_jac_evt - log_PE_prior_evt)

        # selection normalization
        ll = ll_evt - (Nobs * log_mu)

        # optional R0*Tobs term (only if not marginal_R0)
        if not data.marginal_R0:
            # Here we just expose the hook.
            ll += jnp.sum(data.Nevs_per_chunk * jnp.log(data.allTobs)) + Nobs * lR0
            #raise NotImplementedError("non-marginal_R0 hook not wired yet")

        # optional: detatch gradient
        log_var = lax.stop_gradient(log_var)
        
        return ll, log_var

    return loglik