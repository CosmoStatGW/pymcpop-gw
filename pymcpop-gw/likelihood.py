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

from numerical_utils import safe_logsumexp_jax, logdiffexp as logdiffexp_bk

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

def inv_flogitat_bounds(x, xmin, xmax):
    # maps R -> (xmin, xmax), elementwise.
    return xmin + (xmax - xmin) * jax.nn.sigmoid(x)

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

    # Optional event-bounded GMM coordinate transform metadata.
    # If gmm_fit_transform_mode == "event_bounded_flogit", samples drawn in
    # the Gaussian/GMM surrogate space are bounded coordinates z. They must be
    # inverse-mapped with gmm_fit_coord_bounds before conversion to physical PE
    # variables. Legacy mode keeps the historical coordinate interpretation.
    gmm_fit_coord_bounds: Optional[jnp.ndarray] = None  # (N, nd, 2)
    gmm_fit_transform_mode: str = "legacy"
    gmm_fit_coord_names: Optional[Tuple[str, ...]] = None

    # misc constants for optional Poisson term
    Nevs_per_chunk: Optional[jnp.ndarray] = None   # (nchunks,)
    allTobs: Optional[jnp.ndarray] = None          # (nchunks,)
    # number of events
    Nobs: int = 0
    logNobs: Optional[jnp.float64] = 0
    # Regularizer for selction term
    logr : Optional[jnp.float64] = 0

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
    marginal_R0: bool = True,
    taper_kind: str = "sigmoid" ,  # or "power"
    taper_p: float = 12.0           # only used if taper_kind == "power"



@dataclass(frozen=True)
class MarginalSampleData:
    m1det_pe: jnp.ndarray
    m2det_pe: jnp.ndarray
    dL_pe: jnp.ndarray
    spins_pe: jnp.ndarray
    log_PE_prior_pe: jnp.ndarray
    event_id_pe: jnp.ndarray
    Nsamples_evt: jnp.ndarray

    m1inj: jnp.ndarray
    m2inj: jnp.ndarray
    dLinj: jnp.ndarray
    spins_inj: jnp.ndarray
    log_p_draw: jnp.ndarray
    log_p_incl: jnp.ndarray
    Ndraw: jnp.ndarray

    Nevs_per_chunk: Optional[jnp.ndarray] = None
    allTobs: Optional[jnp.ndarray] = None
    Nobs: int = 0
    logNobs: Optional[jnp.float64] = 0
    logr: Optional[jnp.float64] = 0

    spin_model: str = "none"
    rate_model: str = "MD"
    mass_model: str = "DPLDP"
    smoothing: str = "LVK"
    simplex_repair: bool = False
    has_m2_break: bool = False
    norm_gauss: str = "uplow"
    param: str = "vanilla"
    integrate_dc: str = "trapz"
    subtract_log_p_incl: bool = False
    marginal_R0: bool = True
    chunk_pe: int = 0,
    
    taper_kind: str = "sigmoid" ,  # or "power"
    taper_p: float = 12.0           # only used if taper_kind == "power"




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

    Coordinate modes
    ----------------
    legacy:
        samples are interpreted as the historical fit coordinates:
            [logMc, logit(q), logdL, ...]

    event_bounded_flogit:
        samples are bounded-GMM coordinates z.  The GMM density and Gaussian
        proposal density are both evaluated in z-space.  Physical PE variables
        are obtained by inverse-bounded-flogit using per-event coordinate bounds.
    """
    mus_s = data.mus_s
    cho_s = data.cho_s

    # samples = mus_s + sum(cho_s * x[:,None,:], axis=-1)
    samples = mus_s + jnp.einsum("nij,nj->ni", cho_s, x)

    N, nd = samples.shape

    # proposal logpdf for x and Cholesky determinant.
    # nd is now the ACTIVE dimension only.
    log_px = -0.5 * jnp.sum(x * x, axis=1) - 0.5 * nd * jnp.log(2.0 * jnp.pi)
    log_det_L = jnp.sum(jnp.log(jnp.diagonal(cho_s, axis1=1, axis2=2)), axis=1)
    pilik = log_px - log_det_L  # log q_Gauss(samples)

    X = samples

    if data.spin_model in ("default", "default_gauss"):
        if nd != 7:
            raise ValueError(f"Spin model {data.spin_model} requires nd=7, got nd={nd}")
        d_int = 7
        expected_names = ("logMc", "q", "logdL", "chi_1", "chi_2", "cos_t_1", "cos_t_2")

    elif data.spin_model == "none":
        if nd != 3:
            raise ValueError(f"spin_model='none' requires nd=3 after packing, got nd={nd}")
        d_int = 3
        expected_names = ("logMc", "q", "logdL")

    else:
        raise NotImplementedError(f"spin_model={data.spin_model} not yet supported in gauss branch")

    # GMM likelihood gwl, always evaluated in the sampled surrogate coordinate.
    # In legacy mode this is the historical coordinate. In bounded mode this is z.
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

    gwl = safe_logsumexp_jax(logp_components, axis=1)

    if data.gmm_fit_transform_mode == "legacy":
        # Historical interpretation of samples.
        log_Mc_det = samples[:, 0]
        logit_q = samples[:, 1]
        logd = samples[:, 2]

        Mc = jnp.exp(log_Mc_det)
        q = inv_logitat(logit_q)
        dLdet = jnp.exp(logd)

        if data.spin_model in ("default", "default_gauss"):
            chi1 = inv_logitat(samples[:, 3])
            chi2 = inv_logitat(samples[:, 4])
            cost1 = inv_flogitat(samples[:, 5])
            cost2 = inv_flogitat(samples[:, 6])
            spins_evt = jnp.stack([chi1, chi2, cost1, cost2], axis=1)
        else:
            spins_evt = jnp.zeros((N, 0), dtype=jnp.float64)

    elif data.gmm_fit_transform_mode == "event_bounded_flogit":
        if data.gmm_fit_coord_bounds is None:
            raise ValueError("gmm_fit_coord_bounds is required for event_bounded_flogit mode")
        if data.gmm_fit_coord_names is None:
            raise ValueError("gmm_fit_coord_names is required for event_bounded_flogit mode")
        if tuple(data.gmm_fit_coord_names[:d_int]) != expected_names:
            raise NotImplementedError(
                "Unsupported bounded GMM coordinate names. "
                f"Expected {expected_names}, got {tuple(data.gmm_fit_coord_names[:d_int])}."
            )
        if data.gmm_fit_coord_bounds.shape != (N, d_int, 2):
            raise ValueError(
                "gmm_fit_coord_bounds must have shape "
                f"({N}, {d_int}, 2), got {data.gmm_fit_coord_bounds.shape}"
            )

        bounds = data.gmm_fit_coord_bounds
        xmin = bounds[:, :, 0]
        xmax = bounds[:, :, 1]

        # Direct map from bounded-GMM coordinates z to physical/derived PE variables.
        log_Mc_det = inv_flogitat_bounds(samples[:, 0], xmin[:, 0], xmax[:, 0])
        q = inv_flogitat_bounds(samples[:, 1], xmin[:, 1], xmax[:, 1])
        logd = inv_flogitat_bounds(samples[:, 2], xmin[:, 2], xmax[:, 2])

        Mc = jnp.exp(log_Mc_det)
        dLdet = jnp.exp(logd)

        if data.spin_model in ("default", "default_gauss"):
            chi1 = inv_flogitat_bounds(samples[:, 3], xmin[:, 3], xmax[:, 3])
            chi2 = inv_flogitat_bounds(samples[:, 4], xmin[:, 4], xmax[:, 4])
            cost1 = inv_flogitat_bounds(samples[:, 5], xmin[:, 5], xmax[:, 5])
            cost2 = inv_flogitat_bounds(samples[:, 6], xmin[:, 6], xmax[:, 6])
            spins_evt = jnp.stack([chi1, chi2, cost1, cost2], axis=1)
        else:
            spins_evt = jnp.zeros((N, 0), dtype=jnp.float64)

    else:
        raise NotImplementedError(
            f"Unknown gmm_fit_transform_mode={data.gmm_fit_transform_mode!r}"
        )

    m1det, m2det = m1m2_from_Mcq(Mc, q)

    log_jac_evt = jnp.zeros((N,), dtype=jnp.float64)
    if (not data.sample_from_pop):
        # same as PyMC: log_jacobian -= pilik; += gwl
        # Both gwl and pilik are densities in the sampled surrogate coordinate.
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
        logp_pop_evt, log_mu, log_var, var_log_lik_evs = core_fn(
            m1det, m2det, dLdet, spins_evt,
            data.m1inj, data.m2inj, data.dLinj, data.spins_inj, data.log_p_draw, data.log_p_incl,
            Lambda, data.Ndraw
        )

        # PE prior correction (eventwise)
        log_PE_prior_evt = _log_PE_prior_evt(dLdet, logd, data)

        # jax.debug.print("log_PE_prior_evt = {}", log_PE_prior_evt)
        # jax.debug.print("dLdet = {}", dLdet)

        # sum event contributions
        ll_evt = jnp.sum(logp_pop_evt + log_jac_evt - log_PE_prior_evt)

        # selection normalization
        sel_term = lax.cond(
                data.logr == -1,
                lambda _: log_mu,
                lambda _: jnp.logaddexp(log_mu, -data.logr ),
                operand=None
            )

        # jax.debug.print("logmu = {}", log_mu)
        # jax.debug.print("sel_term = {}", sel_term)
        
        ll = ll_evt - (Nobs * sel_term)

        # optional R0*Tobs term (only if not marginal_R0)
        if not data.marginal_R0:
            ll += jnp.sum(data.Nevs_per_chunk * jnp.log(data.allTobs)) + Nobs * lR0
            # missing: also correct var_log_lik_sel

        var_log_lik_sel = jnp.exp(log_var + 2.0 * data.logNobs)
        var_log_lik = lax.stop_gradient(var_log_lik_sel + var_log_lik_evs)
        
        return ll, var_log_lik

    return loglik




def make_loglik_marginal_samples(core_fn, data):
    Nobs = int(data.Nobs)

    @jax.jit
    def loglik(Lambda: jnp.ndarray, lR0=0.0):
        log_evt, log_mu, log_var_sel_u, var_log_lik_evs = core_fn(
            data.m1det_pe, data.m2det_pe, data.dL_pe, data.spins_pe,
            data.m1inj, data.m2inj, data.dLinj, data.spins_inj,
            data.log_p_draw, data.log_p_incl,
            Lambda, data.Ndraw,
            data.log_PE_prior_pe,
            data.event_id_pe,
            data.Nsamples_evt,
        )

        ll_evt = jnp.sum(log_evt)

        sel_term = lax.cond(
            data.logr == -1,
            lambda _: log_mu,
            lambda _: jnp.logaddexp(log_mu, -data.logr),
            operand=None,
        )

        ll = ll_evt - (Nobs * sel_term)

        if not data.marginal_R0:
            ll += jnp.sum(data.Nevs_per_chunk * jnp.log(data.allTobs)) + Nobs * lR0

        var_log_lik_sel = jnp.exp(log_var_sel_u + 2.0 * data.logNobs)
        var_log_lik = lax.stop_gradient(var_log_lik_sel + var_log_lik_evs)

        return ll, var_log_lik

    return loglik



