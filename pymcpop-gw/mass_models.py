from __future__ import annotations

import numpy as np
from numerical_utils import attrapzvec, atcumtrapz, logsumexp2, logaddexp, logdiffexp, log_sigmoid, safe_sigmoid, _interp_indices_nonuniform_safe, interp_1d_nonuniform_multiY, atinterp, safe_logsumexp_jax
from constants import max_m, _tgrid_np


try:
    import jax.numpy as jnp
    PI = jnp.pi
except Exception:
    jnp = None
    
try:
    import jax
except Exception as e:
    print(e)
    raise ValueError()


# ---------------------------------------------------------------------
# Gaussians
# ---------------------------------------------------------------------


def truncGausslowerupper_at_lpdf_safe(
    bk,
    x,
    loc,
    scale,
    xmin=0.0,
    xmax=1.0,
    eps_scale=1e-12,
    eps_Z=1e-300,
    truncate=False,
):
    scale_pos = bk.maximum(scale, eps_scale)

    za = (xmin - loc) / scale_pos
    zb = (xmax - loc) / scale_pos

    logPhia = _log_ndtr_safe(bk, za)
    logPhib = _log_ndtr_safe(bk, zb)

    hi = bk.maximum(logPhib, logPhia)
    lo = bk.minimum(logPhib, logPhia)

    logZ = logdiffexp(bk, hi, lo)
    logZ = bk.maximum(logZ, bk.log(eps_Z))

    z = (x - loc) / scale_pos
    logp = -bk.log(scale_pos) - 0.5 * bk.log(2.0 * PI) - 0.5 * z**2 - logZ

    if truncate:
        in_bounds = (x >= xmin) & (x <= xmax)
        return bk.where(in_bounds, logp, -jnp.inf)
    else:
        return logp


def _log_ndtr_safe(bk, z):
    """
    log Phi(z) using erf. Good enough for most parameter ranges.
    (If you need extreme-tail stability, can swap in a better approximation.)
    """
    sqrt2 = bk.sqrt(2.0)
    Phi = 0.5 * (1.0 + bk.erf(z / sqrt2))
    # guard against log(0)
    Phi = bk.clip(Phi, 1e-300, jnp.inf)
    return bk.log(Phi)


def _log1mexp(bk, t):
    """
    Computes log(1 - exp(-t)) for t >= 0.
    """

    log2 = bk.log(2.0)

    t = bk.maximum(t, 0.0)

    return bk.where(
        t > log2,
        bk.log1p(-bk.exp(-t)),
        bk.log(-bk.expm1(-t)),
    )

def _logdiffexp_posordered(bk, x, y):
    """
    Stable log(exp(x) - exp(y)) assuming x >= y.

    Returns -inf cleanly when x == y, including x = y = -inf.
    """

    same = x == y

    x_safe = bk.where(same, 0.0, x)
    y_safe = bk.where(same, -1e030, y)

    t = x_safe - y_safe
    out = x_safe + _log1mexp(bk, t)

    return bk.where(same, -1e030, out)


def truncGausslowerupper_at_lpdf(
    bk,
    x,
    loc,
    scale,
    xmin=0.0,
    xmax=1.0,
    truncate=False,
):
    # z-scores for truncation bounds
    za = (xmin - loc) / scale
    zb = (xmax - loc) / scale

    # log CDFs
    logPhia = _log_ndtr(bk, za)
    logPhib = _log_ndtr(bk, zb)

    # log SFs: log(1 - Phi(z)) = log Phi(-z)
    logSFa = _log_ndtr(bk, -za)
    logSFb = _log_ndtr(bk, -zb)

    # Two equivalent ways to compute log( Phi(zb) - Phi(za) ):
    #   cdf form: logdiffexp(logPhib, logPhia)
    #   sf  form: logdiffexp(logSFa, logSFb)  [better in right tail]
    #
    # Heuristic: if both bounds are on the right tail, use SF form.
    use_sf = (za > 0.0) & (zb > 0.0)

    
    logZ_cdf = _logdiffexp_posordered(bk, logPhib, logPhia)
    logZ_sf  = _logdiffexp_posordered(bk, logSFa,  logSFb)
    logZ = bk.where(use_sf, logZ_sf, logZ_cdf)

    log_floor = bk.log(bk.finfo(logZ.dtype).tiny)

    logZ = bk.where(
        bk.isfinite(logZ),
        logZ,
        log_floor,
    )

    # Gaussian logpdf
    z = (x - loc) / scale
    logp = -bk.log(scale) - 0.5 * bk.log(2.0 * PI) - 0.5 * z * z - logZ

    if truncate:
        return bk.where((x >= xmin) & (x <= xmax), logp, -jnp.inf)
    return logp


def truncGausslowerupper_at_lpdf_std(
    bk,
    x,
    loc,
    scale,
    xmin=0.0,
    xmax=1.0,
    truncate=False,
):
    # core z-scores
    za = (xmin - loc) / scale
    zb = (xmax - loc) / scale

    # log normalizer: log( Phi(zb) - Phi(za) )
    logPhia = _log_ndtr(bk, za)
    logPhib = _log_ndtr(bk, zb)

    logZ = logdiffexp(bk, logPhib, logPhia)  # assumes zb > za


    # gaussian logpdf part
    z = (x - loc) / scale
    logp = -bk.log(scale) - 0.5 * bk.log(2.0 * PI) - 0.5 * z * z - logZ

    if truncate:
        return bk.where((x >= xmin) & (x <= xmax), logp, -jnp.inf)
    return logp


def truncGausslower_at_lpdf(
    bk,
    x,
    loc,
    scale,
    xmin=0.0,
    xmax=1.0,   # kept for same signature; unused
    truncate=False,
):
    # lower-truncation z-score
    za = (xmin - loc) / scale

    # log normalizer: log(1 - Phi(za)) = log(Phi(-za))
    logZ = _log_ndtr(bk, -za)

    # gaussian logpdf part
    z = (x - loc) / scale
    logp = -bk.log(scale) - 0.5 * bk.log(2.0 * PI) - 0.5 * z * z - logZ

    if truncate:
        return bk.where(x >= xmin, logp, -jnp.inf)
    return logp


def _log_ndtr(bk, z):
    # log Phi(z) via erf; clamp only to avoid log(0)
    Phi = 0.5 * (1.0 + bk.erf(z / bk.sqrt(2.0)))
    Phi = bk.maximum(Phi, 1e-300)
    return bk.log(Phi)


def gaussian_logpdf_pair(bk, m1s, m2s, mu, sd):
    """
    Compute per-component 1D Gaussian log-pdfs for (m1, m2) given
    means mu and std-devs sd.
    """

    m1 = m1s[None, :]          # (1, N)
    m2 = m2s[None, :]          # (1, N)

    mu1 = mu[0][:, None]       # (K, 1)
    mu2 = mu[1][:, None]       # (K, 1)

    sd1 = sd[0][:, None]       # (K, 1)
    sd2 = sd[1][:, None]       # (K, 1)

  
    var1 = sd1 * sd1
    var2 = sd2 * sd2

    diff1 = m1 - mu1                      # (K,N)
    diff2 = m2 - mu2                      # (K,N)

    # 1D Gaussian logpdfs 
    const = -0.5 * bk.log(2.0 * PI)

    logp1 = const - 0.5 * bk.log(var1) - 0.5 * (diff1 * diff1 / var1)
    logp2 = const - 0.5 * bk.log(var2) - 0.5 * (diff2 * diff2 / var2)


    return logp1, logp2

def gaussian_logpdf_pair_corr_xy(
    bk,
    m1s,
    m2s,
    mu,
    sd,
    rho,
):
    """
    Per-component correlated Gaussian log-pdf in the first two coordinates.

    Coordinates:
        x = m1s = logMc
        y = m2s = logitq

    Component covariance in (x,y):
        Sigma_k = [[sx_k^2, rho_k sx_k sy_k],
                   [rho_k sx_k sy_k, sy_k^2]]

    Returns
    -------
    logp_xy : shape (K, N)
        correlated 2D Gaussian logpdf in (x,y)
    """

    x = m1s[None, :]          # (1, N)
    y = m2s[None, :]          # (1, N)

    mux = mu[0][:, None]      # (K, 1)
    muy = mu[1][:, None]      # (K, 1)

    sx = bk.maximum(sd[0][:, None], 1e-300)
    sy = bk.maximum(sd[1][:, None], 1e-300)

    rho = rho[:, None]
    rho = bk.clip(rho, -0.999999, 0.999999)

    one_minus_r2 = bk.maximum(1.0 - rho * rho, 1e-12)

    ax = (x - mux) / sx
    ay = (y - muy) / sy

    Q = (ax * ax + ay * ay - 2.0 * rho * ax * ay) / one_minus_r2

    logp_xy = (
        -bk.log(2.0 * PI)
        -bk.log(sx)
        -bk.log(sy)
        -0.5 * bk.log(one_minus_r2)
        -0.5 * Q
    )


    return logp_xy


def gaussian_logpdf_pair_trunc_q(
    bk,
    xs,
    qs,
    mu,
    sd,
):
    """
    Per-component independent Gaussian log-pdfs in (x, q), with q
    represented by a Gaussian truncated to 0 <= q <= 1.

    Coordinates:
        x = logMc
        q = m2 / m1

    Returns
    -------
    logp_x : shape (K, N)
    logp_q : shape (K, N)
    """

    x = xs[None, :]
    q = qs[None, :]

    mux = mu[0][:, None]
    muq = mu[1][:, None]

    sx = bk.maximum(sd[0][:, None], 1e-300)
    sq = bk.maximum(sd[1][:, None], 1e-300)

    ax = (x - mux) / sx
    logp_x = (
        -0.5 * bk.log(2.0 * PI)
        -bk.log(sx)
        -0.5 * ax * ax
    )

    logp_q = truncGausslowerupper_at_lpdf_safe(
        bk,
        q,
        muq,
        sq,
        xmin=0.0,
        xmax=1.0,
        truncate=True,
    )


    return logp_x, logp_q


def gaussian_logpdf_pair_corr_xy_trunc_q(
    bk,
    xs,
    qs,
    mu,
    sd,
    rho,
):
    """
    Per-component correlated Gaussian log-pdf in (x, q), truncated only
    in q to 0 <= q <= 1.

    The truncation normalization is the marginal q normalizer:

        Z_k = Phi((1 - mu_q) / sigma_q) - Phi((0 - mu_q) / sigma_q)

    This remains correct with correlation because the truncation is only
    on the marginal q coordinate.

    Coordinates:
        x = logMc
        q = m2 / m1
    """

    x = xs[None, :]
    q = qs[None, :]

    mux = mu[0][:, None]
    muq = mu[1][:, None]

    sx = bk.maximum(sd[0][:, None], 1e-300)
    sq = bk.maximum(sd[1][:, None], 1e-300)

    rho = bk.clip(rho[:, None], -0.999999, 0.999999)
    one_minus_r2 = bk.maximum(1.0 - rho * rho, 1e-12)

    ax = (x - mux) / sx
    aq = (q - muq) / sq

    Q = (ax * ax + aq * aq - 2.0 * rho * ax * aq) / one_minus_r2

    za = (0.0 - muq) / sq
    zb = (1.0 - muq) / sq

    logPhia = _log_ndtr_safe(bk, za)
    logPhib = _log_ndtr_safe(bk, zb)

    hi = bk.maximum(logPhib, logPhia)
    lo = bk.minimum(logPhib, logPhia)

    logZ = logdiffexp(bk, hi, lo)
    logZ = bk.maximum(logZ, bk.log(1e-300))

    logp_xy = (
        -bk.log(2.0 * PI)
        -bk.log(sx)
        -bk.log(sq)
        -0.5 * bk.log(one_minus_r2)
        -0.5 * Q
        -logZ
    )

    in_bounds = (q >= 0.0) & (q <= 1.0)
    logp_xy = bk.where(in_bounds, logp_xy, -jnp.inf)


    return logp_xy

M = 40
gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(M)
GH_NODES = gh_nodes_np
GH_WEIGHTS = gh_weights_np


def dpuc_component_logZ_xy_physical(
    bk,
    mux,
    sdx,
    muy,
    sdy,
    *,
    rho=None,
    alpha=0.0,
    mmin,
    mmax,
    DP_truncate=True,
    DP_m1_env=False,
    taper_kind="power",
    taper_p=15.0,
    taper_s=0.003,
    eps_mass=1e-12,
    rho_clip=0.999999,
):
    """
    Per-component physical normalization for DPUC mass kernels.

    Coordinates:
        x = log Mc_src
        y = logit(q)

    Component density before normalization:
        N_k(x,y) * R(m1(x,y), m2(x,y))

    where
        R = G_low(m2; mmin) * G_high(m1; mmax) * m1^(-alpha)

    depending on DP_truncate / DP_m1_env.

    Returns
    -------
    logZk : shape (K,)
        log Z_k, with

        Z_k = E_{(x,y)~N_k}[R(m1(x,y),m2(x,y))]

    Supports both diagonal and correlated x-y kernels.
    For diagonal kernels, pass rho=None.
    """

    if (not DP_truncate) and (not DP_m1_env):
        return jnp.zeros_like(mux, dtype=jnp.float64)

    gh_nodes = jnp.asarray(GH_NODES, dtype=jnp.float64)
    gh_weights = jnp.asarray(GH_WEIGHTS, dtype=jnp.float64)

    mux = jnp.asarray(mux, dtype=jnp.float64)
    muy = jnp.asarray(muy, dtype=jnp.float64)

    sdx = jnp.maximum(jnp.asarray(sdx, dtype=jnp.float64), 1e-300)
    sdy = jnp.maximum(jnp.asarray(sdy, dtype=jnp.float64), 1e-300)

    K = mux.shape[0]
    M = gh_nodes.shape[0]

    a = gh_nodes[None, :, None]  # (1, M, 1)
    b = gh_nodes[None, None, :]  # (1, 1, M)

    sqrt2 = jnp.sqrt(2.0)

    x = mux[:, None, None] + sqrt2 * sdx[:, None, None] * a

    if rho is None:
        y = muy[:, None, None] + sqrt2 * sdy[:, None, None] * b
    else:
        rho = jnp.asarray(rho, dtype=jnp.float64)
        rho = jnp.clip(rho, -rho_clip, rho_clip)

        one_minus_r2 = jnp.maximum(1.0 - rho * rho, 1e-12)
        y_std = rho[:, None, None] * a + jnp.sqrt(one_minus_r2)[:, None, None] * b
        y = muy[:, None, None] + sqrt2 * sdy[:, None, None] * y_std

    # GH expectation weights for E[f(X,Y)] under standard normals.
    log_w_gh = (
        jnp.log(gh_weights[None, :, None])
        + jnp.log(gh_weights[None, None, :])
        - jnp.log(jnp.pi)
    )

    # Stable transform (x,y) -> (m1,m2)
    q = jax.nn.sigmoid(y)
    q = jnp.clip(q, 1e-10, 1.0 - 1e-10)

    logq = jnp.log(q)
    log1pq = jnp.log1p(q)

    # Mc = exp(x)
    # m1 = Mc * (1+q)^(1/5) / q^(3/5)
    # m2 = Mc * (1+q)^(1/5) * q^(2/5)
    log_m1 = x + 0.2 * log1pq - 0.6 * logq
    log_m2 = x + 0.2 * log1pq + 0.4 * logq

    # Avoid overflow in exp for ridiculous quadrature tails.
    log_m1 = jnp.clip(log_m1, -700.0, 700.0)
    log_m2 = jnp.clip(log_m2, -700.0, 700.0)

    m1 = jnp.exp(log_m1)
    m2 = jnp.exp(log_m2)

    m1_safe = jnp.maximum(m1, eps_mass)
    m2_safe = jnp.maximum(m2, eps_mass)

    logR = jnp.zeros_like(m1_safe)

    if DP_truncate:
        eps_trunc = 1e-8
        mmin_safe = jnp.maximum(mmin, eps_mass)
        mmax_safe = jnp.maximum(mmax, mmin_safe * (1.0 + eps_trunc))

        logR = logR + log_low_gate(
            bk,
            m2_safe,
            mmin_safe,
            kind=taper_kind,
            p=taper_p,
            s=taper_s,
        )

        logR = logR + log_high_gate(
            bk,
            m1_safe,
            mmax_safe,
            kind=taper_kind,
            p=taper_p,
            s=taper_s,
        )

    if DP_m1_env:
        logR = logR - alpha * jnp.log(m1_safe)

    logR = jnp.where(jnp.isfinite(logR), logR, -1e30)

    log_terms = (log_w_gh + logR).reshape((K, M * M))
    logZk = safe_logsumexp_jax(log_terms, axis=1)

    return logZk


def dpuc_mixture_logZ_xy_pl_gate(
    bk,
    mux,
    sdx,
    muy,
    sdy,
    logw,
    *,
    rho=None,
    alpha,
    beta,
    m1_low,
    m2_low,
    mmax,
    deltam_PL_m1,
    deltam_PL_m2,
    deltam_PL_high,
    DP_truncate,
    DP_m1_env,
    DP_q_env,
    taper_kind="sigmoid",
    taper_p=12.0,
    eps_mass=1e-12,
    eps_q=1e-10,
    return_components=False,
):
    """
    Native-coordinate normalization for the retained DPUC mass kernels.

    Computes

        Z_DPUC = int dx dy p_GMM(x,y) G(m1(x,y),m2(x,y))

    with

        x = log Mc
        y = logit(q)

    and mass-plane gate

        G = L(m1; mmin, deltam_PL)
          * L(m2; mmin, deltam_PL)
          * H(m1; mmax)

    plus optional m1^{-alpha} and q^beta envelopes.

    This is the normalization of the whole gated GMM mixture:

        logZ_DPUC = logsumexp_k[logw_k + logZ_k]

    not component-normalized mixture semantics.

    Returns
    -------
    logZ_DPUC : scalar
        Default return value.
    logZk : shape (K,)
        Returned when return_components=True.
    """

    # If no modifier is active, every component integrates to one.
    if (not DP_truncate) and (not DP_m1_env) and (not DP_q_env):
        logZk = jnp.zeros_like(logw, dtype=jnp.float64)
        if return_components:
            return logZk
        return safe_logsumexp_jax(logw + logZk, axis=0)

    sdx = bk.maximum(sdx, 1.0e-300)
    sdy = bk.maximum(sdy, 1.0e-300)

    gh_nodes = jnp.asarray(GH_NODES, dtype=jnp.float64)
    gh_weights = jnp.asarray(GH_WEIGHTS, dtype=jnp.float64)

    log_gh_weights = bk.log(gh_weights)
    sqrt2 = bk.sqrt(2.0)

    # Shapes:
    #   a : (1, M, 1)
    #   b : (1, 1, M)
    a = gh_nodes[None, :, None]
    b = gh_nodes[None, None, :]

    mux = mux[:, None, None]
    muy = muy[:, None, None]

    sdx = sdx[:, None, None]
    sdy = sdy[:, None, None]

    x = mux + sqrt2 * sdx * a

    if rho is None:
        y = muy + sqrt2 * sdy * b
    else:
        rho = bk.clip(rho[:, None, None], -0.999999, 0.999999)
        omr2 = bk.maximum(1.0 - rho * rho, 1.0e-12)
        y_std = rho * a + bk.sqrt(omr2) * b
        y = muy + sqrt2 * sdy * y_std

    # Product Gauss-Hermite weights for E_N[f(x,y)].
    log_w_gh = (
        log_gh_weights[None, :, None]
        + log_gh_weights[None, None, :]
        - bk.log(jnp.pi)
    )

    q = bk.sigmoid(y)
    q = bk.clip(q, eps_q, 1.0 - eps_q)

    logq = bk.log(q)
    logfac = 0.2 * bk.log1p(q)

    # m1 = Mc (1+q)^(1/5) q^(-3/5)
    # m2 = Mc (1+q)^(1/5) q^( 2/5)
    log_m1 = x + logfac - 0.6 * logq
    log_m2 = x + logfac + 0.4 * logq

    m1 = bk.exp(log_m1)
    m2 = bk.exp(log_m2)

    m1_safe = bk.maximum(m1, eps_mass)
    m2_safe = bk.maximum(m2, eps_mass)

    logG = bk.zeros_like(m1_safe)

    if DP_truncate:
        eps_trunc = 1.0e-8
        m1_low_safe = jnp.maximum(m1_low, eps_mass)
        m2_low_safe = jnp.maximum(m2_low, eps_mass)
        mmax_safe = jnp.maximum(mmax, m1_low_safe * (1.0 + 1.0e-8))
        
        deltam_PL_m1_safe = bk.maximum(deltam_PL_m1, eps_mass)
        deltam_PL_m2_safe = bk.maximum(deltam_PL_m2, eps_mass)
        deltam_PL_high_safe = bk.maximum(deltam_PL_high, eps_mass)

        logG = logG + (
            log_low_gate_PL_DPUC(
                bk,
                m1_safe,
                m1_low_safe,
                deltam_PL_m1_safe,
            )
            +
            log_low_gate_PL_DPUC(
                bk,
                m2_safe,
                m2_low_safe,
                deltam_PL_m2_safe,
            )
            +
            log_high_gate_PL_DPUC(
                bk,
                m1_safe,
                mmax_safe,
                deltam_PL_high_safe,
            )
        )

    if DP_m1_env:
        logG = logG - alpha * bk.log(m1_safe)

    if DP_q_env:
        logG = logG + beta * bk.log(q)

    valid = (
        bk.isfinite(m1_safe)
        & bk.isfinite(m2_safe)
        & bk.isfinite(logG)
        & (m1_safe > 0.0)
        & (m2_safe > 0.0)
        & (m2_safe < m1_safe)
    )

    logG = bk.where(valid, logG, -jnp.inf)

    K = mux.shape[0]
    log_terms = (log_w_gh + logG).reshape((K, -1))
    logZk = safe_logsumexp_jax(log_terms, axis=1)

    if return_components:
        return logZk

    return safe_logsumexp_jax(logw + logZk, axis=0)

    
# ---------------------------------------------------------------------
# Smoothings
# ---------------------------------------------------------------------

def logS_PLP_std(bk, m, deltam, ml, eps=1e-12):
    t = (m - ml) / bk.maximum(deltam, eps)
    t = bk.clip(t, eps, 1.0 - eps)
    S = t * t * (3.0 - 2.0 * t)
    return bk.log(bk.clip(S, eps, 1.0))


def logS_PLP(bk, m, deltam, ml, eps=1e-12):
    d = bk.maximum(deltam, eps)
    t = (m - ml) / d

    tmid = bk.clip(t, eps, 1.0 - eps)
    log_mid = 2.0 * bk.log(tmid) + bk.log(3.0 - 2.0 * tmid)

    return bk.where(
        t <= 0.0,
        -jnp.inf,
        bk.where(t >= 1.0, 0.0, log_mid),
    )


def logS_PLP_LVK(bk, m, deltam, ml):
    maskL = m <= ml
    maskU = m >= (ml + deltam)
    maskM = ~(maskL | maskU)

    s = bk.where(maskL, -jnp.inf, 0.0)

    mid = bk.log(1.0 / (1.0 + bk.exp(deltam / (m - ml) + deltam / (m - ml - deltam))))
    s1 = bk.where(maskM, mid, s)

    return s1


def log_low_gate(bk, x, low, kind="sigmoid", p=12.0, s=None):
    if kind == "sigmoid":
        return log_sigmoid(bk, x, low, s)

    elif kind == "power":
        # -log(1 + (low/x)^p)
        logx = bk.log(bk.maximum(x, 1e-12))
        return -bk.logaddexp(0.0, p * (bk.log(low) - logx))

    elif kind == "none":
        return bk.where( x<low, -1e030, 0.0 )
    else:
        raise ValueError(f"Unknown taper kind: {kind}")


def log_high_gate(bk, x, high, kind="sigmoid", p=12.0, s=None):
    if kind == "sigmoid":
        return log_sigmoid(bk, high, x, s)

    elif kind == "power":
        # -log(1 + (x/high)^p)
        logx = bk.log(bk.maximum(x, 1e-12))
        return -bk.logaddexp(0.0, p * (logx - bk.log(high)))
    elif kind == "none":
        return bk.where( x>high, -1e030, 0.0 )
    else:
        raise ValueError(f"Unknown taper kind: {kind}")


def _logit_scalar(bk, p):
    return bk.log(p / (1.0 - p))


def pl_dpuc_low_sigmoid_params(
    bk,
    mmin,
    deltam,
    *,
    eps_low=1.0e-6,
    eps_on=1.0e-3,
    eps=1.0e-12,
):
    """
    Parameters for a single low-edge sigmoid L(m).

    The construction enforces
        L(mmin) = eps_low
        L(mmin + deltam) = 1 - eps_on
    using the existing log_low_gate/log_sigmoid convention.
    """

    d = bk.maximum(deltam, eps)
    a = _logit_scalar(bk, eps_low)
    b = _logit_scalar(bk, 1.0 - eps_on)

    s = d / (b - a)
    center = mmin - s * a

    return center, s


def pl_dpuc_high_sigmoid_params(
    bk,
    mmax,
    deltam_high,
    *,
    eps_high=1.0e-6,
    p_at_mmax=0.99,
    eps=1.0e-12,
):
    """
    Parameters for the shared DPUC high-edge sigmoid H(m).

    The construction enforces
        H(mmax) = p_at_mmax
        H(mmax + deltam_high) = eps_high

    using the existing log_high_gate/log_sigmoid convention.
    """

    width = bk.maximum(deltam_high, eps)
    a = _logit_scalar(bk, p_at_mmax)
    b = _logit_scalar(bk, eps_high)

    s = width / (a - b)
    center = mmax + s * a

    return center, s


def log_low_gate_PL_DPUC(bk, x, mmin, deltam):
    center, s = pl_dpuc_low_sigmoid_params(bk, mmin, deltam)
    return log_low_gate(bk, x, center, kind="sigmoid", s=s)


def log_high_gate_PL_DPUC(bk, x, mmax, deltam_high):
    center, s = pl_dpuc_high_sigmoid_params(bk, mmax, deltam_high)
    return log_high_gate(bk, x, center, kind="sigmoid", s=s)

# ---------------------------------------------------------------------
# Secondary mass
# ---------------------------------------------------------------------


def logpdfm2_PLP_noreg(
    bk,
    m,
    beta,
    deltam,
    ml,
    *,
    m_g=45.0,
    w_g=80.0,
    sig_g_low=5.0,
    sig_g_high=5.0,
    has_m2_break=False,
    smoothing="poly",
    taper_kind = 'power',
    taper_p=12,
    
):
    if smoothing == "LVK":
        lS = logS_PLP_LVK(bk, m, deltam, ml)
    else:
        lS = logS_PLP(bk, m, deltam, ml)

        lpdfval = beta * bk.log(m) + lS
        
        if not has_m2_break:
            return lpdfval
        
        if taper_kind == "sigmoid":
            left_edge  = 1.0 - safe_sigmoid(bk, m, m_g, sig_g_low)
            right_edge = safe_sigmoid(bk, m, m_g + w_g, sig_g_high)
        
            mask = bk.log(left_edge + right_edge)
        
        elif taper_kind == "power":
            log_left_edge = log_high_gate(
                bk, m, m_g,
                kind="power",
                p=taper_p,
            )
        
            log_right_edge = log_low_gate(
                bk, m, m_g + w_g,
                kind="power",
                p=taper_p,
            )
        
            # log(exp(log_left_edge) + exp(log_right_edge))
            mask = bk.logaddexp(log_left_edge, log_right_edge)
        
        else:
            raise ValueError(f"Unknown taper_kind: {taper_kind}")
        
        return mask + lpdfval

def logpdfm2_PLP_reg(
    bk,
    m,
    beta,
    deltam,
    ml,
    *,
    sig_l=0.05,
    m_g=45.0,
    w_g=80.0,
    sig_g_low=5.0,
    sig_g_high=5.0,
    has_m2_break=False,
    smoothing="LVK",
    taper_kind="sigmoid",
    taper_p=12.0,
):


    return (
        logpdfm2_PLP_noreg(
            bk,
            m,
            beta,
            deltam,
            ml,
            m_g=m_g,
            w_g=w_g,
            sig_g_low=sig_g_low,
            sig_g_high=sig_g_high,
            has_m2_break=has_m2_break,
            smoothing=smoothing,
            taper_kind=taper_kind,
            taper_p=taper_p,
        )
        + log_low_gate(
            bk,
            m,
            ml,
            kind=taper_kind,
            p=taper_p,
            s=sig_l,
        )
    )


def logpdfq_PL_reg(
    bk,
    m1,
    m2,
    beta,
    m2_low,
):
    """
     secondary mass term in q-space, returned in (m1,m2) variables:
      q = m2/m1
      p(q|m1) ∝ q^beta, q in [m2_low/m1, 1]
      log p(m2|m1) = log p(q|m1) - log(m1)

    Uses analytic truncated power-law normalization via log_norm_truncated_pl.
    """
    q = m2 / m1
    qmin = m2_low / m1

    # analytic norm for ∫ q^beta dq on [qmin, 1]
    # The normalizer integrates m^{-alpha}, so use alpha = -beta.
    lnorm_q = log_norm_truncated_pl(bk, -beta, qmin, 1.0)

    lpdf_q = beta * bk.log(q) - lnorm_q

    # Jacobian: dq/dm2 = 1/m1
    lpdf_m2_given_m1 = lpdf_q - bk.log(m1)

    # hard support (fastest / simplest)
    valid = (m1 > 0.0) & (m2 >= m2_low) & (m2 <= m1)

    return bk.where(valid, lpdf_m2_given_m1, -1e030)


# ---------------------------------------------------------------------
#Power Law Plus Peak with regularized edges (PLPreg)
# ---------------------------------------------------------------------

def log_norm_truncated_pl_num_simple(bk, alpha, mmin, mmax, eps=1e-12, t_floor=1e-12):
    """
    log ∫_{mmin}^{mmax} m^{-alpha} dm
    = log( (mmax^(1-α) - mmin^(1-α)) / (1-α) ).
    """
 
    t = 1. - alpha  # t = 1 - α

    # α ≠ 1: log( |mmax^t - mmin^t| ) - log( |t| )
    num = bk.pow(mmax, t) - bk.pow(mmin, t)
    log_not1 = bk.log(bk.abs(num)) - bk.log(bk.abs(t))

    # α = 1: log( log(mmax/mmin) )

    return log_not1 #bk.switch(close, log_eq1, log_not1)

    
def logpdfm1_PLP_reg(bk, m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass,
                    sl=0.05, sh=0.05, smoothing="LVK",
                    taper_kind="sigmoid", taper_p=12.0):

    log_base = logpdfm1_PLP_noreg(
        bk, m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass,
        smoothing=smoothing, taper_kind=taper_kind, taper_p=taper_p
    )

    log_low  = log_low_gate(bk, m, ml, kind=taper_kind, p=taper_p, s=sl)
    log_high = log_high_gate(bk, m, mh, kind=taper_kind, p=taper_p, s=sh)

    return log_base + log_low + log_high
    

def logpdfm1_PLP_noreg(bk, m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing='LVK'):

     
    log_norm = log_norm_truncated_pl_num_simple(bk, alpha, ml, mh) #norm_truncated_pl_num(alpha, ml, mh)
    log_trunc_component =  -alpha*bk.log(m) - log_norm #1./(m**alpha)/norm
    log_gauss_component = -0.5 * bk.square((m - muMass) / sigmaMass) - bk.log(sigmaMass) - 0.5 * bk.log(2*PI)
 
    if smoothing=='LVK':
        lS = logS_PLP_LVK(bk, m, deltam, ml)
    else:
        lS = logS_PLP(bk, m, deltam, ml)
        
    result = bk.logaddexp( bk.log1p(-lambdaPeak) + log_trunc_component, bk.log(lambdaPeak) + log_gauss_component ) + lS
 
    return result


def logC_PLP_reg(bk, m, beta, deltam, ml, smoothing="LVK", taper_kind="sigmoid", taper_p=12.0):
    _tgrid = _tgrid_np
    xx = ml + (max_m - ml) * _tgrid

    l2 = logpdfm2_PLP_reg(
        bk, xx, beta, deltam, ml,
        smoothing=smoothing, taper_kind=taper_kind, taper_p=taper_p
    )

    l2 = bk.where(bk.isfinite(l2), l2, -1e30)
    a = bk.max(l2)
    p2 = bk.exp(l2 - a)

    cdf_scaled = atcumtrapz(bk, p2, xx)
    cdf_scaled = bk.maximum(bk.where(bk.isfinite(cdf_scaled), cdf_scaled, 0.0), 1e-300)

    return atinterp(bk, m, xx[1:], bk.log(cdf_scaled) + a)


def logNorm_PLP_reg(bk, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass,
                    smoothing="LVK", taper_kind="sigmoid", taper_p=12.0):
    _tgrid = _tgrid_np
    ms = ml + (mh - ml) * _tgrid
    lpdf = logpdfm1_PLP_noreg(
        bk, ms, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass,
        smoothing=smoothing
    )


    lpdf = bk.where(bk.isfinite(lpdf), lpdf, -1e30)
    a = bk.max(lpdf)
    ps = bk.exp(lpdf - a)
    integ = attrapzvec(bk, ps, ms)
    return bk.where((integ > 0.0) & bk.isfinite(integ), a + bk.log(integ), -jnp.inf)


def logpdf_PLP_reg(bk, theta, lambdaBBHmass, smoothing="LVK", taper_kind="sigmoid", taper_p=12.0):
    m1, m2 = theta
    lambdaPeak, alpha, beta, deltam, ml, mh, muMass, sigmaMass = lambdaBBHmass

    lpdfm1 = logpdfm1_PLP_reg(bk, m1, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass,
                              smoothing=smoothing, taper_kind=taper_kind, taper_p=taper_p)
    lpdfm2 = logpdfm2_PLP_reg(bk, m2, beta, deltam, ml,
                              smoothing=smoothing, taper_kind=taper_kind, taper_p=taper_p)
    ln = logNorm_PLP_reg(bk, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass,
                         smoothing=smoothing, taper_kind=taper_kind, taper_p=taper_p)
    lC = logC_PLP_reg(bk, m1, beta, deltam, ml,
                      smoothing=smoothing, taper_kind=taper_kind, taper_p=taper_p)

    finite = bk.isfinite(lpdfm1) & bk.isfinite(lpdfm2) & bk.isfinite(ln) & bk.isfinite(lC)
    val = bk.where(finite, lpdfm1, 0.0) + bk.where(finite, lpdfm2, 0.0) - bk.where(finite, ln, 0.0) - bk.where(finite, lC, 0.0)
    return bk.where(finite, val, -1e30)        


# ---------------------------------------------------------------------
# Double Power Law plus Double Peak (DPLDP)
# ---------------------------------------------------------------------


def logpdfm1_DPLDP(
    bk, m1, alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2,
    m1_low, m_high, delta_m1, lambda0, lambda1, lambda2, epsilon,
    smoothing="LVK", simplex_repair=False, eps_w=1e-15,
    sl=0.1, sh=1.0, norm_gauss="uplow",
    taper_kind="sigmoid", taper_p=12.0,
):


    if not simplex_repair:
        log_lambda0 = bk.log(lambda0)
        log_lambda1 = bk.log(lambda1)
        log_lambda2 = bk.log(lambda2)
    else:
        # Simplex repair
        lam0 = bk.clip(lambda0, eps_w, 1.0 - eps_w)
        lam1 = bk.clip(lambda1, eps_w, 1.0 - eps_w)
        lam2_raw = 1.0 - lam0 - lam1

        lam2 = eps_w + bk.log1p(bk.exp(lam2_raw - eps_w))

        denom = lam0 + lam1 + lam2
        lam0 = lam0 / denom
        lam1 = lam1 / denom
        lam2 = lam2 / denom

        log_lambda0 = bk.log(lam0)
        log_lambda1 = bk.log(lam1)
        log_lambda2 = bk.log(lam2)

    log_ppl = log_broken_power_law_DPLDP_pdf(
        bk, m1, alpha1, alpha2, mb, m1_low, m_high, epsilon=epsilon
    )

    if norm_gauss == "uplow":
        log_pnorm1 = truncGausslowerupper_at_lpdf(bk, m1, mu1, sigma1, xmin=m1_low, xmax=m_high, truncate=True)
        log_pnorm2 = truncGausslowerupper_at_lpdf(bk, m1, mu2, sigma2, xmin=m1_low, xmax=m_high, truncate=True)

    elif norm_gauss == "low-once":
        log_pnorm1 = truncGausslower_at_lpdf(bk, m1, mu1, sigma1, xmin=m1_low, truncate=True)
        log_pnorm2 = -0.5 * ((m1 - mu2) / sigma2) ** 2 - bk.log(sigma2) - 0.5 * bk.log(2.0 * PI)

    elif norm_gauss == "none":
        log_pnorm1 = -0.5 * ((m1 - mu1) / sigma1) ** 2 - bk.log(sigma1) - 0.5 * bk.log(2.0 * PI)
        log_pnorm2 = -0.5 * ((m1 - mu2) / sigma2) ** 2 - bk.log(sigma2) - 0.5 * bk.log(2.0 * PI)

    else:
        raise ValueError("norm_gauss can be uplow, low-once, or none (others not implemented here)")

    if smoothing == "LVK":
        log_S = logS_PLP_LVK(bk, m1, delta_m1, m1_low)
    else:
        log_S = logS_PLP(bk, m1, delta_m1, m1_low)

    term0 = log_lambda0 + log_ppl
    term1 = log_lambda1 + log_pnorm1
    term2 = log_lambda2 + log_pnorm2


    log_mix = logsumexp2(bk, logsumexp2(bk, term0, term1), term2)

    # gate
    log_gate = log_low_gate(bk, m1, m1_low, kind=taper_kind, p=taper_p, s=sl) \
         + log_high_gate(bk, m1, m_high, kind=taper_kind, p=taper_p, s=sh)

    return log_S + log_mix + log_gate


def log_broken_power_law_DPLDP_pdf(
    bk,
    m1,
    alpha1,
    alpha2,
    mb,
    m1_low,
    m_high,
    sh=1.,
    sl=0.1,
    epsilon=0.1,
    eps=1e-12,
    eps_w=1e-12,
    t_floor=1e-12,
):
    
    mb_pos = bk.maximum(mb, eps)
    m1_pos = bk.maximum(m1, eps)

    # safe normalization, in case it is needed
    log_N = log_broken_pl_norm_DPLDP(bk, alpha1, alpha2, mb_pos, m1_low, m_high, ) 
    
    log_m1_over_mb = bk.log(m1_pos / mb_pos)
    log_val1 = -alpha1 * log_m1_over_mb
    log_val2 = -alpha2 * log_m1_over_mb
    
    w = safe_sigmoid(bk, -m1_pos, -mb_pos, epsilon)

    
    ## alternative: Compute log normalization constant directly
    

    log_w = bk.log(w)
    log_1mw = bk.log1p(-w)

    log_mix_val = logaddexp(bk, log_w + log_val1, log_1mw + log_val2)
    
    return log_mix_val - log_N


def log_norm_truncated_pl(bk, alpha, mmin, mmax):
    """
    log ∫_{mmin}^{mmax} m^{-alpha} dm
    Assumes: alpha != 1, 0 < mmin < mmax
    """
    t = 1.0 - alpha  # != 0 by assumption
    b = bk.log(mmin)
    delta = bk.log(mmax) - b  # = log(mmax/mmin) > 0

    # log( (mmax^t - mmin^t) / t )
    # = t*log(mmin) + log(|expm1(t*log(mmax/mmin))|) - log(|t|)
    return t * b + bk.log(bk.abs(bk.expm1( t * delta))) - bk.log(bk.abs(t))


def log_broken_pl_norm_DPLDP(bk, alpha1, alpha2, mb, m1_low, m_high):
    """
    Broken power-law normalization around mb:
      for m < mb: slope alpha1 in u=m/mb on [u_low, 1]
      for m > mb: slope alpha2 in u=m/mb on [1, u_high]

    Returns log(Z), where Z = mb * (I1 + I2).
    Assumes: alpha1!=1, alpha2!=1, 0<m1_low<m_high, mb>0
    """
    u_low  = m1_low / mb
    u_high = m_high / mb

    logI1 = bk.where(
        u_low < 1.0,
        log_norm_truncated_pl(bk, alpha1, u_low, 1.0),
        -jnp.inf,
    )
    logI2 = bk.where(
        u_high > 1.0,
        log_norm_truncated_pl(bk, alpha2, 1.0, u_high),
        -jnp.inf,
    )

    return bk.log(mb) + logaddexp(bk, logI1, logI2)


def log_broken_pl_norm_DPLDP_safe(bk, alpha1, alpha2, mb, m1_low, m_high, eps=1e-12, t_floor=1e-12):
    mb_pos = bk.maximum(mb, eps)

    u_low = bk.maximum(m1_low / mb_pos, eps)
    u_high = bk.maximum(m_high / mb_pos, u_low * (1.0 + 1e-12))

    one = 1.0

    # NOTE: bk.switch not in our minimal backend yet; use bk.where with scalars/tensors.
    logI1 = bk.where(
        u_low < one,
        log_norm_truncated_pl_num_alpha1_safe(bk, alpha1, u_low, one, eps=eps, t_floor=t_floor),
        -jnp.inf,
    )
    logI2 = bk.where(
        u_high > one,
        log_norm_truncated_pl_num_alpha1_safe(bk, alpha2, one, u_high, eps=eps, t_floor=t_floor),
        -jnp.inf,
    )

    return bk.log(mb_pos) + logaddexp(bk, logI1, logI2)


def log_norm_truncated_pl_num_alpha1_safe(bk, alpha, mmin, mmax, eps=1e-12, t_floor=1e-12):
    mmin_c = bk.clip(mmin, eps, jnp.inf)
    mmax_c = bk.clip(mmax, eps, jnp.inf)
    mmax_c = bk.maximum(mmax_c, mmin_c * (1.0 + 1e-12))

    t = 1.0 - alpha
    # avoid exactly zero
    
    t_safe = bk.where(bk.abs(t) < t_floor, bk.where(t >= 0.0, t_floor, -t_floor), t)

    b = bk.log(mmin_c)
    delta = bk.log(mmax_c) - b

    # need expm1; if backend doesn’t expose expm1, we can do exp(x)-1 safely later.
    # For now assume your backends have expm1 or you’ll add it.
    return t_safe * b + bk.log(bk.abs(bk.expm1(t_safe * delta))) - bk.log(bk.abs(t_safe))


def logpdf_DPLDP(bk, theta, lambdaBBHmass, force_m2_less_than_m1=False, has_m2_break=False,
                 smoothing="LVK", interp_vals=None, interp_grids=None, norm=True,
                 simplex_repair=False, norm_gauss="uplow", taper_kind="sigmoid", taper_p=12.0):
    m1, m2 = theta
    (
        alpha1,
        alpha2,
        mb,
        mu1,
        sigma1,
        mu2,
        sigma2,
        m1_low,
        m_high,
        delta_m1,
        lambda0,
        lambda1,
        lambda2,
        beta,
        m2_low,
        delta_m2,
        epsilon,
        m_g,
        w_g,
        sig_g_low,
        sig_g_high,
    ) = lambdaBBHmass

    lpdfm1 = logpdfm1_DPLDP(
        bk, m1, alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high,
        delta_m1, lambda0, lambda1, lambda2, epsilon, smoothing=smoothing,
        simplex_repair=simplex_repair, norm_gauss=norm_gauss,
        taper_kind=taper_kind, taper_p=taper_p
    )
    
    lpdfm2 = logpdfm2_PLP_reg(
        bk, m2, beta, delta_m2, m2_low, m_g=m_g, w_g=w_g,
        sig_g_low=sig_g_low, sig_g_high=sig_g_high, has_m2_break=has_m2_break,
        smoothing=smoothing, taper_kind=taper_kind, taper_p=taper_p
    )
    
    lC = logC_DPLDP(
        bk, m1, beta, delta_m2, m2_low, m_g=m_g, w_g=w_g,
        sig_g_low=sig_g_low, sig_g_high=sig_g_high, has_m2_break=has_m2_break,
        smoothing=smoothing, taper_kind=taper_kind, taper_p=taper_p
    )

    if norm:
        ln = logNorm_DPLDP(
            bk, alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high,
            delta_m1, lambda0, lambda1, lambda2, epsilon, smoothing=smoothing,
            simplex_repair=simplex_repair, norm_gauss=norm_gauss,
            taper_kind=taper_kind, taper_p=taper_p
        )
 
        
    else:
        ln = 0.0

    lpdf = lpdfm1 + lpdfm2 - lC - ln

    if force_m2_less_than_m1:
        ok = bk.all(bk.asarray([(m2 <= m1), (m2 > 0.0), (m1 > 0.0)]), axis=0) if hasattr(bk, "all") else ((m2 <= m1) & (m2 > 0.0) & (m1 > 0.0))
        # (above keeps numpy happy; for ATBackend, & works fine too)
        return bk.where(ok, lpdf, -1.0e30)
    else:
        return lpdf


def logC_DPLDP(bk, m, beta, deltam, m2_low, m_g=45, w_g=80, sig_g_low=5, sig_g_high=5,
               has_m2_break=False, smoothing="LVK", taper_kind="sigmoid", taper_p=12.0):
    
    xx = bk.logspace(bk.log10(m2_low), bk.log10(max_m), 1000)
    l2 = logpdfm2_PLP_reg(
        bk, xx, beta, deltam, m2_low, m_g=m_g, w_g=w_g,
        sig_g_low=sig_g_low, sig_g_high=sig_g_high,
        has_m2_break=has_m2_break, smoothing=smoothing,
        taper_kind=taper_kind, taper_p=taper_p
    )
    l2 = bk.where(bk.isfinite(l2), l2, -1e30)
    a = bk.max(l2)
    p2 = bk.exp(l2 - a)
    cdf = atcumtrapz(bk, p2, xx)
    cdf = bk.maximum(bk.where(bk.isfinite(cdf), cdf, 0.0), 1e-300)
    return atinterp(bk, m, xx[1:], bk.log(cdf) + a)


def logNorm_DPLDP(bk, alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high,
                  delta_m1, lambda0, lambda1, lambda2, epsilon, smoothing="LVK",
                  simplex_repair=False, eps_int=1e-300, norm_gauss="uplow",
                  taper_kind="sigmoid", taper_p=12.0):
    
    ms = bk.logspace(bk.log10(m1_low), bk.log10(m_high), 1000)
    
    lpdf = logpdfm1_DPLDP(
        bk, ms, alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high,
        delta_m1, lambda0, lambda1, lambda2, epsilon, smoothing=smoothing,
        simplex_repair=simplex_repair, norm_gauss=norm_gauss,
        taper_kind=taper_kind, taper_p=taper_p
    )
    
    lpdf = bk.where(bk.isfinite(lpdf), lpdf, -1e30)
    a = bk.max(lpdf)
    ps = bk.exp(lpdf - a)
    integ = attrapzvec(bk, ps, ms)
    return bk.where((integ > 0.0) & bk.isfinite(integ), a + bk.log(integ), -jnp.inf)


def build_m1_grid_DPLDP( bk, 
    alpha1, alpha2, mb,
    mu1, sigma1, mu2, sigma2,
    m1_low, m_high,
    delta_m1,
    n_peak=2500,
    n_tail_low=400,
    n_tail_high=400,
    frac_gauss1=0.2,
    frac_gauss2=0.2,
    k_sigma_gauss=3.0,
    k_sigma_band=4.0,
    n_taper=10,
    n_taper_eff=200,
):
    """
    Symbolic non-uniform m1 grid for non-evolving DPLDP.

    Structure:
      - taper:      [m1_low, m1_low+delta_m1] (clustered near m1_low)
      - low tail:   [taper_hi, band_min)
      - Gaussian 1: [mu1 - kσ1, mu1 + kσ1] (with fallback if degenerate)
      - Gaussian 2: [mu2 - kσ2, mu2 + kσ2] (with fallback if degenerate)
      - mid band:   [band_min, band_max] envelope over peaks + mb
      - high tail:  [band_max, m_high)   (endpoint excluded)

    Guarantees:
      - all points inside (m1_low, m_high)
      - strictly increasing (via tiny ramp)
      - avoids repeated xmin/xmax collapse
    """

    # ---- detach hyperparameters for grid geometry (no grad through geometry) ----
    mb_sg       = bk.stop_grad(mb)
    mu1_sg      = bk.stop_grad(mu1)
    sigma1_sg   = bk.stop_grad(sigma1)
    mu2_sg      = bk.stop_grad(mu2)
    sigma2_sg   = bk.stop_grad(sigma2)
    m1_low_sg   = bk.stop_grad(m1_low)
    m_high_sg   = bk.stop_grad(m_high)
    delta_m1_sg = bk.stop_grad(delta_m1)

    # gentle boundary offset (avoid exact endpoints)
    eps = 1e-4
    xmin = m1_low_sg + eps
    xmax = m_high_sg - eps
    span = bk.maximum(xmax - xmin, 1e-6)

    # ------------------------------------------------------------
    # 0) Taper grid: clustered near xmin (important for logS_PLP)
    # ------------------------------------------------------------
    taper_hi = bk.clip(xmin + bk.maximum(delta_m1_sg, 1e-6), xmin, xmax)
    taper_w  = bk.maximum(taper_hi - xmin, 1e-6)

    if n_taper > 1:
        eps_t = 1e-4  # smallest fraction of taper width for the first interior point
        u = bk.linspace(0.0, 1.0, n_taper)  # [0,1]
        t = bk.exp(bk.log(eps_t) * (1.0 - u))   # eps_t -> 1
        t = (t - eps_t) / (1.0 - eps_t)         # -> [0,1]
        m1_taper = xmin + taper_w * t
    else:
        m1_taper = bk.zeros((0,))

    # ------------------------------------------------------------
    # 1) Gaussian windows (clip to support)
    # ------------------------------------------------------------
    k_g = k_sigma_gauss
    k_b = k_sigma_band

    g1_min_raw = mu1_sg - k_g * bk.abs(sigma1_sg)
    g1_max_raw = mu1_sg + k_g * bk.abs(sigma1_sg)
    g2_min_raw = mu2_sg - k_g * bk.abs(sigma2_sg)
    g2_max_raw = mu2_sg + k_g * bk.abs(sigma2_sg)

    g1_min = bk.clip(g1_min_raw, xmin, xmax)
    g1_max = bk.clip(g1_max_raw, xmin, xmax)
    g2_min = bk.clip(g2_min_raw, xmin, xmax)
    g2_max = bk.clip(g2_max_raw, xmin, xmax)

    tiny = 1e-6 * span
    g1_width = g1_max - g1_min
    g2_width = g2_max - g2_min

    has_g1 = bk.gt(g1_width, tiny)
    has_g2 = bk.gt(g2_width, tiny)

    # ------------------------------------------------------------
    # 2) Envelope "interesting band" over peaks + mb
    # ------------------------------------------------------------
    peak_min_raw = bk.minimum(g1_min_raw, g2_min_raw)
    peak_min_raw = bk.minimum(peak_min_raw, mb_sg)

    peak_max_raw = bk.maximum(g1_max_raw, g2_max_raw)
    peak_max_raw = bk.maximum(peak_max_raw, mb_sg)

    band_min = bk.clip(peak_min_raw, xmin, xmax)
    band_max = bk.clip(peak_max_raw, xmin, xmax)

    band_width = bk.maximum(band_max - band_min, tiny)

    # ------------------------------------------------------------
    # 3) Split n_peak between Gaussians + mid band
    # ------------------------------------------------------------
    n_g1  = int(n_peak * float(frac_gauss1))
    n_g2  = int(n_peak * float(frac_gauss2))
    if n_g1 < 0: n_g1 = 0
    if n_g2 < 0: n_g2 = 0
    if n_g1 + n_g2 > n_peak:
        scale = float(n_peak) / float(n_g1 + n_g2)
        n_g1 = int(round(n_g1 * scale))
        n_g2 = int(round(n_g2 * scale))
    n_mid = max(n_peak - n_g1 - n_g2, 0)

    # ------------------------------------------------------------
    # 4) Low tail: start AFTER taper, keep fixed length
    # ------------------------------------------------------------
    if n_tail_low > 0:
        denom_low = float(n_tail_low + 1)
        t_low = (bk.arange(n_tail_low) + 1.0) / denom_low  # in (0,1)

        low_start = taper_hi
        low_width = band_min - low_start

        fallback_w = bk.maximum(taper_w / bk.maximum(n_taper, 1), 1e-3)

        tail_good = low_start + low_width * t_low
        tail_fallback = low_start + fallback_w * t_low

        m1_low_tail = bk.switch(bk.gt(low_width, 0), tail_good, tail_fallback)
    else:
        m1_low_tail = bk.zeros((0,))

    # ------------------------------------------------------------
    # 5) Gaussian 1 segment (with fallback window if degenerate)
    # ------------------------------------------------------------
    if n_g1 > 0:
        denom_g1 = float(max(n_g1 - 1, 1))
        t_g1 = bk.arange(n_g1) / denom_g1

        m1_g1 = g1_min + g1_width * t_g1

        fallback_width = 1e-8 * span
        g1_center = 0.5 * (g1_min + g1_max)
        g1_center = bk.clip(g1_center, xmin + fallback_width, xmax - fallback_width)
        fallback_g1 = g1_center + fallback_width * (t_g1 - 0.5)

        m1_g1 = bk.switch(has_g1, m1_g1, fallback_g1)
    else:
        m1_g1 = bk.zeros((0,))

    # ------------------------------------------------------------
    # 6) Gaussian 2 segment (with fallback window if degenerate)
    # ------------------------------------------------------------
    if n_g2 > 0:
        denom_g2 = float(max(n_g2 - 1, 1))
        t_g2 = bk.arange(n_g2) / denom_g2

        m1_g2 = g2_min + g2_width * t_g2

        fallback_width = 1e-8 * span
        g2_center = 0.5 * (g2_min + g2_max)
        g2_center = bk.clip(g2_center, xmin + fallback_width, xmax - fallback_width)
        fallback_g2 = g2_center + fallback_width * (t_g2 - 0.5)

        m1_g2 = bk.switch(has_g2, m1_g2, fallback_g2)
    else:
        m1_g2 = bk.zeros((0,))

    # ------------------------------------------------------------
    # 7) Mid band segment
    # ------------------------------------------------------------
    if n_mid > 0:
        denom_mid = float(max(n_mid - 1, 1))
        t_mid = bk.arange(n_mid) / denom_mid
        m1_mid = band_min + band_width * t_mid
    else:
        m1_mid = bk.zeros((0,))

    # ------------------------------------------------------------
    # 8) High tail: endpoint excluded (avoid exact xmax)
    # ------------------------------------------------------------
    if n_tail_high > 0:
        denom_high = float(max(n_tail_high, 1))   # not (n_tail_high-1)
        t_high = bk.arange(n_tail_high) / denom_high  # in [0,1)
        m1_high_tail = band_max + (xmax - band_max) * t_high
    else:
        m1_high_tail = bk.zeros((0,))

    # ------------------------------------------------------------
    # Combine -> clip -> sort -> tiny ramp for strict monotonicity
    # ------------------------------------------------------------
    m1_grid_raw = bk.concatenate(
        [m1_taper, m1_low_tail, m1_g1, m1_g2, m1_mid, m1_high_tail],
        axis=0,
    )

    m1_grid_clipped = bk.clip(m1_grid_raw, xmin, xmax)
    m1_grid_sorted = bk.sort(m1_grid_clipped)

    # tiny ramp ensures strict increase (does not affect resolution)
    ramp_step = 1e-6
    ramp = ramp_step * bk.arange(m1_grid_sorted.shape[0], dtype=m1_grid_sorted.dtype)
    m1_grid_strict = bk.clip(m1_grid_sorted + ramp, xmin, xmax)
    return m1_grid_strict


def logpdf_DPLDP_from_interp(bk, theta, interp_vals, force_m2_less_than_m1=False):

    m1, m2 = theta
    interp_grids, interp_vals_mass = interp_vals

    m1_grid, m2_grid = interp_grids
    lp_m1_grid, lp_m2_grid, lC_of_m1, ln = interp_vals_mass

    ok = (
        (m1 >= m1_grid[0]) & (m1 <= m1_grid[-1]) &
        (m2 >= m2_grid[0]) & (m2 <= m2_grid[-1])
    )

    if force_m2_less_than_m1:
        ok = ok & (m2 <= m1)

    # avoid C(m1)=0 zone (logC=-inf -> +inf in joint)
    ok = ok & (m1 > m2_grid[0])


    Y_m1 = bk.stack([lp_m1_grid, lC_of_m1], axis=0)   # shape (2, Nm1)

    out = interp_1d_nonuniform_multiY(bk, m1, m1_grid, Y_m1, side="left")

    lpdfm1 = out[0]
    lC     = out[1]

    lpdfm2 = atinterp(bk, m2, m2_grid, lp_m2_grid)


    lpdf = lpdfm1 + lpdfm2 - lC - ln
    return bk.where(ok, lpdf, -1e30)


# ---------------------------------------------------------------------
# Double Power Law plus Double Peak Redshift Evolving (DPLDP-z)
# ---------------------------------------------------------------------


def theta_of_z(bk, z, theta_0, theta_inf, z_t, delta_z):
    """
    Generic redshift evolution for a hyperparameter, in the spirit of Eq. (2):
        θ(z) = θ0 + (θ_inf - θ0) * s(z; z_t, Δz),

    where s is a smooth sigmoid between 0 and 1.
    Works with scalar or array z (broadcasts).
    """
    delta_z = bk.maximum(delta_z, 1e-6)
    x = (z - z_t) / delta_z
    # tanh-based sigmoid: smoothly goes from 0 to 1 around z_t
    s = 0.5 * (1.0 + bk.tanh(x))
    return theta_0 + (theta_inf - theta_0) * s


def logpdfm1_DPLDP_z(bk, m1, z, alpha1_0, alpha2_0, mb_0, mu1_0, sigma1_0, mu2_0, sigma2_0,
                     m1_low, m_high, delta_m1, lambda0_0, lambda1_0, lambda2_0, 
                     epsilon,
                     alpha1_inf, z_alpha1, dz_alpha1, alpha2_inf, z_alpha2, dz_alpha2,
                     mb_inf, z_mb, dz_mb, mu1_inf, z_mu1, dz_mu1, sigma1_inf, z_sigma1, dz_sigma1,
                     mu2_inf, z_mu2, dz_mu2, sigma2_inf, z_sigma2, dz_sigma2,
                     lambda0_inf, lambda1_inf, lambda2_inf, 
                     z_lambda, dz_lambda,
                     smoothing="LVK", simplex_repair=False, norm_gauss="uplow",
                     taper_kind="sigmoid", taper_p=12.0):
    """
    Redshift-evolving version of logpdfm1_DPLDP with:
      - shape parameters evolved via theta_of_z(...)
      - mixture weights evolved as a convex combination

        lambda(z) = (1 - S_lambda(z)) * lambda_0 + S_lambda(z) * lambda_inf,

      where S_lambda(z) = 0.5 * (1 + tanh((z - z_lambda)/dz_lambda)).
    """

    # --- shape evolution as before ---
    alpha1  = theta_of_z(bk, z, alpha1_0,  alpha1_inf,  z_alpha1,  dz_alpha1)
    alpha2  = theta_of_z(bk,z, alpha2_0,  alpha2_inf,  z_alpha2,  dz_alpha2)
    mb      = theta_of_z(bk,z, mb_0,      mb_inf,      z_mb,      dz_mb)
    mu1     = theta_of_z(bk,z, mu1_0,     mu1_inf,     z_mu1,     dz_mu1)
    sigma1  = theta_of_z(bk,z, sigma1_0,  sigma1_inf,  z_sigma1,  dz_sigma1)
    mu2     = theta_of_z(bk,z, mu2_0,     mu2_inf,     z_mu2,     dz_mu2)
    sigma2  = theta_of_z(bk, z, sigma2_0,  sigma2_inf,  z_sigma2,  dz_sigma2)

    # --- shared S_lambda(z) for the mixture weights ---
    x_l = (z - z_lambda) / dz_lambda
    S_l = 0.5 * (1.0 + bk.tanh(x_l))  # same shape as in logpdf_DPLDP_z / logNorm_DPLDP_z

    # low-z and high-z λ2 from simplex

    # convex combination: λ(z) = (1-S) λ(0) + S λ(∞)
    lambda0 = (1.0 - S_l) * lambda0_0 + S_l * lambda0_inf
    lambda1 = (1.0 - S_l) * lambda1_0 + S_l * lambda1_inf
    lambda2 = (1.0 - S_l) * lambda2_0 + S_l * lambda2_inf 

    # --- call original m1 logpdf with z-dependent quantities ---
    return logpdfm1_DPLDP(
        bk, m1, alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2,
        m1_low, m_high, delta_m1, lambda0, lambda1, lambda2, epsilon,
        smoothing=smoothing, simplex_repair=simplex_repair, norm_gauss=norm_gauss,
        taper_kind=taper_kind, taper_p=taper_p
    )


def logpdf_DPLDP_z(bk, theta, z, lambdaBBHmass_lowz, evo_params,
                   force_m2_less_than_m1=False, has_m2_break=False,
                   norm_gauss="uplow", smoothing="LVK",
                   interp_vals=None, interp_grids=None, simplex_repair=False,
                   taper_kind="sigmoid", taper_p=12.0):
    """
    Redshift-evolving wrapper around your original logpdf_DPLDP.

    Parameters
    ----------
    theta : (m1, m2)
    z     : redshift (scalar or array broadcasting with m1, m2)
    lambdaBBHmass_lowz :
        Same vector you currently pass to logpdf_DPLDP, interpreted as
        the z≈0 values of the hyperparameters.
    evo_params :
        Flat tuple/array of the *evolution* hyperparameters, ordered as:
          (alpha1_inf, z_alpha1, dz_alpha1,
           alpha2_inf, z_alpha2, dz_alpha2,
           mb_inf,    z_mb,     dz_mb,
           mu1_inf,   z_mu1,    dz_mu1,
           sigma1_inf,z_sigma1, dz_sigma1,
           mu2_inf,   z_mu2,    dz_mu2,
           sigma2_inf,z_sigma2, dz_sigma2,
           lambda0_inf, lambda1_inf, z_lambda, dz_lambda)
    """

    m1, m2 = theta


    # unpack low-z hyperparameters (exactly your current order)
    (alpha1_0, alpha2_0, mb_0,
     mu1_0, sigma1_0, mu2_0, sigma2_0,
     m1_low, m_high, delta_m1,
     lambda0_0, lambda1_0, lambda2_0, 
     beta, m2_low, delta_m2,
     epsilon, m_g, w_g, sig_g_low, sig_g_high) = lambdaBBHmass_lowz

    # unpack evolution parameters
    (alpha1_inf,  z_alpha1,  dz_alpha1,
     alpha2_inf,  z_alpha2,  dz_alpha2,
     mb_inf,      z_mb,      dz_mb,
     mu1_inf,     z_mu1,     dz_mu1,
     sigma1_inf,  z_sigma1,  dz_sigma1,
     mu2_inf,     z_mu2,     dz_mu2,
     sigma2_inf,  z_sigma2,  dz_sigma2,
     lambda0_inf, lambda1_inf, lambda2_inf, z_lambda, dz_lambda) = evo_params

    # --- build z-dependent hyperparameters for the shape parameters ---
    alpha1  = theta_of_z(bk,z, alpha1_0,  alpha1_inf,  z_alpha1,  dz_alpha1)
    alpha2  = theta_of_z(bk,z, alpha2_0,  alpha2_inf,  z_alpha2,  dz_alpha2)
    mb      = theta_of_z(bk,z, mb_0,      mb_inf,      z_mb,      dz_mb)
    mu1     = theta_of_z(bk,z, mu1_0,     mu1_inf,     z_mu1,     dz_mu1)
    sigma1  = theta_of_z(bk,z, sigma1_0,  sigma1_inf,  z_sigma1,  dz_sigma1)
    mu2     = theta_of_z(bk,z, mu2_0,     mu2_inf,     z_mu2,     dz_mu2)
    sigma2  = theta_of_z(bk,z, sigma2_0,  sigma2_inf,  z_sigma2,  dz_sigma2)

    # --- shared S_lambda(z) for the mixture weights ---
    x_l    = (z - z_lambda) / dz_lambda
    S_l    = 0.5 * (1.0 + bk.tanh(x_l))


    lambda0 = (1.0 - S_l) * lambda0_0 + S_l * lambda0_inf
    lambda1 = (1.0 - S_l) * lambda1_0 + S_l * lambda1_inf
    lambda2 = (1.0 - S_l) * lambda2_0 + S_l * lambda2_inf


    # rebuild a z-dependent mass-parameter vector for the downstream calls
    lambdaBBHmass_z = (
        alpha1, alpha2, mb,
        mu1, sigma1, mu2, sigma2,
        m1_low, m_high, delta_m1,
        lambda0, lambda1, lambda2, 
        beta, m2_low, delta_m2,
        epsilon, m_g, w_g, sig_g_low, sig_g_high
    )

    # now just call original logpdf_DPLDP
    lpdf_ = logpdf_DPLDP(
        bk, theta, lambdaBBHmass_z,
        force_m2_less_than_m1=force_m2_less_than_m1,
        has_m2_break=has_m2_break, norm_gauss=norm_gauss,
        smoothing=smoothing, norm=False,
        simplex_repair=simplex_repair,
        taper_kind=taper_kind, taper_p=taper_p,
    )

    ln = logNorm_DPLDP_z(
        bk, z,
        alpha1_0, alpha2_0, mb_0, mu1_0, sigma1_0, mu2_0, sigma2_0,
        m1_low, m_high, delta_m1, lambda0_0, lambda1_0, lambda2_0, epsilon,
        alpha1_inf, z_alpha1, dz_alpha1, alpha2_inf, z_alpha2, dz_alpha2,
        mb_inf, z_mb, dz_mb, mu1_inf, z_mu1, dz_mu1,
        sigma1_inf, z_sigma1, dz_sigma1, mu2_inf, z_mu2, dz_mu2,
        sigma2_inf, z_sigma2, dz_sigma2,
        lambda0_inf, lambda1_inf, lambda2_inf, z_lambda, dz_lambda,
        smoothing=smoothing, simplex_repair=simplex_repair,
        norm_gauss=norm_gauss, taper_kind=taper_kind, taper_p=taper_p,
    )

    return lpdf_ - ln


def logNorm_DPLDP_z(bk, z, alpha1_0, alpha2_0, mb_0, mu1_0, sigma1_0, mu2_0, sigma2_0,
                    m1_low, m_high, delta_m1, lambda0_0, lambda1_0, lambda2_0, epsilon,
                    alpha1_inf, z_alpha1, dz_alpha1, alpha2_inf, z_alpha2, dz_alpha2,
                    mb_inf, z_mb, dz_mb, mu1_inf, z_mu1, dz_mu1, sigma1_inf, z_sigma1, dz_sigma1,
                    mu2_inf, z_mu2, dz_mu2, sigma2_inf, z_sigma2, dz_sigma2,
                    lambda0_inf, lambda1_inf, lambda2_inf, z_lambda, dz_lambda,
                    smoothing="LVK", simplex_repair=False, norm_gauss="uplow",
                    taper_kind="sigmoid", taper_p=12.0):
    """
    Same semantics as original logNorm_DPLDP_z, but:
    - we compute theta(z) once per z,
    - including mixture weights via a shared S_lambda(z),
    - then broadcast over m1_grid.

    Returns vector (Nevt,) of log-normalizations for each z.
    """
 
    
    _tgrid = _tgrid_np

    m1_grid = m1_low + (m_high - m1_low) * _tgrid  # (N1,)


    # --- make z a 1D tensor ---
    z = bk.atleast_1d(z)
    
    K = z.shape[0]          # number of events
    N1 = m1_grid.shape[0]   # grid size in m1

    # --- evolve all shape hyperparameters ONLY over z (shape: (K,)) ---
    alpha1  = theta_of_z(bk, z, alpha1_0,  alpha1_inf,  z_alpha1,  dz_alpha1)
    alpha2  = theta_of_z(bk, z, alpha2_0,  alpha2_inf,  z_alpha2,  dz_alpha2)
    mb      = theta_of_z(bk, z, mb_0,      mb_inf,      z_mb,      dz_mb)
    mu1     = theta_of_z(bk, z, mu1_0,     mu1_inf,     z_mu1,     dz_mu1)
    sigma1  = theta_of_z(bk, z, sigma1_0,  sigma1_inf,  z_sigma1,  dz_sigma1)
    mu2     = theta_of_z(bk, z, mu2_0,     mu2_inf,     z_mu2,     dz_mu2)
    sigma2  = theta_of_z(bk, z, sigma2_0,  sigma2_inf,  z_sigma2,  dz_sigma2)

    # --- shared S_lambda(z) for mixture weights ---
    x_l    = (z - z_lambda) / dz_lambda
    S_l    = 0.5 * (1.0 + bk.tanh(x_l))


    lambda0 = (1.0 - S_l) * lambda0_0 + S_l * lambda0_inf
    lambda1 = (1.0 - S_l) * lambda1_0 + S_l * lambda1_inf
    lambda2 = (1.0 - S_l) * lambda2_0 + S_l * lambda2_inf

    # --- broadcast to (K, N1) and flatten ---
    M_flat = bk.tile(m1_grid, K)  # shape: (K * N1,)

    alpha1_flat  = bk.repeat(alpha1,  N1)
    alpha2_flat  = bk.repeat(alpha2,  N1)
    mb_flat      = bk.repeat(mb,      N1)
    mu1_flat     = bk.repeat(mu1,     N1)
    sigma1_flat  = bk.repeat(sigma1,  N1)
    mu2_flat     = bk.repeat(mu2,     N1)
    sigma2_flat  = bk.repeat(sigma2,  N1)
    lambda0_flat = bk.repeat(lambda0, N1)
    lambda1_flat = bk.repeat(lambda1, N1)
    lambda2_flat = bk.repeat(lambda2, N1) 

    # --- evaluate m1 logpdf in one big vectorized call ---
    lp_flat = logpdfm1_DPLDP(
        bk, M_flat, alpha1_flat, alpha2_flat, mb_flat, mu1_flat, sigma1_flat, mu2_flat, sigma2_flat,
        m1_low, m_high, delta_m1, lambda0_flat, lambda1_flat, lambda2_flat, epsilon,
        smoothing=smoothing, simplex_repair=simplex_repair, norm_gauss=norm_gauss,
        taper_kind=taper_kind, taper_p=taper_p
    )

    logp = bk.where(bk.isfinite(lp_flat.reshape((K, N1))), lp_flat.reshape((K, N1)), -1e30)
    a = bk.max(logp, axis=1, keepdims=True)
    p = bk.exp(logp - a)
    norm = attrapzvec(bk, p, m1_grid[None, :], axis=1)
    return bk.where((norm > 0.0) & bk.isfinite(norm), a[:, 0] + bk.log(norm), -jnp.inf)


def build_m1_grid_DPLDP_z( bk,
    z_bank,
    # low-z hyperparameters
    mu1_0, sigma1_0, mu2_0, sigma2_0, mb_0,
    # high-z (asymptotic) hyperparameters
    mu1_inf, sigma1_inf, mu2_inf, sigma2_inf, mb_inf,
    # evolution hyperparameters
    z_mu1, dz_mu1,
    z_sigma1, dz_sigma1,
    z_mu2, dz_mu2,
    z_sigma2, dz_sigma2,
    z_mb, dz_mb,
    # support for m1
    m1_low, m_high,
    delta_m1,
    # grid resolution controls
    n_peak=2500,      # points in the "interesting" band (peaks + break)
    n_tail_low=400,   # points in low-mass tail
    n_tail_high=400,  # points in high-mass tail
    k_sigma=4.0,      # how many sigmas around each Gaussian to cover
    n_taper=10, 
    n_taper_eff = 200
):
    """
    Symbolic non-uniform m1 grid for the DPLDP-z mass model (with redshift evolution).

    Structure:
      - low tail:   [m1_low, band_min)
      - Gaussian 1 window over all z
      - Gaussian 2 window over all z
      - mid band:   [band_min, band_max] envelope over both peaks + break
      - high tail:  [band_max, m_high]

    n_peak is split into:
      ~20% for Gaussian 1, ~20% for Gaussian 2, rest in the mid band.

    All points are:
      - inside (m1_low, m_high),
      - sorted,
      - deduplicated.
    """

    # ---- detach all hyperparameters for grid geometry (no grad through grid) ----
    mu1_0_s      = bk.stop_grad(mu1_0)
    sigma1_0_s   = bk.stop_grad(sigma1_0)
    mu2_0_s      = bk.stop_grad(mu2_0)
    sigma2_0_s   = bk.stop_grad(sigma2_0)
    mb_0_s       = bk.stop_grad(mb_0)

    mu1_inf_s    = bk.stop_grad(mu1_inf)
    sigma1_inf_s = bk.stop_grad(sigma1_inf)
    mu2_inf_s    = bk.stop_grad(mu2_inf)
    sigma2_inf_s = bk.stop_grad(sigma2_inf)
    mb_inf_s     = bk.stop_grad(mb_inf)

    z_mu1_s      = bk.stop_grad(z_mu1)
    dz_mu1_s     = bk.stop_grad(dz_mu1)
    z_sigma1_s   = bk.stop_grad(z_sigma1)
    dz_sigma1_s  = bk.stop_grad(dz_sigma1)
    z_mu2_s      = bk.stop_grad(z_mu2)
    dz_mu2_s     = bk.stop_grad(dz_mu2)
    z_sigma2_s   = bk.stop_grad(z_sigma2)
    dz_sigma2_s  = bk.stop_grad(dz_sigma2)
    z_mb_s       = bk.stop_grad(z_mb)
    dz_mb_s      = bk.stop_grad(dz_mb)

    m1_low_s     = bk.stop_grad(m1_low)
    m_high_s     = bk.stop_grad(m_high)
    delta_m1_s = bk.stop_grad(delta_m1) 


    eps   = 1e-4 
 
    # global support (slightly shrunken to avoid exact boundaries)
    xmin = m1_low_s + eps
    xmax = m_high_s - eps
    span = bk.maximum(xmax - xmin, 1e-06) 

    # -----  explicit taper window grid -----
    # make sure the window has nonzero width and lies in support
    # ----- explicit taper window grid (clustered near xmin) -----
    taper_hi = bk.clip(xmin + bk.maximum(delta_m1_s, 1e-6), xmin, xmax)
    taper_w  = bk.maximum(taper_hi - xmin, 1e-6)
    
    if n_taper > 1:
        # cluster points near xmin using logarithmic spacing
        eps_t = 1e-4  # controls closeness of the first interior point (fraction of taper width)
        u = bk.linspace(0.0, 1.0, n_taper)  # [0,1]
        t = bk.exp(bk.log(eps_t) * (1.0 - u))   # goes from eps_t -> 1
        t = (t - eps_t) / (1.0 - eps_t)         # rescale to [0,1]
        m1_taper = xmin + taper_w * t
    else:
        m1_taper = bk.zeros((0,))

    # ---- evolve hyperparameters over z_bank (using detached params) ----
    mu1_z = theta_of_z(bk, z_bank, mu1_0_s,  mu1_inf_s,  z_mu1_s,    dz_mu1_s)
    sigma1_z = theta_of_z(bk, z_bank, sigma1_0_s, sigma1_inf_s, z_sigma1_s, dz_sigma1_s)

    mu2_z = theta_of_z(bk, z_bank, mu2_0_s,  mu2_inf_s,  z_mu2_s,    dz_mu2_s)
    sigma2_z = theta_of_z(bk, z_bank, sigma2_0_s, sigma2_inf_s, z_sigma2_s, dz_sigma2_s)

    mb_z = theta_of_z(bk, z_bank, mb_0_s, mb_inf_s, z_mb_s, dz_mb_s)

    k_sigma_t = k_sigma #bk.as_tensor_variable(k_sigma, dtype=dtype)

    # ---- Gaussian windows over all z ----
    # First for each z, then take global min/max over z.
    g1_min_z = mu1_z - k_sigma_t * bk.abs(sigma1_z)
    g1_max_z = mu1_z + k_sigma_t * bk.abs(sigma1_z)

    g2_min_z = mu2_z - k_sigma_t * bk.abs(sigma2_z)
    g2_max_z = mu2_z + k_sigma_t * bk.abs(sigma2_z)

    # global windows (clipped to support)
    g1_min = bk.clip(bk.min(g1_min_z), xmin, xmax)
    g1_max = bk.clip(bk.max(g1_max_z), xmin, xmax)
    g2_min = bk.clip(bk.min(g2_min_z), xmin, xmax)
    g2_max = bk.clip(bk.max(g2_max_z), xmin, xmax)

    tiny = 1e-6 * span
    g1_width = g1_max - g1_min
    g2_width = g2_max - g2_min

    has_g1 = bk.gt(g1_width, tiny)
    has_g2 = bk.gt(g2_width, tiny)

    # ---- global "interesting" band over all z (both peaks + break) ----
    peak_min_z = bk.minimum(g1_min_z, g2_min_z)
    peak_min_z = bk.minimum(peak_min_z, mb_z)

    peak_max_z = bk.maximum(g1_max_z, g2_max_z)
    peak_max_z = bk.maximum(peak_max_z, mb_z)

    band_min = bk.clip(bk.min(peak_min_z), xmin, xmax)
    band_max = bk.clip(bk.max(peak_max_z), xmin, xmax)  # min/max then clip

    band_width = bk.maximum(band_max - band_min, tiny)

    # ---- split n_peak between Gaussians and the mid band ----
    # use Python ints; n_peak is passed as plain int in your code
    frac_gauss1 = 0.2
    frac_gauss2 = 0.2

    n_g1  = int(n_peak * float(frac_gauss1))
    n_g2  = int(n_peak * float(frac_gauss2))
    if n_g1 < 0: n_g1 = 0
    if n_g2 < 0: n_g2 = 0
    if n_g1 + n_g2 > n_peak:
        scale = float(n_peak) / float(n_g1 + n_g2)
        n_g1 = int(round(n_g1 * scale))
        n_g2 = int(round(n_g2 * scale))
    n_mid = max(n_peak - n_g1 - n_g2, 0)

    # ---- segments ----

    # 1) low tail: ideally [taper_hi, band_min), but keep fixed length (n_tail_low)
    # so it always has shape (n_tail_low,) and compiles.
    
    if n_tail_low > 0:
        denom_low = float(n_tail_low + 1)  # +1 so we avoid including the endpoint
        t_low = (bk.arange(n_tail_low) + 1.0) / denom_low   # in (0,1)
    
        low_start = taper_hi
        low_width = band_min - low_start
    
        # fallback width if the segment would be empty or negative.
        # use something comparable to the taper resolution (not microscopic).
        # taper_w ~ delta_m1, so taper_w / n_taper is a sensible spacing scale.
        fallback_w = bk.maximum(taper_w / bk.maximum(n_taper, 1), 1e-3)  # Msun scale floor
    
        tail_good = low_start + low_width * t_low
        tail_fallback = low_start + fallback_w * t_low
    
        # if low_width > 0 -> use the good tail, else use fallback tail
        m1_low_tail = bk.switch(bk.gt(low_width, 0), tail_good, tail_fallback)
    
    else:
        m1_low_tail = bk.zeros((0,))


    # 2) Gaussian 1: [g1_min, g1_max]
    if n_g1 > 0:
        if n_g1 > 1:
            denom_g1 = float(n_g1 - 1)
        else:
            denom_g1 = 1.0
        t_g1 = bk.arange(n_g1) / denom_g1
        m1_g1 = g1_min + g1_width * t_g1
        # if the window is effectively degenerate, kill it

        fallback_width = 1e-08 * span  #  small compared to global support
        
        # center the fallback at the midpoint of the proposed window (≈ mu1 band)
        g1_center = 0.5 * (g1_min + g1_max)
        g1_center = bk.clip(g1_center, xmin + fallback_width, xmax - fallback_width)
        
        fallback_g1 = g1_center + fallback_width * (t_g1 - 0.5)  # monotone in t_g1

        m1_g1 = bk.switch(has_g1, m1_g1, fallback_g1)
    else:
        m1_g1 = bk.zeros((0,))

    # 3) Gaussian 2: [g2_min, g2_max]
    if n_g2 > 0:
        if n_g2 > 1:
            denom_g2 = float(n_g2 - 1)
        else:
            denom_g2 = 1.0
        t_g2 = bk.arange(n_g2) / denom_g2
        m1_g2 = g2_min + g2_width * t_g2

        g2_center = 0.5 * (g2_min + g2_max)
        g2_center = bk.clip(g2_center, xmin + fallback_width, xmax - fallback_width)
        fallback_g2 = g2_center + fallback_width * (t_g2 - 0.5)
        m1_g2 = bk.switch(has_g2, m1_g2, fallback_g2)
        
    else:
        m1_g2 = bk.zeros((0,))

    # 4) mid band: [band_min, band_max]
    if n_mid > 0:
        if n_mid > 1:
            denom_mid = float(n_mid - 1)
        else:
            denom_mid = 1.0
        t_mid = bk.arange(n_mid) / denom_mid
        m1_mid = band_min + band_width * t_mid
    else:
        m1_mid = bk.zeros((0,))

    # 5) high tail: [band_max, xmax]
    if n_tail_high > 0:
        denom_high = float(max(n_tail_high, 1))  # NOTE: n_tail_high (not n_tail_high-1)
        t_high = bk.arange(n_tail_high) / denom_high  # in [0, 1) never hits 1
        m1_high_tail = band_max + (xmax - band_max) * t_high
    else:
        m1_high_tail = bk.zeros((0,))

    # ---- combine, clip, sort, deduplicate ----
    m1_grid_raw = bk.concatenate(
        [m1_taper, m1_low_tail, m1_g1, m1_g2, m1_mid, m1_high_tail],
        axis=0,
    )

    # just in case anything nudged outside [xmin, xmax]
    m1_grid_clipped = bk.clip(m1_grid_raw, xmin, xmax)

    # enforce monotonicity and remove duplicates
    m1_grid_sorted = bk.sort(m1_grid_clipped)
    
    Ntot = m1_grid_sorted.shape[0]

    # pick a ramp small enough that the *total* ramp never exceeds eps/2
    ramp_step = bk.minimum(1e-6, 0.5 * eps / bk.maximum(Ntot - 1, 1))
    
    ramp = ramp_step * bk.arange(Ntot, dtype=m1_grid_sorted.dtype)
    
    m1_grid_strict = bk.clip(m1_grid_sorted + ramp, xmin, xmax)
    return m1_grid_strict


def logpdf_DPLDP_z_from_interp(bk, theta, z, interp_vals, force_m2_less_than_m1=False):

    interp_grids, interp_vals_mass = interp_vals
    m1, m2 = theta

    m1_grid, m2_grid, z_bank = interp_grids
    lp_m1_bank, lp_m2_grid, lC_of_m1, ln_bank = interp_vals_mass


    # ------------------------------------------------------------
    # 0) HARD SUPPORT MASK (this is the production fix)
    # ------------------------------------------------------------
    ok = (
        bk.isfinite(m1) & bk.isfinite(m2) & bk.isfinite(z)
        & (m1 >= m1_grid[0]) & (m1 <= m1_grid[-1])
        & (m2 >= m2_grid[0]) & (m2 <= m2_grid[-1])
        & (z  >= z_bank[0])  & (z  <= z_bank[-1])
    )


    # optional physical constraint
    if force_m2_less_than_m1:
        ok = ok & (m2 <= m1)

    # CRITICAL: avoid C(m1)=0 region which would produce +inf
    ok = ok & (m1 > m2_grid[0])

    # ------------------------------------------------------------
    # 1) SAFE indices + weights
    # ------------------------------------------------------------
    kR, rz = _interp_indices_nonuniform_safe(bk, z,  z_bank)
    kL = kR - 1

    j1, r1 = _interp_indices_nonuniform_safe(bk, m1, m1_grid)
    j2, r2 = _interp_indices_nonuniform_safe(bk, m2, m2_grid)

    # ------------------------------------------------------------
    # 2) Interpolate log p(m1 | z)
    # ------------------------------------------------------------
    yl_m1_L = lp_m1_bank[kL, j1 - 1]
    yh_m1_L = lp_m1_bank[kL, j1]
    lpdfm1_L = (1.0 - r1) * yl_m1_L + r1 * yh_m1_L

    yl_m1_R = lp_m1_bank[kR, j1 - 1]
    yh_m1_R = lp_m1_bank[kR, j1]
    lpdfm1_R = (1.0 - r1) * yl_m1_R + r1 * yh_m1_R

    lpdfm1 = (1.0 - rz) * lpdfm1_L + rz * lpdfm1_R

    # ------------------------------------------------------------
    # 3) Interpolate log C(m1)
    # ------------------------------------------------------------
    yl_C = lC_of_m1[j1 - 1]
    yh_C = lC_of_m1[j1]
    lC   = (1.0 - r1) * yl_C + r1 * yh_C

    # If logC is -inf or nan -> reject safely

    # ------------------------------------------------------------
    # 4) Interpolate log p(m2)
    # ------------------------------------------------------------
    yl_m2 = lp_m2_grid[j2 - 1]
    yh_m2 = lp_m2_grid[j2]
    lpdfm2 = (1.0 - r2) * yl_m2 + r2 * yh_m2

    # ------------------------------------------------------------
    # 5) Interpolate ln_norm(z)
    # ------------------------------------------------------------
    ln_L = ln_bank[kL]
    ln_R = ln_bank[kR]
    ln   = (1.0 - rz) * ln_L + rz * ln_R


    # ------------------------------------------------------------
    # 6) Assemble joint logpdf
    # ------------------------------------------------------------
    

    finite_lpdf = (
        bk.isfinite(lpdfm1)
        & bk.isfinite(lpdfm2)
        & bk.isfinite(lC)
        & bk.isfinite(ln)
    )
    
    ok_final = ok & finite_lpdf
    
    lpdfm1_s = bk.where(ok_final, lpdfm1, 0.0)
    lpdfm2_s = bk.where(ok_final, lpdfm2, 0.0)
    lC_s     = bk.where(ok_final, lC,     0.0)
    ln_s     = bk.where(ok_final, ln,     0.0)
    
    lpdf_safe = lpdfm1_s + lpdfm2_s - lC_s - ln_s
    
    return bk.where(ok_final, lpdf_safe, -1e30)


# ---------------------------------------------------------------------
# FullPop-4.0-inspired all-CBC mass model with conditional 2D structure
# ---------------------------------------------------------------------
#
# Baseline single-object spectrum: Eq. (B8) of the GWTC-4.0 population
# paper, with the targeted modifications agreed for this module:
#   * sigmoid blends replace the three hard power-law transitions;
#   * compact C1 tapers replace the outer Butterworth filters, giving
#     exact support on (m_NSmin, m_BHmax);
#   * the two notch filters use H(lower) * L(upper), following the
#     ancestor Power Law + Dip construction so that they suppress the
#     interval between their two edges.
#
# The final joint density is not the original symmetric FullPop pairing.
# It is
#
#   p(m1, m2) = p_S(m1) p(m2 | m1),
#
# with p(m2 | m1) proportional to q**beta(m1) times a positive sum of
# localized Gaussian modulations in (m1, m2).
# ---------------------------------------------------------------------


def log_compact_high_taper_FULLPOP4(bk, m, deltam, m_high):
    """Reflected ``logS_PLP`` taper: one below the taper, zero at/above m_high."""
    return logS_PLP(bk, m_high - m, deltam, 0.0)


def log_lowpass_FULLPOP4(bk, m, m0, eta, eps=1e-12):
    """log L(m|m0,eta), L = [1 + (m/m0)**eta]**(-1)."""
    m_safe = bk.maximum(m, eps)
    m0_safe = bk.maximum(m0, eps)
    log_ratio = bk.log(m_safe) - bk.log(m0_safe)
    return -logaddexp(bk, 0.0, eta * log_ratio)


def log_highpass_FULLPOP4(bk, m, m0, eta, eps=1e-12):
    """log H(m|m0,eta), H = [1 + (m0/m)**eta]**(-1)."""
    m_safe = bk.maximum(m, eps)
    m0_safe = bk.maximum(m0, eps)
    log_ratio = bk.log(m0_safe) - bk.log(m_safe)
    return -logaddexp(bk, 0.0, eta * log_ratio)


def log_notch_FULLPOP4(
    bk,
    m,
    m_low,
    m_high,
    eta_low,
    eta_high,
    amplitude,
):
    """
    Log notch factor

        N(m) = 1 - A H(m|m_low,eta_low) L(m|m_high,eta_high).

    The H(lower) L(upper) orientation is the ancestor-model convention
    that suppresses density between ordered lower and upper edges.
    """
    log_window = (
        bk.log(amplitude)
        + log_highpass_FULLPOP4(bk, m, m_low, eta_low)
        + log_lowpass_FULLPOP4(bk, m, m_high, eta_high)
    )
    return logdiffexp(bk, 0.0, log_window)


def log_backbone_FULLPOP4(
    bk,
    m,
    m_NSmax,
    m_BHmin,
    m_break,
    alpha1,
    alpha_dip,
    alpha2,
    alpha3,
    delta_NS_transition,
    delta_BH_transition,
    delta_break_transition,
    eps=1e-12,
):
    """
    Smooth continuous four-segment backbone.

    Segments:
        m**alpha1                         below m_NSmax
        m**alpha_dip                      from m_NSmax to m_BHmin
        m**alpha2                         from m_BHmin to m_break
        m**alpha3                         above m_break

    The three hard transitions are replaced by separate sigmoid blends.
    Multiplicative continuity factors are retained at every break.
    The intended ordering is

        m_NSmax < m_BHmin < m_break < m_BHmax.
    """
    m_safe = bk.maximum(m, eps)
    m_NSmax_safe = bk.maximum(m_NSmax, eps)
    m_BHmin_safe = bk.maximum(m_BHmin, eps)
    m_break_safe = bk.maximum(m_break, eps)

    log_m = bk.log(m_safe)
    log_m_NSmax = bk.log(m_NSmax_safe)
    log_m_BHmin = bk.log(m_BHmin_safe)
    log_m_break = bk.log(m_break_safe)

    log_B1 = alpha1 * log_m
    log_Bdip = (
        alpha_dip * log_m
        + (alpha1 - alpha_dip) * log_m_NSmax
    )
    log_B2 = (
        alpha2 * log_m
        + (alpha1 - alpha_dip) * log_m_NSmax
        + (alpha_dip - alpha2) * log_m_BHmin
    )
    log_B3 = (
        alpha3 * log_m
        + (alpha1 - alpha_dip) * log_m_NSmax
        + (alpha_dip - alpha2) * log_m_BHmin
        + (alpha2 - alpha3) * log_m_break
    )

    s_NS = safe_sigmoid(
        bk, m_safe, m_NSmax_safe, delta_NS_transition
    )
    log_B12 = logaddexp(
        bk,
        bk.log1p(-s_NS) + log_B1,
        bk.log(s_NS) + log_Bdip,
    )

    s_BH = safe_sigmoid(
        bk, m_safe, m_BHmin_safe, delta_BH_transition
    )
    log_B123 = logaddexp(
        bk,
        bk.log1p(-s_BH) + log_B12,
        bk.log(s_BH) + log_B2,
    )

    s_break = safe_sigmoid(
        bk, m_safe, m_break_safe, delta_break_transition
    )
    return logaddexp(
        bk,
        bk.log1p(-s_break) + log_B123,
        bk.log(s_break) + log_B3,
    )


def log_peak_factor_FULLPOP4(
    bk,
    m,
    c1,
    c2,
    mu1,
    sigma1,
    mu2,
    sigma2,
    m_NSmin,
    m_BHmax,
):
    """
    Log of the multiplicative peak factor

        1 + c1 N_[m_NSmin,m_BHmax](m|mu1,sigma1)
          + c2 N_[m_NSmin,m_BHmax](m|mu2,sigma2).
    """
    log_G1 = truncGausslowerupper_at_lpdf_safe(
        bk,
        m,
        mu1,
        sigma1,
        xmin=m_NSmin,
        xmax=m_BHmax,
        truncate=False,
    )
    log_G2 = truncGausslowerupper_at_lpdf_safe(
        bk,
        m,
        mu2,
        sigma2,
        xmin=m_NSmin,
        xmax=m_BHmax,
        truncate=False,
    )

    return logaddexp(
        bk,
        logaddexp(bk, 0.0, bk.log(c1) + log_G1),
        bk.log(c2) + log_G2,
    )


def logpdfm1_FULLPOP4_unnorm(bk, m, lambda_S):
    """
    Unnormalized FullPop-4.0 single-object spectrum p_S(m).

    ``lambda_S`` ordering:

        (c1, c2, mu1, sigma1, mu2, sigma2,
         m_NSmin, m_NSmax, m_BHmin, m_BHmax,
         alpha1, alpha_dip, alpha2,
         m_UMGmin, m_UMGmax,
         eta_NSmax, eta_BHmin, A1,
         eta_UMGmin, eta_UMGmax, A2,
         delta_m1_low, delta_m1_high,
         delta_NS_transition, delta_BH_transition,
         m_break, alpha3, delta_break_transition)
    """
    (
        c1,
        c2,
        mu1,
        sigma1,
        mu2,
        sigma2,
        m_NSmin,
        m_NSmax,
        m_BHmin,
        m_BHmax,
        alpha1,
        alpha_dip,
        alpha2,
        m_UMGmin,
        m_UMGmax,
        eta_NSmax,
        eta_BHmin,
        A1,
        eta_UMGmin,
        eta_UMGmax,
        A2,
        delta_m1_low,
        delta_m1_high,
        delta_NS_transition,
        delta_BH_transition,
        m_break,
        alpha3,
        delta_break_transition,
    ) = lambda_S

    log_edges = (
        logS_PLP(bk, m, delta_m1_low, m_NSmin)
        + log_compact_high_taper_FULLPOP4(
            bk, m, delta_m1_high, m_BHmax
        )
    )

    log_peaks = log_peak_factor_FULLPOP4(
        bk,
        m,
        c1,
        c2,
        mu1,
        sigma1,
        mu2,
        sigma2,
        m_NSmin,
        m_BHmax,
    )

    log_n1 = log_notch_FULLPOP4(
        bk,
        m,
        m_NSmax,
        m_BHmin,
        eta_NSmax,
        eta_BHmin,
        A1,
    )
    log_n2 = log_notch_FULLPOP4(
        bk,
        m,
        m_UMGmin,
        m_UMGmax,
        eta_UMGmin,
        eta_UMGmax,
        A2,
    )

    log_backbone = log_backbone_FULLPOP4(
        bk,
        m,
        m_NSmax,
        m_BHmin,
        m_break,
        alpha1,
        alpha_dip,
        alpha2,
        alpha3,
        delta_NS_transition,
        delta_BH_transition,
        delta_break_transition,
    )

    logp = log_edges + log_peaks + log_n1 + log_n2 + log_backbone
    in_support = (m > m_NSmin) & (m < m_BHmax)
    return bk.where(in_support, logp, -jnp.inf)


def logNorm_m1_FULLPOP4(bk, lambda_S, res=1000):
    """Differentiable trapezoidal normalization of p_S on its exact support."""
    m_NSmin = lambda_S[6]
    m_BHmax = lambda_S[9]

    t = bk.linspace(0.0, 1.0, res)
    ms = m_NSmin + (m_BHmax - m_NSmin) * t

    logp = logpdfm1_FULLPOP4_unnorm(bk, ms, lambda_S)
    logp_safe = bk.where(bk.isfinite(logp), logp, -1.0e30)

    shift = bk.max(logp_safe)
    vals = bk.exp(logp_safe - shift)
    integ = attrapzvec(bk, vals, ms)

    valid = (
        (m_BHmax > m_NSmin)
        & (integ > 0.0)
        & bk.isfinite(integ)
        & bk.isfinite(shift)
    )
    return bk.where(valid, shift + bk.log(integ), -jnp.inf)


def logpdfm1_FULLPOP4(bk, m, lambda_S, norm=True, res=1000):
    """Normalized FullPop-4.0 single-object spectrum used as p_S(m1)."""
    logp = logpdfm1_FULLPOP4_unnorm(bk, m, lambda_S)
    if not norm:
        return logp
    return logp - logNorm_m1_FULLPOP4(bk, lambda_S, res=res)


def beta_m1_FULLPOP4(bk, m1, Lambda_q):
    """
    Smooth NS-to-BH transition in the conditional mass-ratio exponent.

    ``Lambda_q`` ordering:

        (beta_NS, beta_BH, m_beta, delta_beta,
         delta_m2_low, delta_m2_high)
    """
    beta_NS, beta_BH, m_beta, delta_beta, _, _ = Lambda_q
    s = safe_sigmoid(bk, m1, m_beta, delta_beta)
    return beta_NS + (beta_BH - beta_NS) * s


def logR_FULLPOP4(bk, m1, m2, bumps=(), eps=1e-12):
    """
    Positive localized modulation in the (m1,m2) plane.

        R = 1 + sum_k A_k exp[-0.5 ((m1-mu1_k)/sig1_k)^2
                              -0.5 ((m2-mu2_k)/sig2_k)^2]

    Each bump is packed as ``(amp, mu1, sigma1, mu2, sigma2)``.
    """
    logR = 0.0
    if bumps is None:
        return logR

    for bump in bumps:
        amp, mu1, sigma1, mu2, sigma2 = bump
        sigma1_safe = bk.maximum(sigma1, eps)
        sigma2_safe = bk.maximum(sigma2, eps)
        log_bump = (
            bk.log(amp)
            - 0.5 * ((m1 - mu1) / sigma1_safe) ** 2
            - 0.5 * ((m2 - mu2) / sigma2_safe) ** 2
        )
        logR = logaddexp(bk, logR, log_bump)

    return logR


def logkernel_m2_given_m1_FULLPOP4(
    bk,
    m1,
    m2,
    Lambda_q,
    bumps,
    m_NSmin,
    m_BHmax,
    eps=1e-12,
):
    """Unnormalized log kernel for p(m2|m1)."""
    _, _, _, _, delta_m2_low, delta_m2_high = Lambda_q

    m1_safe = bk.maximum(m1, eps)
    m2_safe = bk.maximum(m2, eps)
    q = m2_safe / m1_safe
    beta = beta_m1_FULLPOP4(bk, m1, Lambda_q)

    log_edges = (
        logS_PLP(bk, m2, delta_m2_low, m_NSmin)
        + log_compact_high_taper_FULLPOP4(
            bk, m2, delta_m2_high, m_BHmax
        )
    )

    return (
        log_edges
        + beta * bk.log(q)
        + logR_FULLPOP4(bk, m1, m2, bumps=bumps)
    )


def logNorm_m2_given_m1_FULLPOP4(
    bk,
    m1,
    Lambda_q,
    bumps,
    m_NSmin,
    m_BHmax,
    res=500,
    eps=1e-12,
):
    """
    Conditional normalization

        Z_2(m1) = integral_[m_NSmin,m1] K(m1,m2) dm2.

    A fixed t-grid is mapped to the parameter-dependent interval, keeping
    the quadrature shape fixed and differentiable.
    """
    m1a = bk.asarray(m1)
    span = m1a - m_NSmin
    span_safe = bk.maximum(span, eps)

    t = bk.linspace(0.0, 1.0, res)
    m2_grid = m_NSmin + span_safe[..., None] * t
    m1_grid = m1a[..., None]

    logk = logkernel_m2_given_m1_FULLPOP4(
        bk,
        m1_grid,
        m2_grid,
        Lambda_q,
        bumps,
        m_NSmin,
        m_BHmax,
    )
    logk_safe = bk.where(bk.isfinite(logk), logk, -1.0e30)

    shift = bk.max(logk_safe, axis=-1, keepdims=True)
    vals = bk.exp(logk_safe - shift)
    integ = attrapzvec(bk, vals, m2_grid, axis=-1)
    logZ = shift[..., 0] + bk.log(integ)

    valid = (
        (span > 0.0)
        & (m1a < m_BHmax)
        & (integ > 0.0)
        & bk.isfinite(integ)
        & bk.isfinite(logZ)
    )
    return bk.where(valid, logZ, -jnp.inf)


def logpdfm2_given_m1_FULLPOP4(
    bk,
    m1,
    m2,
    Lambda_q,
    bumps,
    m_NSmin,
    m_BHmax,
    res=500,
):
    """Normalized conditional secondary-mass density p(m2|m1)."""
    logk = logkernel_m2_given_m1_FULLPOP4(
        bk,
        m1,
        m2,
        Lambda_q,
        bumps,
        m_NSmin,
        m_BHmax,
    )
    logZ = logNorm_m2_given_m1_FULLPOP4(
        bk,
        m1,
        Lambda_q,
        bumps,
        m_NSmin,
        m_BHmax,
        res=res,
    )

    in_support = (
        (m1 > m_NSmin)
        & (m1 < m_BHmax)
        & (m2 > m_NSmin)
        & (m2 < m_BHmax)
        & (m2 <= m1)
    )
    return bk.where(in_support, logk - logZ, -jnp.inf)


def logpdf_FULLPOP4_conditional(
    bk,
    theta,
    lambdaBBHmass,
    res_m1=1000,
    res_m2=500,
):
    """
    Final normalized joint model

        p(m1,m2) = p_S(m1) p(m2|m1).

    Parameter packing:

        lambdaBBHmass = (lambda_S, Lambda_q, bumps)

    with ``lambda_S`` and ``Lambda_q`` ordered as documented in
    ``logpdfm1_FULLPOP4_unnorm`` and ``beta_m1_FULLPOP4``. ``bumps`` is a
    Python sequence of ``(amp, mu1, sigma1, mu2, sigma2)`` tuples.
    """
    m1, m2 = theta
    lambda_S, Lambda_q, bumps = lambdaBBHmass

    m_NSmin = lambda_S[6]
    m_BHmax = lambda_S[9]

    logp1 = logpdfm1_FULLPOP4(
        bk, m1, lambda_S, norm=True, res=res_m1
    )
    logp2_given_1 = logpdfm2_given_m1_FULLPOP4(
        bk,
        m1,
        m2,
        Lambda_q,
        bumps,
        m_NSmin,
        m_BHmax,
        res=res_m2,
    )

    return logp1 + logp2_given_1
