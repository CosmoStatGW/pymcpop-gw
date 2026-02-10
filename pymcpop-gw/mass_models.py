from __future__ import annotations

import numpy as np
from pytensor_utils import attrapzvec, atcumtrapz, logsumexp2, logaddexp, logdiffexp, sigmoid, log_sigmoid, safe_sigmoid, atinterp_uniform, logsumexp
from constants import _PI as PI
from constants import max_m, _tgrid_np
from pytensor_utils import atinterp
from jax.scipy.special import logsumexp as _logsumexp

try:
    import jax.numpy as jnp
except Exception:
    jnp = None
    
try:
    import jax
except Exception as e:
    print(e)
    raise ValueError()




# Precompute a few sizes you will use
_LEGGAUSS_NP = {
    8:  np.polynomial.legendre.leggauss(8),
    16: np.polynomial.legendre.leggauss(16),
    32: np.polynomial.legendre.leggauss(32),
    64: np.polynomial.legendre.leggauss(64),
}

def leggauss_const(n: int):
    t_np, w_np = _LEGGAUSS_NP[int(n)]  # pure dict lookup, no mutation
    # Convert to JAX arrays (constants captured by XLA)
    t = jnp.asarray(t_np, dtype=jnp.float64)
    w = jnp.asarray(w_np, dtype=jnp.float64)
    return t, w


def grid_diagnostics(name, x):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    dx = np.diff(x)
    print(f"{name}: N={x.size}")
    print(f"  min dx = {dx.min():.3e}")
    print(f"  1% dx  = {np.quantile(dx, 0.01):.3e}")
    print(f"  median = {np.median(dx):.3e}")
    print(f"  max dx = {dx.max():.3e}")
    print(f"  any non-increasing? {np.any(dx <= 0)}")
    # near-duplicates relative to eps
    print(f"  dx < 1e-12: {np.sum(dx < 1e-12)}")
    print(f"  dx < 1e-10: {np.sum(dx < 1e-10)}")
    
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

    # these must exist somewhere (paste next if not)
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
        return bk.where(in_bounds, logp, -np.inf)
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
    Phi = bk.clip(Phi, 1e-300, np.inf)
    return bk.log(Phi)


def truncGausslowerupper_at_lpdf(
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
        return bk.where((x >= xmin) & (x <= xmax), logp, -np.inf)
    return logp


def _log_ndtr(bk, z):
    # log Phi(z) via erf; clamp only to avoid log(0)
    Phi = 0.5 * (1.0 + bk.erf(z / bk.sqrt(2.0)))
    Phi = bk.maximum(Phi, 1e-300)
    return bk.log(Phi)

    
# ---------------------------------------------------------------------
# Smoothings
# ---------------------------------------------------------------------

def logS_PLP(bk, m, deltam, ml, eps=1e-12):
    t = (m - ml) / bk.maximum(deltam, eps)
    t = bk.clip(t, eps, 1.0 - eps)
    S = t * t * (3.0 - 2.0 * t)
    return bk.log(bk.clip(S, eps, 1.0))


def logS_PLP_LVK(bk, m, deltam, ml):
    maskL = m <= ml
    maskU = m >= (ml + deltam)
    maskM = ~(maskL | maskU)

    s = bk.where(maskL, -np.inf, 0.0)

    mid = bk.log(1.0 / (1.0 + bk.exp(deltam / (m - ml) + deltam / (m - ml - deltam))))
    s1 = bk.where(maskM, mid, s)

    return s1


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
    smoothing="LVK",
):
    if smoothing == "LVK":
        lS = logS_PLP_LVK(bk, m, deltam, ml)
    else:
        lS = logS_PLP(bk, m, deltam, ml)

    lpdfval = beta * bk.log(m) + lS

    if not has_m2_break:
        return lpdfval

    # Two edges as in your legacy code:
    left_edge  = 1.0 - safe_sigmoid(bk, m, m_g, sig_g_low)
    right_edge = safe_sigmoid(bk, m, m_g + w_g, sig_g_high)

    # Your original: mask = log(left_edge + right_edge)
    mask = bk.log(left_edge + right_edge)

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
        )
        + log_sigmoid(bk, m, ml, sig_l)
    )



# ---------------------------------------------------------------------
# Double Power Law plus Double Peak (DPLDP)
# ---------------------------------------------------------------------


    
def logpdfm1_DPLDP(
    bk,
    m1,
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
    epsilon,
    smoothing="LVK",
    simplex_repair=False,
    eps_w=1e-15,
    sl=0.1,
    sh=1.,
    norm_gauss="uplow",
):
    if not simplex_repair:
        log_lambda0 = bk.log(lambda0)
        log_lambda1 = bk.log(lambda1)
        log_lambda2 = bk.log(lambda2)
    else:
        # Simplex repair (keep your logic; just bk ops)
        lam0 = bk.clip(lambda0, eps_w, 1.0 - eps_w)
        lam1 = bk.clip(lambda1, eps_w, 1.0 - eps_w)
        lam2_raw = 1.0 - lam0 - lam1

        # bk.softplus exists? If not, paste a bk-softplus utility later.
        # For now assume you have it or will add it to backends.
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
        log_pnorm1 = truncGausslowerupper_at_lpdf(bk, m1, mu1, sigma1, xmin=m1_low, xmax=m_high)
        log_pnorm2 = truncGausslowerupper_at_lpdf(bk, m1, mu2, sigma2, xmin=m1_low, xmax=m_high)

    elif norm_gauss == "low-once":
        log_pnorm1 = truncGausslowerupper_at_lpdf(bk, m1, mu1, sigma1, xmin=m1_low)
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

    # gate: log(sigmoid_low) + log(1 - sigmoid_high)
    log_gate = log_sigmoid(bk, m1, m1_low, sl) + bk.log1p(-safe_sigmoid(bk, m1, m_high, sh))

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

    log_N = log_broken_pl_norm_DPLDP(bk, alpha1, alpha2, mb_pos, m1_low, m_high, ) #eps=eps, t_floor=t_floor)

    log_m1_over_mb = bk.log(m1_pos / mb_pos)
    log_val1 = -alpha1 * log_m1_over_mb
    log_val2 = -alpha2 * log_m1_over_mb

    w = safe_sigmoid(bk, -m1_pos, -mb_pos, epsilon)
    #w = bk.clip(w, eps_w, 1.0 - eps_w)

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
        -np.inf,
    )
    logI2 = bk.where(
        u_high > 1.0,
        log_norm_truncated_pl(bk, alpha2, 1.0, u_high),
        -np.inf,
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
        -np.inf,
    )
    logI2 = bk.where(
        u_high > one,
        log_norm_truncated_pl_num_alpha1_safe(bk, alpha2, one, u_high, eps=eps, t_floor=t_floor),
        -np.inf,
    )

    return bk.log(mb_pos) + logaddexp(bk, logI1, logI2)


def log_norm_truncated_pl_num_alpha1_safe(bk, alpha, mmin, mmax, eps=1e-12, t_floor=1e-12):
    mmin_c = bk.clip(mmin, eps, np.inf)
    mmax_c = bk.clip(mmax, eps, np.inf)
    mmax_c = bk.maximum(mmax_c, mmin_c * (1.0 + 1e-12))

    t = 1.0 - alpha
    # avoid exactly zero
    t_safe = bk.where(bk.abs(t) < t_floor, bk.where(t >= 0.0, t_floor, -t_floor), t)

    b = bk.log(mmin_c)
    delta = bk.log(mmax_c) - b

    # need expm1; if backend doesn’t expose expm1, we can do exp(x)-1 safely later.
    # For now assume your backends have expm1 or you’ll add it.
    return t_safe * b + bk.log(bk.abs(bk.expm1(t_safe * delta))) - bk.log(bk.abs(t_safe))




def logpdf_DPLDP(
    bk,
    theta,
    lambdaBBHmass,
    force_m2_less_than_m1=False,
    has_m2_break=False,
    smoothing="LVK",
    resC=100,
    resN=500,
    interp_vals=None,
    interp_grids=None,
    norm=True,
    simplex_repair=False,
    norm_gauss="uplow",
):
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
        bk,
        m1,
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
        epsilon,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
        norm_gauss=norm_gauss,
    )

    lpdfm2 = logpdfm2_PLP_reg(
        bk,
        m2,
        beta,
        delta_m2,
        m2_low,
        m_g=m_g,
        w_g=w_g,
        sig_g_low=sig_g_low,
        sig_g_high=sig_g_high,
        has_m2_break=has_m2_break,
        smoothing=smoothing,
    )

    lC = logC_DPLDP(
        bk,
        m1,
        beta,
        delta_m2,
        m2_low,
        m_g=m_g,
        w_g=w_g,
        sig_g_low=sig_g_low,
        sig_g_high=sig_g_high,
        has_m2_break=has_m2_break,
        res=resC,
        smoothing=smoothing,
    )

    if norm:
        ln = logNorm_DPLDP(
            bk,
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
            epsilon,
            smoothing=smoothing,
            res=resN,
            simplex_repair=simplex_repair,
            norm_gauss=norm_gauss,
        )
 

        # ln  = logNorm_m1_DPLDP_GL(
        #         bk,
        #         alpha1, alpha2, mb,
        #         mu1, sigma1, mu2, sigma2,
        #         m1_low, m_high, delta_m1,
        #         lambda0, lambda1, lambda2,
        #         epsilon,
        #         smoothing=smoothing,
        #         simplex_repair=simplex_repair,
        #         #sl=sl, sh=sh,
        #         norm_gauss=norm_gauss,
        #         n_gl=16,   
        #     )
        
    else:
        ln = 0.0

    lpdf = lpdfm1 + lpdfm2 - lC - ln

    if force_m2_less_than_m1:
        ok = bk.all(bk.asarray([(m2 <= m1), (m2 > 0.0), (m1 > 0.0)]), axis=0) if hasattr(bk, "all") else ((m2 <= m1) & (m2 > 0.0) & (m1 > 0.0))
        # (above keeps numpy happy; for ATBackend, & works fine too)
        return bk.where(ok, lpdf, -np.inf)
    else:
        return lpdf


def logC_DPLDP(
    bk,
    m,
    beta,
    deltam,
    m2_low,
    m_g=45,
    w_g=80,
    sig_g_low=5,
    sig_g_high=5,
    has_m2_break=False,
    res=500,
    smoothing="LVK",
):
    if res != 500:
        _tgrid = bk.linspace(0.0, 1.0, res)
    else:
        _tgrid = _tgrid_np 

    xx = m2_low + (max_m - m2_low) * _tgrid  

    l2 = logpdfm2_PLP_reg(
        bk,
        xx,
        beta,
        deltam,
        m2_low,
        m_g=m_g,
        w_g=w_g,
        sig_g_low=sig_g_low,
        sig_g_high=sig_g_high,
        has_m2_break=has_m2_break,
        smoothing=smoothing,
    )

    a = bk.max(l2)
    p2 = bk.exp(l2 - a)

    # legacy: cdf = atcumtrapz(p2, xx)
    cdf = atcumtrapz(bk, p2, xx) #bk.stop_grad(xx))  
    #cdf = bk.clip(cdf, 1e-300, np.inf)

    x0 = xx[1]
    x1 = xx[-1]
    nU = xx.shape[0] - 1

    # log(cdf_scaled) + a gives log(cdf_original)
    #itr = atinterp_uniform(bk, m, x0, x1, nU, bk.log(cdf) + a)
    itr = atinterp( bk, m, xx[1:], bk.log(cdf)+ a )
    return itr


# ---- Fast replacements for logNorm_DPLDP and logC_DPLDP using GL ----

def logNorm_m1_DPLDP_GL(
    bk,
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
    epsilon,
    *,
    smoothing="LVK",
    simplex_repair=False,
    eps_w=1e-15,
    sl=0.1,
    sh=1,
    norm_gauss="uplow",
    n_gl=32,
):
    """
    logZ = log ∫_{m1_low}^{m_high} exp(logpdfm1_DPLDP(..., m1)) dm1
    computed by Gauss–Legendre on a small number of intervals.

    This keeps EXACT model (soft break epsilon + gates + smoothing choice).
    """
    #m1_low = jnp.asarray(m1_low, dtype=jnp.float64)
    #m_high = jnp.asarray(m_high, dtype=jnp.float64)
    #mb     = jnp.asarray(mb, dtype=jnp.float64)

    # fixed split point for taper end
    m_taper_end = jnp.minimum(m1_low + delta_m1, m_high)

    # clamp break into [m_taper_end, m_high] so segments are ordered
    mb_c = jnp.clip(mb, m_taper_end, m_high)

    def _logf(x):
        return logpdfm1_DPLDP(
            bk,
            x,
            alpha1, alpha2, mb,
            mu1, sigma1, mu2, sigma2,
            m1_low, m_high, delta_m1,
            lambda0, lambda1, lambda2,
            epsilon,
            smoothing=smoothing,
            simplex_repair=simplex_repair,
            eps_w=eps_w,
            sl=sl,
            sh=sh,
            norm_gauss=norm_gauss,
        )

    # Segment 1: [m1_low, m_taper_end]
    logI1 = _log_integral_gl_1d(_logf, m1_low, m_taper_end, n=n_gl)

    # Segment 2: [m_taper_end, mb_c]
    logI2 = _log_integral_gl_1d(_logf, m_taper_end, mb_c, n=n_gl)

    # Segment 3: [mb_c, m_high]
    logI3 = _log_integral_gl_1d(_logf, mb_c, m_high, n=n_gl)

    # Total = log(exp(logI1)+exp(logI2)+exp(logI3))
    logZ = _logaddexp(_logaddexp(logI1, logI2), logI3)
    return logZ



def _log_integral_gl_1d(logf, a, b, *, n=32):
    """
    Compute log ∫_a^b exp(logf(x)) dx using n-point Gauss–Legendre.
    Works with JAX tracers for a,b.

    logf: callable x -> log density (broadcastable over x)
    a,b: scalars (can be JAX tracers)
    """
    #a = jnp.asarray(a, dtype=jnp.float64)
    #b = jnp.asarray(b, dtype=jnp.float64)

    # ensure ordering
    a0 = jnp.minimum(a, b)
    b0 = jnp.maximum(a, b)

    # if interval is empty, return -inf
    length = b0 - a0
    empty = length <= 0.0

    t, w = leggauss_const(int(n))  # constants

    # map nodes to [a0, b0]
    mid  = 0.5 * (a0 + b0)
    half = 0.5 * (b0 - a0)
    x = mid + half * t

    # log-integrand at nodes
    lv = logf(x)

    # include weights + Jacobian
    logw = jnp.log(w) + jnp.log(half)
    out = _logsumexp(lv + logw)

    return jnp.where(empty, -jnp.inf, out)


def _logaddexp(a, b):
    return jnp.logaddexp(a, b)


def logNorm_DPLDP(
    bk,
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
    epsilon,
    res=500,
    smoothing="LVK",
    simplex_repair=False,
    eps_int=1e-300,
    norm_gauss="uplow",
):
    """
    Overflow-safe log normalization:
      log ∫ exp(logpdfm1_DPLDP(ms)) dms
    using max-subtraction.
    """
    if res != 500:
        _tgrid = bk.linspace(0.0, 1.0, res)
    else:
        _tgrid = _tgrid_np #_get_t_grid()

    ms = m1_low + (m_high - m1_low) * _tgrid

    lpdf = logpdfm1_DPLDP(
        bk,
        ms,
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
        epsilon,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
        norm_gauss=norm_gauss,
    )

    a = bk.max(lpdf)
    ps = bk.exp(lpdf - a)
    integ = attrapzvec(bk, ps, ms)
    #integ = bk.clip(integ, eps_int, np.inf)

    return a + bk.log(integ)



