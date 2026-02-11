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



def gaussian_logpdf_pair(bk, m1s, m2s, mu, sd, z=None, mins=None, maxs=None):
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
    const = -0.5 * at.log(2.0 * PI)

    logp1 = const - 0.5 * at.log(var1) - 0.5 * (diff1 * diff1 / var1)
    logp2 = const - 0.5 * at.log(var2) - 0.5 * (diff2 * diff2 / var2)


    if z is not None:
        z = z[None, :]
        muz = mu[2][:, None]       # (K, 1)
        sdz = sd[2][:, None]

        varz = sdz * sdz
        diffz = z - muz
        logpz = const - 0.5 * at.log(varz) - 0.5 * (diffz * diffz / varz)
    else:
        logpz = bk.zeros_like(logp1)

    return logp1, logp2, logpz

    
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
    # log_ratio = bk.log(mmax_c / mmin_c)
    # log_eq1   = bk.log( log_ratio )

    return log_not1 #bk.switch(close, log_eq1, log_not1)

    
def logpdfm1_PLP_reg(bk, m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, sl=0.05, sh=0.05, smoothing='LVK'):

    return logpdfm1_PLP_noreg(bk, m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing=smoothing)  + log_sigmoid(bk, m, ml, sl) + bk.log1p(-safe_sigmoid(bk, m, mh, sh)) 
    

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



def logC_PLP_reg( bk, m, beta, deltam, ml, smoothing='LVK'):


    _tgrid = _tgrid_np

    xx = ml + (max_m - ml) * _tgrid 

    l2 = logpdfm2_PLP_reg(bk, xx, beta, deltam, ml,
                         smoothing=smoothing)

    
    a = bk.max(l2)
    p2 = bk.exp(l2 - a)

    cdf_scaled = atcumtrapz(bk, p2, xx)
    # cdf_scaled = bk.clip(cdf_scaled, 1e-300, np.inf)

    # x0 = xx[1]
    # x1 = xx[-1]
    # nU = xx.shape[0] - 1

    # # log(cdf_scaled) + a gives log(cdf_original)
    # itr = atinterp_uniform(m, x0, x1, nU, bk.log(cdf_scaled) + a)
    itr = atinterp( bk, m, xx[1:], bk.log(cdf_scaled) + a )
    
    return itr



def logNorm_PLP_reg( bk, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing='LVK', ):

    
    
    '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

    '''


    _tgrid = _tgrid_np 

    ms = ml + (mh - ml) * _tgrid 

    lpdf = logpdfm1_PLP_noreg( bk, ms , lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing=smoothing  )

    a = bk.max(lpdf)
    ps = bk.exp(lpdf - a)                 # <= 1, avoids overflow
    integ = attrapzvec(bk, ps, ms)
    #integ = bk.clip(integ, eps_int, np.inf)

    return a + bk.log(integ)





    
def logpdf_PLP_reg(bk, theta, lambdaBBHmass,  smoothing='LVK'):
    
        m1, m2 = theta
        lambdaPeak, alpha, beta, deltam, ml, mh, muMass, sigmaMass = lambdaBBHmass
                

        lpdfm1 = logpdfm1_PLP_reg( bk, m1, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing=smoothing )
        
        lpdfm2 = logpdfm2_PLP_reg(bk, m2, beta, deltam, ml, smoothing=smoothing)
        
        ln = logNorm_PLP_reg( bk, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, smoothing=smoothing )

        lC = logC_PLP_reg(bk, m1, beta, deltam,  ml, smoothing=smoothing
                         ) 
        
        return lpdfm1 + lpdfm2 - ln - lC
        



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
            simplex_repair=simplex_repair,
            norm_gauss=norm_gauss,
        )
 
        
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
    smoothing="LVK",
):
    
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



