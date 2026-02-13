from __future__ import annotations

import numpy as np
from pytensor_utils import attrapzvec, atcumtrapz, logtrapzexp_streaming, logsumexp2, logaddexp, logdiffexp, sigmoid, log_sigmoid, safe_sigmoid, atinterp_uniform, logsumexp , _interp_indices_nonuniform_safe, interp_1d_nonuniform_multiY
#from constants import _PI as PI
from constants import max_m, _tgrid_np
from pytensor_utils import atinterp

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
        return bk.where((x >= xmin) & (x <= xmax), logp, -jnp.inf)
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
    const = -0.5 * bk.log(2.0 * PI)

    logp1 = const - 0.5 * bk.log(var1) - 0.5 * (diff1 * diff1 / var1)
    logp2 = const - 0.5 * bk.log(var2) - 0.5 * (diff2 * diff2 / var2)


    if z is not None:
        z = z[None, :]
        muz = mu[2][:, None]       # (K, 1)
        sdz = sd[2][:, None]

        varz = sdz * sdz
        diffz = z - muz
        logpz = const - 0.5 * bk.log(varz) - 0.5 * (diffz * diffz / varz)
    else:
        logpz = bk.zeros_like(logp1)

    return logp1, logp2, logpz



M = 40
gh_nodes_np, gh_weights_np = np.polynomial.hermite.hermgauss(M)
GH_NODES = gh_nodes_np   # (M,)
GH_WEIGHTS = gh_weights_np  # (M,)
SQRTPI = np.sqrt(np.pi)

def mixture_logZ_physical_vectorized( bk, 
    mux, sdx, muy, sdy, logw, mmin, mmax,
    #GH_NODES, GH_WEIGHTS,
    eps=1e-10
):
    """
    Returns scalar logZ for hard truncation in (m1,m2) but GMM in (x=logMc,y=logitq).
    Uses 1D Gauss-Hermite in y; x-integral is Normal CDF difference.
    """
 
    # y grid: (K,M)
    y = muy[:, None] + bk.sqrt(2.0) * sdy[:, None] * GH_NODES[None, :]
    q = bk.sigmoid(y)

    fac = (1.0 + q) ** 0.2
    Mc_low  = mmin / (fac * q ** 0.4)
    Mc_high = mmax * (q ** 0.6) / fac

    x_low  = bk.log(Mc_low)
    x_high = bk.log(Mc_high)


    z_high = (x_high - mux[:, None]) / sdx[:, None]
    z_low  = (x_low  - mux[:, None]) / sdx[:, None]

    # log(Phi(z_high)-Phi(z_low)) stably
    lhi = _log_ndtr(bk, z_high)
    llo = _log_ndtr(bk, z_low)
    logPx = logdiffexp(bk, 
                       lhi, llo)

    Px = bk.exp(logPx)  # (K,M), safe enough after log-space diff

    # GH integrate over y: Zk = sum w_i Px / sqrt(pi)
    Zk = bk.sum(GH_WEIGHTS[None, :] * Px, axis=1) / bk.sqrt(np.pi)  # (K,)
    

    logZ = bk.logsumexp(logw + bk.log(Zk))
    return logZ, Zk


    
    
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

    s = bk.where(maskL, -jnp.inf, 0.0)

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
    # cdf_scaled = bk.clip(cdf_scaled, 1e-300, jnp.inf)

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
    #integ = bk.clip(integ, eps_int, jnp.inf)

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
        return bk.where(ok, lpdf, -jnp.inf)
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
    #cdf = bk.clip(cdf, 1e-300, jnp.inf)

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
    #integ = bk.clip(integ, eps_int, jnp.inf)

    return a + bk.log(integ)



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
    #m1_grid_strict = m1_grid_sorted + ramp
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

    #out = interp_1d_nonuniform_multiY_numpyop(m1, m1_grid, Y_m1)
    out = interp_1d_nonuniform_multiY(bk, m1, m1_grid, Y_m1, side="left")

    lpdfm1 = out[0]
    lC     = out[1]

    #lpdfm2 = interp_1d_nonuniform_numpyop(m2, m2_grid, lp_m2_grid)
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
    x = (z - z_t) / delta_z
    # tanh-based sigmoid: smoothly goes from 0 to 1 around z_t
    s = 0.5 * (1.0 + bk.tanh(x))
    return theta_0 + (theta_inf - theta_0) * s



def logpdfm1_DPLDP_z( bk, 
    m1, z,
    # low-z hyperparameters
    alpha1_0, alpha2_0, mb_0,
    mu1_0, sigma1_0, mu2_0, sigma2_0,
    m1_low, m_high, delta_m1,
    lambda0_0, lambda1_0, lambda2_0,
    epsilon,
    # evolution hyperparameters for each θ in {alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2}
    alpha1_inf, z_alpha1, dz_alpha1,
    alpha2_inf, z_alpha2, dz_alpha2,
    mb_inf,    z_mb,     dz_mb,
    mu1_inf,   z_mu1,    dz_mu1,
    sigma1_inf,z_sigma1, dz_sigma1,
    mu2_inf,   z_mu2,    dz_mu2,
    sigma2_inf,z_sigma2, dz_sigma2,
    # NEW: mixture evolution specified by endpoints + shared (z_lambda, dz_lambda)
    lambda0_inf, lambda1_inf, lambda2_inf, z_lambda, dz_lambda,
    smoothing='LVK',
    simplex_repair=False,
    norm_gauss='uplow'
):
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
    #lambda2_0   = 1.0 - lambda0_0 - lambda1_0
    #lambda2_inf = 1.0 - lambda0_inf - lambda1_inf

    # convex combination: λ(z) = (1-S) λ(0) + S λ(∞)
    lambda0 = (1.0 - S_l) * lambda0_0 + S_l * lambda0_inf
    lambda1 = (1.0 - S_l) * lambda1_0 + S_l * lambda1_inf
    lambda2 = (1.0 - S_l) * lambda2_0 + S_l * lambda2_inf 

    # --- call your original m1 logpdf with z-dependent quantities ---
    return logpdfm1_DPLDP( bk, 
        m1,
        alpha1, alpha2, mb,
        mu1, sigma1, mu2, sigma2,
        m1_low, m_high, delta_m1,
        lambda0, lambda1, lambda2,
        epsilon,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
        norm_gauss=norm_gauss
    )





def logpdf_DPLDP_z( bk,
    theta, z,
    lambdaBBHmass_lowz,
    evo_params,
    force_m2_less_than_m1=False,
    has_m2_break=False,
    norm_gauss='uplow',
    smoothing='LVK',
    #resC=100, resN=500,
    interp_vals=None, interp_grids=None,
    simplex_repair=False,
):
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

    lambda2_0   = 1.0 - lambda0_0 - lambda1_0
    lambda2_inf = 1.0 - lambda0_inf - lambda1_inf

    lambda0 = (1.0 - S_l) * lambda0_0 + S_l * lambda0_inf
    lambda1 = (1.0 - S_l) * lambda1_0 + S_l * lambda1_inf
    lambda2 = (1.0 - S_l) * lambda2_0 + S_l * lambda2_inf

    # (we only pass lambda0, lambda1 to logpdfm1_DPLDP; lambda2 is implied)

    # rebuild a z-dependent mass-parameter vector for the downstream calls
    lambdaBBHmass_z = (
        alpha1, alpha2, mb,
        mu1, sigma1, mu2, sigma2,
        m1_low, m_high, delta_m1,
        lambda0, lambda1, lambda2, 
        beta, m2_low, delta_m2,
        epsilon, m_g, w_g, sig_g_low, sig_g_high
    )

    # now just call your original logpdf_DPLDP
    lpdf_ =  logpdf_DPLDP( bk,
        theta,
        lambdaBBHmass_z,
        force_m2_less_than_m1=force_m2_less_than_m1,
        has_m2_break=has_m2_break,
        norm_gauss=norm_gauss,
        smoothing=smoothing,
        #resC=resC, resN=resN,
        #interp_vals=interp_vals,
        #interp_grids=interp_grids,
        norm=False,
        simplex_repair=simplex_repair,
    )

    ln = logNorm_DPLDP_z(bk,
        z,
        alpha1_0, alpha2_0, mb_0, mu1_0, sigma1_0, mu2_0, sigma2_0,
        m1_low, m_high, delta_m1, lambda0_0, lambda1_0, lambda2_0, epsilon,
        alpha1_inf, z_alpha1, dz_alpha1,
        alpha2_inf, z_alpha2, dz_alpha2,
        mb_inf,    z_mb,     dz_mb,
        mu1_inf,   z_mu1,    dz_mu1,
        sigma1_inf,z_sigma1, dz_sigma1,
        mu2_inf,   z_mu2,    dz_mu2,
        sigma2_inf,z_sigma2, dz_sigma2,
        lambda0_inf, lambda1_inf, lambda2_inf, z_lambda, dz_lambda,
        #res=resN, 
        smoothing=smoothing,
        simplex_repair=simplex_repair,
        norm_gauss=norm_gauss
    )

    return lpdf_ - ln



def logNorm_DPLDP_z( bk, 
    z, 
    alpha1_0, alpha2_0, mb_0, mu1_0, sigma1_0, mu2_0, sigma2_0,
    m1_low, m_high, delta_m1, lambda0_0, lambda1_0, lambda2_0, epsilon,
    alpha1_inf, z_alpha1, dz_alpha1,
    alpha2_inf, z_alpha2, dz_alpha2,
    mb_inf,    z_mb,     dz_mb,
    mu1_inf,   z_mu1,    dz_mu1,
    sigma1_inf,z_sigma1, dz_sigma1,
    mu2_inf,   z_mu2,    dz_mu2,
    sigma2_inf,z_sigma2, dz_sigma2,
    lambda0_inf, lambda1_inf, lambda2_inf, z_lambda, dz_lambda,
    smoothing="LVK",
    #res=500,
    simplex_repair=False,
    norm_gauss='uplow'
):
    """
    Same semantics as your original logNorm_DPLDP_z, but:
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
    lp_flat = logpdfm1_DPLDP(bk, 
        M_flat,
        alpha1_flat, alpha2_flat, mb_flat,
        mu1_flat, sigma1_flat, mu2_flat, sigma2_flat,
        m1_low, m_high, delta_m1,
        lambda0_flat, lambda1_flat, lambda2_flat,
        epsilon,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
        norm_gauss=norm_gauss
    )

    # reshape back to (K, N1) and integrate over m1
    logp = lp_flat.reshape((K, N1))


    return bk.log(attrapzvec(bk, bk.exp(logp), m1_grid[None, :]
                             , axis=1))





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
        #fallback_width = bk.maximum(1e-8 * span, 10.0 * ramp_step)
        
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
        #bk.isfinite(m1) & bk.isfinite(m2) & bk.isfinite(z)
        (m1 >= m1_grid[0]) & (m1 <= m1_grid[-1])
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
    #ok = ok & bk.isfinite(lC)

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

    #ok = ok & bk.isfinite(ln)

    # ------------------------------------------------------------
    # 6) Assemble joint logpdf
    # ------------------------------------------------------------
    lpdf = lpdfm1 + lpdfm2 - lC - ln

    return bk.where(ok, lpdf, -jnp.inf)
