import time
import sys
import psutil
import os
import pytensor.tensor as at

from pytensor.graph.basic import graph_inputs
from pytensor.tensor.random.op import RandomVariable
from pytensor.graph.basic import graph_inputs, Variable

from pytensor.graph.basic import io_toposort
from pytensor.printing import debugprint
from pytensor import shared
import numpy as onp
from tqdm import tqdm


import xarray as xr
import arviz as az
import zarr

import pytensor_tools as atools

import pymc as pm






# # --- Helper to draw from the GMM in NumPy only ---
# def sample_from_per_event_gmm(wts, mus, chol_covs, Xwhite, rng=onp.random.default_rng(123)):
#     """
#     wts : (N, K)
#     mus : (N, K, D)
#     chol_covs : (N, K, D, D) lower-tri
#     Xwhite : (N, D) standard normals
#     """
#     N, K = wts.shape
#     u = rng.random((N, 1))
#     cdf = onp.cumsum(wts, axis=1)
#     k = (u < cdf).argmax(axis=1)            # (N,)
#     rows = onp.arange(N)
#     # draw one component per event with provided white noise Xwhite
#     return mus[rows, k, :] + (chol_covs[rows, k, :, :] @ Xwhite[..., None]).squeeze(-1)  # (N, D)


def safe_prefilter_injections(m1_det, m2_det, dL_gpc, zmax_of_dL, mmin_src=3.0):
    zmax = zmax_of_dL(dL_gpc)
    thr = mmin_src * (1.0 + zmax)  # detector-frame threshold per injection
    # discard if either component is guaranteed < mmin_src in source frame
    discard = (m1_det < thr) | (m2_det < thr)
    keep = ~discard
    return keep




def icovs_to_cholesky(icovs_l, jitter=0.0):
    """
    Convert per-event inverse covariances to Cholesky factors of covariances.

    Parameters
    ----------
    icovs_l : array, shape (N, K, D, D)
        Inverse covariance matrices for each event n and component k.
    jitter : float, optional
        Small diagonal term to add to the covariance before Cholesky,
        to guard against numerical issues (e.g. 1e-8).

    Returns
    -------
    cho_covs_l : array, shape (N, K, D, D)
        Lower-triangular Cholesky factors L such that cov = L @ L.T.
    """
    # Invert inverse covariances to get covariances
    covs_l = onp.linalg.inv(icovs_l)  # shape (N, K, D, D)

    if jitter > 0.0:
        # add jitter to the diagonal of each (N,K,D,D)
        D = covs_l.shape[-1]
        covs_l = covs_l + jitter * onp.eye(D)[None, None, :, :]

    # Cholesky factorization (assumes covs_l is SPD)
    cho_covs_l = onp.linalg.cholesky(covs_l)  # shape (N, K, D, D)

    return cho_covs_l


def sample_from_per_event_gmm(wts_l, mus_l, cho_covs_l, x, rng=None):
    """
    Draws one sample per event from a per-event Gaussian mixture.
    Mixture component is chosen per event using wts_l.
    
    Parameters
    ----------
    wts_l      : (N, K) mixture weights per event
    mus_l      : (N, K, D) means
    cho_covs_l : (N, K, D, D) Cholesky factors
    x          : (N, D) standard normal draws (or any external reparam vector)
    rng        : onp.random.Generator or None
    
    Returns
    -------
    samples    : (N, D) drawn samples
    ig         : (N,) chosen mixture index per event
    mu_sel     : (N, D) chosen means
    Lx         : (N, D) transformed x per event
    """

    if rng is None:
        rng = onp.random.default_rng()

    N, K = wts_l.shape
    _, _, D = mus_l.shape

    # 1) sample component indices ig[i] ~ Categorical(wts_l[i])
    u = rng.random(N)
    cdf = onp.cumsum(wts_l, axis=1)
    ig = (u[:, None] <= cdf).argmax(axis=1).astype(int)

    # 2) select per-event means and Cholesky factors
    idx = onp.arange(N)
    mu_sel = mus_l[idx, ig, :]           # (N, D)
    L_sel  = cho_covs_l[idx, ig, :, :]   # (N, D, D)

    # 3) batched multiply L_sel @ x
    #    matches PyTensor: at.sum(L_selected * x[:, None, :], axis=2)
    Lx = onp.einsum("nij,nj->ni", L_sel, x)

    # 4) GMM sample = mu + L x
    samples = mu_sel + Lx

    return samples #, ig, mu_sel, Lx
    

def robust_stat(x, trim=0.05):
    """Trimmed median absolute for stability."""
    x = onp.asarray(x, dtype=onp.float64).ravel()
    a, b = onp.quantile(x, [trim, 1-trim])
    x = x[(x>=a) & (x<=b)]
    return onp.median(x)



def evo_triplet(name, ivals, z_t_prior=None, dz_prior=None, theta0_init=None, theta_inf_prior=None):
    """
    Create (theta_inf, z_t, dz) random variables.
    - theta_inf_prior defaults to theta0 prior if not provided.
    """
    if theta_inf_prior is None:
        # fall back to same prior range as theta0 if possible
        # expected to be provided in priors[name_root]
        pass

    theta_inf = pm.Uniform(f"{name}_inf",
                           lower=theta_inf_prior[0], upper=theta_inf_prior[1],
                           initval=(ivals.get(f"{name}_inf", theta0_init)))

    z_t = pm.Uniform(f"z_{name}",
                     lower=z_t_prior[0], upper=z_t_prior[1],
                     initval=ivals.get(f"z_{name}", None))

    dz = pm.Uniform(f"dz_{name}",
                    lower=dz_prior[0], upper=dz_prior[1],
                    initval=ivals.get(f"dz_{name}", None))
    return theta_inf, z_t, dz


def safe_prefilter_injections_detector_frame(
    m1_det, m2_det, dL_gpc,
    dL_grid, zmax_grid,
    mmin_src=3.0,
):
    """
    Safe discard: if m_det/(1+z_max(dL)) < mmin_src then always out of support.
    """
    zmax = onp.interp(dL_gpc, dL_grid, zmax_grid)
    thr = mmin_src * (1.0 + zmax)  # detector-frame threshold per injection

    discard = (m1_det < thr) | (m2_det < thr)
    keep = ~discard
    return keep

def build_zmax_envelope_from_corners(
    z_from_dL_fn,
    dL_min_gpc, dL_max_gpc,
    priors,
    n_grid=4096,
    dtype= onp.float64,
):
    """
    Build a conservative z_max(dL) envelope by scanning the corners of the cosmology prior.

    Parameters
    ----------
    z_from_dL_fn : callable
        Compiled function like:
          z_from_dL_fn(dL, H0, Om, w0, Xi0, nXi0) -> z
        Must accept vector dL.
    dL_min_gpc, dL_max_gpc : float
        Range of luminosity distances in Gpc.
    priors : dict
        Must contain ranges like priors['H0']=[min,max], etc.
        Keys used: 'H0','Om','w0','Xi0','nXI0'  (note your key spelling)
    n_grid : int
        Resolution of the distance grid used to build the envelope.
    """
    # log-spaced distance grid (better coverage across many decades)
    dL_grid = onp.geomspace(dL_min_gpc, dL_max_gpc, n_grid).astype(dtype)

    H0_min, H0_max     = map(float, priors["H0"])
    Om_min, Om_max     = map(float, priors["Om"])
    w0_min, w0_max     = map(float, priors["w0"])
    Xi0_min, Xi0_max   = map(float, priors["Xi0"])
    nXi0_min, nXi0_max = map(float, priors["nXi0"])

    corners = []
    for H0_ in (H0_min, H0_max):
        for Om_ in (Om_min, Om_max):
            for w0_ in (w0_min, w0_max):
                for Xi0_ in (Xi0_min, Xi0_max):
                    for nXi0_ in (nXi0_min, nXi0_max):
                        corners.append((H0_, Om_, w0_, Xi0_, nXi0_))

    zmax_grid = onp.zeros_like(dL_grid)

    # scan corners
    for (H0_, Om_, w0_, Xi0_, nXi0_) in corners:
        z = z_from_dL_fn(dL_grid, H0_, Om_, w0_, Xi0_, nXi0_)
        z = onp.asarray(z, dtype=dtype)

        # guard against any numerical weirdness
        z = onp.where(onp.isfinite(z), z, 0.0)
        z = onp.clip(z, 0.0, onp.inf)

        zmax_grid = onp.maximum(zmax_grid, z)

    return dL_grid, zmax_grid


def find_mass_redshift_bounds(wts_l_np, mus_l_np, cho_covs_l_np,
                          H0_range, Om_range, w0_range, Xi0_range, nXi0_range,
                          N, nd, 
                          dLinj,
                        m1inj,
                        m2inj,
                          z_from_dL_fn,
                          sampling_GW,
                          trials=1000, 
                      s0=0.10  ,
                              is_observed=False,
                      rng=onp.random.default_rng()
                         ):


    H0_max = H0_range[1]
    H0_min = H0_range[0]

    Om_max = Om_range[1]
    Om_min = Om_range[0]

    w0_max = w0_range[1]
    w0_min = w0_range[0]

    Xi0_max = Xi0_range[1]
    Xi0_min = Xi0_range[0]

    nXi0_max = nXi0_range[1]
    nXi0_min = nXi0_range[0]

    # max injection redshift
    max_dL = onp.max(dLinj)
    min_dL = onp.min(dLinj)
        
    Mc_det, q = atools.Mcq_from_m1m2_at(m1inj, m2inj)

    logit_q = atools.logit(q)
    lq_min_inj = onp.min(logit_q)
    lq_max_inj = onp.max(logit_q)
        
    logz_max_inj = 0
    logz_min_inj = 1e10

    lMc_max_inj = 0
    lMc_min_inj = 1e10
    
    for H0_ in (H0_max, H0_min):
        for Om_ in (Om_min, Om_max):
            for w0_ in (w0_min, w0_max):
                for Xi0_ in (Xi0_min, Xi0_max):
                    for nXi0_ in (nXi0_min, nXi0_max):

                        if not is_observed:   
                            zinj = z_from_dL_fn( onp.squeeze(dLinj), float(H0_), float(Om_), float(w0_), float(Xi0_), float(nXi0_)  )
                        else:
                            zinj = onp.squeeze(dLinj)
                        
                        log_zinj = onp.log1p(zinj)
                        logz_max_inj_ = onp.max(  log_zinj  )
                        logz_min_inj_ = onp.min( log_zinj   )
                        #print("H0=%s, Om=%s, w0=%s, Xi0=%s, n=%s"%(H0_, Om_, w0_, Xi0_, nXi0_))
                        #print("zmin: %s, zmax:%s"%(z_min_inj_, z_max_inj_))

                        if logz_max_inj_>logz_max_inj:
                            logz_max_inj = logz_max_inj_

                        if logz_min_inj_<logz_min_inj:
                            logz_min_inj = logz_min_inj_

                        if not is_observed:    
                            log_Mc_src = onp.log(Mc_det) - onp.log1p(zinj)
                        else:
                            log_Mc_src = onp.log(Mc_det)
                        

                        log_Mc_src_min_ = onp.min(log_Mc_src)
                        log_Mc_src_max_ = onp.max(log_Mc_src)

                        if log_Mc_src_max_>lMc_max_inj:
                            lMc_max_inj = log_Mc_src_max_

                        if log_Mc_src_min_<lMc_min_inj:
                            lMc_min_inj = log_Mc_src_min_

                       

    print("min, max injection distance: %s, %s Gpc"%(min_dL,max_dL))
    if not is_observed:  
        print("min, max injection log(1+redshift): %s, %s "%(logz_min_inj,logz_max_inj))
        print("min, max injection log(Mc_src): %s, %s "%(lMc_min_inj,lMc_max_inj))
    else:
        print("min, max injection log(1+distance): %s, %s "%(logz_min_inj,logz_max_inj))
        print("min, max injection log(Mc_det): %s, %s "%(lMc_min_inj,lMc_max_inj))
    print("min, max injection logit(q): %s, %s "%(lq_min_inj,lq_max_inj))

    print("Finding data redshift and mass range...")
    
    logz_maxs = []
    logz_mins = []
    
    lMc_maxs = []
    lqs_maxs = []
    
    lMc_mins = []
    lqs_mins = []

    logz_diffs = []
    lqs_diffs = []
    lMc_diffs = []

    m1_diffs = []
    m2_diffs = []

    m1_maxs = []
    m2_maxs = []
    m1_mins = []
    m2_mins = []
    
    for _ in tqdm(range(trials)):
        #Xwhite = rng.standard_normal((N, nd))
        # N = number of events, D = dimension
        #rng = onp.random.default_rng()

        Xwhite = rng.standard_normal(size=(N, nd))

        #if 'gmm' in sampling_GW:
        samples = sample_from_per_event_gmm(wts_l_np, mus_l_np, cho_covs_l_np, Xwhite)
        # elif sampling_GW=='gauss':
        #     samples = mus_l_np + onp.einsum("nij,nj->ni", cho_covs_l_np, x)


        d_nodes = onp.exp(samples[:, 2])            

        if H0_range[0]!=H0_range[1]:
            H0 = rng.uniform(*H0_range)
        else:
            H0 = H0_range[0]
        if Om_range[0]!=Om_range[1]:
            Om = rng.uniform(*Om_range)
        else:
            Om = Om_range[0]
        if w0_range[0]!=w0_range[1]:
            w0 = rng.uniform(*w0_range)
        else:
            w0=w0_range[0]

        if Xi0_range[0]!=Xi0_range[1]:
            Xi0 = rng.uniform(*Xi0_range)
        else:
            Xi0=Xi0_range[0]
        if nXi0_range[0]!=nXi0_range[1]:
            nXi0 = rng.uniform(*nXi0_range)
        else:
            nXi0=nXi0_range[0]
                
                  

        # data redshifts
        if not is_observed:  
            z_nodes = z_from_dL_fn(d_nodes, float(H0), float(Om), float(w0), float(Xi0), float(nXi0), )         
        else:
            z_nodes = d_nodes   
        
        z_data = onp.asarray(z_nodes, dtype=onp.float64)

        
        logz_data = onp.log1p(z_data)

        if not is_observed:  
            log_Mc_src = samples[:, 0]-logz_data
        else:
            log_Mc_src = samples[:, 0]

        #print(log_Mc_src.max())
        
        logit_q = samples[:, 1]

        logz_data_max = onp.max(logz_data) 
        logz_data_min = onp.min(logz_data) 

        lMc_data_max = onp.max(log_Mc_src) 
        lMc_data_min = onp.min(log_Mc_src) 

        lq_data_max = onp.max(logit_q) 
        lq_data_min = onp.min(logit_q) 

        # logz_data_max = onp.quantile(z_data, 0.99)
        # logz_data_min = onp.quantile(z_data, 0.01)

        # lMc_data_max = onp.quantile(log_Mc_src, 0.99)
        # lMc_data_min = onp.quantile(log_Mc_src, 0.01)

        # lq_data_max = onp.quantile(logit_q, 0.99)
        # lq_data_min = onp.quantile(logit_q, 0.01)


        m1_src, m2_src = atools.m1m2_from_Mcq_at( onp.exp(log_Mc_src), atools.inv_logit(logit_q) )
        m1_data_min = onp.min(m1_src)
        m2_data_min = onp.min(m2_src)
        m1_data_max = onp.max(m1_src)
        m2_data_max = onp.max(m2_src)
        
        # print("m1 src: ")
        # print(m1_src)

        # print("m2 src: ")
        # print(m2_src)
        
        lMc_maxs.append(lMc_data_max)
        lqs_maxs.append(lq_data_max)

        lMc_mins.append(lMc_data_min)
        lqs_mins.append(lq_data_min)
    
        logz_maxs.append(logz_data_max)
        logz_mins.append(logz_data_min)

        m1_mins.append(m1_data_min)
        m2_mins.append(m2_data_min)
        m1_maxs.append(m1_data_max)
        m2_maxs.append(m2_data_max)


        logz_data.sort()
        log_Mc_src.sort()
        logit_q.sort()
        m1_src.sort()
        m2_src.sort()
        
        tol = 1e-12
        logz_data = logz_data[onp.insert(onp.diff(logz_data) > tol, 0, True)]
        log_Mc_src = log_Mc_src[onp.insert(onp.diff(log_Mc_src) > tol, 0, True)]
        logit_q = logit_q[onp.insert(onp.diff(logit_q) > tol, 0, True)]
        m1_src = m1_src[onp.insert(onp.diff(m1_src) > tol, 0, True)]
        m2_src = m2_src[onp.insert(onp.diff(m2_src) > tol, 0, True)]
            
        dz = onp.diff(logz_data)
        dz_pos = dz[dz > tol]
        
        dMc = onp.diff(log_Mc_src)
        dMc_pos = dMc[dMc > tol]
        
        dq = onp.diff(logit_q)
        dq_pos = dq[dq > tol]

        dm1 = onp.diff(m1_src)
        dm1_pos = dm1[dm1 > tol]

        dm2 = onp.diff(m2_src)
        dm2_pos = dm2[dm2 > tol]

        # logz_diffs.append( onp.quantile(dz_pos, 0.01) ) #onp.min(dz_pos))
        # lMc_diffs.append( onp.quantile(dMc_pos, 0.01) )  #onp.min(dMc_pos))
        # lqs_diffs.append( onp.quantile(dq_pos, 0.01) )  #onp.min(dq_pos))

        # m1_diffs.append(onp.quantile(dm1_pos, 0.01))
        # m2_diffs.append(onp.quantile(dm2_pos, 0.01))

        logz_diffs.append( onp.mean(dz_pos))
        lMc_diffs.append( onp.min(dMc_pos))
        lqs_diffs.append( onp.min(dq_pos))

        m1_diffs.append( onp.min(dm1_pos))
        m2_diffs.append( onp.min(dm2_pos))
    
    logz_max_data = max(logz_maxs)
    logz_min_data = min(logz_mins)

    lMc_max_data = max(lMc_maxs)
    lMc_min_data = min(lMc_mins)

    lq_max_data = max(lqs_maxs)
    lq_min_data = min(lqs_mins)

    m1_max_data = max(m1_maxs)
    m1_min_data = min(m1_mins)

    m2_max_data = max(m2_maxs)
    m2_min_data = min(m2_mins)

    logz_diff = max(logz_diffs)
    lMc_diff = onp.mean(lMc_diffs)
    lq_diff = onp.mean(lqs_diffs)
    m1_diff = onp.mean(m1_diffs)
    m2_diff = onp.mean(m2_diffs)

    if not is_observed:
        print("min, max data log(1+redshift): %s, %s "%(logz_min_data,logz_max_data))
        print("min, max data log(Mc)_src: %s, %s "%(lMc_min_data,lMc_max_data))
        print("min, max data logit(q): %s, %s "%(lq_min_data,lq_max_data))
        print("min, max data m1: %s, %s "%(m1_min_data,m1_max_data))
        print("min, max data m2: %s, %s "%(m2_min_data,m2_max_data))
    
        print("min log(1+redshift) scale: %s "%(logz_diff))
        print("min log(Mc)_src scale: %s "%(lMc_diff))
        print("min logit(q) scale: %s "%(lq_diff))
    
        print("min m1 src scale: %s "%(m1_diff))
        print("min m2 src scale: %s "%(m2_diff))

    else:
        print("min, max data log(1+dL): %s, %s "%(logz_min_data,logz_max_data))
        print("min, max data log(Mc_D): %s, %s "%(lMc_min_data,lMc_max_data))
        print("min, max data logit(q): %s, %s "%(lq_min_data,lq_max_data))
        print("min, max data m1_D: %s, %s "%(m1_min_data,m1_max_data))
        print("min, max data m2_D: %s, %s "%(m2_min_data,m2_max_data))
    
        print("min log(1+dL) scale: %s "%(logz_diff))
        print("min log(Mc_D) scale: %s "%(lMc_diff))
        print("min logit(q) scale: %s "%(lq_diff))
    
        print("min m1_D scale: %s "%(m1_diff))
        print("min m2_D scale: %s "%(m2_diff))
    
    return dict(logz_min_data=logz_min_data, logz_max_data=logz_max_data,  lMc_min_data=lMc_min_data, lMc_max_data=lMc_max_data, lq_min_data=lq_min_data, lq_max_data=lq_max_data, logz_diff=logz_diff, lMc_diff=lMc_diff,  lq_diff=lq_diff, logz_min_inj=logz_min_inj, logz_max_inj=logz_max_inj, m1_diff=m1_diff, m2_diff=m2_diff, lMc_min_inj=lMc_min_inj, lMc_max_inj=lMc_max_inj, lq_max_inj=lq_max_inj, lq_min_inj=lq_min_inj )
    


def find_zgrid_bounds(wts_l_np, mus_l_np, cho_covs_l_np,
                          H0_range, Om_range, w0_range,Xi0_range, nXi0_range,
                          N, nd, 
                          dLinj,
                          z_from_dL_fn,
                          sampling_GW,
                          trials=1000, 
                      s0=0.10  ,
                      rng=onp.random.default_rng(123)
                         ):


    H0_max = H0_range[1]
    H0_min = H0_range[0]

    Om_max = Om_range[1]
    Om_min = Om_range[0]

    w0_max = w0_range[1]
    w0_min = w0_range[0]

    Xi0_max = Xi0_range[1]
    Xi0_min = Xi0_range[0]

    nXi0_max = nXi0_range[1]
    nXi0_min = nXi0_range[0]

    # max injection redshift
    max_dL = onp.max(dLinj)
    min_dL = onp.min(dLinj)
        
        
        
    z_max_inj = 0
    z_min_inj = 1e10
    for H0_ in (H0_max, H0_min):
        for Om_ in (Om_min, Om_max):
            for w0_ in (w0_min, w0_max):
                for Xi0_ in (Xi0_min, Xi0_max):
                    for nXi0_ in (nXi0_min, nXi0_max):
                
                        z_max_inj_ = onp.max( z_from_dL_fn( onp.squeeze(dLinj), float(H0_), float(Om_), float(w0_), float(Xi0_), float(nXi0_)  )   )

                        z_min_inj_ = onp.min( z_from_dL_fn( onp.squeeze(dLinj), float(H0_), float(Om_), float(w0_), float(Xi0_), float(nXi0_)  )   )
                        #print("H0=%s, Om=%s, w0=%s, Xi0=%s, n=%s"%(H0_, Om_, w0_, Xi0_, nXi0_))
                        #print("zmin: %s, zmax:%s"%(z_min_inj_, z_max_inj_))

                        if z_max_inj_>z_max_inj:
                            z_max_inj = z_max_inj_

                        if z_min_inj_<z_min_inj:
                            z_min_inj = z_min_inj_

    print("min, max injection distance: %s, %s Gpc"%(min_dL,max_dL))
    print("min, max injection redshift: %s, %s "%(z_min_inj,z_max_inj))

    print("Finding data redshift range...")
    z_maxs = []
    z_mins = []
    for _ in tqdm(range(trials)):
        
        Xwhite = rng.standard_normal((N, nd))

        #if 'gmm' in sampling_GW:
        samples = sample_from_per_event_gmm(wts_l_np, mus_l_np, cho_covs_l_np, Xwhite)
        # elif sampling_GW=='gauss':
        #     samples = mus_l_np + (cho_covs_l_np @ Xwhite[..., None])[..., 0]  
        #     # mus_s + at.matmul(cho_s, x[..., None])[..., 0]  

        d_nodes = onp.exp(samples[:, 2])            

        if H0_range[0]!=H0_range[1]:
            H0 = rng.uniform(*H0_range)
        else:
            H0 = H0_range[0]
        if Om_range[0]!=Om_range[1]:
            Om = rng.uniform(*Om_range)
        else:
            Om = Om_range[0]
        if w0_range[0]!=w0_range[1]:
            w0 = rng.uniform(*w0_range)
        else:
            w0=w0_range[0]

        if Xi0_range[0]!=Xi0_range[1]:
            Xi0 = rng.uniform(*Xi0_range)
        else:
            Xi0=Xi0_range[0]
        if nXi0_range[0]!=nXi0_range[1]:
            nXi0 = rng.uniform(*nXi0_range)
        else:
            nXi0=nXi0_range[0]
                
                  

        # data redshifts
        z_nodes = z_from_dL_fn(d_nodes, float(H0), float(Om), float(w0), float(Xi0), float(nXi0), )         
        z_data = onp.asarray(z_nodes, dtype=onp.float64)

        z_data_max = onp.quantile(z_data, 0.99)
        z_data_min = onp.quantile(z_data, 0.01)

        z_maxs.append(z_data_max)
        z_mins.append(z_data_min)
    
    z_max_data = max(z_maxs)
    z_min_data = min(z_mins)
    print("min, max data redshift: %s, %s "%(z_min_data,z_max_data))
    
    return z_min_inj, z_max_inj, z_min_data, z_max_data



def _bin_indices(x, edges):
    """
    x:      (B, K) values to bin
    edges:  (NBINS+1,) monotonically increasing bin edges
    returns idx: (B, K) int64 in [0, NBINS-1]
    """
    idx = at.searchsorted(edges, x, side="right") - 1
    idx = at.clip(idx, 0, edges.shape[0] - 2)
    return idx.astype("int64")


def _scatter_sum_batched(values, idx, nbins):
    """
    values: (B, K) floatX  -> per-sample contribution to add to the bin
    idx:    (B, K) int64   -> bin index per sample
    nbins:  python int or 0-d tensor, number of bins
    return: (B, nbins) floatX, sum of 'values' per bin for every batch row
    """
    B, K = values.shape
    base = at.arange(B, dtype="int64") * nbins        # (B,)
    pos = (base[:, None] + idx).flatten()             # (B*K,)
    val = values.flatten()                            # (B*K,)
    out = at.zeros((B * nbins,), values.dtype)        # (B*nbins,)
    out = at.inc_subtensor(out[pos], val, inplace=False)
    return out.reshape((B, nbins))

def pt_vec(x, DT="float64"):
    x = onp.asarray(x, dtype=DT).reshape(-1)
    return shared(x, borrow=True)  # or at.as_tensor_variable(x) if truly tiny


def dump_uniform_sources(outputs, context=2):
    outs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    nodes = io_toposort([], outs)
    rv_nodes = [n for n in nodes if isinstance(getattr(n, "op", None), RandomVariable)]
    print(f"Found {len(rv_nodes)} RandomVariable nodes")
    for k, n in enumerate(rv_nodes, 1):
        print(f"\n[{k}] RV op: {n.op}  | owner: {n}")
        # who uses this random draw?
        for out in n.outputs:
            for client in out.clients:
                print("  used by:", client)
        # small subgraph around it (VERY helpful)
        try:
            print("\n--- debugprint around this node ---")
            print(debugprint(n.outputs, print_type=True, stop_on_name=True, depth=context))
        except Exception as e:
            print("debugprint failed:", e)
            
# ---------- flatten containers ----------
def _flatten(name, obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(f"{name}.{k}", v)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flatten(f"{name}[{i}]", v)
    else:
        yield name, obj

# ---------- classification ----------
def classify_tensor(x, model=None):
    """
    Classify x w.r.t. RandomVariables.
    Returns dict with:
      has_rv: bool
      rv_nodes: list[str] of RV op types
      is_value_var: bool
      source_rv: RV (if x is a value var)
    """
    info = {
        "has_rv": False,
        "rv_nodes": [],
        "is_value_var": False,
        "source_rv": None,
    }

    # Non-PyTensor objects are deterministic
    if not isinstance(x, Variable):
        return info

    # Is x itself a value-var?
    if model is not None and x in model.values_to_rvs:
        info["is_value_var"] = True
        info["source_rv"] = model.values_to_rvs[x]

    # Find any RVs in x's graph
    try:
        ins_gen = graph_inputs([x])
        ins = list(ins_gen)  # materialize here so exceptions are caught
        rv_ops = []
        for v in ins:
            if getattr(v, "owner", None) and isinstance(getattr(v.owner, "op", None), RandomVariable):
                rv_ops.append(type(v.owner.op).__name__)
        if rv_ops:
            info["has_rv"] = True
            info["rv_nodes"] = sorted(set(rv_ops))
    except Exception:
        # If graph introspection fails, assume deterministic for safety in reporting
        return info

    return info

def print_input_rv_report(model=None, **kwargs):
    """
    Example:
        print_input_rv_report(
            model=model,
            m1inj=m1inj, m2inj=m2inj, dLinj=dLinj, spinsInj=spinsInj,
            log_p_draw=lpdinj, Lambda=Lambda_val, dL_grid=dL_grid_val, z_grid=z_grid_val
        )
    """
    lines = []
    for name, obj in kwargs.items():
        for leaf_name, leaf in _flatten(name, obj):
            info = classify_tensor(leaf, model=model)

            status_bits = []
            if info["is_value_var"]:
                status_bits.append("VALUE_VAR")
            if info["has_rv"]:
                status_bits.append("CONTAINS_RV")
            status = "deterministic" if not status_bits else ",".join(status_bits)

            src = ""
            if info["is_value_var"] and info["source_rv"] is not None:
                src_name = getattr(info["source_rv"], "name", "<unnamed RV>")
                src = f" (from {src_name})"

            rv_types = f" rv_ops={info['rv_nodes']}" if info["rv_nodes"] else ""
            dtype = getattr(leaf, "dtype", type(leaf).__name__)
            shape_str = str(getattr(leaf, "shape", ""))

            lines.append(f"- {leaf_name}: {status}{src}{rv_types} | dtype={dtype} | shape={shape_str}")

    print("\n".join(lines))




def as_value_var(x, model):
    # If x is an RV, return its value-var; otherwise return x unchanged
    return model.rvs_to_values.get(x, x)

def stack_as_values(elems, model):
    # Apply as_value_var to each element, then stack
    vals = [as_value_var(e, model) for e in elems]
    return at.stack(vals)


def make_tqdm_callback_full(pbar):
    t0 = time.perf_counter()
    last_refresh = [t0]
    div_count    = [0]
    last_nsteps  = [None]   # last observed n_steps (after warmup)
    last_ss      = [None]   # last observed step_size (after warmup)

    def _get_stat(name, args, kwargs, chain):
        # 1) kwarg directly
        if name in kwargs:
            try:
                return kwargs[name]
            except Exception:
                pass
        # 2) Draw-like object in kwargs
        d = kwargs.get("draw", None)
        if d is not None and hasattr(d, name):
            try:
                return getattr(d, name)
            except Exception:
                pass
        # 3) Trace.get_sampler_stats (first positional arg often is the trace)
        if len(args) >= 2:
            trace = args[0]
            try:
                arr = trace.get_sampler_stats(name, chains=[chain] if chain is not None else None)
                if len(arr):
                    return arr[-1]
            except Exception:
                pass
        return None

    def _get_diverging(args, kwargs, tuning, chain):
        if "diverging" in kwargs:
            return bool(kwargs["diverging"])
        d = kwargs.get("draw", None)
        if d is not None and hasattr(d, "diverging"):
            return bool(getattr(d, "diverging"))
        if len(args) >= 2:
            trace = args[0]
            try:
                arr = trace.get_sampler_stats("diverging", chains=[chain] if chain is not None else None)
                if len(arr):
                    return bool(arr[-1])
            except Exception:
                pass
        return False

    def cb(*args, **kwargs):
        """
        Supports:
          - PyMC >=5: (draw, tuning, chain) or kwargs with a Draw-like object
          - Older patterns: (trace, draw) or (draw,)
        """
        draw = tuning = chain = None

        if len(args) >= 3:
            draw, tuning, chain = args[:3]
        elif len(args) == 2:
            draw = args[1]
            tuning = kwargs.get("tuning")
            chain  = kwargs.get("chain", 0)
        elif len(args) == 1:
            draw = args[0]
            tuning = kwargs.get("tuning")
            chain  = kwargs.get("chain", 0)
        else:
            draw  = kwargs.get("draw", 0)
            tuning = kwargs.get("tuning")
            chain  = kwargs.get("chain", 0)

        # Normalize tuning flag if present on Draw-like object
        if hasattr(draw, "tuning") and tuning is None:
            tuning = bool(getattr(draw, "tuning", False))

        # === stats & divergence (only after warmup) ===
        if tuning is False:
            if _get_diverging(args, kwargs, tuning, chain):
                div_count[0] += 1

            nsteps = _get_stat("n_steps", args, kwargs, chain)
            if nsteps is not None:
                try:
                    last_nsteps[0] = int(nsteps)
                except Exception:
                    pass

            ss = _get_stat("step_size", args, kwargs, chain)
            if ss is not None:
                try:
                    last_ss[0] = float(ss)
                except Exception:
                    pass

        # === progress updates (same cadence as your original, lightly throttled) ===
        pbar.update(1)

        now = time.perf_counter()
        if (pbar.n % 25) == 0 and (now - last_refresh[0]) >= 0.25:
            phase = "warmup" if tuning else "sampling"
            rate  = pbar.n / max(now - t0, 1e-9)
            # Build a tiny postfix string without heavy formatting
            extras = [f"div={div_count[0]}"]
            if last_nsteps[0] is not None:
                extras.append(f"nsteps={last_nsteps[0]}")
            if last_ss[0] is not None:
                # format step size compactly
                extras.append(f"ss={last_ss[0]:.3g}")
            pbar.set_postfix_str(f"{phase} | {rate:5.1f} it/s | " + " ".join(extras), refresh=False)
            last_refresh[0] = now

    return cb

def make_tqdm_callback(pbar):
    t0 = time.perf_counter()
    last_refresh = [t0]  # mutable box to avoid nonlocal

    def cb(*args, **kwargs):
        """
        Supports:
          - PyMC >=5: (draw, tuning, chain) or Draw-like object in kwargs
          - Older patterns: (trace, draw) or (draw,)
        """
        draw = tuning = chain = None

        if len(args) >= 3:
            draw, tuning, chain = args[:3]
        elif len(args) == 2:
            # could be (trace, draw)
            draw = args[1]
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)
        elif len(args) == 1:
            draw = args[0]
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)
        else:
            draw = kwargs.get("draw", 0)
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)

        # If PyMC passed a Draw-like object, grab tuning flag from it (we don't need draw as int)
        if hasattr(draw, "tuning") and tuning is None:
            tuning = bool(getattr(draw, "tuning", False))

        # Update bar every callback (same as your original)
        pbar.update(1)

        # Postfix every 25 iters, but throttle by time to reduce overhead
        now = time.perf_counter()
        if (pbar.n % 25) == 0 and (now - last_refresh[0]) >= 0.25:
            phase = "warmup" if tuning else "sampling"
            rate = pbar.n / max(now - t0, 1e-9)
            pbar.set_postfix_str(f"{phase} | {rate:5.1f} it/s", refresh=False)
            last_refresh[0] = now

    return cb


class TqdmPerChainCallback(pm.callbacks.Callback):
    def __init__(self, draws=None, tune=None):
        # Only simple, picklable state here
        self.draws = draws
        self.tune = tune
        self._pbar = None
        self._t0 = None
        self._last_refresh = None

    def _parse_args(self, *args, **kwargs):
        """
        Mimic your original flexible signature handling.
        Supports:
          - PyMC >=5: (draw, tuning, chain)
          - Older patterns: (trace, draw), (draw,), or kwargs
        """
        draw = tuning = chain = None

        if len(args) >= 3:
            draw, tuning, chain = args[:3]
        elif len(args) == 2:
            # could be (trace, draw)
            draw = args[1]
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)
        elif len(args) == 1:
            draw = args[0]
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)
        else:
            draw = kwargs.get("draw", 0)
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)

        # If PyMC passed a Draw-like object with a .tuning attribute
        if hasattr(draw, "tuning") and tuning is None:
            tuning = bool(getattr(draw, "tuning", False))

        if chain is None:
            chain = 0

        return draw, bool(tuning), int(chain)

    def __call__(self, *args, **kwargs):
        draw, tuning, chain = self._parse_args(*args, **kwargs)

        # Lazily create tqdm inside each worker
        if self._pbar is None:
            total = None
            if self.draws is not None and self.tune is not None:
                total = self.draws + self.tune

            self._pbar = tqdm(
                total=total,
                desc=f"chain {chain}",
                position=chain,  # one bar per chain
                leave=True,
            )
            self._t0 = time.perf_counter()
            self._last_refresh = self._t0

        # Every callback => one step
        self._pbar.update(1)

        # Optional: same warmup/sampling + rate logic
        now = time.perf_counter()
        if (self._pbar.n % 25) == 0 and (now - self._last_refresh) >= 0.25:
            phase = "warmup" if tuning else "sampling"
            rate = self._pbar.n / max(now - self._t0, 1e-9)
            self._pbar.set_postfix_str(f"{phase} | {rate:5.1f} it/s", refresh=False)
            self._last_refresh = now     

def make_tqdm_callback_frequent(pbar):
    t0 = time.perf_counter()

    def cb(*args, **kwargs):
        """
        Supports:
          - PyMC >=5: (draw, tuning, chain)
          - Older patterns: (trace, draw) or (draw,)
        """
        draw = tuning = chain = None

        if len(args) >= 3:
            draw, tuning, chain = args[:3]
        elif len(args) == 2:
            # could be (trace, draw)
            draw = args[1]
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)
        elif len(args) == 1:
            draw = args[0]
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)
        else:
            draw = kwargs.get("draw", 0)
            tuning = kwargs.get("tuning")
            chain = kwargs.get("chain", 0)

        # update bar (throttle if you like)
        pbar.update(1)
        if (pbar.n % 25) == 0:
            phase = "warmup" if tuning else "sampling"
            rate = pbar.n / max(time.perf_counter() - t0, 1e-9)
            pbar.set_postfix_str(f"{phase} | {rate:5.1f} it/s", refresh=False)

    return cb




def _group_keys_list(g):
    # zarr>=2.18 returns generator; normalize to list
    return list(g.group_keys()) if hasattr(g, "group_keys") else []

def _array_keys_list(g):
    return list(g.array_keys()) if hasattr(g, "array_keys") else []

def _infer_chain_draw_shape_from_group(grp):
    """
    Try to infer (n_chain, n_draw) from any array in the group.
    Arrays are expected to be shaped (chain, draw, ...).
    """
    for name in _array_keys_list(grp):
        arr = grp[name][...]
        if arr.ndim >= 2:
            return arr.shape[0], arr.shape[1]
    # fallback: look for attrs the backend sometimes writes
    attrs = getattr(grp, "attrs", {})
    n_chain = attrs.get("n_chain") or attrs.get("chains")
    n_draw  = attrs.get("n_draw")  or attrs.get("draws")
    if n_chain and n_draw:
        return int(n_chain), int(n_draw)
    raise ValueError("Could not infer (chain, draw) from group arrays or attrs")

def _mk_da(name, arr, chain_names=None, draw_names=None):
    import numpy as np
    import xarray as xr

    arr = onp.asarray(arr)
    # ensure at least (chain, draw)
    if arr.ndim < 2:
        if arr.ndim == 1:
            arr = arr[None, :]  # (1, N)
        else:
            arr = arr[None, None]  # scalar -> (1,1)

    n_chain, n_draw = arr.shape[:2]

    # If provided names mismatch sizes, regenerate to match the *data*
    if chain_names is None or len(chain_names) != n_chain:
        chain_coords = onp.arange(n_chain)
    else:
        chain_coords = onp.asarray(chain_names)

    if draw_names is None or len(draw_names) != n_draw:
        draw_coords = onp.arange(n_draw)
    else:
        draw_coords = onp.asarray(draw_names)

    dims = ["chain", "draw"] + [f"{name}_dim{i}" for i in range(arr.ndim - 2)]
    coords = {"chain": chain_coords, "draw": draw_coords}
    for i, L in enumerate(arr.shape[2:]):
        coords[dims[2 + i]] = onp.arange(L)

    return xr.DataArray(arr, dims=dims, coords=coords, name=name)

def _load_group_as_dataset(grp, chain_names=None, draw_names=None):
    import xarray as xr
    if grp is None:
        return None
    keys = list(grp.array_keys()) if hasattr(grp, "array_keys") else []
    if not keys:
        return None

    ds_vars = {}
    for key in keys:
        arr = grp[key][...]
        # NOTE: _mk_da will *override* chain/draw coords if sizes don’t match
        ds_vars[key] = _mk_da(key, arr, chain_names, draw_names)
    return xr.Dataset(ds_vars)

def load_pymc_zarr_trace_robust(store_path):
    """
    Returns an ArviZ InferenceData from a PyMC ZarrTrace directory.
    Tries ArviZ's from_zarr first; if that fails, constructs by hand.
    """
    # 1) Fast path: ArviZ (works when store has CF-ish metadata)
    try:
        # consolidated=False avoids the “failed to open consolidated metadata” warning->fallback
        return az.InferenceData.from_zarr(store_path)
    except Exception as e_fast:
        print("ArviZ.from_zarr path failed, falling back:", repr(e_fast))

    # 2) Manual path
    root = zarr.open_group(store_path, mode="r")
    gkeys = _group_keys_list(root)
    # infer chain/draw once from the richest group we can find
    probe_grp = None
    for candidate in ("posterior", "sample_stats", "warmup_posterior"):
        if candidate in gkeys:
            grp = root[candidate]
            if _array_keys_list(grp):
                probe_grp = grp
                break
    if probe_grp is None:
        raise RuntimeError(f"No posterior/sample_stats arrays found in {store_path}")

    n_chain, n_draw = _infer_chain_draw_shape_from_group(probe_grp)
    chain_names = onp.arange(n_chain)
    draw_names  = onp.arange(n_draw)

    posterior         = _load_group_as_dataset(root["posterior"], chain_names, draw_names) \
                        if "posterior" in gkeys else None
    sample_stats      = _load_group_as_dataset(root["sample_stats"], chain_names, draw_names) \
                        if "sample_stats" in gkeys else None
    warmup_posterior  = _load_group_as_dataset(root.get("warmup_posterior"), chain_names, draw_names) \
                        if "warmup_posterior" in gkeys else None
    warmup_stats      = _load_group_as_dataset(root.get("warmup_sample_stats"), chain_names, draw_names) \
                        if "warmup_sample_stats" in gkeys else None

    return az.InferenceData(
        posterior=posterior,
        sample_stats=sample_stats,
        warmup_posterior=warmup_posterior,
        warmup_sample_stats=warmup_stats,
    )



def drop_object_vars(idata: az.InferenceData) -> az.InferenceData:
    idata = idata.copy()
    for grp in idata.groups():
        ds = getattr(idata, grp, None)
        if ds is None:
            continue
        bad = [v for v in ds.data_vars if ds[v].dtype == object]
        if bad:
            ds = ds.drop_vars(bad)
            setattr(idata, grp, ds)
    return idata


# Writes output both on std output and on log file
class Logger(object):
    
    def __init__(self, fname):
        self.terminal = sys.__stdout__
        self.log = open(fname, "w+")
        self.log.write('--------- LOG FILE ---------\n')
        print('Logger created log file: %s' %fname)
        #self.write('Logger')
       
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.close
        sys.stdout = sys.__stdout__
        
    def isatty(self):
        return False
