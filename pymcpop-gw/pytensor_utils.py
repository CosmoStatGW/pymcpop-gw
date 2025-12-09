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

import pytensor_tools as atools


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
    rng        : np.random.Generator or None
    
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
        
    z_max_inj = 0
    z_min_inj = 1e10

    lMc_max_inj = 0
    lMc_min_inj = 1e10
    
    for H0_ in (H0_max, H0_min):
        for Om_ in (Om_min, Om_max):
            for w0_ in (w0_min, w0_max):
                for Xi0_ in (Xi0_min, Xi0_max):
                    for nXi0_ in (nXi0_min, nXi0_max):

                        zinj = z_from_dL_fn( onp.squeeze(dLinj), float(H0_), float(Om_), float(w0_), float(Xi0_), float(nXi0_)  )
                
                        z_max_inj_ = onp.max(  zinj  )
                        z_min_inj_ = onp.min( zinj   )
                        #print("H0=%s, Om=%s, w0=%s, Xi0=%s, n=%s"%(H0_, Om_, w0_, Xi0_, nXi0_))
                        #print("zmin: %s, zmax:%s"%(z_min_inj_, z_max_inj_))

                        if z_max_inj_>z_max_inj:
                            z_max_inj = z_max_inj_

                        if z_min_inj_<z_min_inj:
                            z_min_inj = z_min_inj_

                        log_Mc_src = onp.log(Mc_det) - onp.log1p(zinj)

                        log_Mc_src_min_ = onp.min(log_Mc_src)
                        log_Mc_src_max_ = onp.max(log_Mc_src)

                        if log_Mc_src_max_>lMc_max_inj:
                            lMc_max_inj = log_Mc_src_max_

                        if log_Mc_src_min_<lMc_min_inj:
                            lMc_min_inj = log_Mc_src_min_

                       

    print("min, max injection distance: %s, %s Gpc"%(min_dL,max_dL))
    print("min, max injection redshift: %s, %s "%(z_min_inj,z_max_inj))
    print("min, max injection log(Mc_src): %s, %s "%(lMc_min_inj,lMc_max_inj))
    print("min, max injection logit(q): %s, %s "%(lq_min_inj,lq_max_inj))

    print("Finding data redshift and mass range...")
    
    z_maxs = []
    z_mins = []
    
    lMc_maxs = []
    lqs_maxs = []
    
    lMc_mins = []
    lqs_mins = []

    z_diffs = []
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
        #rng = np.random.default_rng()

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
        z_nodes = z_from_dL_fn(d_nodes, float(H0), float(Om), float(w0), float(Xi0), float(nXi0), )         
        z_data = onp.asarray(z_nodes, dtype=onp.float64)
        log_Mc_src = samples[:, 0]/(1+z_data)
        logit_q = samples[:, 1]

        z_data_max = onp.max(z_data) #onp.quantile(z_data, 0.99)
        z_data_min = onp.min(z_data) #onp.quantile(z_data, 0.01)

        lMc_data_max = onp.max(log_Mc_src) #onp.quantile(log_Mc_src, 0.99)
        lMc_data_min = onp.min(log_Mc_src) #onp.quantile(log_Mc_src, 0.01)

        lq_data_max = onp.max(logit_q) #onp.quantile(logit_q, 0.99)
        lq_data_min = onp.min(logit_q) #onp.quantile(logit_q, 0.01)


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
    
        z_maxs.append(z_data_max)
        z_mins.append(z_data_min)

        m1_mins.append(m1_data_min)
        m2_mins.append(m2_data_min)
        m1_maxs.append(m1_data_max)
        m2_maxs.append(m2_data_max)

        z_diffs.append(onp.mean(onp.diff(z_data)))
        lMc_diffs.append(onp.mean(onp.diff(log_Mc_src)))
        lqs_diffs.append(onp.mean(onp.diff(logit_q)))

        m1_diffs.append(onp.mean(onp.diff(m1_src)))
        m2_diffs.append(onp.mean(onp.diff(m2_src)))
    
    z_max_data = max(z_maxs)
    z_min_data = min(z_mins)

    lMc_max_data = max(lMc_maxs)
    lMc_min_data = min(lMc_mins)

    lq_max_data = max(lqs_maxs)
    lq_min_data = min(lqs_mins)

    m1_max_data = max(m1_maxs)
    m1_min_data = min(m1_mins)

    m2_max_data = max(m2_maxs)
    m2_min_data = min(m2_mins)

    z_diff = max(z_diffs)
    lMc_diff = max(lMc_diffs)
    lq_diff = max(lqs_diffs)
    m1_diff = max(m1_diffs)
    m2_diff = max(m2_diffs)
    
    print("min, max data redshift: %s, %s "%(z_min_data,z_max_data))
    print("min, max data log(Mc)_src: %s, %s "%(lMc_min_data,lMc_max_data))
    print("min, max data logit(q): %s, %s "%(lq_min_data,lq_max_data))
    print("min, max data m1: %s, %s "%(m1_min_data,m1_max_data))
    print("min, max data m2: %s, %s "%(m2_min_data,m2_max_data))

    print("min redshift scale: %s "%(z_diff))
    print("min log(Mc)_src scale: %s "%(lMc_diff))
    print("min logit(q) scale: %s "%(lq_diff))

    print("min m1 src scale: %s "%(m1_diff))
    print("min m2 src scale: %s "%(m2_diff))
    
    return dict(z_min_data=z_min_data, z_max_data=z_max_data,  lMc_min_data=lMc_min_data, lMc_max_data=lMc_max_data, lq_min_data=lq_min_data, lq_max_data=lq_max_data, z_diff=z_diff, lMc_diff=lMc_diff,  lq_diff=lq_diff, z_min_inj=z_min_inj, z_max_inj=z_max_inj, m1_diff=m1_diff, m2_diff=m2_diff, lMc_min_inj=lMc_min_inj, lMc_max_inj=lMc_max_inj, lq_max_inj=lq_max_inj, lq_min_inj=lq_min_inj )
    


def find_zgrid_bounds(wts_l_np, mus_l_np, cho_covs_l_np,
                          H0_range, Om_range, w0_range,Xi0_range, nXi0_range,
                          N, nd, 
                          dLinj,
                          z_from_dL_fn,
                          sampling_GW,
                          trials=1000, 
                      s0=0.10  ,
                      rng=onp.random.default_rng(123),
                      return_diff=False
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
    if return_diff:
        z_diffs=[]
        max_diffs=[]
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

        if return_diff:

            z_span = onp.quantile(z_data, 0.95) - onp.quantile(z_data, 0.05)  
            max_diffs.append(z_span)

            z_data.sort()
            tol = 1e-12
            z_data = z_data[onp.insert(onp.diff(z_data) > tol, 0, True)]
            
            dz = onp.diff(z_data)
            dz_pos = dz[dz > tol]
            z_diffs.append(onp.mean(dz_pos))

            
    
    z_max_data = max(z_maxs)
    z_min_data = min(z_mins)
    print("min, max data redshift: %s, %s "%(z_min_data,z_max_data))
    if return_diff:
        z_diff = 2*onp.quantile(z_diffs, 0.05)
        z_span = onp.quantile(max_diffs, 0.95) #max(max_diffs)
        print("data min redshift separation: %s "%(z_diff))
        print("data max redshift separation: %s "%(z_span))
    else:
        z_diff = None
        z_span = None
    
    
    return z_min_inj, z_max_inj, z_min_data, z_max_data, z_diff, z_span

def make_tqdm_callback(pbar):
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
