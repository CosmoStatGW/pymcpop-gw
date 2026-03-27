#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import os
import argparse
import json
import sys
import h5py

import numpy as np

from sklearn.mixture import GaussianMixture
from scipy.linalg import inv, det
from scipy.special import logsumexp

from scipy.stats import ks_2samp

from tqdm import tqdm
import pandas as pd

import importlib

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'serif'
plt.rcParams["mathtext.fontset"] = "cm"

import corner

MGCpath = '../../MGCosmoPop/MGCosmoPop'
sys.path.append(MGCpath)


from dataStructures.O1O2data import O1O2Data
from dataStructures.O3adata import O3aData
from dataStructures.O3bdata import O3bData
from dataStructures.O4adata import O4aData
from dataStructures.mockData import GWMockData


import data_tools as dt
import pytensor_utils_old as ptools

#######################################################################################
#######################################################################################
def logit(p):
    return np.log(p) - np.log(1 - p)

def flogit(p, xmin=-1, xmax=1):
    return np.log(-xmin + p) - np.log(xmax - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def inv_flogit(p, xmin=-1, xmax=1):
    return ( np.exp(p)*xmax+xmin ) / (1 + np.exp(p) )

def m1m2_from_Mcq(Mc, q):
    
    m1 = Mc*(1+q)**(1./5.)/q**(3./5.)
    m2 = q*m1

    return m1, m2


def fit_cho(allsamples, allNsamples,spins='default', skymap=False, inclination=False):

    samples_means = []
    samples_cho_covs = []

    nevs = len(allsamples)
    print('There are %s events'%nevs)
    
    
    for i in tqdm(range(nevs), ):
    
        jmax = allNsamples[i]
    
        
        m1ds = allsamples[i, :jmax, 0]
        m2ds = allsamples[i, :jmax, 1]
        dLs = allsamples[i, :jmax, 2]
        
        Mcs = (m1ds*m2ds)**(3/5)/(m1ds+m2ds)**(1/5)
        qs = m2ds/m1ds 
        
        qlogit = logit(qs)

        pts_ = [ np.log(Mcs), qlogit, np.log(dLs),]

        if spins=='default':
            chi1 = allsamples[i, :jmax, 3]
            chi2 = allsamples[i, :jmax, 4]
            cost1 = allsamples[i, :jmax, 5]
            cost2 = allsamples[i, :jmax, 6]
            
            lchi1 = logit(chi1)
            lchi2 = logit(chi2)
            lcost1 = flogit(cost1)
            lcost2 = flogit(cost2)

            pts_.append( lchi1) 
            pts_.append(lchi2) 
            pts_.append( lcost1) 
            pts_.append( lcost2  )

            s_start = 7
            
        elif spins=='aligned':
            chi1z = allsamples[i, :jmax, 3]
            chi2z = allsamples[i, :jmax, 4]
            pts_.append(  flogit(chi1z) ) 
            pts_.append(  flogit(chi2z)  )

            s_start = 5
            
        elif spins=='none':
            s_start = 3
            pass


        if skymap:
            ra = allsamples[i, :jmax, s_start]
            dec = allsamples[i, :jmax, s_start+1]
            pts_.append( flogit(ra, xmin=0,xmax=2*np.pi) )
            pts_.append( flogit(dec, xmin=-np.pi/2, xmax=np.pi/2) )
            istart = s_start+2
        else:
            istart = s_start
                    
        if inclination:
            iota = allsamples[i, :jmax, istart]
            pts_.append( flogit(iota, xmin=0,xmax=np.pi)  )

        print('Number of dimensions of each event: %s'%len(pts_))
            
        pts = np.stack( pts_ ).T
        
        samples_means.append( pts.mean(axis=0) )
        samples_cho_covs.append( np.linalg.cholesky( np.cov(pts, rowvar=False) ) )
            
        
    
    samples_means = np.asarray(samples_means)
    samples_cho_covs = np.asarray(samples_cho_covs)
    
    print("Done.")

    return samples_means, samples_cho_covs




def build_dL_prior_interpolants(dL_samples, prior_vals, prior_floor=1e-300, dL_bounds_full=None):
    """
    Build per-event interpolation tables for log(1 / pi_PE(dL)).

    Parameters
    ----------
    dL_samples : array-like, shape (N_events, N_samples)
        dL sample locations for each event.
    prior_vals : array-like, shape (N_events, N_samples)
        PE prior values evaluated at the corresponding dL sample locations.
    prior_floor : float, optional
        Minimum allowed prior value for numerical stability.
    dL_bounds_full : array-like, shape (N_events, 2), optional
        Full posterior-sample support bounds [dL_min_full, dL_max_full] for each event.
        If None, bounds are taken from the tabulated dL_samples.

    Returns
    -------
    dL_grid : ndarray, shape (N_events, N_samples)
        Sorted dL values for each event.
    loginvprior_grid : ndarray, shape (N_events, N_samples)
        log(1 / prior_vals) evaluated on the sorted dL grid.
    dL_min : ndarray, shape (N_events,)
        Minimum supported dL per event.
    dL_max : ndarray, shape (N_events,)
        Maximum supported dL per event.
    """

    dL_samples = np.asarray(dL_samples)
    prior_vals = np.asarray(prior_vals)

    if dL_samples.shape != prior_vals.shape:
        raise ValueError(
            "dL_samples and prior_vals must have the same shape. "
            "Got %s and %s" % (dL_samples.shape, prior_vals.shape)
        )

    if prior_floor <= 0:
        raise ValueError("prior_floor must be > 0")

    if not np.isfinite(dL_samples).all():
        raise ValueError("dL_samples contains non-finite values")

    if not np.isfinite(prior_vals).all():
        raise ValueError("prior_vals contains non-finite values")

    if (dL_samples <= 0).any():
        raise ValueError("dL_samples must be strictly positive")

    nevs, nsamps = dL_samples.shape

    if dL_bounds_full is not None:
        dL_bounds_full = np.asarray(dL_bounds_full)
        if dL_bounds_full.shape != (nevs, 2):
            raise ValueError(
                "dL_bounds_full must have shape (%s, 2), got %s"
                % (nevs, dL_bounds_full.shape)
            )
        if not np.isfinite(dL_bounds_full).all():
            raise ValueError("dL_bounds_full contains non-finite values")
        if (dL_bounds_full[:, 0] <= 0).any():
            raise ValueError("dL_bounds_full lower bounds must be > 0")
        if (dL_bounds_full[:, 1] <= dL_bounds_full[:, 0]).any():
            raise ValueError("Each full dL bound must satisfy max > min")

    dL_grid = np.empty_like(dL_samples, dtype=float)
    loginvprior_grid = np.empty_like(prior_vals, dtype=float)
    dL_min = np.empty(nevs, dtype=float)
    dL_max = np.empty(nevs, dtype=float)

    for i in range(nevs):
        order = np.argsort(dL_samples[i])

        dL_sorted = dL_samples[i, order]
        prior_sorted = prior_vals[i, order]

        if np.any(np.diff(dL_sorted) <= 0):
            raise ValueError(
                "dL values for event %s are not strictly increasing after sorting. "
                "Interpolation requires unique dL points." % i
            )

        prior_sorted = np.maximum(prior_sorted, prior_floor)

        dL_grid[i] = dL_sorted
        loginvprior_grid[i] = -np.log(prior_sorted)

        if dL_bounds_full is None:
            dL_min[i] = dL_sorted[0]
            dL_max[i] = dL_sorted[-1]
        else:
            dL_min[i] = dL_bounds_full[i, 0]
            dL_max[i] = dL_bounds_full[i, 1]

    return dL_grid, loginvprior_grid, dL_min, dL_max
    
def fit_gmm(allsamples, allNsamples, allNames=None, fout_plot=None, spins='default', skymap=False, inclination=False, n_components=np.arange(0,20), dil_factor=1, refit=False, fname_base=None, safety_number=10, imin=0, imax=-1, fsummaries='summaries', fplots='plots'):

    gmm_log_wts_l = []
    gmm_means_l = []
    gmm_icovs_l = []
    gmm_covs_l = []
    gmm_cho_covs_l = []
    gmm_log_dets_l = []
    all_gmm_l = []
    
    
    nevs = len(allsamples)
    print('There are %s events'%nevs)

    if imax==-1:
        iend=nevs
    elif imax<=nevs:
        iend=imax   
    else:
        raise ValueError()
        
    nevs_fit = iend-imin
    print('Fitting events between %s and %s. Total %s'%(imin, iend, nevs_fit))
    
    for i in tqdm(range(imin, iend), ): 

        jmax = allNsamples[i]
    
        m1ds = allsamples[i, :jmax, 0]
        m2ds = allsamples[i, :jmax, 1]
        dLs = allsamples[i, :jmax, 2]

        Mcs = (m1ds*m2ds)**(3/5)/(m1ds+m2ds)**(1/5)
        qs = m2ds/m1ds 
        qlogit = logit(qs)
        
        pts_ = [dil_factor*np.log(Mcs), qlogit, np.log(dLs),]

        
        if spins=='default':
            chi1 = allsamples[i, :jmax, 3]
            chi2 = allsamples[i, :jmax, 4]
            cost1 = allsamples[i, :jmax, 5]
            cost2 = allsamples[i, :jmax, 6]
            lchi1 = logit(chi1)
            lchi2 = logit(chi2)
            lcost1 = flogit(cost1)
            lcost2 = flogit(cost2)
            

            s_start = 7

            pts_.append(dil_factor*lchi1) 
            pts_.append(dil_factor*lchi2) 
            pts_.append(dil_factor*lcost1) 
            pts_.append(dil_factor*lcost2 )
        
        elif spins=='aligned':
            chi1z = allsamples[i, :jmax, 3]
            chi2z = allsamples[i, :jmax, 4]
            lchi1z = flogit(chi1z)
            lchi2z = flogit(chi2z)
   
            s_start = 5

            pts_.append(dil_factor*lchi1z) 
            pts_.append(dil_factor*lchi2z )
        
        elif spins=='none':
            s_start = 3
            pass


        if skymap:
            ra = allsamples[i, :jmax, s_start]
            dec = allsamples[i, :jmax, s_start+1]
            pts_.append( flogit(ra, xmin=0,xmax=2*np.pi) )
            pts_.append( flogit(dec, xmin=-np.pi/2, xmax=np.pi/2) )
            istart = s_start+2
        else:
            istart = s_start
                    
        if inclination:
            iota = allsamples[i, :jmax, istart]
            pts_.append( flogit(iota, xmin=0,xmax=np.pi)  )
            
        print('Number of dimensions of each event: %s'%len(pts_))
        
        pts = np.stack( pts_ ).T


        print('\nEvent %s is fit between %s and %s components'%(allNames[i], n_components[i][0], n_components[i][1]))
        min_found_already =False
        if refit:
            bics_prev = np.loadtxt( os.path.join(fout_plot, fsummaries, 'BICs_ev_%s_%s.txt'%(i,allNames[i])))
            n_c_prev = np.loadtxt( os.path.join(fout_plot, fsummaries, 'n_comp_BICs_ev_%s_%s.txt'%(i,allNames[i])))
            prev_max=max(n_c_prev)
            min_bic_idx_ = np.argmin(bics_prev)
            nc_best_ = n_c_prev[min_bic_idx_]

            print("Min bic for %s was at %s components. "%(allNames[i], nc_best_))
            if nc_best_ < prev_max-safety_number:
                print("Bic was already minimised safely below max number of components of previous fit: n_max=%s, min bic at %s. Not refitting this event"%(prev_max, nc_best_))
                min_found_already = True
            else:
                print('Refit.')
        
        if  (n_components[i][0]>=  n_components[i][1]) or min_found_already:
            print('No need to refit.')
            
            means_ = np.load( os.path.join(base_fname, '%s_gmm_means.npy'%fname_base) )
            covariances_ = np.load( os.path.join(base_fname, '%s_gmm_covs.npy'%fname_base) )
            gmm_cho_covs_l_ = np.load( os.path.join(base_fname, '%s_gmm_cho_covs.npy'%fname_base) )
            precisions_ = np.load( os.path.join(base_fname, '%s_gmm_icovs.npy'%fname_base) )
            gmm_log_dets_l_  = np.load( os.path.join(base_fname, '%s_gmm_log_dets.npy'%fname_base) )
            weights_ = np.load( os.path.join(base_fname, '%s_gmm_log_wts.npy'%fname_base) )
            
            gmm = None
        else:

            n_components_ = np.arange(n_components[i][0], n_components[i][1]+1 )
            
            models = np.array([GaussianMixture(nc, 
                                               covariance_type='full', #'full', 
                                               random_state=1312, 
                                               tol=1e-3,
                                               n_init=1,
                                               #init_params='random_from_data',
                                               verbose=0 ,
                                               max_iter=int(1e05)
                                            ) for nc in n_components_ ] )

            best_bic_ = 1e100
            idx_best = 0
            stop = False

            bics = []
            
            for idx, model in enumerate(models):

                if not stop:                
                    model.fit(pts)
                    bic_ = model.bic(pts)
                    if bic_<best_bic_:
                        
                        last_best = idx_best
                        
                        idx_best = idx
                        best_bic_ = bic_
                        print("New best bic found at %s components"%n_components_[idx_best])
                        print("Previous was %s components"%n_components_[last_best])
                        #print('idx, last_best: %s, %s'%(idx, last_best))
                    else:
                        if n_components_[idx]-n_components_[idx_best]>safety_number:
                            print('Stopping at %s components'%n_components_[idx])
                            stop=True
                
                bics.append(bic_)
                    
                
            #bics = np.array([gmm.bic(pts) for gmm in models])
                
            min_bic_idx = np.argmin(bics)
            nc_best = n_components_[min_bic_idx]
    
            print("Best bic at %s components"%nc_best)
            
            if fout_plot is not None:
                if allNames is not None:
                    name=allNames[i]
                else:
                    name=i
                #print("N. components for ev %s is %s"%(name, nc_best))
                      
                if refit:
                    bics = np.concatenate([bics_prev, bics])
                    n_components_ = np.concatenate([n_c_prev, n_components_])

                np.savetxt( os.path.join(fout_plot, fsummaries, 'BICs_ev_%s_%s.txt'%(i,name)), bics ) 
                np.savetxt( os.path.join(fout_plot, fsummaries, 'n_comp_BICs_ev_%s_%s.txt'%(i,name)), n_components_ ) 

                #print(n_components_)
                #print(bics)
                plt.plot(n_components_,bics )
                plt.axvline(nc_best)
                plt.title("Ev %s, N. components %s"%(name, nc_best))
                plt.xlabel('BIC')
                plt.xlabel('N')
                fout='BIC_ev_%s_%s'%(i,name)
                if refit:
                    fout+='_refit_imax-%s'%n_components[i][1]
                plt.savefig(os.path.join(fout_plot, fplots, fout+'.pdf'))
                
                #plt.show()
                plt.close()
            
            
            
            gmm = models[min_bic_idx] #GaussianMixture(nc_best)
            
        
            weights_ = gmm.weights_
            means_ = gmm.means_
            covariances_ = gmm.covariances_
            print('shape of cov')
            print(covariances_.shape)
            try:
                gmm_cho_covs_l_ = [  np.linalg.cholesky( gmm.covariances_[k])  for k in range(nc_best)] 
            except:
                gmm_cho_covs_l_ = gmm.covariances_ #[  1/ gmm.covariances_[k]  for k in range(nc_best)]
            precisions_ = gmm.precisions_
            try:
                gmm_log_dets_l_ = [np.log(det(gmm.covariances_[k])) for k in range(nc_best)]
            except:
                gmm_log_dets_l_ = [np.log(np.product(gmm.covariances_)) for k in range(nc_best)]
    
        gmm_log_wts_l.append(np.log(weights_))
        gmm_means_l.append(means_)
        gmm_covs_l.append( covariances_ )
        gmm_cho_covs_l.append( gmm_cho_covs_l_   )
        gmm_icovs_l.append( precisions_  )
        all_gmm_l.append(gmm)
        gmm_log_dets_l.append( gmm_log_dets_l_ )
    
    
    
    print("Done.")

    nd = gmm_means_l[0].shape[1]
    
    allNgm = [len(gmm_log_wts_l[i]) for i in range(len(gmm_log_wts_l))]
    Ngm = max(allNgm)
    
    gmm_log_wts_l_full = np.log(np.zeros( (nevs_fit, Ngm) ))
    gmm_means_l_full = np.zeros( (nevs_fit, Ngm, nd) )
    gmm_icovs_l_full = np.zeros( (nevs_fit, Ngm, nd, nd) )
    gmm_cho_covs_l_full = np.zeros( (nevs_fit, Ngm, nd, nd) )
    gmm_covs_l_full = np.zeros( (nevs_fit, Ngm, nd, nd) )
    gmm_log_dets_l_full = np.log(np.ones( (nevs_fit, Ngm) ))

    for i in range(nevs_fit):
        ngm_  =  allNgm[i]

        gmm_log_wts_l_full[i, :ngm_] = gmm_log_wts_l[i]
        gmm_means_l_full[i, :ngm_] = gmm_means_l[i]
        gmm_icovs_l_full[i, :ngm_] = gmm_icovs_l[i]
        gmm_cho_covs_l_full[i, :ngm_] = gmm_cho_covs_l[i]
        gmm_covs_l_full[i, :ngm_] = gmm_covs_l[i]
        gmm_log_dets_l_full[i, :ngm_] = gmm_log_dets_l[i]

    
    return gmm_means_l_full, gmm_icovs_l_full, gmm_covs_l_full, gmm_cho_covs_l_full, gmm_log_dets_l_full, gmm_log_wts_l_full, all_gmm_l, allNgm



def plot_samples(allsamples, 
                 all_gmm_l,  
                 allNsamples,
                 pval_th=0.05,
                 imax=None, 
                 names_plot=None, 
                 allNames=None, 
                 spins='default',  
                 skymap=False,
                 inclination=False,
                 lmeans=None, 
                 lstds=None, 
                 dil_factor=1,
                 fout=None, 
                 fout_suff='', 
                 KS=False, 
                 show=False, 
                 nbins=50, 
                 ngm=None, 
                 cl=[0.5, 0.90],
                 fplots='plots/'
                ):


    if names_plot is not None and imax is not None:
        raise ValueError()

    elif names_plot is not None:
        if allNames is None:
            raise ValueError("Please provide list of all names")
        if allsamples is None:
            raise ValueError('Provide original posterior samples')
        idx_plot = [ i for i in range(len(allsamples)) if allNames[i] in names_plot ]

    elif imax is not None:
        idx_plot = np.argsort(allNames)[:imax]

    print()    
    print('Making cornerplots...')
    print()
    for i in idx_plot:
        print()
        print('*'*40)
        if allNames is not None:
            name=allNames[i]
        else:
            name=i
        print('Event %s'%name)
        print('*'*40)
        
        nsam =  allNsamples[i]
        print('\nNsamples is %s'%nsam)
        if ngm is not None:
            print('N GMM is %s'%ngm[i])
        ns =  max(10000, nsam)
        plot=True
        try:
            s1s = all_gmm_l[i].sample(ns)[0].T
        except:
            try:
                s1s = all_gmm_l[i].T
                print(s1s.shape)
            except:
                print('No replot for %s'%name)
                plot=False

        if plot:
            lMc = s1s[0]/dil_factor
            lq = s1s[1]
            ld = s1s[2]
    
            qs = inv_logit(lq) 

            m1s, m2s = m1m2_from_Mcq(np.exp(lMc), qs )
            dLs = np.exp(ld)

            a2_ = [m1s, m2s, dLs,]
            
    
            
            if spins=='default':
                chi1 = inv_logit(s1s[3]/dil_factor)
                chi2 = inv_logit(s1s[4]/dil_factor)
                cost1 = inv_flogit(s1s[5]/dil_factor)
                cost2 = inv_flogit(s1s[6]/dil_factor)
                s_start=7
                a2_+=[chi1, chi2, cost1, cost2]
            elif spins=='aligned':
                chi1z = inv_flogit(s1s[3]/dil_factor)
                chi2z = inv_flogit(s1s[4]/dil_factor)
                a2_+=[chi1z, chi2z, ]
                s_start=5
            elif spins=='none':
                s_start=3
                pass

            if skymap:
                ra = inv_flogit( s1s[s_start], xmin=0, xmax=2*np.pi)
                dec = inv_flogit( s1s[s_start+1], xmin=-np.pi/2, xmax=np.pi/2 )
                i_start = s_start+2
                a2_+=[ra, dec,]
            else:
                i_start=s_start
                
            if inclination:
                iota = inv_flogit(s1s[i_start], xmin=0, xmax=np.pi)
                a2_+=[iota]
                
    
                     
    
            a2 = np.asarray(a2_).T

            if KS:
                if allsamples is None:
                    raise ValueError('Provide original posterior samples for KS test')
                kstest_m1 = ks_2samp(allsamples[i, :nsam, 0], m1s)
                kstest_m2 = ks_2samp(allsamples[i, :nsam, 1], m2s)
                kstest_dL = ks_2samp(allsamples[i, :nsam, 2], dLs)
        
                print("\nKS test results:\n")
                
                print("m1: %s"%str(kstest_m1))
                if kstest_m1.pvalue>pval_th:
                    print("For m1, the null hypothesis thatthat the two samples were drawn from the same distribution cannot be rejected")
                else:
                    print("For m1, the null hypothesis thatthat the two samples were drawn from the same distribution IS REJECTED! Do a better fit.")
        
                print("\nm2: %s"%str(kstest_m1))
                if kstest_m2.pvalue>pval_th:
                    print("For m2, the null hypothesis thatthat the two samples were drawn from the same distribution cannot be rejected")
                else:
                    print("For m2, the null hypothesis thatthat the two samples were drawn from the same distribution IS REJECTED! Do a better fit.")
                
                print("\ndL: %s"%str(kstest_dL))
                if kstest_dL.pvalue>pval_th:
                    print("For dL, the null hypothesis thatthat the two samples were drawn from the same distribution cannot be rejected")
                else:
                    print("For dL, the null hypothesis thatthat the two samples were drawn from the same distribution IS REJECTED! Do a better fit.")
            
                
                print()
            
     
            print('---- Plot in the space of detector-frame variables: m1, m2, d_L, spins')

            labels = [r'$m_1^d$', r'$m_2^d$', r'$d_L$', ]
            if spins=='default':
                labels+=[r'$\chi_1$', r'$\chi_2$', r'$\cos \theta_1$', r'$\cos \theta_2$']
            elif spins=='aligned':
                labels+=[r'$\chi_{1,z}$', r'$\chi_{2,z}$',]
            
            if skymap:
                 labels+=[r'$\alpha$', r'$\delta$',]
            if inclination:
                labels+=[r'$\iota$',]
                      
                      
                      
            
            
            if allsamples is not None:
                a1_ =   [ allsamples[i, :nsam, k] for k in range(3)] 
                myr = [ ( max(0, min( np.percentile(allsamples[i, :nsam, 0], .1), np.percentile( m1s, .1) ) ), max( np.percentile(allsamples[i, :nsam, 0], 97), np.percentile(m1s, 97) )  ),
                    ( max(0,min( allsamples[i, :nsam, 1].min(), m2s.min() )*(1-0.1) ), max( allsamples[i, :nsam, 1].max(), m2s.max() )  ),
                    ( max(0, min( allsamples[i, :nsam, 2].min(), dLs.min() )*(1-0.1) ), max( allsamples[i, :nsam, 2].max(), dLs.max() )  ),]
                
                if spins=='default':  
                    a1_+=[allsamples[i, :nsam, k] for k in range(3,7)]
                    s_start=7   
                    myr+=[( -0.1, 1.1 ), (-0.1,1.1), (-1.1,1.1), (-1.1, 1.1)]
                
                elif spins=='aligned':
                    a1_+=[allsamples[i, :nsam, k] for k in range(3,5)]
                    s_start=5
                    myr+=[(-1.1,1.1), (-1.1, 1.1)]
                if skymap:
                    a1_+= [allsamples[i, :nsam, k] for k in range(s_start,s_start+2) ]
                    istart=s_start+2
                    myr+=[( max(0, min( np.percentile(allsamples[i, :nsam, s_start], .1), np.percentile( ra, .1) ) ), max( np.percentile(allsamples[i, :nsam, s_start], 97), np.percentile(ra, 97) )  ),
                         ( max(0, min( np.percentile(allsamples[i, :nsam, s_start+1], .1), np.percentile( dec, .1) ) ), max( np.percentile(allsamples[i, :nsam, s_start+1], 97), np.percentile(dec, 97) )  )
                         ]
                else:
                    istart=s_start
                if inclination:
                    a1_+= [allsamples[i, :nsam, k] for k in range(istart,istart+1)] 
                    myr+=[( max(0, min( np.percentile(allsamples[i, :nsam, istart], .1), np.percentile( iota, .1) ) ), max( np.percentile(allsamples[i, :nsam, istart], 97), np.percentile(iota, 97) )  ),]
                    
                    
                    

                a1 =  np.asarray( a1_ ).T
                    
                    
            
                
                fig = corner.corner(a1,
                                   color='darkred',
                                 
                                 title_fmt='.2f', 
                               levels=cl,
                                density=True,
                                no_fill_contours=True,
                                    plot_datapoints=False,
                                #quantiles=[0.05, 0.5, 0.95]
                                    show_titles=False,
                                range=myr,
                                bins=nbins,     
                            hist_kwargs=dict(density=True, lw=1.5),
                            contour_kwargs=dict(linewidths=1.5),
                               )

            else:
                # No original samples provided 
                
                fig=None
                if spins=='default' and not skymap and not inclination:
                    myr = [ ( max(0,  np.percentile( m1s, .1) ) ,  np.percentile(m1s, 97)  ),
                    ( max(0, m2s.min() )*(1-0.1),  m2s.max() )  ,
                    ( max(0,  dLs.min() )*(1-0.1) ,  dLs.max()   ),
                       ( -0.1, 1.1 ), (-0.1,1.1), (-1.1,1.1), (-1.1, 1.1)
                      ]
                else:
                    myr=None
            
            
            fig=corner.corner( a2, fig=fig,
                                   color='darkblue',
                                 
                                 title_fmt='.2f', 
                               levels=cl,
                                density=True,
                               no_fill_contours=True,
                                    plot_datapoints=False,
                                #quantiles=[0.05, 0.5, 0.95],
                              labels=labels,
                              show_titles=False,
                               label_kwargs={"fontsize": 20},
                              range=myr,
                                bins=nbins,
                              weights=np.ones(len(a2))/len(a2)*ns,
                            hist_kwargs=dict(density=True, lw=1.5, ),
                            contour_kwargs=dict(linewidths=1.5),
                               )
    
      
            
            if fout is not None:
                plt.savefig(os.path.join(fout, fplots, 'corner_ev_%s_%s_theta%s.pdf'%(i, name, fout_suff)))
            
            if show:
                plt.show()
            plt.close()
    
    
            #######
            idx_dil = (0, 3,4,5,6)
            print()
            print('---- Plot in the space of remapped detector-frame variables (see paper)')

            labels_remap = [r'$\log \mathcal{M}_c^D$', r'$\log \frac{q}{1-q}$', r'$\log d_L$', ]
            
            if spins=='default':
                labels_remap+=[r'$\log \frac{\chi_1}{1-\chi_1}$', r'$\log \frac{\chi_2}{1-\chi_2}$', r'$\log \frac{1+\cos \theta_1}{1-\cos \theta_1}$', r'$\log \frac{1+\cos \theta_2}{1-\cos \theta_2}$']
            elif spins=='aligned':
                labels_remap+=[r'$\log \frac{\chi_{1,z}}{1-\chi_{1,z}}$', r'$\log \frac{\chi_{2,z}}{1-\chi_{2,z}}$', ]
            
            if skymap:
                 labels_remap+=[r'$\log \frac{ \alpha }{2\pi - \alpha}$', r'$\log \frac{ \delta + \pi/2}{\pi/2- \delta}$']
            if inclination:
                labels_remap+=[r'$\log \frac{ \iota }{\pi - \iota}$',]
                            
                            
                            
                            

            
            if allsamples is not None:
                m1ds = allsamples[i, :nsam, 0]
                m2ds = allsamples[i, :nsam, 1]
                Mcs = (m1ds*m2ds)**(3/5)/(m1ds+m2ds)**(1/5)
            
                lMC_s = np.log(Mcs)
                lq_s = logit( allsamples[i, :nsam, 1]/allsamples[i, :nsam, 0] )
                ld_s =  np.log(allsamples[i, :nsam, 2])


                a1_ = [ lMC_s, lq_s,ld_s, ]
                myr=myr = [ ( min( np.percentile( lMC_s, 0.1), np.percentile( lMc, 0.1) ), max( np.percentile(lMC_s, 99), np.percentile( lMc , 99)  )),
                    ( min( lq_s.min(), lq.min() ), max( lq_s.max(), lq.max() )  ),
                    ( min( ld_s.min(), ld.min() ), max(ld_s.max(), ld.max() )  ),]

                if spins=='default':
                    
                    lchi1_s = logit(allsamples[i, :nsam, 3])
                    lchi2_s = logit(allsamples[i, :nsam, 4])
                    lcost1_s = flogit(allsamples[i, :nsam, 5])
                    lcost2_s = flogit(allsamples[i, :nsam, 6])
    
                    a1_+=[ lchi1_s, lchi2_s,lcost1_s,lcost2_s ]
                
                    myr+=[
                   (  min( lchi1_s.min(), s1s[3].min() ) , max( lchi1_s.max(), s1s[3].max() )  ),
                   ( min( lchi2_s.min(), s1s[4].min() ) , max( lchi2_s.max(), s1s[4].max() )  ),
                   ( min( lcost1_s.min(), s1s[5].min() ) , max( lcost1_s.max(), s1s[5].max() )  ),
                   ( min( lcost2_s.min(), s1s[6].min() ) , max( lcost2_s.max(), s1s[6].max() ) ) 
                       
                  ]
                    s_start=7
                elif spins=='aligned':
                    lchi1_s = flogit(allsamples[i, :nsam, 3])
                    lchi2_s = flogit(allsamples[i, :nsam, 4])
                    a1_+=[ lchi1_s, lchi2_s ]
                    myr+=[
                   (  min( lchi1_s.min(), s1s[3].min() ) , max( lchi1_s.max(), s1s[3].max() )  ),
                   ( min( lchi2_s.min(), s1s[4].min() ) , max( lchi2_s.max(), s1s[4].max() )  ),
                  ]
                    s_start=5
                else:
                    s_start=3


                if skymap:
                    lra = flogit(allsamples[i, :nsam, s_start], xmin=0, xmax=2*np.pi)
                    ldec = flogit(allsamples[i, :nsam, s_start+1], xmin=-np.pi/2, xmax=np.pi/2)
                    a1_+=[ lra, ldec ]
                    istart=s_start+2
                    myr+=[  (  min( lra.min(), s1s[s_start].min() ) , max( lra.max(), s1s[s_start].max() )  ),
                   ( min( ldec.min(), s1s[s_start+1].min() ) , max( ldec.max(), s1s[s_start+1].max() )  ),
                         ]
                else:
                    istart=s_start
                if inclination:
                    liota = flogit(allsamples[i, :nsam, istart], xmin=0, xmax=np.pi)
                    a1_+=[ liota ]
                    myr+=[  (  min( liota.min(), s1s[istart].min() ) , max( liota.max(), s1s[istart].max() )  ),
                   
                         ]
                
                a1 = np.asarray( a1_ ).T
                
            
                fig = corner.corner( a1,
                                   color='darkred',
                                
                                 title_fmt='.2f', 
                               levels=cl,
                                density=True,
                               no_fill_contours=True,
                                    plot_datapoints=False,
                                #quantiles=[0.05, 0.5, 0.95]
                                #range=myr,
                                    show_titles=False,
                                bins=nbins,
                                 hist_kwargs=dict(density=True, lw=1.5),
                            contour_kwargs=dict(linewidths=1.5),
                                weights=np.ones(len(a1))/len(a1)*len(a1),
                               )

            else:
                # original samples not provided
                # does not work yet withour spin models, inclination and skymap
                fig=None

                if spins=='default':
                    myr = [ ( np.percentile( lMc, 0.1) , np.percentile( lMc , 99)  ),
                    (  lq.min() , lq.max() )  ,
                    (  ld.min() , ld.max() )  ,
                   (  s1s[3].min()  , s1s[3].max()   ),
                   (  s1s[4].min()  , s1s[4].max()  ),
                   (  s1s[5].min()  ,  s1s[5].max()   ),
                   (  s1s[6].min()  ,  s1s[6].max()  ) 
                       
                  ]
    
                a2 = np.asarray([s1s[k] if k not in idx_dil else s1s[k]/dil_factor for k in range(s1s.shape[0])]).T
                fig=corner.corner( a2, fig=fig,
                                   color='darkblue',
                                
                                 title_fmt='.2f', 
                               levels=cl,
                                density=True,
                               no_fill_contours=True,
                                    plot_datapoints=False,
                                #quantiles=[0.05, 0.5, 0.95]
                              #range=myr,
                                  labels=labels_remap,
                                  show_titles=False,
                                   label_kwargs={"fontsize": 20},
                                bins=nbins,
                               weights=np.ones(len(a2))/len(a2)*ns,
                            hist_kwargs=dict(density=True, lw=1.5),
                            contour_kwargs=dict(linewidths=1.5),
                               )
    
      
            if fout is not None:
                plt.savefig(os.path.join(fout, fplots, 'corner_ev_%s_%s_thetatil%s.pdf'%(i, name,fout_suff)))
            if show:
                plt.show()
            plt.close()



def plot_dL_prior_interpolants(
    dL_grid,
    loginvprior_grid,
    dL_min,
    dL_max,
    names,
    fout,
    dL_samples=None,
    prior_vals=None,
    prior_floor=1e-300,
    max_nevs=None,
    show=False,
    fplots='plots/'
):
    """
    Plot per-event log(1 / pi_PE(dL)) interpolation tables and, if provided,
    overlay the raw tabulated values as scatter points.
    """

    dL_grid = np.asarray(dL_grid)
    loginvprior_grid = np.asarray(loginvprior_grid)
    dL_min = np.asarray(dL_min)
    dL_max = np.asarray(dL_max)

    if dL_grid.shape != loginvprior_grid.shape:
        raise ValueError(
            "dL_grid and loginvprior_grid must have the same shape. "
            "Got %s and %s" % (dL_grid.shape, loginvprior_grid.shape)
        )

    nevs = dL_grid.shape[0]

    if len(names) != nevs:
        raise ValueError("names must have length %s, got %s" % (nevs, len(names)))

    if (dL_samples is None) ^ (prior_vals is None):
        raise ValueError("Provide both dL_samples and prior_vals, or neither.")

    if dL_samples is not None:
        dL_samples = np.asarray(dL_samples)
        prior_vals = np.asarray(prior_vals)

        if dL_samples.shape != dL_grid.shape:
            raise ValueError(
                "dL_samples must have shape %s, got %s" % (dL_grid.shape, dL_samples.shape)
            )
        if prior_vals.shape != dL_grid.shape:
            raise ValueError(
                "prior_vals must have shape %s, got %s" % (dL_grid.shape, prior_vals.shape)
            )

    if max_nevs is None:
        idx_plot = range(nevs)
    else:
        idx_plot = range(min(nevs, max_nevs))

    for i in idx_plot:
        name = names[i]

        plt.figure()

        plt.plot(
            dL_grid[i],
            loginvprior_grid[i],
            lw=1.5,
            label=r'$\log(1/\pi_{\rm PE}(d_L))$'
        )

       # plt.plot(dL_grid[i], -2*np.log(dL_grid[i]), label=r'$-2 \log(d_L)$')

        if dL_samples is not None:
            yscat = -np.log(np.maximum(prior_vals[i], prior_floor))
            plt.scatter(
                dL_samples[i],
                yscat,
                s=8,
                alpha=0.5,
                label='tabulated values'
            )

        plt.axvline(dL_min[i], ls='--', lw=1.0)
        plt.axvline(dL_max[i], ls='--', lw=1.0)

        plt.xlabel(r'$d_L$')
        plt.ylabel(r'$\log(1/\pi_{\rm PE}(d_L))$')
        plt.title(f'Event {name}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(fout, fplots, f'dL_prior_interp_ev_{i}_{name}.pdf'))

        if show:
            plt.show()

        plt.close()

#######################################################################################
#######################################################################################


parser = argparse.ArgumentParser()


parser.add_argument("--snr_th", default=1, type=float, required=False)
parser.add_argument("--far_th", default=1, type=float, required=False)
parser.add_argument("--dil_factor", default=1, type=float, required=False)
parser.add_argument("--n_gmm_min", default=1, type=int, required=False)
parser.add_argument("--n_gmm_max", default=10, type=int, required=False)
parser.add_argument("--fin_data", default='', type=str, required=True)
parser.add_argument("--fin_prior_dL", default='', type=str, required=False)
parser.add_argument("--prior_only", default=0, type=int, required=False)
parser.add_argument("--prior_floor", default=1e-30, type=float, required=False)

parser.add_argument("--fnames", nargs='+', type=str, required=True) # this is like O1, O2 , ... leave empty string for sim. data
parser.add_argument("--metadata", nargs='+', type=str, required=False)
parser.add_argument("--fout", default='GWTC-fits', type=str, required=False)
parser.add_argument("--ps_prior", default='nocosmo', type=str, required=False)
parser.add_argument("--plot", default=1, type=int, required=False)
parser.add_argument("--skymap", default=0, type=int, required=False)
parser.add_argument("--inclination", default=0, type=int, required=False)
parser.add_argument("--spins", default='default', type=str, required=False)
parser.add_argument("--imin", default=0, type=int, required=False)
parser.add_argument("--imax", default=-1, type=int, required=False)
parser.add_argument("--reweight", default=0, type=int, required=False)

parser.add_argument("--fsummaries", default='summaries', type=str, required=False)
parser.add_argument("--fplots", default='plots', type=str, required=False)


if __name__=='__main__':
    
    os.chdir("../")
    
    FLAGS = parser.parse_args()

    if not os.path.exists(FLAGS.fout):
            os.makedirs(FLAGS.fout)
            print('Created %s'%FLAGS.fout)
    else:
            print('Using %s'%FLAGS.fout)

    logfile = os.path.join(FLAGS.fout, 'logfile.txt')
    myLog = ptools.Logger(logfile)
    sys.stdout = myLog
    sys.stderr = myLog
    
    print("Will fit the following datasets:")
    print(FLAGS.fnames)
    
            #######################################################################################

    for ir, run_name in enumerate(FLAGS.fnames):
    
        fname_ = os.path.join(FLAGS.fin_data, run_name)
        print()
        print('#################################################')
        print('Fitting %s. Input file: %s'%(run_name, fname_ ))
        print('#################################################')
        print()

        if run_name == 'O1O2':
    
            data = O1O2Data(fname_, SNR_th=FLAGS.snr_th, FAR_th=FLAGS.far_th, which_spins=FLAGS.spins, metadata = FLAGS.metadata[ir] )
        
        elif run_name == 'O3a':
            
            events_names = {'not_use': ['GW150914_095045', 'GW151012_095443', 'GW151226_033853', 'GW170104_101158', 'GW170608_020116', 'GW170729_185629', 'GW170809_082821', 'GW170814_103043', 'GW170818_022509', 'GW170823_131358',
                            'GW190814','GW190814_211039', 'GW190425_081805'], 
                
                'use':None }
            
            data = O3aData(fname_, 
                           GWTC2_1=None, 
              events_use=events_names, SNR_th=FLAGS.snr_th, FAR_th=FLAGS.far_th,
              which_spins=FLAGS.spins,
              suffix_name=FLAGS.ps_prior,
                           metadata = FLAGS.metadata[ir]
             )
        
        elif run_name == 'O3b':
            
            events_names = {'not_use': ['GW150914_095045', 'GW151012_095443', 'GW151226_033853', 'GW170104_101158', 'GW170608_020116', 'GW170729_185629', 'GW170809_082821', 'GW170814_103043', 'GW170818_022509', 'GW170823_131358',
                            'GW190814'], 
                
                'use':None }
            
            data = O3bData(fname_, 
              #GWTC2_1=None, 
              events_use=events_names, SNR_th=FLAGS.snr_th, FAR_th=FLAGS.far_th,
              which_spins=FLAGS.spins,
              suffix_name=FLAGS.ps_prior,
                           metadata = FLAGS.metadata[ir]
             )

        elif run_name == 'O4a' or run_name=='pe_files':
            
            events_names = {'not_use': ['GW230518_125908','GW230529_181500'], 
                        'use':None }
            
            data = O4aData(fname_, 
              events_use=events_names, 
                SNR_th=FLAGS.snr_th, 
                FAR_th=FLAGS.far_th,
              which_spins=FLAGS.spins,
              #suffix_name=FLAGS.ps_prior,
                           metadata = FLAGS.metadata[ir]
             )
            
        else:
            # We are using simulated data 
            data = GWMockData( fname_, 
                                SNR_th=FLAGS.snr_th, 
                                which_spins=FLAGS.spins,
                                inclination=FLAGS.inclination,
                               reweight=FLAGS.reweight
                             )


        if FLAGS.ps_prior=='nocosmo' and run_name in ('O4a', 'pe_files' ):
            print("setting ps_prior = cosmo for O4a")
            FLAGS.ps_prior = 'cosmo'
        
        flist = ['%s'%(FLAGS.fout), 
                  '%s/%s/'%(FLAGS.fout, run_name.split('.')[0] ),
                  '%s/%s/snrth-%s_farth-%s/'%(FLAGS.fout,  run_name.split('.')[0], int(FLAGS.snr_th), int(FLAGS.far_th), ),
                  '%s/%s/snrth-%s_farth-%s/dil_factor-%s'%(FLAGS.fout, run_name.split('.')[0], int(FLAGS.snr_th), int(FLAGS.far_th), FLAGS.dil_factor)]

        base_str = '%s/%s/snrth-%s_farth-%s/dil_factor-%s/'%(FLAGS.fout, run_name.split('.')[0], int(FLAGS.snr_th), int(FLAGS.far_th), FLAGS.dil_factor)
        
        if run_name in ('O3a', 'O3b', 'O4a',  'pe_files'):

            flist+= ['%s/%s/snrth-%s_farth-%s/dil_factor-%s/%s/'%(FLAGS.fout, run_name, int(FLAGS.snr_th), int(FLAGS.far_th), FLAGS.dil_factor, FLAGS.ps_prior) ]

            base_str += '/%s/'%FLAGS.ps_prior

        flist+=[
                base_str+'/spin-%s'%(FLAGS.spins),
                 base_str+'/spin-%s/skymap-%s'%(FLAGS.spins, FLAGS.skymap),
                 base_str+'/spin-%s/skymap-%s/inclination-%s'%(FLAGS.spins, FLAGS.skymap, FLAGS.inclination),
                ]
        
    
        for p in flist:

            if not os.path.exists(p):
                os.makedirs(p)
                print('Created %s'%p)
            else:
                print('Using %s'%p)

        psum = flist[-1]+'/'+FLAGS.fsummaries
        if not os.path.exists(psum):
                os.makedirs(psum)
                print('Created %s'%psum)

        pplot = flist[-1]+'/'+FLAGS.fplots
        if not os.path.exists(pplot):
                os.makedirs(pplot)
                print('Created %s'%pplot)
            
        print()
    
        base_fname = flist[-1]
        print('Output foder is %s'%base_fname)
        
        
        fsam = os.path.join(base_fname, '%s_allnsamples.npy'%run_name.split('.')[0] )
        if not os.path.exists(fsam):
            np.save( fsam, data.Nsamples )
            print('Saved nsamples in %s'%fsam)
        else:
            print('nsamples already saved in %s'%fsam)

        fngmm = os.path.join(base_fname, '%s_allNgm.txt'%run_name.split('.')[0] )
        if os.path.exists(fngmm):
            ngmm_prev = np.loadtxt(fngmm)
            print()
            allnames_prev = [] 
            with open(os.path.join(base_fname, '%s_allNames.txt'%run_name.split('.')[0] )) as f:
                for line in f:
                    # Remove the newline character at the end of the line
                    line = line.strip()
                    # Append the line to the list
                    allnames_prev.append(line)
            print(allnames_prev)
                
            assert data.events==allnames_prev
            print('Previous fit found')
            n_comp_all = np.asarray( [ ( ngmm_prev[i].astype(int), FLAGS.n_gmm_max  ) for i in range(len(data.events))] )
           
            refit = True
            fout_suff_plot = '_refit_imax-%s'%FLAGS.n_gmm_max
            
        else:
            print('No previous gmm fit')
            n_comp_all = np.asarray( [ (FLAGS.n_gmm_min, FLAGS.n_gmm_max)   for i in range(len(data.events))] )
            refit = False
            fout_suff_plot =''
        
        #print('N. of components to fit:')
        #print( [ ( data.events[i], n_comp_all[i] ) for i in range(len(n_comp_all)) ]  )

        
        allsamples_ = [ data.m1z, data.m2z, data.dL, ]
        print('data m1z shape: %s'%str(data.m1z.shape))

        if FLAGS.fin_prior_dL != '':
            print("Also interpolating prior. loading from %s" % FLAGS.fin_prior_dL)
            prior_vals = np.load(FLAGS.fin_prior_dL)
        
            assert prior_vals.shape == data.dL.shape
        
            dL_bounds_full = None
            try:
                with h5py.File(fname_, "r") as f_in:
                    dL_bounds_full = np.asarray(f_in["properties"]["dL_bounds"])
                print("Loaded full dL bounds from input file.")
                print("dL_bounds_full shape: %s" % (dL_bounds_full.shape,))
                assert dL_bounds_full.shape == (data.dL.shape[0], 2)
            except Exception as e:
                print("Could not load full dL bounds from input file.")
                print(e)
                print("Falling back to bounds from the tabulated subsample.")
        
            dL_grid_, loginvprior_grid_, dL_min_, dL_max_ = build_dL_prior_interpolants(
                data.dL,
                prior_vals,
                prior_floor=FLAGS.prior_floor,
                dL_bounds_full=dL_bounds_full,
            )
        
            np.savez(
                os.path.join(base_fname, f"{run_name}_dL_prior_interp.npz"),
                dL_grid=dL_grid_,
                loginvprior_grid=loginvprior_grid_,
                dL_min=dL_min_,
                dL_max=dL_max_,
            )
        
            print("Prior interpolated.\n")
        
            plot_dL_prior_interpolants(
                dL_grid_,
                loginvprior_grid_,
                dL_min_,
                dL_max_,
                data.events,
                base_fname,
                dL_samples=data.dL,
                prior_vals=prior_vals,
                prior_floor=FLAGS.prior_floor,
            )

        if FLAGS.prior_only:
            print("Done.")

        else:
        
            if FLAGS.spins!='none':
                for i in range(len(data.spins)):
                    allsamples_.append(data.spins[i])
    
            if FLAGS.skymap:
                print('data ra shape: %s'%str(data.ra.shape))
                print('data dec shape: %s'%str(data.dec.shape))
                allsamples_.append(data.ra)
                allsamples_.append(data.dec)
            if FLAGS.inclination:
                allsamples_.append(data.iota,)
            
            for k in range(len(allsamples_)):
                print('allsamples_ %s comp shape: %s'%(k,allsamples_[k].shape ))
            allsamples_ = np.stack(allsamples_).transpose(1,2,0)
        
            gmm_means_, gmm_icovs_, gmm_covs_, gmm_cho_covs_, gmm_log_dets_, gmm_log_wts_, all_gmm_, allNgm_ = fit_gmm( allsamples_, 
                                                                                                                                                 data.Nsamples, 
                                                                                                                                                 allNames = data.events, 
                                                                                                                                                 fout_plot = base_fname, 
                                                                                                                                                 spins = FLAGS.spins, 
                                                                                                                                                 n_components = n_comp_all, 
                                                                                                                                                 dil_factor = FLAGS.dil_factor, 
                                                                                                                                                 refit = refit, 
                                                                                                                                                 fname_base = run_name.split('.')[0] ,
                                                                                                                                                skymap=FLAGS.skymap,
                                    inclination=FLAGS.inclination,
                                                                                                                       imin=FLAGS.imin,
                                                                                                                       imax=FLAGS.imax,
                                                                                                                                        fplots=FLAGS.fplots,
                                                                                                                                        fsummaries=FLAGS.fsummaries


                                                                                                                                                )
        
            means_, cho_covs_ = fit_cho(allsamples_, data.Nsamples, spins=FLAGS.spins, skymap=FLAGS.skymap, inclination=FLAGS.inclination, )
        
            np.savetxt( fngmm, allNgm_ ) 
        
            np.savetxt( os.path.join(base_fname, '%s_allNames.txt'%run_name.split('.')[0]), data.events, delimiter=" ", fmt="%s" ) 
            np.save( os.path.join(base_fname, '%s_cho-means.npy'%run_name.split('.')[0]), means_, )
            np.save( os.path.join(base_fname, '%s_cho-covs.npy'%run_name.split('.')[0]), cho_covs_, )
            
            
            np.save( os.path.join(base_fname, '%s_gmm_means.npy'%run_name.split('.')[0]), gmm_means_, )
            np.save( os.path.join(base_fname, '%s_gmm_icovs.npy'%run_name.split('.')[0]), gmm_icovs_, )
            np.save( os.path.join(base_fname, '%s_gmm_covs.npy'%run_name.split('.')[0]), gmm_covs_, )
            np.save( os.path.join(base_fname, '%s_gmm_cho_covs.npy'%run_name.split('.')[0]), gmm_cho_covs_, )
            np.save( os.path.join(base_fname, '%s_gmm_log_dets.npy'%run_name.split('.')[0]), gmm_log_dets_, )
            np.save( os.path.join(base_fname, '%s_gmm_log_wts.npy'%run_name.split('.')[0]), gmm_log_wts_, )
        
            if FLAGS.plot:
                plot_samples(allsamples_, 
                     all_gmm_,   
                     data.Nsamples,
                     #means_O1O2, 
                     #cho_covs_O1O2,
                     imax=len(data.events), 
                     names_plot=None, 
                     allNames=data.events,
                     dil_factor=FLAGS.dil_factor,
                         fout=base_fname,
                         fout_suff=fout_suff_plot,
                    spins=FLAGS.spins, 
                             skymap=FLAGS.skymap,
                     inclination=FLAGS.inclination,
                             fplots=FLAGS.fplots,
                    )
    




    print()
    print('*'*80)
    print('END. Results are saved in: %s'%FLAGS.fout)
    print('*'*80)
    print()

    
    myLog.close()



