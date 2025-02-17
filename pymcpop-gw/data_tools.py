#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import numpy as onp
import sys



def load_data_samples(fin, nmax=None):

    for i,fid in enumerate(fin):

        print("\nLoading data from %s"%fid)

        print('Loading posterior samples...')

        m1d_samples_ = onp.load( fid+'m1d_samples.npy' ) 
        m2d_samples_ = onp.load( fid+'m2d_samples.npy' )
        dL_samples_ = onp.load( fid+'dL_samples.npy' )

        chi1_samples_ = onp.load( fid+'chi1_samples.npy' )
        chi2_samples_ = onp.load( fid+'chi2_samples.npy' )
        cost1_samples_ = onp.load( fid+'cost1_samples.npy' )
        cost2_samples_ = onp.load( fid+'cost2_samples.npy' )
        
        where_compute_=~onp.isnan(m1d_samples_)
        allNsamples_ = where_compute_.sum(axis=-1) #onp.load( fid+'allnsamples.npy' )

        if i==0:

            m1d_samples = m1d_samples_
            m2d_samples = m2d_samples_
            dL_samples = dL_samples_
            chi1_samples = chi1_samples_
            chi2_samples = chi2_samples_
            cost1_samples = cost1_samples_
            cost2_samples = cost2_samples_
            allNsamples = allNsamples_
            where_compute = where_compute_
            

        else:

            m1d_samples = onp.concatenate([m1d_samples, m1d_samples_])
            m2d_samples = onp.concatenate([m2d_samples, m2d_samples_])
            dL_samples = onp.concatenate([dL_samples, dL_samples_])

            chi1_samples = onp.concatenate([chi1_samples, chi1_samples_])
            chi2_samples = onp.concatenate([chi2_samples, chi2_samples_])
            cost1_samples = onp.concatenate([cost1_samples, cost1_samples_])
            cost2_samples = onp.concatenate([cost2_samples, cost2_samples_])

            allNsamples = onp.concatenate([allNsamples, allNsamples_])
            where_compute = onp.concatenate([where_compute, where_compute_])
    
    print('Done.')
    print('allNsamples is')
    print(allNsamples)

    if nmax==-1:
        nmax=None
    else:
        allNsamples = onp.array([min(nmax, allNsamples[i]) for i in range(len(allNsamples)) ])
    
    res = {'m1d_samples': m1d_samples[:, :nmax], 
            'm2d_samples': m2d_samples[:, :nmax],
            'dL_samples': dL_samples[:, :nmax],
            'chi1_samples': chi1_samples[:, :nmax],
            'chi2_samples': chi2_samples[:, :nmax],
            'cost1_samples': cost1_samples[:, :nmax],
            'cost2_samples': cost2_samples[:, :nmax],
            'allNsamples': allNsamples,
            'where_compute': where_compute[:, :nmax]
           }
    print('allNsamples after cut is')
    print(res['allNsamples'])
    return res
        


def load_data_interp(fin):


    samples_means_dict =  {}
    samples_cho_covs_dict = {}

    gmm_log_wts_dict = {}
    gmm_means_dict = {}
    gmm_icovs_dict = {}
    gmm_covs_dict = {}
    gmm_cho_covs_dict = {}
    gmm_log_dets_dict = {}
    allNgm_dict = {}
    nevs_dict = {}

    nevs_all = 0
    Ngm_max = 0

    for i,fid in enumerate(fin):

        print("\nLoading data from %s"%fid)

        print('Loading sample means and covs...')

        try:
            samples_means_ =  onp.load( fid+'cho-means.npy' ) 
            samples_cho_covs_ = onp.load( fid+'cho-covs.npy' )
        except:
                try:
                    samples_means_ =  onp.load( fid+'cho_means.npy' ) 
                    samples_cho_covs_ = onp.load( fid+'cho_covs.npy' )
                except Exception as e:
                    print(e)
                    samples_means_ = onp.zeros( (1,1,1) )
                    samples_cho_covs_ = onp.zeros( (1,1,1,1) )
                
        if i==0:

            samples_means = samples_means_
            samples_cho_covs = samples_cho_covs_

        else:

            samples_means = onp.concatenate([samples_means, samples_means_])
            samples_cho_covs = onp.concatenate([samples_cho_covs, samples_cho_covs_])

    
        print('Done.')
        
        # load samples interpolants 
    
        print('Loading gmm parameters...')
    
        gmm_log_wts_dict[fid] = onp.load( fid+'gmm_log_wts.npy' ) 
        gmm_means_dict[fid] =  onp.load( fid+'gmm_means.npy' ) 
        gmm_icovs_dict[fid] =  onp.load( fid+'gmm_icovs.npy' ) 
        try:
            gmm_cho_covs_dict[fid] =  onp.load( fid+'gmm_cho_covs.npy' )
        except:
            print('Cholesky not available.')
            gmm_cho_covs = onp.zeros(gmm_icovs_dict[fid].shape)
            gmm_cho_covs_dict[fid] = onp.asarray(gmm_cho_covs)
        try:
            gmm_covs_dict[fid] =  onp.load( fid+'gmm_covs.npy' )
        except:
            print('Covariance not available.')
            gmm_covs = onp.zeros(gmm_icovs_dict[fid].shape)
            gmm_covs_dict[fid] = onp.asarray(gmm_covs)
            
        gmm_log_dets_dict[fid] =  onp.load( fid+'gmm_log_dets.npy' ) 
        allNgm_dict[fid] = onp.loadtxt( fid+'allNgm.txt' ).astype('int') 


        if i==0:
            allNgm = allNgm_dict[fid]
        else:
            allNgm = onp.concatenate([allNgm, allNgm_dict[fid]])

        try:
            if max(allNgm_dict[fid]) > Ngm_max:
                Ngm_max = max(allNgm_dict[fid])
        except TypeError:
            # array is 1d
            if allNgm_dict[fid] > Ngm_max:
                Ngm_max = allNgm_dict[fid]

        nevs_dict[fid] = len(gmm_log_wts_dict[fid])
        nevs_all += len(gmm_log_wts_dict[fid])

        nd = gmm_means_dict[fid][0].shape[1]

    nevs_arr = onp.asarray([ nevs_dict[k] for k in nevs_dict.keys() ])
    print('\nDone. Events:%s. Total: %s events. Max GMM number: %s. Number of dimensions: %s'%(nevs_arr,nevs_all, Ngm_max, nd))

    print("\nConcatenating data...")

    gmm_log_wts = onp.log( onp.zeros( (nevs_all, Ngm_max) ))
    gmm_means = onp.zeros( (nevs_all, Ngm_max, nd) )
    gmm_icovs = onp.zeros( (nevs_all, Ngm_max, nd, nd) )
    gmm_covs = onp.zeros( (nevs_all, Ngm_max, nd, nd) )
    gmm_cho_covs = onp.zeros( (nevs_all, Ngm_max, nd, nd) )
    gmm_log_dets = onp.log( onp.ones( (nevs_all, Ngm_max) ))

    iev = 0
    for k,fid in enumerate(fin):

        for i in range(nevs_dict[fid]):
            try:
                ngm_  =  max(allNgm_dict[fid]) 
            except TypeError:
                ngm_  =  allNgm_dict[fid]
    
            gmm_log_wts[iev, :ngm_] = gmm_log_wts_dict[fid][i]
            gmm_means[iev, :ngm_] = gmm_means_dict[fid][i]
            gmm_icovs[iev, :ngm_] = gmm_icovs_dict[fid][i]
            gmm_cho_covs[iev, :ngm_] = gmm_cho_covs_dict[fid][i]
            gmm_covs[iev, :ngm_] = gmm_covs_dict[fid][i]
            gmm_log_dets[iev, :ngm_] = gmm_log_dets_dict[fid][i]

            iev+=1
    

    return {'samples_means': samples_means, 
            'samples_cho_covs': samples_cho_covs,
            'gmm_log_wts': gmm_log_wts,
            'gmm_means': gmm_means,
            'gmm_icovs': gmm_icovs,
            'gmm_covs': gmm_covs,
            'gmm_cho_covs': gmm_cho_covs,
            'gmm_log_dets': gmm_log_dets,
            'allNgm': allNgm,
            'Nevents': nevs_arr
           }




def load_injections(fin_injections, allPercUse=None):


    Ttot = 0.
    ndetInj_max = 0

    alldLInj_dict = {}
    allm1truesInj_dict = {}
    allm2truesInj_dict = {}
    logPdrawInj_dict = {}
    ngenInj_dict = {}
    Tobs_dict = {}
    ndetInj_dict = {}
    
    allchiefftruesInj_dict = {}
    allchiptruesInj_dict = {}

    allspin1xInj_dict = {}
    allspin2xInj_dict = {}

    allspin1yInj_dict = {}
    allspin2yInj_dict = {}

    allspin1zInj_dict = {}
    allspin2zInj_dict = {}

    allChi1Inj_dict = {}
    allChi2Inj_dict=  {}

    allCost1Inj_dict=  {}
    allCost2Inj_dict = {}
    
    for i,fiinj in enumerate(fin_injections):

        print("\nLoading injections from %s"%fiinj)

        alldLInj_dict[fiinj] = onp.load( fiinj+'dL.npy') 
        allm1truesInj_dict[fiinj] = onp.load( fiinj+'m1d.npy')
        allm2truesInj_dict[fiinj] = onp.load( fiinj+'m2d.npy')
        logPdrawInj_dict[fiinj] = onp.load(fiinj+'log_p_draw.npy')
        
        try:
            allchiefftruesInj_dict[fiinj] =  onp.load(fiinj+'chieff.npy')
            allchiptruesInj_dict[fiinj] =  onp.load(fiinj+'chip.npy')
            spins=True
        except Exception as e:
            print(e)
            print("Chi_eff, chi_p not in spin injections.")
            spins_chieffchip=False

        try:
            allspin1xInj_dict[fiinj] =  onp.load(fiinj+'spin1x.npy')
            allspin2xInj_dict[fiinj] =  onp.load(fiinj+'spin2x.npy')

            allspin1yInj_dict[fiinj] =  onp.load(fiinj+'spin1y.npy')
            allspin2yInj_dict[fiinj] =  onp.load(fiinj+'spin2y.npy')

            allspin1zInj_dict[fiinj] =  onp.load(fiinj+'spin1z.npy')
            allspin2zInj_dict[fiinj] =  onp.load(fiinj+'spin2z.npy')

            spins_123xyz=True
        except Exception as e:
            print(e)
            print("Spin x,y z not in spin injections.")
            spins_123xyz=False

        try:
            allChi1Inj_dict[fiinj] =  onp.load(fiinj+'chi1.npy')
            allChi2Inj_dict[fiinj] =  onp.load(fiinj+'chi2.npy')

            allCost1Inj_dict[fiinj] =  onp.load(fiinj+'ct1.npy')
            allCost2Inj_dict[fiinj] =  onp.load(fiinj+'ct2.npy')

            spins_default=True
            print("Using chi1, chi2, cost1, cost2 for spin injections.")
        
        except Exception as e:
            print(e)
            print("Chi1, chi2, cost1, cost2 not in spin injections.")
            spins_default=False
    
        if (spins_chieffchip or spins_123xyz or spins_default):
            spins=True
        else:
            spins=False
            print('No spins in injections.')
        
        ngenInj_dict[fiinj] = onp.load( fiinj+'Ngen.npy').astype(int)

        if allPercUse is not None:
            percUse = allPercUse[i]
        else:
            percUse = 1

        if percUse <1 :

            Ngen_new = int(percUse*ngenInj_dict[fiinj])

            Ndet = len(alldLInj_dict[fiinj])#.astype(int)

            Nuse = int(Ndet*percUse)

            print('Downsampling injections to %s detections, which corresponds to using %s of those available. '%(Nuse, percUse ))
            print('Original number of detected injections: %s'%Ndet)
            
            idxs = onp.random.permutation(Ndet)

            alldLInj_dict[fiinj] = alldLInj_dict[fiinj][idxs][:Nuse]
            allm1truesInj_dict[fiinj] = allm1truesInj_dict[fiinj][idxs][:Nuse]
            allm2truesInj_dict[fiinj] = allm2truesInj_dict[fiinj][idxs][:Nuse]
            logPdrawInj_dict[fiinj] = logPdrawInj_dict[fiinj][idxs][:Nuse]

            ngenInj_dict[fiinj] = Ngen_new

            if spins_chieffchip:
                allchiefftruesInj_dict[fiinj] = allchiefftruesInj_dict[fiinj][idxs][:Nuse]
                allchiptruesInj_dict[fiinj] = allchiptruesInj_dict[fiinj][idxs][:Nuse]
            if spins_123xyz:
                allspin1xInj_dict[fiinj] =  allspin1xInj_dict[fiinj][idxs][:Nuse]
                allspin2xInj_dict[fiinj] =  allspin2xInj_dict[fiinj][idxs][:Nuse]

                allspin1yInj_dict[fiinj] =  allspin1yInj_dict[fiinj][idxs][:Nuse]
                allspin2yInj_dict[fiinj] =  allspin2yInj_dict[fiinj][idxs][:Nuse]
    
                allspin1zInj_dict[fiinj] =  allspin1zInj_dict[fiinj][idxs][:Nuse]
                allspin2zInj_dict[fiinj] = allspin2zInj_dict[fiinj][idxs][:Nuse]
            if spins_default:
                allChi1Inj_dict[fiinj] =  allChi1Inj_dict[fiinj][idxs][:Nuse]
                allChi2Inj_dict[fiinj] =  allChi2Inj_dict[fiinj][idxs][:Nuse]
    
                allCost1Inj_dict[fiinj] =  allCost1Inj_dict[fiinj][idxs][:Nuse]
                allCost2Inj_dict[fiinj] =  allCost2Inj_dict[fiinj][idxs][:Nuse]
            
        elif percUse >1:
            raise ValueError()
        
        
        Tobs_dict[fiinj] = onp.loadtxt( fiinj+'Tobs.txt')
        
        Ttot += Tobs_dict[fiinj]

        ndetInj_dict[fiinj] = len(alldLInj_dict[fiinj])#.astype(int)
        if ndetInj_dict[fiinj]>ndetInj_max:
            ndetInj_max = ndetInj_dict[fiinj]
        
        print("N. of det injections: %s"%ndetInj_dict[fiinj])
        
        print('Done.')

    
    print("\nConcatenating injections...")

    ndatasets = len(fin_injections)
    
    alldLInj = onp.zeros( (ndatasets,  ndetInj_max))
    allm1truesInj = onp.zeros( (ndatasets,  ndetInj_max))
    allm2truesInj = onp.zeros( (ndatasets,  ndetInj_max))
    logPdrawInj = onp.full( (ndatasets,  ndetInj_max), onp.inf)
    ngenInj = onp.zeros( ndatasets)
    TobsInj = onp.zeros( ndatasets)
    ndetInj = onp.zeros( ndatasets)
    if spins_chieffchip:
        allchieffInj = onp.zeros( (ndatasets,  ndetInj_max))
        allchipInj = onp.zeros( (ndatasets,  ndetInj_max))
    if spins_123xyz:

        allspin1xInj = onp.zeros( (ndatasets,  ndetInj_max))
        allspin2xInj = onp.zeros( (ndatasets,  ndetInj_max))

        allspin1yInj = onp.zeros( (ndatasets,  ndetInj_max))
        allspin2yInj = onp.zeros( (ndatasets,  ndetInj_max))

        allspin1zInj = onp.zeros( (ndatasets,  ndetInj_max))
        allspin2zInj = onp.zeros( (ndatasets,  ndetInj_max))

    if spins_default:
        allChi1Inj = onp.zeros( (ndatasets,  ndetInj_max))
        allChi2Inj = onp.zeros( (ndatasets,  ndetInj_max))

        allCost1Inj = onp.zeros( (ndatasets,  ndetInj_max))
        allCost2Inj = onp.zeros( (ndatasets,  ndetInj_max))
        

        
    
    for i,fiinj in enumerate(fin_injections):

        nmax = ndetInj_dict[fiinj]
        
        alldLInj[i, :nmax] = alldLInj_dict[fiinj]
        allm1truesInj[i, :nmax] = allm1truesInj_dict[fiinj]
        allm2truesInj[i, :nmax] = allm2truesInj_dict[fiinj]
        logPdrawInj[i, :nmax] = logPdrawInj_dict[fiinj]
        ngenInj[i] = ngenInj_dict[fiinj]
        TobsInj[i] = Tobs_dict[fiinj]
        ndetInj[i] = ndetInj_dict[fiinj]
        if spins_chieffchip:
            allchieffInj[i, :nmax] = allchiefftruesInj_dict[fiinj]
            allchipInj[i:nmax] = allchiptruesInj_dict[fiinj]
        if spins_123xyz:
            allspin1xInj[i, :nmax] = allspin1xInj_dict[fiinj]
            allspin2xInj[i, :nmax] = allspin2xInj_dict[fiinj]
    
            allspin1yInj[i, :nmax] = allspin1yInj_dict[fiinj]
            allspin2yInj[i, :nmax] = allspin2yInj_dict[fiinj]
    
            allspin1zInj[i, :nmax] = allspin1zInj_dict[fiinj]
            allspin2zInj[i, :nmax] = allspin2zInj_dict[fiinj]
        if spins_default:
            allChi1Inj[i, :nmax] =  allChi1Inj_dict[fiinj]
            allChi2Inj[i, :nmax] = allChi2Inj_dict[fiinj]
    
            allCost1Inj[i, :nmax] = allCost1Inj_dict[fiinj]
            allCost2Inj[i, :nmax]=  allCost2Inj_dict[fiinj]


            

    inj = {}
    inj['dL'] = alldLInj
    inj['m1d'] = allm1truesInj
    inj['m2d'] = allm2truesInj
    inj['log_wt'] = logPdrawInj
    inj['Ngen'] = ngenInj
    inj['Tobs'] = TobsInj
    inj['Ndet'] = ndetInj.astype(int)
    if spins_chieffchip:
        inj['chieff'] = allchieffInj
        inj['chip'] = allchipInj
    if spins_123xyz:
        inj['spin1x'] = allspin1xInj
        inj['spin2x'] = allspin2xInj

        inj['spin1y'] = allspin1yInj
        inj['spin2y'] = allspin2yInj

        inj['spin1z'] = allspin1zInj
        inj['spin2z'] = allspin2zInj
    if spins_default:
        inj['chi1'] = allChi1Inj
        inj['chi2'] = allChi2Inj

        inj['cost1'] = allCost1Inj
        inj['cost2'] = allCost2Inj
    
    return inj




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
