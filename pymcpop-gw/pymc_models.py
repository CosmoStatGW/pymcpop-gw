#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import pytensor_tools as atools
import pytensor_utils_old as putils

from pytensor_utils import atinterp

from pytensor_tools_new import PopAndSelJAXOp
import cosmology as cosmo
from backends import NPBackend, JAXBackend
import constants

import pymc_models_or as pmmor
import mass_models as mm


import pytensor.tensor as at
import pytensor
import pymc as pm

#from pytensor.gradient import disconnected_grad as stop_grad
from pytensor.compile.mode import get_default_mode
from pymc.distributions import transforms as tr
#from pymc.pytensorf import collect_default_updates
from pytensor import config
import h5py

PLPeakO3params = {'H0': 67.66, 'Om':0.31, 'w0':-1, 'Xi0': 1, 'nXi0':0}


import numpy as np


eps   = 1e-30
tinyL = 1e-300
NEG_BIG = -np.inf



#####################################################
#####################################################

# MODEL

#####################################################
#####################################################


def make_model(  priors,
                 GWData,
                 InjData,
                 ivals=None,
                 eps_init = 0.01,
                 sampling_GW = 'gmm',
                 rate_model = 'MD',
                 mass_model = 'PLP',
                 smoothing='LVK',
                 simplex_repair=False,
                 interp_mass = 0,
                 interp_z = 0,
                 has_m2_break = False,
                 norm_gauss = 'uplow',
                 spin_model = 'none',
                 spin_inj = 'none',
                 marginal_R0 = True,
                 dLprior = ['none'],
                 fix_inj_len = False,
                 chunk_inj = -1,
                 chunk_reduce = False,
                 use_float32 = False,
                 use_float32_bias=False,
                 sel_method='Tobs',
                 N_DP_comp_max = 100,
                 alpha_tail = 0.2,
                 alpha_small = 0.01,
                 L_small_1 = 0.5,
                 L_small_2 = 0.5,
                 L_small_3 = 0.1,
                 s_local = 0.5,
                 find_m_bounds = False,
                 q_mbound = 0.05,
                 alpha_inv_params = (1, 1),
                 fix_H0 = True,
                fix_Om = True,
               fix_w0 = True,
                 fix_Xi0n = True,
                 z_pivot=0.5,
               pade=False,
               zres=150,
                z_grid_mode='cheb',
                zmin_a=1e-05, zmin_b=1e-03, zmid_b=3.0, zmax_c=10.0, hi_boost=0.20,
                 find_z_bounds = False,
               params_fix=None,
                 Neff_min=4,
                Neff_min_lik=1,
               log_lik_var_min=1,
                 use_sel_spin=True,
                 pop_only = False,
               N_successes_l=None,
               Nsamplesuse = -1,
               include_sel_uncertainty=False,
               sel_smoothing='poly',
               alpha_beta_prior='poly',
               dil_factor=1,
               use_log_alpha_beta=False ,
               allTobs=None,
                 use_updates=True,
                 inj_loop='vec',
                 save_thetas=False,
                 interp_inj=True,
                 param='vanilla',
                 DP_prior='SB',
                 sigma_softmax=0.75,
                 gamma_DP_params = (4, 0.8),
                 is_observed = False,
                 sample_from_pop = False,
                 mmin_inj=-1,
                 is_compressed_inj=False,
                 debug_sel_batch=False,
                 reparam_z = True,
                 reparam_mass = False,
                 priors_for_mmin='',
                 normalize_PE_prior=True,
                 linear_mass=False,
                 linear_z=False
                ):



    ################################################
    # Read in data and set dimensions
    ################################################

    
    ## GW data
    if not pop_only:
        # gw data are interpolants of single-event posteriors
        if sampling_GW=='gauss':
            # we sample single-event parameters from broad gaussian approximations of the posteriors
            mus_s, cho_s, log_wts_l, mus_l, icovs_l, log_dets_l, cho_covs_l, Tobs, Nevs = GWData
            import numpy as np
            wts_l = np.exp(log_wts_l)
            
        elif 'gmm' in sampling_GW or sampling_GW=='gumbel':
            # we sample single-event parameters from the actual single-event posteriors
            wts_l, mus_l, cho_covs_l, Tobs, Nevs = GWData
        else:
            raise ValueError('sampling_GW can be gmm, gmm_cat, gumbel,  gauss ')
            
        

    else:
        # gw data are single-event posterior samples
        # shape of each has to be n_events, n_samples
        m1det, m2det, d, spin_samples, Tobs, allNsamples, where_compute = GWData            

        if Nsamplesuse !=-1 :
            if Nsamplesuse>allNsamples:
                raise ValueError("Must use less samples than those available.")
            print("allNsamples availabe is %s, but %s will be used"%(allNsamples, Nsamplesuse))
            allNsamples =  Nsamplesuse   
            allNsamples_np = allNsamples #allNsamples.eval()
        
        if (spin_model=='default') or (spin_model=='default_gauss'):
           chi1, chi2, cost1, cost2 = spin_samples
        else:
            raise NotImplementedError()

    ## Injections data
    if spin_inj == 'none':
        dLinj, m1inj, m2inj, lpdinj, Ndraw, Ndet, lp_incl_inj = InjData
    elif spin_inj == 'chieffchip':
        dLinj, m1inj, m2inj, chiefffInj, chipInj, lpdinj, Ndraw, Ndet, lp_incl_inj = InjData
    elif (spin_inj == 'chi12xyz' or spin_inj == 'default'):
        if (spin_model=='default') or (spin_model=='default_gauss'):
            dLinj, m1inj, m2inj, chi1Inj, chi2Inj, cost1Inj, cost2Inj, lpdinj, Ndraw, Ndet, lp_incl_inj = InjData
        elif spin_model == 'none':
            dLinj, m1inj, m2inj, lpdinj, Ndraw, Ndet, lp_incl_inj = InjData
            
    ndata = m1inj.shape[0] # number of observing runs to combine
    ndata_np = ndata #ndata.eval()
    ninj = m1inj.shape[1] # max number of injections
    ninj_np = ninj #ninj.eval()
    Ndraw = float(Ndraw)

    if not use_sel_spin and spin_model!='none':
        raise ValueError("You are using spin_model=%s but not use_sel_spin. "%spin_model)

    if ndata_np==1:
        
        if use_sel_spin:
            spin_model_name = spin_model
            
            if spin_model == 'chieffchip' or spin_model == 'chieffchip_uc' :
                spinsInj = [ chiefffInj[0], chipInj[0] ]
                
            elif (spin_model == 'default') or (spin_model == 'default_gauss'):
                spinsInj = [ chi1Inj[0], chi2Inj[0], cost1Inj[0], cost2Inj[0] ]
                
            else:
                raise ValueError("use_sel_spin is True, but no valid spin model name was given. Use use_sel_spin=False or provide valid spin model.")
                spinsInj = []
    
        else:
            print("Spin distribution will not be used in the sel effect")
            spinsInj = []
            spin_model_name = 'none'

        if spin_model_name in ("default", "default_gauss"):
            spinsInj = at.stack(spinsInj, axis=1)   # from [chi1,chi2,cost1,cost2] -> (ninj,4)
        elif spin_model_name in ("chieffchip", "chieffchip_uc"):
            spinsInj = at.stack(spinsInj, axis=1)   # (ninj,2)
        else:
            spinsInj = at.zeros((m1inj[0].shape[0], 0), dtype="float64")
    
    Ndet_np = Ndet #Ndet.eval()
    N_DP_comp_max_np = N_DP_comp_max #N_DP_comp_max.eval()
    Nevs_np = np.atleast_1d(Nevs) #Nevs.eval()

    Tobs_np = Tobs #Tobs.eval()

        
    if not pop_only:
        N = mus_l.shape[0]#.astype(int_dtype) # number of events in total
        N_np = N#.astype(int_dtype) #N.eval()
        ngmm = mus_l.shape[1]#.astype(int_dtype)
        ngmm_np = ngmm#.astype(int_dtype) #ngmm.eval()
        nd = mus_l.shape[2]#.astype(int_dtype)
        nd_np = nd#.astype(int_dtype) #nd.eval()
        print('N:%s, max ngmm: %s, nd: %s '%(N_np, ngmm_np, nd_np))
        print('N evs is %s'%Nevs_np)
        print('Tobs is %s'%Tobs_np)
    else:
        N = m1det.shape[0] # number of events in total
        N_np = N #N.eval()
        Nsamples = m1det.shape[1]
        Nsamples_np = Nsamples #Nsamples.eval()
        print("N samples max will be ")
        print(Nsamples_np)
        print('N:%s, n samples: %s '%(N_np, allNsamples_np))

    logN = np.log(N)


    
    event_index = np.arange(N_np, dtype=int)
    Ttot = np.sum(Tobs)

    
    print('Injections: :%s, '%(ninj_np))

    print('ninj: :%s, %s datasets,'%(Ndet_np, ndata_np))

    coords = {'event_index': event_index}

    

    if mass_model in ('DP', 'DPUC'):
        coords['component'] = np.arange(N_DP_comp_max_np, dtype=int)
        
        if rate_model in ('DPUC','DPUC-vol'):
            ndim_GMM = 3
        else:
            ndim_GMM = 2

        print('GMM dimension is %s'%ndim_GMM)
        
        coords['GMMdimension'] = np.arange(ndim_GMM, dtype=int)
        coords['GMMdimension_1'] = np.arange(ndim_GMM, dtype=int)
        coords['GMMdimension_2'] = np.arange(ndim_GMM, dtype=int)
        p = ndim_GMM*(ndim_GMM+1)//2  # packed length = 3 for n=2
        
        coords["packed_cholesky"] = np.arange(p)

    if pop_only:
        coords['nsamples'] = np.arange( Nsamples_np, dtype=int )
    else:
         coords['GWdimension'] = np.arange(nd_np, dtype=int)


    if params_fix is None:
        print('No values for parameters to fix passed. Default values will be used. If fixing parameters, check that the values are consistent. Values of fixed parameters:')
        print(PLPeakO3params)
        params_fix=PLPeakO3params


 


    
    if ( find_z_bounds or (mass_model in ('DPUC', 'DP') and find_m_bounds) or mmin_inj!=-1 ):

    
        rng = np.random.default_rng()
        
    
        # --- Compile once: z_from_dL and midpoint derivative ---
        z_sym      = at.dvector('z_nodes')    # if you need it
        d_sym      = at.dvector('dL_nodes')
        H0_sym     = at.dscalar('H0')
        Om_sym     = at.dscalar('Om')
        w0_sym     = at.dscalar('w0')
        Xi0_sym     = at.dscalar('Xi0')
        n_sym     = at.dscalar('nXi0')

        
        z_from_dL_sym = atools.z_from_dL_at(d_sym, H0_sym, Om_sym, w0_sym, Xi0_sym, n_sym, interp=pade, param=param)
        z_from_dL_fn = pytensor.function([d_sym, H0_sym, Om_sym, w0_sym, Xi0_sym, n_sym], z_from_dL_sym)

        
        if fix_H0:
            priors['H0'] = ( params_fix['H0'], params_fix['H0'])
        if fix_Om:
            priors['Om'] = ( params_fix['Om'], params_fix['Om'])
        if fix_w0:
            priors['w0'] = ( -1, -1)
        if fix_Xi0n:
            priors['Xi0'] = ( 1, 1)
            priors['nXi0'] = ( 0, 0)



        if find_z_bounds:
            print("\nFinding optimal points for redshift interpolation...")
            print("min, max redshift search grid: %s, %s"%(atools.zGridGlobals_at.eval().min(), atools.zGridGlobals_at.eval().max()))
        
            min_z, max_z, z_min_data, z_max_data = putils.find_zgrid_bounds(wts_l, mus_l, cho_covs_l,
                                          priors['H0'], priors['Om'], priors['w0'], priors['Xi0'], priors['nXi0'], 
                                          int(N), int(nd),
                                        dLinj,
                                        z_from_dL_fn,
                                          sampling_GW,
                                          trials=1000, 
                                         )
    
            
            
            zmin_b = min(zmin_b, max(min_z, z_min_data))
    
            zmin_a = min( zmin_a, min(min_z, z_min_data))
            
            zmid_b = min( zmid_b, z_max_data )
            zmax_c = max(zmax_c, max(z_max_data, max_z))*(1+0.1)
    
            print("Redshift values found, overwriting default:")
            print("zmin_a=%s, zmin_b=%s, zmid_b=%s, zmax_c=%s"%(zmin_a, zmin_b, zmid_b, zmax_c))


        if (mass_model in ('DPUC', 'DP') and find_m_bounds):

            print("\nFinding prior range for DP-GMM.")
       
            scales = putils.find_mass_redshift_bounds(wts_l, mus_l, cho_covs_l,
                                          priors['H0'], priors['Om'], priors['w0'], priors['Xi0'], priors['nXi0'], 
                                          int(N), int(nd),
                                        dLinj,
                            m1inj,
                            m2inj,
                              z_from_dL_fn,
                              sampling_GW,
                              trials=1000, 
                            is_observed = False, #is_observed
                          #rng=onp.random.default_rng(123)
                            q_mbound=q_mbound
                             )
    
            lowmu1 = scales['lMc_min_data']#.astype(X)
            upmu1 = scales['lMc_max_data']#.astype(X)
    
            lowmu2 = scales['lq_min_data']#.astype(X)
            upmu2 = scales['lq_max_data']#.astype(X)

            lowmu3 = scales['logz_min_data']#.astype(X)
            upmu3 = scales['logz_max_data']#.astype(X)


            lowmu1_inj = scales['lMc_min_inj']#.astype(X)
            upmu1_inj = scales['lMc_max_inj']#.astype(X)
    
            lowmu2_inj = scales['lq_min_inj']#.astype(X)
            upmu2_inj = scales['lq_max_inj']#.astype(X)

            lowmu3_inj = scales['logz_min_inj']#.astype(X)
            upmu3_inj = scales['logz_max_inj']#.astype(X)
    
            L_small_1_data = scales['lMc_diff']#.astype(X)
            L_small_2_data = scales['lq_diff']#.astype(X)

            L_small_m1 = scales['m1_diff']#.astype(X)
            L_small_m2 = scales['m2_diff']#.astype(X)
    
            L_small_3_data = scales['logz_diff']#.astype(X)

            print("Mass/redshift DP-GMM prior values found, overwriting default:")
            print("lowmu1=%s, upmu1=%s, lowmu2=%s, upmu2=%s"%(lowmu1, upmu1, lowmu2, upmu2))
            print("L_small_1_data=%s, L_small_2_data=%s, L_small_3_data=%s"%(L_small_1_data, L_small_2_data,L_small_3_data ))
            print("L_small_m1=%s, L_small_m2=%s"%(L_small_m1, L_small_m2, ))


            

            if L_small_1>0 and L_small_2>0:
                print("Finding min spacing based on requested spacing in m1-m2...")

                m1_min = min(scales['m1_min_inj'], scales['m1_min_data'] )
                m2_min = min(scales['m2_min_inj'], scales['m2_min_data']  )
                m1_max = min(scales['m1_max_inj'], scales['m1_max_data'] )
                m2_max = min(scales['m2_max_inj'], scales['m2_max_data']  )

                dmin = putils.min_sep_logMc_logitq(
                    m1_min=m1_min, m1_max=m1_max,
                    m2_min=m2_min, m2_max=m2_max,
                    dm1=L_small_1, dm2=L_small_2
                )


                L_small_1 = max(L_small_1_data, dmin[0])#.astype(X) 
                L_small_2 = max(L_small_2_data, dmin[1])#.astype(X)
            else:
                L_small_1 = L_small_1_data
                L_small_2 = L_small_2_data

                if L_small_3>0:

                    L_small_3 = max(L_small_3_data, np.log(1+L_small_3/max(z_max_data, max_z)) )#.astype(X)  
                else:
                    L_small_3 = L_small_3_data
            print(" Final L_small_1=%s, L_small_2a=%s, L_small_3=%s"%(L_small_1, L_small_2,L_small_3 ))

            
        if mmin_inj!=-1:
            if 'BNS' in mass_model:
                raise ValueError()
            print("Pre-filtering injections to exclude those with mass<%s solar masses."%mmin_inj)

            if priors_for_mmin=='':
                priors_for_mmin = priors
                print("Computing source-frame mass across prior range equal to your prior")
            else:
                print("Comupting source-frame mass across prior range with input file %s"%priors_for_mmin)
            dL_min, dL_max = dLinj[0].min(), dLinj[0].max()
            
            # 1) build envelope once 
            dL_grid, zmax_grid = putils.build_zmax_envelope_from_corners(
                z_from_dL_fn, dL_min, dL_max, priors_for_mmin, n_grid=4096
            )
            
            # 2) apply safe filter once
            keep = putils.safe_prefilter_injections_detector_frame(
                m1inj[0], m2inj[0], dLinj[0],
                dL_grid, zmax_grid,
                mmin_src=mmin_inj,
            )
            ninj_or = m1inj.shape[1]
            ninj_new = keep.sum()
            print("Will keep %s injections out of %s"%(ninj_new, ninj_or))

            dLinj, m1inj, m2inj, lpdinj = [ d_[keep] for d_ in dLinj ], [ m_[keep] for m_ in m1inj], [ m_[keep] for m_ in m2inj], [l_[keep] for l_ in lpdinj ]
            
            #spinsInj = [sI[keep] for sI in spinsInj ]
            if isinstance(spinsInj, (list, tuple)):
                spinsInj = [sI[keep] for sI in spinsInj]
            else:
                # PyTensor variable or numpy array
                spinsInj = spinsInj[keep]
            
            
            Ndet[0] = ninj_new

            if is_compressed_inj:
                lp_incl_inj = [ l_[keep] for l_ in lp_incl_inj]
            else:
                lp_incl_inj = lp_incl_inj

            print("lp_incl_inj is ")
            print(lp_incl_inj)
            
    
    #####################################################################################################
    if not linear_z:
        zgrid_np = atools.make_z_grid_fixed(
        total=zres,
        zmin_a=zmin_a, zmin_b=zmin_b, zmid_b=zmid_b, zmax_c=zmax_c,
        mid_boost=8.0, edge_frac=0.08, end_boost=0.5
    )
    else:
        zgrid_np = np.linspace(zmin_a, zmax_c, zres)

 
    zgrid_ = at.constant(zgrid_np)
    

    
    print("z grid for interpolation built. Resolution: %s"%zres)
    print("z min: %s , z max: %s"%(zmin_a, zmax_c))
    print("is z grid constant check:")
    print(isinstance(zgrid_, at.TensorConstant))

    #####################################################################################################

    if z_pivot!=0:

        H0_min, H0_max = priors['H0']
        Om_min, Om_max = priors['Om']
        w0_min, w0_max = priors['w0']

        # Evaluate E(z_pivot) at the corners of the (Om, w0) prior box
        Ez_corners = [
            atools.Efun_num(z_pivot, Om_min, w0_min), #.astype(X),
            atools.Efun_num(z_pivot, Om_min, w0_max), #.astype(X),
            atools.Efun_num(z_pivot, Om_max, w0_min), #.astype(X),
            atools.Efun_num(z_pivot, Om_max, w0_max), #.astype(X),
        ]
        Ez_min = min(Ez_corners)#.astype(X)
        Ez_max = max(Ez_corners)#.astype(X)


    #####################################################################################################


    vol_in_prior = any('UniformSourceFrame' in s or 'UniformComovingVolume' in s for s in dLprior)
    
    all_dLsq_prior = all(s == 'dLsq' for s in dLprior)
    all_no_dL_prior = all(s == 'none' for s in dLprior)

    edges = [0]
    for n in Nevs_np:
        edges.append(edges[-1] + int(n))

    if vol_in_prior:

        zgrid_dLp = at.constant( atools.make_z_grid(total=zres, zmin_a=zmin_a, zmin_b=zmin_b, zmid_b=zmid_b, zmax_c=zmax_c, hi_boost=hi_boost) ) 

        dc_grid_Planck15 = atools.dcfun_at(zgrid_dLp, 67.74, 0.3075, -1., interp=False)#.astype(work_dtype)
        dL_grid_Planck15 = atools.dLfun_at(zgrid_dLp, 67.74, 0.3075, -1., 1., 0., interp=False, dc=dc_grid_Planck15, param='vanilla') #.astype(work_dtype)

        if normalize_PE_prior:
            z_bounds = atools.z_from_dL_at( np.asarray([0.1/1000, 40000/1000]), 67.74, 0.3075, -1., 1., 0. , interp=False)
            z_min_PE_prior, z_max_PE_prior = float(z_bounds[0].eval() ), float(z_bounds[1].eval())
            print(
    f"normalization of uniform-in-com-vol prior between dL=[{0.1/1000}, {40000/1000}] Gpc, "
    f"i.e. z=[{z_min_PE_prior}, {z_max_PE_prior}]"
)
            
            log_norm_PE_prior = cosmo.compute_log_norm_UniformSourceFrame(NPBackend(), z_min_PE_prior, z_max_PE_prior, 67.74, 0.3075, -1,  constants._x01_np, constants._w01_np)
            
        

    ################################################
    # Build model
    ################################################

            
    if 'gmm' in sampling_GW:
        # we sample single-event parameters from the actual single-event posteriors
        # need tensor variables to correctly slice inside model
        wts_l, mus_l, cho_covs_l = at.constant(wts_l), at.constant(mus_l), at.constant(cho_covs_l)



    
    with pm.Model(coords=coords) as model:




        
        ################################################
        # Cosmological parameters
        ################################################

        if fix_Om:
            Om_ = params_fix['Om']
        else:
            Om_ = pm.Uniform('Om', lower=priors['Om'][0], upper=priors['Om'][1], initval=ivals.get('Om')) 

        if fix_w0:
            w0_ = at.as_tensor_variable(-1.)
        else:
            if pade:
                raise NotImplementedError("Pade appproximation with varying w0 not implemented yet. Use pade=False")
            w0_ =  pm.Uniform('w0', lower=priors['w0'][0], upper=priors['w0'][1], initval=ivals.get('w0'))

            
        
        if fix_H0:
            H0_ =  params_fix['H0']
        else:
            if z_pivot!=0:
                print("Sampling in H(z=%s)"%z_pivot)
                # Define a broad prior for Hp = H(z_pivot)
                # Just choose constants that safely cover the H0 prior range once divided by Ez_pivot.
                Hp_min = H0_min * Ez_min
                Hp_max = H0_max * Ez_max

                H0_init = ivals.get('H0', at.as_tensor_variable(67.7) )
                
                Hp_ = pm.Uniform(
                    'Hp',
                    lower=Hp_min,
                    upper=Hp_max,
                    # optional: if you have numeric Om_init, w0_init, you can precompute a good initval
                    initval=H0_init * atools.Efun_at(z_pivot, ivals.get('Om', 0.3), ivals.get('w0', -1)).eval()
                )

                # E(z_pivot; Om, w0) using your helper
                Ez_pivot = atools.Efun_at(z_pivot, Om_, w0_)
                
                # Physical H0 as a deterministic transform of (Hp, Om, w0)
                H0_ = pm.Deterministic('H0', Hp_ / Ez_pivot)
                
                # Jacobian term: prior flat in H0, but sampling in Hp
                _ = pm.Potential('J_H0_from_Hp', -at.log(Ez_pivot))

            else:
                
                H0_ =  pm.Uniform('H0', lower=priors['H0'][0], upper=priors['H0'][1], initval=ivals.get('H0'))
        
        
        
        if fix_Xi0n:
            Xi0_ =  at.as_tensor_variable(1.) #, dtype=work_dtype)
            nXi0_ = at.as_tensor_variable(0.) #, dtype=work_dtype)
        else:
            Xi0_ =  pm.Uniform('Xi0', lower=priors['Xi0'][0], upper=priors['Xi0'][1], initval=ivals.get('Xi0'))
            nXi0_ = pm.Uniform('nXi0', lower=priors['nXi0'][0], upper=priors['nXi0'][1], initval=ivals.get('nXi0')) 

            print("For Xi0-n, we use the %s parameterization"%param)


        Lambda_ = [H0_, Om_, w0_, Xi0_, nXi0_]

        ################################################
        # Redshift evolution of merger rate
        ################################################
        
        if rate_model=='MD':
            
            print('Modeling evolution of merger rate with redshift with Madau-Dickinson profile')
            gamma_ = pm.Uniform('gamma', lower=priors['gamma'][0], upper=priors['gamma'][1], initval=ivals.get('gamma'))    
            kappa_ = pm.Uniform('kappa', lower=priors['kappa'][0], upper=priors['kappa'][1], initval=ivals.get('kappa'))
            zp_ = pm.Uniform('zp', lower=priors['zp'][0], upper=priors['zp'][1], initval=ivals.get('zp'))

       
            Lambda_ += [gamma_, kappa_, zp_]

        elif rate_model=='PL':
            print('Modeling evolution of merger rate with a power law')
            gamma_ = pm.Uniform('gamma', lower=priors['gamma'][0], upper=priors['gamma'][1], initval=ivals.get('gamma'))

            Lambda_ += [gamma_]

        elif rate_model in ('DPUC', 'DPUC-vol'):

            assert mass_model in ('DP', 'DPUC')
            print('Modeling evolution of merger rate with a DP-GMM together with mass')

        ################################################
        # Spin
        ################################################

        if spin_model == 'chieffchip':
            print('Modeling spin distribution with a gaussian in chieff-chip')
            muEff_ = pm.Uniform('muEff', lower=priors['muEff'][0], upper=priors['muEff'][1])
            sigEff_ = pm.Uniform('sigEff', lower=priors['sigEff'][0], upper=priors['sigEff'][1])
            muP_ = pm.Uniform('muP', lower=priors['muP'][0], upper=priors['muP'][1])
            sigP_ = pm.Uniform('sigP', lower=priors['sigP'][0], upper=priors['sigP'][1])
            rho_ = pm.Uniform('rho', lower=priors['rho'][0], upper=priors['rho'][1])

            Lambda_ += [muEff_, sigEff_, muP_, sigP_, rho_]

        elif spin_model=='chieffchip_uc':

            print('Modeling spin distribution with uncorrelated gaussians in chieff-chip')
            muEff_ = pm.Uniform('muEff', lower=priors['muEff'][0], upper=priors['muEff'][1])
            sigEff_ = pm.Uniform('sigEff', lower=priors['sigEff'][0], upper=priors['sigEff'][1])
            muP_ = pm.Uniform('muP', lower=priors['muP'][0], upper=priors['muP'][1])
            sigP_ = pm.Uniform('sigP', lower=priors['sigP'][0], upper=priors['sigP'][1])

            Lambda_ += [muEff_, sigEff_, muP_, sigP_]

        elif spin_model=='default':

            print('Modeling spin distribution with default spin model')

            if not use_log_alpha_beta:
                muChi_ = pm.Uniform('muChi', lower=priors['muChi'][0], upper=priors['muChi'][1])
                varChi_ = pm.Uniform('varChi', lower=priors['varChi'][0], upper=priors['varChi'][1])
                zeta_ = pm.Uniform('zeta', lower=priors['zeta'][0], upper=priors['zeta'][1])
                sigmat_ = pm.Uniform('sigmat', lower=priors['sigmat'][0], upper=priors['sigmat'][1])
    
                kappa_ = muChi_*(1-muChi_)/varChi_-1
    
                alphaChi_ = pm.Deterministic('alphaChi',  muChi_*kappa_ )
                betaChi_ = pm.Deterministic('betaChi',  (1-muChi_)*kappa_ )
                stdChi_ = pm.Deterministic('stdChi',  at.sqrt(varChi_) )
    
    
                Lambda_ += [alphaChi_, betaChi_, zeta_, sigmat_]
    
                # Bound alpha, beta > 1    
                
                if alpha_beta_prior=='poly':
                    print("Tapering prior on alpha_chi, beta_chi with polynomial smoothing")
                    _ = pm.Potential('bound_alphaChi', atools.log_f_smooth_poly(alphaChi_, 5e-4,  1 )  )
                    _ = pm.Potential('bound_betaChi', atools.log_f_smooth_poly(betaChi_, 5e-4,  1  ))
                elif alpha_beta_prior=='sigmoid':
                    print("Tapering prior on alpha_chi, beta_chi with sigmoid smoothing")
                    _ = pm.Potential('bound_alphaChi', atools.log_sigmoid(alphaChi_,  1+3e-04, 1e-04)  )
                    _ = pm.Potential('bound_betaChi', atools.log_sigmoid(betaChi_, 1+3e-04, 1e-04)  )
                else:
                    print("Putting prior on alpha_chi, beta_chi with hard cut")
                    _ = pm.Potential('bound_alphaChi', at.switch( at.le(alphaChi_, 1. ), -np.inf, 0. ) )
                    _ = pm.Potential('bound_betaChi', at.switch( at.le(betaChi_, 1. ), -np.inf, 0.0 ) )
        
            else:
                # still to be tested. Might improve sampling/divergences
                print("Sampling in log(alpha-1), log(beta-1)")
                raise NotImplementedError()
                
        elif spin_model=='default_gauss':

            print('Modeling spin distribution with default spin model, gaussian distribution for magnitudes')

            muChi_ = pm.Uniform('muChi', lower=priors['muChi'][0], upper=priors['muChi'][1])
            sigmaChi_ = pm.Uniform('sigmaChi', lower=priors['sigmaChi'][0], upper=priors['sigmaChi'][1])
            
            zeta_ = pm.Uniform('zeta', lower=priors['zeta'][0], upper=priors['zeta'][1])
            sigmat_ = pm.Uniform('sigmat', lower=priors['sigmat'][0], upper=priors['sigmat'][1])

            Lambda_ += [muChi_, sigmaChi_, zeta_, sigmat_]
            
        else:
            print('No model of the spin distribution.')
                

            

        ################################################
        # Mass distribution
        ################################################
            
        if mass_model=='PLPreg':

            ### BBH
            
            # Power law + peak
            print('Modeling mass distribution with LVK Power Law + Peak with regularized edge')
            if smoothing=='LVK':
                print('Using LVK smoothing')
            elif smoothing=='poly':
                print('using differentiable polynomial smoothing')
            
            alpha_  = pm.Uniform("alpha",      lower=priors["alpha"][0],      upper=priors["alpha"][1],      initval=ivals.get("alpha"))
            beta_   = pm.Uniform("beta",       lower=priors["beta"][0],       upper=priors["beta"][1],       initval=ivals.get("beta"))
            muM_    = pm.Uniform("muMass",     lower=priors["muMass"][0],     upper=priors["muMass"][1],     initval=ivals.get("muMass"))
                            
            lamP_   = pm.Uniform("lambdaPeak", lower=priors["lambdaPeak"][0], upper=priors["lambdaPeak"][1], initval=ivals.get("lambdaPeak"))        
            ml_     = pm.Uniform("ml",         lower=priors["ml"][0],         upper=priors["ml"][1],         initval=ivals.get("ml"))
            mh_     = pm.Uniform("mh",         lower=priors["mh"][0],         upper=priors["mh"][1],         initval=ivals.get("mh"))
            deltam_ = pm.Uniform("deltam",     lower=priors["deltam"][0],     upper=priors["deltam"][1],     initval=ivals.get("deltam"))
            sM_     = pm.Uniform("sigmaMass",  lower=priors["sigmaMass"][0],  upper=priors["sigmaMass"][1],  initval=ivals.get("sigmaMass"))
            


            Lambda_ += [lamP_, alpha_, beta_, deltam_, ml_, mh_, muM_, sM_ ]


        elif mass_model=='DPLDP':

            print('Modeling mass distribution with Double Power Law + Double Peak ')

            alpha1_   = pm.Uniform("alpha1",   lower=priors["alpha1"][0],   upper=priors["alpha1"][1],   initval=ivals.get("alpha1"))
            alpha2_   = pm.Uniform("alpha2",   lower=priors["alpha2"][0],   upper=priors["alpha2"][1],   initval=ivals.get("alpha2"))
            mb_       = pm.Uniform("mb",       lower=priors["mb"][0],       upper=priors["mb"][1],       initval=ivals.get("mb"))
            mu1_      = pm.Uniform("mu1",      lower=priors["mu1"][0],      upper=priors["mu1"][1],      initval=ivals.get("mu1"))
            sigma1_   = pm.Uniform("sigma1",   lower=priors["sigma1"][0],   upper=priors["sigma1"][1],   initval=ivals.get("sigma1"))
            mu2_      = pm.Uniform("mu2",      lower=priors["mu2"][0],      upper=priors["mu2"][1],      initval=ivals.get("mu2"))
            sigma2_   = pm.Uniform("sigma2",   lower=priors["sigma2"][0],   upper=priors["sigma2"][1],   initval=ivals.get("sigma2"))
            u         = pm.Uniform("u", 0, 1, initval=ivals.get("u"))
            m1_low_   = pm.Deterministic("m1_low", 3 + (10 - 3) * at.sqrt(u))
            v         = pm.Uniform("v", 0, 1, initval=ivals.get("v"))
            m2_low_   = pm.Deterministic("m2_low", 3 + v * (m1_low_ - 3))
            m_high_   = pm.Deterministic("m_high", at.as_tensor_variable(300.0)) #.astype(X)  )
            delta_m1_ = pm.Uniform("delta_m1", lower=priors["delta_m1"][0], upper=priors["delta_m1"][1], initval=ivals.get("delta_m1"))
            lambda_vec = pm.Dirichlet("lambda", a=np.asarray([1, 1, 1]), initval=np.asarray(ivals.get("lambda")))
            lambda0_  = pm.Deterministic("lambda0", lambda_vec[0])
            lambda1_  = pm.Deterministic("lambda1", lambda_vec[1])
            lambda2_  = pm.Deterministic("lambda2", lambda_vec[2])
            beta_     = pm.Uniform("beta",     lower=priors["beta"][0],     upper=priors["beta"][1],     initval=ivals.get("beta"))
            delta_m2_ = pm.Uniform("delta_m2", lower=priors["delta_m2"][0], upper=priors["delta_m2"][1], initval=ivals.get("delta_m2"))
            epsilon_  = pm.Deterministic("epsilon", at.as_tensor_variable(0.01) )
            
            if has_m2_break:
                print("Including gap for secondary mass")
                m_g_     =  pm.Uniform("m_g", lower=priors["m_g"][0], upper=priors["m_g"][1], initval=ivals.get("m_g")) 
                w_g_     = pm.Uniform("w_g", lower=priors["w_g"][0], upper=priors["w_g"][1], initval=ivals.get("w_g")) 
                sig_g_l_ = at.as_tensor_variable(1e-02)#.astype(X)
                sig_g_h_ = at.as_tensor_variable(1e-02)#.astype(X)
            else:
                m_g_     = at.as_tensor_variable(45.)#.astype(X)
                w_g_     = at.as_tensor_variable(70.)#.astype(X)
                sig_g_l_ = at.as_tensor_variable(1e-02)#.astype(X)
                sig_g_h_ = at.as_tensor_variable(1e-02)#.astype(X)
            
            Lambda_ += [alpha1_, alpha2_, mb_, mu1_, sigma1_, mu2_, sigma2_, m1_low_, m_high_, delta_m1_, lambda0_, lambda1_, lambda2_, beta_, m2_low_, delta_m2_, epsilon_, m_g_, w_g_, sig_g_l_, sig_g_h_]

        
        
        elif mass_model=='DPLDP-z':


            print("Modeling mass distribution with DPLDP + redshift-evolving hyperparameters")

            # -------------------------
            # Low-z (z≈0) hyperparameters (same as before)
            # -------------------------
            alpha1_0  = pm.Uniform("alpha1_0",  lower=priors["alpha1_0"][0],  upper=priors["alpha1_0"][1],  initval=ivals.get("alpha1_0"))
            alpha2_0  = pm.Uniform("alpha2_0",  lower=priors["alpha2_0"][0],  upper=priors["alpha2_0"][1],  initval=ivals.get("alpha2_0"))
            mb_0      = pm.Uniform("mb_0",      lower=priors["mb_0"][0],      upper=priors["mb_0"][1],      initval=ivals.get("mb_0"))
            mu1_0     = pm.Uniform("mu1_0",     lower=priors["mu1_0"][0],     upper=priors["mu1_0"][1],     initval=ivals.get("mu1_0"))
            
            #sigma1_0  = pm.Uniform("sigma1_0",  lower=priors["sigma1_0"][0],  upper=priors["sigma1_0"][1],  initval=ivals.get("sigma1_0"))
            sigma1_0 = pm.Truncated(
                        "sigma1_0",
                        pm.LogNormal.dist(mu=np.log(0.6), sigma=0.9),
                        lower=priors["sigma1_0"][0],
                        upper=priors["sigma1_0"][1],
                        initval=ivals.get("sigma1_0"),
                    )
            
            mu2_0     = pm.Uniform("mu2_0",     lower=priors["mu2_0"][0],     upper=priors["mu2_0"][1],     initval=ivals.get("mu2_0"))
            #sigma2_0  = pm.Uniform("sigma2_0",  lower=priors["sigma2_0"][0],  upper=priors["sigma2_0"][1],  initval=ivals.get("sigma2_0"))
            sigma2_0 = pm.Truncated(
                        "sigma2_0",
                        pm.LogNormal.dist(mu=np.log(4.0), sigma=0.9),
                        lower=priors["sigma2_0"][0],
                        upper=priors["sigma2_0"][1],
                        initval=ivals.get("sigma2_0"),
                    )
            
            
            delta_m1_ = pm.Uniform("delta_m1",  lower=priors["delta_m1"][0],upper=priors["delta_m1"][1],initval=ivals.get("delta_m1"))
            
            # m1_low, m2_low, m_high as in your original block
            u        = pm.Uniform("u", 0, 1, initval=ivals.get("u"))
            m1_low_  = pm.Deterministic("m1_low", (3 + (10 - 3) * at.sqrt(u)) ) #.astype(X) )
            v        = pm.Uniform("v", 0, 1, initval=ivals.get("v"))
            m2_low_  = pm.Deterministic("m2_low", (3 + v * (m1_low_ - 3)) ) #.astype(X))
            m_high_  = pm.Deterministic("m_high", at.as_tensor_variable(300.0)) #.astype(X))
            

            
            # secondary-mass hyperparams (unchanged unless you also evolve them)
            beta_     = pm.Uniform("beta",     lower=priors["beta"][0],     upper=priors["beta"][1],     initval=ivals.get("beta"))
            delta_m2_ = pm.Uniform("delta_m2", lower=priors["delta_m2"][0], upper=priors["delta_m2"][1], initval=ivals.get("delta_m2"))
            epsilon_  = pm.Deterministic("epsilon", at.as_tensor_variable(0.1) ) #.astype(X))
            
            if has_m2_break:
                print("Including gap for secondary mass")
                m_g_     = pm.Uniform("m_g", lower=priors["m_g"][0], upper=priors["m_g"][1], initval=ivals.get("m_g"))
                w_g_     = pm.Uniform("w_g", lower=priors["w_g"][0], upper=priors["w_g"][1], initval=ivals.get("w_g"))
                sig_g_l_ = at.as_tensor_variable(1e-02)#.astype(X)
                sig_g_h_ = at.as_tensor_variable(1e-02)#.astype(X)
            else:
                m_g_     = at.as_tensor_variable(45.)#.astype(X)
                w_g_     = at.as_tensor_variable(70.)#.astype(X)
                sig_g_l_ = at.as_tensor_variable(1e-02)#.astype(X)
                sig_g_h_ = at.as_tensor_variable(1e-02)#.astype(X)



            # # mixture weights at z≈0
            #eps_w = at.as_tensor_variable(1e-12)#.astype(X)

            # read lambda prior from priors file
            lam0_prior = priors.get("lambda0_vec_0", "Dirichlet(1,1,1)")
            
            if isinstance(lam0_prior, str) and lam0_prior.startswith("Dirichlet"):
                # parse "Dirichlet(a,b,c)"
                inside = lam0_prior[len("Dirichlet("):-1]
                alphas = [float(x.strip()) for x in inside.split(",")]
            else:
                # alternatively allow direct numeric lists in future
                alphas = lam0_prior
            
            alphas = np.asarray(alphas)
            lambda_vec0 = pm.Dirichlet(
                                    "lambda0_vec",
                                    a=alphas,
                                    initval=np.asarray(ivals.get("lambda"))
                                )



            lambda0_0 = pm.Deterministic("lambda0_0", lambda_vec0[0])
            lambda1_0 = pm.Deterministic("lambda1_0", lambda_vec0[1])
            lambda2_0 = pm.Deterministic("lambda2_0", lambda_vec0[2])


            alpha1_inf_,  z_alpha1_,  dz_alpha1_ = putils.evo_triplet(
                "alpha1",
                theta0_rv=alpha1_0,
                ivals=ivals,
                priors=priors,
            )
        
            alpha2_inf_,  z_alpha2_,  dz_alpha2_ = putils.evo_triplet(
                "alpha2",
                theta0_rv=alpha2_0,
                ivals=ivals,
                priors=priors,
            )
        
            
            mb_inf_ = pm.Deterministic("mb_inf", mb_0) 
            z_mb_   = pm.Deterministic("z_mb", at.as_tensor_variable(0.0) ) #.astype(X)) 
            dz_mb_  = pm.Deterministic("dz_mb", at.as_tensor_variable(1.0)) #.astype(X))  
            
        
            mu1_inf_,     z_mu1_,     dz_mu1_    = putils.evo_triplet(
                "mu1",
                theta0_rv=mu1_0,
                ivals=ivals,
                priors=priors,
            )
        
            sigma1_inf_,  z_sigma1_,  dz_sigma1_ = putils.evo_triplet(
                "sigma1",
                theta0_rv=sigma1_0,
                ivals=ivals,
                priors=priors,
            )
        
            mu2_inf_,     z_mu2_,     dz_mu2_    = putils.evo_triplet(
                "mu2",
                theta0_rv=mu2_0,
                ivals=ivals,
                priors=priors,
            )
        
            sigma2_inf_,  z_sigma2_,  dz_sigma2_ = putils.evo_triplet(
                "sigma2",
                theta0_rv=sigma2_0,
                ivals=ivals,
                priors=priors,
            )


            # -------------------------
            # High-z mixture weights + shared evolution S_lambda(z)
            # -------------------------

            # Global redshift–transition priors for all evolving hyperparameters
            z_t_prior = priors.get("z_t", (0.05, 1.5))   # lower/upper for z_transition
            dz_prior  = priors.get("dz",  (0.05, 2.0))   # lower/upper for Δz

            # High-z endpoint on the simplex
            lambda_vec_inf = pm.Dirichlet(
                "lambda_inf_vec",
                a=np.asarray([1, 1, 1]),
                initval=np.asarray(ivals.get("lambda_inf_vec", [0.10, 0.05, 0.85])),
            )
            lambda0_inf_ = pm.Deterministic("lambda0_inf", lambda_vec_inf[0])
            lambda1_inf_ = pm.Deterministic("lambda1_inf", lambda_vec_inf[1])
            lambda2_inf_ = pm.Deterministic("lambda2_inf", lambda_vec_inf[2])
            
            # Shared transition redshift and width for the mixture weights
            z_lambda_ = pm.Uniform(
                "z_lambda",
                lower=z_t_prior[0],
                upper=z_t_prior[1],
                initval=ivals.get("z_lambda", 1.1),
            )
            log_dz_lambda_ = pm.Uniform(
                "log_dz_lambda",
                lower=np.log(dz_prior[0]), #.astype(X),
                upper=np.log(dz_prior[1]), #.astype(X),
                initval=ivals.get("log_dz_lambda", np.log(0.5) ),
            )
            dz_lambda_ = pm.Deterministic("dz_lambda", at.exp(log_dz_lambda_))

            
            if simplex_repair:
                print("Will enforce lambda0(z), lambda1(z), lambda2(z) on the simplex")
                raise ValueError()
            # -------------------------
            # Pack hyperparameters for logpdf_DPLDP_z wrapper
            #   - low-z vector: same order, but with *_0 values
            #   - evolution params: (theta_inf, z_theta, dz_theta) for each evolving parameter
            # -------------------------
            lambdaBBHmass_lowz_ = [
                alpha1_0, alpha2_0, mb_0,
                mu1_0, sigma1_0, mu2_0, sigma2_0,
                m1_low_, m_high_, delta_m1_,
                lambda0_0, lambda1_0, lambda2_0, 
                beta_, m2_low_, delta_m2_,
                epsilon_, m_g_, w_g_, sig_g_l_, sig_g_h_
            ]
            
            evo_params_ = [
                alpha1_inf_,  z_alpha1_,  dz_alpha1_,
                alpha2_inf_,  z_alpha2_,  dz_alpha2_,
                mb_inf_,      z_mb_,      dz_mb_,
                mu1_inf_,     z_mu1_,     dz_mu1_,
                sigma1_inf_,  z_sigma1_,  dz_sigma1_,
                mu2_inf_,     z_mu2_,     dz_mu2_,
                sigma2_inf_,  z_sigma2_,  dz_sigma2_,
                lambda0_inf_, lambda1_inf_, lambda2_inf_, z_lambda_, dz_lambda_,
            ]
            
            # If your code expects a single list Lambda_, append both
            Lambda_ += [*lambdaBBHmass_lowz_, *evo_params_]
            
        
        
        
        ### BNS
        elif 'BNSgauss' in mass_model:

            if mass_model=='BNSgauss':
                # Uncorrelated gaussians
                print('Modeling mass distribution with uncorrelated gaussian distributions')
            elif mass_model=='BNSgaussCond':
                # Conditioned gaussians
                print('Modeling mass distribution with gaussian distributions with p(m1, m2) = p(m1) p(m2|m1) H(m1-m2)')
                
            muM_ = pm.Uniform('muMass', lower=priors['muMass'][0], upper=priors['muMass'][1])
            sM_ = pm.Uniform('sigmaMass', lower=priors['sigmaMass'][0], upper=priors['sigmaMass'][1] )  
            Lambda_ += [muM_, sM_ ]

        ### Non - parametric
        elif mass_model in ('DPUC', 'DP'):

            print("Modeling mass distribution as Dirichelet Process. Max number of components: %s"%N_DP_comp_max)

            if DP_prior=='SB':

                print("Prior for the process is stick-breaking")
                #### Stick Breaking Prior
                alpha_inv_init = 10. #alpha_inv_params[0] / alpha_inv_params[1]
                alpha_inv = pm.Gamma("alpha_inv", alpha_inv_params[0], alpha_inv_params[1], initval=alpha_inv_init )
                print("alpha_inv prior has parameters %s"%str(alpha_inv_params))
                alpha = 1/alpha_inv
    
                #beta_init = np.full(N_DP_comp_max_np, 1e-02)#.astype(X)
                #beta_init[0] = 0.99
                beta_init = np.full(N_DP_comp_max_np, 0.02)
                beta_init[:5] = [0.80, 0.60, 0.35, 0.20, 0.10]
    
                beta = pm.Beta("beta", 1.0, alpha, dims="component" , initval=beta_init)

                #b0 = pm.draw(beta, draws=1)
                print("beta init top10:", np.sort(beta_init)[::-1][:10])
                print("beta init first10:", beta_init[:10])


                #w = pm.Deterministic("w", atools.stick_breaking(beta), dims="component")

                w_raw = atools.stick_breaking(beta)
                w = pm.Deterministic("w", w_raw / at.sum(w_raw), dims="component")

                # ---- quick init diagnostics (runs once at model build) ----
                w0 =  atools.stick_breaking(beta_init) #pm.draw(w, draws=1)
                w0 = (w0/at.sum(w0)).eval()
                print("w init: sum=", w0.sum(), " max=", w0.max(), " min=", w0.min())
                print("w init top10:", np.sort(w0)[::-1][:10])
                print("# comps for 90% mass:", np.searchsorted(np.cumsum(np.sort(w0)[::-1]), 0.90) + 1)

            elif DP_prior=='dirichelet':
                print("Prior for the process is dirichelet")

                print("alpha_total prior is Gamma with parameters %s"%str(gamma_DP_params))
                
                ### Dirichelet Prior
                alpha_total = pm.Gamma("alpha_total", alpha=gamma_DP_params[0], beta=gamma_DP_params[1])  # mean ≈ 5
                a = alpha_total / N_DP_comp_max
                w = pm.Dirichlet("w", a=at.ones(N_DP_comp_max) * a, dims="component")

            elif DP_prior=='softmax':
                print("Prior for the process is softmax")
                print("sigma_w sampled from halfnormal with std=%s"%sigma_softmax)
                
                ### Uniform Prior
                sigma_w = pm.HalfNormal("sigma_w", sigma=sigma_softmax)
                raw_w = pm.Normal("raw_w", 0, sigma_w, dims="component")  # small variance
                w = pm.Deterministic("w", pm.math.softmax(raw_w), dims="component")

            else:
                raise ValueError()

            
            logw = at.log(w)

        

            #### Mean prior 


            U1, U2 = (upmu1-lowmu1) , (upmu2-lowmu2)    # "too-wide" typical std per dim 

            mu1_center = (lowmu1 + upmu1) / 2.0  # 3.55
            mu2_center = (lowmu2 + upmu2) / 2.0

            M_active = 5

            mu1_init = np.full(N_DP_comp_max_np, mu1_center)
            mu2_init = np.full(N_DP_comp_max_np, mu2_center)
            
            mu1_init[:M_active] = np.linspace(lowmu1 + 0.1*(upmu1-lowmu1),
                                              upmu1  - 0.1*(upmu1-lowmu1),
                                              M_active)
            
            mu2_init[:M_active] = np.linspace(lowmu2 + 0.1*(upmu2-lowmu2),
                                              upmu2  - 0.1*(upmu2-lowmu2),
                                              M_active)
            
     
            mu1 = pm.Uniform('mulMc', lower=lowmu1, upper=upmu1, dims= ("component" ), initval=np.full(N_DP_comp_max_np, mu1_center)) #.astype(X) )
            mu2 = pm.Uniform('mulq', lower=lowmu2, upper=upmu2, dims= ("component" ), initval=np.full(N_DP_comp_max_np, mu2_center)) #.astype(X))

            if rate_model in ('DPUC','DPUC-vol' ):

                mu3_center = ( lowmu3+ upmu3) / 2.0
                mu3_init = np.full(N_DP_comp_max_np, mu3_center)
                mu3_init[:M_active] = np.linspace(lowmu3 + 0.1*(upmu3-lowmu3),
                                              upmu3  - 0.1*(upmu3-lowmu3),
                                              M_active)
                
                
                mu3 = pm.Uniform('mulz', lower=lowmu3, upper=upmu3, dims= ("component" ), initval=np.full(N_DP_comp_max_np, mu3_center)) #.astype(X))

                mus = at.stack([mu1, mu2, mu3], axis=0)
                
            else:
                mus = at.stack([mu1, mu2], axis=0)     
                
            

            mu = pm.Deterministic("mu", mus, dims=("GMMdimension", "component") )

            
            #### Sigma prior 
            
            print("L_small_1 = %s "%L_small_1)
            print("L_small_2 = %s "%L_small_2)

            print("U1 = %s "%U1)
            print("U2 = %s "%U2)


            # # Fréchet shape for 1D marginal: alpha = d/2 with d=1 -> 0.5
            # print("P( sigma < L_small ) = %s "%alpha_small)

            # alpha_shape = 0.5
            #lambda_ell_1 = -at.log(alpha_small) * L_small_1**(alpha_shape) # small scale
            #lambda_ell_2 = -at.log(alpha_small) * L_small_2**(alpha_shape) # small scale
            
            # tau1 = pm.CustomDist("tau1", lambda_ell_1, 1,
            #               logp=atools.frechet_logp_full,
            #               transform=tr.log, initval=0.2,
            #               random=atools.frechet_random, )

            # tau2 = pm.CustomDist("tau2", lambda_ell_2, 1,
            #               logp=atools.frechet_logp_full,
            #               transform=tr.log, initval=0.2,
            #               random=atools.frechet_random, )

            tau1 = pm.Uniform("tau1", lower=L_small_1, upper=U1, initval= (U1 / 2.0 )  )
            tau2 = pm.Uniform("tau2", lower=L_small_2, upper=U2, initval= (U2 / 2.0 )  )

            print("s_local = %s "%s_local)

            eps1 = pm.Normal("eps1", 0.0, s_local, dims=("component",), initval=np.zeros(N_DP_comp_max_np)) #.astype(X))
            eps2 = pm.Normal("eps2", 0.0, s_local, dims=("component",), initval=np.zeros(N_DP_comp_max_np)) #.astype(X))

            # eps1 = pm.SkewNormal("eps1", mu=0, sigma=s_local, alpha=+2, dims=("component",), initval=np.zeros(N_DP_comp_max_np).astype(X) )
            # eps2 = pm.SkewNormal("eps2", mu=0, sigma=s_local, alpha=+2, dims=("component",), initval=np.zeros(N_DP_comp_max_np).astype(X))


            sig1 = pm.Deterministic("sig1", tau1 * at.exp(eps1) , dims="component")   
            sig2 = pm.Deterministic("sig2", tau2 * at.exp(eps2), dims="component")  

            
            if rate_model in ('DPUC', 'DPUC-vol'):

                
                U3 = (upmu3-lowmu3)

                print("L_small_3 = %s "%L_small_3)
                print("U3 = %s "%U3)

                tau3 = pm.Uniform("tau3", lower=L_small_3, upper=U3,initval= (U3 / 2.0 )  )
                # eps3 = pm.SkewNormal("eps3", mu=0, sigma=s_local, alpha=+2, dims=("component",), initval=np.zeros(N_DP_comp_max_np).astype(X))
                eps3 = pm.Normal("eps3", 0.0, s_local, dims=("component",), initval=np.zeros(N_DP_comp_max_np)) #.astype(X))
                sig3 = pm.Deterministic("sig3", tau3 * at.exp(eps3), dims="component")  

                sigs = at.stack([sig1, sig2, sig3], axis=0)
                
            else:
                sigs = at.stack([sig1, sig2], axis=0)

            if alpha_tail!=-1:

                # ----- Penalize large sigma -----
                
                
                print("P(tau_1,2 > U_1,2) = %s "%alpha_tail)
                
                lambda_large_1 = -np.log(alpha_tail) / U1   
                lambda_large_2 = -np.log(alpha_tail) / U2   
    
    
                _ = pm.Potential( "pc_large_ell_1", -lambda_large_1 * tau1,  )
                _ = pm.Potential( "pc_large_ell_2", -lambda_large_2 * tau2, )

          
            if mass_model=='DPUC':
                print("No m1-m2 correlation.")
                
                sd = pm.Deterministic("sig", sigs, dims=("GMMdimension", "component"))

                Lambda_ += [ w, mu, sd, logw ]

            elif mass_model=='DP':
                print("Including m1-m2 correlation.")
                # -------- Correlation prior --------

                eta=1.
                print("eta = %s"%eta)
                rho_u = pm.Beta("rho_u", alpha=eta, beta=eta, dims=("component",))

                # #rho_max = 0.9  # cap on |rho|
                # # choose fraction f of L_small you allow for the minor axis
                f = 0.5   # minor axis at least 100xf% of L_small in worst case
                rho_max = np.sqrt(1.0 - f**2)  # ≈ 0.866
                print("rho_max = %s, with f=%s, i.e minor axis is at least %s of L_small in worst case"%(rho_max,f,f))
                rho   = pm.Deterministic("rho", rho_max * (2.0 * rho_u - 1.0), dims="component")

                # rho = pm.Uniform("rho", lower=-rho_max, upper=rho_max, dims="component")
                # pm.Potential(
                #     "lkj_corr_prior",
                #     (eta - 1.0) * at.log(1.0 - rho**2).sum()
                # )
    
                # # Useful terms
                one_minus_r2 = 1.0 - rho**2
                sqrt1mr2     = at.sqrt(one_minus_r2)
                
                # ----- Cholesky of Σ (for reference / if you need solves) -----
                # Σ = [[s1^2, ρ s1 s2], [ρ s1 s2, s2^2]]
                # Cholesky L = diag([s1, s2]) @ [[1, 0], [ρ, sqrt(1-ρ^2)]]
                row0 = at.stack([sig1,               at.zeros_like(sig1)], axis=1)          # (K,2)
                row1 = at.stack([rho * sig2,         sig2 * sqrt1mr2     ], axis=1)          # (K,2)
                L    = at.stack([row0, row1], axis=1)     
                Cho_cov = pm.Deterministic("Cho_cov", L, dims=("component","GMMdimension","GMMdimension_1"))
                
                # ----- log |Σ^{-1}| (no inverses) -----
                # det Σ = s1^2 * s2^2 * (1 - ρ^2)
                # log |Σ^{-1}| = - log det Σ
                ldets_inv = pm.Deterministic(
                    "ldets_inv",
                    -2.0 * at.log(sig1) - 2.0 * at.log(sig2) - at.log(one_minus_r2),
                    dims="component",
                )
                
                # ----- Precision Σ^{-1} in closed form (Fisher) -----
                # Σ^{-1} = 1 / [ (1-ρ^2) s1^2 s2^2 ] * [[ s2^2, -ρ s1 s2 ], [ -ρ s1 s2, s1^2 ]]
                # variances
                var1 = sig1**2          # (K,)
                var2 = sig2**2          # (K,)
                cov12 = rho * sig1 * sig2
                
                den = one_minus_r2 * (var1) * (var2)
                F11 =  (var2)            / den
                F22 =  (var1)            / den
                F12 = -(cov12)    / den
                
                Fisher = pm.Deterministic( "Fisher", at.stack([
                    at.stack([F11, F12], axis=1),
                    at.stack([F12, F22], axis=1)
                ], axis=1), dims=("component","GMMdimension_1","GMMdimension_2"))  # shape: (K, 2, 2)
    

                ################################################
    
                Lambda_ += [ alpha, beta, w, mu, Fisher, ldets_inv, logw ]

            Lambda_ += [N_DP_comp_max]


            
        ################################################
        # If including total normalization of the rate, add it here
        ################################################
        
        if not marginal_R0:
            R0 = pm.Uniform('R0', lower=priors['R0'][0], upper=priors['R0'][1])
        else:
            R0 = at.as_tensor_variable(1.)   
        lR0 = at.log(R0)

    
            
        ################################################
        # Individual event mass and distance
        ###############################################

        x = pm.Normal( 'x', mu=0, sigma=1, dims= ("event_index" , "GWdimension" ), initval = (np.random.randn(N, nd) * eps_init)) #.astype(X) )    

            
        if 'gmm' in sampling_GW:
    
            print('Sampling m1d, m2d, dL from GMM')

                
            ###################################
            # categorical way

            ig = pm.Categorical('idx', p=wts_l, dims= "event_index",  initval=at.argmax(wts_l, axis=1)) #.astype(int_dtype) )

   
            # Select means and Cholesky factors per batch
            mu_selected = mus_l[ np.arange(N), ig, :]         # shape (N, D)
            L_selected = cho_covs_l[ np.arange(N), ig, :, :]  # shape (N, D, D)
             
            # Batched matrix multiplication: (N, D, D) @ (N, D, 1) → (N, D, 1)
            Lx = at.sum(L_selected * x[:, None, :], axis=2)  # → shape (N, D)

                      
            # Final transformed sample
            samples = mu_selected + Lx                # shape (N, D)

            
            log_Mc_det = samples[:,0]/dil_factor
            logit_q = samples[:,1]
            logd = samples[:,2]
            

            if (spin_model == 'chieffchip') or (spin_model == 'chieffchip_uc') :
    
                chieff = atools.inv_flogitat(samples[:,3])
                chip = atools.inv_logitat(samples[:,4])
    
            elif (spin_model == 'default') or (spin_model == 'default_gauss'):
                # we have chi1, chi2, cost1, cost2
                if save_thetas:
                    chi1 = pm.Deterministic('chi1', atools.inv_logitat(samples[:,3]))
                    chi2 = pm.Deterministic('chi2', atools.inv_logitat(samples[:,4]))
        
                    cost1 = pm.Deterministic('cost1', atools.inv_flogitat(samples[:,5]))
                    cost2 = pm.Deterministic('cost2', atools.inv_flogitat(samples[:,6]))
                else:
                    chi1 = atools.inv_logitat(samples[:,3])
                    chi2 = atools.inv_logitat(samples[:,4])
                    cost1 =atools.inv_flogitat(samples[:,5])
                    cost2 =atools.inv_flogitat(samples[:,6])
                    
            else:
                print("No spins computed")
        

        
        elif sampling_GW=='gauss' : # to be tested with spins
            
            print('Sampling log(Mc), logit(q), log(dL) from Gaussian approximant')

            
            # sample = mu + L @ x   (batched)
            #samples = mus_s + at.matmul(cho_s, x[..., None])[..., 0]      # (N, d)
            samples = mus_s + at.sum(cho_s * x[:, None, :], axis=-1)

          
        
            # logp = log p(x) - log|L|
            # d = x.shape[1]
            log_px = -0.5 * at.sum(x**2, axis=1) - 0.5 * nd_np * at.log(2*np.pi)    # (N,)


            log_det_L = at.sum(at.log(at.diagonal(cho_s, axis1=1, axis2=2)), axis=1)  # (N,)

            
            pilik = log_px - log_det_L                                               # (N,)

            # unpack coordinates:
            log_Mc_det = samples[:, 0]
            logit_q    = samples[:, 1]
            logd       = samples[:, 2]
            

            if spin_model == 'none' :
                
                X = at.stack([log_Mc_det, logit_q, logd ], axis=1)
                d_int  = 3


            elif spin_model == 'default' or spin_model == 'default_gauss':

                chi1 = atools.inv_logitat(samples[:,3])
                chi2 = atools.inv_logitat(samples[:,4])
    
                cost1 = atools.inv_flogitat(samples[:,5])
                cost2 = atools.inv_flogitat(samples[:,6])

                X = at.stack([log_Mc_det, logit_q, logd,  samples[:,3],  samples[:,4],  samples[:,5],  samples[:,6]], axis=1)
                d_int  = 7


        

            # X as (N, d)
            #X = vals.T                                   # (N, d)
            #print("X shape is %s"%(X[:, None, :].shape.eval()))
            #print("mus_l shape is %s"%(mus_l.shape.eval()))
            
            # Broadcast X against component-wise parameters
            # diff: (N, ngmm, d)
            diff = X[:, None, :] - mus_l[:, :, :d_int]                  # (N, 1, d) - (N, ngmm, d)
            
            # Quadratic form using precision F = Σ^{-1}
            # tmp = F @ diff[..., None]  -> (N, ngmm, d, 1) -> squeeze to (N, ngmm, d)

            
            tmp = at.matmul(icovs_l[:, :, :d_int, :d_int], diff[..., None])[..., 0]   # (N, ngmm, d)


            
            # r^T F r for each (obs, comp)
            quad = at.sum(diff * tmp, axis=-1)            # (N, ngmm)

            
            # Component logpdfs (Multivariate Normal)
            log_norm = (-0.5 * d_int * at.log(2*np.pi)) #.astype(work_dtype)     # scalar

            logp_components = (
                -0.5 * quad
                + log_norm
                - 0.5 * log_dets_l
                + log_wts_l
            )                                             # (N, ngmm)

            # Mixture log-likelihood per observation: logsumexp over components
            gwl = at.logsumexp(logp_components, axis=1, )   # (N,)

        
        else:
            raise NotImplementedError()


        Mc = at.exp(log_Mc_det)            
        q = atools.inv_logitat(logit_q)
        m1det, m2det = atools.m1m2_from_Mcq_at(Mc, q)
        d = at.exp(logd)                   
            
            
        ################################################
        # Population prior and selection computation
        ################################################



        if spin_model in ("default", "default_gauss"):
            spins = at.stack([chi1, chi2, cost1, cost2], axis=1)  # (N,4)
        elif spin_model in ("chieffchip", "chieffchip_uc"):
            spins = at.stack([chieff, chip], axis=1)              # (N,2)
        else:
            spins = at.zeros((m1src.shape[0], 0), dtype="float64")  # (N,0)

        if mass_model not in ('DP', 'DPUC', 'DPLDP-z'):
            Lambda_ = at.stack(Lambda_, axis=0)

        if chunk_inj:
            print("Will process injections in chunks of %s"%chunk_inj)

        
        fused = PopAndSelJAXOp(
        zgrid=zgrid_np,
        x01=constants._x01_np,
        w01=constants._w01_np,
        rate_model=rate_model,
        mass_model=mass_model,
        spin_model=spin_model_name,
        smoothing=smoothing,
        simplex_repair=simplex_repair,
        has_m2_break=has_m2_break,
        norm_gauss=norm_gauss,
        param=param,
        #interp_mass=interp_mass,
        #mass_grids = interp_grids_mass,
        #is_observed=is_observed,
        subtract_log_p_incl=False,       
        linear_mass=linear_mass,
        linear_z=linear_z,
        chunk_inj=chunk_inj

        )
        
        if lp_incl_inj[0] is None:
            log_p_incl_ = at.zeros_like(lpdinj[0])
        else:
            log_p_incl_ = lp_incl_inj[0]
        
        log_p_pop, log_mu_, var_ll_u_ = fused(
            m1det, m2det, d, spins,           # EVENT side
            m1inj[0], m2inj[0], dLinj[0], spinsInj,
            lpdinj[0], log_p_incl_,
            Lambda_, Ndraw
        )


        ################################################
        
        if all_dLsq_prior:
            #dLprior=='dLsq':
            # Remove \pi(d)~dL^2 prior on distance 
            log_p_pop -= 2*logd
            print('Removing dL^2 prior for all events.')

        elif all_no_dL_prior:    
            print("No dL prior removed for all events.")
            
        elif vol_in_prior:

            zs_Planck15 = atools.atinterp(d, dL_grid_Planck15, zgrid_dLp)
            dc_Planck15 = atools.dcfun_at(zs_Planck15, 67.74, 0.3075, -1., interp=False)

            lpi = at.zeros_like(log_p_pop )

            # apply chunk-wise prior removal
            for i, lab in enumerate(dLprior):
                sl = slice(edges[i], edges[i+1])

                print('For events between %s and %s, removing prior %s'%(edges[i], edges[i+1], lab))


                if lab == 'dLsq':
                    print('chunk is dLsq')
                    print(sl)
                    chunk = 2 * logd[sl]

                elif lab == 'none':
                    print('chunk is zero')
                    print(sl)
                    chunk = at.zeros_like(log_p_pop[sl])

                else:
                    # base label + whether we apply the -J correction
                    use_J = lab.endswith('-J')
                    base = lab[:-2] if use_J else lab

                    if base == 'UniformComovingVolume':
                        print('chunk is UniformComovingVolume')
                        print(sl)
                        chunk = atools.log_dV_dz_at(
                            zs_Planck15[sl], 67.74, 0.3075, -1., dc=dc_Planck15[sl]
                        )

                    elif base == 'UniformSourceFrame':
                        print('chunk is UniformSourceFrame')
                        print(sl)
                        chunk = (
                            atools.log_dV_dz_at(
                                zs_Planck15[sl], 67.74, 0.3075, -1., dc=dc_Planck15[sl]
                            )
                            - at.log1p(zs_Planck15[sl])
                        )

                    else:
                        raise ValueError(f"Unknown dL prior label: {lab}")

                    if use_J:
                        print('removing log_ddL_dz ')
                        chunk -= atools.log_ddL_dz(
                            zs_Planck15[sl], 67.74, 0.3075, -1., 1., 0.,
                            dc=dc_Planck15[sl], interp=False, param='vanilla'
                        )

                    if normalize_PE_prior:
                        print('normalizing PE prior')
                        chunk -= log_norm_PE_prior

                lpi = at.set_subtensor(lpi[sl], chunk)
            
            log_p_pop -= lpi
            

        
        
        else:
            raise ValueError("Check dL prior choices.")
            

        if sampling_GW=='gauss' and not sample_from_pop:
                # Add gw likelihood and correct for sampling prior pdf
                log_p_pop -= pilik
                log_p_pop += gwl

        ################################################
        
        #  sum log likelihoods

        likelihood_val = at.sum( log_p_pop )

       
        # add R0*Tobs if needed. 
        if not marginal_R0:
            print("Will not marginalise over R0.")
            # each term p_pop is multiplied by
            # R0*T_obs . So we get a factor (R0*T_obs)**N_i for every
            # observing run. R0 is the same for every run so I just have
            # (R0)**{\sum N_i} . For T_obs I have T_{obs,1}**N_1 * T_{obs,2}**N_2 * ...
            poiss_term = at.sum(Nevs*at.log(allTobs))+N*lR0
            likelihood_val += poiss_term
        else:
            print("Will marginalise over R0 with flat-in-log prior.")

        
        
        _ = pm.Potential("likelihood", likelihood_val ) 



        ################################################
        # Selection effect
        ################################################
        
        if sel_method=='skip':
            print('No selection bias!')
        else:
            # add sel effects    
            if ndata_np==1:
                # we passed a single injection set corresponding to multiple observing runs,
                # with injections already containing the correct weights
                print("Using selection effects from a single injection campaign")
       
                if not marginal_R0:
                    # This is really the number of expected events 
                    sel_effect = -R0*Ttot*at.exp(log_mu_)
                else:
                    sel_effect = -N*log_mu_
    
            else:
                raise NotImplementedError()

            
            ################################################
            # Sel effect computed. Now exclude high-variance regions in the integral
     

            if marginal_R0:
                log_lik_var = pm.Deterministic('log_lik_var', at.exp(var_ll_u_+2*logN ) )
            else:
                log_lik_var = pm.Deterministic('log_lik_var', at.exp(  var_ll_u_+2*at.log( R0*Ttot ) + 2*log_mu_ ) )
            
     

            if ((Neff_min==0) and (log_lik_var_min==0)):
                print("No condition on number of effective points in MC integral for sel. effect")
                selection_bias =  sel_effect #pm.Deterministic("sel_bias", sel_effect )
            else:
                if log_lik_var_min==0:

                   raise NotImplementedError()

                
                elif Neff_min==0:

                    # Thresholding on likelihood variance
                    print("MC integral for sel. effect thresholded on log lik. variance")
                    
                    if sel_smoothing=='sigmoid':
                        # smooth with sigmoid 
                        print("Tapering sel effect with sigmoid smoothing")
                        
                        selection_bias = sel_effect + atools.logdiffexp( at.log(1), atools.log_sigmoid(log_lik_var, log_lik_var_min*(1+0.002), 0.001 )) 

                    
                    elif sel_smoothing=='poly':
                        print("Tapering sel effect with polynomial smoothing")

                        selection_bias = sel_effect
                        _ = pm.Potential("bound_log_lik_var", atools.logS_PLP(log_lik_var_min - log_lik_var, deltam=0.01, ml=-0.01))

                    else:
                        print("Tapering sel effect with hard cut")

                        selection_bias = sel_effect
                        _ = pm.Potential("bound_log_lik_var", at.switch(log_lik_var <= log_lik_var_min, 0.0, -np.inf ))

            
            _ = pm.Potential('selection_bias', selection_bias)

            if marginal_R0:
                if include_sel_uncertainty:
                    
                    
                    # from Farr 2019
                    # print("Including selection function uncertainty as in Farr 2019")
                    #sel_uncertainty = (3*N+N**2)/(2*Neff)

                    # from heinzel-Vitale 2025
                    print("Including selection function uncertainty as in Heinzel-Vitale 2025")
                    sel_uncertainty = - N*(N+1)/(2) * var_ll_u_
                    
                    _ = pm.Potential('selection_uncertainty', sel_uncertainty)
            

    return model



