#
#    Copyright (c) 2025 Michele Mancarella <mancarella@cpt.univ-mrs.fr>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import pytensor.tensor as at
import jax.numpy as np
import pymc as pm
import jax
from pytensor.graph import Apply, Op
import pytensor
from pytensor.gradient import grad

from jax.numpy import array
from jax.numpy import concatenate
from jax.numpy import ones
from jax.numpy import zeros

c_light = 299792458*1e-03
c_light_at = at.as_tensor_variable(c_light)
MIN = at.as_tensor_variable(-np.inf)
INF = at.as_tensor_variable(np.inf)
 
#if int(pytensor.__version__.split('.')[1])>25: #=='2.30.3':
try:
        zGridGlobals_at = at.sort(at.unique(at.concatenate([ 
            #[at.as_tensor_variable(0.)],
            #at.logspace(start=-100, stop=-15, base=10, steps=50), 
            at.logspace(start=-10, stop=-4, base=10, steps=5), 
                     at.logspace(start=-4, stop=1, base=10, steps=100), 
                     at.logspace(start=1, stop=2, base=10, steps=10), 
            #at.logspace(start=2, stop=5, base=10, steps=50) 
        ])))

except:
    
    zGridGlobals_at = at.sort(at.unique(at.concatenate([ 
        #[at.as_tensor_variable(0.)],
        #at.logspace(start=-100, end=-15, base=10, steps=50), 
        at.logspace(start=-10, end=-4, base=10, steps=5), 
                     at.logspace(start=-4, end=1, base=10, steps=100), 
                     at.logspace(start=1, end=2, base=10, steps=10), 
        #at.logspace(start=2, end=5, base=10, steps=50) 
    ])))


#zGridGlobals_at = at.linspace(start=0, end=3, steps=500) 

zGridGlobals = np.array(zGridGlobals_at.eval())






##########################
####### Auxiliary functions ########
##########################


def logsubexp(x, y):
    """`log(exp(x)-exp(y))` """
    return x + at.log1p(-at.exp(y-x))

def logsumexp(x, y):
    """`log(exp(x)+exp(y))` """
    return x + at.log1p(at.exp(y-x))

def logitat(p):
    return at.log(p) - at.log(1. - p)

def inv_logitat(p):
    return 1. / (1 + at.exp(-p))

def inv_flogitat(p):
    return (at.exp(p) - 1. ) / (1. + at.exp(p))

def logaddexp(x, y):
    """`log(exp(x)+exp(y))` """
    return x + at.log1p(at.exp(y-x))
    
def logdiffexp(x, y):
    '''
    computes log( e^x - e^y)
    '''
    return x + at.log1p(-at.exp(y-x))

 
def flogitat(p):
    return at.log(1 + p) - at.log(1 - p)

def m1m2_from_Mcq_at(Mc, q):
    
    m1 = Mc*(1+q)**(1./5.)/q**(3./5.)
    m2 = q*m1

    return m1, m2

def Mcq_from_m1m2_at(m1, m2):
   
    Mc  = ((m1*m2)**(3./5.))/((m1+m2)**(1./5.))
    q = m2/m1
    
    return Mc, q

def log_sigmoid(x, m, sig):
    return -at.log1p(at.exp(-(x-m)/sig))

def sigmoid(x, m, sig):
    return 1/(1+at.exp((-(x-m)/sig)))


def stick_breaking(beta):
    portion_remaining = at.concatenate([[1], at.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

##########################
####### Interpolators and integrators ########
##########################



def atinterp(x, xs, ys, return_grad=False):
    """
    Linearly interpolate ys(x) from (xs, ys) to x.
    Optionally returns gradient dy/dx.

    Args:
        x: TensorVariable (N,) — interpolation points
        xs: TensorVariable (M,) — fixed grid (sorted)
        ys: TensorVariable (M,) — values on the grid
        return_grad: bool — whether to return dy/dx

    Returns:
        y_interp: interpolated values at x
        (optional) grad: dy/dx at x
    """
    x = x.ravel()
    xs = xs.ravel()
    ys = ys.ravel()

    # Inject NaN if out-of-bounds
    #out_of_bounds = ~at.all((x >= xs[0]) & (x <= xs[-1]))
    #_ = at.switch(out_of_bounds, float("nan"), 0.0)

    # Interpolation indices
    idxs = at.searchsorted(xs, x, side='left')
    idxs = at.clip(idxs, 1, xs.shape[0] - 1)

    xl = xs[idxs - 1]
    xh = xs[idxs]
    yl = ys[idxs - 1]
    yh = ys[idxs]

    r = (x - xl) / (xh - xl)
    y_interp = r * yh + (1.0 - r) * yl

    if return_grad:
        dy_dx = (yh - yl) / (xh - xl)
        return y_interp, dy_dx
    else:
        return y_interp

def jnptinterp(x, xs, ys):

  idxs = jax.numpy.searchsorted(xs, x,  side='left', sorter=None)

  xl = xs[idxs-1]
  yl = ys[idxs-1]
  xh = xs[idxs]
  yh = ys[idxs]

  r = (x-xl)/(xh-xl);

  return r*yh + (1.0-r)*yl;




def atcumtrapz(y, x=None, dx=1.0, axis=-1, initial=None):

    
    if x.ndim == 1:
        d = at.diff(x)
        # reshape to correct shape
        shape = [1] * y.ndim
        shape[axis] = -1
        d = d.reshape(shape)
    elif len(x.shape) != len(y.shape):
        raise ValueError("If given, shape of x must be 1-d or the "
                         "same as y.I got: d.shape=%s, y.shape=%s"%(d.shape.eval(), x.shape.eval()))
    else:
        d = at.diff(x, axis=axis)

    nd = y.ndim
    
    if x.ndim==1:
        res = at.cumsum(d * (y[1:] + y[:-1]) / 2.0, axis=axis)
    elif (x.ndim==2) and ((axis==1) or (axis==-1)):        
        res = at.sum( d * (y[:, 1: ]+y[:, :-1])/2.0, axis )

    return res


def attrapzvec11(y, x,  axis=-1):

    # works in 1D and 2D

    if True:
        if x.ndim == 1:
            d = at.diff(x)
            # reshape to correct shape
            shape = [1]*y.ndim
            shape[axis] = d.shape[0]
            d = at.reshape(d, shape)
        else:
            d = at.diff(x, axis=axis)
    nd = y.ndim
    
    if x.ndim == 1:
        ret = at.sum(d * (y[1:] + y[:-1]) / 2.0)#.sum(axis)
    elif (x.ndim==2) and ((axis==1) or (axis==-1)):
        # Operations didn't work, cast to ndarray
        # d = np.asarray(d)
        # y = np.asarray(y)        
        ret = at.sum( d * (y[:, 1: ]+y[:, :-1])/2.0, axis )    
    else:
      raise NotImplementedError()
    return ret


def attrapzvec(y, x,  dx=1., axis=-1):
        if x is None:
                d = dx
        else:
                #x = asanyarray(x)
                if x.ndim == 1:
                    d = at.diff(x)
                    # reshape to correct shape
                    shape = [1]*y.ndim
                    shape[axis] = d.shape[0]
                    d = at.reshape(d, shape)
                else:
                    d = at.diff(x, axis=axis)
        
        nd = y.ndim
        slice1 = [slice(None)]*nd
        slice2 = [slice(None)]*nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        try:
            ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
        except ValueError:
            # Operations didn't work, cast to ndarray
            d = np.asarray(d)
            y = np.asarray(y)
            ret = add.reduce(d * (y[tuple(slice1)]+y[tuple(slice2)])/2.0, axis)
        return ret




##########################
####### Distances and cosmology ########
##########################


PI = at.as_tensor_variable(np.pi)

def dcfun_at(z, H0, Om, w0, interp=False):
    """Comoving distance at redshift ``z``, in Gpc, H0 in km/s/Mpc"""
    if interp:
      return c_light_at/H0 * _int_dC_hyperbolic(z, Om)*1e-03
    else:
      zz = at.linspace(0, z, steps=100).T
      E = Efun_at(zz,Om,w0 )
      return c_light_at/H0 * attrapzvec(1/E, zz)*1e-03


def dcfun_np(z, H0, Om, interp=False):
    """Comoving distance at redshift ``z``, in Gpc, H0 in km/s/Mpc"""
    if interp:
      return c_light/H0 * _int_dC_hyperbolic(z, Om)*1e-03
    else:
      zz = np.linspace(0, z, 100).T
      #print(zz ok)
      E = np.sqrt( Om*(1+zz)**3+(1-Om)  )
      return np.array(c_light/H0 * np.trapz(1/E, zz)*1e-03)


def Xifun_at(z, Xi0, n):
    return Xi0+(1-Xi0)/(1+z)**n

def dLfun_np(z, H0, Om, interp=False):
    """Luminosity distance at redshift ``z``."""
    return np.array((z+1.0)*dcfun_np(z, H0, Om, interp=interp))


def dLfun_at(z, H0, Om, w0, Xi0, n, interp=False):
    """Luminosity distance at redshift ``z``."""
    return Xifun_at(z, Xi0, n)*(z+1.0)*dcfun_at(z, H0, Om, w0, interp=interp)


def Efun_at(z,Om,w0 ):
    return at.sqrt( Om*(1+z)**3+(1-Om)  )


def z_from_dL_np(r, H0, Om, w0, Xi0, n ):
    dLGrid = np.array(dLfun_np( zGridGlobals, H0, Om=Om ))
    z2dL = jnptinterp( r, dLGrid, zGridGlobals ) 
    return np.array(z2dL)


def z_from_dL_at(r, H0, Om, w0, Lambda_MG, is_GP_dL, data_range=None, res=1000, GP_zero_point=False):
    if not is_GP_dL:
        Xi0, n = Lambda_MG
        dLGrid_at = at.concatenate([ at.constant([0.0]), dLfun_at( zGridGlobals_at, H0, Om, w0, Xi0, n )])
        return atinterp( r, dLGrid_at, at.concatenate([ at.constant([0.0]), zGridGlobals_at]) )  
    else:
        gp = Lambda_MG[0]
        
        dLGrid_EM_at = dLfun_at( zGridGlobals_at, H0, Om, w0, 1., 0 )

        log_distance_ratio, grad_log_distance_ratio, X_test, log_distance_ratio_grid = compute_gp_interp_dist_ratio( zGridGlobals_at, gp, name="f", res=res, data_range=data_range, GP_zero_point=GP_zero_point)
        
        dLGrid_at = at.exp(log_distance_ratio)*dLGrid_EM_at

        return dLGrid_at, log_distance_ratio, grad_log_distance_ratio


    
    
def log_j_at(z, Om, H0=70, dc=None, ):
    if dc is None:
        dc = dcfun_at(z, H0, Om)
    dc*=H0/c_light*1e03
    return at.log(4*PI)+2*at.log(dc)-at.log(Efun_at(z, Om=Om))


def log_dV_dz_at(z, Lambda_c, dc=None):

    H0, Om0, w0 = Lambda_c
    if dc is None:
        dc = dcfun_at(z, H0, Om0, w0)    
    res =  at.log(4*PI)+at.log(c_light)-at.log(H0)+2*at.log(dc)-at.log(Efun_at(z, Om0, w0))-3*at.log(10)

    return res

    
def log_ddL_dz(z, H0, Om0,  w0, Xi0, n, dc=None):
    
    # H0 in Mpc, dLs in Gpc
    
    if dc is None:
        dc = dcfun_at(z, H0, Om0,  w0, interp=False)*H0/c_light
    
    Xi = Xifun_at(z, Xi0, n)
    res = at.log( ( Xi -n*(1-Xi0)/(1+z)**n )* dc + Xi*c_light*(1+z)/(1e03*H0*Efun_at(z,Om0,  w0)) )  
        
    return res


# no dependence on H0 (as in Finke et.al.)
# dc * H0/c
def u_z_at(z, Om, w0):
    zz = at.linspace(0, z, 100).T
    E = Efun_at(zz, Om, w0)
    u = attrapzvec(1./E, zz)
    return u

# dV/dzdOm * H0^3/c^3/4pi
def log_j_z_at(z, Om, w0, ):
    E = Efun_at(z, Om, w0)
    u = u_z_at(z, Om, w0).T
    logj = 2*at.log(u) - at.log(E)
    return logj

def log_j_z_at_norm(z, Om, w0, zmax):
    logj = log_j_z_at(z, Om, w0)
    zz = at.geomspace(1e-7, zmax, 10000) # fixed (zmin, zmax)
    log_norm = at.log(attrapzvec(at.exp(log_j_z_at(zz, Om, w0)), zz))
    return logj - log_norm


##########################
####### Redshift distributions ########
##########################

def zdist_at(z, gamma, kappa):
  return z**2*(1+z)**gamma*at.exp(-z**2/kappa)


def p_z_at(z, gamma, kappa, normalize=True, zmax=15):
    
    if normalize:
        zz = at.linspace(0, zmax, steps=500).T
        pz =  zdist_at(zz, gamma, kappa)
        norm = attrapzvec(pz, zz)
    else:
        norm=1

    return  zdist_at(z, gamma, kappa)/norm



def zdist_at_MD(z, gamma, kappa, zp):
    return at.exp(log_zdist_at_MD(z, gamma, kappa, zp))


def log_zdist_at_MD(z, gamma, kappa, zp):
    lrate =  gamma*at.log1p(z)-at.log(1+((1+z)/(1+zp))**(gamma+kappa))
    lC0 = at.log( 1+(1+zp)**(-gamma-kappa))
    return lC0+lrate


def psi_MD(z, gamma, kappa, zp, normalize=True, zmax=15):
    
    if normalize:
        zz = at.linspace(0, zmax, steps=500).T
        pz =  zdist_at_MD(zz, gamma, kappa, zp)
        norm = attrapzvec(pz, zz)
    else:
        norm=1

    return  zdist_at_MD(z, gamma, kappa, zp)/norm



def p_z_MD(z, gamma, kappa, zp, Om, normalize=True, zmax=20, dc=None):
    
    psiz = psi_MD(z, gamma, kappa, zp, normalize=False, zmax=zmax)
    dVdz = at.exp(log_j_at(z, Om, H0=70, dc=dc, ))
    
    if normalize:
        zz = at.linspace(0, zmax, steps=500).T
        pz =  psi_MD(zz, gamma, kappa, zp, normalize=False,)*at.exp(log_j_at(zz, Om, H0=70, dc=None, ))/(1+zz)
        norm = attrapzvec(pz, zz)
    else:
        norm=1
        
    return psiz*dVdz/(1+z)/norm


def log_p_z_MD_unnorm(z, gamma, kappa, zp, Lambda_c, dc=None):
    #lC0 = at.log( 1+(1+zp)**(-gamma-kappa))
    
    log_psiz = log_psi_z_MD(z, gamma, kappa, zp) #gamma*at.log1p(z)-at.log(1+((1+z)/(1+zp))**(gamma+kappa))

    log_dVdz = log_dV_dz_at(z, Lambda_c, dc=dc )
    
    return log_psiz+log_dVdz


def log_psi_z_MD(z, gamma, kappa, zp):
    lC0 = at.log( 1+(1+zp)**(-gamma-kappa))
    log_psiz = lC0+gamma*at.log1p(z)-at.log(1+((1+z)/(1+zp))**(gamma+kappa))
    return log_psiz-at.log1p(z)


def log_p_z_PL_unnorm(z, gamma, H0, Om, w0, dc=None):
    log_psiz = gamma*at.log1p(z)
    log_dVdz = log_dV_dz_at(z, H0, Om, w0, dc=dc )

    return log_psiz+log_dVdz-at.log1p(z)


def log_p_z_PL_norm(z, gamma, H0, Om, w0, dc=None):
    log_psiz = gamma*at.log1p(z)
    log_dVdz = log_dV_dz_at(z, H0, Om, w0, dc=dc )

    zz = at.geomspace(1e-07, 500, steps=2000).T #at.linspace(0, 5, steps=2000).T
    pz = at.exp( gamma*at.log1p(zz)+log_dV_dz_at(zz, H0, Om, w0,dc=dc )-at.log1p(zz) )
    norm = attrapzvec(pz, zz)
    
    return log_psiz+log_dVdz-at.log1p(z)-at.log(norm)



#####################################################
# Gaussian processes for d
#####################################################



def min_max_scaler(X_raw, data_range, feature_range=(0, 1)):
    data_min, data_max = data_range
    feature_min, feature_max = feature_range

    X_std = (X_raw - data_min) / (data_max - data_min)
    X_scaled = X_std * (feature_max - feature_min) + feature_min
    return X_scaled



def min_max_inverse_transform(X_scaled, data_range, feature_range=(0, 1)):
    data_min, data_max = data_range
    feature_min, feature_max = feature_range

    X_std = (X_scaled - feature_min) / (feature_max - feature_min)
    X_raw = X_std * (data_max - data_min) + data_min
    return X_raw



U = at.as_tensor_variable(2.5)         # upper bound for σ with high probability
alpha = at.as_tensor_variable(0.01)    # small tail probability
lambda_ = at.log(1 / alpha) / U

alpha_ell = at.as_tensor_variable(0.01)
#L = at.as_tensor_variable(0.01)

d_GP = at.as_tensor_variable(1)


def frechet_logp_full(l, lambda_ell, d):
    return at.log(d * lambda_ell / 2) \
         - (d / 2 + 1) * at.log(l) \
         - lambda_ell * l ** (-d / 2)


def find_beta(L, alpha, p0=0.01):
    import scipy.stats as stats
    from scipy.optimize import bisect
    # Define function for root-finding: GammaCDF(L; alpha, beta) - p0 = 0
    def func(beta):
        return stats.gamma.cdf(L, a=alpha, scale=1/beta) - p0

    # beta must be positive, try searching between a small number and a large number
    beta_opt = bisect(func, 1e-6, 100)
    return beta_opt

def find_al(L, beta, p0=0.01):
    import scipy.stats as stats
    from scipy.optimize import bisect
    # Define function for root-finding: GammaCDF(L; alpha, beta) - p0 = 0
    def func(al):
        return stats.invgamma.cdf(L, a=al, scale=1/beta) - p0

    # beta must be positive, try searching between a small number and a large number
    alpha_opt = bisect(func, 1e-6, 100)
    return alpha_opt


#####################################################



def compute_gp_interp(X_list, gp, data_range, name="f", res=100,):
    """
    Evaluate GP on fixed grid, and interpolate function + gradient at input points.

    Args:
        X_list: list of two tensors (e.g., [X_data, X_inj])
        ℓ: lengthscale (symbolic)
        η: amplitude (symbolic)
        name: name of the GP random variable
        res: number of grid points

    Returns:
        zs_list: interpolated GP values at X_list
        grads_list: interpolated gradients at X_list
    """
    
        
    X_test = at.linspace(0, 1, res)[:, None]
    dx = X_test[1] - X_test[0]
    f_test =  at.cumsum(at.softplus( gp.prior( name, X_test, reparameterize=True) ))*dx

         
    # Interpolate values and gradients at requested points
    z_data, grad_data_scaled = atinterp( min_max_scaler( X_list[0], data_range=data_range), X_test, f_test, return_grad=True)
    z_inj, grad_inj_scaled   = atinterp( min_max_scaler( X_list[1], data_range=data_range), X_test, f_test, return_grad=True)
    grad_data = grad_data_scaled/ (dmax - dmin)
    grad_inj = grad_inj_scaled/ (dmax - dmin)
                
    
    return [z_data, z_inj], [grad_data, grad_inj], [X_test, f_test]



def compute_gp_interp_dist_ratio( z_grid, gp, data_range=None, name="f", res=1000, GP_zero_point=False ):

    if data_range is not None:
        zmin, zmax = data_range
    
        X_test = at.linspace(0, 1, res)[:, None]
        #dx = X_test[1] - X_test[0]
        X_eval = min_max_scaler(z_grid, data_range=data_range)
    else:
        # this is just a trick, since we need the gradient.
        X_test = at.concatenate( [ at.constant([0.0]), z_grid])[:, None] #at.linspace(0, z_grid.max(), res)[:, None]
        X_eval = z_grid #at.concatenate( [ at.constant([0.0]), z_grid])

    log_distance_ratio_grid = gp.prior( name, X=X_test, reparameterize=True) 

    # enforce distance ratio(z=0) = 1, i.e. log(distance_ratio)(z=0) = 0
    f_pseudo = log_distance_ratio_grid[0]
    pseudo_obs = pm.Normal( "dr_of_zero_constr", mu=f_pseudo, sigma=1e-6, observed=0.0 )
           
    # Interpolate values and gradients at requested points
    log_distance_ratio, grad_log_distance_ratio = atinterp( X_eval, X_test, log_distance_ratio_grid, return_grad=True)
    if data_range is not None:
        grad_log_distance_ratio /= (zmax - zmin)
                
    
    return log_distance_ratio, grad_log_distance_ratio, X_test[1:], log_distance_ratio_grid[1:]

    

#####################################################
#####################################################


##########################
####### Spin distributions ########
##########################


def logpdf_multivariate_trunc_2D( x1, x2, m1, m2, s1, s2, rho, l1, u1, l2, u2 ):

    
    where_inf =  ( x1 < l1 ) | ( x1 > u1 ) | ( x2 < l2 ) | ( x2 > u2 )

    mean = at.as_tensor_variable([m1, m2])
    x = at.as_tensor_variable([x1, x2]).T

    sEsP = rho*s1*s2 

    
    detC = s1**2* s2**2 - sEsP**2
    logdetC = at.log(detC)

    Cinv = at.zeros( (2, 2) )
    Cinv = at.set_subtensor( Cinv[0,0], s2**2/detC )
    Cinv = at.set_subtensor( Cinv[1,1], s1**2/detC )
    Cinv = at.set_subtensor( Cinv[0,1], -sEsP/detC )
    Cinv = at.set_subtensor( Cinv[1,0], -sEsP/detC )


    return at.where( where_inf, MIN, pm.logp( pm.MvNormal.dist( mu=mean, tau=Cinv, shape=(x.shape[0], 3)), x ))



def logpdf_default_spin(theta, lambdaBBHspin):

    chi1, chi2, cost1, cost2 = theta
    alphaChi, betaChi, zeta, sigmat = lambdaBBHspin
  
    normBeta =  at.gammaln(alphaChi) + at.gammaln(betaChi) - at.gammaln(alphaChi + betaChi)
        
    lpdfs1 = (alphaChi-1.0)*at.log(chi1) + (betaChi-1.0)*at.log1p(-chi1)
    lpdfs2 = (alphaChi-1.0)*at.log(chi2) + (betaChi-1.0)*at.log1p(-chi2)

    logpdfampl = lpdfs1 + lpdfs2 - 2*normBeta
   
  
    lpdfcos1_gauss = -0.5*(1.0-cost1)**2/(sigmat**2)-at.log(sigmat)-at.log(at.erf(at.sqrt(2.)/sigmat))
    lpdfcos2_gauss = -0.5*(1.0-cost2)**2/(sigmat**2)-at.log(sigmat)-at.log(at.erf(at.sqrt(2.)/sigmat))

    return logpdfampl + logsumexp( at.log(2.0)+at.log(zeta)-at.log(PI) + lpdfcos1_gauss + lpdfcos2_gauss, at.log(1.0-zeta)-at.log(4.0) )


def logpdf_default_spin_gauss(theta, lambdaBBHspin):

    chi1, chi2, cost1, cost2 = theta
    muChi, sigmaChi, zeta, sigmat = lambdaBBHspin
  
        
    lpdfs1 = truncGausslowerupper_at_lpdf_nonly(chi1, muChi, sigmaChi, xmin=0, xmax=1)
    lpdfs2 = truncGausslowerupper_at_lpdf_nonly(chi2, muChi, sigmaChi, xmin=0, xmax=1)

    logpdfampl = lpdfs1 + lpdfs2
   
  
    lpdfcos1_gauss = -0.5*(1.0-cost1)**2/(sigmat**2)-at.log(sigmat)-at.log(at.erf(at.sqrt(2.)/sigmat))
    lpdfcos2_gauss = -0.5*(1.0-cost2)**2/(sigmat**2)-at.log(sigmat)-at.log(at.erf(at.sqrt(2.)/sigmat))

    return logpdfampl + logsumexp( at.log(2.0)+at.log(zeta)-at.log(PI) + lpdfcos1_gauss + lpdfcos2_gauss, at.log(1.0-zeta)-at.log(4.0) )

    
        

##########################
####### Mass distributions ########
##########################


####### Uncorrelated flat ########


def logpdf_flat_sharp(theta, lambdaBBHmass):  
    m1, m2 = theta
    ml, mh = lambdaBBHmass

    return at.where( (m1>=ml) & (m1<=mh) & (m2>=ml) & (m2<=mh) & (m2<=m1), -2*at.log( mh-ml ) , MIN  )


def logpdf_flat(theta, lambdaBBHmass):  
    m1, m2 = theta
    ml, mh = lambdaBBHmass

    return -2*at.log( mh-ml ) + at.log(1-sigmoid(m1, mh, 0.05))+log_sigmoid(m1, ml, 0.05)+ at.log(1-sigmoid(m2, mh, 0.05))+log_sigmoid(m2, ml, 0.05)

    
    
    
####### Uncorrelated gaussian ########

def truncGausslower_at(x, loc, scale, xmin=0, ):    

    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    return at.where(x>xmin, 1./(at.sqrt(2.*PI)*scale)/(1.-Phialpha) * at.exp(-(x-loc)**2/(2*scale**2)) , 0.)


def truncGaussLowerUpper_at(x, loc, scale, xmin=0, xmax=1 ):    

    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    Phibeta = 0.5*(1.+at.erf((xmax-loc)/(at.sqrt(2.)*scale)))
    return at.where(  at.le(xmin,x) & at.le(x,xmax), 1./(at.sqrt(2.*PI)*scale)/(Phibeta-Phialpha) * at.exp(-(x-loc)**2/(2*scale**2)) , 0.)


def truncGausslowerupper_at_lpdf(x, loc, scale, xmin=0, xmax=1):    

    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    Phibeta = 0.5*(1.+at.erf((xmax-loc)/(at.sqrt(2.)*scale)))
    
    return at.where( (x>=xmin) & (x<=xmax), 
                    -at.log(scale)-0.5*at.log(2*PI)-at.log(Phibeta-Phialpha) + 0.5*(-(x-loc)**2/(scale**2)) , MIN)


def truncGausslowerupper_at_lpdf_nonly(x, loc, scale, xmin=0, xmax=1):    

    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    Phibeta = 0.5*(1.+at.erf((xmax-loc)/(at.sqrt(2.)*scale)))
    
    return -at.log(scale)-0.5*at.log(2*PI)-at.log(Phibeta-Phialpha) + 0.5*(-(x-loc)**2/(scale**2)) 

def truncGausslower_at_lpdf(x, loc, scale, xmin=0):    

    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    #Phibeta = 0.5*(1.+at.erf((xmax-loc)/(at.sqrt(2.)*scale)))
    
    return at.where( x>=xmin, 
                    -at.log(scale)-0.5*at.log(2*PI)-at.log(1.-Phialpha) + 0.5*(-(x-loc)**2/(scale**2)) , MIN)


def double_gauss_norm(mu, sigma):
    z = -mu / sigma
    C = 0.5 * (1 + at.erf(z / at.sqrt(2)))
    return 0.5 - C + 0.5 * C**2


def logpdf_gauss_single(x, loc, scale, xmin=0):  
    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    return at.where(x>xmin, at.log(1./(at.sqrt(2.*PI)*scale)/(1.-Phialpha)) + -(x-loc)**2/(2*scale**2) , MIN )
    #return -at.log(scale)-0.5*at.log(2.*PI) -0.5*(x-loc)**2/(scale**2)


def logpdf_gauss(theta, lambdaBBHmass):  
    m1, m2 = theta
    loc, scale = lambdaBBHmass
    
    return logpdf_gauss_single(m1, loc, scale, xmin=0) + logpdf_gauss_single(m2, loc, scale, xmin=0) -at.log(double_gauss_norm(loc, scale))

def logpdf_gauss_cond(theta, lambdaBBHmass):  
    m1, m2 = theta
    loc, scale = lambdaBBHmass
    
    logpdfm1 = truncGausslower_at_lpdf( m1, xmin=0., loc=loc, scale=scale)
    logpdfm2 = truncGausslowerupper_at_lpdf( m2, xmin=0., xmax=m1, loc=loc, scale=scale)
    return logpdfm1+logpdfm2



####### Power Law + Peak ########


def truncated_power_law(m, alpha, ml, mh):
        
        where_compute = (ml < m) & (m < mh )

        result = at.where(where_compute, at.log(m)*(-alpha), MIN)
        
        return result



def logpdf_PLP(theta, lambdaBBHmass, pairing=True):
    
        m1, m2 = theta
        lambdaPeak, alpha, beta, deltam, ml, mh, muMass, sigmaMass = lambdaBBHmass
                
        where_compute = (m2 <= m1) & (ml <= m2) & (m1 <= mh ) 

        lpdfm1 = at.where(where_compute, logpdfm1_PLP(m1,  lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass ), MIN )
        lpdfm2 = at.where(where_compute,logpdfm2_PLP(m2, beta, deltam, ml), MIN )
        if pairing:
            lC = at.where(where_compute, logC_PLP(m1, beta, deltam,  ml, ), MIN )
        ln = at.where(where_compute, logNorm_PLP( lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass), MIN )
        
        return at.where( where_compute, lpdfm1+lpdfm2+lC-ln, MIN )
        
        

    
    
def logS_PLP(m, deltam, ml,):
        maskL = m <= ml 
        maskU = m >= (ml + deltam) 
        
        maskM = ~(maskL | maskU)
        
        s = at.where(maskL, MIN, at.as_tensor_variable(0.)  )
        
        s1 = at.where( maskM, at.log(1/(1+ at.exp(deltam/(m-ml) + deltam/(m-ml - deltam) ) )) , s  )
        
        return s1   



def logpdfm1_PLP(m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass):

    where_compute = (ml <= m) & (m <= mh )

    norm = norm_truncated_pl_num(alpha, ml, mh)
    trunc_component = at.where(where_compute, 1./m**alpha/norm, MIN)
    gauss_component = at.where(where_compute, at.exp(-(m-muMass)**2/(2*sigmaMass**2))/(at.sqrt(2*PI)*sigmaMass), MIN)

    lS = logS_PLP(m, deltam, ml) 
        
    result =  at.where( where_compute, at.log( (1-lambdaPeak)*trunc_component+lambdaPeak*gauss_component)+lS
                       , MIN )
    return result

    

def logpdfm2_PLP(m2, beta, deltam, ml):

    where_compute = (ml<= m2) #& (~where_nan)
    res = at.log(m2)*(beta)+logS_PLP(m2, deltam, ml)
    result = at.where( where_compute, res, MIN )
           
    return result

        

def logC_PLP( m, beta, deltam, ml, res=100):
    '''
    Gives inverse log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
    '''
    

    max_m = at.as_tensor_variable(500)
  
    
    x2 = at.linspace(ml, 15, res )
    x3 = at.linspace(15.01, 100, res )
    x4 = at.linspace(101.1, max_m, int(res/2) )
    xx = at.concatenate([ x2, x3, x4 ] )

    p2 = at.exp(logpdfm2_PLP( xx , beta, deltam, ml))
    cdf = atcumtrapz(p2, xx, )
    itr = atinterp( m, xx[1:], at.log(cdf))
    return itr




    

def logNorm_PLP( lambdaPeak, alpha,  deltam, ml, mh, muMass, sigmaMass  , res=1000 ):
    
    '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

    '''
    
    ms = at.linspace(ml, mh, res)
    ps = at.exp( logpdfm1_PLP( ms , lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass  ))
    p1 = at.where( (ms>=ml) & (ms<=mh), ps, 0.)
    return at.log(attrapzvec(p1,ms))

            
    
            
def norm_truncated_pl_num(alpha, mmin, mmax):

    return 1/(1-alpha)*(mmax**(1-alpha)-mmin**(1-alpha))




####### Power Law + Peak smooth edges , LVK low-end ########



def logpdf_PLP_reg(theta, lambdaBBHmass,  pairing=True):
    
        m1, m2 = theta
        lambdaPeak, alpha, beta, deltam, ml, mh, muMass, sigmaMass = lambdaBBHmass
                

        lpdfm1 = logpdfm1_PLP_reg(m1, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass )
        lpdfm2 = logpdfm2_PLP_reg(m2, beta, deltam, ml)
        ln = logNorm_PLP_reg( lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass)
        lpdf = lpdfm1+lpdfm2-ln
        if pairing:
            return lpdf-logC_PLP_reg(m1, beta, deltam,  ml) 
        else:
            return lpdf

        


def logpdfm1_PLP_reg(m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, sl=0.05, sh=0.05):

    return logpdfm1_PLP_noreg(m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass)+at.log(1-sigmoid(m, mh, sh))+log_sigmoid(m, ml, sl)


def logpdfm1_PLP_noreg(m, lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass):

    norm = norm_truncated_pl_num(alpha, ml, mh)
    trunc_component =  1./(m**alpha)/norm
    gauss_component = at.exp(-(m-muMass)**2/(2*sigmaMass**2))/(at.sqrt(2*PI)*sigmaMass)
        
    result =  at.log( (1-lambdaPeak)*trunc_component+lambdaPeak*gauss_component)+logS_PLP(m, deltam, ml) 
 
    return result

def logpdfm2_PLP_reg(m, beta, deltam, ml, sig_l=0.05):
    return logpdfm2_PLP_noreg(m, beta, deltam, ml,)+log_sigmoid(m, ml, sig_l) 


def logpdfm2_PLP_noreg(m, beta, deltam, ml,):
    return beta*at.log(m)+logS_PLP(m, deltam, ml)
           
        

def logC_PLP_reg( m, beta, deltam, ml, res=1000):
    '''
    Gives log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
    '''

    #max_m = at.as_tensor_variable(500)
  
   
    # lower edge
    #ms1 = at.linspace(ml, 15, res)
    
    # before gaussian peak
    #ms2 = at.linspace( 15.1, 25, res )
    
    # around gaussian peak
    #ms3= at.linspace( 25.1, 40, res)
    
    # after gaussian peak
    #ms4 = at.linspace(40.1, 100, res )

    # after gaussian peak
    #ms5 = at.linspace(100.1, max_m, int(res/2) )
    
    #xx=at.concatenate([ms1,ms2, ms3, ms4, ms5] )

    xx = at.linspace(ml, 500, res)
    
    p2 = at.exp(logpdfm2_PLP_noreg( xx , beta, deltam, ml))
    cdf = atcumtrapz(p2, xx, )
    itr = atinterp( m, xx[1:], at.log(cdf))
    return itr


def logNorm_PLP_reg( lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, res=1000):
    
    '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

    '''
     
            
    # lower edge
    #ms1 = at.linspace(ml, 15, res)
    
    # before gaussian peak
    #ms2 = at.linspace( 15.1, 25, res )
    
    # around gaussian peak
    #ms3= at.linspace( 25.1, 40, res)
    
    # after gaussian peak
    #ms4 = at.linspace(40.1, mh, int(res/2) )
    
    #ms=at.concatenate([ms1,ms2, ms3, ms4] )
    ms = at.linspace(ml, mh, res)
    
    ps = at.exp( logpdfm1_PLP_noreg( ms , lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass  ))
    return at.log(attrapzvec(ps,ms))



####### double Power Law + double Peak  LVK low-end ########


def log_broken_power_law_pdf(m1, alpha1, alpha2, mb, m1_low, m_high, sh=0.05, sl=0.05):
    """
    Log of the broken power-law PDF (JAX-compatible, log-space)
    """
    
    
    # Compute log normalization constant
    norm1 = (m_high * (m_high / mb) ** (-alpha2) - mb) / (-alpha2 + 1)
    norm2 = (mb - m1_low * (m1_low / mb) ** (-alpha1)) / (-alpha1 + 1)
    log_N = at.log(norm1 + norm2)

    # log(pdf) in each regime
    log_val1 = -alpha1 * at.log(m1 / mb)
    log_val2 = -alpha2 * at.log(m1 / mb)

  
    # Smooth weight function (sigmoid transition)
    w = sigmoid( -m1, -mb, epsilon)
    #1.0 / (1.0 + at.exp((m1 - mb) / epsilon))


    # Use log-sum-exp trick to compute:
    # log(w * exp(log_val1) + (1-w) * exp(log_val2))
    log_mix_val = logsumexp(
        at.log(w) + log_val1,
        at.log1p(-w) + log_val2
    )

    # Outside bounds, set log-prob to -inf
    #valid_mask = (m1 >= m1_low) & (m1 < m_high)
    #return at.where(valid_mask, log_mix_val - log_N, MIN)
    return log_mix_val - log_N + at.log(1-sigmoid(m1, m_high, sh)) + log_sigmoid(m1, m1_low, sl)


def logpdfm1_DPLDP(
    m1, alpha1, alpha2, mb,
    mu1, sigma1, mu2, sigma2,
    m1_low, m_high, delta_m1,
    lambda0, lambda1,
    ):
    """
    Log of the mixture model. Assumes other components return log-probabilities.
    """
    log_lambda0 = at.log(lambda0)
    log_lambda1 = at.log(lambda1)
    log_lambda2 = at.log1p(-lambda0 - lambda1)  # log(1 - λ0 - λ1)

    log_ppl = log_broken_power_law_pdf(m1, alpha1, alpha2, mb, m1_low, m_high)
    log_pnorm1 = truncGausslowerupper_at_lpdf_nonly(m1, mu1, sigma1, xmin=m1_low, xmax=m_high) 
    log_pnorm2 = truncGausslowerupper_at_lpdf_nonly(m1, mu2, sigma2, xmin=m1_low, xmax=m_high) 
    log_S = logS_PLP(m1, delta_m1, m1_low,)

    # logsumexp of the weighted logs
    log_mix = logsumexp(
        logsumexp(log_lambda0 + log_ppl, log_lambda1 + log_pnorm1),
        log_lambda2 + log_pnorm2
    )

    return log_mix + log_S


def logpdf_DPLDP(theta, lambdaBBHmass):
    
        m1, m2 = theta
        alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, beta, m2_low, delta_m2 = lambdaBBHmass
                

        lpdfm1 = logpdfm1_DPLDP( m1, alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1,)
    
        lpdfm2 = logpdfm2_PLP_reg(m2, beta, delta_m2, m2_low)
        
        lC = logC_DPLDP(m1, beta, delta_m2,  m2_low) 
        ln = logNorm_DPLDP(  alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1)

        lpdf = lpdfm1+lpdfm2-lC-ln

        return  lpdf
        
     

def logC_DPLDP( m, beta, deltam, ml, res=500):
    '''
    Gives log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
    '''

    max_m = at.as_tensor_variable(500)
  
   
    # lower edge
    ms1 = at.linspace(ml, 100, res)

    # upper edge
    ms5 = at.linspace(100.1, max_m, 100 )
    
    xx=at.concatenate([ms1, ms5] )
    
    p2 = at.exp(logpdfm2_PLP_noreg( xx , beta, deltam, ml))
    
    cdf = atcumtrapz(p2, xx, )
    itr = atinterp( m, xx[1:], at.log(cdf))
    
    return itr


def logNorm_DPLDP( alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, res=1000):
    
    '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

    '''
    # lower edge
    ms1 = at.linspace(ml, 100, res)

    # after max
    ms5 = at.linspace(100.1, m_high, 100 )
    
    ms=at.concatenate([ms1, ms5] )
            
    #ms = at.linspace(m1_low, m_high, res)
    
    ps = at.exp( logpdfm1_DPLDP( ms , alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1  ))
    return at.log(attrapzvec(ps,ms))
