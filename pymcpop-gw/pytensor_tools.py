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

from jax.numpy import array
from jax.numpy import concatenate
from jax.numpy import ones
from jax.numpy import zeros

c_light = 299792458*1e-03
c_light_at = at.as_tensor_variable(c_light)
MIN = at.as_tensor_variable(-np.inf)
INF = at.as_tensor_variable(np.inf)
 

zGridGlobals_at = at.sort(at.unique(at.concatenate([ at.logspace(start=-100, end=-15, base=10, steps=5), at.logspace(start=-30, end=-4, base=10, steps=30), 
                     #at.linspace(start=1.1e-03, end=10, steps=50),
                     at.logspace(start=-4, end=1, base=10, steps=100), 
                     at.logspace(start=1, end=2, base=10, steps=20), at.logspace(start=2, end=5, base=10, steps=5) ])))

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

def log_sigmoid(x, m, sig):
    return -at.log1p(at.exp(-(x-m)/sig))

def sigmoid(x, m, sig):
    return 1/(1+at.exp((-(x-m)/sig)))

##########################
####### Interpolators and integrators ########
##########################



def atinterp(x, xs, ys):

  idxs = at.searchsorted(xs, x,  side='left', sorter=None)

  xl = xs[idxs-1]
  yl = ys[idxs-1]
  xh = xs[idxs]
  yh = ys[idxs]

  r = (x-xl)/(xh-xl);

  return r*yh + (1.0-r)*yl;


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


def attrapzvec(y, x,  axis=-1):

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


def z_from_dL_at(r, H0, Om, w0, Xi0, n ):
    dLGrid_at = dLfun_at( zGridGlobals_at, H0, Om, w0, Xi0, n )
    z2dL = atinterp( r, dLGrid_at, zGridGlobals_at ) 
    return z2dL 


    
    
def log_j_at(z, Om, H0=70, dc=None, ):
    if dc is None:
        dc = dcfun_at(z, H0, Om)
    dc*=H0/c_light*1e03
    return at.log(4*PI)+2*at.log(dc)-at.log(Efun_at(z, Om=Om))

def log_dV_dz_at(z, H0, Om0, w0, dc=None):
    if dc is None:
        dc = dcfun_at(z, H0, Om0, w0)    
    res =  at.log(4*PI)+at.log(c_light)-at.log(H0)+2*at.log(dc)-at.log(Efun_at(z, Om0, w0))-3*at.log(10)
    return res
    
def log_ddL_dz(z, H0, Om0,  w0, Xi0, n, dL=None):
    
    # H0 in Mpc, dLs in Gpc
    
    if dL is None:
        dc = dcfun_at(z, H0, Om0,  w0, Xi0, n, interp=False)*H0/c_light
    else:
        dc = dL/(1+z)
    
    Xi = Xifun_at(z, Xi0, n)
    res = at.log( ( Xi -n*(1-Xi0)/(1+z)**n )* dc + Xi*c_light*(1+z)/(1e03*H0*Efun_at(z,Om0,  w0)) )  
        
    return res



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


def log_p_z_MD_unnorm(z, gamma, kappa, zp, H0, Om, w0, dc=None):
    #lC0 = at.log( 1+(1+zp)**(-gamma-kappa))
    
    log_psiz = log_psi_z_MD(z, gamma, kappa, zp) #gamma*at.log1p(z)-at.log(1+((1+z)/(1+zp))**(gamma+kappa))
    
    log_dVdz = log_dV_dz_at(z, H0, Om, w0, dc=dc )

    return log_psiz+log_dVdz


def log_psi_z_MD(z, gamma, kappa, zp):

    log_psiz = gamma*at.log1p(z)-at.log(1+((1+z)/(1+zp))**(gamma+kappa))

    return log_psiz-at.log1p(z)


def log_p_z_PL_unnorm(z, gamma, H0, Om, w0, dc=None):
    log_psiz = gamma*at.log1p(z)
    log_dVdz = log_dV_dz_at(z, H0, Om, w0, dc=dc )

    return log_psiz+log_dVdz-at.log1p(z)


def log_p_z_PL_norm(z, gamma, H0, Om, w0, dc=None):
    log_psiz = gamma*at.log1p(z)
    log_dVdz = log_dV_dz_at(z, H0, Om, w0, dc=dc )

    zz = at.geomspace(1e-07, 500, steps=2000).T #at.linspace(0, 5, steps=2000).T
    pz = at.exp( gamma*at.log1p(zz)+log_dV_dz_at(zz, H0, Om, w0,dc=None )-at.log1p(zz) )
    norm = attrapzvec(pz, zz)
    
    return log_psiz+log_dVdz-at.log1p(z)-at.log(norm)






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

def truncGausslower_at_lpdf(x, loc, scale, xmin=0):    

    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    #Phibeta = 0.5*(1.+at.erf((xmax-loc)/(at.sqrt(2.)*scale)))
    
    return at.where( x>=xmin, 
                    -at.log(scale)-0.5*at.log(2*PI)-at.log(1.-Phialpha) + 0.5*(-(x-loc)**2/(scale**2)) , MIN)



def logpdf_gauss_single(x, loc, scale, xmin=0):  
    #Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    #return at.where(x>xmin, at.log(1./(at.sqrt(2.*PI)*scale)/(1.-Phialpha)) + -(x-loc)**2/(2*scale**2) , MIN )
    return -at.log(scale)-0.5*at.log(2.*PI) -0.5*(x-loc)**2/(scale**2)



def logpdf_gauss(theta, lambdaBBHmass):  
    m1, m2 = theta
    loc, scale = lambdaBBHmass
    
    return logpdf_gauss_single(m1, loc, scale, xmin=0) + logpdf_gauss_single(m2, loc, scale, xmin=0)

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



def logpdf_PLP(theta, lambdaBBHmass):
    
        m1, m2 = theta
        lambdaPeak, alpha, beta, deltam, ml, mh, muMass, sigmaMass = lambdaBBHmass
                
        where_compute = (m2 <= m1) & (ml <= m2) & (m1 <= mh ) 

        lpdfm1 = at.where(where_compute, logpdfm1_PLP(m1,  lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass ), MIN )
        lpdfm2 = at.where(where_compute,logpdfm2_PLP(m2, beta, deltam, ml), MIN )
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



def logpdf_PLP_reg(theta, lambdaBBHmass):
    
        m1, m2 = theta
        lambdaPeak, alpha, beta, deltam, ml, mh, muMass, sigmaMass = lambdaBBHmass
                

        lpdfm1 = logpdfm1_PLP_reg(m1,  lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass )
        lpdfm2 = logpdfm2_PLP_reg(m2, beta, deltam, ml)
        
        lC = logC_PLP_reg(m1, beta, deltam,  ml) 
        ln = logNorm_PLP_reg( lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass)

        lpdf = lpdfm1+lpdfm2-lC-ln

        return  lpdf
        


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
           
        

def logC_PLP_reg( m, beta, deltam, ml, res=200):
    '''
    Gives log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
    '''

    max_m = at.as_tensor_variable(500)
  
   
    # lower edge
    ms1 = at.linspace(ml, 15, res)
    
    # before gaussian peak
    ms2 = at.linspace( 15.1, 25, res )
    
    # around gaussian peak
    ms3= at.linspace( 25.1, 40, res)
    
    # after gaussian peak
    ms4 = at.linspace(40.1, 100, res )

    # after gaussian peak
    ms5 = at.linspace(100.1, max_m, int(res/2) )
    
    xx=at.concatenate([ms1,ms2, ms3, ms4, ms5] )

    
    p2 = at.exp(logpdfm2_PLP_noreg( xx , beta, deltam, ml))
    cdf = atcumtrapz(p2, xx, )
    itr = atinterp( m, xx[1:], at.log(cdf))
    return itr


def logNorm_PLP_reg( lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass, res=200):
    
    '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

    '''
    
    
            
    # lower edge
    ms1 = at.linspace(ml, 15, res)
    
    # before gaussian peak
    ms2 = at.linspace( 15.1, 25, res )
    
    # around gaussian peak
    ms3= at.linspace( 25.1, 40, res)
    
    # after gaussian peak
    ms4 = at.linspace(40.1, mh, int(res/2) )
    
    ms=at.concatenate([ms1,ms2, ms3, ms4] )
    
    
    ps = at.exp( logpdfm1_PLP_noreg( ms , lambdaPeak, alpha, deltam, ml, mh, muMass, sigmaMass  ))
    return at.log(attrapzvec(ps,ms))

         
