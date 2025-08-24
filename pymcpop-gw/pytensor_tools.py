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
 
#if int(pytensor.__version__.split('.')[1])>25: #=='2.30.3':
try:
        zGridGlobals_at = at.sort(at.unique(at.concatenate([ at.logspace(start=-100, stop=-15, base=10, steps=50), at.logspace(start=-30, stop=-4, base=10, steps=100), 
                     #at.linspace(start=1.1e-03, end=10, steps=50),
                     at.logspace(start=-4, stop=1, base=10, steps=1000), 
                     at.logspace(start=1, stop=2, base=10, steps=100), at.logspace(start=2, stop=5, base=10, steps=50) ])))

except:
    
    zGridGlobals_at = at.sort(at.unique(at.concatenate([ at.logspace(start=-100, end=-15, base=10, steps=50), at.logspace(start=-30, end=-4, base=10, steps=100), 
                     #at.linspace(start=1.1e-03, end=10, steps=50),
                     at.logspace(start=-4, end=1, base=10, steps=1000), 
                     at.logspace(start=1, end=2, base=10, steps=100), at.logspace(start=2, end=5, base=10, steps=50) ])))

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

def safe_sigmoid(x, x0, eps):
    s = 1.0 / (1.0 + at.exp(-(x - x0) / eps))
    return at.clip(s, 1e-15, 1 - 1e-15)

def stick_breaking(beta):
    portion_remaining = at.concatenate([[1], at.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

##########################
####### Interpolators and integrators ########
##########################

def meshgrid_at(x, y):
    x = at.as_tensor_variable(x)
    y = at.as_tensor_variable(y)
    nx = x.shape[0]
    ny = y.shape[0]

    X = at.alloc(x, nx, ny)      # Broadcast x along columns
    Y = at.alloc(y, nx, ny).T    # Broadcast y along rows, then transpose

    return X.T, Y.T

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



class TrapzOp(Op):
    itypes = [at.dmatrix, at.dmatrix]   # y: (M,N), x: (1,N)
    otypes = [at.dvector]               # output: (M,)

    def __init__(self, axis=1):
        self.axis = axis

    def perform(self, node, inputs, outputs):
        y, x = inputs                    # y: (M,N), x: (1,N)
        # Broadcast x to (M,N)
        x_b = np.broadcast_to(x, y.shape)
        out = np.trapz(y, x_b, axis=self.axis)  # (M,)
        outputs[0][0] = out

    def grad(self, inputs, output_grads):
        y, x = inputs                    # y: (M,N), x: (1,N)
        (gz,) = output_grads             # (M,)

        class JaxTrapzGrad(Op):
            itypes = [at.dmatrix, at.dmatrix]   # y: (M,N), x: (1,N)
            otypes = [at.dmatrix, at.dmatrix]   # dy: (M,N), dx: (1,N)

            def perform(inner_self, node, inputs, outputs):
                yv, xv = inputs                  # yv: (M,N), xv: (1,N)
                # Broadcast x to (M,N)
                xv_b = jnp.broadcast_to(xv, yv.shape)

                def trapz_sum(y_, x_):
                    return jnp.sum(jnp.trapz(y_, x_, axis=1))

                dy = jax.grad(trapz_sum, argnums=0)(yv, xv_b)  # (M,N)
                dx_full = jax.grad(trapz_sum, argnums=1)(yv, xv_b)  # (M,N)

                # Sum dx over M dimension to get (N,)
                dx = jnp.sum(dx_full, axis=0)   # (N,)
                dx = dx[None, :]                # (1,N)

                outputs[0][0] = np.asarray(dy)
                outputs[1][0] = np.asarray(dx)

        jax_grad_op = JaxTrapzGrad()
        dy, dx = jax_grad_op(y, x)

        return [gz[:, None] * dy, gz[:, None] * dx]  # dx broadcasted

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


def log_p_z_MD_unnorm(z, gamma, kappa, zp, H0, Om, w0, dc=None):
    #lC0 = at.log( 1+(1+zp)**(-gamma-kappa))
    
    log_psiz = log_psi_z_MD(z, gamma, kappa, zp) #gamma*at.log1p(z)-at.log(1+((1+z)/(1+zp))**(gamma+kappa))
    
    log_dVdz = log_dV_dz_at(z, H0, Om, w0, dc=dc )

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


def truncGausslower_at_logpdf(x, loc, scale, xmin=0):  
    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    return at.where(x>xmin, at.log(1./(at.sqrt(2.*PI)*scale)/(1.-Phialpha)) + -(x-loc)**2/(2*scale**2) , MIN )
    #return -at.log(scale)-0.5*at.log(2.*PI) -0.5*(x-loc)**2/(scale**2)

def truncGausslower_at_pdf(x, loc, scale, xmin=0):  
    Phialpha = 0.5*(1.+at.erf((xmin-loc)/(at.sqrt(2.)*scale)))
    return at.where(x>xmin, at.exp( -(x-loc)**2/(2*scale**2))/(at.sqrt(2.*PI)*scale)/(1.-Phialpha) , at.as_tensor_variable(0.) )
    #return -at.log(scale)-0.5*at.log(2.*PI) -0.5*(x-loc)**2/(scale**2)



def logpdf_gauss(theta, lambdaBBHmass):  
    m1, m2 = theta
    loc, scale = lambdaBBHmass
    
    return truncGausslower_at_logpdf(m1, loc, scale, xmin=0) + truncGausslower_at_logpdf(m2, loc, scale, xmin=0) -at.log(double_gauss_norm(loc, scale))

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


def log_broken_power_law_DPLDP_pdf(m1, alpha1, alpha2, mb, m1_low, m_high, sh=0.05, sl=0.05, epsilon=0.01):
    """
    Log of the broken power-law PDF 
    """    
    
    # Compute log normalization constant
    norm1 = (m_high * (m_high / mb) ** (-alpha2) - mb) / (-alpha2 + 1)
    norm2 = (mb - m1_low * (m1_low / mb) ** (-alpha1)) / (-alpha1 + 1)
    log_N = at.log(norm1 + norm2)


    # log(pdf) in each regime
    log_val1 = -alpha1 * at.log(m1 / mb)
    log_val2 = -alpha2 * at.log(m1 / mb)

  
    # Smooth weight function (sigmoid transition)
    w = safe_sigmoid( -m1, -mb, epsilon)

    # Use log-sum-exp to compute:
    # log(w * exp(log_val1) + (1-w) * exp(log_val2))
    log_mix_val = logsumexp(
        at.log(w) + log_val1,
        at.log1p(-w) + log_val2
    )

    
    s1 = at.log1p(-sigmoid(m1, m_high, sh))
    s2 = log_sigmoid(m1, m1_low, sl)
    
    return log_mix_val - log_N + s1 + s2


def logpdfm1_DPLDP(
    m1, alpha1, alpha2, mb,
    mu1, sigma1, mu2, sigma2,
    m1_low, m_high, delta_m1,
    lambda0, lambda1,
    epsilon
    ):
    """
    Log of the mixture model. Assumes other components return log-probabilities.
    """
    log_lambda0 = at.log(lambda0)
    log_lambda1 = at.log(lambda1)
    log_lambda2 = at.log1p(-lambda0 - lambda1)  # log(1 - λ0 - λ1)

    log_ppl = log_broken_power_law_DPLDP_pdf(m1, alpha1, alpha2, mb, m1_low, m_high, epsilon=epsilon)
    log_pnorm1 = truncGausslowerupper_at_lpdf(m1, mu1, sigma1, xmin=m1_low, xmax=m_high) 
    log_pnorm2 = truncGausslowerupper_at_lpdf(m1, mu2, sigma2, xmin=m1_low, xmax=m_high) 
    log_S = logS_PLP(m1, delta_m1, m1_low,)

    # logsumexp of the weighted logs
    log_mix = logsumexp(
        logsumexp(log_lambda0 + log_ppl, log_lambda1 + log_pnorm1),
        log_lambda2 + log_pnorm2
    )

    return log_mix + log_S


def logpdf_DPLDP(theta, lambdaBBHmass, force_m2_less_than_m1=False):
    
        m1, m2 = theta
        alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, beta, m2_low, delta_m2, epsilon = lambdaBBHmass
                

        lpdfm1 = logpdfm1_DPLDP( m1, alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, epsilon)
    
        lpdfm2 = logpdfm2_PLP_reg(m2, beta, delta_m2, m2_low)
        
        lC = logC_DPLDP(m1, beta, delta_m2,  m2_low) 
    
        ln = logNorm_DPLDP(  alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, epsilon)

        lpdf = lpdfm1 + lpdfm2 -lC -ln

        if force_m2_less_than_m1:
            eval = at.and_(at.and_(m2 <= m1, m2 > 0), m1 > 0)
            return at.where(eval, lpdf, MIN)
        else:
            return lpdf
        
     

def logC_DPLDP( m, beta, deltam, m2_low, res=1000):
    '''
    Gives log integral of  p(m1, m2) dm2 (i.e. log C(m1) in the LVC notation )
    '''

    max_m = at.as_tensor_variable(500)
  
   
    # lower edge
    ms1 = at.linspace(m2_low, 100, res)

    # upper edge
    ms5 = at.linspace(100.1, max_m, 100 )
    
    xx=at.concatenate([ms1, ms5] )
    
    p2 = at.exp(logpdfm2_PLP_noreg( xx , beta, deltam, m2_low))
    
    cdf = atcumtrapz(p2, xx, )
    itr = atinterp( m, xx[1:], at.log(cdf))
    
    return itr


def logNorm_DPLDP( alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, epsilon, res=1000):
    
    '''
        Gives log integral of  p(m1, m2) dm1 dm2 (i.e. total normalization of mass function )

    '''
    # lower edge
    ms1 = at.linspace(m1_low, 100, res)

    # after max
    ms5 = at.linspace(100.1, m_high, 100 )
    
    ms = at.sort( at.unique( at.concatenate([ms1, ms5] ) ))
            
    lpdf = logpdfm1_DPLDP( ms , alpha1, alpha2, mb, mu1, sigma1, mu2, sigma2, m1_low, m_high, delta_m1, lambda0, lambda1, epsilon  )
    ps = at.exp( lpdf)
    
    return at.log( attrapzvec(ps, ms) )



####### FullPop-4.0 ########


def log1mexp(x):
    """
    Numerically stable log(1 - exp(x)) for x < 0
    """
    log2 = at.log(2.0)
    return at.switch(
        x <= -log2,
        at.log1p(-at.exp(x)),
        at.log(-at.expm1(x))
    )

# --- Broken power law ---
def log_broken_power_law_FP_pdf(m, λ_p, norm=False, maxm=100, epsilon=0.1):
    m_NSmax, m_BHmin, α1, α2, α_dip = λ_p

    log_region1 = α1 * at.log(m)
    log_region2 = (α1 - α_dip) * at.log(m_NSmax) + α_dip * at.log(m)
    log_region3 = (α1 - α_dip) * at.log(m_NSmax) + (α_dip - α2) * at.log(m_BHmin) + α2 * at.log(m)

    s1 = safe_sigmoid(m, m_NSmax, epsilon)
    s2 = safe_sigmoid(m, m_BHmin, epsilon)

    log_part1 = log_region1 + at.log1p(-s1)
    log_part2 = log_region2 + at.log(s1) + at.log1p(-s2)
    log_part3 = log_region3 + at.log(s2)

    result = logsumexp(
        logsumexp(log_part1, log_part2),
        log_part3
    )

    if norm:
        mgrid = at.logspace(at.log10(1.0), at.log10(maxm), 2000)
        log_vals = log_broken_power_law_FP_pdf(mgrid, λ_p, norm=False, maxm=maxm)
        vals = at.exp(log_vals)
        norm_factor = attrapzvec(vals, mgrid)
        return result - at.log(norm_factor)
    else:
        return result


def log_l_filter_at(m, m0, η):
    log_x = η * (at.log(m0) - at.log(m))
    return -logsumexp(0.0, log_x)

def log_h_filter_at(m, m0, η):
    log_x = η * (at.log(m0) - at.log(m))
    return log_x - logsumexp(0.0, log_x)


def log_notch_filter_at(m, γlow, γhigh, ηlow, ηhigh, A):
    log_l = log_l_filter_at(m, γlow, ηlow)
    log_h = log_h_filter_at(m, γhigh, ηhigh)
    log_prod = log_l + log_h + at.log(A)
    return log1mexp(log_prod)  # safe: log(1 - A * l * h)

def log_f_q_FP(q, m2, Λ_q, epsilon=0.1):
    beta_low, beta_high, m_break = Λ_q
    s = safe_sigmoid(m2, m_break, epsilon)

    log_s = at.log(s)
    log1m_s = at.log1p(-s)
    log_q = at.log(q)

    log_term1 = log1m_s + beta_low * log_q
    log_term2 = log_s + beta_high * log_q

    return logsumexp(log_term1, log_term2)


def log_B_notches(m, λ_b):
    γlow_1, γhigh_1, ηlow_1, ηhigh_1, A1 = λ_b[0:5]
    γlow_2, γhigh_2, ηlow_2, ηhigh_2, A2 = λ_b[5:10]
    η_NSmin, m_NSmin = λ_b[10:12]
    η_BHmax, m_BHmax = λ_b[12:14]

    log_n1 = log_notch_filter_at(m, γlow_1, γhigh_1, ηlow_1, ηhigh_1, A1)
    log_n2 = log_notch_filter_at(m, γlow_2, γhigh_2, ηlow_2, ηhigh_2, A2)
    log_l = log_l_filter_at(m, m_NSmin, η_NSmin)
    log_h = log_h_filter_at(m, m_BHmax, η_BHmax)

    return log_l + log_h + log_n1 + log_n2


def logpdfm1_FP(m, λ_m, norm=False):
    m_BHmax = λ_m[11:][-1]

    def my_logp(m, λ_m):
        c1, c2, μ1, σ1, μ2, σ2 = λ_m[0:6]
        λ_p = λ_m[6:11]
        λ_b = λ_m[11:]
        _, m_NSmin = λ_b[10:12]
        _, m_BHmax = λ_b[12:14]

        log_G1 = truncGausslowerupper_at_lpdf(m, μ1, σ1, xmin=m_NSmin, xmax=m_BHmax)
        log_G2 = truncGausslowerupper_at_lpdf(m, μ2, σ2, xmin=m_NSmin, xmax=m_BHmax)

        logP = log_broken_power_law_FP_pdf(m, λ_p, norm=False, maxm=m_BHmax * 1.1)
        logB = log_B_notches(m, λ_b)

        log_terms1 = 0.0  # log(1)
        log_terms2 = at.log(c1) + log_G1
        log_terms3 = at.log(c2) + log_G2
            
        logsum = logsumexp(
                logsumexp(log_terms1, log_terms2),
                log_terms3
            )

        return logsum + logP + logB

    log_unnorm = my_logp(m, λ_m)

    if norm:
        mgrid = at.logspace(at.log10(1.0), at.log10(m_BHmax * 1.1), 2000)
        log_vals = my_logp(mgrid, λ_m)
        vals = at.exp(log_vals)
        norm_factor = attrapzvec(vals, mgrid)
        return log_unnorm - at.log(norm_factor)
    else:
        return log_unnorm


def logpdf_FP(theta, λ_m, Λ_q, norm=True, norm_p1=False, res=1000, force_m2_less_than_m1=False):
    m1, m2 = theta
    logp1 = logpdfm1_FP(m1, λ_m, norm=norm_p1)
    logp2 = logpdfm1_FP(m2, λ_m, norm=norm_p1)
    q = m2 / m1
    logf = log_f_q_FP(q, m2, Λ_q)
    lpdfval = logp1 + logp2 + logf

    if force_m2_less_than_m1:
        eval = at.and_(at.and_(m2 <= m1, m2 > 0), m1 > 0)
        joint = at.where(eval, lpdfval, MIN)
    else:
        joint = lpdfval

    if norm:
        #m_min = 1e-05
        λ_b = λ_m[11:]
        _, m_NSmin = λ_b[10:12]
        m_max = λ_m[11:][-1] * 1.5
        m_min = m_NSmin * 0.5

        m1_grid_ = at.geomspace(m_min, m_max, res)
        m2_grid_ = at.geomspace(m_min, m_max, res)
        m1_vals_, m2_vals_ = meshgrid_at(m1_grid_, m2_grid_)

        m1_stack = at.flatten(m1_vals_)
        logp1_grid = logpdfm1_FP(m1_stack, λ_m, norm=norm_p1)

        m2_stack = at.flatten(m2_vals_)
        logp2_grid = logpdfm1_FP(m2_stack, λ_m, norm=norm_p1)

        q_grid = m2_stack / m1_stack
        logf_grid = log_f_q_FP(q_grid, m2_stack, Λ_q)

        joint_grid = logp1_grid + logp2_grid + logf_grid

        joint_grid = at.where(m2_stack <= m1_stack, at.exp(joint_grid), 0.0)

        trapz = TrapzOp(axis=1)
        inner = trapz(at.reshape(joint_grid, m2_vals_.shape), m2_grid_[None, :])

        trapz0 = TrapzOp(axis=0)
        norm_factor = trapz0(inner.dimshuffle(0, 'x'), m1_grid_[:, None])

        return joint - at.log(norm_factor)
    else:
        return joint

