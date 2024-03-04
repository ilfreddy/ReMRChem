from scipy.special import erf
from vampyr import vampyr3d as vp
from vampyr import vampyr1d as vp1
import numpy as np

def point_charge(position, center , charge):
    d2 = ((position[0] - center[0])**2 +
          (position[1] - center[1])**2 +
          (position[2] - center[2])**2)
    distance = np.sqrt(d2)
    return charge / distance

def coulomb_HFYGB(position, center, charge, precision):
    d2 = ((position[0] - center[0])**2 +
          (position[1] - center[1])**2 +
          (position[2] - center[2])**2)
    distance = np.sqrt(d2)
    def smoothing_HFYGB(charge, prec):
        factor = 0.00435 * prec / charge**5
        return factor**(1./3.)
    def uHFYGB(r):
        u = erf(r)/r + (1/(3*np.sqrt(np.pi)))*(np.exp(-(r**2)) + 16*np.exp(-4*r**2))
        return u
    factor = smoothing_HFYGB(charge, precision)
    value = uHFYGB(distance/factor)
    return charge * value / factor

