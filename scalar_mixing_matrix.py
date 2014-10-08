import numpy as np
from scipy.misc import factorial as fac
from sympy.physics.wigner import wigner_3j


def Mll(wl,l1,l2):
	l3=np.arange(len(wl))
	array=[(2*i+1)*wl[i]*wigner_3j(l1,l2,i,0,0,0)**2 for i in l3]
	ml = (2*l2+1)/(4*np.pi)*np.sum(array)
	return ml
