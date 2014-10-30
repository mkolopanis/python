import numpy as np
from scipy.misc import factorial as fac
from decimal import Decimal
import multiprocessing
def wigner_3j(l1,l2,l3):
	if not ( abs( l1-l2 ) <= l3 <= l1 +l2):
		return 0
	L=l1+l2+l3
	if L % 2 != 0:
		return 0
	return float(Decimal((-1)**(L/2.))*np.sqrt((fac(L - 2*l1,exact=1)*fac(L-2*l2,exact=1)*fac(L-2*l3,exact=1))/Decimal(fac(L+1,exact=1)))*(Decimal(fac(L/2,exact=1))/(fac(L/2 - l1,exact=1)*fac(L/2 - l2,exact=1)*fac(L/2 - l3,exact=1))))


def mll_value(inputs):
	wl,l1,l2=inputs
	l3=np.arange(len(wl))
	array=[(2*i+1)*wl[i]*wigner_3j(l1,l2,i)**2 for i in l3]
	ml = (2*l2+1)/(4*np.pi)*np.sum(array)
	return ml

def Mll(wl):
	x,y=np.mgrid[0:len(wl),0:len(wl)]
	pool=multiprocessing.Pool()
	results=pool.map(mll_value,[[wl,x.flat[i],y.flat[i]] for i in xrange(len(x.flat))])
	pool.close()
	pool.join()
	#Mll=np.matrix(np.reshape(results,(len(x.flat),len(x.flat))))
	return results
