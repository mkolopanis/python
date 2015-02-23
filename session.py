# coding: utf-8

import triangle
import emcee
import numpy as np
import matplotlib.pyplot as plt
m_true=-.9594
b_true=4.294
f_true=.534
N=50
x=np.sort(10*np.random.rand(N))
yerr=.3+.5*np.random.rand(N)
y=m_true*x+b_true
plt.plot(x,y)
plt.errorbar(x,y,yerr=yerr)
get_ipython().magic(u'whos ')
y_true=m_true*x+b_true
y +=np.abs(f_true*y)*np.random.randn(N)
plt.scatter(x,y)
plt.scatter(x,y)
plt.plot(x,y,'k.')
plt.errorbar(x,y,yerr=yerr)
plt.errorbar(x,y,yerr=yerr,fmt='.',color='k')
plt.plot(x,y_true,'r-')

A=np.vstack((np.ones_like(x),x)).T
C=np.diag(yerr*yerr)



cov=np.linalg.inv(np.dot(A.T,np.linalg.solve(C,A)))
b_ls,m_ls=np.dot(cov,np.dot(A.T,np.linalg.solve(C,y)))
b_ls,m_ls
cov

plt.plot(x,m_ls*x+b_ls,'g-')
def lnlike(theta,x,y,yerr):
    m,b,lnf=theta
    model=m*x+b
    inv_sigma2=1.0/(yerr**2+model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2-np.log(inv_sigma2)))
def lnprior(theta):
    m,b,lnf=theta
    if -5.0<m<.5 and 0.0<b<10.0 and -10.<lnf<1.0:
        return 0.0
    return -np.inf
def lnprob(theta,x,y,yerr):
    lp=lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp+lnlike(theta,x,y,yerr)
ndim,nwalkers=3,100
pos= [ [m_ls,b_ls,.4] + 1e-4*np.random.randn(ndim) for i in xrange(nwalkers)]
pos
sampler=emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(x,y,yerr))
sampler.run_mcmc(pos,500)
sampler.run_mcmc(pos,500)
plt.figure()
this=sampler.chain
this
samples=sampler.chain[:,50:,:].reshape((-1,ndim))
samples
np.shape(samples)
fig=triangle.corner(samples,labels=["$m$","$b$","$\ln\,f$"], truths=[m_true,b_true,np.log(f_true)])
