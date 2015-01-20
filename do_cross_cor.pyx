import numpy as np
cimport numpy as np
import healpy as hp
from libc.math cimport sqrt, floor, fabs
cimport libc
ctypedef unsigned size_t
ctypedef size_t tsize
import os
import cython

def alm2cl(alms,alms2 = None, lmax = None, mmax = None, lmax_out= None, weight = None):
    #############################
    # Check alm and number of spectra
    #
    cdef int Nspec, Nspec2
    if not hasattr(alms, '__len__'):
        raise ValueError('alms must be an array or a sequence of arrays')
    if not hasattr(alms[0], '__len__'):
        alms_lonely = True
        alms = [alms]
    else:
        alms_lonely = False

    Nspec = len(alms)

    if alms2 is None:
        alms2 = alms
    
    if not hasattr(alms2, '__len__'):
        raise ValueError('alms2 must be an array or a sequence of arrays')
    if not hasattr(alms2[0], '__len__'):
        alms2 = [alms2]
    Nspec2 = len(alms2)
    
    if Nspec != Nspec2:
        raise ValueError('alms and alms2 must have same number of spectra')
    
    ##############################################
    # Check sizes of alm's and lmax/mmax/lmax_out
    #
    cdef int almsize
    almsize = alms[0].size
    for i in xrange(Nspec):
        if alms[i].size != almsize or alms2[i].size != almsize:
            raise ValueError('all alms must have same size')

    if lmax is None:
        if mmax is None:
            lmax = alm_getlmax(almsize)
            mmax = lmax
        else:
            lmax = alm_getlmax2(almsize, mmax)

    if mmax is None:
        mmax = lmax

    if lmax_out is None:
        lmax_out = lmax

    if weight is None:
        weight = np.repeat(1.+1J,alms[0].size)

    #######################
    # Computing the spectra
    #
    cdef int j, l, m, limit
    cdef int lmax_ = lmax, mmax_ = mmax
    cdef int lmax_out_ = lmax_out
    
    cdef np.ndarray[double, ndim=1] powspec_
    cdef np.ndarray[np.complex128_t, ndim=1] alm1_
    cdef np.ndarray[np.complex128_t, ndim=1] alm2_
    cdef np.ndarray[double,ndim=1] tmp_w

    for n in xrange(Nspec): 
        for m in xrange(0,Nspec-n):
            spectra = []
            powspec_ = np.zeros(lmax + 1)
            alm1_ = alms[m]
            alm2_ = alms2[m + n]
            # compute cross-spectrum alm1[n] x alm2[n+m]
            # and place result in result list
            for l in range(lmax_ + 1):
                tmp_w=np.zeros( lmax + 1 )
                j = alm_getidx(lmax_, l, 0)
                powspec_[l] = alm1_[j].real * alm2_[j].real * weight[j].real**2
                tmp_w[l]=weight[j].real**2
                limit = l if l <= mmax else mmax
                for m in range(1, limit + 1):
                    j = alm_getidx(lmax_, l, m)
                    powspec_[l] += 2 * (alm1_[j].real * alm2_[j].real * weight[j].real**2 +  alm1_[j].imag * alm2_[j].imag * weight[j].imag**2)
                    tmp_w[l]+= ( weight[j].real**2 + weight[j].imag**2)
                powspec_[l] /= tmp_w[l]
            spectra.append(powspec_)

    if alms_lonely:
        spectra = spectra[0]

    return spectra


@cython.cdivision(True)
cdef inline int alm_getidx(int lmax, int l, int m):
    return m*(2*lmax+1-m)/2+l

@cython.cdivision(True)
cdef inline int alm_getlmax(int s):
    cdef double x
    x=(-3+np.sqrt(1+8*s))/2
    if x != floor(x):
        return -1
    else:
        return <int>floor(x)

@cython.cdivision(True)
cdef inline int alm_getlmax2(int s, int mmax):
    cdef double x
    x = (2 * s + mmax ** 2 - mmax - 2.) / (2 * mmax + 2.)
    if x != floor(x):
        return -1
    else:
        return <int>floor(x)
