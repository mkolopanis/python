import numpy as np
import healpy as hp
from astropy.io import fits
from rotate_tqu import rotate_tqu
import matplotlib.pyplot as plt

cl_file='/home/matt/wmap/simul_scalCls.fits'
radio_file='/data/wmap/faraday_MW_realdata.fits'
fwhm=[27.3,11.7]
bands=[43.1,94.5]
wl=np.array([299792458./(b*1e9) for b in bands])
fwhm_gen=[0,5,11.7,27.3,45,60]
nside=512
npix=hp.nside2npix(512)

cl_gen=hp.read_cl(cl_file)
alpha=hp.read_map(radio_file,hdu='maps/phi')
alpha=hp.ud_grade(alpha,nside)
const=2*(wl[0]**2-wl[1]**2)
plt.figure()
for f in fwhm_gen:
	simul_cmb=hp.synfast(cl_gen,nside,pol=1,new=1,fwhm=f*np.pi/(180.*60))
	rot_1=rotate_tqu(simul_cmb,wl[0],alpha)
	rot_2=rotate_tqu(simul_cmb,wl[1],alpha)
	Delta_Q=(rot_1[1]-rot_2[1])/const
	alpha_U=alpha*rot_1[2]

	dQ=hp.ma(Delta_Q)
	aU=hp.ma(alpha_U)
	dQ=hp.smoothing(dQ,fwhm=np.pi/180.)
	aU=hp.smoothing(dQ,fwhm=np.pi/180.)
	cls=hp.anafast(dQ,map2=aU)
	l=np.arange(len(cls))
	ll=np.array([i*(i+1)/(2*np.pi) for i in l])
	plt.plot(l[:384],ll[:384]*cls[:384]*1e12,'.',label='fwhm= {:2d}'.format(int(f)))
plt.legend(loc='upper right',numpoints=1)
plt.show()
