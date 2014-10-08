import numpy as np
import healpy as hp
from astropy.io import fits
from rotate_tqu import rotate_tqu
import matplotlib.pyplot as plt
import json
cl_file='/home/matt/wmap/simul_scalCls.fits'
radio_file='/data/wmap/faraday_MW_realdata.fits'
fwhm=[27.3,11.7]
bands=[43.1,94.5]
wl=np.array([299792458./(b*1e9) for b in bands])
sky_cut=np.arange(200,0,-5)/200.
sky_cut=np.append(sky_cut,(225*(np.pi/180.)**2/(4*np.pi)))
nside=512
npix=hp.nside2npix(512)

cl_gen=hp.read_cl(cl_file)
alpha=hp.read_map(radio_file,hdu='maps/phi')
alpha=hp.ud_grade(alpha,nside)
const=2*(wl[0]**2-wl[1]**2)
cl_array=[]
simul_cmb=hp.synfast(cl_gen,nside,pol=1,new=1,fwhm=0)
rot_1=rotate_tqu(simul_cmb,wl[0],alpha)
rot_2=rotate_tqu(simul_cmb,wl[1],alpha)
Delta_Q=(rot_1[1]-rot_2[1])/const
alpha_U=alpha*rot_1[2]
dQ=hp.ma(Delta_Q)
aU=hp.ma(alpha_U)
for fsky in sky_cut:
	pix=np.arange((1-fsky)*npix).astype(int)	##Pixels to be masked
	mask=np.repeat(False,npix)
	if len(pix) > 0:		##Check if number pixels > 0
		mask[pix]=True		##Mask "Bad" pixels
	dQ.mask = mask
	aU.mask = mask
	cls=hp.anafast(dQ,map2=aU)
	hp.write_cl('cl_FR_QxAU_fsky_{:03d}.fits'.format(int(100*fsky)),cls,dtype='float')
	cl_array.append(cls)

fracs=[array/cl_array[0] for array in cl_array]
mean_fraction=[x.mean() for x in fracs]
median_fraction=[np.median(x) for x in fracs]
cut_list=sky_cut.tolist()
f=open('mean_fraction','w')
json.dump([cut_list,mean_fraction],f)
f.close()
f=open('median_fraction','w')
json.dump([cut_list,median_fraction],f)
f.close()

plt.figure()
plt.plot(sky_cut,mean_fraction,'x',label='mean')
plt.plot(sky_cut,median_fraction,'+',label='median')
plt.legend(loc='upper right')
plt.axis([1,0,0,1.05])
plt.xlabel('Sky Cut Fraction')
plt.ylabel('Fraction of Full Power Spectrum')
plt.title('Fractional Power in Cut Sky, FR QxaU')
plt.locator_params(axis='x',nbins=10)
plt.savefig('fractional_sky_cut.eps')
plt.savefig('fractional_sky_cut.png',format='png')


#plt.figure()
#plt.plot(sky_cut,mean_fraction,'x',label='mean')
#plt.plot(sky_cut,median_fraction,'+',label='median')
plt.legend(loc='lower left')
plt.yscale('log')
plt.axis([1,0,1e-5,2])
plt.locator_params(axis='x',nbins=10)
#plt.xlabel('Sky Cut Fraction')
#plt.ylabel('Fraction of Full Power Spectrum')
#plt.title('Fractional Power in Cut Sky, FR QxaU')
plt.savefig('fractional_sky_cut_log.eps')
plt.savefig('fractional_sky_cut_log.png',format='png')
