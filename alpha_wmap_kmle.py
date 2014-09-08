import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import ipdb; 
from time import time
import emcee 
import triangle
import multiprocessing
from kapteyn import kmpfit
from rotate_tqu import rotate_tqu
from alpha_function import alpha_function

def fit_pixel(inputs):
	p0=[1,1,10]
	fitobj=kmpfit.Fitter(residuals=alpha_function,data=inputs)
	fitobj.fit(params0=p0)
	parms=np.array(fitobj.params)
	parms[:2]*=1e-7
	dparms=np.array(fitobj.stderr)
	dparms[:2]*=1e-7
	return np.concatenate([parms,dparms,[fitobj.chi2_min],[fitobj.dof]])

if __name__=='__main__':
	t1=time()
	radio_file='/data/wmap/faraday_MW_realdata.fits'
	cl_file='/home/matt/wmap/simul_scalCls.fits'
	nside=128
	npix=hp.nside2npix(nside)
	t2=time()
	names=['K','Ka','Q','V','W']
	bands=[23,33,41,61,94]
	w_fwhm=np.array([.93,.68,.53,.35,.23])
	wl=np.array([299792458/(band*1e9) for band in bands])
	num_wl=len(wl)
	npix1=hp.nside2npix(512)
	
	map_prefix='/data/wmap/wmap_band_iqumap_r9_9yr_'
	simul_prefix='/data/mwap/simul_fr_rotated_'
	
	wmap_files=[ map_prefix+name+'_v5.fits' for name in names]
	simul_files=[simul_prefix+str(band).zfill(3)+'.fits' for band in bands]
	
	noise_const_t=np.asarray([1.429,1.466,2.188,3.131,6.544])*1e-3
	noise_const_q=np.asarray([1.435,1.472,2.197,3.141,6.560])*1e-3
	
	q_array=np.zeros((num_wl,npix))
	u_array=np.zeros((num_wl,npix))
	sigma_q=np.zeros((num_wl,npix))
	sigma_u=np.zeros((num_wl,npix))
	
	cls=hp.read_cl(cl_file)
	simul_cmb=hp.sphtfunc.synfast(cls,512,fwhm=5.*np.pi/(180.*60.),new=1,pol=1);
	simul_cmb=hp.reorder(simul_cmb,r2n=1);
	
	alpha_radio=hp.read_map(radio_file,hdu='maps/phi');
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=512,order_in='ring',order_out='nested')
	
	for i in range(num_wl):
		tmp_cmb=hp.sphtfunc.smoothing(simul_cmb,pol=1,fwhm=np.pi/(180.)*w_fwhm[i])
		wmap_counts=hp.read_map(wmap_files[i],nest=1,field=3);
		tmp_cmb=rotate_tqu(tmp_cmb,wl[i],alpha_radio);
		tmp_q=np.random.normal(0,1,npix1)*noise_const_q[i]
		tmp_u=np.random.normal(0,1,npix1)*noise_const_q[i]
		tmp_out=hp.ud_grade(tmp_cmb+np.array([np.zeros(npix1),tmp_q,tmp_u]),nside_out=nside,order_in='nested',order_out='nested')
		tmp_out=hp.sphtfunc.smoothing(tmp_out,lmax=383,pol=1,fwhm=np.pi/180.)
		q_array[i]=tmp_out[1]
		u_array[i]=tmp_out[2]
		sigma_q[i]=hp.sphtfunc.smoothing(hp.ud_grade(tmp_q,nside_out=nside,order_in='nested',order_out='nested'),fwhm=np.pi/180.,lmax=383);
		sigma_u[i]=hp.sphtfunc.smoothing(hp.ud_grade(tmp_u,nside_out=nside,order_in='nested',order_out='nested'),fwhm=np.pi/180.,lmax=383);
	
	#fit these pixels
	t0=time()
	pool=multiprocessing.Pool()
	outputs=pool.map(fit_pixel,[[wl, q_array[:,i],u_array[:,i],sigma_q[:,i],sigma_u[:,i]] for i in xrange(npix)])
	pool.close()
	pool.join()
	fit_time=time()-t0
	#reconstruct 4 QUIET fields
	#emcee code will go here

	#real_alpha=hp.sphtfunc.smoothing(hp.ud_grade(alpha_radio,128,order_in='nested'),fwhm=np.pi/180.,lmax=383)
	#real_t,real_q,real_u=hp.sphtfunc.smoothing(hp.ud_grade(simul_cmb,128,order_in='nested'),fwhm=np.pi/180.,lmax=383,pol=1)
	
	#samp1=sampler.flatchain
	
	#fig=triangle.corner(samp1,labels=["$q$","$u$","$alpha_{RM}$"],truths=[real_q[0],real_u[0],real_alpha[0]])
	#plt.show()
	
	#q_mcmc,u_mcmc,a_mcmc=map(lambda v: (v[1],v[2]-v[1],v[1]-v[0]), zip(*np.percentile(samp1,[16,50,84], axis=0)))
	
	print 'Fitting occurred in '+'{:.3f}'.format(fit_time/60.)+' minutes'
	q_map,u_map,alpha_map,dq_map,du_map,dalpha_map,chisquare,dof=list(zip(*outputs))
	alpha_head=fits.ImageHDU(alpha_map,name="ALPHA RM")
	dalpha_head=fits.ImageHDU(dalpha_map,name="ALPHA UNCERTAINTY")
	alpha_head.header['TUNIT1']=('rad/m^2', 'Physical Units of Map')	
	dalpha_head.header['TUNIT1']=('rad/m^2', 'Physical Units of Map')	
	q_head=fits.ImageHDU(q_map,name="STOKES Q")
	q_head.header['TUNIT1']=('K_{CMB} Thermodynamic', 'Physical Units of Map')
	dq_head=fits.ImageHDU(dq_map,name="UNCERTAINTY Q")
	dq_head.header['TUNIT1']=('K_{CMB} Thermodynamic', 'Physical Units of Map')
	u_head=fits.ImageHDU(u_map,name="STOKES U")
	u_head.header['TUNIT1']=('K_{CMB} Thermodynamic', 'Physical Units of Map')
	du_head=fits.ImageHDU(du_map,name="UNCERTAINTY U")
	du_head.header['TUNIT1']=('K_{CMB} Thermodynamic', 'Physical Units of Map')
	chi_head=fits.ImageHDU(chisquare,name='CHISQUARE')
	dof_head=fits.ImageHDU(dof,name='DOF')
	prim=fits.PrimaryHDU()
	hdulist=fits.HDUList([prim,alpha_head,q_head,u_head,dalpha_head,dq_head,du_head])
	hdulist.writeto('alpha_wmap_kmle_cmb.fits')
	
	total_time=time()-t1
	print 'Total computation time: '+'{:.3f}'.format(total_time/60.)+' minutes'
