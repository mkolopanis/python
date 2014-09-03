import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import ipdb
from time import time
import emcee 
import triangle
import multiprocessing
def rotate_tqu(map_in,wl,alpha):  #rotates tqu map by phase
	npix=hp.nside2npix(hp.get_nside(map_in))
	tmp_map=np.zeros((3,npix))
	tmp_map[0]=map_in[0]
	tmp_map[1]=map_in[1]*np.cos(2*alpha*wl**2) + map_in[2]*np.sin(2*alpha*wl**2)
	tmp_map[2]=-map_in[1]*np.sin(2*alpha*wl**2) + map_in[2]*np.cos(2*alpha*wl**2)
	return tmp_map

def lnlike(theta,x,q_in,u_in,q_err,u_err):
	q,u,a=theta
	pol_q=np.array([q+2*a*w**2*u for w in x])
	pol_u=np.array([u-2*a*w**2*q for w in x])
	inv_sigmaq=1.0/(q_err**2)
	inv_sigmau=1.0/(u_err**2)
	return -0.5*(np.sum((pol_q-q_in)**2*inv_sigmaq + (pol_u-u_in)**2*inv_sigmau))

def lnprior(theta):
	q,u,a=theta
	if np.abs(q)<8e-6 and np.abs(u)<8e-6 and np.abs(a)<2250.:
		return 0.0 
	return -np.inf
	#return -0.5*(((np.random.normal(q,1e-6)-q)/(1e-6))**2+((np.random.normal(u,1e-6)-u)/1e-6)**2+((np.random.normal(a,100)-a)/100)**2-np.log(1e-6*1e-6*100))

def lnprob(theta,x,q_in,u_in,q_err,u_err):
	lp=lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp+lnlike(theta,x,q_in,u_in,q_err,u_err)

def fit_pixel(inputs):
	wl,q_array,u_array,sigma_q,sigma_u=inputs
	ndim,nwalkers=3, 100
	pos=[[1e-7,1e-7,10]*np.random.randn(ndim) for i in range(nwalkers)]
	sampler=emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(wl,q_array,u_array,sigma_q,sigma_u))
	sampler.run_mcmc(pos,800)
	data=np.array(sampler.chain[:,100:,:].reshape((-1,ndim)))
	output=np.zeros(3)
	deltas=np.zeros(3)
	for i in xrange(3):
		tmp1,out1,tmp2=np.percentile(data[:,i],[16,50,84])
		delta=np.max([out1-tmp1,tmp2-out1])
		output[i]=out1
		deltas[i]=delta
	return np.reshape([output,deltas],6)
	
if __name__=="__main__":
	t1=time()
	radio_file='/data/wmap/faraday_MW_realdata.fits'
	cl_file='/home/matt/wmap/simul_scalCls.fits'
	nside=128
	npix=hp.nside2npix(nside)
	cls=hp.read_cl(cl_file)
	simul_cmb=hp.sphtfunc.synfast(cls,512,fwhm=13.*np.pi/(180.*60.),new=1,pol=1);
	simul_cmb=hp.reorder(simul_cmb,r2n=1);
	
	alpha_radio=hp.read_map(radio_file,hdu='maps/phi');
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=512,order_in='ring',order_out='nested')
	
	names=['K','Ka','Q','V','W']
	bands=[23,33,41,61,94]
	wl=np.array([299792458./(band*1e9) for band in bands])
	num_wl=len(wl)
	npix1=hp.nside2npix(512)
	
	map_prefix='/data/wmap/wmap_band_forered_iqumap_r9_9yr_'
	simul_prefix='/data/mwap/simul_fr_rotated_'
	
	wmap_files=[ map_prefix+name+'_v5.fits' for name in names]
	simul_files=[simul_prefix+str(band).zfill(3)+'.fits' for band in bands]
	
	noise_const_t=np.asarray([1.429,1.466,2.188,3.131,6.544])*1e-3
	noise_const_q=np.asarray([1.435,1.472,2.197,3.141,6.560])*1e-3
	
	q_array=np.zeros((num_wl,npix))
	u_array=np.zeros((num_wl,npix))
	sigma_q=np.zeros((num_wl,npix))
	sigma_u=np.zeros((num_wl,npix))
	for i in range(num_wl):
		wmap_counts=hp.read_map(wmap_files[i],nest=1,field=3);
		tmp_cmb=rotate_tqu(simul_cmb,wl[i],alpha_radio);
		tmp_out=hp.ud_grade(np.random.normal(0,1,npix1)*noise_const_q[i]/np.sqrt(wmap_counts),nside_out=nside,order_in='nested')
		sigma_q[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.)
		tmp_out=hp.ud_grade(np.random.normal(0,1,npix1)*noise_const_q[i]/np.sqrt(wmap_counts),nside_out=nside,order_in='nested')
		sigma_u[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.)
		tmp_out=hp.ud_grade(tmp_cmb,nside_out=nside,order_in='nested')
		tmp_out=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383,pol=1)
		q_array[i]=tmp_out[1]+sigma_q[i]
		u_array[i]=tmp_out[2]+sigma_u[i]
	#emcee code will go here
	
	t0=time()
	pool=multiprocessing.Pool()
	outputs=pool.map(fit_pixel,[[wl, q_array[:,i],u_array[:,i],sigma_q[:,i],sigma_u[:,i]] for i in xrange(npix)])
	pool.close()
	pool.join()
	#samp1=fit_pixel(wl,q_array[:,0],u_array[:,0],sigma_q[:,0],sigma_u[:,0])
	fit_time=time()-t0
	#real_alpha=hp.sphtfunc.smoothing(hp.ud_grade(alpha_radio,128,order_in='nested'),fwhm=np.pi/180.,lmax=383)
	#real_t,real_q,real_u=hp.sphtfunc.smoothing(hp.ud_grade(simul_cmb,128,order_in='nested'),fwhm=np.pi/180.,lmax=383,pol=1)
	
	#samp1=sampler.flatchain
	
	#fig=triangle.corner(samp1,labels=["$q$","$u$","$alpha_{RM}$"],truths=[real_q[0],real_u[0],real_alpha[0]])
	#plt.show()
	
	#q_mcmc,u_mcmc,a_mcmc=map(lambda v: (v[1],v[2]-v[1],v[1]-v[0]), zip(*np.percentile(samp1,[16,50,84], axis=0)))
	
	total_time=time()-t1
	print 'Fitting occurred in '+'{:.3f}'.format(fit_time/60.)+' minutes'
	print 'Total computation time: '+'{:.3f}'.format(total_time/60.)+' minutes'
	#outputs=np.array(outputs)
	q_map,u_map,alpha_map,dq_map,du_map,dalpha_map=list(zip(*outputs))
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
	prim=fits.PrimaryHDU()
	hdulist=fits.HDUList([prim,alpha_head,q_head,u_head,dalpha_head,dq_head,du_head])
	hdulist.writeto('wmap_alpha_noise_emcee.fits')

