import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import ipdb; 
from time import time
import emcee 
import triangle
import multiprocessing
from rotate_tqu import rotate_tqu

def lnlike(theta,x,q_in,u_in,q_err,u_err):
	q,u,a=theta
	pol_q=np.array([q+2*a*w**2*u for w in x])
	pol_u=np.array([u-2*a*w**2*q for w in x])
	inv_sigmaq=1.0/(q_err**2)
	inv_sigmau=1.0/(u_err**2)
	return -0.5*(np.sum((pol_q-q_in)**2*inv_sigmaq  + (pol_u-u_in)**2*inv_sigmau ))
#- np.log(inv_sigmaq)- np.log(inv_sigmau)
def lnprior(theta):
	q,u,a=theta
	if np.abs(q)<8e-6 and np.abs(u)<8e-6 and np.abs(a)<2250.:
		return 0.0
	else:
		return -np.inf
	#return -0.5*(((q)/(1000))**2+((u)/1000)**2+((a)/1000)**2+2*np.log(1000*1000*1000))

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

if __name__=='__main__':
	t1=time()
	radio_file='/data/wmap/faraday_MW_realdata.fits'
	cl_file='/home/matt/wmap/simul_scalCls.fits'
	nside=128
	npix=hp.nside2npix(nside)
	
	
	bands_p=[30,44,70,100,143,217,353]
	wl_p=np.array([299792458/(band*1e9) for band in bands_p])
	num_wl=len(wl_p)
	beamfwhm = np.array([33.,24.,14.,9.5,7.1,5.0,5.0])
	pixarea = np.array([(4*np.pi/x)*(180./np.pi*60.)**2 for x in [hp.nside2npix(1024),hp.nside2npix(2048)]])
	
	#beamscale =np.zeros(num_wl)
	#for i in range(3):
	#	beamscale[i]=beamfwhm[i]/np.sqrt(pixarea[0])
	#for i in range(3,7):
	#	beamscale[i]=beamfwhm[i]/np.sqrt(pixarea[1])
	#
	#tmp_noise=[2.0,2,7,4.7,2.5,2.2,4.8,14.7]
	#noise_const_t=np.array([tmp_noise[x]/beamscale[x] for x in range(num_wl)])*2.725e-6
	#tmp_noise=[2.8,3.9,6.7,4.0,4.2,9.8,29.8]
	#noise_const_q=np.array([tmp_noise[x]/beamscale[x] for x in range(num_wl)])*2.725e-6
	noise_const_t=np.array([9.2+21.02,12.5+2.61,23.2+7.87,11,6,12,43])
	noise_const_q=noise_const_t	
	q_array_1=np.zeros((num_wl,npix))
	u_array_1=np.zeros((num_wl,npix))
	sigma_q_1=np.zeros((num_wl,npix))
	sigma_u_1=np.zeros((num_wl,npix))
	
	
	cls=hp.read_cl(cl_file)
	simul_cmb=hp.sphtfunc.synfast(cls,2048,fwhm=5.*np.pi/(180.*60.),new=1,pol=1);
	simul_cmb=hp.reorder(simul_cmb,r2n=1);
	
	alpha_radio=hp.read_map(radio_file,hdu='maps/phi');
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=2048,order_in='ring',order_out='nested')
	
	npix1=hp.nside2npix(2048)
	t2=time()
	
	for i in range(num_wl):
		if i in range(3):
			npix1=hp.nside2npix(1024)
		if i==3:
			npix1=hp.nside2npix(2048)
		tmp_cmp=hp.sphtfunc.smoothing(simul_cmb,fwhm=beamfwhm[i]*np.pi/(180.*60.))	
		tmp_cmb=rotate_tqu(tmp_cmb,wl_p[i],alpha_radio);
		tmp_q=np.random.normal(0,1,npix1)*noise_const_q[i]
		tmp_u=np.random.normal(0,1,npix1)*noise_const_q[i]
		tmp_out=hp.ud_grade(tmp_cmb+np.array([np.zeros(npix1),tmp_q,tmp_u]),nside_out=nside,order_in='nested',order_out='nested');
		tmp_out=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383,pol=1)
		q_array_1[i]=tmp_out[1]
		u_array_1[i]=tmp_out[2]
		sigma_q_1[i]=hp.sphtfunc.smoothing(hp.ud_grade(tmp_q,nside_out=nside,order_in='nested',order_out='nested'),fwhm=np.pi/180.,lmax=383);
		sigma_u_1[i]=hp.sphtfunc.smoothing(hp.ud_grade(tmp_u,nside_out=nside,order_in='nested',order_out='nested'),fwhm=np.pi/180.,lmax=383);
		
	t3=time()
	print 'This computation took '+"{:.3f}".format((t3-t2)/60.)+' minutes'
	names=['K','Ka','Q','V','W']
	bands_w=[23,33,41,61,94]
	w_fwhm=np.array([.93,.68,.53,.35,.23])
	wl_w=np.array([299792458/(band*1e9) for band in bands_w])
	num_wl=len(wl_w)
	npix1=hp.nside2npix(512)
	
	map_prefix='/data/wmap/wmap_band_iqumap_r9_9yr_'
	simul_prefix='/data/mwap/simul_fr_rotated_'
	
	wmap_files=[ map_prefix+name+'_v5.fits' for name in names]
	simul_files=[simul_prefix+str(band).zfill(3)+'.fits' for band in bands_w]
	
	noise_const_t=np.asarray([1.429,1.466,2.188,3.131,6.544])*1e-3
	noise_const_q=np.asarray([1.435,1.472,2.197,3.141,6.560])*1e-3
	
	q_array_2=np.zeros((num_wl,npix))
	u_array_2=np.zeros((num_wl,npix))
	sigma_q_2=np.zeros((num_wl,npix))
	sigma_u_2=np.zeros((num_wl,npix))
	
	simul_cmb=hp.ud_grade(simul_cmb,nside_out=512,order_in='nested',order_out='nested')
	
	
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=512,order_in='nested',order_out='nested');
	
	for i in range(num_wl):
		tmp_cmb=hp.sphtfunc.smoothing(simul_cmb,fwhm=w_fwhm*np.pi/180.)
		wmap_counts=hp.read_map(wmap_files[i],nest=1,field=3);
		tmp_cmb=rotate_tqu(tmp_cmb,wl_w[i],alpha_radio);
		tmp_q=np.random.normal(0,1,npix1)*noise_const_q[i]/np.sqrt(wmap_counts)
		tmp_u=np.random.normal(0,1,npix1)*noise_const_q[i]/np.sqrt(wmap_counts)
		tmp_out=hp.ud_grade(tmp_cmb+np.array([np.zeros(npix1),tmp_q,tmp_u]),nside_out=nside,order_in='nested',order_out='nested');
		tmp_out=hp.sphtfunc.smoothing(tmp_out,lmax=383,pol=1,fwhm=np.pi/180.)
		q_array_2[i]=tmp_out[1]
		u_array_2[i]=tmp_out[2]
		sigma_q_2[i]=hp.sphtfunc.smoothing(hp.ud_grade(tmp_q,nside_out=nside,order_in='nested',order_out='nested'),fwhm=np.pi/180.,lmax=383);
		sigma_u_2[i]=hp.sphtfunc.smoothing(hp.ud_grade(tmp_u,nside_out=nside,order_in='nested',order_out='nested'),fwhm=np.pi/180.,lmax=383);
	
	wl=np.concatenate((wl_p,wl_w))
	
	q_array=np.concatenate((q_array_1,q_array_2))
	u_array=np.concatenate((u_array_1,u_array_2))
	sigma_q=np.concatenate((sigma_q_1,sigma_q_2))
	sigma_u=np.concatenate((sigma_u_1,sigma_u_2))
	
	new_index=np.argsort(wl)
	new_index=new_index[::-1]
	wl=wl[new_index]
	nuw_wl=len(new_index)
	q_array=q_array[new_index]
	u_array=u_array[new_index]
	sigma_q=sigma_q[new_index]
	sigma_u=sigma_u[new_index]
	
	bands=bands_w+bands_p
	bands.sort()
	
	#emcee code will go here

	t0=time()
	pool=multiprocessing.Pool()
	outputs=pool.map(fit_pixel,[[wl, q_array[:,i],u_array[:,i],sigma_q[:,i],sigma_u[:,i]] for i in xrange(npix)])
	pool.close()
	pool.join()
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
	hdulist.writeto('planck_wmap_alpha_emcee.fits')
