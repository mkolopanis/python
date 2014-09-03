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

def convertcenter(ra,dec):
	return np.array([(ra[0]+ra[1]/60.)/24.*360,dec])

def regioncoords(ra,dec,dx,nx,ny):
	decs=np.array([ dec +dx*( j - ny/2.) for j in xrange(ny)])
	coords= np.array([ [np.mod(ra+dx*(i-nx/2)/np.cos(dec*np.pi/180.),360),dec]  for i in xrange(nx) for dec in decs])
	return coords

if __name__=='__main__':
	t1=time()
	radio_file='/data/wmap/faraday_MW_realdata.fits'
	cl_file='/home/matt/wmap/simul_scalCls.fits'
	nside=128
	npix=hp.nside2npix(nside)
	
	bands=[43.1,94.5]
	q_fwhm=[27.3,11.7]
	noise_const_q=np.array([6./q_fwhm[0],6./q_fwhm[1]])*1e-6
	centers=np.array([convertcenter([12,4],-39),convertcenter([5,12],-39),convertcenter([0,48],-48),convertcenter([22,44],-36)])
	wl=np.array([299792458./(band*1e9) for band in bands])
	num_wl=len(wl)
	q_array=np.zeros((num_wl,npix))
	u_array=np.zeros((num_wl,npix))
	sigma_q=np.zeros((num_wl,npix))
	sigma_u=np.zeros((num_wl,npix))	
	cls=hp.read_cl(cl_file)
	simul_cmb=hp.sphtfunc.synfast(cls,128,fwhm=5.*np.pi/(180.*60.),new=1,pol=1);
	simul_cmb=hp.reorder(simul_cmb,r2n=1);
	
	alpha_radio=hp.read_map(radio_file,hdu='maps/phi');
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=128,order_in='ring',order_out='nested')
	t2=time()
	for i in range(num_wl):
		tmp_cmb=hp.sphtfunc.smoothing(simul_cmb,fwhm=np.pi/180*q_fwhm[i],lmax=383)
		tmp_cmb=rotate_tqu(tmp_cmb,wl_q[i],alpha_radio);
		tmp_out=np.random.normal(0,1,npix)*noise_const_q[i]
		sigma_q[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
		tmp_out=np.random.normal(0,1,npix)*noise_const_q[i]
		sigma_u[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
		tmp_out=hp.sphtfunc.smoothing(tmp_cmb,fwhm=np.pi/180.,lmax=383,pol=1)
		q_array[i]=tmp_out[1]+sigma_q_3[i]
		u_array[i]=tmp_out[2]+sigma_u_3[i]

	#need to find pixels for just the CMB field from QUIET
	dx=1./(60.)*6
	nx=np.int(15/dx)
	ny=nx
	for p in xrange(len(centers)):
		coords=regioncoords(centers[p,0],centers[p,1],dx,nx,ny)
		coords_sky=SkyCoords(ra=coords[:,0],dec=coords[:,1],unit=u.degree,frame='fk5')
		phi=coords_sky.galactic.l.deg
		theta=coords_sky.galactic.b.deg
		pixels=np.reshape(hp.ang2pix(128,theta*np.pi/180.,phi*np.pi/180.),(nx,ny))
		#fit these pixels
		t0=time()
		pool=multiprocessing.Pool()
		outputs1=pool.map(fit_pixel,[[wl, q_array[:,i],u_array[:,i],sigma_q[:,i],sigma_u[:,i]] for i in pixels.flat])
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
		q_map,u_map,alpha_map,dq_map,du_map,dalpha_map,chisquare=list(zip(*outputs))
		q_map=np.reshape(q_map,(nx,ny))
		u_map=np.reshape(u_map,(nx,ny))
		alpha_map=np.reshape(alpha_map,(nx,ny))
		dq_map=np.reshape(dq_map,(nx,ny))
		du_map=np.reshape(du_map,(nx,ny))
		dalpha_map=np.reshape(dalpha_map,(nx,ny))
		chisquare=np.resharp(chisquare,(nx,ny))
		ipdb.set_trace()

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
		du_head.header['TUNIT1']=('K_{CMB} Thermodynamic', 	'Physical Units of Map')
		chi_head=fits.ImageHDU(chisquare,name='CHISQUARE')
		prim=fits.PrimaryHDU()
		hdulist=fits.HDUList([prim,alpha_head,q_head,u_head,dalpha_head,dq_head,du_head,chi_head])
		hdulist.writeto('alpha_quiet_2-band_mle_cmb'+str(p).zfill(1)+'.fits')
	
	total_time=time()-t1
	print 'Total computation time: '+'{:.3f}'.format(total_time/60.)+' minutes'
