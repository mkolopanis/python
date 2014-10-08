import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import ipdb; 
from time import time
import triangle
import multiprocessing
from kapteyn import kmpfit
from rotate_tqu import rotate_tqu
from alpha_function import alpha_function

def fit_pixel(inputs):
	wl,q_array,u_array,sigma_q,sigma_u=inputs
	sb1=sigma_q[0]**2/u_array[-1]**2 + sigma_q[-1]**2/u_array[-1]**2+(sigma_q[0]**2+sigma_q[-1]**2)*sigma_u[-1]**2/u_array[-1]**4
	sb2=sigma_u[0]**2/q_array[-1]**2 + sigma_u[-1]**2/q_array[-1]**2+(sigma_u[0]**2-sigma_u[-1]**2)*sigma_q[-1]**2/q_array[-1]**4
	b0=( (q_array[0]-q_array[-1])/(u_array[-1]*sb1**2) - (u_array[0]-u_array[-1])/(q_array[-1]*sb2**2) )/(1./sb1**2+1./sb2**2)
	a0=b0/(2*(wl[0]**2-wl[-1]**2))
	p0=[q_array[-1]*1e7,u_array[-1]*1e7,a0]
	fitobj=kmpfit.Fitter(residuals=alpha_function,data=inputs)
	fitobj.fit(params0=p0)
	parms=np.array(fitobj.params)
	parms[:2]*=1e-7
	dparms=np.array(fitobj.xerror)
	dparms[:2]*=1e-7
	return np.concatenate([parms,dparms,[fitobj.chi2_min],[fitobj.dof]])
#def fit_pixel(inputs):
#	p0=[1.,1.,10]
#	fitobj=kmpfit.Fitter(residuals=alpha_function,data=inputs)
#	fitobj.fit(params0=p0)
#	parms=np.array(fitobj.params)
#	parms[:2]*=1e-7
#	dparms=np.array(fitobj.stderr)
#	dparms[:2]*=1e-7
#	return np.concatenate([parms,dparms,[fitobj.chi2_min],[fitobj.dof]])

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
	
	cls=hp.read_cl(cl_file)
	simul_cmb=hp.sphtfunc.synfast(cls,128,fwhm=5.*np.pi/(180.*60.),new=1,pol=1);
	
	alpha_radio=hp.read_map(radio_file,hdu='maps/phi');
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=128,order_in='ring',order_out='ring')
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
	
	for i in range(num_wl):
		tmp_cmb=hp.sphtfunc.smoothing(simul_cmb,fwhm=np.pi/(180.*60.)*q_fwhm[i],pol=1)
		rot_cmb=rotate_tqu(tmp_cmb,wl[i],alpha_radio);
		tmp_q=np.random.normal(0,1,npix)*noise_const_q[i]
		tmp_u=np.random.normal(0,1,npix)*noise_const_q[i]
		tmp_out=hp.sphtfunc.smoothing(rot_cmb,fwhm=np.pi/180.,lmax=383,pol=1)
		q_array[i]=tmp_out[1]
		u_array[i]=tmp_out[2]
		sigma_q[i]=hp.sphtfunc.smoothing(tmp_q,fwhm=np.pi/180)
		sigma_u[i]=hp.sphtfunc.smoothing(tmp_u,fwhm=np.pi/180)
	#need to find pixels for just the CMB field from QUIET
	dx=1./(60.)*3
	nx=np.int(15/dx)
	ny=nx
	alpha_all_sky=[[],[]]
	quiet_mask=np.repeat(hp.UNSEEN,npix)
	q_all_sky=[[],[]]
	u_all_sky=[[],[]]
	for p in xrange(len(centers)):
		coords=regioncoords(centers[p,0],centers[p,1],dx,nx,ny)
		coords_sky=SkyCoord(ra=coords[:,0],dec=coords[:,1],unit=u.degree,frame='fk5')
		phi=coords_sky.galactic.l.deg*np.pi/180.
		theta=(90-coords_sky.galactic.b.deg)*np.pi/180.
		pixels=np.reshape(hp.ang2pix(128,theta,phi),(nx,ny))
		unique_pix=np.unique(pixels)
		#fit these pixels
		t0=time()
		pool=multiprocessing.Pool()
		outputs=pool.map(fit_pixel,[[wl, q_array[:,i],u_array[:,i],sigma_q[:,i],sigma_u[:,i]] for i in pixels.flat])
		pool.close()
		pool.join()
		fit_time=time()-t0
		#reconstruct 4 QUIET fields
		
		print 'Fitting occurred in '+'{:.3f}'.format(fit_time/60.)+' minutes'
		q_map,u_map,alpha_map,dq_map,du_map,dalpha_map,chisquare,dof=list(zip(*outputs))
		q_map=np.reshape(q_map,(nx,ny))
		u_map=np.reshape(u_map,(nx,ny))
		alpha_map=np.reshape(alpha_map,(nx,ny))
		dq_map=np.reshape(dq_map,(nx,ny))
		du_map=np.reshape(du_map,(nx,ny))
		dalpha_map=np.reshape(dalpha_map,(nx,ny))
		chisquare=np.reshape(chisquare,(nx,ny))
		dof=np.reshape(dof,(nx,ny))
		
		for pix in unique_pix:
			ind=np.where(pixels == pix)
			alpha_all_sky[0].append(pix)
			alpha_all_sky[1].append(np.average([alpha_map[ind[0][i],ind[1][i]] for i in xrange(len(ind[0]))]))
			q_all_sky[0].append(pix)
			q_all_sky[1].append(np.average([alpha_map[ind[0][i],ind[1][i]] for i in xrange(len(ind[0]))]))
			u_all_sky[0].append(pix)
			u_all_sky[1].append(np.average([alpha_map[ind[0][i],ind[1][i]] for i in xrange(len(ind[0]))]))
			quiet_mask[pix]=1

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
		alpha_sky_head=fits.ImageHDU(alpha_all_sky,name='ALPHA ALL SKY')
		alpha_sky_head.header['TUNIT1']=('rad/m^2', 'Physical Units of Map')	
		alpha_sky_head.header["NSIDE"]=1024
		alpha_sky_head.header["ORDERING"]='RING'
		alpha_sky_head.header["UNSEEN"]=hp.UNSEEN
		q_sky_head=fits.ImageHDU(q_all_sky,name='STOKES Q ALL SKY')
		q_sky_head.header['TUNIT1']=('K_{CMB} Thermodynamic', 'Physical Units of Map')	
		q_sky_head.header["NSIDE"]=1024
		q_sky_head.header["ORDERING"]='RING'
		q_sky_head.header["UNSEEN"]=hp.UNSEEN
		u_sky_head=fits.ImageHDU(u_all_sky,name='STOKES U ALL SKY')
		u_sky_head.header['TUNIT1']=('K_{CMB} Thermodynamic', 'Physical Units of Map')	
		u_sky_head.header["NSIDE"]=1024
		u_sky_head.header["ORDERING"]='RING'
		u_sky_head.header["UNSEEN"]=hp.UNSEEN
		hdulist=fits.HDUList([prim,alpha_head,q_head,u_head,dalpha_head,dq_head,du_head,alpha_sky_head,q_sky_head,u_sky_head])
		hdulist.writeto('alpha_quiet_simul_kmle_2-band_cmb'+str(p+1).zfill(1)+'_simul.fits')
	
	total_time=time()-t1
	print 'Total computation time: '+'{:.3f}'.format(total_time/60.)+' minutes'
