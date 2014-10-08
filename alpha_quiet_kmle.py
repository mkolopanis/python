import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import ipdb; 
from time import time
import multiprocessing
from kapteyn import kmpfit
from rotate_tqu import rotate_tqu
from alpha_function import alpha_function
def fit_pixel(inputs):
	p0=[1.,1.,10]
	fitobj=kmpfit.Fitter(residuals=alpha_function,data=inputs)
	fitobj.fit(params0=p0)
	parms=np.array(fitobj.params)
	parms[:2]*=1e-7
	dparms=np.array(fitobj.xerror)
	dparms[:2]*=1e-7
	return np.concatenate([parms,dparms,[fitobj.chi2_min],[fitobj.dof]])

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
	
	alpha_radio=hp.read_map(radio_file,hdu='maps/phi');
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=1024,order_in='ring',order_out='ring')
	npix1=hp.nside2npix(2048)
	t2=time()
	
	for i in range(num_wl):
		if i in range(3):	
			npix1=hp.nside2npix(1024)
			tmp_cmb1=hp.ud_grade(simul_cmb,nside_out=1024,order_in='ring')
			
		if i==3:
			npix1=hp.nside2npix(2048)
			tmp_cmb1=simul_cmb
			alpha_radio=hp.ud_grade(alpha_radio,nside_out=2048,order_in='ring',order_out='ring')

		tmp_cmb=hp.sphtfunc.smoothing(tmp_cmb1,pol=1,fwhm=beamfwhm[i]*np.pi/(180.*60.))
		tmp_cmb=rotate_tqu(tmp_cmb,wl_p[i],alpha_radio);
		tmp_q=np.random.normal(0,1,npix1)*noise_const_q[i]
		tmp_u=np.random.normal(0,1,npix1)*noise_const_q[i]
		tmp_out=hp.ud_grade(tmp_cmb+np.array([np.zeros(npix1),tmp_q,tmp_u]),nside_out=nside,order_in='ring',order_out='ring');
		tmp_out=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383,pol=1)
		q_array_1[i]=tmp_out[1]
		u_array_1[i]=tmp_out[2]
		sigma_q_1[i]=hp.sphtfunc.smoothing(hp.ud_grade(tmp_q,nside_out=nside,order_in='ring',order_out='ring'),fwhm=np.pi/180.,lmax=383);
		sigma_u_1[i]=hp.sphtfunc.smoothing(hp.ud_grade(tmp_u,nside_out=nside,order_in='ring',order_out='ring'),fwhm=np.pi/180.,lmax=383);
	t3=time()
	print 'This computation took '+"{:.3f}".format((t3-t2)/60.)+' minutes'
	tmp_cmp1=bytearray()
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
	
	simul_cmb=hp.ud_grade(simul_cmb,nside_out=512,order_in='ring',order_out='ring')
	
	
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=512,order_in='ring',order_out='ring');
	
	for i in range(num_wl):
		tmp_cmb=hp.sphtfunc.smoothing(simul_cmb,pol=1,fwhm=np.pi/(180.)*w_fwhm[i])
		wmap_counts=hp.read_map(wmap_files[i],field=3);
		tmp_cmb=rotate_tqu(tmp_cmb,wl_w[i],alpha_radio);
		tmp_q=np.random.normal(0,1,npix1)*noise_const_q[i]/np.sqrt(wmap_counts)
		tmp_u=np.random.normal(0,1,npix1)*noise_const_q[i]/np.sqrt(wmap_counts)
		tmp_out=hp.ud_grade(tmp_cmb+np.array([np.zeros(npix1),tmp_q,tmp_u]),nside_out=nside,order_in='ring',order_out='ring');
		tmp_out=hp.sphtfunc.smoothing(tmp_out,lmax=383,pol=1,fwhm=np.pi/180.)
		q_array_2[i]=tmp_out[1]
		u_array_2[i]=tmp_out[2]
		sigma_q_2[i]=hp.sphtfunc.smoothing(hp.ud_grade(tmp_q,nside_out=nside,order_in='ring',order_out='ring'),fwhm=np.pi/180.,lmax=383);
		sigma_u_2[i]=hp.sphtfunc.smoothing(hp.ud_grade(tmp_u,nside_out=nside,order_in='ring',order_out='ring'),fwhm=np.pi/180.,lmax=383);
	
	

	bands_q=[43.1,94.5]
	q_fwhm=[27.3,11.7]
	noise_const_q=np.array([6./fwhm for fwhm in q_fwhm])*1e-6
	centers=np.array([convertcenter([12,4],-39),convertcenter([5,12],-39),convertcenter([0,48],-48),convertcenter([22,44],-36)])
	wl_q=np.array([299792458./(band*1e9) for band in bands_q])
	num_wl=len(wl_q)
	q_array_3=np.zeros((num_wl,npix))
	u_array_3=np.zeros((num_wl,npix))
	sigma_q_3=np.zeros((num_wl,npix))
	sigma_u_3=np.zeros((num_wl,npix))	
	simul_cmb=hp.ud_grade(simul_cmb,nside_out=128,order_in='ring',order_out='ring')
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=128,order_in='ring',order_out='ring');
	for i in range(num_wl):
		tmp_cmb=hp.sphtfunc.smoothing(simul_cmb,pol=1,fwhm=np.pi/(180.*60.)*q_fwhm[i])
		tmp_cmb=rotate_tqu(tmp_cmb,wl_q[i],alpha_radio);
		sigma_q_3[i]=np.random.normal(0,1,npix)*noise_const_q[i]
		sigma_u_3[i]=np.random.normal(0,1,npix)*noise_const_q[i]
		tmp_out=hp.sphtfunc.smoothing(tmp_cmb+np.array([np.zeros(npix),sigma_q_3[i],sigma_u_3[i]]),fwhm=np.pi/180.,lmax=383,pol=1)
		q_array_3[i]=tmp_out[1]
		u_array_3[i]=tmp_out[2]
		sigma_q_3[i]=hp.sphtfunc.smoothing(sigma_q_3[i],fwhm=np.pi/180.,lmax=383);
		sigma_u_3[i]=hp.sphtfunc.smoothing(sigma_u_3[i],fwhm=np.pi/180.,lmax=383);

	wl=np.concatenate((wl_p,wl_w,wl_q))
	
	q_array=np.concatenate((q_array_1,q_array_2,q_array_3))
	u_array=np.concatenate((u_array_1,u_array_2,u_array_3))
	sigma_q=np.concatenate((sigma_q_1,sigma_q_2,sigma_q_3))
	sigma_u=np.concatenate((sigma_u_1,sigma_u_2,sigma_u_3))
	
	new_index=np.argsort(wl)
	new_index=new_index[::-1]
	wl=wl[new_index]
	nuw_wl=len(new_index)
	q_array=q_array[new_index]
	u_array=u_array[new_index]
	sigma_q=sigma_q[new_index]
	sigma_u=sigma_u[new_index]
	
	bands=bands_w+bands_p+bands_q
	bands.sort()
	
	#need to find pixels for just the CMB field from QUIET
	dx=1./(60.)*6
	nx=np.int(15/dx)
	ny=nx
	quiet_mask=np.zeros(npix)
	q_all_sky=np.zeros(npix)
	u_all_sky=np.zeros(npix)
	alpha_all_sky=np.zeros(npix)
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
			alpha_all_sky[pix]=np.average([alpha_map[ind[0][i],ind[1][i]] for i in xrange(len(ind[0]))])
			q_all_sky[pix]=np.average([q_map[ind[0][i],ind[1][i]] for i in xrange(len(ind[0]))])
			u_all_sky[pix]=np.average([u_map[ind[0][i],ind[1][i]] for i in xrange(len(ind[0]))])
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
		q_sky_head=fits.ImageHDU(q_all_sky,name='STOKES Q ALL SKY')
		u_sky_head=fits.ImageHDU(u_all_sky,name='STOKES U ALL SKY')
		q_sky_head.header['TUNIT1']=('K_{CMB} Thermodynamic', 'Physical Units of Map')	
		u_sky_head.header['TUNIT1']=('K_{CMB} Thermodynamic', 'Physical Units of Map')	
		hdulist=fits.HDUList([prim,alpha_head,q_head,u_head,dalpha_head,dq_head,du_head,alpha_sky_head,q_sky_head,u_sky_head])
		hdulist.writeto('alpha_quiet_kmle_cmb'+str(p+1).zfill(1)+'.fits')
	
	total_time=time()-t1
	print 'Total computation time: '+'{:.3f}'.format(total_time/60.)+' minutes'
