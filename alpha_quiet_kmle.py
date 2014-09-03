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
def rotate_tqu(map_in,wl,alpha):  #rotates tqu map by phase
	npix=hp.nside2npix(hp.get_nside(map_in))
	tmp_map=np.zeros((3,npix))
	tmp_map[0]=map_in[0]
	tmp_map[1]=map_in[1]*np.cos(2*alpha*wl**2) + map_in[2]*np.sin(2*alpha*wl**2)
	tmp_map[2]=-map_in[1]*np.sin(2*alpha*wl**2) + map_in[2]*np.cos(2*alpha*wl**2)
	return tmp_map

def alpha_function(P,data):
	x,q,u,q_err,u_err=data
	Q_mod= np.array(P[0] + 2*P[2]*x**2*P[1])*1e-9
	U_mod= np.array(P[1] - 2*P[2]*x**2*P[0])*1e-9
	small_angle=(np.sin(2*P[2]*x**2)-2*P[2]*x**2)/np.sin(2*P[2]*x**2)
	return np.concatenate([(q-Q_mod)/q_err,(u-U_mod)/u_err,small_angle/1.])

def fit_pixel(inputs):
	p0=[1e2,1e2,10]
	fitobj=kmpfit.Fitter(residuals=alpha_function,data=inputs)
	fitobj.fit(params0=p0)
	parms=np.array(fitobj.params)
	parms[:2]*=1e-9
	dparms=np.array(fitobj.stderr)
	dparms[:2]*=1e-9
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
		tmp_cmb=rotate_tqu(simul_cmb,wl_p[i],alpha_radio);
		tmp_out=hp.ud_grade(np.random.normal(0,1,npix1)*noise_const_q[i],nside_out=nside,order_in='nested',order_out='nested');
		sigma_q_1[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
		tmp_out=hp.ud_grade(np.random.normal(0,1,npix1)*noise_const_q[i],nside_out=nside,order_in='nested',order_out='nested');
		sigma_u_1[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
		tmp_out=hp.ud_grade(tmp_cmb,nside_out=nside,order_in='nested',order_out='nested');
		tmp_out=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383,pol=1)
		q_array_1[i]=tmp_out[1]+sigma_q_1[i]
		u_array_1[i]=tmp_out[2]+sigma_u_1[i]
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
		wmap_counts=hp.read_map(wmap_files[i],nest=1,field=3);
		tmp_cmb=rotate_tqu(simul_cmb,wl_w[i],alpha_radio);
		tmp_out=hp.ud_grade(np.random.normal(0,1,npix1)*noise_const_q[i]/np.sqrt(wmap_counts),nside_out=nside,order_in='nested',order_out='nested');
		sigma_q_2[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
		tmp_out=hp.ud_grade(np.random.normal(0,1,npix1)*noise_const_q[i]/np.sqrt(wmap_counts),nside_out=nside,order_in='nested',order_out='nested');
		sigma_u_2[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
		tmp_out=hp.ud_grade(tmp_cmb,nside_out=nside,order_in='nested',order_out='nested');
		tmp_out=hp.sphtfunc.smoothing(tmp_out,lmax=383,pol=1,fwhm=np.pi/180.)
		q_array_2[i]=tmp_out[1]+sigma_q_2[i]
		u_array_2[i]=tmp_out[2]+sigma_u_2[i]
	
	

	bands_q=[43.1,94.5]
	q_fwhm=[27.3,11.7]
	noise_const_q=np.array([6./q_fwhm[0],6./q_fwhm[1]])*1e-6
	centers=np.array([convertcenter([12,4],-39),convertcenter([5,12],-39),convertcenter([0,48],-48),convertcenter([22,44],-36)])
	wl_q=np.array([299792458./(band*1e9) for band in bands_q])
	num_wl=len(wl_q)
	q_array_3=np.zeros((num_wl,npix))
	u_array_3=np.zeros((num_wl,npix))
	sigma_q_3=np.zeros((num_wl,npix))
	sigma_u_3=np.zeros((num_wl,npix))	
	simul_cmb=hp.ud_grade(simul_cmb,nside_out=128,order_in='nested',order_out='nested')
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=128,order_in='nested',order_out='nested');
	for i in range(num_wl):
		tmp_cmb=rotate_tqu(simul_cmb,wl_q[i],alpha_radio);
		tmp_out=np.random.normal(0,1,npix)*noise_const_q[i]
		sigma_q_3[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
		tmp_out=np.random.normal(0,1,npix)*noise_const_q[i]
		sigma_u_3[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
		tmp_out=hp.sphtfunc.smoothing(tmp_cmb,fwhm=np.pi/180.,lmax=383,pol=1)
		q_array_3[i]=tmp_out[1]+sigma_q_3[i]
		u_array_3[i]=tmp_out[2]+sigma_u_3[i]

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
	for p in xrange(len(centers)):
		coords=regioncoords(centers[p,0],centers[p,1],dx,nx,ny)
		coords_sky=SkyCoord(ra=coords[:,0],dec=coords[:,1],unit=u.degree,frame='fk5')
		phi=coords_sky.galactic.l.deg*np.pi/180.
		theta=(90-coords_sky.galactic.b.deg)*np.pi/180.
		pixels=np.reshape(hp.ang2pix(128,theta*np.pi/180.,phi*np.pi/180.),(nx,ny))
		#fit these pixels
		t0=time()
		pool=multiprocessing.Pool()
		outputs=pool.map(fit_pixel,[[wl, q_array[:,i],u_array[:,i],sigma_q[:,i],sigma_u[:,i]] for i in pixels.flat])
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
		q_map=np.reshape(q_map,(nx,ny))
		u_map=np.reshape(u_map,(nx,ny))
		alpha_map=np.reshape(alpha_map,(nx,ny))
		dq_map=np.reshape(dq_map,(nx,ny))
		du_map=np.reshape(du_map,(nx,ny))
		dalpha_map=np.reshape(dalpha_map,(nx,ny))
		chisquare=np.reshape(chisquare,(nx,ny))
		dof=np.reshape(dof,(nx,ny))
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
		hdulist=fits.HDUList([prim,alpha_head,q_head,u_head,dalpha_head,dq_head,du_head,chi_head,dof_head])
		hdulist.writeto('alpha_quiet_kmle_cmb'+str(p+1).zfill(1)+'.fits')
	
	total_time=time()-t1
	print 'Total computation time: '+'{:.3f}'.format(total_time/60.)+' minutes'
