import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from astropy.io import fits
import ipdb
import make_quiet_field as simulate_fields
import radialProfile
import plot_binned
import subprocess
import json
import rotate_tqu
import scipy.fftpack as fft

smoothing_scale=30.0

def faraday_correlate_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands,beam=False,polar_mask=False):
	print "Computing Cross Correlations for Bands "+str(bands)

	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi')
	alpha_radio=hp.ud_grade(alpha_radio,1024)
	iqu_band_i=hdu_i['stokes iqu'].data
	iqu_band_j=hdu_j['stokes iqu'].data
	sigma_i=hdu_i['Q/U UNCERTAINTIES'].data
	sigma_j=hdu_j['Q/U UNCERTAINTIES'].data
	field_pixels=hdu_i['SQUARE PIXELS'].data
	
	q_fwhm=[27.3,11.7]
	noise_const=np.array([36./f for f in q_fwhm])*1e-6
	npix=hp.nside2npix(1024)
	sigma_i=[noise_const[0]*np.random.normal(0,1,npix),noise_const[1]*np.random.normal(0,1,npix)]
	sigma_j=[noise_const[0]*np.random.normal(0,1,npix),noise_const[1]*np.random.normal(0,1,npix)]
	iqu_band_i[1]+=sigma_i[0]
	iqu_band_i[2]+=sigma_i[1]
	iqu_band_j[1]+=sigma_j[0]
	iqu_band_j[2]+=sigma_j[1]
	hdu_i.close()
	hdu_j.close()
	
	
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),lmax=383)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),lmax=383)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	const=2.*(wl_i**2-wl_j**2)	

	Delta_Q=(iqu_band_i[1]-iqu_band_j[1])/const
	Delta_U=(iqu_band_i[2]-iqu_band_j[2])/const
	alpha_u=alpha_radio*iqu_band_j[2] 
	alpha_q=-alpha_radio*iqu_band_j[1]
	
	if polar_mask:
		P=np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)
		bad_pix=np.where( P < .2e-6)
		Delta_Q[bad_pix]=0
		Delta_U[bad_pix]=0
		alpha_u[bad_pix]=0
		alpha_q[bad_pix]=0

	cross1_array=[]
	cross2_array=[]
	cross3_array=[]
	L=15*(np.pi/180.)
	k=np.arange(1,np.round(500*(L/(2*np.pi))))
	l=2*np.pi*k/L
	Bl_factor=np.repeat(1,len(k))
	if beam:
		Bl_factor=hp.gauss_beam(np.pi/180.,383)
	for field1 in xrange(4):
		pix_cmb=field_pixels.field(field1)	
		nx=np.sqrt(pix_cmb.shape[0])	
		flat_dq=np.reshape(Delta_Q[pix_cmb],(nx,nx))	
		flat_du=np.reshape(Delta_U[pix_cmb],(nx,nx))
		flat_aq=np.reshape(alpha_q[pix_cmb],(nx,nx))	
		flat_au=np.reshape(alpha_u[pix_cmb],(nx,nx))	
		
		dq_alm=fft.fftshift(fft.fft2(flat_dq,shape=[450,450]))
		du_alm=fft.fftshift(fft.fft2(flat_du,shape=[450,450]))
		aq_alm=fft.fftshift(fft.fft2(flat_aq,shape=[450,450]))
		au_alm=fft.fftshift(fft.fft2(flat_au,shape=[450,450]))
		pw2d_qau=np.real(dq_alm*np.conjugate(au_alm))		
		pw2d_uaq=np.real(du_alm*np.conjugate(aq_alm))		
		pw1d_qau=radialProfile.azimuthalAverage(pw2d_qau)
		pw1d_uaq=radialProfile.azimuthalAverage(pw2d_uaq)
		tmp_cl1=pw1d_qau[k.astype(int)-1]*L**2
		tmp_cl2=pw1d_uaq[k.astype(int)-1]*L**2
		#	index=np.where( (np.sqrt(x**2+y**2) <= k[num_k] +1)  & ( np.sqrt(x**2 + y**2) >= k[num_k] -1) )
		#	tmp1= np.sum(pw2d_qau[index])/(np.pi*( (k[num_k]+1)**2 -(k[num_k]-1)**2 ) )
		#	tmp2= np.sum(pw2d_uaq[index])/(np.pi*( (k[num_k]+1)**2 -(k[num_k]-1)**2 ) )
		#	tmp_cl1[num_k]=L**2*tmp1
		#	tmp_cl2[num_k]=L**2*tmp2
		cross1_array.append(tmp_cl1/Bl_factor)
		cross2_array.append(tmp_cl2/Bl_factor)
	
	cross1=np.mean(cross1_array,axis=0)	##Average over all Cross Spectra
	cross2=np.mean(cross2_array,axis=0)	##Average over all Cross Spectra
	hp.write_cl('cl_'+bands+'_FR_QxaU.fits',cross1)
	hp.write_cl('cl_'+bands+'_FR_UxaQ.fits',cross2)
	return (cross1,cross2)

def faraday_noise_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands,beam=False,polar_mask=False):

	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi')
	alpha_radio=hp.ud_grade(alpha_radio,1024)
	iqu_band_i=hdu_i['stokes iqu'].data
	iqu_band_j=hdu_j['stokes iqu'].data
	sigma_i=hdu_i['Q/U UNCERTAINTIES'].data
	sigma_j=hdu_j['Q/U UNCERTAINTIES'].data
	field_pixels=hdu_i['SQUARE PIXELS'].data
	
	q_fwhm=[27.3,11.7]
	noise_const=np.array([36./f for f in q_fwhm])*1e-6
	npix=hp.nside2npix(1024)
	sigma_i=[noise_const[0]*np.random.normal(0,1,npix),noise_const[1]*np.random.normal(0,1,npix)]
	sigma_j=[noise_const[0]*np.random.normal(0,1,npix),noise_const[1]*np.random.normal(0,1,npix)]
	iqu_band_i[1]=sigma_i[0]
	iqu_band_i[1]=sigma_i[0]
	iqu_band_i[2]=sigma_i[1]
	iqu_band_j[1]=sigma_j[0]
	iqu_band_j[2]=sigma_j[1]
	hdu_i.close()
	hdu_j.close()
	
	
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),lmax=383)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),lmax=383)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	const=2.*(wl_i**2-wl_j**2)	

	Delta_Q=(iqu_band_i[1]-iqu_band_j[1])/const
	Delta_U=(iqu_band_i[2]-iqu_band_j[2])/const
	alpha_u=alpha_radio*iqu_band_j[2] 
	alpha_q=-alpha_radio*iqu_band_j[1]

	if polar_mask:
		P=np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)
		bad_pix=np.where( P < .2e-6)
		Delta_Q[bad_pix]=0
		Delta_U[bad_pix]=0
		alpha_u[bad_pix]=0
		alpha_q[bad_pix]=0

	cross1_array=[]
	cross2_array=[]
	cross3_array=[]
	L=15*(np.pi/180.)
	k=np.arange(1,np.round(500*(L/(2*np.pi))))
	l=2*np.pi*k/L
	Bl_factor=np.repeat(1,len(k))
	if beam:
		Bl_factor=hp.gauss_beam(np.pi/180.,383)
	for field1 in xrange(4):
		pix_cmb=field_pixels.field(field1)	
		nx=np.sqrt(pix_cmb.shape[0])	
		flat_dq=np.reshape(Delta_Q[pix_cmb],(nx,nx))	
		flat_du=np.reshape(Delta_U[pix_cmb],(nx,nx))
		flat_aq=np.reshape(alpha_q[pix_cmb],(nx,nx))	
		flat_au=np.reshape(alpha_u[pix_cmb],(nx,nx))	
		
		dq_alm=fft.fftshift(fft.fft2(flat_dq,shape=[450,450]))
		du_alm=fft.fftshift(fft.fft2(flat_du,shape=[450,450]))
		aq_alm=fft.fftshift(fft.fft2(flat_aq,shape=[450,450]))
		au_alm=fft.fftshift(fft.fft2(flat_au,shape=[450,450]))
	
		pw2d_qau=np.real(dq_alm*np.conjugate(au_alm))		
		pw2d_uaq=np.real(du_alm*np.conjugate(aq_alm))		
		pw1d_qau=radialProfile.azimuthalAverage(pw2d_qau)
		pw1d_uaq=radialProfile.azimuthalAverage(pw2d_uaq)
		tmp_cl1=pw1d_qau[k.astype(int)-1]*L**2
		tmp_cl2=pw1d_uaq[k.astype(int)-1]*L**2
		#	index=np.where( (np.sqrt(x**2+y**2) <= k[num_k] +1)  & ( np.sqrt(x**2 + y**2) >= k[num_k] -1) )
		#	tmp1= np.sum(pw2d_qau[index])/(np.pi*( (k[num_k]+1)**2 -(k[num_k]-1)**2 ) )
		#	tmp2= np.sum(pw2d_uaq[index])/(np.pi*( (k[num_k]+1)**2 -(k[num_k]-1)**2 ) )
		#	tmp_cl1[num_k]=L**2*tmp1
		#	tmp_cl2[num_k]=L**2*tmp2
		cross1_array.append(tmp_cl1/Bl_factor)
		cross2_array.append(tmp_cl2/Bl_factor)
	
	
	cross1=np.mean(cross1_array,axis=0)	##Average over all Cross Spectra
	cross2=np.mean(cross2_array,axis=0)	##Average over all Cross Spectra
	hp.write_cl('cl_'+bands+'_FR_QxaU.fits',cross1)
	hp.write_cl('cl_'+bands+'_FR_UxaQ.fits',cross2)
	return (cross1,cross2)

def faraday_theory_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands_name,beam=False,polar_mask=False):
	print "Computing Cross Correlations for Bands "+str(bands_name)

#	radio_file='/data/wmap/faraday_MW_realdata.fits'
#	cl_file='/home/matt/wmap/simul_scalCls.fits'
#	nside=1024
#	npix=hp.nside2npix(nside)
#	
#	cls=hp.read_cl(cl_file)
#	simul_cmb=hp.sphtfunc.synfast(cls,nside,fwhm=0.,new=1,pol=1);
#	
#	alpha_radio=hp.read_map(radio_file,hdu='maps/phi');
#	alpha_radio=hp.ud_grade(alpha_radio,nside_out=nside,order_in='ring',order_out='ring')
#	bands=[43.1,94.5]
	q_fwhm=[27.3,11.7]
#	wl=np.array([299792458./(band*1e9) for band in bands])
#	num_wl=len(wl)
#	t_array=np.zeros((num_wl,npix))	
#	q_array=np.zeros((num_wl,npix))
#	u_array=np.zeros((num_wl,npix))
#	for i in range(num_wl):
#		tmp_cmb=rotate_tqu.rotate_tqu(simul_cmb,wl[i],alpha_radio);
#		t_array[i],q_array[i],u_array[i]=tmp_cmb
#	iqu_band_i=[t_array[0],q_array[0],u_array[0]]	
#	iqu_band_j=[t_array[1],q_array[1],u_array[1]]	


	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi')
	alpha_radio=hp.ud_grade(alpha_radio,1024)
	iqu_band_i=hdu_i['stokes iqu'].data
	iqu_band_j=hdu_j['stokes iqu'].data
	field_pixels=hdu_i['SQUARE PIXELS'].data
	hdu_i.close()
	hdu_j.close()
	
	
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),lmax=383)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),lmax=383)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	const=2.*(wl_i**2-wl_j**2)	

	Delta_Q=(iqu_band_i[1]-iqu_band_j[1])/const
	Delta_U=(iqu_band_i[2]-iqu_band_j[2])/const
	alpha_u=alpha_radio*iqu_band_j[2] 
	alpha_q=-alpha_radio*iqu_band_j[1]
	
	if polar_mask:
		P=np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)
		bad_pix=np.where( P < .2e-6)
		Delta_Q[bad_pix]=0
		Delta_U[bad_pix]=0
		alpha_u[bad_pix]=0
		alpha_q[bad_pix]=0

	cross1_array=[]
	cross2_array=[]
	L=15*(np.pi/180.)
	k=np.arange(1,np.round(500*(L/(2*np.pi))))
	l=2*np.pi*k/L
	Bl_factor=np.repeat(1,len(k))
	if beam:
		Bl_factor=hp.gauss_beam(np.pi/180.,383)
	for field1 in xrange(4):
		pix_cmb=field_pixels.field(field1)	
		nx=np.sqrt(pix_cmb.shape[0])	
		flat_dq=np.reshape(Delta_Q[pix_cmb],(nx,nx))	
		flat_du=np.reshape(Delta_U[pix_cmb],(nx,nx))
		flat_aq=np.reshape(alpha_q[pix_cmb],(nx,nx))	
		flat_au=np.reshape(alpha_u[pix_cmb],(nx,nx))	
		
		dq_alm=fft.fftshift(fft.fft2(flat_dq,shape=[450,450]))
		du_alm=fft.fftshift(fft.fft2(flat_du,shape=[450,450]))
		aq_alm=fft.fftshift(fft.fft2(flat_aq,shape=[450,450]))
		au_alm=fft.fftshift(fft.fft2(flat_au,shape=[450,450]))
	
		pw2d_qau=np.real(dq_alm*np.conjugate(au_alm))		
		pw2d_uaq=np.real(du_alm*np.conjugate(aq_alm))		
		pw1d_qau=radialProfile.azimuthalAverage(pw2d_qau)
		pw1d_uaq=radialProfile.azimuthalAverage(pw2d_uaq)
		tmp_cl1=pw1d_qau[k.astype(int)-1]*L**2
		tmp_cl2=pw1d_uaq[k.astype(int)-1]*L**2
		#	index=np.where( (np.sqrt(x**2+y**2) <= k[num_k] +1)  & ( np.sqrt(x**2 + y**2) >= k[num_k] -1) )
		#	tmp1= np.sum(pw2d_qau[index])/(np.pi*( (k[num_k]+1)**2 -(k[num_k]-1)**2 ) )
		#	tmp2= np.sum(pw2d_uaq[index])/(np.pi*( (k[num_k]+1)**2 -(k[num_k]-1)**2 ) )
		#	tmp_cl1[num_k]=L**2*tmp1
		#	tmp_cl2[num_k]=L**2*tmp2
		cross1_array.append(tmp_cl1/Bl_factor)
		cross2_array.append(tmp_cl2/Bl_factor)

	cross1=np.mean(cross1_array,axis=0)	##Average over all Cross Spectra
	cross2=np.mean(cross2_array,axis=0)	##Average over all Cross Spectra
	hp.write_cl('cl_'+bands_name+'_FR_QxaU.fits',cross1)
	hp.write_cl('cl_'+bands_name+'_FR_UxaQ.fits',cross2)
	return (cross1,cross2)
def plot_mc():

	theory1=hp.read_cl('cl_theory_FR_QxaU.fits')
	theory2=hp.read_cl('cl_theory_FR_UxaQ.fits')
	bins=[1,5,10,20,50]
	f=open('cl_array_FR_QxaU.json','r')
	cross1_array=json.load(f)
	f.close()	
	f=open('cl_array_FR_UxaQ.json','r')
	cross2_array=json.load(f)
	f.close()	
	f=open('cl_noise_FR_QxaU.json','r')
	noise1_array=json.load(f)
	f.close()	
	f=open('cl_noise_FR_UxaQ.json','r')
	noise2_array=json.load(f)
	f.close()	
#	bl_27=hp.gauss_beam(27.3*(np.pi/(180*60.)),lmax=383)
#	bl_11=hp.gauss_beam(11.7*(np.pi/(180*60.)),lmax=383)
#	bl_factor=bl_11*bl_27
	L=15*(np.pi/180.)
	k=np.arange(1,500*(L/(2*np.pi)))
	l=2*np.pi*k/L
	cross1=np.mean(cross1_array,axis=0)
	noise1=np.mean(noise1_array,axis=0)
	dcross1=np.std(cross1_array,axis=0)
	plt.figure()
	plt.plot(l,l*(l+1)/(2*np.pi)*theory1,'r-')
	plt.plot(l,l*(l+1)/(2*np.pi)*(cross1-noise1),'k')
	plt.errorbar(l,l*(l+1)/(2*np.pi)*(cross1-noise1),yerr=l*(l+1)/(2*np.pi)*dcross1,color='black')
	plt.title('Cross 43x95 FR QxaU')
	plt.ylabel('$\\frac{\ell(\ell+1)}{2\pi}C_{\ell}\ \\frac{\mu K^{2}rad}{m^{4}}$')
	plt.xlabel('$\ell$')

	plt.savefig('Cross_43x95_FR_QxaU_flat.eps')
	plt.savefig('Cross_43x95_FR_QxaU_flat.png',format='png')
	
	cross2=np.mean(cross2_array,axis=0)
	noise2=np.mean(noise2_array,axis=0)
	dcross2=np.std(cross2_array,axis=0)
	plt.figure()
	plt.plot(l,l*(l+1)/(2*np.pi)*theory2,'r-')
	plt.plot(l,l*(l+1)/(2*np.pi)*(cross2-noise2),'k')
	plt.errorbar(l,l*(l+1)/(2*np.pi)*(cross2-noise2),yerr=l*(l+1)/(2*np.pi)*dcross2,color='black')
	plt.title('Cross 43x95 FR UxaQ')
	plt.ylabel('$\\frac{\ell(\ell+1)}{2\pi}C_{\ell}\ \\frac{\mu K^{2}rad}{m^{4}}$')
	plt.xlabel('$\ell$')

	plt.savefig('Cross_43x95_FR_UxaQ_flat.eps')
	plt.savefig('Cross_43x95_FR_UxaQ_flat.png',format='png')

def main():
	map_prefix='/home/matt/quiet/quiet_maps/'
	i_file=map_prefix+'quiet_simulated_43.1'
	j_file=map_prefix+'quiet_simulated_94.5'
	alpha_file='/data/wmap/faraday_MW_realdata.fits'
	bands=[43.1,94.5]
	names=['43','95']
	wl=np.array([299792458./(band*1e9) for band in bands])
	N_runs=100
	bins=[1,5,10,20,50]
	cross1_array=[]
	cross2_array=[]
	noise1_array=[]
	noise2_array=[]
	theory1_array=[]
	theory2_array=[]
	
	for i in xrange(N_runs):	
	
		simulate_fields.main()
		#for n in xrange(N_runs):
		ttmp1,ttmp2=faraday_theory_quiet(i_file+'.fits',j_file+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1])
		theory1_array.append(ttmp1)
		theory2_array.append(ttmp2)

		tmp1,tmp2=faraday_correlate_quiet(i_file+'.fits',j_file+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1])

		ntmp1,ntmp2=faraday_noise_quiet(i_file+'.fits',j_file+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1])

		cross1_array.append(tmp1)
		cross2_array.append(tmp2)
		noise1_array.append(ntmp1)
		noise2_array.append(ntmp2)

	f=open('cl_theory_FR_QxaU.json','w')
	json.dump([[a for a in theory1_array[i]] for i in xrange(len(cross1_array))],f)
	f.close()	
	f=open('cl_theory_FR_UxaQ.json','w')
	json.dump([[a for a in theory2_array[i]] for i in xrange(len(cross2_array))],f)
	f.close()	
	theory1=np.mean(theory1_array,axis=0)
	theory2=np.mean(theory2_array,axis=0)
	hp.write_cl('cl_theory_FR_QxaU.fits',theory1)
	hp.write_cl('cl_theory_FR_UxaQ.fits',theory2)
	
	f=open('cl_array_FR_QxaU.json','w')
	json.dump([[a for a in cross1_array[i]] for i in xrange(len(cross1_array))],f)
	f.close()	
	f=open('cl_array_FR_UxaQ.json','w')
	json.dump([[a for a in cross2_array[i]] for i in xrange(len(cross2_array))],f)
	f.close()	
	f=open('cl_noise_FR_QxaU.json','w')
	json.dump([[a for a in noise1_array[i]] for i in xrange(len(noise1_array))],f)
	f.close()	
	f=open('cl_noise_FR_UxaQ.json','w')
	json.dump([[a for a in noise2_array[i]] for i in xrange(len(noise2_array))],f)
	f.close()	
	
	L=15*(np.pi/180.)
	k=np.arange(1,500*(L/(2*np.pi)))
	l=2*np.pi*k/L
	cross1=np.mean(cross1_array,axis=0)
	noise1=np.mean(noise1_array,axis=0)
	dcross1=np.std(np.subtract(cross1_array,noise1),axis=0)
	plt.figure()
	plt.plot(l,l*(l+1)/(2*np.pi)*theory1,'r-')
	plt.plot(l,l*(l+1)/(2*np.pi)*(cross1-noise1),'k')
	plt.errorbar(l,l*(l+1)/(2*np.pi)*(cross1-noise1),yerr=l*(l+1)/(2*np.pi)*dcross1,color='black')
	plt.title('Cross 43x95 FR QxaU')
	plt.ylabel('$\\frac{\ell(\ell+1)}{2\pi}C_{\ell}\ \\frac{\mu K^{2}rad}{m^{4}}$')
	plt.xlabel('$\ell$')

	plt.savefig('Cross_43x95_FR_QxaU_flat.eps')
	plt.savefig('Cross_43x95_FR_QxaU_flat.png',format='png')
	
	cross2=np.mean(cross2_array,axis=0)
	noise2=np.mean(noise2_array,axis=0)
	dcross2=np.std(np.subtract(cross2_array,noise2),axis=0)
	plt.figure()
	plt.plot(l,l*(l+1)/(2*np.pi)*theory2,'r-')
	plt.plot(l,l*(l+1)/(2*np.pi)*(cross2-noise2),'k')
	plt.errorbar(l,l*(l+1)/(2*np.pi)*(cross2-noise2),yerr=l*(l+1)/(2*np.pi)*dcross2,color='black')
	plt.title('Cross 43x95 FR UxaQ')
	plt.ylabel('$\\frac{\ell(\ell+1)}{2\pi}C_{\ell}\ \\frac{\mu K^{2}rad}{m^{4}}$')
	plt.xlabel('$\ell$')

	plt.savefig('Cross_43x95_FR_UxaQ_flat.eps')
	plt.savefig('Cross_43x95_FR_UxaQ_flat.png',format='png')
	
	subprocess.call('mv *.eps eps/', shell=True)
	
if __name__=='__main__':
	main()
