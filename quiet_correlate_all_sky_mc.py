import matplotlib
matplotlib.use('Agg')
import numpy as np
import healpy as hp
from astropy.io import fits
import ipdb
import make_quiet_field as simulate_fields
import plot_binned
import bin_llcl
import subprocess
import json
import rotate_tqu
from colorama import Fore, Back, Style

##Global smoothing size parameter##

smoothing_scale=30.0
###map_bl=hp.gauss_beam(np.sqrt(hp.nside2pixarea(128)+hp.nside2pixarea(1024)),383)

def faraday_correlate_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands,beam=False,weight=None):
	print "Computing Cross Correlations for Bands "+str(bands)

	q_fwhm=[27.3,11.7]
	noise_const=np.array([36./f for f in q_fwhm])*1e-6
	npix=hp.nside2npix(1024)
	sigma_i=[noise_const[0]*np.random.normal(0,1,npix),noise_const[0]*np.random.normal(0,1,npix)]
	sigma_j=[noise_const[1]*np.random.normal(0,1,npix),noise_const[1]*np.random.normal(0,1,npix)]


	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi')
	delta_alpha_radio=hp.read_map(alpha_file,hdu='uncertainty/phi')*np.random.normal(0,1,hp.nside2npix(128))
	iqu_band_i=hdu_i['stokes iqu'].data
	iqu_band_j=hdu_j['stokes iqu'].data
	#sigma_i=hdu_i['Q/U UNCERTAINTIES'].data
	#sigma_j=hdu_j['Q/U UNCERTAINTIES'].data
	#mask_hdu=fits.open('/data/wmap/wmap_polarization_analysis_mask_r9_9yr_v5.fits')
	#mask=mask_hdu[1].data.field(0)
	#mask_hdu.close()
	#mask=hp.reorder(mask,n2r=1)
	#mask=hdu_i['mask'].data
	#mask=hp.ud_grade(mask,nside_out=128)
	#pix=np.where(mask != 0)
	#pix=np.array(pix).reshape(len(pix[0]))
	#pix_bad=np.where(mask == 0)
	field_pixels=hdu_i['FIELD PIXELS'].data
	
	iqu_band_i[1]+=sigma_i[0]
	iqu_band_i[2]+=sigma_i[1]
	iqu_band_j[1]+=sigma_j[0]
	iqu_band_j[2]+=sigma_j[1]
	hdu_i.close()
	hdu_j.close()
	
	sigma_q_i=hp.smoothing(sigma_i[0],fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_u_i=hp.smoothing(sigma_i[1],fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_q_j=hp.smoothing(sigma_j[0],fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_u_j=hp.smoothing(sigma_j[1],fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),verbose=False)	

	sigma_q_i=hp.ud_grade(sigma_q_i,128)
	sigma_u_i=hp.ud_grade(sigma_u_i,128)
	sigma_q_j=hp.ud_grade(sigma_q_j,128)
	sigma_u_j=hp.ud_grade(sigma_u_j,128)
		
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),verbose=False)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),verbose=False)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	iqu_band_i=hp.ud_grade(iqu_band_i,nside_out=128,order_in='ring')
	iqu_band_j=hp.ud_grade(iqu_band_j,nside_out=128,order_in='ring')
	
	const=2.*(wl_i**2-wl_j**2)	

	Delta_Q=(iqu_band_i[1]-iqu_band_j[1])/const
	Delta_U=(iqu_band_i[2]-iqu_band_j[2])/const
	alpha_u=alpha_radio*iqu_band_j[2] 
	alpha_q=-alpha_radio*iqu_band_j[1]

	DQm=hp.ma(Delta_Q)
	DUm=hp.ma(Delta_U)
	aQm=hp.ma(alpha_q)
	aUm=hp.ma(alpha_u)
	sqi=hp.ma(sigma_q_i)
	sui=hp.ma(sigma_u_i)
	sqj=hp.ma(sigma_q_j)
	suj=hp.ma(sigma_u_j)
	salpha=hp.ma(delta_alpha_radio)
	alpham=hp.ma(alpha_radio)
	um=hp.ma(iqu_band_j[2])
	qm=hp.ma(iqu_band_j[1])
	
	cross1_array=[]
	cross2_array=[]
	Ndq_array=[]
	Ndu_array=[]
	Nau_array=[]
	Naq_array=[]
	Bl_factor=np.repeat(1,3*128)
	l=np.arange(3*128)
	#ipdb.set_trace()
	Weight1=np.repeat(1,len(iqu_band_i[0]))
	Weight2=np.repeat(1,len(iqu_band_i[0]))
	if weight:
		Weight1=(iqu_band_i[1]**2+iqu_band_i[2]**2)/(sigma_u_i[0]**2+sigma_q_i[1]**2)
		Weight2=(iqu_band_j[1]**2+iqu_band_i[2]**2)/(sigma_u_j[0]**2+sigma_q_j[1]**2)
	if beam:
		Bl_factor=hp.gauss_beam(smoothing_scale*(np.pi/180.*60.),383)
	pix_area=hp.nside2pixarea(128)
	for field1 in xrange(1):
		mask_bool1=np.repeat(True,len(Delta_Q))
		pix_cmb1=field_pixels.field(field1)	
		pix_cmb1=pix_cmb1[np.nonzero(pix_cmb1)]	##Take Pixels From Field 1
		tmp=np.zeros(hp.nside2npix(1024))
		tmp[pix_cmb1]=1
		tmp=hp.ud_grade(tmp,128)
		mask_bool1[np.nonzero(tmp)]=False
		#mask_bool1[np.where(np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)<.2e-6)]=True
		
		DQm.mask=mask_bool1
		DUm.mask=mask_bool1
		aQm.mask=mask_bool1
		aUm.mask=mask_bool1
		sqi.mask=mask_bool1
		sui.mask=mask_bool1
		sqj.mask=mask_bool1
		suj.mask=mask_bool1
		salpha.mask=mask_bool1
		alpham.mask=mask_bool1
		um.mask=mask_bool1
		qm.mask=mask_bool1

		cross1_array.append(hp.anafast(DQm*Weight1,map2=aUm*Weight1)/Bl_factor**2)
		cross2_array.append(hp.anafast(DUm*Weight1,map2=aQm*Weight1)/Bl_factor**2)
		

		##calculate theoretical variance for correlations
		Ndq_array.append((abs(Weight1*(sqi-sqj))**2).sum()*(pix_area/const)**2/(4.*np.pi))
		Ndu_array.append((abs(Weight1*(sui-suj))**2).sum()*(pix_area/const)**2/(4.*np.pi))
		Nau_array.append((abs(Weight1*(salpha*um+alpham*suj+salpha*suj))**2).sum()*pix_area**2/(4.*np.pi))
		Naq_array.append((abs(Weight1*(salpha*qm+alpham*sqj+salpha*sqj))**2).sum()*pix_area**2/(4.*np.pi))
		#ipdb.set_trace()
		
        


	#cross1=np.mean(cross1_array,axis=0)	##Average over all Cross Spectra
	#cross2=np.mean(cross2_array,axis=0)	##Average over all Cross Spectra
	#N_dq=np.mean(Ndq_array,axis=0)	##Average over all Cross Spectra
	#N_du=np.mean(Ndu_array,axis=0)	##Average over all Cross Spectra
	#N_au=np.mean(Nau_array,axis=0)	##Average over all Cross Spectra
	#N_aq=np.mean(Naq_array,axis=0)	##Average over all Cross Spectra
	cross1=cross1_array[0]
	cross2=cross2_array[0]
	N_dq=Ndq_array[0]
	N_du=Ndu_array[0]
	N_au=Nau_array[0]
	N_aq=Naq_array[0]
	hp.write_cl('cl_'+bands+'_FR_QxaU.fits',cross1)
	hp.write_cl('cl_'+bands+'_FR_UxaQ.fits',cross2)


	return (cross1,cross2,N_dq,N_du,N_au,N_aq)

def faraday_noise_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands,beam=False,weight=None):
	print "Computing Cross Correlations for Bands "+str(bands)


	q_fwhm=[27.3,11.7]
	noise_const=np.array([36./f for f in q_fwhm])*1e-6
	npix=hp.nside2npix(1024)
	sigma_i=[noise_const[0]*np.random.normal(0,1,npix),noise_const[0]*np.random.normal(0,1,npix)]
	sigma_j=[noise_const[1]*np.random.normal(0,1,npix),noise_const[1]*np.random.normal(0,1,npix)]
	
	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	iqu_band_i=hdu_i['stokes iqu'].data
	iqu_band_j=hdu_j['stokes iqu'].data
	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi')
	field_pixels=hdu_i['FIELD PIXELS'].data
	hdu_i.close()
	hdu_j.close()
	
	Weight1=(iqu_band_i[1]**2+iqu_band_i[2]**2)/(sigma_i[0]**2+sigma_i[1]**2)
	Weight2=(iqu_band_j[1]**2+iqu_band_i[2]**2)/(sigma_j[0]**2+sigma_j[1]**2)
	iqu_band_i=np.zeros((3,npix))	
	iqu_band_j=np.zeros((3,npix))	
	iqu_band_i[1]=sigma_i[0]
	iqu_band_i[2]=sigma_i[1]
	iqu_band_j[1]=sigma_j[0]
	iqu_band_j[2]=sigma_j[1]
	
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),verbose=False)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),verbose=False)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	iqu_band_i=hp.ud_grade(iqu_band_i,nside_out=128,order_in='ring')
	iqu_band_j=hp.ud_grade(iqu_band_j,nside_out=128,order_in='ring')
	
	Weight1=np.repeat(1,len(iqu_band_i[0]))
	Weight2=np.repeat(1,len(iqu_band_i[0]))
	if weight:
		Weight1=hp.smoothing(Weight1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),verbose=False)	
		Weight2=hp.smoothing(Weight2,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),verbose=False)	
	
	Weight1=hp.ud_grade(Weight1,128)
	Weight2=hp.ud_grade(Weight2,128)
	
	const=2.*(wl_i**2-wl_j**2)	

	Delta_Q=(iqu_band_i[1]-iqu_band_j[1])/const
	Delta_U=(iqu_band_i[2]-iqu_band_j[2])/const
	alpha_u=alpha_radio*iqu_band_j[2] 
	alpha_q=-alpha_radio*iqu_band_j[1]

	DQm=hp.ma(Delta_Q)
	DUm=hp.ma(Delta_U)
	aQm=hp.ma(alpha_q)
	aUm=hp.ma(alpha_u)
	cross1_array=[]
	cross2_array=[]
	
	Bl_factor=np.repeat(1,3*128)
	if beam:
		Bl_factor=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),383)
	for field1 in xrange(1):
		mask_bool1=np.repeat(True,len(Delta_Q))
		pix_cmb1=field_pixels.field(field1)	
		pix_cmb1=pix_cmb1[np.nonzero(pix_cmb1)]	##Take Pixels From Field 1
		tmp=np.zeros(hp.nside2npix(1024))
		tmp[pix_cmb1]=1
		tmp=hp.ud_grade(tmp,128)
		mask_bool1[np.nonzero(tmp)]=False
		#mask_bool1[np.where(np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)<.2e-6)]=True
		
		DQm.mask=mask_bool1
		DUm.mask=mask_bool1
		aQm.mask=mask_bool1
		aUm.mask=mask_bool1

		
		cross1_array.append(hp.anafast(DQm*Weight1,map2=aUm*Weight1)/Bl_factor**2)
		cross2_array.append(hp.anafast(DUm*Weight1,map2=aQm*Weight1)/Bl_factor**2)
	
	#cross1=np.mean(cross1_array,axis=0)	##Average over all Cross Spectra
	#cross2=np.mean(cross2_array,axis=0)	##Average over all Cross Spectra
	cross1=cross1_array[0]
	cross2=cross2_array[0]
	hp.write_cl('cl_'+bands+'_FR_noise_QxaU.fits',cross1)
	hp.write_cl('cl_'+bands+'_FR_noise_UxaQ.fits',cross2)
	return (cross1,cross2)

def faraday_theory_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands_name,beam=False,weight=None):
	print "Computing Cross Correlations for Bands "+str(bands_name)

	radio_file='/data/wmap/faraday_MW_realdata.fits'
	cl_file='/home/matt/wmap/simul_scalCls.fits'
	nside=1024
	npix=hp.nside2npix(nside)
	
	cls=hp.read_cl(cl_file)
	simul_cmb=hp.sphtfunc.synfast(cls,nside,fwhm=0.,new=1,pol=1);
	
	alpha_radio=hp.read_map(radio_file,hdu='maps/phi');
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=nside,order_in='ring',order_out='ring')
	bands=[43.1,94.5]
        q_fwhm=[27.3,11.7]
	noise_const=np.array([36./f for f in q_fwhm])*1e-6
	npix=hp.nside2npix(128)
	sigma_i=[noise_const[0]*np.random.normal(0,1,npix),noise_const[0]*np.random.normal(0,1,npix)]
	sigma_j=[noise_const[1]*np.random.normal(0,1,npix),noise_const[1]*np.random.normal(0,1,npix)]
	npix=hp.nside2npix(1024)
	wl=np.array([299792458./(band*1e9) for band in bands])
	num_wl=len(wl)
	t_array=np.zeros((num_wl,npix))	
	q_array=np.zeros((num_wl,npix))
	u_array=np.zeros((num_wl,npix))
	for i in range(num_wl):
		tmp_cmb=rotate_tqu.rotate_tqu(simul_cmb,wl[i],alpha_radio);
		t_array[i],q_array[i],u_array[i]=hp.smoothing(tmp_cmb,fwhm=q_fwhm[i]*np.pi/(180.*60.),verbose=False)
	iqu_band_i=[t_array[0],q_array[0],u_array[0]]	
	iqu_band_j=[t_array[1],q_array[1],u_array[1]]	


	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	#iqu_band_i=hdu_i[1].data
	#iqu_band_j=hdu_j[1].data
	field_pixels=hdu_i['FIELD PIXELS'].data
	hdu_i.close()
	hdu_j.close()
	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi')
	
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),verbose=False)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),verbose=False)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	iqu_band_i=hp.ud_grade(iqu_band_i,nside_out=128,order_in='ring')
	iqu_band_j=hp.ud_grade(iqu_band_j,nside_out=128,order_in='ring')
	
	const=2.*(wl_i**2-wl_j**2)	

	Delta_Q=(iqu_band_i[1]-iqu_band_j[1])/const
	Delta_U=(iqu_band_i[2]-iqu_band_j[2])/const
	alpha_u=alpha_radio*iqu_band_j[2] 
	alpha_q=-alpha_radio*iqu_band_j[1]

	DQm=hp.ma(Delta_Q)
	DUm=hp.ma(Delta_U)
	aQm=hp.ma(alpha_q)
	aUm=hp.ma(alpha_u)
	cross1_array=[]
	cross2_array=[]

	Weight1=np.repeat(1,len(iqu_band_i[0]))
	if weight:
		Weight1=(iqu_band_i[1]**2+iqu_band_i[2]**2)/(sigma_i[0]**2+sigma_i[1]**2)
		Weight2=(iqu_band_j[1]**2+iqu_band_i[2]**2)/(sigma_j[0]**2+sigma_j[1]**2)
	
	Bl_factor=np.repeat(1,3*128)
	if beam:
		Bl_factor=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),383)
	for field1 in xrange(1):
		mask_bool1=np.repeat(True,len(Delta_Q))
		pix_cmb1=field_pixels.field(field1)	
		pix_cmb1=pix_cmb1[np.nonzero(pix_cmb1)]	##Take Pixels From Field 1
		tmp=np.zeros(hp.nside2npix(1024))
		tmp[pix_cmb1]=1
		tmp=hp.ud_grade(tmp,128)
		mask_bool1[np.nonzero(tmp)]=False
		#mask_bool1[np.where(np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)<.1e-6)]=True
		
		DQm.mask=mask_bool1
		DUm.mask=mask_bool1
		aQm.mask=mask_bool1
		aUm.mask=mask_bool1

		
		cross1_array.append(hp.anafast(DQm*Weight1,map2=aUm*Weight1)/Bl_factor**2)
		cross2_array.append(hp.anafast(DUm*Weight1,map2=aQm*Weight1)/Bl_factor**2)
	
	#cross1=np.mean(cross1_array,axis=0)	##Average over all Cross Spectra
	#cross2=np.mean(cross2_array,axis=0)	##Average over all Cross Spectra
	cross1=cross1_array[0]
	cross2=cross2_array[0]
	hp.write_cl('cl_'+bands_name+'_FR_QxaU.fits',cross1)
	hp.write_cl('cl_'+bands_name+'_FR_UxaQ.fits',cross2)
	return (cross1,cross2)
def plot_mc():

	f=open('cl_theory_FR_QxaU.json','r')
	theory1_array_in=json.load(f)
	f.close()	
	f=open('cl_theory_FR_UxaQ.json','r')
	theory2_array_in=json.load(f)
	f.close()	
	f=open('cl_array_FR_QxaU.json','r')
	cross1_array_in=json.load(f)
	f.close()	
	f=open('cl_array_FR_UxaQ.json','r')
	cross2_array_in=json.load(f)
	f.close()	
	f=open('cl_noise_FR_QxaU.json','r')
	noise1_array_in=json.load(f)
	f.close()	
	f=open('cl_noise_FR_UxaQ.json','r')
	noise2_array_in=json.load(f)
	f.close()	
	f=open('cl_Nau_FR_QxaU.json','r')
	Nau_array_in=json.load(f)
	f.close()	
	f=open('cl_Ndq_FR_QxaU.json','r')
	Ndq_array_in=json.load(f)
	f.close()	
	f=open('cl_Naq_FR_UxaQ.json','r')
	Naq_array_in=json.load(f)
	f.close()	
	f=open('cl_Ndu_FR_UxaQ.json','r')
	Ndu_array_in=json.load(f)
	f.close()	
	
	bins=[1,5,10,20,50]
	fsky=225.*(np.pi/180.)**2/(4*np.pi)
	l=np.arange(len(cross1_array_in[0]))
	ll=l*(l+1)/(2*np.pi)
	L=np.sqrt(fsky*4*np.pi)
	dl_eff=2*np.pi/L
	
	for b in bins:
		N_dq=np.mean(Ndq_array_in)
		N_au=np.mean(Nau_array_in)
		delta1=np.sqrt(2.*abs((np.mean(cross1_array_in,axis=0)-np.mean(noise1_array_in,axis=0))**2+(np.mean(cross1_array_in,axis=0)-np.mean(noise1_array_in,axis=0))/2.*(N_dq+N_au)+N_dq*N_au/2.)/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
	
		cosmic1=np.sqrt(2./((2.*l+1)*np.sqrt(b**2+dl_eff**2)*fsky)*np.mean(theory1_array_in,axis=0)**2)

		N_du=np.mean(Ndu_array_in)
		N_aq=np.mean(Naq_array_in)
		delta2=np.sqrt(2.*abs((np.mean(cross2_array_in,axis=0)-np.mean(noise2_array_in,axis=0))**2+(np.mean(cross2_array_in,axis=0)-np.mean(noise2_array_in,axis=0))/2.*(N_dq+N_au)+N_dq*N_au/2.)/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
		cosmic2=np.sqrt(2./((2*l+1)*np.sqrt(b**2+dl_eff**2)*fsky)*np.mean(theory2_array_in,axis=0)**2)

        	theory1_array=[]
        	theory2_array=[]
        	cross1_array=[]
        	cross2_array=[]
        	noise1_array=[]
        	noise2_array=[]
        	
		Ndq_array=[]
        	Ndu_array=[]
        	Nau_array=[]
        	Naq_array=[]
        	
		plot_l=[]
		if( b != 1):
	        	for n in xrange(len(cross1_array_in)):
	        	        tmp_t1=bin_llcl.bin_llcl(ll*theory1_array_in[n],b)
	        	        tmp_t2=bin_llcl.bin_llcl(ll*theory2_array_in[n],b)
				tmp_c1=bin_llcl.bin_llcl(ll*cross1_array_in[n],b)
	        	        tmp_c2=bin_llcl.bin_llcl(ll*cross2_array_in[n],b)
				tmp_n1=bin_llcl.bin_llcl(ll*noise1_array_in[n],b)
	        	        tmp_n2=bin_llcl.bin_llcl(ll*noise2_array_in[n],b)
	        	        
				theory1_array.append(tmp_t1['llcl'])
				theory2_array.append(tmp_t2['llcl'])
				
				cross1_array.append(tmp_c1['llcl'])
				cross2_array.append(tmp_c2['llcl'])
				
				noise1_array.append(tmp_n1['llcl'])
				noise2_array.append(tmp_n2['llcl'])
	        	        
				if n == len(cross1_array_in)-1:
	        	                plot_l=tmp_c1['l_out']
			tmp_c1=bin_llcl.bin_llcl(ll*cosmic1,b)
			tmp_d1=bin_llcl.bin_llcl(ll*delta1,b)
		
			cosmic1=tmp_c1['llcl']
			delta1=tmp_d1['llcl']

			tmp_c2=bin_llcl.bin_llcl(ll*cosmic2,b)
			tmp_d2=bin_llcl.bin_llcl(ll*delta2,b)
			cosmic2=tmp_c2['llcl']
			delta2=tmp_d2['llcl']
			
		else:
			plot_l=l
			theory1_array=np.multiply(ll,theory1_array_in)
			cross1_array=np.multiply(ll,cross1_array_in)
			noise1_array=np.multiply(ll,noise1_array_in)
			theory2_array=np.multiply(ll,theory2_array_in)
			cross2_array=np.multiply(ll,cross2_array_in)
			noise2_array=np.multiply(ll,noise2_array_in)
			cosmic1*=ll
			cosmic2*=ll
			delta1*=ll
			delta2*=ll
	
		theory1=np.mean(theory1_array,axis=0)
		dtheory1=np.std(theory1_array,axis=0,ddof=1)
		cross1=np.mean(cross1_array,axis=0)
		noise1=np.mean(noise1_array,axis=0)
		dcross1=np.std(np.subtract(cross1_array,noise1),axis=0,ddof=1)
		#ipdb.set_trace()
		plot_binned.plotBinned((cross1-noise1)*1e12,dcross1*1e12,plot_l,b,'Cross_43x95_FR_QxaU', title='Cross 43x95 FR QxaU',theory=theory1*1e12,dtheory=dtheory1*1e12,delta=delta1*1e12,cosmic=cosmic1*1e12)

		theory2=np.mean(theory2_array,axis=0)
		dtheory2=np.std(theory2_array,axis=0,ddof=1)
		cross2=np.mean(cross2_array,axis=0)
		#delta2=np.mean(delta2_array,axis=0)
		noise2=np.mean(noise2_array,axis=0)
		dcross2=np.std(np.subtract(cross2_array,noise2),axis=0,ddof=1)
		#ipdb.set_trace()
		plot_binned.plotBinned((cross2-noise2)*1e12,dcross2*1e12,plot_l,b,'Cross_43x95_FR_UxaQ', title='Cross 43x95 FR UxaQ',theory=theory2*1e12,dtheory=dtheory2*1e12,delta=delta2*1e12,cosmic=cosmic2*1e12)
		#ipdb.set_trace()
	
	subprocess.call('mv *01*.png bin_01/', shell=True)
	subprocess.call('mv *05*.png bin_05/', shell=True)
	subprocess.call('mv *10*.png bin_10/', shell=True)
	subprocess.call('mv *20*.png bin_20/', shell=True)
	subprocess.call('mv *50*.png bin_50/', shell=True)
	subprocess.call('mv *.eps eps/', shell=True)

def plot_mc_mll():
	f=open('cl_theory_FR_QxaU.json','r')
	theory1_array=json.load(f)
	f.close()	
	f=open('cl_theory_FR_UxaQ.json','r')
	theory2_array=json.load(f)
	f.close()	
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
	f=open('cl_Nau_FR_QxaU.json','r')
	Nau_array=json.load(f)
	f.close()	
	f=open('cl_Ndq_FR_QxaU.json','r')
	Ndq_array=json.load(f)
	f.close()	
	f=open('cl_Naq_FR_UxaQ.json','r')
	Naq_array=json.load(f)
	f.close()	
	f=open('cl_Ndu_FR_UxaQ.json','r')
	Ndu_array=json.load(f)
	f.close()	
	f=open('cl_delta_FR_UxaQ.json','r')
	delta2_array=json.load(f)
	f.close()	
	
	f=open('mixing_matrix.json','r')
	mll=json.load(f)
	f.close()
	mll=np.matrix(mll)
	kll=mll.getI()
	kll=kll.tolist()	
	theory1=np.array([np.sum(np.mean(theory1_array,axis=0)[j]*kll[i][j] for j in xrange(len(kll[i]))) for i in xrange(len(mll))])
	dtheory1=np.std(theory1_array,axis=0,ddof=1)
	cross1=np.array([np.sum(np.mean(cross1_array,axis=0)[j]*kll[i][j] for j in xrange(len(kll[i]))) for i in xrange(len(mll))])
	N_dq=np.array([np.sum(np.mean(Ndq_array)*m[j] for j in xrange(len(m))) for m in kll])
	N_au=np.array([np.sum(np.mean(Nau_array)*m[j] for j in xrange(len(m))) for m in kll])
	noise1=np.array([np.sum(np.mean(noise1_array,axis=0)[j]*kll[i][j] for j in xrange(len(kll[i]))) for i in xrange(len(mll))])
	l=np.arange(len(cross1))
	dcross1=np.std(np.subtract(cross1_array,noise1),axis=0,ddof=1)
	delta1=np.sqrt(2.*((cross1-noise1)**2+abs(cross1-noise1)/2.*(N_dq+N_au)+N_dq*N_au/2.)/(2.*l+1.))
	plot_binned.plotBinned((cross1-noise1)*1e12,dcross1*1e12,plot_l,b,'Cross_43x95_FR_QxaU', title='Cross 43x95 FR QxaU',theory=theory1*1e12,dtheory=dtheory1*1e12,delta=delta1*1e12)

	theory2=np.array([np.sum(np.mean(theory2_array,axis=0)[j]*kll[i][j] for j in xrange(len(kll[i]))) for i in xrange(len(mll))])
	dtheory2=np.std(theory2_array,axis=0,ddof=1)
	cross2=np.array([np.sum(np.mean(cross2_array,axis=0)[j]*kll[i][j] for j in xrange(len(kll[i]))) for i in xrange(len(mll))])
	#delta2=np.mean(delta2_array,axis=0)
	N_du=np.array([np.sum(np.mean(Ndu_array)*m[i] for i in xrange(len(m))) for m in kll])
	N_aq=np.array([np.sum(np.mean(Naq_array)*m[i] for i in xrange(len(m))) for m in kll])
	noise2=np.array([np.sum(np.mean(noise2_array,axis=0)[j]*kll[i][j] for j in xrange(len(kll[i]))) for i in xrange(len(mll))])
	dcross2=np.std(np.subtract(cross2_array,noise2),axis=0,ddof=1)
	delta2=np.sqrt(2.*((cross2-noise2)**2+abs(cross2-noise2)/2.*(N_du+N_aq)+N_du*N_aq/2.)/(2.*l+1.))
	#ipdb.set_trace()
	plot_binned.plotBinned((cross2-noise2)*1e12,dcross2*1e12,plot_l,b,'Cross_43x95_FR_UxaQ', title='Cross 43x95 FR UxaQ',theory=theory2*1e12,dtheory=dtheory2*1e12,delta=delta2*1e12)

	
	subprocess.call('mv *01*.png mll/bin_01/', shell=True)
	subprocess.call('mv *05*.png mll/bin_05/', shell=True)
	subprocess.call('mv *10*.png mll/bin_10/', shell=True)
	subprocess.call('mv *20*.png mll/bin_20/', shell=True)
	subprocess.call('mv *50*.png mll/bin_50/', shell=True)
	subprocess.call('mv *.eps mll/eps/', shell=True)


def main():
	##Parameters for Binning, Number of Runs
	##	Beam correction and, pixel weighting
	use_beam=1
	use_weight=0
	N_runs=100
	bins=[1,5,10,20,50]

	map_prefix='/home/matt/quiet/quiet_maps/'
	i_file=map_prefix+'quiet_simulated_43.1'
	j_file=map_prefix+'quiet_simulated_94.5'
	alpha_file='/data/wmap/faraday_MW_realdata.fits'
	bands=[43.1,94.5]
	names=['43','95']
	wl=np.array([299792458./(band*1e9) for band in bands])
	cross1_array_in=[]
	cross2_array_in=[]
	Ndq_array_in=[]
	Ndu_array_in=[]
	Nau_array_in=[]
	Naq_array_in=[]
	noise1_array_in=[]
	noise2_array_in=[]
	theory1_array_in=[]
	theory2_array_in=[]
	

	simulate_fields.main()
	#for n in xrange(N_runs):
	for i in xrange(N_runs):	
		print(Fore.WHITE+Back.GREEN+Style.BRIGHT+'Correlation #{:03d}'.format(i+1)+Back.RESET+Fore.RESET+Style.RESET_ALL)
		ttmp1,ttmp2=faraday_theory_quiet(i_file+'.fits',j_file+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1],beam=use_beam,weight=use_weight)
		theory1_array_in.append(ttmp1)
		theory2_array_in.append(ttmp2)
		tmp1,tmp2,n1,n2,n3,n4=faraday_correlate_quiet(i_file+'.fits',j_file+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1],beam=use_beam,weight=use_weight)
		ntmp1,ntmp2=faraday_noise_quiet(i_file+'.fits',j_file+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1],beam=use_beam,weight=use_weight)
		cross1_array_in.append(tmp1)
		cross2_array_in.append(tmp2)
		Ndq_array_in.append(n1)
		Ndu_array_in.append(n2)
		Nau_array_in.append(n3)
		Naq_array_in.append(n4)
		noise1_array_in.append(ntmp1)
		noise2_array_in.append(ntmp2)


	f=open('cl_theory_FR_QxaU.json','w')
	json.dump(np.array(theory1_array_in).tolist(),f)
	f.close()	
	f=open('cl_theory_FR_UxaQ.json','w')
	json.dump(np.array(theory2_array_in).tolist(),f)
	f.close()	
	theory1=np.mean(theory1_array,axis=0)
	theory2=np.mean(theory2_array,axis=0)
	hp.write_cl('cl_theory_FR_QxaU.fits',theory1)
	hp.write_cl('cl_theory_FR_UxaQ.fits',theory2)
	#f=open('cl_theory_FR_QxaU.json','r')
	#theory1_array=json.load(f)
	#f.close()	
	#f=open('cl_theory_FR_UxaQ.json','r')
	#theory2_array=json.load(f)
	#f.close()	
	f=open('cl_array_FR_QxaU.json','w')
	json.dump(np.array(cross1_array_in).tolist(),f)
	f.close()	
	f=open('cl_array_FR_UxaQ.json','w')
	json.dump(np.array(cross2_array_in).tolist(),f)
	f.close()	
	f=open('cl_Ndq_FR_QxaU.json','w')
	json.dump(np.array(Ndq_array_in).tolist(),f)
	f.close()	
	f=open('cl_Ndu_FR_UxaQ.json','w')
	json.dump(np.array(Ndu_array_in).tolist(),f)
	f.close()	
	f=open('cl_Nau_FR_QxaU.json','w')
	json.dump(np.array(Nau_array_in).tolist(),f)
	f.close()	
	f=open('cl_Naq_FR_UxaQ.json','w')
	json.dump(np.array(Naq_array_in).tolist(),f)
	f.close()	
	f=open('cl_noise_FR_QxaU.json','w')
	json.dump(np.array(noise1_array_in).tolist(),f)
	f.close()	
	f=open('cl_noise_FR_UxaQ.json','w')
	json.dump(np.array(noise2_array_in).tolist(),f)
	f.close()	
	bins=[1,5,10,20,50]
	fsky=225.*(np.pi/180.)**2/(4*np.pi)
	l=np.arange(len(cross1_array_in[0]))
	ll=l*(l+1)/(2*np.pi)
	L=np.sqrt(fsky*4*np.pi)
	dl_eff=2*np.pi/L
	for b in bins:
		N_dq=np.mean(Ndq_array_in)
		N_au=np.mean(Nau_array_in)
		delta1=np.sqrt(2.*abs((np.mean(cross1_array_in,axis=0)-np.mean(noise1_array_in,axis=0))**2+(np.mean(cross1_array_in,axis=0)-np.mean(noise1_array_in,axis=0))/2.*(N_dq+N_au)+N_dq*N_au/2.)/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
	
		cosmic1=np.sqrt(2./((2.*l+1)*np.sqrt(b**2+dl_eff**2)*fsky)*np.mean(theory1_array_in,axis=0)**2)

		N_du=np.mean(Ndu_array_in)
		N_aq=np.mean(Naq_array_in)
		delta2=np.sqrt(2.*abs((np.mean(cross2_array_in,axis=0)-np.mean(noise2_array_in,axis=0))**2+(np.mean(cross2_array_in,axis=0)-np.mean(noise2_array_in,axis=0))/2.*(N_dq+N_au)+N_dq*N_au/2.)/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
		cosmic2=np.sqrt(2./((2*l+1)*np.sqrt(b**2+dl_eff**2)*fsky)*np.mean(theory2_array_in,axis=0)**2)

        	theory1_array=[]
        	theory2_array=[]
        	cross1_array=[]
        	cross2_array=[]
        	noise1_array=[]
        	noise2_array=[]
        	
		Ndq_array=[]
        	Ndu_array=[]
        	Nau_array=[]
        	Naq_array=[]
        	
		plot_l=[]
		if( b != 1):
	        	for n in xrange(len(cross1_array_in)):
	        	        tmp_t1=bin_llcl.bin_llcl(ll*theory1_array_in[n],b)
	        	        tmp_t2=bin_llcl.bin_llcl(ll*theory2_array_in[n],b)
				tmp_c1=bin_llcl.bin_llcl(ll*cross1_array_in[n],b)
	        	        tmp_c2=bin_llcl.bin_llcl(ll*cross2_array_in[n],b)
				tmp_n1=bin_llcl.bin_llcl(ll*noise1_array_in[n],b)
	        	        tmp_n2=bin_llcl.bin_llcl(ll*noise2_array_in[n],b)
	        	        
				theory1_array.append(tmp_t1['llcl'])
				theory2_array.append(tmp_t2['llcl'])
				
				cross1_array.append(tmp_c1['llcl'])
				cross2_array.append(tmp_c2['llcl'])
				
				noise1_array.append(tmp_n1['llcl'])
				noise2_array.append(tmp_n2['llcl'])
	        	        
				if n == len(cross1_array_in)-1:
	        	                plot_l=tmp_c1['l_out']
			tmp_c1=bin_llcl.bin_llcl(ll*cosmic1,b)
			tmp_d1=bin_llcl.bin_llcl(ll*delta1,b)
		
			cosmic1=tmp_c1['llcl']
			delta1=tmp_d1['llcl']

			tmp_c2=bin_llcl.bin_llcl(ll*cosmic2,b)
			tmp_d2=bin_llcl.bin_llcl(ll*delta2,b)
			cosmic2=tmp_c2['llcl']
			delta2=tmp_d2['llcl']
			
		else:
			plot_l=l
			theory1_array=np.multiply(ll,theory1_array_in)
			cross1_array=np.multiply(ll,cross1_array_in)
			noise1_array=np.multiply(ll,noise1_array_in)
			theory2_array=np.multiply(ll,theory2_array_in)
			cross2_array=np.multiply(ll,cross2_array_in)
			noise2_array=np.multiply(ll,noise2_array_in)
			cosmic1*=ll
			cosmic2*=ll
			delta1*=ll
			delta2*=ll
	
		theory1=np.mean(theory1_array,axis=0)
		dtheory1=np.std(theory1_array,axis=0,ddof=1)
		cross1=np.mean(cross1_array,axis=0)
		noise1=np.mean(noise1_array,axis=0)
		dcross1=np.std(np.subtract(cross1_array,noise1),axis=0,ddof=1)
		#ipdb.set_trace()
		plot_binned.plotBinned((cross1-noise1)*1e12,dcross1*1e12,plot_l,b,'Cross_43x95_FR_QxaU', title='Cross 43x95 FR QxaU',theory=theory1*1e12,dtheory=dtheory1*1e12,delta=delta1*1e12,cosmic=cosmic1*1e12)

		theory2=np.mean(theory2_array,axis=0)
		dtheory2=np.std(theory2_array,axis=0,ddof=1)
		cross2=np.mean(cross2_array,axis=0)
		#delta2=np.mean(delta2_array,axis=0)
		noise2=np.mean(noise2_array,axis=0)
		dcross2=np.std(np.subtract(cross2_array,noise2),axis=0,ddof=1)
		#ipdb.set_trace()
		plot_binned.plotBinned((cross2-noise2)*1e12,dcross2*1e12,plot_l,b,'Cross_43x95_FR_UxaQ', title='Cross 43x95 FR UxaQ',theory=theory2*1e12,dtheory=dtheory2*1e12,delta=delta2*1e12,cosmic=cosmic2*1e12)
		#ipdb.set_trace()
	
	subprocess.call('mv *01*.png bin_01/', shell=True)
	subprocess.call('mv *05*.png bin_05/', shell=True)
	subprocess.call('mv *10*.png bin_10/', shell=True)
	subprocess.call('mv *20*.png bin_20/', shell=True)
	subprocess.call('mv *50*.png bin_50/', shell=True)
	subprocess.call('mv *.eps eps/', shell=True)
	
if __name__=='__main__':
	main()
