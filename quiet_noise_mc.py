import numpy as np
import healpy as hp
from astropy.io import fits
import ipdb
import make_quiet_field as simulate_fields
import plot_binned
import subprocess
import json
def faraday_correlate_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands):
	print "Computer Cross Correlations for Bands "+str(bands)

	temperature_file='/data/Planck/COM_CompMap_CMB-smica_2048.fits'
	planck_T=hp.read_map(temperature_file)
	planck_T*=1e-6

	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi')
	iqu_band_i=hdu_i['stokes iqu'].data
	iqu_band_j=hdu_j['stokes iqu'].data
	sigma_i=hdu_i['Q/U UNCERTAINTIES'].data
	sigma_j=hdu_j['Q/U UNCERTAINTIES'].data
	#mask_hdu=fits.open('/data/wmap/wmap_polarization_analysis_mask_r9_9yr_v5.fits')
	#mask=mask_hdu[1].data.field(0)
	#mask_hdu.close()
	#mask=hp.reorder(mask,n2r=1)
	mask=hdu_i['mask'].data
	mask=hp.ud_grade(mask,nside_out=128)
	pix=np.where(mask != 0)
	pix=np.array(pix).reshape(len(pix[0]))
	pix_bad=np.where(mask == 0)
	field_pixels=hdu_i['FIELD PIXELS'].data
	iqu_band_i[1]=sigma_i[0]
	iqu_band_i[2]=sigma_i[1]
	iqu_band_j[1]=sigma_j[0]
	iqu_band_j[2]=sigma_j[1]
	hdu_i.close()
	hdu_j.close()
	
	iqu_band_i=hp.ud_grade(iqu_band_i,nside_out=128,order_in='ring')
	iqu_band_j=hp.ud_grade(iqu_band_j,nside_out=128,order_in='ring')
	planck_T=hp.ud_grade(planck_T,nside_out=128,order_in='ring')
	
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.pi/180.,lmax=383)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.pi/180.,lmax=383)
	planck_T=hp.smoothing(planck_T,fwhm=np.pi/180.,lmax=383)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	const=2.*(wl_i**2-wl_j**2)	

	Delta_Q=(iqu_band_i[1]-iqu_band_j[1])/const
	Delta_U=(iqu_band_i[2]-iqu_band_j[2])/const
	alpha_u=alpha_radio*iqu_band_i[2] 
	alpha_q=-alpha_radio*iqu_band_i[1]

	DQm=hp.ma(Delta_Q)
	DUm=hp.ma(Delta_U)
	aQm=hp.ma(alpha_q)
	aUm=hp.ma(alpha_u)
	mask_bool1=np.repeat(True,len(Delta_Q))
	mask_bool2=np.repeat(True,len(Delta_Q))
	cross1_array=[]
	cross2_array=[]
	cross3_array=[]
	
	l=np.arange(3*128)
	Bl_60=np.exp(-l*(l+1)*((60.0*np.pi/(180.*60.)/(np.sqrt(8.0*np.log(2.))))**2)/2.)
	Bl_11=np.exp(-l*(l+1)*((11.7*np.pi/(180.*60.)/(np.sqrt(8.0*np.log(2.))))**2)/2.)
	Bl_27=np.exp(-l*(l+1)*((27.3*np.pi/(180.*60.)/(np.sqrt(8.0*np.log(2.))))**2)/2.)
	Bl_factor=Bl_60**2*Bl_11*Bl_27
	for field1 in xrange(4):
		for field2 in xrange(field1,4):
	
			pix_cmb1=field_pixels.field(field1)	
			pix_cmb1=pix_cmb1[np.nonzero(pix_cmb1)]	##Take Pixels From Field 1
			tmp=np.zeros(hp.nside2npix(1024))
			tmp[pix_cmb1]=1
			tmp=hp.ud_grade(tmp,128)
			mask_bool1[np.nonzero(tmp)]=False
			
			pix_cmb2=field_pixels.field(field2)
			pix_cmb2=pix_cmb2[np.nonzero(pix_cmb2)]	##Take Pixels From Field 2
			tmp=np.zeros(hp.nside2npix(1024))
			tmp[pix_cmb2]=1
			tmp=hp.ud_grade(tmp,128)
			mask_bool2[np.nonzero(tmp)]=False	##Create Masks for each QUIET FIELD

			DQm.mask=mask_bool1
			DUm.mask=mask_bool1
			aQm.mask=mask_bool2
			aUm.mask=mask_bool2

			TE_map=np.array([planck_T*alpha_radio,Delta_Q,Delta_U])
			TEm=hp.ma(TE_map)
			TEm[0].mask=mask_bool1
			TEm[1].mask=mask_bool2
			TEm[2].mask=mask_bool2
			
			cross1_array.append(hp.anafast(DQm,map2=aUm)/Bl_factor)
			cross2_array.append(hp.anafast(DUm,map2=aQm)/Bl_factor)
			cross_tmp=hp.anafast(TEm,pol=1,nspec=4)
			cross3_array.append(cross_tmp[-1]/Bl_factor)
	
	cross1=np.mean(cross1_array,axis=0)	##Average over all Cross Spectra
	cross2=np.mean(cross2_array,axis=0)	##Average over all Cross Spectra
	cross3=np.mean(cross3_array,axis=0)	##Average over all Cross Spectra
	hp.write_cl('cl_'+bands+'_FR_noise_QxaU.fits',cross1)
	hp.write_cl('cl_'+bands+'_FR_noise_UxaQ.fits',cross2)
	hp.write_cl('cl_'+bands+'_FR_noise_TE_cmb.fits',cross3)
	return (cross1,cross2,cross3)


if __name__=='__main__':
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
	cross3_array=[]
	
	for i in xrange(N_runs):	
		simulate_fields.main()
		tmp1,tmp2,tmp3=faraday_correlate_quiet(i_file+'.fits',j_file+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1])
		cross1_array.append(tmp1)
		cross2_array.append(tmp2)
		cross3_array.append(tmp3)

	f=open('cl_noise_FR_QxaU.json','w')
	json.dump([[a for a in cross1_array[i]] for i in xrange(len(cross1_array))],f)
	f.close()	
	f=open('cl_noise_FR_UxaQ.json','w')
	json.dump([[a for a in cross2_array[i]] for i in xrange(len(cross2_array))],f)
	f.close()	
	f=open('cl_noise_FR_TE.json','w')
	json.dump([[a for a in cross3_array[i]] for i in xrange(len(cross3_array))],f)
	f.close()	
	
