import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.io import fits
import ipdb
import make_quiet_field as simulate_fields
import rotate_tqu
import plot_binned
import subprocess
import json

def faraday_correlate_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands,beam=False):
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
	
	iqu_band_i=hp.ud_grade(iqu_band_i,nside_out=128,order_in='ring')
	iqu_band_j=hp.ud_grade(iqu_band_j,nside_out=128,order_in='ring')
	planck_T=hp.ud_grade(planck_T,nside_out=128,order_in='ring')
	
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.pi/180.,lmax=383)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.pi/180.,lmax=383)
	planck_T=hp.smoothing(planck_T,fwhm=np.pi/180.,lmax=383)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	P=np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)
	weights=np.repeat(1,len(P))	
	num,bins,junk=plt.hist(P,bins=40)
	index=np.argmax(num)
	weights[np.where(P <= bins[index+1]/2.)]=.75
	weights[np.where(P <= bins[index+1]/4.)]=.5
	weights[np.where(P <= bins[index+1]/8.)]=.25
	const=2.*(wl_i**2-wl_j**2)	

	Delta_Q=(iqu_band_i[1]-iqu_band_j[1])/const*weights
	Delta_U=(iqu_band_i[2]-iqu_band_j[2])/const*weights
	alpha_u=alpha_radio*iqu_band_j[2]*weights
	alpha_q=-alpha_radio*iqu_band_j[1]*weights

	DQm=hp.ma(Delta_Q)
	DUm=hp.ma(Delta_U)
	aQm=hp.ma(alpha_q)
	aUm=hp.ma(alpha_u)
	cross1_array=[]
	cross2_array=[]
	cross3_array=[]
	if beam:
		l=np.arange(3*128)
		Bl_60=np.exp(-l*(l+1)*((60.0*np.pi/(180.*60.)/(np.sqrt(8.0*np.log(2.))))**2)/2.)
		Bl_11=np.exp(-l*(l+1)*((11.7*np.pi/(180.*60.)/(np.sqrt(8.0*np.log(2.))))**2)/2.)
		Bl_27=np.exp(-l*(l+1)*((27.3*np.pi/(180.*60.)/(np.sqrt(8.0*np.log(2.))))**2)/2.)
		Bl_factor=Bl_60**2*Bl_11*Bl_27
	else:
		Bl_factor=hp.gauss_beam(11.7*np.pi/(180.*60),lmax=383)*hp.gauss_beam(27.3*np.pi/(180.*60.),lmax=383)

	for field1 in xrange(4):
		mask_bool1=np.repeat(True,len(Delta_Q))
		pix_cmb1=field_pixels.field(field1)	
		pix_cmb1=pix_cmb1[np.nonzero(pix_cmb1)]	##Take Pixels From Field 1
		tmp=np.zeros(hp.nside2npix(1024))
		tmp[pix_cmb1]=1
		tmp=hp.ud_grade(tmp,128)
		mask_bool1[np.nonzero(tmp)]=False
	#	mask_bool1[np.where(P<.7e-6)]=True
		DQm.mask=mask_bool1
		DUm.mask=mask_bool1
		aQm.mask=mask_bool1
		aUm.mask=mask_bool1

		TE_map=np.array([planck_T*alpha_radio,Delta_Q,Delta_U])
		TEm=hp.ma(TE_map)
		TEm[0].mask=mask_bool1
		TEm[1].mask=mask_bool1
		TEm[2].mask=mask_bool1
		
		cross1_array.append(hp.anafast(DQm,map2=aUm)/Bl_factor)
		cross2_array.append(hp.anafast(DUm,map2=aQm)/Bl_factor)
		cross_tmp=hp.anafast(TEm,pol=1,nspec=4)
		cross3_array.append(cross_tmp[-1]/Bl_factor)
	
	cross1=np.mean(cross1_array,axis=0)	##Average over all Cross Spectra
	cross2=np.mean(cross2_array,axis=0)	##Average over all Cross Spectra
	cross3=np.mean(cross3_array,axis=0)	##Average over all Cross Spectra
	hp.write_cl('cl_'+bands+'_FR_QxaU.fits',cross1)
	hp.write_cl('cl_'+bands+'_FR_UxaQ.fits',cross2)
	hp.write_cl('cl_'+bands+'_FR_TE_cmb.fits',cross3)
	return (cross1,cross2,cross3)

def faraday_noise_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands,beam=False):
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

	P=np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)
	weights=np.repeat(1,len(P))	
	num,bins,junk=plt.hist(P,bins=40)
	index=np.argmax(num)
	weights[np.where(P <= bins[index+1]/2.)]=.75
	weights[np.where(P <= bins[index+1]/4.)]=.5
	weights[np.where(P <= bins[index+1]/8.)]=.25
	const=2.*(wl_i**2-wl_j**2)	

	Delta_Q=(iqu_band_i[1]-iqu_band_j[1])/const*weights
	Delta_U=(iqu_band_i[2]-iqu_band_j[2])/const*weights
	alpha_u=alpha_radio*iqu_band_j[2]*weights 
	alpha_q=-alpha_radio*iqu_band_j[1]*weights

	DQm=hp.ma(Delta_Q)
	DUm=hp.ma(Delta_U)
	aQm=hp.ma(alpha_q)
	aUm=hp.ma(alpha_u)
	cross1_array=[]
	cross2_array=[]
	cross3_array=[]
	
	if beam:
		l=np.arange(3*128)
		Bl_60=np.exp(-l*(l+1)*((60.0*np.pi/(180.*60.)/(np.sqrt(8.0*np.log(2.))))**2)/2.)
		Bl_11=np.exp(-l*(l+1)*((11.7*np.pi/(180.*60.)/(np.sqrt(8.0*np.log(2.))))**2)/2.)
		Bl_27=np.exp(-l*(l+1)*((27.3*np.pi/(180.*60.)/(np.sqrt(8.0*np.log(2.))))**2)/2.)
		Bl_factor=Bl_60**2*Bl_11*Bl_27
	else:
		Bl_factor=hp.gauss_beam(11.7*np.pi/(180.*60),lmax=383)*hp.gauss_beam(27.3*np.pi/(180.*60.),lmax=383)
	for field1 in xrange(4):
		mask_bool1=np.repeat(True,len(Delta_Q))
		pix_cmb1=field_pixels.field(field1)	
		pix_cmb1=pix_cmb1[np.nonzero(pix_cmb1)]	##Take Pixels From Field 1
		tmp=np.zeros(hp.nside2npix(1024))
		tmp[pix_cmb1]=1
		tmp=hp.ud_grade(tmp,128)
		mask_bool1[np.nonzero(tmp)]=False
	#	mask_bool1[np.where(P<.7e-6)]=True
		
		DQm.mask=mask_bool1
		DUm.mask=mask_bool1
		aQm.mask=mask_bool1
		aUm.mask=mask_bool1

		TE_map=np.array([planck_T*alpha_radio,Delta_Q,Delta_U])
		TEm=hp.ma(TE_map)
		TEm[0].mask=mask_bool1
		TEm[1].mask=mask_bool1
		TEm[2].mask=mask_bool1
		
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

def faraday_theory_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands_name,beam=False):
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
	wl=np.array([299792458./(band*1e9) for band in bands])
	num_wl=len(wl)
	t_array=np.zeros((num_wl,npix))	
	q_array=np.zeros((num_wl,npix))
	u_array=np.zeros((num_wl,npix))
	for i in range(num_wl):
		tmp_cmb=rotate_tqu.rotate_tqu(simul_cmb,wl[i],alpha_radio);
		t_array[i],q_array[i],u_array[i]=tmp_cmb
	iqu_band_i=[t_array[0],q_array[0],u_array[0]]	
	iqu_band_j=[t_array[1],q_array[1],u_array[1]]	


	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi')
	temperature_file='/data/Planck/COM_CompMap_CMB-smica_2048.fits'
	planck_T=hp.read_map(temperature_file)
	planck_T*=1e-6
	hdu_i=fits.open(i_file)
	field_pixels=hdu_i['FIELD PIXELS'].data
	hdu_i.close()
	
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
	alpha_u=alpha_radio*iqu_band_j[2]
	alpha_q=-alpha_radio*iqu_band_j[1]

	DQm=hp.ma(Delta_Q)
	DUm=hp.ma(Delta_U)
	aQm=hp.ma(alpha_q)
	aUm=hp.ma(alpha_u)
	cross1_array=[]
	cross2_array=[]
	cross3_array=[]
	Bl_factor=np.repeat(1.,3*128)
	for field1 in xrange(4):
		mask_bool1=np.repeat(True,len(Delta_Q))
		pix_cmb1=field_pixels.field(field1)	
		pix_cmb1=pix_cmb1[np.nonzero(pix_cmb1)]	##Take Pixels From Field 1
		tmp=np.zeros(hp.nside2npix(1024))
		tmp[pix_cmb1]=1
		tmp=hp.ud_grade(tmp,128)
		mask_bool1[np.nonzero(tmp)]=False
	#	mask_bool1[np.where(P<.7e-6)]=True
		
		DQm.mask=mask_bool1
		DUm.mask=mask_bool1
		aQm.mask=mask_bool1
		aUm.mask=mask_bool1

		TE_map=np.array([planck_T*alpha_radio,Delta_Q,Delta_U])
		TEm=hp.ma(TE_map)
		TEm[0].mask=mask_bool1
		TEm[1].mask=mask_bool1
		TEm[2].mask=mask_bool1
		
		cross1_array.append(hp.anafast(DQm,map2=aUm)/Bl_factor)
		cross2_array.append(hp.anafast(DUm,map2=aQm)/Bl_factor)
		cross_tmp=hp.anafast(TEm,pol=1,nspec=4)
		cross3_array.append(cross_tmp[-1]/Bl_factor)
	
	cross1=np.mean(cross1_array,axis=0)	##Average over all Cross Spectra
	cross2=np.mean(cross2_array,axis=0)	##Average over all Cross Spectra
	cross3=np.mean(cross3_array,axis=0)	##Average over all Cross Spectra
	hp.write_cl('cl_'+bands_name+'_FR_QxaU.fits',cross1)
	hp.write_cl('cl_'+bands_name+'_FR_UxaQ.fits',cross2)
	hp.write_cl('cl_'+bands_name+'_FR_TE_cmb.fits',cross3)
	return (cross1,cross2,cross3)

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
	cross3_array=[]
	noise1_array=[]
	noise2_array=[]
	noise3_array=[]
	
	for i in xrange(N_runs):	
		simulate_fields.main()
		tmp1,tmp2,tmp3=faraday_correlate_quiet(i_file+'.fits',j_file+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1])
		ntmp1,ntmp2,ntmp3=faraday_noise_quiet(i_file+'.fits',j_file+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1])
		cross1_array.append(tmp1)
		cross2_array.append(tmp2)
		cross3_array.append(tmp3)
		noise1_array.append(ntmp1)
		noise2_array.append(ntmp2)
		noise3_array.append(ntmp3)

	theory1,theory2,theory3=faraday_theory_quiet(i_file+'.fits',j_file+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1])
	hp.write_cl('cl_theory_FR_QxaU.fits',theory1)
	hp.write_cl('cl_theory_FR_UxaQ.fits',theory2)
	hp.write_cl('cl_theory_FR_TE.fits',theory3)
	f=open('cl_array_FR_QxaU.json','w')
	json.dump([[a for a in cross1_array[i]] for i in xrange(len(cross1_array))],f)
	f.close()	
	f=open('cl_array_FR_UxaQ.json','w')
	json.dump([[a for a in cross2_array[i]] for i in xrange(len(cross2_array))],f)
	f.close()	
	f=open('cl_array_FR_TE.json','w')
	json.dump([[a for a in cross3_array[i]] for i in xrange(len(cross3_array))],f)
	f.close()	
	f=open('cl_noise_FR_QxaU.json','w')
	json.dump([[a for a in noise1_array[i]] for i in xrange(len(noise1_array))],f)
	f.close()	
	f=open('cl_noise_FR_UxaQ.json','w')
	json.dump([[a for a in noise2_array[i]] for i in xrange(len(noise2_array))],f)
	f.close()	
	f=open('cl_noise_FR_TE.json','w')
	json.dump([[a for a in noise3_array[i]] for i in xrange(len(noise3_array))],f)
	f.close()	

	cross1=np.mean(cross1_array,axis=0)
	noise1=np.mean(noise1_array,axis=0)
	dcross1=np.std(cross1_array,axis=0)
	plot_binned.plotBinned((cross1-noise1)*1e12,dcross1*1e12,bins,'Cross_43x95_FR_QxaU', title='Cross 43x95 FR QxaU',theory=theory1*1e12)

	cross2=np.mean(cross2_array,axis=0)
	noise2=np.mean(noise2_array,axis=0)
	dcross2=np.std(cross2_array,axis=0)
	plot_binned.plotBinned((cross2-noise2)*1e12,dcross2*1e12,bins,'Cross_43x95_FR_UxaQ', title='Cross 43x95 FR UxaQ',theory=theory2*1e12)

	cross3=np.mean(cross3_array,axis=0)
	noise3=np.mean(noise3_array,axis=0)
	dcross3=np.std(cross3_array,axis=0)
	plot_binned.plotBinned((cross3-noise3)*1e12,dcross3*1e12,bins,'Cross_43x95_FR_TE', title='Cross 43x95 FR TE',theory=theory3*1e12)
	
	subprocess.call('mv *01*.png bin_01/', shell=True)
	subprocess.call('mv *05*.png bin_05/', shell=True)
	subprocess.call('mv *10*.png bin_10/', shell=True)
	subprocess.call('mv *20*.png bin_20/', shell=True)
	subprocess.call('mv *50*.png bin_50/', shell=True)
	subprocess.call('mv *.eps eps/', shell=True)
	
if __name__=='__main__':
	main()
