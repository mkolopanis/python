import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from astropy.io import fits
import ipdb
import simulate_prism as simulate_fields
import glob
import plot_binned
import bin_llcl
import subprocess
import json
import rotate_tqu
from colorama import Fore, Back, Style
from scipy import linalg as lin

##Global smoothing size parameter##
smoothing_scale=40.0
nside_out=128
npix_out=hp.nside2npix(nside_out)
bands=np.array([30.,36.,51.,105.,135.,160.])
wl=np.array([299792458./(b*1e9) for b in bands])
beam_fwhm=np.array([17.,14.,10.,4.8,3.8,3.2])
n_det=np.array([50,100,150,250,300,350])
noise_const_temp=np.array([63.4,59.7,53.7,45.6,44.9,45.5])/np.sqrt(n_det)*1e-6
noise_const_pol=np.array([89.7,84.5,75.9,64.4,63.4,64.3])/np.sqrt(n_det)*1e-6

##ONLY use THESE Bands
#good_index=[0,1,2]
#bands=bands[good_index]
#noise_const_temp=noise_const_temp[good_index]
#noise_const_pol=noise_const_pol[good_index]

###map_bl=hp.gauss_beam(np.sqrt(hp.nside2pixarea(128)+hp.nside2pixarea(1024)),383)

mask_array=[ '/data/wmap/wmap_polarization_analysis_mask_r9_9yr_v5.fits','/data/Planck/COM_CMB_IQU-common-field-MaskPol_1024_R2.00.fits']
mask_name=['wmap','cmask']

###map_bl=hp.gauss_beam(np.sqrt(hp.nside2pixarea(128)+hp.nside2pixarea(1024)),383)

def correlate_signal(i_file,j_file,wl_i,wl_j,alpha_file,bands,beam=False,gal_cut=0.,mask_file=None):
	print "Computing Cross Correlations for Bands "+str(bands)


	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi')
	delta_alpha_radio=hp.read_map(alpha_file,hdu='uncertainty/phi')
	iqu_band_i=hdu_i['stokes iqu'].data
	iqu_band_j=hdu_j['stokes iqu'].data
	nside_i=hdu_i['stokes iqu'].header['nside']
	nside_j=hdu_j['stokes iqu'].header['nside']
	hdu_i.close()
	hdu_j.close()
	

	ind_i=np.argwhere( wl == wl_i)[0][0]
	ind_j=np.argwhere( wl == wl_j)[0][0]
	npix_i=hp.nside2npix(nside_i)	
	npix_j=hp.nside2npix(nside_j)	
	#ipdb.set_trace()
	if npix_i != iqu_band_i[1].shape[0]:
		print 'NSIDE parameter not equal to size of map for file I'
		print 'setting npix to larger parameter'
		npix_i=iqu_band_i[1].shape[0]
	
	if npix_j != iqu_band_j[1].shape[0]:
		print 'NSIDE parameter not equal to size of map for file J'
		print 'setting npix to larger parameter'
		npix_j=iqu_band_j[1].shape[0]

	sigma_i=[noise_const_pol[ind_i]*np.random.normal(0,1,npix_i),noise_const_pol[ind_i]*np.random.normal(0,1,npix_i)]
	sigma_j=[noise_const_pol[ind_j]*np.random.normal(0,1,npix_j),noise_const_pol[ind_j]*np.random.normal(0,1,npix_j)]
	
	iqu_band_i[1]+=sigma_i[0]
	iqu_band_i[2]+=sigma_i[1]
	iqu_band_j[1]+=sigma_j[0]
	iqu_band_j[2]+=sigma_j[1]
	
	sigma_q_i=hp.smoothing(sigma_i[0],fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_i])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_u_i=hp.smoothing(sigma_i[1],fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_i])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_q_j=hp.smoothing(sigma_j[0],fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_j])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_u_j=hp.smoothing(sigma_j[1],fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_j])**2)*np.pi/(180.*60.),verbose=False)	

	sigma_q_i=hp.ud_grade(sigma_q_i,nside_out)
	sigma_u_i=hp.ud_grade(sigma_u_i,nside_out)
	sigma_q_j=hp.ud_grade(sigma_q_j,nside_out)
	sigma_u_j=hp.ud_grade(sigma_u_j,nside_out)
		
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_i])**2)*np.pi/(180.*60.),verbose=False)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_j])**2)*np.pi/(180.*60.),verbose=False)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	iqu_band_i=hp.ud_grade(iqu_band_i,nside_out=nside_out,order_in='ring')
	iqu_band_j=hp.ud_grade(iqu_band_j,nside_out=nside_out,order_in='ring')
	
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
	
	Bl_factor=np.repeat(1.,3*nside_out)
	#ipdb.set_trace()
	if beam:
		Bl_factor=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),3*nside_out-1)
	pix_area=hp.nside2pixarea(nside_out)
	#ipdb.set_trace()
	mask_bool=np.repeat(False,npix_out)

	if gal_cut > 0:
		pix=np.arange(hp.nside2npix(nside_out))
		x,y,z=hp.pix2vec(nside,pix,nest=0)
		mask_bool= np.abs(z)<= np.sin(gal_cut*np.pi/180.)
	#mask_bool1[np.where(np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)<.2e-6)]=True
	if not (mask_file is None):
		mask_hdu=fits.open(mask_file)
		mask=mask_hdu[1].data.field(0)
		mask_hdu.close()
		
		mask=hp.reorder(mask,n2r=1)
		mask=hp.ud_grade(mask,nside_out=128)
		
		mask_bool=~mask.astype(bool)
		
		fsky= 1. - np.sum(mask)/float(len(mask))	
		L=np.sqrt(fsky*4*np.pi)
		dl_eff=2*np.pi/L
	
	DQm.mask=mask_bool
	DUm.mask=mask_bool
	aQm.mask=mask_bool
	aUm.mask=mask_bool
	sqi.mask=mask_bool
	sui.mask=mask_bool
	sqj.mask=mask_bool
	suj.mask=mask_bool
	salpha.mask=mask_bool
	alpham.mask=mask_bool
	um.mask=mask_bool
	qm.mask=mask_bool
	#ipdb.set_trace()
	cross1=hp.anafast(DQm,map2=aUm)/Bl_factor**2
	cross2=hp.anafast(DUm,map2=aQm)/Bl_factor**2
	

	##calculate theoretical variance for correlations
	N_dq=abs((sqi-sqj)**2).sum()*(pix_area/const)**2/(4.*np.pi)
	N_du=abs((sui-suj)**2).sum()*(pix_area/const)**2/(4.*np.pi)
	N_au=abs((salpha*um+alpham*suj+salpha*suj)**2).sum()*pix_area**2/(4.*np.pi)
	N_aq=abs((salpha*qm+alpham*sqj+salpha*sqj)**2).sum()*pix_area**2/(4.*np.pi)
	#ipdb.set_trace()

	return (cross1,cross2,N_dq,N_du,N_au,N_aq)

def correlate_noise(i_file,j_file,wl_i,wl_j,alpha_file,bands,beam=False,gal_cut=0.,mask_file=None):
	print "Computing Noise Correlation for Bands "+str(bands)


	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi')
	delta_alpha_radio=hp.read_map(alpha_file,hdu='uncertainty/phi')
	#iqu_band_i=hdu_i['stokes iqu'].data
	#iqu_band_j=hdu_j['stokes iqu'].data
	nside_i=hdu_i['stokes iqu'].header['nside']
	nside_j=hdu_j['stokes iqu'].header['nside']
	hdu_i.close()
	hdu_j.close()
	

	ind_i=np.argwhere( wl == wl_i)[0][0]
	ind_j=np.argwhere( wl == wl_j)[0][0]

	npix_i=hp.nside2npix(nside_i)
	npix_j=hp.nside2npix(nside_j)
	iqu_band_i=np.zeros((3,npix_i))
	iqu_band_j=np.zeros((3,npix_j))
	
	sigma_i=[noise_const_pol[ind_i]*np.random.normal(0,1,npix_i),noise_const_pol[ind_i]*np.random.normal(0,1,npix_i)]
	sigma_j=[noise_const_pol[ind_j]*np.random.normal(0,1,npix_j),noise_const_pol[ind_j]*np.random.normal(0,1,npix_j)]
	
	iqu_band_i[1]=np.copy(sigma_i[0])
	iqu_band_i[2]=np.copy(sigma_i[1])
	iqu_band_j[1]=np.copy(sigma_j[0])
	iqu_band_j[2]=np.copy(sigma_j[1])
	
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_i])**2)*np.pi/(180.*60.),verbose=False)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_j])**2)*np.pi/(180.*60.),verbose=False)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	iqu_band_i=hp.ud_grade(iqu_band_i,nside_out=nside_out,order_in='ring')
	iqu_band_j=hp.ud_grade(iqu_band_j,nside_out=nside_out,order_in='ring')
	
	const=2.*(wl_i**2-wl_j**2)	

	Delta_Q=(iqu_band_i[1]-iqu_band_j[1])/const
	Delta_U=(iqu_band_i[2]-iqu_band_j[2])/const
	alpha_u=alpha_radio*iqu_band_j[2] 
	alpha_q=-alpha_radio*iqu_band_j[1]

	DQm=hp.ma(Delta_Q)
	DUm=hp.ma(Delta_U)
	aQm=hp.ma(alpha_q)
	aUm=hp.ma(alpha_u)
	
	Bl_factor=np.repeat(1.,3*nside_out)
	#ipdb.set_trace()
	if beam:
		Bl_factor=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),3*nside_out-1)
	pix_area=hp.nside2pixarea(nside_out)
	#ipdb.set_trace()
	mask_bool=np.repeat(False,npix_out)

	if gal_cut > 0:
		pix=np.arange(hp.nside2npix(nside_out))
		x,y,z=hp.pix2vec(nside,pix,nest=0)
		mask_bool= np.abs(z)<= np.sin(gal_cut*np.pi/180.)
	#mask_bool1[np.where(np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)<.2e-6)]=True
	if not (mask_file is None):
		mask_hdu=fits.open(mask_file)
		mask=mask_hdu[1].data.field(0)
		mask_hdu.close()
		
		mask=hp.reorder(mask,n2r=1)
		mask=hp.ud_grade(mask,nside_out=128)
		
		mask_bool=~mask.astype(bool)
		
		fsky= 1. - np.sum(mask)/float(len(mask))	
		L=np.sqrt(fsky*4*np.pi)
		dl_eff=2*np.pi/L
	
	DQm.mask=mask_bool
	DUm.mask=mask_bool
	aQm.mask=mask_bool
	aUm.mask=mask_bool
	#ipdb.set_trace()
	cross1=hp.anafast(DQm,map2=aUm)/Bl_factor**2
	cross2=hp.anafast(DUm,map2=aQm)/Bl_factor**2
	#cross1=np.mean(cross1_array,axis=0)	##Average over all Cross Spectra
	#cross2=np.mean(cross2_array,axis=0)	##Average over all Cross Spectra
	#hp.write_cl('cl_'+bands+'_FR_noise_QxaU.fits',cross1)
	#hp.write_cl('cl_'+bands+'_FR_noise_UxaQ.fits',cross2)
	return (cross1,cross2)

def correlate_theory(i_file,j_file,wl_i,wl_j,alpha_file,bands_name,beam=False,gal_cut=0.,mask_file=None):
	print "Computing Cross Correlations for Bands "+str(bands_name)

	radio_file='/data/wmap/faraday_MW_realdata.fits'
	cl_file='/home/matt/wmap/simul_scalCls.fits.lens'
	
	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	#iqu_band_i=hdu_i['stokes iqu'].data
	#iqu_band_j=hdu_j['stokes iqu'].data
	nside_i=hdu_i['stokes iqu'].header['nside']
	nside_j=hdu_j['stokes iqu'].header['nside']
	hdu_i.close()
	hdu_j.close()
	ind_i=np.where( wl == wl_i)[0][0]
	ind_j=np.where( wl == wl_j)[0][0]
	
	cls=hp.read_cl(cl_file)
	simul_cmb=hp.sphtfunc.synfast(cls,max(nside_i,nside_j),fwhm=0.,new=1,pol=1);
	
	alpha_radio=hp.read_map(radio_file,hdu='maps/phi');

	##Generate CMB for file J
	
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=nside_j,order_in='ring',order_out='ring')
	simul_cmb=hp.ud_grade(simul_cmb,nside_out=nside_j)
	tmp_cmb=rotate_tqu.rotate_tqu(simul_cmb,wl_j,alpha_radio);
	iqu_band_j=hp.smoothing(tmp_cmb,fwhm=np.sqrt((beam_fwhm[ind_j]*np.pi/(180.*60.))**2-hp.nside2pixarea(nside_i)),verbose=False)
	
	##Generate CMB for file I
	
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=nside_i,order_in='ring',order_out='ring')
	simul_cmb=hp.ud_grade(simul_cmb,nside_out=nside_i)
	tmp_cmb=rotate_tqu.rotate_tqu(simul_cmb,wl_i,alpha_radio);
	#ipdb.set_trace()
	iqu_band_i=hp.smoothing(tmp_cmb,fwhm=np.sqrt((beam_fwhm[ind_i]*np.pi/(180.*60.))**2-hp.nside2pixarea(nside_i)),verbose=False)
	


	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi')
	
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_i])**2)*np.pi/(180.*60.),verbose=False)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_j])**2)*np.pi/(180.*60.),verbose=False)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	iqu_band_i=hp.ud_grade(iqu_band_i,nside_out=nside_out,order_in='ring')
	iqu_band_j=hp.ud_grade(iqu_band_j,nside_out=nside_out,order_in='ring')
	
	const=2.*(wl_i**2-wl_j**2)	

	Delta_Q=(iqu_band_i[1]-iqu_band_j[1])/const
	Delta_U=(iqu_band_i[2]-iqu_band_j[2])/const
	alpha_u=alpha_radio*iqu_band_j[2] 
	alpha_q=-alpha_radio*iqu_band_j[1]

	DQm=hp.ma(Delta_Q)
	DUm=hp.ma(Delta_U)
	aQm=hp.ma(alpha_q)
	aUm=hp.ma(alpha_u)
	
	Bl_factor=np.repeat(1.,3*nside_out)
	#ipdb.set_trace()
	if beam:
		Bl_factor=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),3*nside_out-1)
	pix_area=hp.nside2pixarea(nside_out)
	#ipdb.set_trace()
	mask_bool=np.repeat(False,npix_out)

	if gal_cut > 0:
		pix=np.arange(hp.nside2npix(nside_out))
		x,y,z=hp.pix2vec(nside,pix,nest=0)
		mask_bool= np.abs(z)<= np.sin(gal_cut*np.pi/180.)
	#mask_bool1[np.where(np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)<.2e-6)]=True
	if not (mask_file is None):
		mask_hdu=fits.open(mask_file)
		mask=mask_hdu[1].data.field(0)
		mask_hdu.close()
		
		mask=hp.reorder(mask,n2r=1)
		mask=hp.ud_grade(mask,nside_out=128)
		
		mask_bool=~mask.astype(bool)
		
		fsky= 1. - np.sum(mask)/float(len(mask))	
		L=np.sqrt(fsky*4*np.pi)
		dl_eff=2*np.pi/L
	
	DQm.mask=mask_bool
	DUm.mask=mask_bool
	aQm.mask=mask_bool
	aUm.mask=mask_bool
	cross1=hp.anafast(DQm,map2=aUm)/Bl_factor**2
	cross2=hp.anafast(DUm,map2=aQm)/Bl_factor**2
	#cross1=np.mean(cross1_array,axis=0)	##Average over all Cross Spectra
	#cross2=np.mean(cross2_array,axis=0)	##Average over all Cross Spectra
	return (cross1,cross2)
def plot_mc():
	bins=[1,5,10,20,25,50]
	l=np.arange(3*nside_out)
	ll=l*(l+1)/(2.*np.pi)
	bls=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),3*nside_out-1)**2
	for num, mask_file in enumerate(mask_array):
		f=np.load('prism_simul_'+mask_name[num]+'.npz')
		theory1_array_in=f['the1_in']
		theory2_array_in=f['the2_in']
		cross1_array_in=f['c1_in']
		cross2_array_in=f['c2_in']
		noise1_array_in=f['n1_in']
		noise2_array_in=f['n2_in']
		Ndq_array_in=f['ndq_in']
		Ndu_array_in=f['ndu_in']
		Nau_array_in=f['nau_in']
		Naq_array_in=f['naq_in']

		mask_hdu=fits.open(mask_file)
		mask=mask_hdu[1].data.field(0)
		mask_hdu.close()
		
		mask=hp.reorder(mask,n2r=1)
		mask=hp.ud_grade(mask,nside_out=128)
		
		mask_bool=~mask.astype(bool)
		
		fsky= 1. - np.sum(mask)/float(len(mask))	
		L=np.sqrt(fsky*4*np.pi)
		dl_eff=2*np.pi/L

		theory1_array_in=np.array(theory1_array_in)/fsky
		theory2_array_in=np.array(theory2_array_in)/fsky
		cross1_array_in=np.array(cross1_array_in)/fsky
		cross2_array_in=np.array(cross2_array_in)/fsky
		Ndq_array_in=np.array(Ndq_array_in)/fsky
		Ndu_array_in=np.array(Ndu_array_in)/fsky
		Nau_array_in=np.array(Nau_array_in)/fsky
		Naq_array_in=np.array(Naq_array_in)/fsky
		noise1_array_in=np.array(noise1_array_in)/fsky
		noise2_array_in=np.array(noise2_array_in)/fsky


		for b in bins:
			N_dq=np.mean(Ndq_array_in,axis=1)
			N_au=np.mean(Nau_array_in,axis=1)
			delta1_in=np.sqrt(2.*abs((np.mean(cross1_array_in,axis=1).T-np.mean(noise1_array_in,axis=1).T)**2+(np.mean(cross1_array_in,axis=1).T-np.mean(noise1_array_in,axis=1).T)/2.*(N_dq+N_au)+N_dq*N_au/2.).T/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
		
			cosmic1_in=np.sqrt(2./((2.*l+1)*np.sqrt(b**2+dl_eff**2)*fsky)*np.mean(theory1_array_in,axis=1)**2)

			N_du=np.mean(Ndu_array_in,axis=1)
			N_aq=np.mean(Naq_array_in,axis=1)
			delta2_in=np.sqrt(2.*abs((np.mean(cross2_array_in,axis=1).T-np.mean(noise2_array_in,axis=1).T)**2+(np.mean(cross2_array_in,axis=1).T-np.mean(noise2_array_in,axis=1).T)/2.*(N_dq+N_au)+N_dq*N_au/2.).T/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
			cosmic2_in=np.sqrt(2./((2*l+1)*np.sqrt(b**2+dl_eff**2)*fsky)*np.mean(theory2_array_in,axis=1)**2)

			cross1_array=[[],[],[]]
			cross2_array=[[],[],[]]
			Ndq_array=[[],[],[]]
			Ndu_array=[[],[],[]]
			Nau_array=[[],[],[]]
			Naq_array=[[],[],[]]
			noise1_array=[[],[],[]]
			noise2_array=[[],[],[]]
			theory1_array=[[],[],[]]
			theory2_array=[[],[],[]]
			cosmic1=[[],[],[]]
			cosmic2=[[],[],[]]
			delta1=[[],[],[]]
			delta2=[[],[],[]]
        		
			plot_l=[]
			if( b != 1):
				for m in xrange(len(cross1_array_in)):
		        		for n in xrange(len(cross1_array_in[0])):
		        		        tmp_t1=bin_llcl.bin_llcl(ll*theory1_array_in[m][n]/bls,b)
		        		        tmp_t2=bin_llcl.bin_llcl(ll*theory2_array_in[m][n]/bls,b)
						tmp_c1=bin_llcl.bin_llcl(ll*cross1_array_in[m][n]/bls,b)
		        		        tmp_c2=bin_llcl.bin_llcl(ll*cross2_array_in[m][n]/bls,b)
						tmp_n1=bin_llcl.bin_llcl(ll*noise1_array_in[m][n]/bls,b)
		        		        tmp_n2=bin_llcl.bin_llcl(ll*noise2_array_in[m][n]/bls,b)
		        		        
						theory1_array[m].append(tmp_t1['llcl'])
						theory2_array[m].append(tmp_t2['llcl'])
						
						cross1_array[m].append(tmp_c1['llcl'])
						cross2_array[m].append(tmp_c2['llcl'])
						
						noise1_array[m].append(tmp_n1['llcl'])
						noise2_array[m].append(tmp_n2['llcl'])
		        		        
						if n == len(cross1_array_in[0])-1:
		        		                plot_l=tmp_c1['l_out']
					tmp_c1=bin_llcl.bin_llcl(ll*cosmic1_in[m]/bls,b)
					tmp_d1=bin_llcl.bin_llcl(ll*delta1_in[m]/bls,b)
					cosmic1[m]=tmp_c1['llcl']
					delta1[m]=tmp_d1['llcl']

					tmp_c2=bin_llcl.bin_llcl(ll*cosmic2_in[m]/bls,b)
					tmp_d2=bin_llcl.bin_llcl(ll*delta2_in[m]/bls,b)
					cosmic2[m]=tmp_c2['llcl']
					delta2[m]=tmp_d2['llcl']
					
			else:
				plot_l=l
				theory1_array=np.multiply(ll/bls,theory1_array_in)
				cross1_array=np.multiply(ll/bls,cross1_array_in)
				noise1_array=np.multiply(ll/bls,noise1_array_in)
				theory2_array=np.multiply(ll/bls,theory2_array_in)
				cross2_array=np.multiply(ll/bls,cross2_array_in)
				noise2_array=np.multiply(ll/bls,noise2_array_in)
				cosmic1=cosmic1_in*ll/bls
				cosmic2=cosmic2_in*ll/bls
				delta1=delta1_in*ll/bls
				delta2=delta2_in*ll/bls
			#noise1=np.mean(noise1_array,axis=1)
			#noise2=np.mean(noise2_array,axis=1)
        		theory_array = np.add(theory1_array,theory2_array)
        		theory=np.mean(theory_array,axis=1)
        		dtheory=np.std(theory_array,axis=1,ddof=1)
        		cross_array = np.add(np.subtract(cross1_array,noise1_array),np.subtract(cross2_array,noise2_array))
        		cross=np.mean(cross_array,axis=1)
        		dcross=np.std(cross_array,axis=1,ddof=1)
        		cosmic=np.sqrt(np.array(cosmic1)**2+np.array(cosmic2)**2)
        		delta=np.sqrt(np.array(delta1)**2+np.array(delta2)**2)

			cross=np.average(cross,weights=1./dcross**2,axis=0)
			theory=np.average(theory,weights=1./dcross**2,axis=0)
			dtheory=np.average(dtheory,weights=1./dcross**2,axis=0)
			cosmic=np.average(cosmic,weights=1./dcross**2,axis=0)
			delta=np.average(delta,weights=1./dcross**2,axis=0)
			dcross=np.sqrt(np.average(dcross**2,weights=1./dcross**2,axis=0))

			#theory1=np.mean(theory1_array,axis=0)
			#dtheory1=np.std(theory1_array,axis=0,ddof=1)
			#cross1=np.mean(cross1_array,axis=0)
			#dcross1=np.std(np.subtract(cross1_array,noise1),axis=0,ddof=1)
			#ipdb.set_trace()
			plot_binned.plotBinned((cross)*1e12,dcross*1e12,plot_l,b,'prism_FR_simulation',title='PRISM FR Correlator',theory=theory*1e12,dtheory=dtheory*1e12,delta=delta*1e12,cosmic=cosmic*1e12)

			#theory2=np.mean(theory2_array,axis=0)
			#dtheory2=np.std(theory2_array,axis=0,ddof=1)
			#cross2=np.mean(cross2_array,axis=0)
			##delta2=np.mean(delta2_array,axis=0)
			#dcross2=np.std(np.subtract(cross2_array,noise2),axis=0,ddof=1)
			##ipdb.set_trace()
			#plot_binned.plotBinned((cross2-noise2)*1e12,dcross2*1e12,plot_l,b,'Cross_43x95_FR_UxaQ', title='Cross 43x95 FR UxaQ',theory=theory2*1e12,dtheory=dtheory2*1e12,delta=delta2*1e12,cosmic=cosmic2*1e12)
			#ipdb.set_trace()
    
			if b == 25 :
				a_scales=np.linspace(-2,4,121)
				chi_array=[]
				for a in a_scales:
					chi_array.append(np.sum( (cross - a*theory)**2/(dcross)**2))
				ind = np.argmin(chi_array)
			#likelihood=np.exp(np.multiply(-1./2.,chi_array))/np.sqrt(2*np.pi)
				likelihood=np.exp(np.multiply(-1./2.,chi_array))/np.sum(np.exp(np.multiply(-1./2.,chi_array))*.05)

				Sig=np.sum(cross/(dcross**2))/np.sum(1./dcross**2)
				Noise=np.std(np.sum(cross_array/dcross**2,axis=1)/np.sum(1./dcross**2))
				Sig1=np.sum(cross*(theory/dcross)**2)/np.sum((theory/dcross)**2)
				Noise1=np.std(np.sum(cross_array*(theory/dcross)**2,axis=1)/np.sum((theory/dcross)**2))
				SNR=Sig/Noise
				SNR1=Sig1/Noise1
				
				Sig2=np.sum(cross/(dcross**2))/np.sum(1./dcross**2)
				Noise2=np.sqrt(1./np.sum(1./dcross**2))
				Sig3=np.sum(cross*(theory/dcross)**2)/np.sum((theory/dcross)**2)
				Noise3=np.sqrt(np.sum(theory**2)/np.sum(theory**2/dcross**2))
				SNR2=Sig2/Noise2
				SNR3=Sig3/Noise3
				
				#ipdb.set_trace()
				fig,ax1=plt.subplots(1,1)

				ax1.plot(a_scales,likelihood,'k.')
				ax1.set_title('Faraday Rotation Correlator')
				ax1.set_xlabel('Likelihood scalar')
				ax1.set_ylabel('Likelihood of Correlation')
				fig.savefig('FR_Correlation_Likelihood.png',format='png')
				fig.savefig('FR_Correlation_Likelihood.eps',format='eps')
				#ipdb.set_trace()
				f=open('Maximum_likelihood.txt','w')
				f.write('Maximum Likelihood: {0:2.5f}%  for scale factor {1:.2f} \n'.format(float(likelihood[ind]*100),float(a_scales[ind])))
				f.write('Probability of scale factor =1: {0:2.5f}% \n \n'.format(float(likelihood[np.where(a_scales ==1)])*100))
				f.write('Detection Levels using Standard Deviation \n')
				f.write('Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n'.format(SNR,Sig, Noise))
				f.write('Weighted Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n \n'.format(SNR1,Sig1,Noise))
				f.write('Detection using Theoretical Noise \n')
				f.write('Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n'.format(SNR2,Sig2, Noise2))
				f.write('Weighted Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n'.format(SNR3,Sig3,Noise3))
				f.close()

			#if b == 1 :
			#	xbar= np.matrix(ll[1:]*(cross-np.mean(cross))[1:]).T
			#	vector=np.matrix(ll[1:]*cross[1:]).T
			#	mu=np.matrix(ll[1:]*theory[1:]).T
			#	fact=len(xbar)-1
			#	cov=(np.dot(xbar,xbar.T)/fact).squeeze()
			#	ipdb.set_trace()
			#	likelihood=np.exp(-np.dot(np.dot((vector-mu).T,lin.inv(cov)),(vector-mu))/2. )/(np.sqrt(2*np.pi*lin.det(cov)))
			#	print('Likelihood of fit is #{0:.5f}'.format(likelihood[0,0]))
			#	f=open('FR_likelihood.txt','w')
			#	f.write('Likelihood of fit is #{0:.5f}'.format(likelihood[0,0]))
			#	f.close()

				#subprocess.call('mv Maximum_likelihood.txt  gal_cut_{0:0>2d}/'.format(cut), shell=True)
				subprocess.call('mv *01*.png bin_01/', shell=True)
				subprocess.call('mv *05*.png bin_05/', shell=True)
				subprocess.call('mv *10*.png bin_10/', shell=True)
				subprocess.call('mv *20*.png bin_20/', shell=True)
				subprocess.call('mv *25*.png bin_25/', shell=True)
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
	##	Beam correction
	use_beam=0
	N_runs=100
	bins=[1,5,10,20,25,50]
	gal_cut=[00,05,10,20,30]
	bls=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),3*nside_out-1)**2
	l=np.arange(3*nside_out)
	ll=l*(l+1)/(2*np.pi)

	map_prefix='/home/matt/Planck/data/faraday/simul_maps/'
	file_prefix=map_prefix+'prism_simulated_'
	alpha_file='/data/wmap/faraday_MW_realdata.fits'
	#wl=np.array([299792458./(band*1e9) for band in bands])
	cross1_array_in=[[],[],[]]
	cross2_array_in=[[],[],[]]
	Ndq_array_in=[[],[],[]]
	Ndu_array_in=[[],[],[]]
	Nau_array_in=[[],[],[]]
	Naq_array_in=[[],[],[]]
	noise1_array_in=[[],[],[]]
	noise2_array_in=[[],[],[]]
	theory1_array_in=[[],[],[]]
	theory2_array_in=[[],[],[]]
	

	#simulate_fields.main()
	for num, mask_file in enumerate(mask_array):
		print(Fore.WHITE+Back.RED+Style.BRIGHT+'Mask: '+mask_name[num]+Back.RESET+Fore.RESET+Style.RESET_ALL)
		count=0
		for i in [0,1,2]:
			for j in [3,4,5]:
				#for n in xrange(N_runs):
				for run in xrange(N_runs):	
					print(Fore.WHITE+Back.GREEN+Style.BRIGHT+'Correlation #{:03d}'.format(run+1)+Back.RESET+Fore.RESET+Style.RESET_ALL)
					print('Bands: {0:0>3.0f} and {1:0>3.0f}'.format(bands[i],bands[j]))
					ttmp1,ttmp2=correlate_theory(file_prefix+'{0:0>3.0f}.fits'.format(bands[i]),file_prefix+'{0:0>3.0f}.fits'.format(bands[j]),wl[i],wl[j],alpha_file,'{0:0>3.0f}x{1:0>3.0f}'.format(bands[i],bands[j]),beam=use_beam,mask_file=mask_file)
				#f=open('cl_noise_FR_{0:0>3.0f}x{1:0>3.0f}_cut{2:0>2d}_UxaQ.json'.format(bands[i],bands[j],cut),'w')
					theory1_array_in[count].append(ttmp1)
					theory2_array_in[count].append(ttmp2)
					tmp1,tmp2,n1,n2,n3,n4=correlate_signal(file_prefix+'{0:0>3.0f}.fits'.format(bands[i]),file_prefix+'{0:0>3.0f}.fits'.format(bands[j]),wl[i],wl[j],alpha_file,'{0:0>3.0f}x{1:0>3.0f}'.format(bands[i],bands[j]),beam=use_beam,mask_file=mask_file)
					ntmp1,ntmp2=correlate_noise(file_prefix+'{0:0>3.0f}.fits'.format(bands[i]),file_prefix+'{0:0>3.0f}.fits'.format(bands[j]),wl[i],wl[j],alpha_file,'{0:0>3.0f}x{1:0>3.0f}'.format(bands[i],bands[j]),beam=use_beam,mask_file=mask_file)
					cross1_array_in[count].append(tmp1)
					cross2_array_in[count].append(tmp2)
					Ndq_array_in[count].append(n1)
					Ndu_array_in[count].append(n2)
					Nau_array_in[count].append(n3)
					Naq_array_in[count].append(n4)
					noise1_array_in[count].append(ntmp1)
					noise2_array_in[count].append(ntmp2)
				count+=1
		np.savez('prism_simul_'+mask_name[num]+'.npz',the1_in=theory1_array_in,the2_in=theory2_array_in,c1_in=cross1_array_in,c2_in=cross2_array_in,ndq_in=Ndq_array_in,ndu_in=Ndu_array_in,nau_in=Nau_array_in,naq_in=Naq_array_in,n1_in=noise1_array_in,n2_in=noise2_array_in)
				#f=open('cl_theory_FR_{0:0>3.0f}x{1:0>3.0f}_cut{2:0>2d}_QxaU.json'.format(bands[i],bands[j],cut),'w')
				#json.dump(np.array(theory1_array_in).tolist(),f)
				#f.close()	
				#f=open('cl_theory_FR_{0:0>3.0f}x{1:0>3.0f}_cut{2:0>2d}_UxaQ.json'.format(bands[i],bands[j],cut),'w')
				#json.dump(np.array(theory2_array_in).tolist(),f)
				#f.close()	
				#f=open('cl_array_FR_{0:0>3.0f}x{1:0>3.0f}_cut{2:0>2d}_QxaU.json'.format(bands[i],bands[j],cut),'w')
				#json.dump(np.array(cross1_array_in).tolist(),f)
				#f.close()	
				#f=open('cl_array_FR_{0:0>3.0f}x{1:0>3.0f}_cut{2:0>2d}_UxaQ.json'.format(bands[i],bands[j],cut),'w')
				#json.dump(np.array(cross2_array_in).tolist(),f)
				#f.close()	
				#f=open('cl_noise_FR_{0:0>3.0f}x{1:0>3.0f}_cut{2:0>2d}_QxaU.json'.format(bands[i],bands[j],cut),'w')
				#json.dump(np.array(noise1_array_in).tolist(),f)
				#f.close()	
				#json.dump(np.array(noise2_array_in).tolist(),f)
				#f.close()	
				#f=open('cl_Nau_FR_{0:0>3.0f}x{1:0>3.0f}_cut{2:0>2d}_QxaU.json'.format(bands[i],bands[j],cut),'w')
				#json.dump(np.array(Nau_array_in).tolist(),f)
				#f.close()	
				#f=open('cl_Ndq_FR_{0:0>3.0f}x{1:0>3.0f}_cut{2:0>2d}_QxaU.json'.format(bands[i],bands[j],cut),'w')
				#json.dump(np.array(Ndq_array_in).tolist(),f)
				#f.close()	
				#f=open('cl_Naq_FR_{0:0>3.0f}x{1:0>3.0f}_cut{2:0>2d}_UxaQ.json'.format(bands[i],bands[j],cut),'w')
				#json.dump(np.array(Naq_array_in).tolist(),f)
				#f.close()	
				#f=open('cl_Ndu_FR_{0:0>3.0f}x{1:0>3.0f}_cut{2:0>2d}_UxaQ.json'.format(bands[i],bands[j],cut),'w')
				#json.dump(np.array(Ndu_array_in).tolist(),f)
				#f.close()	
			
				#fsky= 1. - np.sin(cut*np.pi/180.)
				#L=np.sqrt(fsky*4*np.pi)
				#dl_eff=2*np.pi/L

		mask_hdu=fits.open(mask_file)
		mask=mask_hdu[1].data.field(0)
		mask_hdu.close()
		
		mask=hp.reorder(mask,n2r=1)
		mask=hp.ud_grade(mask,nside_out=128)
		
		mask_bool=~mask.astype(bool)
		
		fsky= 1. - np.sum(mask)/float(len(mask))	
		L=np.sqrt(fsky*4*np.pi)
		dl_eff=2*np.pi/L

		theory1_array_in=np.array(theory1_array_in)/fsky
		theory2_array_in=np.array(theory2_array_in)/fsky
		cross1_array_in=np.array(cross1_array_in)/fsky
		cross2_array_in=np.array(cross2_array_in)/fsky
		Ndq_array_in=np.array(Ndq_array_in)/fsky
		Ndu_array_in=np.array(Ndu_array_in)/fsky
		Nau_array_in=np.array(Nau_array_in)/fsky
		Naq_array_in=np.array(Naq_array_in)/fsky
		noise1_array_in=np.array(noise1_array_in)/fsky
		noise2_array_in=np.array(noise2_array_in)/fsky


		for b in bins:
			N_dq=np.mean(Ndq_array_in,axis=1)
			N_au=np.mean(Nau_array_in,axis=1)
			delta1_in=np.sqrt(2.*abs((np.mean(cross1_array_in,axis=1).T-np.mean(noise1_array_in,axis=1).T)**2+(np.mean(cross1_array_in,axis=1).T-np.mean(noise1_array_in,axis=1).T)/2.*(N_dq+N_au)+N_dq*N_au/2.).T/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
		
			cosmic1_in=np.sqrt(2./((2.*l+1)*np.sqrt(b**2+dl_eff**2)*fsky)*np.mean(theory1_array_in,axis=1)**2)

			N_du=np.mean(Ndu_array_in,axis=1)
			N_aq=np.mean(Naq_array_in,axis=1)
			delta2_in=np.sqrt(2.*abs((np.mean(cross2_array_in,axis=1).T-np.mean(noise2_array_in,axis=1).T)**2+(np.mean(cross2_array_in,axis=1).T-np.mean(noise2_array_in,axis=1).T)/2.*(N_dq+N_au)+N_dq*N_au/2.).T/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
			cosmic2_in=np.sqrt(2./((2*l+1)*np.sqrt(b**2+dl_eff**2)*fsky)*np.mean(theory2_array_in,axis=1)**2)

			cross1_array=[[],[],[]]
			cross2_array=[[],[],[]]
			Ndq_array=[[],[],[]]
			Ndu_array=[[],[],[]]
			Nau_array=[[],[],[]]
			Naq_array=[[],[],[]]
			noise1_array=[[],[],[]]
			noise2_array=[[],[],[]]
			theory1_array=[[],[],[]]
			theory2_array=[[],[],[]]
			cosmic1=[[],[],[]]
			cosmic2=[[],[],[]]
			delta1=[[],[],[]]
			delta2=[[],[],[]]
        		
			plot_l=[]
			if( b != 1):
				for m in xrange(len(cross1_array_in)):
		        		for n in xrange(len(cross1_array_in[0])):
		        		        tmp_t1=bin_llcl.bin_llcl(ll*theory1_array_in[m][n]/bls,b)
		        		        tmp_t2=bin_llcl.bin_llcl(ll*theory2_array_in[m][n]/bls,b)
						tmp_c1=bin_llcl.bin_llcl(ll*cross1_array_in[m][n]/bls,b)
		        		        tmp_c2=bin_llcl.bin_llcl(ll*cross2_array_in[m][n]/bls,b)
						tmp_n1=bin_llcl.bin_llcl(ll*noise1_array_in[m][n]/bls,b)
		        		        tmp_n2=bin_llcl.bin_llcl(ll*noise2_array_in[m][n]/bls,b)
		        		        
						theory1_array[m].append(tmp_t1['llcl'])
						theory2_array[m].append(tmp_t2['llcl'])
						
						cross1_array[m].append(tmp_c1['llcl'])
						cross2_array[m].append(tmp_c2['llcl'])
						
						noise1_array[m].append(tmp_n1['llcl'])
						noise2_array[m].append(tmp_n2['llcl'])
		        		        
						if n == len(cross1_array_in[0])-1:
		        		                plot_l=tmp_c1['l_out']
					tmp_c1=bin_llcl.bin_llcl(ll*cosmic1_in[m]/bls,b)
					tmp_d1=bin_llcl.bin_llcl(ll*delta1_in[m]/bls,b)
					cosmic1[m]=tmp_c1['llcl']
					delta1[m]=tmp_d1['llcl']

					tmp_c2=bin_llcl.bin_llcl(ll*cosmic2_in[m]/bls,b)
					tmp_d2=bin_llcl.bin_llcl(ll*delta2_in[m]/bls,b)
					cosmic2[m]=tmp_c2['llcl']
					delta2[m]=tmp_d2['llcl']
					
			else:
				plot_l=l
				theory1_array=np.multiply(ll/bls,theory1_array_in)
				cross1_array=np.multiply(ll/bls,cross1_array_in)
				noise1_array=np.multiply(ll/bls,noise1_array_in)
				theory2_array=np.multiply(ll/bls,theory2_array_in)
				cross2_array=np.multiply(ll/bls,cross2_array_in)
				noise2_array=np.multiply(ll/bls,noise2_array_in)
				cosmic1=cosmic1_in*ll/bls
				cosmic2=cosmic2_in*ll/bls
				delta1=delta1_in*ll/bls
				delta2=delta2_in*ll/bls
			#noise1=np.mean(noise1_array,axis=1)
			#noise2=np.mean(noise2_array,axis=1)
        		theory_array = np.add(theory1_array,theory2_array)
        		theory=np.mean(theory_array,axis=1)
        		dtheory=np.std(theory_array,axis=1,ddof=1)
        		cross_array = np.add(np.subtract(cross1_array,noise1_array),np.subtract(cross2_array,noise2_array))
        		cross=np.mean(cross_array,axis=1)
        		dcross=np.std(cross_array,axis=1,ddof=1)
        		cosmic=np.sqrt(np.array(cosmic1)**2+np.array(cosmic2)**2)
        		delta=np.sqrt(np.array(delta1)**2+np.array(delta2)**2)

			cross=np.average(cross,weights=1./dcross**2,axis=0)
			theory=np.average(theory,weights=1./dcross**2,axis=0)
			dtheory=np.average(dtheory,weights=1./dcross**2,axis=0)
			cosmic=np.average(cosmic,weights=1./dcross**2,axis=0)
			delta=np.average(delta,weights=1./dcross**2,axis=0)
			dcross=np.sqrt(np.average(dcross**2,weights=1./dcross**2,axis=0))

			#theory1=np.mean(theory1_array,axis=0)
			#dtheory1=np.std(theory1_array,axis=0,ddof=1)
			#cross1=np.mean(cross1_array,axis=0)
			#dcross1=np.std(np.subtract(cross1_array,noise1),axis=0,ddof=1)
			#ipdb.set_trace()
			plot_binned.plotBinned((cross)*1e12,dcross*1e12,plot_l,b,'prism_FR_simulation',title='PRISM FR Correlator',theory=theory*1e12,dtheory=dtheory*1e12,delta=delta*1e12,cosmic=cosmic*1e12)

			#theory2=np.mean(theory2_array,axis=0)
			#dtheory2=np.std(theory2_array,axis=0,ddof=1)
			#cross2=np.mean(cross2_array,axis=0)
			##delta2=np.mean(delta2_array,axis=0)
			#dcross2=np.std(np.subtract(cross2_array,noise2),axis=0,ddof=1)
			##ipdb.set_trace()
			#plot_binned.plotBinned((cross2-noise2)*1e12,dcross2*1e12,plot_l,b,'Cross_43x95_FR_UxaQ', title='Cross 43x95 FR UxaQ',theory=theory2*1e12,dtheory=dtheory2*1e12,delta=delta2*1e12,cosmic=cosmic2*1e12)
			#ipdb.set_trace()
    
			if b == 25 :
				a_scales=np.linspace(-2,4,121)
				chi_array=[]
				for a in a_scales:
					chi_array.append(np.sum( (cross - a*theory)**2/(dcross)**2))
				ind = np.argmin(chi_array)
			#likelihood=np.exp(np.multiply(-1./2.,chi_array))/np.sqrt(2*np.pi)
				likelihood=np.exp(np.multiply(-1./2.,chi_array))/np.sum(np.exp(np.multiply(-1./2.,chi_array))*.05)

				Sig=np.sum(cross/(dcross**2))/np.sum(1./dcross**2)
				Noise=np.std(np.sum(cross_array/dcross**2,axis=1)/np.sum(1./dcross**2))
				Sig1=np.sum(cross*(theory/dcross)**2)/np.sum((theory/dcross)**2)
				Noise1=np.std(np.sum(cross_array*(theory/dcross)**2,axis=1)/np.sum((theory/dcross)**2))
				SNR=Sig/Noise
				SNR1=Sig1/Noise1
				
				Sig2=np.sum(cross/(dcross**2))/np.sum(1./dcross**2)
				Noise2=np.sqrt(1./np.sum(1./dcross**2))
				Sig3=np.sum(cross*(theory/dcross)**2)/np.sum((theory/dcross)**2)
				Noise3=np.sqrt(np.sum(theory**2)/np.sum(theory**2/dcross**2))
				SNR2=Sig2/Noise2
				SNR3=Sig3/Noise3
				
				#ipdb.set_trace()
				fig,ax1=plt.subplots(1,1)

				ax1.plot(a_scales,likelihood,'k.')
				ax1.set_title('Faraday Rotation Correlator')
				ax1.set_xlabel('Likelihood scalar')
				ax1.set_ylabel('Likelihood of Correlation')
				fig.savefig('FR_Correlation_Likelihood.png',format='png')
				fig.savefig('FR_Correlation_Likelihood.eps',format='eps')
				#ipdb.set_trace()
				f=open('Maximum_likelihood.txt','w')
				f.write('Maximum Likelihood: {0:2.5f}%  for scale factor {1:.2f} \n'.format(float(likelihood[ind]*100),float(a_scales[ind])))
				f.write('Probability of scale factor =1: {0:2.5f}% \n \n'.format(float(likelihood[np.where(a_scales ==1)])*100))
				f.write('Detection Levels using Standard Deviation \n')
				f.write('Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n'.format(SNR,Sig, Noise))
				f.write('Weighted Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n \n'.format(SNR1,Sig1,Noise))
				f.write('Detection using Theoretical Noise \n')
				f.write('Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n'.format(SNR2,Sig2, Noise2))
				f.write('Weighted Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n'.format(SNR3,Sig3,Noise3))
				f.close()

			#if b == 1 :
			#	xbar= np.matrix(ll[1:]*(cross-np.mean(cross))[1:]).T
			#	vector=np.matrix(ll[1:]*cross[1:]).T
			#	mu=np.matrix(ll[1:]*theory[1:]).T
			#	fact=len(xbar)-1
			#	cov=(np.dot(xbar,xbar.T)/fact).squeeze()
			#	ipdb.set_trace()
			#	likelihood=np.exp(-np.dot(np.dot((vector-mu).T,lin.inv(cov)),(vector-mu))/2. )/(np.sqrt(2*np.pi*lin.det(cov)))
			#	print('Likelihood of fit is #{0:.5f}'.format(likelihood[0,0]))
			#	f=open('FR_likelihood.txt','w')
			#	f.write('Likelihood of fit is #{0:.5f}'.format(likelihood[0,0]))
			#	f.close()

				#subprocess.call('mv Maximum_likelihood.txt  gal_cut_{0:0>2d}/'.format(cut), shell=True)
				subprocess.call('mv *01*.png bin_01/', shell=True)
				subprocess.call('mv *05*.png bin_05/', shell=True)
				subprocess.call('mv *10*.png bin_10/', shell=True)
				subprocess.call('mv *20*.png bin_20/', shell=True)
				subprocess.call('mv *25*.png bin_25/', shell=True)
				subprocess.call('mv *50*.png bin_50/', shell=True)
				subprocess.call('mv *.eps eps/', shell=True)
if __name__=='__main__':
	main()
