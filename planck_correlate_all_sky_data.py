import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from astropy.io import fits
import ipdb
import glob
import plot_binned
import bin_llcl
import subprocess
import rotate_tqu
#from scipy import linalg as lin

##Global smoothing size parameter##

smoothing_scale=35.0
nside_out=128
npix_out=hp.nside2npix(nside_out)
bands=np.array([30.,44.,70.,100.,143.,217.,353.])
wl=np.array([299792458./(b*1e9) for b in bands])
beam_fwhm=np.array([32.29,27,13.21,9.67,7.26,4.96,4.93])
#beam_fwhm=np.array([33.,24.,14.,10.,7.1,5.0,5.0])
#noise_const_temp=np.array([2.0,2.7,4.7,2.5,2.2,4.8,14.7])*2.7255e-6
#noise_const_pol=np.array([2.8,3.9,6.7,4.0,4.2,9.8,29.8])*2.7255e-6
noise_const_temp=np.array([2.5,2.7,3.5,1.29,.555,.78,2.56])*60.e-6
noise_const_pol=np.array([3.5,4.0,5.0,1.96,1.17,1.75,7.31])*60.e-6

##ONLY use THESE Bands
good_index=[0,1,2]
bands=bands[good_index]
noise_const_temp=noise_const_temp[good_index]
noise_const_pol=noise_const_pol[good_index]

###map_bl=hp.gauss_beam(np.sqrt(hp.nside2pixarea(128)+hp.nside2pixarea(1024)),383)

def correlate_signal(i_file,j_file,wl_i,wl_j,alpha_file,bands,beam=False,gal_cut=0.):
	print "\t \tSignal Correlation"

	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi',verbose=False)
	delta_alpha_radio=hp.read_map(alpha_file,hdu='uncertainty/phi',verbose=False)
	iqu_band_i=np.array([hdu_i['freq-map'].data.field(i) for i in xrange(3)])
	iqu_band_j=np.array([hdu_j['freq-map'].data.field(i) for i in xrange(3)])
	sigma_q_i=np.sqrt(hdu_i['freq-map'].data.field('QQ_cov '))
	sigma_u_i=np.sqrt(hdu_i['freq-map'].data.field('UU_cov '))
	sigma_q_j=np.sqrt(hdu_j['freq-map'].data.field('QQ_cov '))
	sigma_u_j=np.sqrt(hdu_j['freq-map'].data.field('UU_cov '))
	nside_i=hdu_i['freq-map'].header['nside']
	nside_j=hdu_j['freq-map'].header['nside']
	order_i=hdu_i['freq-map'].header['ordering']
	order_j=hdu_j['freq-map'].header['ordering']
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
	sigma_i=np.array([sigma_q_i*np.random.normal(0,1,npix_i),sigma_u_i*np.random.normal(0,1,npix_i)])
	sigma_j=np.array([sigma_q_j*np.random.normal(0,1,npix_j),sigma_u_j*np.random.normal(0,1,npix_j)])
#	sigma_i=[noise_const_pol[ind_i]*np.random.normal(0,1,npix_i),noise_const_pol[ind_i]*np.random.normal(0,1,npix_i)]
#	sigma_j=[noise_const_pol[ind_j]*np.random.normal(0,1,npix_j),noise_const_pol[ind_j]*np.random.normal(0,1,npix_j)]
	
#	iqu_band_i[1]+=sigma_i[0]
#	iqu_band_i[2]+=sigma_i[1]
#	iqu_band_j[1]+=sigma_j[0]
#	iqu_band_j[2]+=sigma_j[1]
	sigma_i=hp.reorder(sigma_i,n2r=1)	
	sigma_j=hp.reorder(sigma_j,n2r=1)	

	sigma_q_i=hp.smoothing(sigma_i[0],fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_i])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_u_i=hp.smoothing(sigma_i[1],fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_i])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_q_j=hp.smoothing(sigma_j[0],fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_j])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_u_j=hp.smoothing(sigma_j[1],fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_j])**2)*np.pi/(180.*60.),verbose=False)	

	sigma_q_i=hp.ud_grade(sigma_q_i,nside_out=nside_out,order_in='ring',order_out='ring')
	sigma_u_i=hp.ud_grade(sigma_u_i,nside_out=nside_out,order_in='ring',order_out='ring')
	sigma_q_j=hp.ud_grade(sigma_q_j,nside_out=nside_out,order_in='ring',order_out='ring')
	sigma_u_j=hp.ud_grade(sigma_u_j,nside_out=nside_out,order_in='ring',order_out='ring')
	iqu_band_i=hp.reorder(iqu_band_i,n2r=1)		
	iqu_band_j=hp.reorder(iqu_band_j,n2r=1)		
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_i])**2)*np.pi/(180.*60.),verbose=False)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_j])**2)*np.pi/(180.*60.),verbose=False)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	iqu_band_i=hp.ud_grade(iqu_band_i,nside_out=nside_out,order_in='ring',order_out='ring')
	iqu_band_j=hp.ud_grade(iqu_band_j,nside_out=nside_out,order_in='ring',order_out='ring')
	
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
	#ipdb.set_trace()	
	Bl_factor=np.repeat(1.,3*nside_out)
	#ipdb.set_trace()
	if beam:
		Bl_factor=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),3*nside_out-1)
	pix_area=hp.nside2pixarea(nside_out)
	#ipdb.set_trace()
	mask_bool=np.repeat(False,npix_out)

	if gal_cut > 0:
		pix=np.arange(hp.nside2npix(nside_out))
		x,y,z=hp.pix2vec(nside_out,pix,nest=0)
		mask_bool= np.abs(z)<= np.sin(gal_cut*np.pi/180.)
	elif gal_cut < 0:
		pix=np.arange(hp.nside2npix(nside_out))
		x,y,z=hp.pix2vec(nside_out,pix,nest=0)
		mask_bool= np.abs(z)>= np.sin(np.abs(gal_cut)*np.pi/180.)
	#mask_bool1[np.where(np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)<.2e-6)]=True
	
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

def correlate_noise(i_file,j_file,wl_i,wl_j,alpha_file,bands,beam=False,gal_cut=0.):
	print "\t \tNoise Correlation"


	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi',verbose=False)
	delta_alpha_radio=hp.read_map(alpha_file,hdu='uncertainty/phi',verbose=False)
	iqu_band_i=np.array([hdu_i['freq-map'].data.field(i) for i in xrange(3)])
	iqu_band_j=np.array([hdu_j['freq-map'].data.field(i) for i in xrange(3)])
	sigma_q_i=np.sqrt(hdu_i['freq-map'].data.field('QQ_cov '))
	sigma_u_i=np.sqrt(hdu_i['freq-map'].data.field('UU_cov '))
	sigma_q_j=np.sqrt(hdu_j['freq-map'].data.field('QQ_cov '))
	sigma_u_j=np.sqrt(hdu_j['freq-map'].data.field('UU_cov '))
	nside_i=hdu_i['freq-map'].header['nside']
	nside_j=hdu_j['freq-map'].header['nside']
	order_i=hdu_i['freq-map'].header['ordering']
	order_j=hdu_j['freq-map'].header['ordering']
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
	sigma_i=np.array([sigma_q_i*np.random.normal(0,1,npix_i),sigma_u_i*np.random.normal(0,1,npix_i)])
	sigma_j=np.array([sigma_q_j*np.random.normal(0,1,npix_j),sigma_u_j*np.random.normal(0,1,npix_j)])
#	sigma_i=[noise_const_pol[ind_i]*np.random.normal(0,1,npix_i),noise_const_pol[ind_i]*np.random.normal(0,1,npix_i)]
#	sigma_j=[noise_const_pol[ind_j]*np.random.normal(0,1,npix_j),noise_const_pol[ind_j]*np.random.normal(0,1,npix_j)]
	sigma_i=hp.reorder(sigma_i,n2r=1)
	sigma_j=hp.reorder(sigma_j,n2r=1)

	iqu_band_i[1]=np.copy(sigma_i[0])
	iqu_band_i[2]=np.copy(sigma_i[1])
	iqu_band_j[1]=np.copy(sigma_j[0])
	iqu_band_j[2]=np.copy(sigma_j[1])
	
		
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_i])**2)*np.pi/(180.*60.),verbose=False)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(beam_fwhm[ind_j])**2)*np.pi/(180.*60.),verbose=False)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	iqu_band_i=hp.ud_grade(iqu_band_i,nside_out=nside_out,order_in=order_i,order_out='ring')
	iqu_band_j=hp.ud_grade(iqu_band_j,nside_out=nside_out,order_in=order_j,order_out='ring')
	
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
		x,y,z=hp.pix2vec(nside_out,pix,nest=0)
		mask_bool= np.abs(z) <= np.sin(gal_cut*np.pi/180.)
	elif gal_cut > 0:
		pix=np.arange(hp.nside2npix(nside_out))
		x,y,z=hp.pix2vec(nside_out,pix,nest=0)
		mask_bool= np.abs(z) >= np.sin(np.abs(gal_cut)*np.pi/180.)
	#mask_bool1[np.where(np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)<.2e-6)]=True
	
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

def plot_mc():
	bins=[1,5,10,20,25,50]
	gal_cut=[-20,-10,-05,00,05,10,20]
	bls=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),3*nside_out-1)**2
	l=np.arange(3*nside_out)
	ll=l*(l+1)/(2*np.pi)
	for cut in gal_cut:
		data=np.load('FR_planck_cut_{0:2>02d}.npz'.format(cut))
		
		theory_in=data['theory']
		cross1_array_in=data['c1']
		cross2_array_in=data['c2']
		noise1_array_in=data['n1']
		noise2_array_in=data['n2']
		Ndq_array_in=data['ndq']	
		Ndu_array_in=data['ndu']	
		Naq_array_in=data['naq']	
		Nau_array_in=data['nau']	
		
		#print( str(np.shape(cross1_array_in)) + ' line 596')
		if cut >= 0:	
			fsky= 1. - np.sin(cut*np.pi/180.)
		else:
			fsky= np.abs(np.sin(cut*np.pi/180.))
		L=np.sqrt(fsky*4*np.pi)
		dl_eff=2*np.pi/L
		
		#print( str(np.shape(cross1_array_in)) + ' line 604')
		#ipdb.set_trace()
		cross1_array_in=np.array(cross1_array_in)/fsky
		cross2_array_in=np.array(cross2_array_in)/fsky
		noise1_array_in=np.array(noise1_array_in)/fsky
		noise2_array_in=np.array(noise2_array_in)/fsky
		Nau_array_in=np.array(Nau_array_in)/fsky
		Naq_array_in=np.array(Naq_array_in)/fsky
		Ndu_array_in=np.array(Ndu_array_in)/fsky
		Ndq_array_in=np.array(Ndq_array_in)/fsky

		#ipdb.set_trace()

		for b in bins:
			#N_dq=np.mean(Ndq_array_in)
			#N_au=np.mean(Nau_array_in)
			#Transpose arrays to match dimensions for operations
			#cross1_array_in=cross1_array_in.T
			#cross2_array_in=cross2_array_in.T
			#noise1_array_in=noise1_array_in.T
			#noise2_array_in=noise2_array_in.T
			
			#ipdb.set_trace()

			delta1=np.sqrt( np.divide(2.*abs( (cross1_array_in - noise1_array_in).T**2 + (cross1_array_in - noise1_array_in). T * (Ndq_array_in+Nau_array_in)/2. + Ndq_array_in*Nau_array_in/2.).T,((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky)))

			#N_du=np.mean(Ndu_array_in)
			#N_aq=np.mean(Naq_array_in)
			delta2=np.sqrt( np.divide(2.*abs( (cross2_array_in - noise2_array_in).T**2 + (cross2_array_in - noise2_array_in).T * (Ndu_array_in+Naq_array_in)/2. + Ndu_array_in*Naq_array_in/2.).T,((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky)))
		
			#Transpose arrays back to match for plotting 
			#cross1_array=cross1_array_in.T
			#cross2_array=cross2_array_in.T
			#noise1_array=noise1_array_in.T
			#noise2_array=noise2_array_in.T
			#ipdb.set_trace()	
			
			cosmic=np.sqrt(2./((2.*l+1)*np.sqrt(b**2+dl_eff**2)*fsky)*theory_in**2)
			
        		delta_array=np.sqrt(delta1**2+delta2**2)	
        		cross_array = np.add(np.subtract(cross1_array_in,noise1_array_in),np.subtract(cross2_array_in,noise2_array_in))
        		cross=np.average(cross_array,weights=1./delta_array**2,axis=0)
			dcross=np.average(delta_array,weights=1./delta_array**2,axis=0)
			plot_l=[]
			if( b != 1):
				tmp_c1=bin_llcl.bin_llcl(ll*cross/bls,b)
		        	                            
				cross=tmp_c1['llcl']
				
				tmp_dc1=bin_llcl.bin_llcl(ll*dcross/bls,b)
		        	                            
				dcross= tmp_dc1['llcl']
		        	
		        	plot_l=tmp_c1['l_out']
				tmp_t1=bin_llcl.bin_llcl(ll*theory_in/bls,b)
				theory=tmp_t1['llcl']
				tmp_c1=bin_llcl.bin_llcl(ll*cosmic/bls,b)
				cosmic=tmp_c1['llcl']
				
			
				
			else:
				plot_l=l
				theory=np.multiply(ll/bls,theory_in)
				#cross1_array=np.multiply(ll/bls,cross1_array_in)
				#noise1_array=np.multiply(ll/bls,noise1_array_in)
				#cross2_array=np.multiply(ll/bls,cross2_array_in)
				#noise2_array=np.multiply(ll/bls,noise2_array_in)
				cross*=ll/bls
				cosmic*=ll/bls
				dcross*=ll/bls
				#cosmic2*=ll/bls
				#delta1*=ll/bls
				#delta2*=ll/bls
				#ipdb.set_trace()
			bad=np.where(plot_l < dl_eff)
			#noise1=np.mean(noise1_array,axis=0)
			#noise2=np.mean(noise2_array,axis=0)
        		#theory_array = np.add(theory1_array,theory2_array)
        		#theory=np.mean(theory_array,axis=0)
        		#dtheory=np.std(theory_array,axis=0,ddof=1)
        		#cross_array = np.add(np.subtract(cross1_array,noise1_array),np.subtract(cross2_array,noise2_array))
        		#delta_array=np.sqrt(delta1**2+delta2**2)
			##cross_array=np.add(cross1_array,cross2_array)
			#ipdb.set_trace()
        		#cross=np.average(cross_array,weights=1./delta_array**2,axis=0)
        		#cosmic=np.sqrt(cosmic1**2+cosmic2**2)
			#theory1=np.mean(theory1_array,axis=0)
			#dtheory1=np.std(theory1_array,axis=0,ddof=1)
			#cross1=np.mean(cross1_array,axis=0)
			#dcross1=np.std(np.subtract(cross1_array,noise1),axis=0,ddof=1)
			#dcross=np.average(delta_array,weights=1./delta_array**2,axis=0)
			#ipdb.set_trace()
			plot_binned.plotBinned((cross)*1e12,dcross*1e12,plot_l,b,'Cross_FR_cut_{0:0>2d}'.format(cut), title='Faraday Rotation Correlator',theory=theory*1e12,dtheory=cosmic*1e12)

			#theory2=np.mean(theory2_array,axis=0)
			#dtheory2=np.std(theory2_array,axis=0,ddof=1)
			#cross2=np.mean(cross2_array,axis=0)
			##delta2=np.mean(delta2_array,axis=0)
			#dcross2=np.std(np.subtract(cross2_array,noise2),axis=0,ddof=1)
			##ipdb.set_trace()
			#plot_binned.plotBinned((cross2-noise2)*1e12,dcross2*1e12,plot_l,b,'Cross_43x95_FR_UxaQ', title='Cross 43x95 FR UxaQ',theory=theory2*1e12,dtheory=dtheory2*1e12,delta=delta2*1e12,cosmic=cosmic2*1e12)
			#ipdb.set_trace()
    
			if b == 25 :
				a_scales=np.linspace(-10,10,1001)
				chi_array=[]
				for a in a_scales:
					chi_array.append(np.sum( (cross - a*theory)**2/(dcross)**2))
				ind = np.argmin(chi_array)
			#likelihood=np.exp(np.multiply(-1./2.,chi_array))/np.sqrt(2*np.pi)
				likelihood=np.exp(np.multiply(-1./2.,chi_array))/np.sum(np.exp(np.multiply(-1./2.,chi_array))*.05)

				Sig=np.sum(cross/(dcross**2))/np.sum(1./dcross**2)
				#Noise=np.std(np.sum(cross_array/dcross**2,axis=1)/np.sum(1./dcross**2))	\
				Noise=np.sqrt(1./np.sum(1./dcross**2))
				Sig1=np.sum(cross*(theory/dcross)**2)/np.sum((theory/dcross)**2)
				Noise1=np.sum(dcross*(theory/dcross)**2)/np.sum((theory/dcross)**2)
				SNR=Sig/Noise
				SNR1=Sig1/Noise1
				
				#Sig2=np.sum(cross/(dcross**2))/np.sum(1./dcross**2)
				#Noise2=np.sqrt(1./np.sum(1./dcross**2))
				#Sig3=np.sum(cross*(theory/dcross)**2)/np.sum((theory/dcross)**2)
				#Noise3=np.sqrt(np.sum(theory**2)/np.sum(theory**2/dcross**2))
				#SNR2=Sig2/Noise2
				#SNR3=Sig3/Noise3
				
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
				#f.write('Detection using Theoretical Noise \n')
				#f.write('Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n'.format(SNR2,Sig2, Noise2))
				#f.write('Weighted Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n'.format(SNR3,Sig3,Noise3))
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

	subprocess.call('mv Maximum_likelihood_cut_{:2>02d}.txt' .format(cut), shell=True)
	subprocess.call('mv *01*.png bin_01/', shell=True)
	subprocess.call('mv *05*.png bin_05/', shell=True)
	subprocess.call('mv *10*.png bin_10/', shell=True)
	subprocess.call('mv *20*.png bin_20/', shell=True)
	subprocess.call('mv *25*.png bin_25/', shell=True)
	subprocess.call('mv *50*.png bin_50/', shell=True)
	subprocess.call('mv *.eps eps/'.format(cut), shell=True)

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
	N_runs=10
	bins=[1,5,10,20,25,50]
	gal_cut=[-20,-10,-05,00,05,10,20]
	bls=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),3*nside_out-1)**2
	l=np.arange(3*nside_out)
	ll=l*(l+1)/(2*np.pi)

	#map_prefix='/home/matt/Planck/data/faraday/simul_maps/'
	#file_prefix=map_prefix+'planck_simulated_'
	alpha_file='/data/wmap/faraday_MW_realdata.fits'
	#wl=np.array([299792458./(band*1e9) for band in bands])
	#theory1_array_in=[]
	#theory2_array_in=[]
	
	dsets=glob.glob('/data/Planck/LFI*1024*R2.00*.fits')
	dsets.sort()
	#simulate_fields.main()
	for cut in gal_cut:
		cross1_array_in=[]
		cross2_array_in=[]
		dcross1_array_in=[]
		dcross2_array_in=[]
		Ndq_array_in=[]
		Ndu_array_in=[]
		Nau_array_in=[]
		Naq_array_in=[]
		noise1_array_in=[]
		noise2_array_in=[]
		
		print('Galactic cut: {:2d}'.format(cut))
		for i in xrange(len(bands)-1):
			for j in xrange(i+1,len(bands)):
				print('    Bands: {0:0>3.0f} x {1:0>3.0f}'.format(bands[i],bands[j]))
				tmp_cross1_array=[]
				tmp_cross2_array=[]
				tmp_noise1_array=[]
				tmp_noise2_array=[]
				##arrays used to average over noise realizations
				tmp_Ndq_array=[]
				tmp_Ndu_array=[]
				tmp_Nau_array=[]
				tmp_Naq_array=[]
				
				for n in xrange(N_runs):
					print('\t Correlation #{:0>3d}'.format(n+1))
					tmp_c1,tmp_c2,tmp_dq,tmp_du,tmp_au,tmp_aq=correlate_signal(dsets[i],dsets[j],wl[i],wl[j],alpha_file,'{0:0>3.0f}x{1:0>3.0f}'.format(bands[i],bands[j]),beam=use_beam,gal_cut=cut)
					tmp_cross1_array.append(tmp_c1)
					tmp_cross2_array.append(tmp_c2)
					tmp_Ndq_array.append(tmp_dq)
					tmp_Ndu_array.append(tmp_du)
					tmp_Naq_array.append(tmp_aq)
					tmp_Nau_array.append(tmp_au)
					
					tmp_n1,tmp_n2=correlate_noise(dsets[i],dsets[j],wl[i],wl[j],alpha_file,'{0:0>3.0f}x{1:0>3.0f}'.format(bands[i],bands[j]),beam=use_beam,gal_cut=cut)
					tmp_noise1_array.append(tmp_n1)
					tmp_noise2_array.append(tmp_n2)
				
				cross1_array_in.append(np.mean(tmp_cross1_array,axis=0))
				cross2_array_in.append(np.mean(tmp_cross2_array,axis=0))
				noise1_array_in.append(np.mean(tmp_noise1_array,axis=0))
				noise2_array_in.append(np.mean(tmp_noise2_array,axis=0))
				Ndq_array_in.append(np.mean(tmp_Ndq_array))
				Ndu_array_in.append(np.mean(tmp_Ndu_array))
				Nau_array_in.append(np.mean(tmp_Nau_array))
				Naq_array_in.append(np.mean(tmp_Naq_array))
				#print( str(np.shape(cross1_array_in) ) + ' line 587')

		#read in theory_cls
		theory_in=hp.read_cl('/home/matt/Planck/data/faraday/correlation/fr_theory_cl.fits')
		
		#print( str(np.shape(cross1_array_in)) + ' line 592' )
		#out_dic={'theory':theory_in,'c1':cross1_array_in,'c2':cross2_array_in,'n1':noise1_array_in,'n2':noise2_array_in,'ndq':Ndq_array_in,'nau':Nau_array_in,'ndu':Ndu_array_in,'naq':Naq_array_in}	
		np.savez('FR_planck_cut_{0:2>02d}.npz'.format(cut),theory=theory_in,c1=cross1_array_in,c2=cross2_array_in,n1=noise1_array_in,n2=noise2_array_in,ndq=Ndq_array_in,nau=Nau_array_in,ndu=Ndu_array_in,naq=Naq_array_in)
	
		#print( str(np.shape(cross1_array_in)) + ' line 596')
		if cut >= 0:	
			fsky= 1. - np.sin(cut*np.pi/180.)
		else:
			fsky= np.abs(np.sin(cut*np.pi/180.))
		L=np.sqrt(fsky*4*np.pi)
		dl_eff=2*np.pi/L
		
		#print( str(np.shape(cross1_array_in)) + ' line 604')
		#ipdb.set_trace()
		cross1_array_in=np.array(cross1_array_in)/fsky
		cross2_array_in=np.array(cross2_array_in)/fsky
		noise1_array_in=np.array(noise1_array_in)/fsky
		noise2_array_in=np.array(noise2_array_in)/fsky
		Nau_array_in=np.array(Nau_array_in)/fsky
		Naq_array_in=np.array(Naq_array_in)/fsky
		Ndu_array_in=np.array(Ndu_array_in)/fsky
		Ndq_array_in=np.array(Ndq_array_in)/fsky

		#ipdb.set_trace()

		for b in bins:
			#N_dq=np.mean(Ndq_array_in)
			#N_au=np.mean(Nau_array_in)
			#Transpose arrays to match dimensions for operations
			#cross1_array_in=cross1_array_in.T
			#cross2_array_in=cross2_array_in.T
			#noise1_array_in=noise1_array_in.T
			#noise2_array_in=noise2_array_in.T
			
			#ipdb.set_trace()

			delta1=np.sqrt( np.divide(2.*abs( (cross1_array_in - noise1_array_in).T**2 + (cross1_array_in - noise1_array_in). T * (Ndq_array_in+Nau_array_in)/2. + Ndq_array_in*Nau_array_in/2.).T,((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky)))

			#N_du=np.mean(Ndu_array_in)
			#N_aq=np.mean(Naq_array_in)
			delta2=np.sqrt( np.divide(2.*abs( (cross2_array_in - noise2_array_in).T**2 + (cross2_array_in - noise2_array_in).T * (Ndu_array_in+Naq_array_in)/2. + Ndu_array_in*Naq_array_in/2.).T,((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky)))
		
			#Transpose arrays back to match for plotting 
			#cross1_array=cross1_array_in.T
			#cross2_array=cross2_array_in.T
			#noise1_array=noise1_array_in.T
			#noise2_array=noise2_array_in.T
			#ipdb.set_trace()	
			
			cosmic=np.sqrt(2./((2.*l+1)*np.sqrt(b**2+dl_eff**2)*fsky)*theory_in**2)
			
        		delta_array=np.sqrt(delta1**2+delta2**2)	
        		cross_array = np.add(np.subtract(cross1_array_in,noise1_array_in),np.subtract(cross2_array_in,noise2_array_in))
        		cross=np.average(cross_array,weights=1./delta_array**2,axis=0)
			dcross=np.average(delta_array,weights=1./delta_array**2,axis=0)
			plot_l=[]
			if( b != 1):
				tmp_c1=bin_llcl.bin_llcl(ll*cross/bls,b)
		        	                            
				cross=tmp_c1['llcl']
				
				tmp_dc1=bin_llcl.bin_llcl(ll*dcross/bls,b)
		        	                            
				dcross= tmp_dc1['llcl']
		        	
		        	plot_l=tmp_c1['l_out']
				tmp_t1=bin_llcl.bin_llcl(ll*theory_in/bls,b)
				theory=tmp_t1['llcl']
				tmp_c1=bin_llcl.bin_llcl(ll*cosmic/bls,b)
				cosmic=tmp_c1['llcl']
				
			
				
			else:
				plot_l=l
				theory=np.multiply(ll/bls,theory_in)
				#cross1_array=np.multiply(ll/bls,cross1_array_in)
				#noise1_array=np.multiply(ll/bls,noise1_array_in)
				#cross2_array=np.multiply(ll/bls,cross2_array_in)
				#noise2_array=np.multiply(ll/bls,noise2_array_in)
				cross*=ll/bls
				cosmic*=ll/bls
				dcross*=ll/bls
				#cosmic2*=ll/bls
				#delta1*=ll/bls
				#delta2*=ll/bls
				#ipdb.set_trace()
			bad=np.where(plot_l < dl_eff)
			#noise1=np.mean(noise1_array,axis=0)
			#noise2=np.mean(noise2_array,axis=0)
        		#theory_array = np.add(theory1_array,theory2_array)
        		#theory=np.mean(theory_array,axis=0)
        		#dtheory=np.std(theory_array,axis=0,ddof=1)
        		#cross_array = np.add(np.subtract(cross1_array,noise1_array),np.subtract(cross2_array,noise2_array))
        		#delta_array=np.sqrt(delta1**2+delta2**2)
			##cross_array=np.add(cross1_array,cross2_array)
			#ipdb.set_trace()
        		#cross=np.average(cross_array,weights=1./delta_array**2,axis=0)
        		#cosmic=np.sqrt(cosmic1**2+cosmic2**2)
			#theory1=np.mean(theory1_array,axis=0)
			#dtheory1=np.std(theory1_array,axis=0,ddof=1)
			#cross1=np.mean(cross1_array,axis=0)
			#dcross1=np.std(np.subtract(cross1_array,noise1),axis=0,ddof=1)
			#dcross=np.average(delta_array,weights=1./delta_array**2,axis=0)
			#ipdb.set_trace()
			plot_binned.plotBinned((cross)*1e12,dcross*1e12,plot_l,b,'Cross_FR_cut_{0:0>2d}'.format(cut), title='Faraday Rotation Correlator',theory=theory*1e12,dtheory=cosmic*1e12)

			#theory2=np.mean(theory2_array,axis=0)
			#dtheory2=np.std(theory2_array,axis=0,ddof=1)
			#cross2=np.mean(cross2_array,axis=0)
			##delta2=np.mean(delta2_array,axis=0)
			#dcross2=np.std(np.subtract(cross2_array,noise2),axis=0,ddof=1)
			##ipdb.set_trace()
			#plot_binned.plotBinned((cross2-noise2)*1e12,dcross2*1e12,plot_l,b,'Cross_43x95_FR_UxaQ', title='Cross 43x95 FR UxaQ',theory=theory2*1e12,dtheory=dtheory2*1e12,delta=delta2*1e12,cosmic=cosmic2*1e12)
			#ipdb.set_trace()
    
			if b == 25 :
				a_scales=np.linspace(-10,10,1001)
				chi_array=[]
				for a in a_scales:
					chi_array.append(np.sum( (cross - a*theory)**2/(dcross)**2))
				ind = np.argmin(chi_array)
			#likelihood=np.exp(np.multiply(-1./2.,chi_array))/np.sqrt(2*np.pi)
				likelihood=np.exp(np.multiply(-1./2.,chi_array))/np.sum(np.exp(np.multiply(-1./2.,chi_array))*.05)

				Sig=np.sum(cross/(dcross**2))/np.sum(1./dcross**2)
				#Noise=np.std(np.sum(cross_array/dcross**2,axis=1)/np.sum(1./dcross**2))	\
				Noise=np.sqrt(1./np.sum(1./dcross**2))
				Sig1=np.sum(cross*(theory/dcross)**2)/np.sum((theory/dcross)**2)
				Noise1=np.sum(dcross*(theory/dcross)**2)/np.sum((theory/dcross)**2)
				SNR=Sig/Noise
				SNR1=Sig1/Noise1
				
				#Sig2=np.sum(cross/(dcross**2))/np.sum(1./dcross**2)
				#Noise2=np.sqrt(1./np.sum(1./dcross**2))
				#Sig3=np.sum(cross*(theory/dcross)**2)/np.sum((theory/dcross)**2)
				#Noise3=np.sqrt(np.sum(theory**2)/np.sum(theory**2/dcross**2))
				#SNR2=Sig2/Noise2
				#SNR3=Sig3/Noise3
				
				#ipdb.set_trace()
				fig,ax1=plt.subplots(1,1)

				ax1.plot(a_scales,likelihood,'k.')
				ax1.set_title('Faraday Rotation Correlator')
				ax1.set_xlabel('Likelihood scalar')
				ax1.set_ylabel('Likelihood of Correlation')
				fig.savefig('FR_Correlation_Likelihood.png',format='png')
				fig.savefig('FR_Correlation_Likelihood.eps',format='eps')
				#ipdb.set_trace()
				f=open('Maximum_likelihood_{0:0>2d}.txt'.format(cut),'w')
				f.write('Maximum Likelihood: {0:2.5f}%  for scale factor {1:.2f} \n'.format(float(likelihood[ind]*100),float(a_scales[ind])))
				f.write('Probability of scale factor =1: {0:2.5f}% \n \n'.format(float(likelihood[np.where(a_scales ==1)])*100))
				f.write('Detection Levels using Standard Deviation \n')
				f.write('Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n'.format(SNR,Sig, Noise))
				f.write('Weighted Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n \n'.format(SNR1,Sig1,Noise))
				#f.write('Detection using Theoretical Noise \n')
				#f.write('Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n'.format(SNR2,Sig2, Noise2))
				#f.write('Weighted Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n'.format(SNR3,Sig3,Noise3))
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

	subprocess.call('mv Maximum_likelihood_cut_{:2>02d}.txt' .format(cut), shell=True)
	subprocess.call('mv *01.png bin_01/', shell=True)
	subprocess.call('mv *05.png bin_05/', shell=True)
	subprocess.call('mv *10.png bin_10/', shell=True)
	subprocess.call('mv *20.png bin_20/', shell=True)
	subprocess.call('mv *25.png bin_25/', shell=True)
	subprocess.call('mv *50.png bin_50/', shell=True)
	subprocess.call('mv *.eps eps/'.format(cut), shell=True)
if __name__=='__main__':
	main()
