import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from scipy import linalg as lin
from  IPython import embed
##Global smoothing size parameter##

smoothing_scale=40.0
nside_out=128
nside_in=1024
###map_bl=hp.gauss_beam(np.sqrt(hp.nside2pixarea(128)+hp.nside2pixarea(1024)),383)

pix_area= np.sqrt(hp.nside2pixarea(1024))*60*180./np.pi
q_fwhm=[27.3,11.7]
noise_const=np.array([36./pix_area for f in q_fwhm])*1e-6
mask_array=[ '/data/wmap/wmap_polarization_analysis_mask_r9_9yr_v5.fits','/data/Planck/COM_CMB_IQU-common-field-MaskPol_1024_R2.00.fits']
mask_name=['wmap','cmask']
synchrotron_file='/data/Planck/COM_CompMap_SynchrotronPol-commander_0256_R2.00.fits'
dust_file='/data/Planck/COM_CompMap_DustPol-commander_1024_R2.00.fits'

def likelihood(cross,dcross,theory,name,title,nruns=1):
	
	Sig=np.sum(cross/(dcross**2))/np.sum(1./dcross**2)
	#Noise=np.std(np.sum(cross_array/dcross**2,axis=1)/np.sum(1./dcross**2))	\
	Noise=np.sqrt(1./np.sum(1./dcross**2))
	Sig1=np.sum(cross*(theory/dcross)**2)/np.sum((theory/dcross)**2)
	Noise1=np.sqrt(np.sum(dcross**2*(theory/dcross)**2)/np.sum((theory/dcross)**2))
	SNR=Sig/Noise
	SNR1=Sig1/Noise1
	
        dcross=np.copy(dcross)/np.sqrt(nruns)

	a_scales=np.linspace(-1000,1000,100000)
	chi_array=[]
	for a in a_scales:
		chi_array.append(np.exp(-.5*np.sum( (cross - a*theory)**2/(dcross)**2)))
	chi_array /= np.max(chi_array)
	chi_sum = np.cumsum(chi_array)
	chi_sum /= chi_sum[-1]
	
	mean = a_scales[np.argmax(chi_array)]
	fig,ax1=plt.subplots(1,1)
	try:
		s1lo,s1hi = a_scales[chi_sum<0.1586][-1],a_scales[chi_sum>1-0.1586][0]
		s2lo,s2hi = a_scales[chi_sum<0.0227][-1],a_scales[chi_sum>1-0.0227][0]
	
	        ax1.vlines(s1lo,0,1,linewidth=2,color='blue')
	        ax1.vlines(s1hi,0,1,linewidth=2,color='blue')
	        ax1.vlines(s2lo,0,1,linewidth=2,color='orange')
	        ax1.vlines(s2hi,0,1,linewidth=2,color='orange')
	        if np.max(abs(np.array([s2hi,s2lo]))) > 5000:
                    ax1.set_xlim([-10000,10000])
	        elif np.max(abs(np.array([s2hi,s2lo]))) > 1000:
	        	ax1.set_xlim([-5000,5000])
	        elif np.max(abs(np.array([s2hi,s2lo]))) > 500:
	        	ax1.set_xlim([-1000,1000])
	        elif np.max(abs(np.array([s2hi,s2lo]))) > 200:
	        	ax1.set_xlim([-500,500])
	        elif np.max(abs(np.array([s2hi,s2lo]))) > 100:
	        	ax1.set_xlim([-200,200])
	        elif np.max(abs(np.array([s2hi,s2lo]))) > 50:
	        	ax1.set_xlim([-100,100])
	        elif np.max(abs(np.array([s2hi,s2lo]))) > 20:
	        	ax1.set_xlim([-50,50])
	        elif np.max(abs(np.array([s2hi,s2lo]))) > 10:
	        	ax1.set_xlim([-20,20])
	        elif np.max(abs(np.array([s2hi,s2lo]))) > 5:
	        	ax1.set_xlim([-10,10])
                elif np.max(abs(np.array([s2hi,s2lo]))) > 2:
	        	ax1.set_xlim([-5,5])
                elif np.min(np.array([s2hi,s2lo])) > 0:
                        ax1.set_xlim([s2lo-1,s2hi+1])
	        f=open('Maximum_likelihood_simulation_'+name+'_'+title+'.txt','w')
	        f.write('Maximum Likelihood: {0:2.5f}%  for scale factor {1:.2f} \n'.format(float(chi_array[np.argmax(chi_array)]*100),mean))
	        f.write('Posterior: Mean,\tsigma,\t(1siglo,1sighi),\t(2sighlo,2sighi)\n')
                f.write('Posterior: {0:.3f},\t{1:.3f} ,\t({2:.3f},{3:.3f})\t({4:.3f},{5:.3f})\n '.format(mean,np.mean([s1hi-mean,mean-s1lo]) ,s1lo,s1hi, s2lo,s2hi))
	        f.write('Posterior SNR:\t {0:.3f}'.format(1./(np.mean( [s1hi-mean,mean-s1lo] )*np.sqrt(nruns)) ) )
	        f.write('\n\n')
	        f.write('Detection Levels using Standard Deviation \n')
	        f.write('Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n'.format(SNR,Sig, Noise))
	        f.write('Weighted Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n \n'.format(SNR1,Sig1,Noise1))
	        f.close()
	except:

		print('Scale exceeded for possterior. \n Plotting anyway')

	ax1.plot(a_scales,chi_array,'k',linewidth=2)
        ax1.set_xlim([s2lo-1,s2hi+1])
	ax1.set_title('Faraday Rotatior Posterior')
	ax1.set_xlabel('Likelihood scalar')
	ax1.set_ylabel('Likelihood of Correlation')
		
	fig.savefig('FR_simulation_likelihood_'+name+'_'+title+'.png',format='png')
	fig.savefig('FR_simulation_likelihood_'+name+'_'+title+'.eps',format='eps')

def faraday_correlate_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands,beam=False):
	print "Computing Cross Correlations for Bands "+str(bands)

	q_fwhm=[27.3,11.7]
	#noise_const=np.array([36./f for f in q_fwhm])*1e-6
	npix=hp.nside2npix(1024)
	sigma_q_i,sigma_u_i=[noise_const[0]*np.random.normal(0,1,npix),noise_const[0]*np.random.normal(0,1,npix)]
	sigma_q_j,sigma_u_j=[noise_const[1]*np.random.normal(0,1,npix),noise_const[1]*np.random.normal(0,1,npix)]

	sigma_q_i=hp.smoothing(sigma_q_i,fwhm=q_fwhm[0]*np.pi/(180.*60.),verbose=False)	
	sigma_u_i=hp.smoothing(sigma_u_i,fwhm=q_fwhm[0]*np.pi/(180.*60.),verbose=False)	
	sigma_q_j=hp.smoothing(sigma_q_j,fwhm=q_fwhm[1]*np.pi/(180.*60.),verbose=False)	
	sigma_u_j=hp.smoothing(sigma_u_j,fwhm=q_fwhm[1]*np.pi/(180.*60.),verbose=False)	

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
	
	iqu_band_i[1]+=np.copy(sigma_q_i)
	iqu_band_i[2]+=np.copy(sigma_u_i)
	iqu_band_j[1]+=np.copy(sigma_q_j)
	iqu_band_j[2]+=np.copy(sigma_u_j)
	hdu_i.close()
	hdu_j.close()
	
	sigma_q_i=hp.smoothing(sigma_q_i,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_u_i=hp.smoothing(sigma_u_i,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_q_j=hp.smoothing(sigma_q_j,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_u_j=hp.smoothing(sigma_u_j,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),verbose=False)	

	sigma_q_i=hp.ud_grade(sigma_q_i,128)
	sigma_u_i=hp.ud_grade(sigma_u_i,128)
	sigma_q_j=hp.ud_grade(sigma_q_j,128)
	sigma_u_j=hp.ud_grade(sigma_u_j,128)
		
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),verbose=False)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),verbose=False)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	iqu_band_i=hp.ud_grade(iqu_band_i,nside_out=128,order_in='ring')
	iqu_band_j=hp.ud_grade(iqu_band_j,nside_out=128,order_in='ring')
	
        dust_t_file='/data/Planck/COM_CompMap_dust-commander_0256_R2.00.fits'
        dust_b_file='/data/Planck/COM_CompMap_ThermalDust-commander_2048_R2.00.fits'
        
        ##Dust intensity scaling factor
        hdu_dust_t=fits.open(dust_t_file)
        dust_t=hdu_dust_t[1].data.field('TEMP_ML')
        hdu_dust_t.close()
        
        dust_t=hp.reorder(dust_t,n2r=1)
        dust_t=hp.ud_grade(dust_t,nside_out)
        
        hdu_dust_b=fits.open(dust_b_file)
        dust_beta=hdu_dust_b[1].data.field('BETA_ML_FULL')
        hdu_dust_b.close
        
        dust_beta=hp.reorder(dust_beta,n2r=1)	
        dust_beta=hp.ud_grade(dust_beta,nside_out)
        
        gamma_dust=6.626e-34/(1.38e-23*dust_t)
        freqs=[43.1,94.5]
        krj_to_kcmb=np.ones_like(freqs)
        dust_factor=np.array([krj_to_kcmb[i]*1e-6*(np.exp(gamma_dust*353e9)-1.)/(np.exp(gamma_dust*x*1e9)-1.)* (x/353.)**(1.+dust_beta) for i,x in enumerate(freqs)])
        sync_factor=krj_to_kcmb*np.array([1e-6*(30./x)**2 for x in freqs])

	hdu_sync=fits.open(synchrotron_file)
	sync_q=hdu_sync[1].data.field(0)
	sync_u=hdu_sync[1].data.field(1)
	
	sync_q=hp.reorder(sync_q,n2r=1)
	sync_q=hp.ud_grade(sync_q,nside_out=128)

	
	sync_u=hp.reorder(sync_u,n2r=1)
	sync_u=hp.ud_grade(sync_u,nside_out=128)
	hdu_sync.close()
	
	hdu_dust=fits.open(dust_file)
	dust_q=hdu_dust[1].data.field(0)
	dust_u=hdu_dust[1].data.field(1)
	hdu_dust.close()
	
	dust_q=hp.reorder(dust_q,n2r=1)
	dust_q=hp.smoothing(dust_q,fwhm=np.sqrt(smoothing_scale**2-10.0**2)*np.pi/(180.*60.),verbose=False)
	dust_q=hp.ud_grade(dust_q,128)
	
	dust_u=hp.reorder(dust_u,n2r=1)
	dust_u=hp.smoothing(dust_u,fwhm=np.sqrt(smoothing_scale**2-10.0**2)*np.pi/(180.*60.),verbose=False)
	dust_u=hp.ud_grade(dust_u,128)
    
        gamma_sync_q=[]
        gamma_sync_u=[]
        delta_dust_q=[]
        delta_dust_u=[]
        iqu_array=[iqu_band_i,iqu_band_j]
        iqu_array_fr=[]
        for cnt,iqu in enumerate(iqu_array):
                #redefine temps to make math easier
                mask_bool1=np.repeat(True,len(dust_q))
                pix_cmb1=field_pixels.field(0)	
                pix_cmb1=pix_cmb1[np.nonzero(pix_cmb1)]	##Take Pixels From Field 1
                tmp=np.zeros(hp.nside2npix(1024))
                tmp[pix_cmb1]+=1
                tmp=hp.ud_grade(tmp,128)
                mask_bool1[np.nonzero(tmp)]=False

                dq=hp.ma(dust_factor[cnt]*dust_q)
                du=hp.ma(dust_factor[cnt]*dust_u)
                sq=hp.ma(sync_factor[cnt]*sync_q)
                su=hp.ma(sync_factor[cnt]*sync_u)

                dq.mask=mask_bool1
                du.mask=mask_bool1
                sq.mask=mask_bool1
                su.mask=mask_bool1
                #normalization factors for scaling 
                gamma_sync_q= np.sum(iqu[1]*sq)/np.sum(sq**2)- np.sum(dq*sq)/np.sum(sq**2)*( (np.sum(sq**2)*np.sum(iqu[1]*dq)-np.sum(iqu[1]*sq)*np.sum(sq*dq))/(np.sum(dq**2)*np.sum(sq**2)-np.sum(sq*dq)**2) )
                delta_dust_q= (np.sum(sq**2)*np.sum(iqu[1]*dq)-np.sum(iqu[1]*sq)*np.sum(sq*dq))/( np.sum(dq**2)*np.sum(sq**2)-np.sum(sq*dq)**2)

                gamma_sync_u= np.sum(iqu[2]*su)/np.sum(su**2)- np.sum(du*su)/np.sum(su**2)*( (np.sum(su**2)*np.sum(iqu[2]*du)-np.sum(iqu[2]*su)*np.sum(su*du))/(np.sum(du**2)*np.sum(su**2)-np.sum(su*du)**2) )
                delta_dust_u= (np.sum(su**2)*np.sum(iqu[2]*du)-np.sum(iqu[2]*su)*np.sum(su*du))/( np.sum(du**2)*np.sum(su**2)-np.sum(su*du)**2)

                iqu_array_fr.append(np.array([iqu[0],iqu[1]-gamma_sync_q*sq-delta_dust_q*dq,iqu[2]-gamma_sync_u*su-delta_dust_u*du]))

        iqu_band_i=np.copy(iqu_array_fr[0])
        iqu_band_j=np.copy(iqu_array_fr[1])
	#iqu_band_i[1]-= gamma_sync_q[0]*sync_q + delta_dust_q[0]*dust_q
	#iqu_band_i[2]-= gamma_sync_u[0]*sync_u + delta_dust_u[0]*dust_u
	#iqu_band_j[1]-= gamma_sync_q[1]*sync_q + delta_dust_q[1]*dust_q
	#iqu_band_j[2]-= gamma_sync_u[1]*sync_u + delta_dust_u[1]*dust_u

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
	Bl_factor=np.repeat(1.,3*128)
	l=np.arange(3*128)
	#ipdb.set_trace()
	if beam:
		Bl_factor=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),383)
	pix_area=hp.nside2pixarea(128)
	pix_area=(smoothing_scale*np.pi/(180.*60))**2
	#ipdb.set_trace()
	for field1 in xrange(1):
		mask_bool1=np.repeat(True,len(Delta_Q))
		pix_cmb1=field_pixels.field(field1)	
		pix_cmb1=pix_cmb1[np.nonzero(pix_cmb1)]	##Take Pixels From Field 1
		tmp=np.zeros(hp.nside2npix(1024))
		tmp[pix_cmb1]+=1
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
		#ipdb.set_trace()
		cross1_array.append(hp.anafast(DQm,map2=aUm)/Bl_factor**2)
		cross2_array.append(hp.anafast(DUm,map2=aQm)/Bl_factor**2)

                ##calculate theoretical variance for correlations
		Ndq_array.append((sqi**2+sqj**2).sum()*(pix_area/const)**2/(4.*np.pi))
		Ndu_array.append((sui**2+suj**2).sum()*(pix_area/const)**2/(4.*np.pi))
		Nau_array.append(((salpha*um+alpham*suj+salpha*suj)**2).sum()*pix_area**2/(4.*np.pi))
		Naq_array.append(((salpha*qm+alpham*sqj+salpha*sqj)**2).sum()*pix_area**2/(4.*np.pi))
		#Ndq_array.append(hp.anafast(sqi,map2=sqj))
		#Ndu_array.append(hp.anafast(sui,map2=suj))
		#Nau_array.append(hp.anafast(um,map2=salpha)+hp.anafast(suj,map2=alpham)+hp.anafast(suj,map2=salpha))
		#Naq_array.append(hp.anafast(qm,map2=salpha)+hp.anafast(suj,map2=alpham)+hp.anafast(sqj,map2=salpha))
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

def faraday_noise_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands,beam=False):
	print "Computing Cross Correlations for Bands "+str(bands)


	q_fwhm=[27.3,11.7]
	#noise_const=np.array([36./f for f in q_fwhm])*1e-6
	npix=hp.nside2npix(1024)
	sigma_i=[noise_const[0]*np.random.normal(0,1,npix),noise_const[0]*np.random.normal(0,1,npix)]
	sigma_j=[noise_const[1]*np.random.normal(0,1,npix),noise_const[1]*np.random.normal(0,1,npix)]
	
	sigma_q_i=hp.smoothing(sigma_i[0],fwhm=q_fwhm[0]*np.pi/(180.*60.),verbose=False)	
	sigma_u_i=hp.smoothing(sigma_i[1],fwhm=q_fwhm[0]*np.pi/(180.*60.),verbose=False)	
	sigma_q_j=hp.smoothing(sigma_j[0],fwhm=q_fwhm[1]*np.pi/(180.*60.),verbose=False)	
	sigma_u_j=hp.smoothing(sigma_j[1],fwhm=q_fwhm[1]*np.pi/(180.*60.),verbose=False)	

	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	iqu_band_i=hdu_i['stokes iqu'].data
	iqu_band_j=hdu_j['stokes iqu'].data
	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi')
	field_pixels=hdu_i['FIELD PIXELS'].data
	hdu_i.close()
	hdu_j.close()
	
	iqu_band_i=np.zeros((3,npix))	
	iqu_band_j=np.zeros((3,npix))	
	iqu_band_i[1]=np.copy(sigma_i[0])
	iqu_band_i[2]=np.copy(sigma_i[1])
	iqu_band_j[1]=np.copy(sigma_j[0])
	iqu_band_j[2]=np.copy(sigma_j[1])
	
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),verbose=False)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),verbose=False)
	#alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)

	iqu_band_i=hp.ud_grade(iqu_band_i,nside_out=128,order_in='ring')
	iqu_band_j=hp.ud_grade(iqu_band_j,nside_out=128,order_in='ring')
	
	sigma_q_i=hp.smoothing(sigma_i[0],fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_u_i=hp.smoothing(sigma_i[1],fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[0])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_q_j=hp.smoothing(sigma_j[0],fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),verbose=False)	
	sigma_u_j=hp.smoothing(sigma_j[1],fwhm=np.sqrt((smoothing_scale)**2-(q_fwhm[1])**2)*np.pi/(180.*60.),verbose=False)	

	sigma_q_i=hp.ud_grade(sigma_q_i,128)
	sigma_u_i=hp.ud_grade(sigma_u_i,128)
	sigma_q_j=hp.ud_grade(sigma_q_j,128)
	sigma_u_j=hp.ud_grade(sigma_u_j,128)
	
        dust_t_file='/data/Planck/COM_CompMap_dust-commander_0256_R2.00.fits'
        dust_b_file='/data/Planck/COM_CompMap_ThermalDust-commander_2048_R2.00.fits'
           
        ##Dust intensity scaling factor
        hdu_dust_t=fits.open(dust_t_file)
        dust_t=hdu_dust_t[1].data.field('TEMP_ML')
        hdu_dust_t.close()
        
        dust_t=hp.reorder(dust_t,n2r=1)
        dust_t=hp.ud_grade(dust_t,nside_out)
        
        hdu_dust_b=fits.open(dust_b_file)
        dust_beta=hdu_dust_b[1].data.field('BETA_ML_FULL')
        hdu_dust_b.close
        
        dust_beta=hp.reorder(dust_beta,n2r=1)	
        dust_beta=hp.ud_grade(dust_beta,nside_out)
        
        gamma_dust=6.626e-34/(1.38e-23*dust_t)
        freqs=[43.1,94.5]
        krj_to_kcmb=np.ones_like(freqs)
        dust_factor=np.array([krj_to_kcmb[i]*1e-6*(np.exp(gamma_dust*353e9)-1.)/(np.exp(gamma_dust*x*1e9)-1.)* (x/353.)**(1.+dust_beta) for i,x in enumerate(freqs)])
        sync_factor=krj_to_kcmb*np.array([1e-6*(30./x)**2 for x in freqs])

	hdu_sync=fits.open(synchrotron_file)
	sync_q=hdu_sync[1].data.field(0)
	sync_u=hdu_sync[1].data.field(1)
	
	sync_q=hp.reorder(sync_q,n2r=1)
	sync_q=hp.ud_grade(sync_q,nside_out=128)

	
	sync_u=hp.reorder(sync_u,n2r=1)
	sync_u=hp.ud_grade(sync_u,nside_out=128)
	hdu_sync.close()
	
	hdu_dust=fits.open(dust_file)
	dust_q=hdu_dust[1].data.field(0)
	dust_u=hdu_dust[1].data.field(1)
	hdu_dust.close()
	
	dust_q=hp.reorder(dust_q,n2r=1)
	dust_q=hp.smoothing(dust_q,fwhm=np.sqrt(smoothing_scale**2-10.0**2)*np.pi/(180.*60.),verbose=False)
	dust_q=hp.ud_grade(dust_q,128)
	
	dust_u=hp.reorder(dust_u,n2r=1)
	dust_u=hp.smoothing(dust_u,fwhm=np.sqrt(smoothing_scale**2-10.0**2)*np.pi/(180.*60.),verbose=False)
	dust_u=hp.ud_grade(dust_u,128)
    
        gamma_sync_q=[]
        gamma_sync_u=[]
        delta_dust_q=[]
        delta_dust_u=[]
        iqu_array=[iqu_band_i,iqu_band_j]
        iqu_array_fr=[]
        for cnt,iqu in enumerate(iqu_array):
                #redefine temps to make math easier
                mask_bool1=np.repeat(True,len(dust_q))
                pix_cmb1=field_pixels.field(0)	
                pix_cmb1=pix_cmb1[np.nonzero(pix_cmb1)]	##Take Pixels From Field 1
                tmp=np.zeros(hp.nside2npix(1024))
                tmp[pix_cmb1]+=1
                tmp=hp.ud_grade(tmp,128)
                mask_bool1[np.nonzero(tmp)]=False

                dq=hp.ma(dust_factor[cnt]*dust_q)
                du=hp.ma(dust_factor[cnt]*dust_u)
                sq=hp.ma(sync_factor[cnt]*sync_q)
                su=hp.ma(sync_factor[cnt]*sync_u)

                dq.mask=mask_bool1
                du.mask=mask_bool1
                sq.mask=mask_bool1
                su.mask=mask_bool1
                #normalization factors for scaling 
                gamma_sync_q= np.sum(iqu[1]*sq)/np.sum(sq**2)- np.sum(dq*sq)/np.sum(sq**2)*( (np.sum(sq**2)*np.sum(iqu[1]*dq)-np.sum(iqu[1]*sq)*np.sum(sq*dq))/(np.sum(dq**2)*np.sum(sq**2)-np.sum(sq*dq)**2) )
                delta_dust_q= (np.sum(sq**2)*np.sum(iqu[1]*dq)-np.sum(iqu[1]*sq)*np.sum(sq*dq))/( np.sum(dq**2)*np.sum(sq**2)-np.sum(sq*dq)**2)

                gamma_sync_u= np.sum(iqu[2]*su)/np.sum(su**2)- np.sum(du*su)/np.sum(su**2)*( (np.sum(su**2)*np.sum(iqu[2]*du)-np.sum(iqu[2]*su)*np.sum(su*du))/(np.sum(du**2)*np.sum(su**2)-np.sum(su*du)**2) )
                delta_dust_u= (np.sum(su**2)*np.sum(iqu[2]*du)-np.sum(iqu[2]*su)*np.sum(su*du))/( np.sum(du**2)*np.sum(su**2)-np.sum(su*du)**2)

                iqu_array_fr.append(np.array([iqu[0],iqu[1]-gamma_sync_q*sq-delta_dust_q*dq,iqu[2]-gamma_sync_u*su-delta_dust_u*du]))

        iqu_band_i=np.copy(iqu_array_fr[0])
        iqu_band_j=np.copy(iqu_array_fr[1])
	Weight1=np.repeat(1.,len(iqu_band_i[0]))
	Weight2=np.repeat(1.,len(iqu_band_i[0]))
	
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
	
	Bl_factor=np.repeat(1.,3*128)
	if beam:
		Bl_factor=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),383)
	for field1 in xrange(1):
		mask_bool1=np.repeat(True,len(Delta_Q))
		pix_cmb1=field_pixels.field(field1)	
		pix_cmb1=pix_cmb1[np.nonzero(pix_cmb1)]	##Take Pixels From Field 1
		tmp=np.zeros(hp.nside2npix(1024))
		tmp[pix_cmb1]=+1
		tmp=hp.ud_grade(tmp,128)
		mask_bool1[np.nonzero(tmp)]=False
		#mask_bool1[np.where(np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)<.2e-6)]=True
		
		DQm.mask=mask_bool1
		DUm.mask=mask_bool1
		aQm.mask=mask_bool1
		aUm.mask=mask_bool1

		
		cross1_array.append(hp.anafast(DQm,map2=aUm)/Bl_factor**2)
		cross2_array.append(hp.anafast(DUm,map2=aQm)/Bl_factor**2)
	
	#cross1=np.mean(cross1_array,axis=0)	##Average over all Cross Spectra
	#cross2=np.mean(cross2_array,axis=0)	##Average over all Cross Spectra
	cross1=cross1_array[0]
	cross2=cross2_array[0]
	hp.write_cl('cl_'+bands+'_FR_noise_QxaU.fits',cross1)
	hp.write_cl('cl_'+bands+'_FR_noise_UxaQ.fits',cross2)
	return (cross1,cross2)

def faraday_theory_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands_name,beam=False):
	print "Computing Cross Correlations for Bands "+str(bands_name)

	radio_file='/data/wmap/faraday_MW_realdata.fits'
	cl_file='/home/matt/wmap/simul_scalCls.fits'
	nside=1024
	npix=hp.nside2npix(nside)
	
	#cls=hp.read_cl(cl_file)
	#simul_cmb=hp.sphtfunc.synfast(cls,nside,fwhm=0.,new=1,pol=1);
	#
	#alpha_radio=hp.read_map(radio_file,hdu='maps/phi');
	#alpha_radio=hp.ud_grade(alpha_radio,nside_out=nside,order_in='ring',order_out='ring')
	bands=[43.1,94.5]
        q_fwhm=[27.3,11.7]
	#noise_const=np.array([36./f for f in q_fwhm])*1e-6
	npix=hp.nside2npix(128)
	sigma_i=[noise_const[0]*np.random.normal(0,1,npix),noise_const[0]*np.random.normal(0,1,npix)]
	sigma_j=[noise_const[1]*np.random.normal(0,1,npix),noise_const[1]*np.random.normal(0,1,npix)]
	npix=hp.nside2npix(1024)
	wl=np.array([299792458./(band*1e9) for band in bands])
	num_wl=len(wl)
	#t_array=np.zeros((num_wl,npix))	
	#q_array=np.zeros((num_wl,npix))
	#u_array=np.zeros((num_wl,npix))
	#for i in range(num_wl):
	#	tmp_cmb=rotate_tqu.rotate_tqu(simul_cmb,wl[i],alpha_radio);
	#	t_array[i],q_array[i],u_array[i]=hp.smoothing(tmp_cmb,fwhm=q_fwhm[i]*np.pi/(180.*60.),verbose=False)
	#iqu_band_i=np.array([t_array[0],q_array[0],u_array[0]])
	#iqu_band_j=np.array([t_array[1],q_array[1],u_array[1]])	


	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	iqu_band_i=hdu_i['no noise iqu'].data
	iqu_band_j=hdu_j['no noise iqu'].data
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

	Bl_factor=np.repeat(1.,3*128)
	if beam:
		Bl_factor=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),383)
	for field1 in xrange(1):
		mask_bool1=np.repeat(True,len(Delta_Q))
		pix_cmb1=field_pixels.field(field1)	
		pix_cmb1=pix_cmb1[np.nonzero(pix_cmb1)]	##Take Pixels From Field 1
		tmp=np.zeros(hp.nside2npix(1024))
		tmp[pix_cmb1]+=1
		tmp=hp.ud_grade(tmp,128)
		mask_bool1[np.nonzero(tmp)]=False
		#mask_bool1[np.where(np.sqrt(iqu_band_j[1]**2+iqu_band_j[2]**2)<.1e-6)]=True
		
		DQm.mask=mask_bool1
		DUm.mask=mask_bool1
		aQm.mask=mask_bool1
		aUm.mask=mask_bool1

		#ipdb.set_trace()	
		cross1_array.append(hp.anafast(DQm,map2=aUm)/Bl_factor**2)
		cross2_array.append(hp.anafast(DUm,map2=aQm)/Bl_factor**2)
	
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
#	f=open('cl_noise_FR_QxaU.json','r')
#	noise1_array_in=json.load(f)
#	f.close()	
#	f=open('cl_noise_FR_UxaQ.json','r')
#	noise2_array_in=json.load(f)
#	f.close()	
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
	
	bins=[1,5,10,20,25,50]
	N_runs=500
#	bls=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),383)**2
	#bls=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),3*nside_out-1)**2
	bls=(hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),3*nside_out-1)*hp.pixwin(nside_out)[:3*nside_out])**2
	#bls=np.repeat(1,3*nside_out)
	fsky=225.*(np.pi/180.)**2/(4*np.pi)
	l=np.arange(len(cross1_array_in[0]))
	ll=l*(l+1)/(2*np.pi)
	L=np.sqrt(fsky*4*np.pi)
	dl_eff=2*np.pi/L

        theory1_array_in=np.array(theory1_array_in)/(fsky*bls)
	theory2_array_in=np.array(theory2_array_in)/(fsky*bls)
	cross1_array_in=np.array(cross1_array_in)/(fsky*bls)
	cross2_array_in=np.array(cross2_array_in)/(fsky*bls)
	Ndq_array_in=np.array(Ndq_array_in)/(fsky)
	Ndu_array_in=np.array(Ndu_array_in)/(fsky)
	Nau_array_in=np.array(Nau_array_in)/(fsky)
	Naq_array_in=np.array(Naq_array_in)/(fsky)
	#noise1_array_in=np.array(noise1_array_in)/(fsky*bls)
	#noise2_array_in=np.array(noise2_array_in)/(fsky*bls)

	Ndq_array_in.shape += (1,)
	Ndu_array_in.shape += (1,)
	Nau_array_in.shape += (1,)
	Naq_array_in.shape += (1,)


	for b in bins:
		theory_cls=hp.read_cl('/home/matt/Planck/data/faraday/correlation/fr_theory_cl.fits')
	#	N_dq=np.mean(Ndq_array_in)
	#	N_au=np.mean(Nau_array_in)
	#	#delta1=np.sqrt(2.*abs((np.mean(cross1_array_in,axis=0)-np.mean(noise1_array_in,axis=0))**2+(np.mean(cross1_array_in,axis=0)-np.mean(noise1_array_in,axis=0))/2.*(N_dq+N_au)+N_dq*N_au/2.)/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
	#	delta1=np.sqrt(2.*((np.mean(theory1_array_in,axis=0))**2+(np.mean(theory1_array_in,axis=0))/2.*(N_dq+N_au)+N_dq*N_au/2.)/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
	#
		cosmic1=np.sqrt(2./((2.*l+1)*np.sqrt(b**2+dl_eff**2)*fsky)*np.mean(theory1_array_in,axis=0)**2)

	#	N_du=np.mean(Ndu_array_in)
	#	N_aq=np.mean(Naq_array_in)
	#	#delta2=np.sqrt(2.*abs((np.mean(cross2_array_in,axis=0)-np.mean(noise2_array_in,axis=0))**2+(np.mean(cross2_array_in,axis=0)-np.mean(noise2_array_in,axis=0))/2.*(N_dq+N_au)+N_dq*N_au/2.)/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
	#	delta2=np.sqrt(2.*((np.mean(theory2_array_in,axis=0))**2+(np.mean(theory2_array_in,axis=0))/2.*(N_du+N_aq)+N_du*N_aq/2.)/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
		cosmic2=np.sqrt(2./((2*l+1)*np.sqrt(b**2+dl_eff**2)*fsky)*np.mean(theory2_array_in,axis=0)**2)

        	theory1_array=[]
        	theory2_array=[]
        	cross1_array=[]
        	cross2_array=[]
        #	noise1_array=[]
        #	noise2_array=[]
                    	
            	Ndq_array=[]
        	Ndu_array=[]
        	Nau_array=[]
        	Naq_array=[]
        	
		plot_l=[]
		if( b != 1):
	        	tmp_t1=bin_llcl.bin_llcl(ll*theory1_array_in,b)
	        	tmp_t2=bin_llcl.bin_llcl(ll*theory2_array_in,b)
			tmp_c1=bin_llcl.bin_llcl(ll*cross1_array_in,b)
	        	tmp_c2=bin_llcl.bin_llcl(ll*cross2_array_in,b)
		#	tmp_n1=bin_llcl.bin_llcl(ll*noise1_array_in,b)
	        #	tmp_n2=bin_llcl.bin_llcl(ll*noise2_array_in,b)
	        	
			theory1_array=tmp_t1['llcl']
			theory2_array=tmp_t2['llcl']
                        theory1_array.shape += (1,)
                        theory2_array.shape += (1,)
                        theory1_array=theory1_array.T
                        theory2_array=theory2_array.T
			plot_l= tmp_t1['l_out']
			cross1_array=tmp_c1['llcl']
			cross2_array=tmp_c2['llcl']
			
		#	noise1_array=tmp_n1['llcl']
		#	noise2_array=tmp_n2['llcl']
	        	
			Ndq_array=bin_llcl.bin_llcl(ll*Ndq_array_in,b)['llcl']
			Ndu_array=bin_llcl.bin_llcl(ll*Ndu_array_in,b)['llcl']
			Naq_array=bin_llcl.bin_llcl(ll*Naq_array_in,b)['llcl']
			Nau_array=bin_llcl.bin_llcl(ll*Nau_array_in,b)['llcl']
			tmp_c1=bin_llcl.bin_llcl((ll*cosmic1)**2,b)
			#tmp_d1=bin_llcl.bin_llcl((ll*delta1)**2,b)
		
			cosmic1=np.sqrt(tmp_c1['llcl'])
			#delta1=np.sqrt(tmp_d1['llcl'])

			tmp_c2=bin_llcl.bin_llcl((ll*cosmic2)**2,b)
			#tmp_d2=bin_llcl.bin_llcl((ll*delta2)**2,b)
			cosmic2=np.sqrt(tmp_c2['llcl'])
			#delta2=np.sqrt(tmp_d2['llcl'])
			t_tmp=bin_llcl.bin_llcl(ll*theory_cls,b)
			theory_cls=t_tmp['llcl']
		else:
			plot_l=l
			theory1_array=np.multiply(ll,theory1_array_in)
			cross1_array=np.multiply(ll,cross1_array_in)
		#	noise1_array=np.multiply(ll,noise1_array_in)
			theory2_array=np.multiply(ll,theory2_array_in)
			cross2_array=np.multiply(ll,cross2_array_in)
		#	noise2_array=np.multiply(ll,noise2_array_in)
			cosmic1*=ll
			cosmic2*=ll
			#delta1*=ll
			#delta2*=ll
			Ndq_array=np.multiply(ll,Ndq_array_in)
			Ndu_array=np.multiply(ll,Ndu_array_in)
			Naq_array=np.multiply(ll,Naq_array_in)
			Nau_array=np.multiply(ll,Nau_array_in)
			theory_cls*=ll
		#ipdb.set_trace()
		bad=np.where(plot_l < 24)
		N_dq=np.mean(Ndq_array,axis=0)
		N_du=np.mean(Ndu_array,axis=0)
		N_aq=np.mean(Naq_array,axis=0)
		N_au=np.mean(Nau_array,axis=0)
		#noise1=np.mean(noise1_array,axis=0)
		#noise2=np.mean(noise2_array,axis=0)
		theory1=np.mean(theory1_array,axis=0)
		theory2=np.mean(theory1_array,axis=0)
        	theory_array = np.add(theory1_array,theory2_array)
        	theory=np.mean(theory_array,axis=0)
        	#dtheory=np.sqrt(np.var(theory1_array,ddof=1) + np.var(theory2_array,ddof=1))
        	#cross_array = np.add(np.subtract(cross1_array,noise1),np.subtract(cross2_array,noise2))
        	cross_array = np.add(cross1_array,cross2_array)
        	cross=np.mean(cross_array,axis=0)
        	#dcross=np.std(cross_array,axis=0,ddof=1)
        	dcross=np.sqrt( ( np.var(cross1_array,axis=0,ddof=1) + np.var(cross2_array,axis=0,ddof=1)))
        	cosmic=np.sqrt(cosmic1**2+cosmic2**2)
	
		delta1=np.sqrt(2./((2*plot_l+1)*fsky*np.sqrt(b**2+dl_eff**2))*(theory1**2 + theory1*(N_dq+N_au)/2. + N_dq*N_au/2.))
		delta2=np.sqrt(2./((2*plot_l+1)*fsky*np.sqrt(b**2+dl_eff**2))*(theory2**2 + theory2*(N_du+N_aq)/2. + N_du*N_aq/2.))
        	delta=np.sqrt(delta1**2+delta2**2)
		#cosmic=np.abs(theory_cls)*np.sqrt(2./((2*plot_l+1)*fsky*np.sqrt(dl_eff**2+b**2)))
		#theory1=np.mean(theory1_array,axis=0)
		#dtheory1=np.std(theory1_array,axis=0,ddof=1)
		#cross1=np.mean(cross1_array,axis=0)
		#dcross1=np.std(np.subtract(cross1_array,noise1),axis=0,ddof=1)
		#ipdb.set_trace()
                good_l=np.logical_and(plot_l <= 250,plot_l >25)
		plot_binned.plotBinned((cross)*1e12,dcross*1e12,plot_l,b,'Cross_43x95_FR', title='QUIET FR Correlator',theory=theory*1e12,delta=delta*1e12,cosmic=cosmic*1e12)

		#theory2=np.mean(theory2_array,axis=0)
		#dtheory2=np.std(theory2_array,axis=0,ddof=1)
		#cross2=np.mean(cross2_array,axis=0)
		##delta2=np.mean(delta2_array,axis=0)
		#dcross2=np.std(np.subtract(cross2_array,noise2),axis=0,ddof=1)
		##ipdb.set_trace()
		#plot_binned.plotBinned((cross2-noise2)*1e12,dcross2*1e12,plot_l,b,'Cross_43x95_FR_UxaQ', title='Cross 43x95 FR UxaQ',theory=theory2*1e12,dtheory=dtheory2*1e12,delta=delta2*1e12,cosmic=cosmic2*1e12)
		#ipdb.set_trace()
    
		if b == 25 :
                        good_l=np.logical_and(plot_l <= 250,plot_l >25)
			likelihood(cross[good_l],delta[good_l],theory[good_l],'field1','c2bfr')

		#if b == 1 :
		#	xbar= np.matrix(ll[1:]*(cross-np.mean(cross))[1:]).T
		#	vector=np.matrix(ll[1:]*cross[1:]).T
		#	mu=np.matrix(ll[1:]*theory[1:]).T
		#	fact=len(xbar)-1
		#	cov=(np.dot(xbar,xbar.T)/fact).squeeze()
		##	ipdb.set_trace()
		#	U,S,V =np.linalg.svd(cov)
		#	_cov= np.einsum('ij,j,jk', V.T,1./S,U.T)
		#	likelhd=np.exp(-np.dot(np.dot((vector-mu).T,_cov),(vector-mu))/2. )/(np.sqrt(2*np.pi*np.prod(S)))
		##	print('Likelihood of fit is #{0:.5f}'.format(likelihood[0,0]))
		#	f=open('FR_likelihood.txt','w')
		#	f.write('Likelihood of fit is #{0:.5f}'.format(likelhd[0,0]))
		#	f.close()

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
#	bls=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),383)**2
	#bls=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),3*nside_out-1)**2
	bls=(hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),3*nside_out-1)*hp.pixwin(nside_out)[:3*nside_out])**2
	N_runs=500
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
	

	#simulate_fields.main()
	ttmp1,ttmp2=faraday_theory_quiet(i_file+'.fits',j_file+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1],beam=use_beam)
	theory1_array_in.append(ttmp1)
	theory2_array_in.append(ttmp2)
	#for n in xrange(N_runs):
	for i in xrange(N_runs):	
		print(Fore.WHITE+Back.GREEN+Style.BRIGHT+'Correlation #{:03d}'.format(i+1)+Back.RESET+Fore.RESET+Style.RESET_ALL)
		tmp1,tmp2,n1,n2,n3,n4=faraday_correlate_quiet(i_file+'.fits',j_file+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1],beam=use_beam)
	#	ntmp1,ntmp2=faraday_noise_quiet(i_file+'.fits',j_file+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1],beam=use_beam)
		cross1_array_in.append(tmp1)
		cross2_array_in.append(tmp2)
		Ndq_array_in.append(n1)
		Ndu_array_in.append(n2)
		Nau_array_in.append(n3)
		Naq_array_in.append(n4)
	#	noise1_array_in.append(ntmp1)
	#	noise2_array_in.append(ntmp2)


	f=open('cl_theory_FR_QxaU.json','w')
	json.dump(np.array(theory1_array_in).tolist(),f)
	f.close()	
	f=open('cl_theory_FR_UxaQ.json','w')
	json.dump(np.array(theory2_array_in).tolist(),f)
	f.close()	
	theory1=np.mean(theory1_array_in,axis=0)
	theory2=np.mean(theory2_array_in,axis=0)
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
	#f=open('cl_noise_FR_QxaU.json','w')
	#json.dump(np.array(noise1_array_in).tolist(),f)
	#f.close()	
	#f=open('cl_noise_FR_UxaQ.json','w')
	#json.dump(np.array(noise2_array_in).tolist(),f)
	#f.close()	
	bins=[1,5,10,20,25,50]
	fsky=225.*(np.pi/180.)**2/(4*np.pi)
	l=np.arange(len(cross1_array_in[0]))
	ll=l*(l+1)/(2*np.pi)
	L=np.sqrt(fsky*4*np.pi)
	dl_eff=2*np.pi/L
	
        theory1_array_in=np.array(theory1_array_in)/(fsky*bls)
	theory2_array_in=np.array(theory2_array_in)/(fsky*bls)
	cross1_array_in=np.array(cross1_array_in)/(fsky*bls)
	cross2_array_in=np.array(cross2_array_in)/(fsky*bls)
	Ndq_array_in=np.array(Ndq_array_in)/(fsky)
	Ndu_array_in=np.array(Ndu_array_in)/(fsky)
	Nau_array_in=np.array(Nau_array_in)/(fsky)
	Naq_array_in=np.array(Naq_array_in)/(fsky)
	#noise1_array_in=np.array(noise1_array_in)/(fsky*bls)
	#noise2_array_in=np.array(noise2_array_in)/(fsky*bls)

	Ndq_array_in.shape += (1,)
	Ndu_array_in.shape += (1,)
	Nau_array_in.shape += (1,)
	Naq_array_in.shape += (1,)


	for b in bins:
		theory_cls=hp.read_cl('/home/matt/Planck/data/faraday/correlation/fr_theory_cl.fits')
	#	N_dq=np.mean(Ndq_array_in)
	#	N_au=np.mean(Nau_array_in)
	#	#delta1=np.sqrt(2.*abs((np.mean(cross1_array_in,axis=0)-np.mean(noise1_array_in,axis=0))**2+(np.mean(cross1_array_in,axis=0)-np.mean(noise1_array_in,axis=0))/2.*(N_dq+N_au)+N_dq*N_au/2.)/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
	#	delta1=np.sqrt(2.*((np.mean(theory1_array_in,axis=0))**2+(np.mean(theory1_array_in,axis=0))/2.*(N_dq+N_au)+N_dq*N_au/2.)/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
	#
		cosmic1=np.sqrt(2./((2.*l+1)*np.sqrt(b**2+dl_eff**2)*fsky)*np.mean(theory1_array_in,axis=0)**2)

	#	N_du=np.mean(Ndu_array_in)
	#	N_aq=np.mean(Naq_array_in)
	#	#delta2=np.sqrt(2.*abs((np.mean(cross2_array_in,axis=0)-np.mean(noise2_array_in,axis=0))**2+(np.mean(cross2_array_in,axis=0)-np.mean(noise2_array_in,axis=0))/2.*(N_dq+N_au)+N_dq*N_au/2.)/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
	#	delta2=np.sqrt(2.*((np.mean(theory2_array_in,axis=0))**2+(np.mean(theory2_array_in,axis=0))/2.*(N_du+N_aq)+N_du*N_aq/2.)/((2.*l+1.)*np.sqrt(b**2+dl_eff**2)*fsky))
		cosmic2=np.sqrt(2./((2*l+1)*np.sqrt(b**2+dl_eff**2)*fsky)*np.mean(theory2_array_in,axis=0)**2)

        	theory1_array=[]
        	theory2_array=[]
        	cross1_array=[]
        	cross2_array=[]
        #	noise1_array=[]
        #	noise2_array=[]
                    	
            	Ndq_array=[]
        	Ndu_array=[]
        	Nau_array=[]
        	Naq_array=[]
        	
		plot_l=[]
		if( b != 1):
	        	tmp_t1=bin_llcl.bin_llcl(ll*theory1_array_in,b)
	        	tmp_t2=bin_llcl.bin_llcl(ll*theory2_array_in,b)
			tmp_c1=bin_llcl.bin_llcl(ll*cross1_array_in,b)
	        	tmp_c2=bin_llcl.bin_llcl(ll*cross2_array_in,b)
		#	tmp_n1=bin_llcl.bin_llcl(ll*noise1_array_in,b)
	        #	tmp_n2=bin_llcl.bin_llcl(ll*noise2_array_in,b)
	        	
			theory1_array=tmp_t1['llcl']
			theory2_array=tmp_t2['llcl']
                        theory1_array.shape += (1,)
                        theory2_array.shape += (1,)
                        theory1_array=theory1_array.T
                        theory2_array=theory2_array.T
			plot_l= tmp_t1['l_out']
			cross1_array=tmp_c1['llcl']
			cross2_array=tmp_c2['llcl']
			
		#	noise1_array=tmp_n1['llcl']
		#	noise2_array=tmp_n2['llcl']
	        	
			Ndq_array=bin_llcl.bin_llcl(ll*Ndq_array_in,b)['llcl']
			Ndu_array=bin_llcl.bin_llcl(ll*Ndu_array_in,b)['llcl']
			Naq_array=bin_llcl.bin_llcl(ll*Naq_array_in,b)['llcl']
			Nau_array=bin_llcl.bin_llcl(ll*Nau_array_in,b)['llcl']
			tmp_c1=bin_llcl.bin_llcl((ll*cosmic1)**2,b)
			#tmp_d1=bin_llcl.bin_llcl((ll*delta1)**2,b)
		
			cosmic1=np.sqrt(tmp_c1['llcl'])
			#delta1=np.sqrt(tmp_d1['llcl'])

			tmp_c2=bin_llcl.bin_llcl((ll*cosmic2)**2,b)
			#tmp_d2=bin_llcl.bin_llcl((ll*delta2)**2,b)
			cosmic2=np.sqrt(tmp_c2['llcl'])
			#delta2=np.sqrt(tmp_d2['llcl'])
			t_tmp=bin_llcl.bin_llcl(ll*theory_cls,b)
			theory_cls=t_tmp['llcl']
		else:
			plot_l=l
			theory1_array=np.multiply(ll,theory1_array_in)
			cross1_array=np.multiply(ll,cross1_array_in)
		#	noise1_array=np.multiply(ll,noise1_array_in)
			theory2_array=np.multiply(ll,theory2_array_in)
			cross2_array=np.multiply(ll,cross2_array_in)
		#	noise2_array=np.multiply(ll,noise2_array_in)
			cosmic1*=ll
			cosmic2*=ll
			#delta1*=ll
			#delta2*=ll
			Ndq_array=np.multiply(ll,Ndq_array_in)
			Ndu_array=np.multiply(ll,Ndu_array_in)
			Naq_array=np.multiply(ll,Naq_array_in)
			Nau_array=np.multiply(ll,Nau_array_in)
			theory_cls*=ll
		#ipdb.set_trace()
		bad=np.where(plot_l < 24)
		N_dq=np.mean(Ndq_array,axis=0)
		N_du=np.mean(Ndu_array,axis=0)
		N_aq=np.mean(Naq_array,axis=0)
		N_au=np.mean(Nau_array,axis=0)
		#noise1=np.mean(noise1_array,axis=0)
		#noise2=np.mean(noise2_array,axis=0)
		theory1=np.mean(theory1_array,axis=0)
		theory2=np.mean(theory1_array,axis=0)
        	theory_array = np.add(theory1_array,theory2_array)
        	theory=np.mean(theory_array,axis=0)
        	#dtheory=np.sqrt(np.var(theory1_array,ddof=1) + np.var(theory2_array,ddof=1))
        	#cross_array = np.add(np.subtract(cross1_array,noise1),np.subtract(cross2_array,noise2))
        	cross_array = np.add(cross1_array,cross2_array)
        	cross=np.mean(cross_array,axis=0)
        	#dcross=np.std(cross_array,axis=0,ddof=1)
        	dcross=np.sqrt( ( np.var(cross1_array,axis=0,ddof=1) + np.var(cross2_array,axis=0,ddof=1)))
        	cosmic=np.sqrt(cosmic1**2+cosmic2**2)
	
		delta1=np.sqrt(2./((2*plot_l+1)*fsky*np.sqrt(b**2+dl_eff**2))*(theory1**2 + theory1*(N_dq+N_au)/2. + N_dq*N_au/2.))
		delta2=np.sqrt(2./((2*plot_l+1)*fsky*np.sqrt(b**2+dl_eff**2))*(theory2**2 + theory2*(N_du+N_aq)/2. + N_du*N_aq/2.))
        	delta=np.sqrt(delta1**2+delta2**2)
		#cosmic=np.abs(theory_cls)*np.sqrt(2./((2*plot_l+1)*fsky*np.sqrt(dl_eff**2+b**2)))
		#theory1=np.mean(theory1_array,axis=0)
		#dtheory1=np.std(theory1_array,axis=0,ddof=1)
		#cross1=np.mean(cross1_array,axis=0)
		#dcross1=np.std(np.subtract(cross1_array,noise1),axis=0,ddof=1)
		#ipdb.set_trace()
		plot_binned.plotBinned((cross)*1e12,dcross*1e12,plot_l,b,'Cross_43x95_FR', title='QUIET FR Correlator',theory=theory*1e12,delta=delta*1e12,cosmic=cosmic*1e12)

		#theory2=np.mean(theory2_array,axis=0)
		#dtheory2=np.std(theory2_array,axis=0,ddof=1)
		#cross2=np.mean(cross2_array,axis=0)
		##delta2=np.mean(delta2_array,axis=0)
		#dcross2=np.std(np.subtract(cross2_array,noise2),axis=0,ddof=1)
		##ipdb.set_trace()
		#plot_binned.plotBinned((cross2-noise2)*1e12,dcross2*1e12,plot_l,b,'Cross_43x95_FR_UxaQ', title='Cross 43x95 FR UxaQ',theory=theory2*1e12,dtheory=dtheory2*1e12,delta=delta2*1e12,cosmic=cosmic2*1e12)
		#ipdb.set_trace()
    
		if b == 25 :
                        good_l=np.logical_and(plot_l <= 200,plot_l >25)
			likelihood(cross[good_l],delta[good_l],theory[good_l],'field1','c2bfr')

		#if b == 1 :
		#	xbar= np.matrix(ll[1:]*(cross-np.mean(cross))[1:]).T
		#	vector=np.matrix(ll[1:]*cross[1:]).T
		#	mu=np.matrix(ll[1:]*theory[1:]).T
		#	fact=len(xbar)-1
		#	cov=(np.dot(xbar,xbar.T)/fact).squeeze()
		##	ipdb.set_trace()
		#	U,S,V =np.linalg.svd(cov)
		#	_cov= np.einsum('ij,j,jk', V.T,1./S,U.T)
		#	likelhd=np.exp(-np.dot(np.dot((vector-mu).T,_cov),(vector-mu))/2. )/(np.sqrt(2*np.pi*np.prod(S)))
		##	print('Likelihood of fit is #{0:.5f}'.format(likelihood[0,0]))
		#	f=open('FR_likelihood.txt','w')
		#	f.write('Likelihood of fit is #{0:.5f}'.format(likelhd[0,0]))
		#	f.close()

	subprocess.call('mv *01*.png bin_01/', shell=True)
	subprocess.call('mv *05*.png bin_05/', shell=True)
	subprocess.call('mv *10*.png bin_10/', shell=True)
	subprocess.call('mv *20*.png bin_20/', shell=True)
	subprocess.call('mv *25*.png bin_25/', shell=True)
	subprocess.call('mv *50*.png bin_50/', shell=True)
	subprocess.call('mv *.eps eps/', shell=True)
	
if __name__=='__main__':
	main()
