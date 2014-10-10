import numpy as np
import healpy as hp
from astropy.io import fits
import ipdb
import scalar_mixing_matrix as Mll
def faraday_correlate_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands):
	print "Computer Cross Correlations for Bands "+str(bands)

	temperature_file='/data/Planck/COM_CompMap_CMB-smica_2048.fits'
	planck_T=hp.read_map(temperature_file)
	planck_T*=1e-6

	hdu_i=fits.open(i_file)
	hdu_j=fits.open(j_file)
	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi')
	delta_alpha_radio=hp.read_map(alpha_file,hdu='uncertainty/phi')
	iqu_band_i=hdu_i['stokes iqu'].data
	iqu_band_j=hdu_j['stokes iqu'].data
	sigma_i=hdu_i['Q/U UNCERTAINTIES'].data
	sigma_j=hdu_j['Q/U UNCERTAINTIES'].data
	mask_hdu=fits.open('/data/wmap/wmap_polarization_analysis_mask_r9_9yr_v5.fits')
	mask=mask_hdu[1].data.field(0)
	mask=hp.reorder(mask,n2r=1)
	#mask=hdu_i['mask'].data
	mask=hp.ud_grade(mask,nside_out=128)
	pix=np.where(mask != 0)
	pix=np.array(pix).reshape(len(pix[0]))
	pix_bad=np.where(mask == 0)
	field_pixels=hdu_i['FIELD PIXELS'].data
	#iqu_band_i[1]+=sigma_i[0]/1.
	#iqu_band_i[2]+=sigma_i[1]/1.
	#iqu_band_j[1]+=sigma_j[0]/1.
	#iqu_band_j[2]+=sigma_j[1]/1.
	
	iqu_band_i=hp.ud_grade(iqu_band_i,nside_out=128,order_in='ring')
	iqu_band_j=hp.ud_grade(iqu_band_j,nside_out=128,order_in='ring')
	planck_T=hp.ud_grade(planck_T,nside_out=128,order_in='ring')
		
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.pi/180.,lmax=383)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.pi/180.,lmax=383)
	planck_T=hp.smoothing(planck_T,fwhm=np.pi/180.,lmax=383)
	alpha_radio=hp.smoothing(alpha_radio,fwhm=np.pi/180.,lmax=383)
	
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

			#wl=hp.anafast((~mask_bool1).astype(int),map2=(~mask_bool2).astype(int))
			#l=np.arange(len(wl))
			#mixing=[[Mll.Mll(wl,l1,l2) for l1 in l] for l2 in l]
			#Bl=np.exp(-l(l+1)*(60**2+11.7**2)*((np.pi/(180.*60.))/(2*np.sqrt(2*np.log(2))))**2/2)
			#K=[m*Bl**2 for m in mixing]
			#K_inv=K.getI
			tmp_cl1=hp.anafast(DQm,map2=aUm)
			tmp_cl2=hp.anafast(DUm,map2=aQm)			
			l=np.arange(len(tmp_cl1))
			Bl_60=np.exp(-l*(l+1)*((60.0*np.pi/(180.*60.)/(np.sqrt(8.0*np.log(2.))))**2)/2.)
			Bl_11=np.exp(-l*(l+1)*((11.7*np.pi/(180.*60.)/(np.sqrt(8.0*np.log(2.))))**2)/2.)
			Bl_27=np.exp(-l*(l+1)*((27.3*np.pi/(180.*60.)/(np.sqrt(8.0*np.log(2.))))**2)/2.)
			Bl_factor=Bl_60**2*Bl_11*Bl_27
			cross1_array.append(tmp_cl1/Bl_factor)
			cross2_array.append(tmp_cl2/Bl_factor)
			cross_tmp=hp.anafast(TEm,pol=1,nspec=4)
			cross3_array.append(cross_tmp[-1]/Bl_factor)
	
	cross1=np.mean(cross1_array,axis=0)	##Average over all Cross Spectra
	cross2=np.mean(cross2_array,axis=0)	##Average over all Cross Spectra
	cross3=np.mean(cross1_array,axis=0)	##Average over all Cross Spectra
	dcross1=np.std(cross1_array,axis=0)	##Average over all Cross Spectra
	dcross2=np.std(cross2_array,axis=0)	##Average over all Cross Spectra
	hp.write_cl('cl_'+bands+'_FR_QxaU.fits',cross1)
	hp.write_cl('cl_'+bands+'_FR_UxaQ.fits',cross2)
	hp.write_cl('cl_'+bands+'_FR_TE_cmb.fits',cross3)
	hp.write_cl('Nl_'+bands+'_FR_QxaU.fits',dcross1)
	hp.write_cl('Nl_'+bands+'_FR_UxaQ.fits',dcross2)

	Delta_Q[pix_bad]=hp.UNSEEN
	Delta_U[pix_bad]=hp.UNSEEN
	alpha_u[pix_bad]=hp.UNSEEN
	alpha_q[pix_bad]=hp.UNSEEN
	planck_T[pix_bad]=hp.UNSEEN
	prim=fits.PrimaryHDU()
	pix_col=fits.Column(name='PIXEL',format='1J',array=pix)
	col_dq=fits.Column(name='SIGNAL',format='1E',unit='K/m^2',array=Delta_Q[pix])
	col_du=fits.Column(name='SIGNAL',format='1E',unit='K/m^2',array=Delta_U[pix])
	col_aq=fits.Column(name='SIGNAL',format='1E',unit='K*rad/m^2',array=alpha_q[pix])
	col_au=fits.Column(name='SIGNAL',format='1E',unit='K*rad/m^2',array=alpha_u[pix])
	col_te1=fits.Column(name='SIGNAL',format='1E',unit='K*rad/m^2',array=TE_map[0][pix])
	col_te2=fits.Column(name='STOKES Q',format='1E',unit='K/m^2',array=TE_map[1][pix])
	col_te3=fits.Column(name='STOKES U',format='1E',unit='K/m^2',array=TE_map[2][pix])	
	data=[col_dq,col_du,col_aq,col_au]
	names=['Q','aU','U','aQ','TE']
	for i in xrange(len(data)):
		cols=fits.ColDefs([pix_col,data[i]])
		tbhdu=fits.BinTableHDU.from_columns(cols)
		tbhdu.header['PIXTYPE']=("HEALPIX","HEALPIX pixelisation")
                tbhdu.header['ORDERING']=("RING","Pixel order scheme, either RING or NESTED")
                tbhdu.header["COORDSYS"]=('G','Pixelization coordinate system')
                tbhdu.header["NSIDE"]=(128,'Healpix Resolution paramter')
                tbhdu.header['OBJECT']=('PARTIAL','Sky coverage, either FULLSKY or PARTIAL')
                tbhdu.header['OBS_NPIX']=(len(pix),'Number of pixels observed')
                tbhdu.header['INDXSCHM']=('EXPLICIT','indexing : IMPLICIT of EXPLICIT')
                tblist=fits.HDUList([prim,tbhdu])
		tblist.writeto('quiet_cross_'+names[i]+'.fits',clobber=True)
	
	return (cross1,cross2,cross3,dcross1,dcross2)


if __name__=='__main__':
	map_prefix='/home/matt/quiet/quiet_maps/'
	i_file=map_prefix+'quiet_simulated_43.1'
	j_file=map_prefix+'quiet_simulated_94.5'
	alpha_file='/data/wmap/faraday_MW_realdata.fits'
	bands=[43.1,94.5]
	names=['43','95']
	wl=np.array([299792458./(band*1e9) for band in bands])
	
	cross1,cross2,cross3,dcross1,dcross2=faraday_correlate_quiet(i_file+'.fits',j_file+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1])
