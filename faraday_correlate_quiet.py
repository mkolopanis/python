import numpy as np
import healpy as hp
from astropy.io import fits
import ipdb
def faraday_correlate_quiet(i_file,j_file,wl_i,wl_j,alpha_file,bands,field):
	print "Computer Cross Correlations for Bands "+str(bands)+" cmb field"+str(field)

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
	mask=hdu_i['mask'].data
	mask=hp.ud_grade(mask,nside_out=128)
	pix=np.where(mask != 0)
	pix=np.array(pix).reshape(len(pix[0]))
	pix_bad=np.where(mask == 0)
	iqu_band_i[1:]+=sigma_i
	iqu_band_j[1:]+=sigma_j
	
	iqu_band_i=hp.ud_grade(iqu_band_i,nside_out=128,order_in='ring')
	iqu_band_j=hp.ud_grade(iqu_band_j,nside_out=128,order_in='ring')
	planck_T=hp.ud_grade(planck_T,nside_out=128,order_in='ring')
	
	iqu_band_i=hp.smoothing(iqu_band_i,pol=1,fwhm=np.pi/180.,lmax=383)
	iqu_band_j=hp.smoothing(iqu_band_j,pol=1,fwhm=np.pi/180.,lmax=383)
	planck_T=hp.smoothing(planck_T,pol=1,fwhm=np.pi/180.,lmax=383)
	alpha_radio=hp.smoothing(alpha_radio,pol=1,fwhm=np.pi/180.,lmax=383)

	const=2.*(wl_i**2-wl_j**2)	

	Delta_Q=(iqu_band_i[1]-iqu_band_j[1])/const
	Delta_U=(iqu_band_i[2]-iqu_band_j[2])/const 
	alpha_u=alpha_radio*iqu_band_i[1] 
	alpha_q=-alpha_radio*iqu_band_i[2]
	
	DQm=hp.ma(Delta_Q)
	DUm=hp.ma(Delta_U)
	aQm=hp.ma(alpha_q)
	aUm=hp.ma(alpha_u)
	mask_bool=np.repeat(True,len(Delta_Q))
	mask_bool[np.nonzero(mask)]=False

	DQm.mask=mask_bool
	DUm.mask=mask_bool
	aQm.mask=mask_bool
	aUm.mask=mask_bool

	Delta_Q[pix_bad]=hp.UNSEEN
	Delta_U[pix_bad]=hp.UNSEEN
	alpha_u[pix_bad]=hp.UNSEEN
	alpha_q[pix_bad]=hp.UNSEEN
	planck_T[pix_bad]=hp.UNSEEN
	TE_map=np.array([planck_T*alpha_radio,Delta_Q,Delta_U])
	TEm=hp.ma(TE_map)
	TEm[0].mask=mask_bool
	TEm[1].mask=mask_bool
	TEm[2].mask=mask_bool

	cross1=hp.anafast(DQm,map2=aUm)
	cross2=hp.anafast(DUm,map2=aQm)
	cross_tmp=hp.anafast(TEm,pol=1,nspec=4)
	cross3=cross_tmp[-1]

	hp.write_cl('cl_'+bands+'_FR_QxaU_cmb'+str(field)+'.fits',cross1)
	hp.write_cl('cl_'+bands+'_FR_UxaQ_cmb'+str(field)+'.fits',cross2)
	hp.write_cl('cl_'+bands+'_FR_TE_cmb'+str(field)+'.fits',cross3)

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
		tblist.writeto('quiet_cross_'+names[i]+'_cmb'+str(field)+'.fits')
	
	return (cross1,cross2,cross3)


if __name__=='__main__':
	map_prefix='/home/matt/quiet/quiet_maps/'
	i_file=map_prefix+'quiet_simulated_43.1_cmb'
	j_file=map_prefix+'quiet_simulated_94.5_cmb'
	alpha_file='/data/wmap/faraday_MW_realdata.fits'
	bands=[43.1,94.5]
	names=['43','95']
	wl=np.array([299792458./(band*1e9) for band in bands])
	
	for p in xrange(4):
		cross1,cross2,cross3=faraday_correlate_quiet(i_file+str(p+1)+'.fits',j_file+str(p+1)+'.fits',wl[0],wl[1],alpha_file,names[0]+'x'+names[1],p+1)
