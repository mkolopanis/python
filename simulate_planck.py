import healpy as hp
import numpy as np
from astropy.io import fits
import ipdb; 
from rotate_tqu import rotate_tqu
from alpha_function import alpha_function

def main():	
	##Define Files used to make maps
	radio_file='/data/wmap/faraday_MW_realdata.fits'
	cl_file='/home/matt/wmap/simul_scalCls.fits.lens'
	output_prefix='/home/matt/Planck/data/faraday/simul_maps/'
	synchrotron_file='/data/Planck/COM_CompMap_SynchrotronPol-commander_0256_R2.00.fits'
	dust_file='/data/Planck/COM_CompMap_DustPol-commander_1024_R2.00.fits'
	gamma_dust=6.626e-34/(1.38e-23*21)
	
	##Define Parameters used to simulate Planck Fields
	bands=np.array([30.,44.,70.,100.,143.,217.,353.])
	#beam_fwhm=np.array([33.,24.,14.,10.,7.1,5.0,5.0])
	#noise_const_temp=np.array([2.0,2.7,4.7,2.5,2.2,4.8,14.7])*2.7255e-6
	#noise_const_pol=np.array([2.8,3.9,6.7,4.0,4.2,9.8,29.8])*2.7255e-6
	#beam_fwhm=np.array([33.,24.,14.,10.,7.1,5.0,5.0])
	beam_fwhm=np.array([32.29,27,13.21,9.67,7.26,4.96,4.93])
	

	pix_area_array=np.array([np.repeat(hp.nside2pixarea(1024),3),np.repeat(hp.nise2pixarea(2048),4)])
	pix_area_array=np.sqrt(pix_area_array)*60*180./np.pi
	#beam_fwhm=np.array([33.,24.,14.,10.,7.1,5.0,5.0])
	#noise_const_temp=np.array([2.0,2.7,4.7,2.5,2.2,4.8,14.7])*2.7255e-6
	#noise_const_pol=np.array([2.8,3.9,6.7,4.0,4.2,9.8,29.8])*2.7255e-6
	noise_const_temp=np.array([2.5,2.7,3.5,1.29,.555,.78,2.56])/pix_area_array*60.e-6
	noise_const_pol=np.array([3.5,4.0,5.0,1.96,1.17,1.75,7.31])/pix_area_array*60.e-6
	krj_to_kcmb=np.array([1.0217,1.0517,np.mean([1.1360,1.1405,1.1348]), np.mean([1.3058,1.3057]),np.mean([1.6735,1.6727]),np.mean([3,2203,3.2336,3.2329,3.2161]),np.mean([14.261,14.106])])*1e-6
	sync_factor=krj_to_kcmb*np.array([20.*(.408/x)**2 for x in bands])
	dust_factor=krj_to_kcmb*np.array([163.e-6*(np.exp(gamma_dust*353e9)-1)/(np.exp(gamma_dust*x*1e9)-1)* (x/353)**2.54 for x in bands])
	nside=2048
	npix=hp.nside2npix(nside)
	pix_area=hp.nside2pixarea(nside)
	
	##Reverse ordre of arrays to Simulate larger NSIDE maps first
	bands = bands[::-1]
	beam_fwhm= beam_fwhm[::-1]
	noise_const_temp = noise_const_temp[::-1]
	noise_const_pol = noise_const_pol[::-1]

	wl=np.array([299792458./(band*1e9) for band in bands])
	num_wl=len(wl)
	tqu_array=[]
	sigma_array=[]
	
	LFI=False
	LFI_IND=np.where(bands == 70)[0][0]

	cls=hp.read_cl(cl_file)
	print 'Generating Map'
	simul_cmb=hp.sphtfunc.synfast(cls,nside,fwhm=0.,new=1,pol=1);
	
	alpha_radio=hp.read_map(radio_file,hdu='maps/phi');
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=nside,order_in='ring',order_out='ring')
	bl_40=hp.gauss_beam(40.*np.pi/(180.*60.),3*nside-1)
	hdu_sync=fits.open(synchrotron_file)
	sync_q=hdu_sync[1].data.field(0)
	sync_u=hdu_sync[1].data.field(1)
	
	sync_q=hp.reorder(sync_q,n2r=1)
	tmp_alm=hp.map2alm(sync_q)
	tmp_alm=hp.almxfl(tmp_alm,1./bl_40)
	sync_q=hp.alm2map(tmp_alm,nside)
	#sync_q=hp.smoothing(sync_q,fwhm=40.*np.pi/(180.*60.),verbose=False,invert=True)
	sync_q=hp.ud_grade(sync_q,nside_out=nside)
	
	sync_u=hp.reorder(sync_u,n2r=1)
	tmp_alm=hp.map2alm(sync_u)
	tmp_alm=hp.almxfl(tmp_alm,1./bl_40)
	sync_u=hp.alm2map(tmp_alm,nside)
	#sync_u=hp.smoothing(sync_u,fwhm=40.*np.pi/(180.*60.),verbose=False,invert=True)
	sync_u=hp.ud_grade(sync_u,nside_out=nside)
	hdu_sync.close()
	

	bl_10=hp.gauss_beam(10*np.pi/(180.*60.),3*nside-1)
	hdu_dust=fits.open(dust_file)
	dust_q=hdu_dust[1].data.field(0)
	dust_u=hdu_dust[1].data.field(1)
	hdu_dust.close()
	
	dust_q=hp.reorder(dust_q,n2r=1)
	tmp_alm=hp.map2alm(dust_q)
	tmp_alm=hp.almxfl(tmp_alm,1./bl_10)
	dust_q=hp.alm2map(tmp_alm,nside)
	#dust_q=hp.smoothing(dust_q,fwhm=10.0*np.pi/(180.*60.),verbose=False,invert=True)
	dust_q=hp.ud_grade(dust_q,nside)
	
	dust_u=hp.reorder(dust_u,n2r=1)
	tmp_alm=hp.map2alm(dust_u)
	tmp_alm=hp.almxfl(tmp_alm,1./bl_10)
	dust_u=hp.alm2map(tmp_alm,nside)
	#dust_q=hp.smoothing(dust_q,fwhm=10.0*np.pi/(180.*60.),verbose=False,invert=True)
	dust_u=hp.ud_grade(dust_u,nside)

	nside=2048
	pix_area=hp.nside2pixarea(nside)
	
	prim=fits.PrimaryHDU()
	prim.header['COMMENT']="Simulated Planck Data with Polarization"
	prim.header['COMMENT']="Created using CAMB"
	#ipdb.set_trace()
	for i in range(num_wl):
		if LFI:
			nside=1024
			npix=hp.nside2npix(1024)
			simul_cmb=hp.ud_grade(simul_cmb,nside)
			alpha_radio=hp.ud_grade(alpha_radio,nside)
			sync_q=hp.ud_grade(sync_q,nside)
			sync_u=hp.ud_grade(sync_u,nside)
			dust_q=hp.ud_grade(sync_q,nside)
			dust_u=hp.ud_grade(sync_u,nside)
			pix_area=hp.nside2pixarea(nside)
		
		tmp_cmb=rotate_tqu(simul_cmb,wl[i],alpha_radio);
		tmp_didqdu=np.array([np.random.normal(0,1,npix)*noise_const_temp[i], np.random.normal(0,1,npix)*noise_const_pol[i] , np.random.normal(0,1,npix)*noise_const_pol[i]])
		tmp_tqu=np.copy(tmp_cmb)
		
		#Add Polarized Foreground emission
		tmp_tqu[1]+= np.copy( dust_factor[i]*dust_q+sync_factor[i]*sync_q    )
		tmp_tqu[2]+= np.copy( dust_factor[i]*dust_u+sync_factor[i]*sync_u    )
	#	tmp_tqu[1]+= np.copy(sync_factor[i]*sync_q)
	#	tmp_tqu[2]+= np.copy(sync_factor[i]*sync_u)
		tmp_tqu=hp.sphtfunc.smoothing(tmp_tqu,fwhm=beam_fwhm[i]*np.pi/(180.*60.),pol=1)
	
		#Add Noise After smooothing
		#tmp_tqu+=tmp_didqdu 

		sig_hdu=fits.ImageHDU(tmp_tqu)
		sig_hdu.header['TFIELDS']=(len(tmp_tqu),'number of fields in each row')
		sig_hdu.header["TTYPE1"]=("STOKES I")
		sig_hdu.header["TTYPE2"]=("STOKES Q")
		sig_hdu.header["TTYPE3"]=("STOKES U")
		sig_hdu.header["TUNIT1"]=("K_{CMB} Thermodynamic", 'Physical Units of Map')
		sig_hdu.header["TUNIT2"]=("K_{CMB} Thermodynamic", 'Physical Units of Map')
		sig_hdu.header["TUNIT3"]=("K_{CMB} Thermodynamic", 'Physical Units of Map')
		sig_hdu.header["TFORM1"]='E'
		sig_hdu.header["TFORM2"]='E'
		sig_hdu.header["TFORM3"]='E'
		
		sig_hdu.header["EXTNAME"]="STOKES IQU"
		sig_hdu.header['POLAR']= 'T'
		sig_hdu.header['POLCCONV']=('COSMO','Coord. Convention for polarisation COSMO/IAU')
		sig_hdu.header['PIXTYPE']=("HEALPIX","HEALPIX pixelisation")
		sig_hdu.header['ORDERING']=("RING","Pixel order scheme, either RING or NESTED")
		sig_hdu.header["NSIDE"]=(nside,'Healpix Resolution paramter')
		sig_hdu.header['OBJECT']=('FULLSKY','Sky coverage, either FULLSKY or PARTIAL')
		sig_hdu.header['OBS_NPIX']=(npix,'Number of pixels observed')
		sig_hdu.header['INDXSCHM']=('IMPLICIT','indexing : IMPLICIT of EXPLICIT')
		sig_hdu.header["COORDSYS"]=('G','Pixelization coordinate system')
		


		err_hdu=fits.ImageHDU(tmp_didqdu)
		err_hdu.header['TFIELDS']=(len(tmp_didqdu),'number of fields in each row')
		err_hdu.header["TTYPE1"]=("UNCERTAINTY I")
		err_hdu.header["TTYPE2"]=("UNCERTAINTY Q")
		err_hdu.header["TTYPE3"]=("UNCERTAINTY U")
		err_hdu.header["TUNIT1"]=("K_{CMB} Thermodynamic", 'Physical Units of Map')
		err_hdu.header["TUNIT2"]=("K_{CMB} Thermodynamic", 'Physical Units of Map')
		err_hdu.header["TUNIT3"]=("K_{CMB} Thermodynamic", 'Physical Units of Map')
		err_hdu.header["TFORM1"]='E'
		err_hdu.header["TFORM2"]='E'
		err_hdu.header["TFORM3"]='E'
		
		err_hdu.header["EXTNAME"]="UNCERTAINTIES"
		err_hdu.header['POLAR']= 'T'
		err_hdu.header['POLCCONV']=('COSMO','Coord. Convention for polarisation COSMO/IAU')
		err_hdu.header['PIXTYPE']=("HEALPIX","HEALPIX pixelisation")
		err_hdu.header['ORDERING']=("RING","Pixel order scheme, either RING or NESTED")
		err_hdu.header["NSIDE"]=(nside,'Healpix Resolution paramter')
		err_hdu.header['OBJECT']=('FULLSKY','Sky coverage, either FULLSKY or PARTIAL')
		err_hdu.header['OBS_NPIX']=(npix,'Number of pixels observed')
		err_hdu.header['INDXSCHM']=('IMPLICIT','indexing : IMPLICIT of EXPLICIT')
		err_hdu.header["COORDSYS"]=('G','Pixelization coordinate system')


	#	ipdb.set_trace()
		tblist=fits.HDUList([prim,sig_hdu,err_hdu])
		tblist.writeto(output_prefix+'planck_simulated_{0:0>3.0f}.fits'.format(bands[i]),clobber=True)
		print "planck_simulated_{:0>3.0f}.fits".format(bands[i])
		print "Nside = {:0>4d}".format(nside)
		if i+1 == LFI_IND:
			LFI=True


if __name__ == '__main__':
	main()
