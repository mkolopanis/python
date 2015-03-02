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
	
	
	##Define Parameters used to simulate Planck Fields
	bands=np.array([30.,44.,70.,100.,143.,217.,353.])
	beam_fwhm=np.array([33.,24.,14.,10.,7.1,5.0,5.0])
	noise_const_temp=np.array([2.0,2.7,4.7,2.5,2.2,4.8,14.7])*2.7255e-6
	noise_const_pol=np.array([2.8,3.9,6.7,4.0,4.2,9.8,29.8])*2.7255e-6
	#beam_fwhm=np.array([33.,24.,14.,10.,7.1,5.0,5.0])
	beam_fwhm=np.array([32.29,27,13.21,9.67,7.26,4.96,4.93])
	#noise_const_temp=np.array([2.0,2.7,4.7,2.5,2.2,4.8,14.7])*2.7255e-6/2.
	#noise_const_pol=np.array([2.8,3.9,6.7,4.0,4.2,9.8,29.8])*2.7255e-6/2.
	noise_const_temp=np.array([2.5,2.7,3.5,1.29,.555,.78,2.56])*60.e-6
	noise_const_pol=np.array([3.5,4.0,5.0,1.96,1.17,1.75,7.31])*60e-6
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
	LFI=False ##Flag used to find where to switch to NISDE=1024
	LFI_IND= np.where(bands == 70.)[0][0] ##Index used to switch flag
	

	cls=hp.read_cl(cl_file)
	print 'Generating Map'
	simul_cmb=hp.sphtfunc.synfast(cls,nside,fwhm=0.,new=1,pol=1);
	
	alpha_radio=hp.read_map(radio_file,hdu='maps/phi');
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=nside,order_in='ring',order_out='ring')

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
			pix_area=hp.nside2pixarea(nside)
		
		tmp_cmb=rotate_tqu(simul_cmb,wl[i],alpha_radio);
		tmp_didqdu=np.array([np.random.normal(0,1,npix)*noise_const_temp[i], np.random.normal(0,1,npix)*noise_const_pol[i] , np.random.normal(0,1,npix)*noise_const_pol[i]])
		tmp_tqu=hp.sphtfunc.smoothing(tmp_cmb,fwhm=np.sqrt((beam_fwhm[i]*np.pi/(180.*60.))**2 - pix_area),pol=1)

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
		if i+1 >= LFI_IND:
			LFI=True


if __name__ == '__main__':
	main()
