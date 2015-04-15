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
	output_prefix='/home/matt/prism/simul_maps/'
	
	
	##Define Parameters used to simulate PRISM Fields
	nside_out=128
	npix_out=hp.nside2npix(nside_out)
	bands=np.array([30.,36.,51.,105.,135.,160.])
	wl=np.array([299792458./(b*1e9) for b in bands])
	beam_fwhm=np.array([17.,14.,10.,4.8,3.8,3.2])
	n_det=np.array([50,100,150,250,300,350])
	noise_const_temp=np.array([63.4,59.7,53.7,45.6,44.9,45.5])/np.sqrt(n_det)*1e-6
	noise_const_pol=np.array([89.7,84.5,75.9,64.4,63.4,64.3])/np.sqrt(n_det)*1e-6

	nside=2048
	npix=hp.nside2npix(nside)
	pix_area=hp.nside2pixarea(nside)

	num_wl=len(wl)
	tqu_array=[]
	sigma_array=[]

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
		tblist.writeto(output_prefix+'prism_simulated_{0:0>3.0f}.fits'.format(bands[i]),clobber=True)
		print "prism_simulated_{:0>3.0f}.fits".format(bands[i])
		print "Nside = {:0>4d}".format(nside)


if __name__ == '__main__':
	main()
