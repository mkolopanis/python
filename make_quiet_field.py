import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import ipdb; 
from time import time
from rotate_tqu import rotate_tqu
from alpha_function import alpha_function

def convertcenter(ra,dec):
	return np.array([(ra[0]+ra[1]/60.)/24.*360,dec])

def regioncoords(ra,dec,dx,nx,ny):
	decs=np.array([ dec +dx*( j - ny/2.) for j in xrange(ny)])
	coords= np.array([ [np.mod(ra+dx*(i-nx/2)/np.cos(dec*np.pi/180.),360),dec]  for i in xrange(nx) for dec in decs])
	return coords

def main():	
	t1=time()
	radio_file='/data/wmap/faraday_MW_realdata.fits'
	cl_file='/home/matt/wmap/simul_scalCls.fits'
	output_prefix='/home/matt/quiet/quiet_maps/'
	nside=1024
	npix=hp.nside2npix(nside)
	
	cls=hp.read_cl(cl_file)
	print 'Generating Map'
	simul_cmb=hp.sphtfunc.synfast(cls,nside,fwhm=11.7*np.pi/(180.*60.),new=1,pol=1);
	
	alpha_radio=hp.read_map(radio_file,hdu='maps/phi');
	alpha_radio=hp.ud_grade(alpha_radio,nside_out=nside,order_in='ring',order_out='ring')
	bands=[43.1,94.5]
	q_fwhm=[27.3,11.7]
	noise_const_q=np.array([6./fwhm for fwhm in q_fwhm])*1e-6
	centers=np.array([convertcenter([12,4],-39),convertcenter([5,12],-39),convertcenter([0,48],-48),convertcenter([22,44],-36)])
	wl=np.array([299792458./(band*1e9) for band in bands])
	num_wl=len(wl)
	t_array=np.zeros((num_wl,npix))	
	q_array=np.zeros((num_wl,npix))
	sigma_q=np.zeros((num_wl,npix))
	u_array=np.zeros((num_wl,npix))
	sigma_u=np.zeros((num_wl,npix))
	for i in range(num_wl):
		tmp_cmb=rotate_tqu(simul_cmb,wl[i],alpha_radio);
		tmp_q=np.random.normal(0,1,npix)*noise_const_q[i]
		tmp_u=np.random.normal(0,1,npix)*noise_const_q[i]
		tmp_out=hp.sphtfunc.smoothing(tmp_cmb,fwhm=np.pi/180.,pol=1)
		t_array[i],q_array[i],u_array[i]=tmp_out
		sigma_q[i]=hp.sphtfunc.smoothing(tmp_q,fwhm=np.pi/180.)
		sigma_u[i]=hp.sphtfunc.smoothing(tmp_u,fwhm=np.pi/180.)
	
	print "Time to Write Fields"
	dx=1./(60.)*3
	nx=np.int(15/dx)
	ny=nx
	all_pix=[]
	field_pix=[]
	quiet_mask=np.zeros(npix)
	prim=fits.PrimaryHDU()
	prim.header['COMMENT']="Simulated Quiet Data"
	prim.header['COMMENT']="Created using CAMB"
	for p in xrange(len(centers)):
		coords=regioncoords(centers[p,0],centers[p,1],dx,nx,ny)
		coords_sky=SkyCoord(ra=coords[:,0],dec=coords[:,1],unit=u.degree,frame='fk5')
		phi=coords_sky.galactic.l.deg*np.pi/180.
		theta=(90-coords_sky.galactic.b.deg)*np.pi/180.
		pixels=hp.ang2pix(nside,theta,phi)
		quiet_mask[pixels]=1
		unique_pix=(np.unique(pixels).tolist())
		field_pix.append(unique_pix)
		all_pix.extend(unique_pix)
		pix_col=fits.Column(name='PIXEL',format='1J',array=unique_pix)
		for f in xrange(num_wl):
			region_mask=np.zeros(npix)
			region_mask[pixels]=1
			region_map_t=np.array(t_array[f][pixels]).reshape((nx,ny))
			region_map_q=np.array(q_array[f][pixels]).reshape((nx,ny))
			region_map_u=np.array(u_array[f][pixels]).reshape((nx,ny))
			region_delta_q=np.array(sigma_q[f][pixels]).reshape((nx,ny))
			region_delta_u=np.array(sigma_u[f][pixels]).reshape((nx,ny))
			prim=fits.PrimaryHDU()
			q_head=fits.ImageHDU([region_map_t,region_map_q,region_map_u],name="STOKES IQU")
			q_head.header['TFIELDS']=(3,'number of fields in each row')
			q_head.header['TTYPE1']=('SIGNAL', "STOKES I, Temperature")
			q_head.header['TUNIT1']=('K_{CMB} Thermodynamic', 'Physical Units of Map')
			q_head.header['TTYPE2']='STOKES Q'
			q_head.header['TUNIT2']=('K_{CMB} Thermodynamic', 'Physical Units of Map')
			q_head.header['TTYPE3']='STOKES U'
			q_head.header['TUNIT3']=('K_{CMB} Thermodynamic', 'Physical Units of Map')
			q_head.header['TFORM1']='E'
			q_head.header['TFORM1']='E'
			q_head.header['TFORM2']='E'
			q_head.header['PIXTYPE']=("HEALPIX","HEALPIX pixelisation")
			q_head.header['ORDERING']=("RING","Pixel order scheme, either RING or NESTED")
			q_head.header["COORDSYS"]=('G','Pixelization coordinate system')
			q_head.header['NSIDE']=(1024,'Healpix Resolution paramter')
			q_head.header['OBJECT']=('FULLSKY','Sky coverage, either FULLSKY or PARTIAL')
			q_head.header['INDXSCHM']=('EXPLICIT','indexing : IMPLICIT of EXPLICIT')
			err_head=fits.ImageHDU([region_delta_q,region_delta_u],name="Q/U UNCERTAINTIES")
			err_head.header['TFIELDS']=(2,'number of fields in each row')
			err_head.header['NSIDE']=1024
			err_head.header['ORDERING']='RING'
			err_head.header['TTYPE1']='SIGMA Q'
			err_head.header['TUNIT1']=('K_{CMB} Thermodynamic', 'Physical Units of Map')
			err_head.header['TTYPE2']='SIGMA U'
			err_head.header['TUNIT2']=('K_{CMB} Thermodynamic', 'Physical Units of Map')
			err_head.header['PIXTYPE']=("HEALPIX","HEALPIX pixelisation")
			err_head.header['OBJECT']=('FULLSKY','Sky coverage, either FULLSKY or PARTIAL')
			err_head.header['INDXSCHM']=('EXPLICIT','indexing : IMPLICIT of EXPLICIT')
			m_head=fits.ImageHDU(region_mask,name='MASK')	
			hdulist=fits.HDUList([prim,q_head,err_head,m_head])
			hdulist.writeto(output_prefix+"quiet_simulated_{:.1f}_cmb{:1d}.fits".format(bands[f],p+1),clobber=True)
			print '{:.1f}_cmb{:1d}.fits'.format(bands[f],p+1)
	
	
	mask_head=fits.ImageHDU(quiet_mask,name='MASK')
	pix_col=fits.Column(name='PIXEL',format='1J',array=all_pix)
	field_pix_col1=fits.Column(name='PIXELS FIELD 1',format='1J',array=field_pix[0])
	field_pix_col2=fits.Column(name='PIXELS FIELD 2',format='1J',array=field_pix[1])
	field_pix_col3=fits.Column(name='PIXELS FIELD 3',format='1J',array=field_pix[2])
	field_pix_col4=fits.Column(name='PIXELS FIELD 4',format='1J',array=field_pix[3])
	for i in xrange(num_wl):
		cut_t,cut_q,cut_u=t_array[i][all_pix],q_array[i][all_pix],u_array[i][all_pix]
		cut_dq,cut_du=sigma_q[i][all_pix],sigma_u[i][all_pix]
		col_t=fits.Column(name='SIGNAL',format='1E',unit='K_{CMB}',array=cut_t)
		col_q=fits.Column(name='STOKES Q',format='1E',unit='K_{CMB}',array=cut_q)
		col_u=fits.Column(name='STOKES U',format='1E',unit='K_{CMB}',array=cut_u)
		col_dq=fits.Column(name='Q ERROR',format='1E',unit='K_{CMB}',array=cut_dq)
		col_du=fits.Column(name='U ERROR',format='1E',unit='K_{CMB}',array=cut_du)
		cols=fits.ColDefs([pix_col,col_q,col_u,col_dq,col_du])
		tbhdu=fits.BinTableHDU.from_columns(cols)
		tbhdu.header['TFIELDS']=(5,'number of fields in each row')
		tbhdu.header["TTYPE2"]=("SIGNAL","STOKES T")
		tbhdu.header["EXTNAME"]="SIGNAL"
		tbhdu.header['POLAR']= 'T'
		tbhdu.header['POLCCONV']=('COSMO','Coord. Convention for polarisation COSMO/IAU')
		tbhdu.header['PIXTYPE']=("HEALPIX","HEALPIX pixelisation")
		tbhdu.header['ORDERING']=("RING","Pixel order scheme, either RING or NESTED")
		tbhdu.header["NSIDE"]=(1024,'Healpix Resolution paramter')
		tbhdu.header['OBJECT']=('PARTIAL','Sky coverage, either FULLSKY or PARTIAL')
		tbhdu.header['OBS_NPIX']=(len(all_pix),'Number of pixels observed')
		tbhdu.header['INDXSCHM']=('IMPLICIT','indexing : IMPLICIT of EXPLICIT')
		tbhdu.header["COORDSYS"]=('G','Pixelization coordinate system')
		tblist=fits.HDUList([prim,tbhdu])
		tblist.writeto(output_prefix+'quiet_partial_simulated_{:.1f}.fits'.format(bands[i]),clobber=True)
	
		q_head=fits.ImageHDU(np.array([t_array[i],q_array[i],u_array[i]]), name='STOKES IQU')
		q_head.header['TFIELDS']=(3,'number of fields in each row')
		q_head.header['TYPE1']=('SIGNAL', "STOKES I, Temperature")
		q_head.header['TYPE2']='STOKES Q'
		q_head.header['TYPE3']='STOKES U'
		q_head.header['TUNIT1']=('K_{CMB} Thermodynamic', 'Physical Units of Map')
		q_head.header['TUNIT2']=('K_{CMB} Thermodynamic', 'Physical Units of Map')
		q_head.header['TUNIT3']=('K_{CMB} Thermodynamic', 'Physical Units of Map')
		q_head.header['TFORM1']='E'
		q_head.header['TFORM1']='E'
		q_head.header['TFORM2']='E'
		q_head.header['EXTNAME']='STOKES IQU'
		q_head.header['POLAR']= 'T'
		q_head.header['POLCCONV']=('COSMO','Coord. Convention for polarisation COSMO/IAU')
		q_head.header['PIXTYPE']=("HEALPIX","HEALPIX pixelisation")
		q_head.header['ORDERING']=("RING","Pixel order scheme, either RING or NESTED")
		q_head.header['NSIDE']=(1024,'Healpix Resolution paramter')
		q_head.header['OBJECT']=('FULLSKY','Sky coverage, either FULLSKY or PARTIAL')
		q_head.header['INDXSCHM']=('IMPLICIT','indexing : IMPLICIT of EXPLICIT')
		q_head.header['BAD_DATA']=(hp.UNSEEN,'Sentinel value given to bad pixels')
		q_head.header["COORDSYS"]=('G','Pixelization coordinate system')
		
		cols=fits.ColDefs([field_pix_col1,field_pix_col2,field_pix_col3,field_pix_col4])
		tbhdu=fits.BinTableHDU.from_columns(cols)
		tbhdu.header['TFIELDS']=(4,'number of fields in each row')
		tbhdu.header["TTYPE1"]=("PIXELS CMB FIELD 1","PIXEL NUMBER BY FIELD")
		tbhdu.header["TTYPE2"]=("PIXELS CMB FIELD 2","PIXEL NUMBER BY FIELD")
		tbhdu.header["TTYPE3"]=("PIXELS CMB FIELD 3","PIXEL NUMBER BY FIELD")
		tbhdu.header["TTYPE4"]=("PIXELS CMB FIELD 4","PIXEL NUMBER BY FIELD")
		tbhdu.header["EXTNAME"]="FIELD PIXELS"
		tbhdu.header['PIXTYPE']=("HEALPIX","HEALPIX pixelisation")
		tbhdu.header['ORDERING']=("RING","Pixel order scheme, either RING or NESTED")
		tbhdu.header["NSIDE"]=(nside,'Healpix Resolution paramter')
		tbhdu.header['OBJECT']=('PARTIAL','Sky coverage, either FULLSKY or PARTIAL')
		tbhdu.header['OBS_NPIX']=(len(all_pix),'Number of pixels observed')
		tbhdu.header['INDXSCHM']=('IMPLICIT','indexing : IMPLICIT of EXPLICIT')
		tbhdu.header["COORDSYS"]=('G','Pixelization coordinate system')
		#tblist=fits.HDUList([prim,tbhdu])
		err_head=fits.ImageHDU(np.array([sigma_q[i],sigma_u[i]]),name='Q/U UNCERTAINTIES')
		err_head.header['TFIELDS']=(2,'number of fields in each row')
		err_head.header['NSIDE']=1024
		err_head.header['ORDERING']='RING'
		err_head.header['TTYPE1']='SIGMA Q'
		err_head.header['TTYPE2']='SIGMA U'
		err_head.header['TUNIT1']=('K_{CMB} Thermodynamic', 'Physical Units of Map')
		err_head.header['TUNIT2']=('K_{CMB} Thermodynamic', 'Physical Units of Map')
		err_head.header['TFORM1']='E'
		err_head.header['TFORM2']='E'
		err_head.header['EXTNAME']='Q/U UNCERTAINTIES'
		err_head.header['PIXTYPE']=("HEALPIX","HEALPIX pixelisation")
		err_head.header['OBJECT']=('FULLSKY','Sky coverage, either FULLSKY or PARTIAL')
		err_head.header['INDXSCHM']=('IMPLICIT','indexing : IMPLICIT of EXPLICIT')
		err_head.header['BAD_DATA']=(hp.UNSEEN,'Sentinel value given to bad pixels')
		hdulist=fits.HDUList([prim,q_head,err_head,mask_head,tbhdu])
		hdulist.writeto(output_prefix+"quiet_simulated_{:.1f}.fits".format(bands[i]),clobber=True)
		print "quiet_simulated_{:.1f}.fits".format(bands[i])
