import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as units
import ipdb
from IPython import embed
import glob
import plot_binned
import bin_llcl
import subprocess
import rotate_tqu
from matplotlib.ticker import MaxNLocator

def convertcenter(ra,dec):
	return np.array([(ra[0]+ra[1]/60.)/24.*360,dec])

def regioncoords(ra,dec,dx,nx,ny):
	decs=np.array([ dec +dx*( j - ny/2.) for j in xrange(ny)])
	coords= np.array([ [np.mod(ra+dx*(i-nx/2)/np.cos(dec*np.pi/180.),360),dec]  for i in xrange(nx) for dec in decs])
	return coords

survey='KECK'
centers=convertcenter([0.0,0],-57.5)
#centers=convertcenter([50.*24/360.,0],-35.)
#centers=convertcenter([1.5,0],-52.5)
smoothing_scale=40.0
bins=25
N_runs=500
nside_out=128
gal_cut=[0,5,10,15,20,30]
npix_out=hp.nside2npix(nside_out)
#pix_area=hp.nside2pixarea(nside_out)
pix_area=(smoothing_scale*np.pi/180/60.)**2


nside_in=1024
npix_in=hp.nside2npix(nside_in)

bands=np.array([95,150])
wl=np.array([299792458./(b*1e9) for b in bands])
beam_fwhm=np.array([.22,.22])*60*np.sqrt(8*np.log(2))


pix_area_array=np.repeat(hp.nside2pixarea(nside_in),len(bands))
pix_area_array=np.sqrt(pix_area_array)*60*180./np.pi
#noise_const_temp=np.array([11.])/pix_area_array*1.e-6
noise_const_array=np.array([3.4,3.4])/pix_area_array*1.e-6

#ukdet=np.array([150.,150.,380.])
#ndet=np.array([288.,512.,512.])
#det_eff=.85
#ndays=15.*24*3600
#FPU=np.array([5,5,3])
#
#noise_const_array=ukdet*np.sqrt(4125.*60**2)/np.sqrt(FPU*ndet*det_eff*ndays)/pix_area_array*1e-6

#gamma_dust=6.626e-34/(1.38e-23*21)
#dust_factor=krj_to_kcmb*np.array([1e-6*(np.exp(gamma_dust*353e9)-1)/(np.exp(gamma_dust*x*1e9)-1)* (x/353.)**2.54 for x in bands])
#krj_to_kcmb=np.array([1.,1.,1.])
krj_to_kcmb=np.ones_like(bands)
#
sync_factor=krj_to_kcmb*np.array([1e-6*(30./x)**2 for x in bands])

beam=hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),3*nside_out-1)

bls=(hp.gauss_beam(smoothing_scale*np.pi/(180.*60.),3*nside_out-1)*hp.pixwin(nside_out)[:3*nside_out])**2


l=np.arange(3*nside_out)
ll=l*(l+1)/(2*np.pi)


def likelihood(cross,dcross,theory,name,title):

	Sig=np.sum(cross/(dcross**2))/np.sum(1./dcross**2)
	#Noise=np.std(np.sum(cross_array/dcross**2,axis=1)/np.sum(1./dcross**2))	\
	Noise=np.sqrt(1./np.sum(1./dcross**2))
	Sig1=np.sum(cross*(theory/dcross)**2)/np.sum((theory/dcross)**2)
	Noise1=np.sqrt(np.sum(dcross**2*(theory/dcross)**2)/np.sum((theory/dcross)**2))
	SNR=Sig/Noise
	SNR1=Sig1/Noise1

	a_scales=np.linspace(-10000,10000,100000)
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
		else:
			ax1.set_xlim([-5,5])
		f=open('Maximum_likelihood_simulation_'+name+'_'+title+'.txt','w')
		f.write('Maximum Likelihood: {0:2.5f}%  for scale factor {1:.2f} \n'.format(float(chi_array[np.argmax(chi_array)]*100),mean))
		f.write('Posterior: Mean,\tsigma,\t(1siglo,1sighi),\t(2sighlo,2sighi)\n')
                f.write('Posterior: {0:.3f},\t{1:.3f} ,\t({2:.3f},{3:.3f})\t({4:.3f},{5:.3f})\n '.format(mean,np.mean([s1hi-mean,mean-s1lo]) ,s1lo,s1hi, s2lo,s2hi))
		f.write('Posterior SNR:\t {0:.3f}'.format(1./np.mean( [s1hi-mean,mean-s1lo] ) ) )
		f.write('\n\n')
		f.write('Detection Levels using Standard Deviation \n')
		f.write('Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n'.format(SNR,Sig, Noise))
		f.write('Weighted Detection Level: {0:.4f} sigma, Signal= {1:.4e}, Noise= {2:.4e} \n \n'.format(SNR1,Sig1,Noise1))
		f.close()

	except:
		print('Scale exceeded for possterior. \n Plotting anyway')
	ax1.plot(a_scales,chi_array,'k',linewidth=2)
	ax1.set_title('Faraday Rotatior Posterior')
	ax1.set_xlabel('Likelihood scalar')
	ax1.set_ylabel('Likelihood of Correlation')

	fig.savefig('FR_simulation_likelihood_'+name+'_'+title+'.png',format='png')
	fig.savefig('FR_simulation_likelihood_'+name+'_'+title+'.eps',format='eps')





alpha_file='/data/wmap/faraday_MW_realdata.fits'
#theory_cls=hp.read_cl('/home/matt/Planck/data/faraday/correlation/fr_theory_cl.fits')


cmb_cls=hp.read_cl('/home/matt/wmap/simul_scalCls.fits.lens')

synchrotron_file='/data/Planck/COM_CompMap_SynchrotronPol-commander_0256_R2.00.fits'
dust_file='/data/Planck/COM_CompMap_DustPol-commander_1024_R2.00.fits'
dust_t_file='/data/Planck/COM_CompMap_dust-commander_0256_R2.00.fits'
dust_b_file='/data/Planck/COM_CompMap_ThermalDust-commander_2048_R2.00.fits'

##Dust intensity scaling factor
hdu_dust_t=fits.open(dust_t_file)
dust_t=hdu_dust_t[1].data.field('TEMP_ML')
hdu_dust_t.close()

dust_t=hp.reorder(dust_t,n2r=1)
dust_t=hp.ud_grade(dust_t,nside_in)

hdu_dust_b=fits.open(dust_b_file)
dust_beta=hdu_dust_b[1].data.field('BETA_ML_FULL')
hdu_dust_b.close

dust_beta=hp.reorder(dust_beta,n2r=1)	
dust_beta=hp.ud_grade(dust_beta,nside_in)

gamma_dust=6.626e-34/(1.38e-23*dust_t)
dust_factor=np.array([krj_to_kcmb[i]*1e-6*(np.exp(gamma_dust*353e9)-1)/(np.exp(gamma_dust*x*1e9)-1)* (x/353.)**(1+dust_beta) for i,x in enumerate(bands)])
df_sub=[ hp.ud_grade(df,nside_out) for  df in dust_factor]

##Testing for noise levels
##Create Field for analysis
dx=1./(60.)*3
field_size=400
nx=np.int(np.sqrt(field_size)/dx)
ny=nx
coords=regioncoords(centers[0],centers[1],dx,nx,ny)
coords_sky=SkyCoord(ra=coords[:,0],dec=coords[:,1],unit=units.degree,frame='fk5')
phi=coords_sky.galactic.l.deg*np.pi/180.
theta=(90-coords_sky.galactic.b.deg)*np.pi/180.
good_pix=hp.ang2pix(nside_in,theta,phi)

print('Keck Array')
print('\tPreparing Foreground Maps')


bl_40=hp.gauss_beam(40.*np.pi/(180.*60.),3*nside_in-1)
bl_10=hp.gauss_beam(10.*np.pi/(180.*60.),3*nside_in-1)

hdu_sync=fits.open(synchrotron_file)
sync_q=hdu_sync[1].data.field(0)
sync_u=hdu_sync[1].data.field(1)

sync_q=hp.reorder(sync_q,n2r=1)
tmp_alm=hp.map2alm(sync_q)
tmp_alm=hp.almxfl(tmp_alm,1./bl_40)
#simul_sync_q=hp.smoothing(sync_q,fwhm=40.*np.pi/(180.*60.),verbose=False,invert=True)
simul_sync_q=hp.alm2map(tmp_alm,nside_in,verbose=False)
#simul_sync_q=hp.ud_grade(simul_sync_q,1024)
sync_q=hp.ud_grade(sync_q,nside_out=nside_out)
sync_q_back=np.copy(sync_q)

sync_u=hp.reorder(sync_u,n2r=1)
tmp_alm=hp.map2alm(sync_u)
tmp_alm=hp.almxfl(tmp_alm,1./bl_40)
#simul_sync_q=hp.smoothing(sync_q,fwhm=40.*np.pi/(180.*60.),verbose=False,invert=True)
simul_sync_u=hp.alm2map(tmp_alm,nside_in,verbose=False)
sync_u=hp.ud_grade(sync_u,nside_out=nside_out)
sync_u_back=np.copy(sync_u)
hdu_sync.close()

hdu_dust=fits.open(dust_file)
dust_q=hdu_dust[1].data.field(0)
dust_u=hdu_dust[1].data.field(1)
hdu_dust.close()

dust_q=hp.reorder(dust_q,n2r=1)
tmp_alm=hp.map2alm(dust_q)
tmp_alm=hp.almxfl(tmp_alm,1./bl_10)
#simul_dust_q=hp.smoothing(dust_q,fwhm=10.*np.pi/(180.*60.),verbose=False,invert=True)
simul_dust_q=hp.alm2map(tmp_alm,nside_in,verbose=False)
#simul_dust_q=np.copy(dust_q)
dust_q=hp.smoothing(dust_q,fwhm=np.sqrt(smoothing_scale**2-10.0**2)*np.pi/(180.*60.),verbose=False)
dust_q=hp.ud_grade(dust_q,nside_out)
dust_q_back=np.copy(dust_q)

dust_u=hp.reorder(dust_u,n2r=1)
tmp_alm=hp.map2alm(dust_u)
tmp_alm=hp.almxfl(tmp_alm,1./bl_10)
#simul_dust_q=hp.smoothing(dust_q,fwhm=10.*np.pi/(180.*60.),verbose=False,invert=True)
simul_dust_u=hp.alm2map(tmp_alm,nside_in,verbose=False)
#simul_dust_u=np.copy(dust_u)
dust_u=hp.smoothing(dust_u,fwhm=np.sqrt(smoothing_scale**2-10.0**2)*np.pi/(180.*60.),verbose=False)
dust_u=hp.ud_grade(dust_u,nside_out)
dust_u_back=np.copy(dust_u)
#
#mask_hdu=fits.open(mask_file)
#mask=mask_hdu[1].data.field(0)
#mask_hdu.close()
#
#mask=hp.reorder(mask,n2r=1)
mask=np.zeros(npix_in)
mask[good_pix]+= 1.
mask=hp.ud_grade(mask,nside_out=nside_out)

mask_bool=~mask.astype(bool)
hp.mollview(mask_bool)
plt.savefig(survey+'_region.png',format='png')
plt.savefig(survey+'_region.eps',format='eps')
plt.clf()
plt.close()
fsky= 1. - np.sum(mask_bool)/float(len(mask_bool))	
L=np.sqrt(fsky*4*np.pi)
dl_eff=2*np.pi/L


print('\tGenerating Control Map')
alpha_radio=hp.read_map(alpha_file,hdu='maps/phi',verbose=False)
tmp_alpha=hp.ud_grade(alpha_radio,nside_in)
alpha_radio=hp.read_map(alpha_file,hdu='maps/phi',verbose=False)
sigma_alpha=hp.read_map(alpha_file,hdu='uncertainty/phi',verbose=False)

iqu_cmb=hp.synfast(cmb_cls,new=1,fwhm=0,pol=1,nside=nside_in,verbose=False)
iqu_array=[rotate_tqu.rotate_tqu(iqu_cmb,w,tmp_alpha) for w in wl] 

iqu_array = [ np.array([iqu[0], \
	iqu[1] + np.copy(simul_sync_q*sync_factor[i] + simul_dust_q*dust_factor[i]), iqu[2] + np.copy(simul_sync_u*sync_factor[i] + simul_dust_u*dust_factor[i])]) for i,iqu in enumerate(iqu_array)] 

iqu_array= [ hp.smoothing(iqu,pol=1,fwhm=beam_fwhm[cnt]*np.pi/(180.*60.),verbose=False) for cnt,iqu in enumerate(iqu_array)]
#iqu_array_fr= [ hp.smoothing(iqu,pol=1,fwhm=beam_fwhm[cnt]*np.pi/(180.*60.),verbose=False) for cnt,iqu in enumerate(iqu_array_fr)]

iqu_back=np.copy(iqu_array)
const_array=[]
band_names=[]
the_au_index=[]
au_index=[]
for i in xrange(len(bands)-1):
	for j in xrange(i+1,len(bands)):
		const_array.append(2*(wl[i]**2-wl[j]**2))
		band_names.append('{0:0>3d}_{1:0>3d}'.format(bands[i],bands[j]))
		au_index.append(j)
		the_au_index.append(j)

const_array=np.array(const_array)
band_names=np.array(band_names)

cross_2band_array=[]
cross_dq_2band_array=[]
cross_du_2band_array=[]

cross_2band_array_fr=[]
cross_dq_2band_array_fr=[]
cross_du_2band_array_fr=[]

theory_2band_array=[]
theory_dq_2band_array=[]
theory_du_2band_array=[]

N_dq_2band_array=[]
N_du_2band_array=[]

N_au_array=[]
N_aq_array=[]
N_au_array_fr=[]
N_aq_array_fr=[]

for run in xrange(N_runs):
	print('\tRealization {0:d}'.format(run+1))
	print('\t\tSmoothing and Downgrading Maps')

	iqu_array=iqu_back.copy().tolist()

	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi',verbose=False)
        sigma_alpha=hp.read_map(alpha_file,hdu='uncertainty/phi',verbose=False)

	sigma_array=[]
	for noise_const in noise_const_array:
		sigma_array.append( noise_const*np.random.normal(0,1,(3,npix_in)))
	for nb in xrange(len(sigma_array)):
		sigma_array[nb]=hp.smoothing(sigma_array[nb], pol=1,fwhm=beam_fwhm[nb]*np.pi/(180.*60.),verbose=False)

	#sigma_array=np.array(sigma_array)
	iqu_array+=np.copy(sigma_array)
	iqu_array=iqu_array.tolist()
	for nb in xrange(len(iqu_array)):
		iqu_array[nb]=hp.smoothing(iqu_array[nb],pol=1,fwhm=np.sqrt( (smoothing_scale)**2 - (beam_fwhm[nb])**2 )*np.pi/(180.*60.),verbose=False)
		iqu_array[nb]=hp.ud_grade(iqu_array[nb],nside_out=nside_out)

		sigma_array[nb]=hp.smoothing(sigma_array[nb],pol=1,fwhm=np.sqrt(smoothing_scale**2 - beam_fwhm[nb]**2 )*np.pi/(180.*60.),verbose=False)
		sigma_array[nb]=hp.ud_grade(sigma_array[nb],nside_out=nside_out)

	iqu_array=[hp.ma(iqu) for iqu in iqu_array]
	sigma_array=[hp.ma(sigma) for sigma in sigma_array]
	alpha_radio = hp.ma(alpha_radio)
	sigma_alpha = hp.ma(sigma_alpha)
	alpha_radio.mask=mask_bool

	for nb in xrange(len(iqu_array)):
		for m in xrange(3):
			iqu_array[nb][m].mask=mask_bool
			sigma_array[nb][m].mask=mask_bool
        print "\t\tRemoving Foregrounds"
        iqu_array_fr=[]
        for cnt,iqu in enumerate(iqu_array):
                #redefine temps to make math easier
                dq=hp.ma(df_sub[cnt]*dust_q)
                du=hp.ma(df_sub[cnt]*dust_u)
                sq=hp.ma(sync_factor[cnt]*sync_q)
                su=hp.ma(sync_factor[cnt]*sync_u)

                dq.mask=mask_bool
                du.mask=mask_bool
                sq.mask=mask_bool
                su.mask=mask_bool
                #normalization factors for scaling 
                gamma_sync_q= np.sum(iqu[1]*sq)/np.sum(sq**2)- np.sum(dq*sq)/np.sum(sq**2)*( (np.sum(sq**2)*np.sum(iqu[1]*dq)-np.sum(iqu[1]*sq)*np.sum(sq*dq))/(np.sum(dq**2)*np.sum(sq**2)-np.sum(sq*dq)**2) )
                delta_dust_q= (np.sum(sq**2)*np.sum(iqu[1]*dq)-np.sum(iqu[1]*sq)*np.sum(sq*dq))/( np.sum(dq**2)*np.sum(sq**2)-np.sum(sq*dq)**2)

                gamma_sync_u= np.sum(iqu[2]*su)/np.sum(su**2)- np.sum(du*su)/np.sum(su**2)*( (np.sum(su**2)*np.sum(iqu[2]*du)-np.sum(iqu[2]*su)*np.sum(su*du))/(np.sum(du**2)*np.sum(su**2)-np.sum(su*du)**2) )
                delta_dust_u= (np.sum(su**2)*np.sum(iqu[2]*du)-np.sum(iqu[2]*su)*np.sum(su*du))/( np.sum(du**2)*np.sum(su**2)-np.sum(su*du)**2)

                iqu_array_fr.append( np.array([iqu[0], iqu[1]-gamma_sync_q*sq-delta_dust_q*dq, iqu[2]-gamma_sync_u*su-delta_dust_u*du]))

	iqu_array_fr=[hp.ma(iqu) for iqu in iqu_array_fr]
	for nb in xrange(len(iqu_array_fr)):
		for m in xrange(len(iqu_array_fr[nb])):
			iqu_array_fr[nb][m].mask=mask_bool
#################################################
	DQ_array=[]
	DU_array=[]
	DQ_fr_array=[]
	DU_fr_array=[]

	for i in xrange(len(iqu_array)-1):
		for j in xrange(i+1,len(iqu_array)):
			DQ_array.append( ( iqu_array[i][1] - iqu_array[j][1])/(2*(wl[i]**2-wl[j]**2)))
			DU_array.append( ( iqu_array[i][2] - iqu_array[j][2])/(2*(wl[i]**2-wl[j]**2)))
			DQ_fr_array.append( ( iqu_array_fr[i][1] - iqu_array_fr[j][1])/(2*(wl[i]**2-wl[j]**2)))
			DU_fr_array.append( ( iqu_array_fr[i][2] - iqu_array_fr[j][2])/(2*(wl[i]**2-wl[j]**2)))

	aQ_array= [ -im[1]*alpha_radio for im in iqu_array]
	aU_array= [ im[2]*alpha_radio  for im in iqu_array]

	aQ_fr_array= [ -im[1]*alpha_radio for im in iqu_array_fr]
	aU_fr_array= [ im[2]*alpha_radio for im in iqu_array_fr]

	print('\t\tCreating Correlators')
	#DQ_array=hp.ma(DQ_array)
	#DU_array=hp.ma(DU_array)
	##DQ_fr_array=hp.ma(DQ_fr_array)
	##DU_fr_array=hp.ma(DU_fr_array)
	#aQ_array=hp.ma(aQ_array)
	#aU_array=hp.ma(aU_array)
	#aQ_fr_array=hp.ma(aQ_fr_array)
	#aU_fr_array=hp.ma(aU_fr_array)

	#print('Commander Polarization Analysis Mask')
	for num in xrange(len(DQ_array)):
		DQ_array[num].mask=mask_bool
		DU_array[num].mask=mask_bool
		DQ_fr_array[num].mask=mask_bool
		DU_fr_array[num].mask=mask_bool
	for num in xrange(len(aQ_array)):
		aQ_array[num].mask=mask_bool
		aU_array[num].mask=mask_bool
		aQ_fr_array[num].mask=mask_bool
		aU_fr_array[num].mask=mask_bool
	#sync_q.mask=mask_bool
	#sync_u.mask=mask_bool
	cross_2band_array.append( [ hp.anafast(DQ_array[i], map2=aU_array[au_index[i]]) + hp.anafast(DU_array[i], map2=aQ_array[au_index[i]]) for i in xrange(len(DQ_array))])
	cross_dq_2band_array.append([ hp.anafast(DQ_array[i], map2=aU_array[au_index[i]]) for i in xrange(len(DQ_array))])
	cross_du_2band_array.append([ hp.anafast(DU_array[i], map2=aQ_array[au_index[i]]) for i in xrange(len(DQ_array))])

	cross_2band_array_fr.append([ hp.anafast(DQ_fr_array[i], map2=aU_fr_array[au_index[i]]) + hp.anafast(DU_fr_array[i], map2=aQ_fr_array[au_index[i]]) for i in xrange(len(DQ_array))])
	cross_dq_2band_array_fr.append([ hp.anafast(DQ_fr_array[i], map2=aU_fr_array[au_index[i]]) for i in xrange(len(DQ_array))])
	cross_du_2band_array_fr.append([ hp.anafast(DU_fr_array[i], map2=aQ_fr_array[au_index[i]]) for i in xrange(len(DQ_array))])

	
	print('\t\tCreating Noise/Theory Estimator')

	alpha_radio=hp.read_map(alpha_file,hdu='maps/phi',verbose=False)
	sigma_alpha=hp.read_map(alpha_file,hdu='uncertainty/phi',verbose=False)*np.random.normal(0,1,len(alpha_radio))
	tmp_alpha=hp.ud_grade(alpha_radio,nside_in)
	iqu_cmb=hp.synfast(cmb_cls,new=1,fwhm=0,pol=1,nside=nside_in,verbose=False)
	sim_array=[rotate_tqu.rotate_tqu(iqu_cmb,w,tmp_alpha) for w in wl]
	for nb in xrange(len(iqu_array)):
		sim_array[nb]=hp.smoothing(sim_array[nb],pol=1,fwhm=beam_fwhm[nb]*np.pi/(180.*60.),verbose=False)
		sim_array[nb]=hp.smoothing(sim_array[nb],pol=1,fwhm=np.sqrt( (smoothing_scale)**2 - (beam_fwhm[nb])**2 )*np.pi/(180.*60.),verbose=False)
		sim_array[nb]=hp.ud_grade(sim_array[nb],nside_out=nside_out)
	
	the_DQ_array=[]
	the_DU_array=[]
	for i in xrange(len(sim_array)-1):
		for j in xrange(i+1,len(sim_array)):	
			the_DQ_array.append( hp.ma( sim_array[i][1] - sim_array[j][1])/(2*(wl[i]**2-wl[j]**2)))
			the_DU_array.append( hp.ma( sim_array[i][2] - sim_array[j][2])/(2*(wl[i]**2-wl[j]**2)))

	the_aQ_array= [hp.ma( -im[1]*alpha_radio) for im in sim_array]
	the_aU_array= [hp.ma( im[2]*alpha_radio ) for im in sim_array]

	
	#the_DQ_array=hp.ma(the_DQ_array)
	#the_DU_array=hp.ma(the_DU_array)
	#the_aQ_array=hp.ma(the_aQ_array)
	#the_aU_array=hp.ma(the_aU_array)

	#print('Commander Polarization Analysis Mask')
	for num in xrange(len(DQ_array)):
		the_DQ_array[num].mask=mask_bool
		the_DU_array[num].mask=mask_bool
	for num in xrange(len(aQ_array)):
		the_aQ_array[num].mask=mask_bool
		the_aU_array[num].mask=mask_bool
	#sync_q.mask=mask_bool
	#sync_u.mask=mask_bool
	theory_2band_array.append([ hp.anafast(the_DQ_array[i], map2=the_aU_array[the_au_index[i]]) + hp.anafast(the_DU_array[i], map2=the_aQ_array[the_au_index[i]]) for i in xrange(len(the_DQ_array))])
	theory_dq_2band_array.append([ hp.anafast(the_DQ_array[i], map2=the_aU_array[au_index[i]]) for i in xrange(len(the_DQ_array))])
	theory_du_2band_array.append([ hp.anafast(the_DU_array[i], map2=the_aQ_array[au_index[i]]) for i in xrange(len(the_DQ_array))])

	#Now Noise Realization
	
	alpha_radio = hp.ma(alpha_radio)
	sigma_alpha = hp.ma(sigma_alpha)
        alpha_radio.mask=mask_bool
        sigma_alpha.mask=mask_bool
	tmp1=[]
	tmp2=[]
	for i in xrange(len(sigma_array)-1):
		for j in xrange(i+1,len(sigma_array)):	
			tmp1.append( [ ((sigma_array[i][1]-sigma_array[j][1])**2   ).sum() *(pix_area/(2*(wl[i]**2-wl[j]**2) ))**2/(4*np.pi) ] ) 
			tmp2.append( [ ((sigma_array[i][2]-sigma_array[j][2])**2   ).sum() *(pix_area/(2*(wl[i]**2-wl[j]**2) ))**2/(4*np.pi) ] ) 
		
	N_dq_2band_array.append(tmp1)
	N_du_2band_array.append(tmp2)
	N_au_array.append( [ [( (sigma_alpha*iqu_array[i][2] + sigma_array[i][2]*alpha_radio+ sigma_array[i][2]*sigma_alpha)**2 ).sum()*pix_area**2/(4*np.pi) ] for i in au_index ] )
	N_aq_array.append( [ [( (sigma_alpha*iqu_array[i][1] + sigma_array[i][1]*alpha_radio+ sigma_array[i][1]*sigma_alpha)**2).sum()*pix_area**2/(4*np.pi) ] for i in au_index ] )
	N_au_array_fr.append( [ [( (sigma_alpha*iqu_array_fr[i][2] + sigma_array[i][2]*alpha_radio+ sigma_array[i][2]*sigma_alpha)**2 ).sum()*pix_area**2/(4*np.pi) ] for i in au_index ] )
	N_aq_array_fr.append( [ [( (sigma_alpha*iqu_array_fr[i][1] + sigma_array[i][1]*alpha_radio+ sigma_array[i][1]*sigma_alpha)**2).sum()*pix_area**2/(4*np.pi) ] for i in au_index ] )


#Save raw data
np.savez(survey+'_simulation_array_{0:0>2d}'.format(bins)+'.npz', c2b=cross_2band_array,c2b_dq=cross_dq_2band_array,c2b_du=cross_du_2band_array,t2b=theory_2band_array,t2b_dq=theory_dq_2band_array,t2b_du=theory_du_2band_array ,ndq=N_dq_2band_array,ndu=N_du_2band_array,nau=N_au_array,naq=N_aq_array, c2bfr=cross_2band_array_fr, c2b_dqfr=cross_dq_2band_array_fr,c2b_dufr=cross_du_2band_array_fr,naufr=N_au_array_fr,naqfr=N_aq_array_fr)

##Add corrections for sky cut
cross_2band_array=np.array(cross_2band_array).swapaxes(0,1).tolist()
cross_dq_2band_array=np.array(cross_dq_2band_array).swapaxes(0,1).tolist()
cross_du_2band_array=np.array(cross_du_2band_array).swapaxes(0,1).tolist()

cross_2band_array_fr=np.array(cross_2band_array_fr).swapaxes(0,1).tolist()
cross_dq_2band_array_fr=np.array(cross_dq_2band_array_fr).swapaxes(0,1).tolist()
cross_du_2band_array_fr=np.array(cross_du_2band_array_fr).swapaxes(0,1).tolist()


theory_2band_array=np.array(theory_2band_array).swapaxes(0,1).tolist()
theory_dq_2band_array=np.array(theory_dq_2band_array).swapaxes(0,1).tolist()
theory_du_2band_array=np.array(theory_du_2band_array).swapaxes(0,1).tolist()

N_dq_2band_array=np.array(N_dq_2band_array).swapaxes(0,1).tolist()
N_du_2band_array=np.array(N_du_2band_array).swapaxes(0,1).tolist()

N_au_array=np.array(N_au_array).swapaxes(0,1).tolist()
N_aq_array=np.array(N_aq_array).swapaxes(0,1).tolist()

N_au_array_fr=np.array(N_au_array_fr).swapaxes(0,1).tolist()
N_aq_array_fr=np.array(N_aq_array_fr).swapaxes(0,1).tolist()


##Add Noies sky Corrections
#N_dq_2band_array*=1./fsky*np.mean(1./bls)
#N_du_2band_array*=1./fsky*np.mean(1./bls)
#
#N_au_array*=1./fsky*np.mean(1./bls)
#N_aq_array*=1./fsky*np.mean(1./bls)

plot_l=[]
#factor accounts of l(l+1), fksy fraction and beam
fact=ll/fsky/bls
fact_n=ll/fsky
for m in xrange(len(cross_2band_array)):
	cross_2band_array[m]=bin_llcl.bin_llcl(fact*cross_2band_array[m],bins)['llcl']

	cross_dq_2band_array[m]=bin_llcl.bin_llcl(fact*cross_dq_2band_array[m],bins)['llcl']
	cross_du_2band_array[m]=bin_llcl.bin_llcl(fact*cross_du_2band_array[m],bins)['llcl']

	cross_2band_array_fr[m]=bin_llcl.bin_llcl(fact*cross_2band_array_fr[m],bins)['llcl']

	cross_dq_2band_array_fr[m]=bin_llcl.bin_llcl(fact*cross_dq_2band_array_fr[m],bins)['llcl']
	cross_du_2band_array_fr[m]=bin_llcl.bin_llcl(fact*cross_du_2band_array_fr[m],bins)['llcl']


	theory_2band_array[m]=bin_llcl.bin_llcl(fact*theory_2band_array[m],bins)['llcl']
	theory_dq_2band_array[m]=bin_llcl.bin_llcl(fact*theory_dq_2band_array[m],bins)['llcl']

	N_au_array[m]=bin_llcl.bin_llcl(fact_n*N_au_array[m],bins)['llcl']
	N_aq_array[m]=bin_llcl.bin_llcl(fact_n*N_aq_array[m],bins)['llcl']
	
	N_au_array_fr[m]=bin_llcl.bin_llcl(fact_n*N_au_array_fr[m],bins)['llcl']
	N_aq_array_fr[m]=bin_llcl.bin_llcl(fact_n*N_aq_array_fr[m],bins)['llcl']

	N_dq_2band_array[m]=bin_llcl.bin_llcl(fact_n*N_dq_2band_array[m],bins)['llcl']
	N_du_2band_array[m]=bin_llcl.bin_llcl(fact_n*N_du_2band_array[m],bins)['llcl']

	if m== len(cross_2band_array) -1:
		tmp=bin_llcl.bin_llcl(fact*theory_du_2band_array[m],bins)
		theory_du_2band_array[m]=tmp['llcl']
		plot_l=tmp['l_out']
	
	else:
		theory_du_2band_array[m]=bin_llcl.bin_llcl(fact*theory_du_2band_array[m],bins)['llcl']



cross_2band=np.mean(cross_2band_array,axis=1)
cross_dq_2band=np.mean(cross_dq_2band_array,axis=1)
cross_du_2band=np.mean(cross_du_2band_array,axis=1)

cross_2band_fr=np.mean(cross_2band_array_fr,axis=1)
cross_dq_2band_fr=np.mean(cross_dq_2band_array_fr,axis=1)
cross_du_2band_fr=np.mean(cross_du_2band_array_fr,axis=1)

theory_2band=np.mean(theory_2band_array,axis=1)
theory_dq_2band=np.mean(theory_dq_2band_array,axis=1)
theory_du_2band=np.mean(theory_du_2band_array,axis=1)

N_au=np.mean(N_au_array,axis=1)
N_aq=np.mean(N_aq_array,axis=1)

N_au_fr=np.mean(N_au_array_fr,axis=1)
N_aq_fr=np.mean(N_aq_array_fr,axis=1)

N_du_2band=np.mean(N_du_2band_array,axis=1)
N_dq_2band=np.mean(N_dq_2band_array,axis=1)

#STD over realizations
dcross_2band=np.sqrt(np.var(cross_du_2band_array,ddof=1,axis=1) + np.var(cross_dq_2band_array,ddof=1,axis=1)) 
dcross_2band_fr=np.sqrt(np.var(cross_du_2band_array_fr,ddof=1,axis=1) + np.var(cross_dq_2band_array_fr,ddof=1,axis=1)) 

#create variance for theory cls

cosmic_2band=abs(theory_2band)*np.sqrt(2./((2*plot_l+1)*fsky*np.sqrt(bins**2 +dl_eff**2)))


#Variance for cross correlation
sigma1_2band=np.sqrt( 2.*( (theory_dq_2band)**2  + (theory_dq_2band) * (N_dq_2band + N_au)/2. + (N_dq_2band*N_au)/2.)/((2.*plot_l+1.)*np.sqrt(bins**2 +dl_eff**2)*fsky))
sigma2_2band=np.sqrt( 2.*( (theory_du_2band)**2  + (theory_du_2band) * (N_du_2band + N_aq)/2. + (N_du_2band*N_aq)/2.)/((2.*plot_l+1.)*np.sqrt(bins**2 +dl_eff**2)*fsky))
sigma_2band=np.sqrt(sigma1_2band**2 + sigma2_2band**2)

sigma1_2band_fr=np.sqrt( 2.*( (theory_dq_2band)**2  + (theory_dq_2band) * (N_dq_2band + N_au_fr)/2. + (N_dq_2band*N_au_fr)/2.)/((2.*plot_l+1.)*np.sqrt(bins**2 +dl_eff**2)*fsky))
sigma2_2band_fr=np.sqrt( 2.*( (theory_du_2band)**2  + (theory_du_2band) * (N_du_2band + N_aq_fr)/2. + (N_du_2band*N_aq_fr)/2.)/((2.*plot_l+1.)*np.sqrt(bins**2 +dl_eff**2)*fsky))
sigma_2band_fr=np.sqrt(sigma1_2band_fr**2 + sigma2_2band_fr**2)

#average over band selection

theory_2band=np.average(theory_2band,axis=0,weights=1./cosmic_2band**2)

cosmic_2band=np.sqrt(1./np.average(1./cosmic_2band**2,axis=0))
#cosmic_2band=np.sqrt(np.average(cosmic_2band_array**2,axis=0,weights=1./cosmic_2band_array**2))



cross_2band=np.average(cross_2band,axis=0,weights=1./sigma_2band**2)
cross_2band_fr=np.average(cross_2band_fr,axis=0,weights=1./sigma_2band_fr**2)

#dcross_2band=np.sqrt(1./np.average(1./dcross_2band**2,axis=0))
#
#sigma_2band=np.sqrt(1./np.average(1./sigma_2band**2,axis=0))

dcross_2band=np.sqrt(np.average(dcross_2band**2,axis=0,weights=1./dcross_2band**2))

sigma_2band=np.sqrt(np.average(sigma_2band**2,axis=0,weights=1./sigma_2band**2))

dcross_2band_fr=np.sqrt(np.average(dcross_2band_fr**2,axis=0,weights=1./dcross_2band_fr**2))

sigma_2band_fr=np.sqrt(np.average(sigma_2band_fr**2,axis=0,weights=1./sigma_2band_fr**2))
#ipdb.set_trace()
np.savez(survey+'_simulation_{0:0>2d}'.format(bins)+'.npz', c2b=cross_2band, dc2b=dcross_2band,s2b=sigma_2band,the2b=theory_2band,cos2b=cosmic_2band,c2bfr=cross_2band_fr,dc2bfr=dcross_2band_fr,s2bfr=sigma_2band_fr)
f=np.load(survey+'_simulation_{0:0>2d}'.format(bins)+'.npz')
for key in f.keys():
	np.savetxt( survey+'_simulation_'+key+'_'+'.txt',f[key],fmt='%.8e')
f.close()

likelihood(cross_2band[1:],dcross_2band[1:],theory_2band[1:],survey,title='c2b')
likelihood(cross_2band_fr[1:],dcross_2band_fr[1:],theory_2band[1:],survey,title='c2bfr')

##Plot 2 Band
fig, ax=plt.subplots(1)

ax.plot(plot_l,theory_2band*1e12,'-r')
ax.fill_between(plot_l,(theory_2band-cosmic_2band)*1e12,(theory_2band+cosmic_2band)*1e12,color='red',alpha=.5)

ax.errorbar(plot_l,cross_2band*1e12,dcross_2band*1e12,fmt='dk',alpha=.5)
ax.fill_between(plot_l,(cross_2band-sigma_2band)*1e12,(cross_2band+sigma_2band)*1e12,color='gray',alpha=.3)

ax.set_title(survey+' FR Cross Correlation')
ax.set_ylabel('$\\frac{\ell (\ell +1)}{2 \pi} C_{\ell} \\left( \\frac{\mu K}{m^{2}} \\right)^{2}$')
ax.set_xlabel('$\ell$')
ax.set_xlim([0,375])
#ax.set_ybound([1e7,-1e7])
fig.subplots_adjust(right=.95,left=.15)

fig.savefig(survey+'_simulation_2band_linear_{0:2d}'.format(bins)+'.png',format='png')
ax.set_yscale('log')
fig.savefig(survey+'_simulation_2band_log_{0:2d}'.format(bins)+'.png',format='png')
plt.close()

##Plot 2 Band Foreground Reduced
fig, ax=plt.subplots(1)

ax.plot(plot_l,theory_2band*1e12,'-r')
ax.fill_between(plot_l,(theory_2band-cosmic_2band)*1e12,(theory_2band+cosmic_2band)*1e12,color='red',alpha=.5)

ax.errorbar(plot_l,cross_2band_fr*1e12,dcross_2band_fr*1e12,fmt='dk',alpha=.5)
ax.fill_between(plot_l,(cross_2band_fr-sigma_2band_fr)*1e12,(cross_2band_fr+sigma_2band_fr)*1e12,color='gray',alpha=.3)

ax.set_title(survey+' FR Cross Correlation')
ax.set_ylabel('$\\frac{\ell (\ell +1)}{2 \pi} C_{\ell} \\left( \\frac{\mu K}{m^{2}} \\right)^{2}$')
ax.set_xlabel('$\ell$')
ax.set_xlim([0,375])
#ax.set_ybound([1e7,-1e7])
fig.subplots_adjust(right=.95,left=.15)

fig.savefig(survey+'_simulation_2band_fr_linear_{0:2d}'.format(bins)+'.png',format='png')
ax.set_yscale('log')
fig.savefig(survey+'_simulation_2band_fr_log_{0:2d}'.format(bins)+'.png',format='png')
plt.close()
