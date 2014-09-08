import healpy as hp
import numpy as np
import astropy as ap
import matplotlib.pyplot as plt
from astropy.io import fits
import ipdb; 
import multiprocessing as mp
import cPickle as pickle 
from time import time
from rotate_tqu import rotate_tqu

def model_qu(parms,wl):
	#npix=hp.nside2npix(hp.get_nside(parms))
	#qu=np.zeros((2,npix))
	qu=np.zeros(2)
	qu[0]=(parms[0]+2*wl**2*parms[2]*parms[1])*1e-6
	qu[1]=(parms[1]-2*wl**2*parms[2]*parms[0])*1e-6
	return qu

def chi_sum(param,q,u,sig_q,sig_u,wl):
	num_wl=len(wl)
	qu_num=[model_qu(param,wl[g]) for g in range(num_wl)]
	return np.sum([np.sum( ((q[h]-qu_num[h][0])/sig_q[h])**2 +((u[h]-qu_num[h][1])/sig_u[h])**2  ) for h in range(num_wl)])

#+ ((2*parms[2]*wl[h]**2-np.sin(2*parms[2]*wl[h]**2))/(np.sin(2*parms[2]*wl[h]**2)))**2

def mcmc_alpha(inputs):
	theta1,q_a,u_a,sq,su,wl=inputs
	runtime=100000
	burntime=1000
	chi1=chi_sum(theta1,q_a,u_a,sq,su,wl)
	save_chi=np.zeros(runtime-burntime)
	theta_array=np.zeros((runtime-burntime,len(theta1)))
	theta2=np.zeros((runtime,len(theta1)))
	theta2[0,:]=theta1[:]
	chi_array=np.zeros(runtime)
	chi_array[0]=chi1
	var2=[.1,.1,1]
	#ipdb.set_trace()
	#deltas=np.zeros((3,runtime))
	#deltas[0]=np.random.uniform(-1,1,runtime)
	#deltas[1]=np.random.uniform(-1,1,runtime)
	#deltas[2]=np.random.uniform(-5,5,runtime)
	for t in range(1,runtime):
		#if ( i > burntime):
			#plt.plot(i-burntime,chi1,'b.')
			#plt.draw()
		proposed=np.zeros(3)
		proposed=theta2[t-1]
		chi1=np.zeros(1)
		chi1=chi_array[t-1]
		for m in range(3):
			proposed[m]=np.random.normal(theta2[t-1,m],var2[m])
			#theta2[0]=theta1[0]+deltas[0,i]
			#theta2[1]=theta1[1]+deltas[1,i]
			#theta2[2]=theta1[2]+deltas[2,i]
			chi2=chi_sum(proposed,q_a,u_a,sq,su,wl)
			ratio=min([0,(chi1-chi2)/2])
			if (np.log(np.random.rand()) < ratio ):
				chi_array[t]=chi2
				chi1=chi2
				theta2[t,m]=proposed[m]	
			else:
				chi_array[t]=chi1
				theta2[t,m]=theta2[t-1,m]
				proposed[m]=theta2[t-1,m]
		if ( t >= burntime):
			save_chi[t-burntime]=chi_array[t]
			theta_array[t-burntime]=theta2[t]
	index=np.argmin(save_chi)
	theta_best=theta_array[index]
	delta_theta=np.std(theta_array,axis=0)
	chi_best=save_chi[index]
	#plt.clf()
	#plt.xlim([0,1000])
	plt.plot(chi_array)
	plt.show(block=False)
	ipdb.set_trace()
	return theta_best,delta_theta,chi_best

names=['K','Ka','Q','V','W']
bands=[23,33,41,61,94]
wl=[299792458/(band*1e9) for band in bands]
num_wl=len(wl)
npix1=hp.nside2npix(512)

map_prefix='/data/wmap/wmap_band_iqumap_r9_9yr_'
simul_prefix='/data/mwap/simul_fr_rotated_'

wmap_files=[ map_prefix+name+'_v5.fits' for name in names]
simul_files=[simul_prefix+str(band).zfill(3)+'.fits' for band in bands]
radio_file='/data/wmap/faraday_MW_realdata.fits'
cl_file='/home/matt/wmap/simul_scalCls.fits'

noise_const_t=np.asarray([1.429,1.466,2.188,3.131,6.544])*1e-3
noise_const_q=np.asarray([1.435,1.472,2.197,3.141,6.560])*1e-3

npix=hp.nside2npix(128)
q_array=np.zeros((num_wl,npix))
u_array=np.zeros((num_wl,npix))
sigma_q=np.zeros((num_wl,npix))
sigma_u=np.zeros((num_wl,npix))

cls=hp.read_cl(cl_file)
simul_cmb=hp.sphtfunc.synfast(cls,512,fwhm=13*np.pi/(180.*60.),new=1,pol=1);
simul_cmb=hp.reorder(simul_cmb,r2n=1);


alpha_radio=hp.read_map(radio_file,hdu='maps/phi');
alpha_radio=hp.ud_grade(alpha_radio,nside_out=512,order_in='ring',order_out='nested');

for i in range(num_wl):
	wmap_counts=hp.read_map(wmap_files[i],nest=1,field=3);
	tmp_cmb=rotate_tqu(simul_cmb,wl[i],alpha_radio);
	tmp_out=hp.ud_grade(tmp_cmb[1],nside_out=128,order_in='nested',order_out='nested');
	q_array[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
	tmp_out=hp.ud_grade(tmp_cmb[2],nside_out=128,order_in='nested',order_out='nested');
	u_array[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
	tmp_out=hp.ud_grade(np.random.normal(0,1,npix1)*noise_const_q[i]/np.sqrt(wmap_counts),nside_out=128,order_in='nested',order_out='nested');
	sigma_q[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
	tmp_out=hp.ud_grade(np.random.normal(0,1,npix1)*noise_const_q[i]/np.sqrt(wmap_counts),nside_out=128,order_in='nested',order_out='nested');
	sigma_u[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);

parms=np.ones((3,npix))
dparms=np.zeros((3,npix))
chis=np.zeros(npix)
#chi1=np.sum([chi_sum(parms1,q_array[i],u_array[i],sigma_q[i],sigma_u[i],wl[i]) for i in range(num_wl)])

#fig=plt.figure()
#plt.xlim([0,1000])
#plt.ion()
#plt.show()
t0=time()
for j in range(5):
#ipdb.set_trace()
#pool=mp.Pool(4)
#outputs=pool.map(mcmc_alpha,[(parms[:,j],q_array[:,j],u_array[:,j],sigma_q[:,j],sigma_u[:,j],wl) for j in range(10)])
#pool.terminate()
#ipdb.set_trace()
#parms,dparm,chi=list(zip(*output))

	parms[:,j],dparms[:,j],chis[j] = mcmc_alpha([parms[:,j],q_array[:,j],u_array[:,j],sigma_q[:,j],sigma_u[:,j],wl])

t1=time()
time_elapsed=t1-t0
parms[:2]*=1e-6
parms[2]
#hp.fitsfunc.mwrfits('info/chi_array.fits',chi_array,colnames='Chi-Sqaured')
#
