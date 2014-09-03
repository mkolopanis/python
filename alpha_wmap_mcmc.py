import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import ipdb; 
from time import time
import pymc
from kabuki import Hierarchical, Knode

def rotate_tqu(map_in,wl,alpha):  #rotates tqu map by phase
	npix=hp.nside2npix(hp.get_nside(map_in))
	tmp_map=np.zeros((3,npix))
	tmp_map[0]=map_in[0]
	tmp_map[1]=map_in[1]*np.cos(2*alpha*wl**2) + map_in[2]*np.sin(2*alpha*wl**2)
	tmp_map[2]=-map_in[1]*np.sin(2*alpha*wl**2) + map_in[2]*np.cos(2*alpha*wl**2)
	return tmp_map


t4=time()
radio_file='/data/wmap/faraday_MW_realdata.fits'
cl_file='/home/matt/wmap/simul_scalCls.fits'
nside=128
npix=hp.nside2npix(nside)


bands_p=[30,44,70,100,143,217,353]
wl_p=np.array([299792458/(band*1e9) for band in bands_p])
num_wl=len(wl_p)
beamfwhm = np.array([33.,24.,14.,9.5,7.1,5.0,5.0])
pixarea = np.array([(4*np.pi/x)*(180./np.pi*60.)**2 for x in [hp.nside2npix(1024),hp.nside2npix(2048)]])

beamscale =np.zeros(num_wl)
for i in range(3):
	beamscale[i]=beamfwhm[i]/np.sqrt(pixarea[0])
for i in range(3,7):
	beamscale[i]=beamfwhm[i]/np.sqrt(pixarea[1])

tmp_noise=[2.0,2,7,4.7,2.5,2.2,4.8,14.7]
noise_const_t=np.array([tmp_noise[x]/beamscale[x] for x in range(num_wl)])*2.725e-6
tmp_noise=[2.8,3.9,6.7,4.0,4.2,9.8,29.8]
noise_const_q=np.array([tmp_noise[x]/beamscale[x] for x in range(num_wl)])*2.725e-6

q_array_1=np.zeros((num_wl,npix))
u_array_1=np.zeros((num_wl,npix))
sigma_q_1=np.zeros((num_wl,npix))
sigma_u_1=np.zeros((num_wl,npix))


cls=hp.read_cl(cl_file)
simul_cmb=hp.sphtfunc.synfast(cls,2048,fwhm=5.*np.pi/(180.*60.),new=1,pol=1);
simul_cmb=hp.reorder(simul_cmb,r2n=1);

alpha_radio=hp.read_map(radio_file,hdu='maps/phi');
alpha_radio=hp.ud_grade(alpha_radio,nside_out=2048,order_in='ring',order_out='nested')

npix1=hp.nside2npix(2048)
t2=time()

for i in range(num_wl):
	if i in range(3):
		npix1=hp.nside2npix(1024)
	if i==3:
		npix1=hp.nside2npix(2048)

	tmp_cmb=rotate_tqu(simul_cmb,wl_p[i],alpha_radio);
	tmp_out=hp.ud_grade(tmp_cmb[1],nside_out=nside,order_in='nested',order_out='nested');
	q_array_1[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
	tmp_out=hp.ud_grade(tmp_cmb[2],nside_out=nside,order_in='nested',order_out='nested');
	u_array_1[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
	tmp_out=hp.ud_grade(np.random.normal(0,1,npix1)*noise_const_q[i],nside_out=nside,order_in='nested',order_out='nested');
	sigma_q_1[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
	tmp_out=hp.ud_grade(np.random.normal(0,1,npix1)*noise_const_q[i],nside_out=nside,order_in='nested',order_out='nested');
	sigma_u_1[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
t3=time()
print 'This computation took '+"{:.3f}".format((t3-t2)/60.)+' minutes'
names=['K','Ka','Q','V','W']
bands_w=[23,33,41,61,94]
wl_w=np.array([299792458/(band*1e9) for band in bands_w])
num_wl=len(wl_w)
npix1=hp.nside2npix(512)

map_prefix='/data/wmap/wmap_band_iqumap_r9_9yr_'
simul_prefix='/data/mwap/simul_fr_rotated_'

wmap_files=[ map_prefix+name+'_v5.fits' for name in names]
simul_files=[simul_prefix+str(band).zfill(3)+'.fits' for band in bands_w]

noise_const_t=np.asarray([1.429,1.466,2.188,3.131,6.544])*1e-3
noise_const_q=np.asarray([1.435,1.472,2.197,3.141,6.560])*1e-3

q_array_2=np.zeros((num_wl,npix))
u_array_2=np.zeros((num_wl,npix))
sigma_q_2=np.zeros((num_wl,npix))
sigma_u_2=np.zeros((num_wl,npix))

simul_cmb=hp.ud_grade(simul_cmb,nside_out=512,order_in='nested',order_out='nested')


alpha_radio=hp.ud_grade(alpha_radio,nside_out=512,order_in='nested',order_out='nested');

for i in range(num_wl):
	wmap_counts=hp.read_map(wmap_files[i],nest=1,field=3);
	tmp_cmb=rotate_tqu(simul_cmb,wl_w[i],alpha_radio);
	tmp_out=hp.ud_grade(tmp_cmb[1],nside_out=nside,order_in='nested',order_out='nested');
	q_array_2[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
	tmp_out=hp.ud_grade(tmp_cmb[2],nside_out=nside,order_in='nested',order_out='nested');
	u_array_2[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
	tmp_out=hp.ud_grade(np.random.normal(0,1,npix1)*noise_const_q[i]/np.sqrt(wmap_counts),nside_out=nside,order_in='nested',order_out='nested');
	sigma_q_2[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);
	tmp_out=hp.ud_grade(np.random.normal(0,1,npix1)*noise_const_q[i]/np.sqrt(wmap_counts),nside_out=nside,order_in='nested',order_out='nested');
	sigma_u_2[i]=hp.sphtfunc.smoothing(tmp_out,fwhm=np.pi/180.,lmax=383);


wl=np.concatenate((wl_p,wl_w))

q_array=np.concatenate((q_array_1,q_array_2))
u_array=np.concatenate((u_array_1,u_array_2))
sigma_q=np.concatenate((sigma_q_1,sigma_q_2))
sigma_u=np.concatenate((sigma_u_1,sigma_u_2))

new_index=np.argsort(wl)
new_index=new_index[::-1]
wl=wl[new_index]
nuw_wl=len(new_index)
q_array=q_array[new_index]
u_array=u_array[new_index]
sigma_q=sigma_q[new_index]
sigma_u=sigma_u[new_index]

bands=bands_w+bands_p
bands.sort()
bands.insert(0,'npix')

data_q=[[0 for x in range(13)] for y in range(npix+1)]
data_q[0]=bands

for x in range(1,npix+1):
	data_q[x][0]=x-1
for x in range(1,npix+1):
	for y in range(1,13):
		data_q[x][y]=q_array[y-1,x-1]

data_u=[[0 for x in range(13)] for y in range(npix+1)]
data_u[0]=bands

for x in range(1,npix+1):
	data_u[x][0]=x-1
for x in range(1,npix+1):
	for y in range(1,13):
		data_u[x][y]=q_array[y-1,x-1]


ipdb.set_trace()
t0=time()
parms=np.zeros((4,npix))
dparms=np.zeros((4,npix))
#pymc code will go here

centers=pymc.Normal('centers',mu=0,tau=1./(1)**2,size=(10,2))
taus=1./pymc.Uniform('sigmas',0,1,size=(10,2))**2
alpha_tau=1./pymc.Uniform('sigma_alpha',1,500,size=(10,1))**2
qu_values=pymc.Normal('qu_values',mu=centers,tau=taus,size=(10,2))
alpha_center=pymc.Normal('alpha_center',mu=0,tau=1./(500)*2,size=(10,1))
alpha_rm=pymc.Normal('alpha_rm',mu=alpha_center,tau=alpha_tau,size=(10,1))

@pymc.deterministic
def pol_u(q_values=qu_values[:,0],u_values=qu_values[:,1],alpha_rm=alpha_rm,wl=wl):
	return np.reshape([[(u_values[y]-2*w**2*alpha_rm[y]*q_values[y] )*1e-7 for y in range(10) ] for w in wl],(10,12))


@pymc.deterministic
def pol_q(q_values=qu_values[:,0],u_values=qu_values[:,1],alpha_rm=alpha_rm,wl=wl):
	return np.reshape([[(q_values[y]+2*w**2*alpha_rm[y]*u_values[y] )*1e-7 for y in range(10) ] for w in wl],(10,12))


obs=pymc.Normal('obs',mu=[pol_q,pol_u],tau=[1./sigma_q[:,:10]**2,1./sigma_u[:,:10]**2] ,value=[q_array[:,:10],u_array[:,:10]],observed=True)
mol=pymc.Model({'centers':centers,'taus':taus,'alpha_tau':alpha_tau,'alpha_center':alpha_center,'alpha_rm':alpha_rm,'pol_q':pol_q,'pol_u':pol_u,'qu_values':qu_values})
mcmc=pymc.MCMC(mol)
mcmc.sample(51000,burn=6000)
t1=time()
time_elapsed=t1-t0
total_time=time()-t4
#1
