import numpy as np
from faraday_correlate_quiet import faraday_correlate_quiet
import healpy as hp
from astropy.io import fits
import matplotlib.pyplot as plt

map_prefix='/home/matt/quiet/quiet_maps/'
i_file=map_prefix+'quiet_simulated_43.1_cmb1.fits'
j_file=map_prefix+'quiet_simulated_94.5_cmb1.fits'
alpha_file='/data/wmap/faraday_MW_realdata.fits'
bands=[43.1,94.5]
names=['43','95']
wl=np.array([299792458./(band*1e9) for band in bands])


cross1,cross2,cross3=faraday_correlate_quiet(i_file,j_file,wl[0],wl[1],alpha_file,names[0]+'x'+names[1])

l=np.arange(len(cross1))
ll=np.array([i*(i+1)/(2*np.pi) for i in l])
plt.plot(l,ll*cross1*1e12)
plt.show()
