import numpy as np
import healpy as hp
def rotate_tqu(map_in,wl,alpha):  #rotates tqu map by phase
	npix=hp.nside2npix(hp.get_nside(map_in))
	tmp_map=np.zeros((3,npix))
	tmp_map[0]=map_in[0]
	tmp_map[1]=np.array([map_in[1][x]*np.cos(2*alpha[x]*wl**2) + map_in[2][x]*np.sin(2*alpha[x]*wl**2) for x in xrange(npix) ])
	tmp_map[2]=np.array([map_in[2][x]*np.cos(2*alpha[x]*wl**2) - map_in[1][x]*np.sin(2*alpha[x]*wl**2) for x in xrange(npix) ])
	return tmp_map
