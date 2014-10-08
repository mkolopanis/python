import numpy as np
import healpy as hp

def pix_rot_q(wl,q_map,u_map,alpha):
	return q_map*np.cos(2*alpha*wl**2) + u_map*np.sin(2*alpha*wl**2)

def pix_rot_u(wl,q_map,u_map,alpha):
	return u_map*np.cos(2*alpha*wl**2) - q_map*np.sin(2*alpha*wl**2)

def rotate_tqu(map_in,wl,alpha):  #rotates tqu map by phase
	npix=hp.nside2npix(hp.get_nside(map_in))
	tmp_map=np.zeros((3,npix))
	tmp_map[0]=map_in[0]
	tmp_map[1]=map_in[1]*np.cos(2*wl**2*alpha)+map_in[2]*np.sin(2*wl**2*alpha) 
	tmp_map[2]=map_in[2]*np.cos(2*wl**2*alpha)-map_in[1]*np.sin(2*wl**2*alpha)
	return tmp_map
