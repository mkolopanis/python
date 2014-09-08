import numpy as np

def alpha_function(P,data):
	wl,q,u,q_err,u_err=data
	Q_mod= np.array([P[0] + 2*P[2]*x**2*P[1] for x in wl])*1e-7
	U_mod= np.array([P[1] - 2*P[2]*x**2*P[0] for x in wl])*1e-7
	small_angle=np.array([(np.sin(2*P[2]*x**2)-2*P[2]*x**2)/np.sin(2*P[2]*x**2) for x in wl])
	return np.concatenate([(q-Q_mod)/q_err,(u-U_mod)/u_err,small_angle/1.])

