import numpy as np
import bin_llcl
import matplotlib.pyplot as plt

def plotBinned(cls_in,dcls_in,bins,output_prefix,title=None,theory=None):
	l=np.arange(len(cls_in))
	ll=l*(l+1)/(2*np.pi)
	for b in bins:
		b_cl=bin_llcl.bin_llcl(ll*cls_in,b)	
		b_dcl=bin_llcl.bin_llcl(ll*dcls_in,b)
		plt.figure()
		plt.clf()
		if not (theory is None) :
			plt.plot(l,ll*theory,'r-')	
		plt.errorbar(b_cl['l_out'],b_cl['llcl'],yerr=b_dcl['llcl']/np.sqrt(b),xerr=b_cl['deltal']/np.sqrt(b),color='black')
		plt.plot(b_cl['l_out'],b_cl['llcl'],'k.')
		plt.xlim([0,np.max(b_cl['l_out']+b_cl['deltal'])])
		#plt.ylim([np.min(b_cl['llcl']-b_dcl['llcl']),np.max(b_cl['llcl']+b_dcl['llcl'])])
		plt.xlabel('$\ell$')
		plt.ylabel('$\\frac{\ell(\ell+1)}{2\pi}C_{\ell}\ \\frac{\mu K^{2}rad}{m^{4}}$')
		if title:
			plt.title(title)
		else:
			plt.title('Binned Cls {:02d}'.format(b))
		plt.savefig(output_prefix+'_{:02d}'.format(b))
		plt.savefig(output_prefix+'_{:02d}.png'.format(b),format='png')
