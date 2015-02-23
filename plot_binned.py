import numpy as np
import bin_llcl
import matplotlib.pyplot as plt
import ipdb
def plotBinned(cls_in,dcls_in,l_out,bins,output_prefix,title=None,theory=None,dtheory=None,delta=None,cosmic=None):
	plt.figure()
	plt.clf()
	if not (theory is None) :
		plt.plot(l_out,theory,'r-')	
	if not (dtheory is None) :
		plt.fill_between(l_out,(theory-dtheory),(theory+dtheory),alpha=.5,facecolor='red')
	if not (cosmic is None) :
		plt.errorbar(l_out,theory,yerr=dtheory,color='red')
	plt.errorbar(l_out,cls_in,yerr=dcls_in,xerr=bins,color='black',fmt='k.',linestyle='None')
	if not (delta is None) :
		plt.fill_between(l_out,cls_in-delta,cls_in+delta,color='gray',alpha=0.5)
	plt.xlim([0,np.max(l_out+bins)])
	#plt.ylim([np.min(b_cl['llcl']-b_dcl['llcl']),np.max(b_cl['llcl']+b_dcl['llcl'])])
	plt.xlabel('$\ell$')
	plt.ylabel('$\\frac{\ell(\ell+1)}{2\pi}C_{\ell}\ \\frac{\mu K^{2}}{m^{4}}$')
	if title:
		plt.title(title)
	else:
		plt.title('Binned Cls {:02d}'.format(bins))
	plt.savefig(output_prefix+'_{:02d}'.format(bins))
	plt.savefig(output_prefix+'_{:02d}.png'.format(bins),format='png')
