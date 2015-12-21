import numpy as np
import bin_llcl
import matplotlib.pyplot as plt
import ipdb

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def plotBinned(cls_in,dcls_in,l_out,bins,output_prefix,title=None,theory=None,dtheory=None,delta=None,cosmic=None):
	fig=plt.figure(1)
	plt.clf()
        ax=fig.add_subplot(111)
        good_l=np.logical_and( l_out > 25 , l_out <= 250)

	if not (theory is None) :
		ax.plot(l_out,theory,'r-')
	if not (cosmic is None) :
		ax.fill_between(l_out,(theory-cosmic),(theory+cosmic),alpha=.5,facecolor='red')
		#plt.fill_between(l_out,(theory-dtheory),(theory+dtheory),alpha=.5,facecolor='red')
	if not (dtheory is None) :
		ax.errorbar(l_out,theory,yerr=dtheory,color='red')
	ax.errorbar(l_out,cls_in,yerr=dcls_in,color='black',fmt='k.',linestyle='None')
	if not (delta is None) :
		ax.fill_between(l_out,cls_in-delta,cls_in+delta,color='gray',alpha=0.5)
	#plt.xlim([0,np.max(l_out+bins)])
	#plt.ylim([np.min(b_cl['llcl']-b_dcl['llcl']),np.max(b_cl['llcl']+b_dcl['llcl'])])
	ax.set_xlabel('$\ell$')
	ax.set_ylabel('$\\frac{\ell(\ell+1)}{2\pi}C_{\ell}\ \\frac{\mu K^{2}}{m^{4}}$')
        #ax.set_xlim([0,np.max(l_out+bins)])
#        ax.autoscale(axis='y',tight=True)

	if title:
		ax.set_title(title)
	else:
		ax.set_title('Binned Cls {:02d}'.format(bins))

        axins = inset_axes(ax,width="50%",height="30%",loc=9)

	if not (theory is None) :
		axins.plot(l_out[good_l],theory[good_l],'r-')
	if not (cosmic is None) :
		axins.fill_between(l_out[good_l],(theory-cosmic)[good_l],(theory+cosmic)[good_l],alpha=.5,facecolor='red')
		#plt.fill_between(l_out,(theory-dtheory),(theory+dtheory),alpha=.5,facecolor='red')
	if not (dtheory is None) :
		axins.errorbar(l_out[good_l],theory[good_l],yerr=dtheory[good_l],color='red')
	axins.errorbar(l_out[good_l],cls_in[good_l],yerr=dcls_in[good_l],color='black',fmt='k.',linestyle='None')
	if not (delta is None) :
		axins.fill_between(l_out[good_l],(cls_in-delta)[good_l],(cls_in+delta)[good_l],color='gray',alpha=0.5)
        axins.set_xlim(25,255)
        axins.set_yscale('log')
        #axins.set_ylim(ymax=1e5)
        #axins.set_ylim(-1e5,1e5)

        mark_inset(ax,axins,loc1=2,loc2=4,fc='none',ec='0.1')
        #axins.set_xlim([25,250])
        axins.autoscale('y',tight=True)


        plt.draw()
        #plt.show(block=True)
	fig.savefig(output_prefix+'_linear_{:02d}.eps'.format(bins),format='eps')
	fig.savefig(output_prefix+'_linear_{:02d}.png'.format(bins),format='png')
        ax.set_yscale('log')
	fig.savefig(output_prefix+'_log_{:02d}.eps'.format(bins),format='eps')
	fig.savefig(output_prefix+'_log_{:02d}.png'.format(bins),format='png')
