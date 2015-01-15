import numpy as np
import ipdb 
def bin_llcl(llcl_in,ubin,uniform=False,flatten=False):
	if np.min(ubin) < 0:
		print 'bins must be positive'
		return
	bins=ubin
	if isinstance(bins,int):
		nb=1
	else:
		nb=len(bins)
	if len(llcl_in) <= 6:
		lmax_in=len(llcl_in[0])-1
	else:
		lmax_in=len(llcl_in)-1
	if (nb > 1):
		k=np.where(bins <= (lmax_in +1))[0]
		if ( len(k) >= (nb-1)):
			bins=bins[bins < (lmax_in+1)]
			nb=len(bins)
		if (len(k) < (nb -1)):
			bins = np.append(bins[k],lmax_in+1)
			nb =len(k) +1
			print 'BIN_LLCL: too many bins'
			print '{} bins avaiable {} bins given'.format(len(k),nb)
	if nb == 1: ##regular binning
		nbins=(lmax_in-bins/2)/long(bins)
		lmax=nbins*bins+bins/2
		l=np.arange(lmax)
		w=2*l+1
		if uniform:
			w=np.repeat(1,lmax)
		y=llcl_in[:lmax]
		if flatten:
			y*=(l*(l+1)/(2*np.pi))
		w1=np.reshape(w[bins/2:],(nbins,bins))
		y1=np.reshape(y[bins/2:]*w[bins/2:],(nbins,bins))
		l1=np.reshape(l[bins/2:],(nbins,bins))
		n1=np.tile(1,(nbins,bins))
		y2=np.reshape(y[bins/2:],(nbins,bins))
		
		llcl_out=(np.sum(y1,1)/np.sum(w1,1)).tolist()
		llcl_out.insert(0,np.sum(y[:bins/2]*w[:bins/2])/np.sum(w[:bins/2]))	
		std_llcl=(np.std(y2,1)).tolist()
		std_llcl.insert(0,np.std(y[:bins/2]))
		l_out = (np.sum(l1,1)/np.sum(n1,1)).tolist()
		l_out.insert(0,np.mean(l[:bins/2]))
		llcl_out=np.array(llcl_out)
		l_out=np.array(l_out)
		std_llcl=np.array(std_llcl)
		dl=np.repeat(bins,nbins+1)
	else:	##irregular binning
		lmax=np.max(bins) -1
		nbins = nb-1
		good=np.where(bins < lmax)[0]
		ng=len(good)
		#bins=bins[good]
		#nbins=len(bins)-1
		if ng == 0:
			print 'l-range of binning does not intersect that of data'
			return
		l=np.arange(lmax)
		n1=np.repeat(1,lmax)
		w=2*l+1
		if uniform:
			w=np.repeat(1,lmax)
		y=llcl_in[:np.ceil(lmax)]
		if flatten:
			y*=(l*(l+1)/(2*np.pi))
		l_out=np.zeros(nbins)
		llcl_out=np.zeros(nbins)
		std_llcl=np.zeros(nbins)
		dl=np.zeros(nbins)
		for i in xrange(nbins):
			l_out[i] = np.mean(l[bins[i]:bins[i+1] ])
			dl[i]=bins[i+1]-bins[i]
			llcl_out[i] = np.sum((y*w)[bins[i]:bins[i+1]])/np.sum(w[bins[i]:bins[i+1]])
			std_llcl[i] = np.std(y[bins[i]:bins[i+1]])
	dllcl=abs(llcl_out)*np.sqrt(2./(2*l_out+1)/dl)
	deltal=dl
	return {'llcl':llcl_out,'l_out':l_out,'dllcl':dllcl,'deltal':deltal,'std_llcl':std_llcl}

