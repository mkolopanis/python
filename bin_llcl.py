import numpy as np
def bin_llcl(llcl_in,ubin,uniform=False,flatten=False):
	"""Bins Continuous Angular Power Spectrum into discrete 
	    bins of even or irregularly spaced sizes
	
	Parameters
	---------
	llcl_in: float, array-like shape (lmax,) or (Nspec,lmax)
		Spectra are assumed to be multiplied by l*(l+1)/(2 * pi).
		Either an singular continuous angular power spectrum,
		or a sequence of Nspec continuous angular power spectrum.
	ubin: int, scalar or array
		 bin size of output power spectrum
		 Accepts integer argument for constant l-space bins or array 
		 of irregularly sized bins
	uniform: bool, scalar, optional
		 if True uses uniform weighting of Cls (instead of 2*l+1)
		 default = False
	flatten: bool, scalar, optional
		if True multiplies input power spectrum by l*(l+1)/(2 * pi)
		default = False


	Returns
	-------
	bn_spec: dictionary of arrays or dictionary of sequence of arrays
	    keys:
		l_out: int, array, (nbins,)
		    l values of binned power spectra
		
		deltal: int, arrray, (nbins,)
		    size of l-bins		

	        llcl: float, array or sequence of arrays, (nbins,) or (nspec,nbins)
		    Binned Angular Power spectrum

		dllcl: float, array or sequence of arrays, (nbins,) or (nspec,nbins)
		    Cosmic Variance of Binned Angular Power Spectrum

		std_llcl: float, array or sequence of arrays, (nbins,) or (nspec,nbins)
		    Standard Deviation of angular power spectrum in l-bin

	"""




	if np.min(ubin) < 0:
		print 'bins must be positive'
		return
	bins=ubin
	if isinstance(bins,int):
		nb=1
	else:
		nb=len(bins)
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

	if len(np.shape(llcl_in)) == 1: llcl_in = [ llcl_in ]
	nspec=np.shape(llcl_in)[0]
	lmax_in=len(llcl_in[0])-1
	llcl_in = np.swapaxes(llcl_in,0,1).tolist()
	if nb == 1: ##regular binning
		nbins=(lmax_in)/long(bins)
		lmax=nbins*bins - 1
		l=np.arange(lmax+1)
		w=2*l+1
		if uniform:
			w=np.repeat(1,lmax+1)
		y=np.array(llcl_in[:lmax+1]).T
		if flatten:
			y*=(l*(l+1)/(2*np.pi))
		w1=np.reshape(w,(nbins,bins))
		y1=np.reshape(y*w,(nspec,nbins,bins))
		l1=np.reshape(l,(nbins,bins))
		n1=np.tile(1,(nbins,bins))
		y2=np.reshape(y,(nspec,nbins,bins))
		
		llcl_out=np.sum(y1,-1)/np.sum(w1,-1)
		#llcl_out.insert(0,np.sum(y[:bins/2]*w[:bins/2])/np.sum(w[:bins/2]))	
		std_llcl=np.std(y2,-1)
		#std_llcl.insert(0,np.std(y[:bins/2]))
		l_out = np.sum(l1,-1)/np.sum(n1,-1)
		#l_out.insert(0,np.mean(l[:bins/2]))
		#llcl_out=np.array(llcl_out)
		#l_out=np.array(l_out)
		#std_llcl=np.array(std_llcl)
		dl=np.full((nspec,nbins),bins)
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
		llcl_out=np.zeros(nbins,nspec)
		std_llcl=np.zeros(nbins,nspec)
		dl=np.zeros(nbins,npsec)
		for i in xrange(nbins):
			l_out[i] = np.mean(l[bins[i]:bins[i+1] ])
			dl[i]=bins[i+1]-bins[i]
			llcl_out[i] = np.sum((y*w)[bins[i]:bins[i+1]])/np.sum(w[bins[i]:bins[i+1]])
			std_llcl[i] = np.std(y[bins[i]:bins[i+1]])
	dllcl=abs(llcl_out)*np.sqrt(2./(2*l_out+1)/dl)
	deltal=dl

	if nspec==1:
		llcl_out = llcl_out.flatten()
		dllcl = dllcl.flatten()
		deltal = deltal.flatten()
		std_llcl = std_llcl.flatten()

	bn_spec = {'l_out':l_out,'deltal':deltal,'llcl':llcl_out,'dllcl':dllcl,'std_llcl':std_llcl}

	return  bn_spec
