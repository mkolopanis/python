import numpy as np
import healpy as hp
from time import time
from rotate_tqu import rotate_tqu
npix=hp.nside2npix(1024)
this=np.random.randint(1,10,(3,npix))
stuff=np.random.randint(2,200,npix)

t0=time()
new=rotate_tqu(this,.013,stuff)
final=time()-t0
print "this took "+"{:.4f}".format(final/60.)+" minutes"
