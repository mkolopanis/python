def alpha_function(P, x,y,err):
	import numpy
	Q= numpy.array(P[0] + 2*P[2]*x**2*P[1])*1e-5
	U= numpy.array(P[1] - 2*P[2]*x**2*P[0])*1e-5
	small_angle=(numpy.sin(2*P[2]*x**2)-2*P[2]*X**2)/numpy.sin(2*P[2]*x**2)

	return [(y[:,0]-Q)/err[:,0],(y[:,1]-U)/err[:,1],small_angle/1.]

