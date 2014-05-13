#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl


r_range = np.linspace(0,4,500)

N_pre = 400
N = 10
x = np.zeros(( N, 1 ))
x_init = 0.5


f = lambda x,r : r*x*(1-x)

for r in r_range:
	x[0,0] = x_init
	
	for ite in range(0,N_pre+1):
		x[0,0] = f(x[0,0] , r)
		
		for i in range(0,N-1):
			x[i+1,0] = f(x[i,0] , r) 
	
	pl.scatter(r*np.ones((N,1)), x, 0.2, color='blue')

pl.xlabel('r', fontsize = 20)
pl.ylabel('$x_n$', fontsize =20)
pl.axis((r_range[0],r_range[-1],0,1))
pl.show()
