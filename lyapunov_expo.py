#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import numpy as np
import random as rn
import math as mt
import pylab as pl

import numpy as np
import random as rn
import math as mt
import pylab as pl

r_range = np.linspace(3,4,1000)
Ly_Ex = np.zeros( len(r_range))

f = lambda x,r : r*x*(1-x)
g = lambda x,r : mt.log(abs(r-2*r*x))

N_pre = 300
N = 2000

n = 0

for r in r_range :
	x_init = rn.random()
	summ = 0

	for i in range(0,N):
		x_old = x_init
		# Logistic Map
		x_init = f(x_old,r)

		if i > N_pre:
			summ = summ + g(x_init,r)

	summ = float(summ)/N
	Ly_Ex[n] = summ
	n = n+1

pl.xlabel('r', fontsize = 20)
pl.ylabel('Lyapunov Exponent, $\lambda$', fontsize =20)
pl.axis((r_range[0], r_range[-1],-1,1))
pl.plot(r_range , Ly_Ex, 'r')
pl.show()	
