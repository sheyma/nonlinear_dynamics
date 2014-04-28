#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# Exercise 1.4

import scipy.integrate as integ
import numpy as np
import pylab as pl
from pylab import *

# parameters of Lotka-Volterra Model
params = { 
			'r' : 0.1 ,
			'alpha' : 0.01,
			'q' : 0.5,
			'beta': 0.01
			}

# time of evolution
t=np.linspace(0,1000,1000)


def pop(x, t):
     dxdt=np.zeros_like(x)
     dxdt[0]=params['r']*x[0] - params['alpha']*x[0]*x[1]
     dxdt[1]= -params['q']*x[1] + params['beta']*x[0]*x[1]
     return dxdt

# initial conditions for x0,y0
init_con = [20, 20]

# solving differential equations
sol=integ.odeint(pop,init_con,t)

# fixed points
fix0 = np.array([	0. , 0. ])
fix1 = np.array([float(params['q'])/params['beta'], params['r']/float(params['alpha'])])

## time evolution of x and y
#pl.figure(1)  
#pl.plot(t,sol[:,0],'r-',label='S')
#pl.plot(t,sol[:,1],'b-',label='M')
#pl.xlabel('t')
#pl.ylabel('S , M')
#pl.title('Time Evolution of S and M Populations')
#lg = legend()
#pl.legend()

# phase space of x and y
pl.figure(2) 
#pl.plot(sol[:,0] , sol[:,1],'k', label='X0=(%.f, %.f)' % ( init_con[0], init_con[1]))

values = np.linspace(0.2, 0.9, 5)
vcolors = pl.cm.autumn_r(linspace(0.3, 1., len(values)))
print values
for v, col in zip(values,vcolors):
	new_init_con = float(v)*fix1
	print "colors", col
	print new_init_con
	sol = integ.odeint(pop,new_init_con,t)
	pl.plot(sol[:,0],sol[:,1],lw=3.5*v, color=col, label='X0=(%.f, %.f)' % ( new_init_con[0], new_init_con[1]) )

ymax = pl.ylim(ymin=0)[1]
xmax = pl.xlim(xmin=0)[1]

n_points = 20

y = np.linspace(0, ymax, n_points)
x = np.linspace(0, xmax, n_points)

X, Y = meshgrid( x , y )

# nullclines found from pop - function by setting t=0
dX, dY = pop( [X , Y] , t=0 )

M = (hypot(dX , dY))
M [M==0] = 1.
dX /= M
dY /= M

pl.quiver(X,Y,dX,dY,M, pivot='mid', scale=50, cmap=pl.cm.jet)

# indicate fixpoint on phase-space
pl.plot([fix0[0]],[fix0[1]],'k', markersize = 10, marker='o')
pl.annotate('FP :'+ str(fix0[0])+','+str(fix0[1]), xy=(fix0[0],fix0[1]))

pl.plot([fix1[0]],[fix1[1]],'k', markersize = 8, marker='o')
pl.annotate('FP :'+ str(fix1[0])+','+str(fix1[1]), xy=(fix1[0],fix1[1]))

pl.suptitle('Lotka-Volterra Model in Phase Space', fontweight='bold')
pl.title( 'r = ' +str(params['r'])+ ',' +
	      r'  $\alpha$ = '+str(params['alpha']) + ',' + 
	      '  q = '+ str(params['q']) + ',' +
	       r'  $\beta$ = '+str(params['beta'])   )
pl.xlabel('S')
pl.ylabel('M')

pl.legend()
pl.show()



