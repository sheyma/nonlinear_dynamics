#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import itertools
import random
import math
import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph
import matplotlib.pyplot as pl

B = np.array([[0,1,1,1],[1,0,0,1],[1,0,0,1],[1,1,1,0]])
G = nx.from_numpy_matrix(B)
print "triangles : ", nx.triangles(G)

N = nx.number_of_nodes(G)
#ADJ = {'a':[]}
ADJ = {}
def _triangles_and_degree_iter(G,nodes=None):
	if nodes is None:
		
		for i in G.nodes():
			ADJ[i] = [] 
			for j in G.nodes():
				if G.has_edge(i , j):
					ADJ[i].append(j)
						
		print ADJ
		nodes_nbrs = G.adj.items()
		
		print (nodes_nbrs)
		
	for node_i in ADJ:
		 
		adjacent_i= set(ADJ[node_i]) 
		print adjacent_i
		count_tri = 0
		for node_j in adjacent_i:
			print "benim set : ", node_j
	
	for v,v_nbrs in nodes_nbrs:
		
		
		print "v : ", v
		print "v_nbrs", v_nbrs
		
		
		vs=set(v_nbrs)-set([v])
		print vs
		
		ntriangles=0
		for w in vs:
			print "w is : ", w
			
			ws=set(G[w])-set([w])
			ntriangles+=len(vs.intersection(ws))
		#yield (v,len(vs),ntriangles)
		#print (v,len(vs),ntriangles)


_triangles_and_degree_iter(G, nodes = None)

#def _triangles_and_degree_iter(G,nodes=None):
    #""" Return an iterator of (node, degree, triangles).  

    #This double counts triangles so you may want to divide by 2.
    #See degree() and triangles() for definitions and details.

    #"""
	#if nodes is None:
		#nodes_nbrs = G.adj.items()
	#else:
		#nodes_nbrs= ( (n,G[n]) for n in G.nbunch_iter(nodes) )
		#print nodes_nbrs
		
    ##for v,v_nbrs in nodes_nbrs:
        ##vs=set(v_nbrs)-set([v])
        ##ntriangles=0
        ##for w in vs:
            ##ws=set(G[w])-set([w])
            ##ntriangles+=len(vs.intersection(ws))
        ##yield (v,len(vs),ntriangles)




def plot_graph(G):
	pos = nx.shell_layout(G)
	nx.draw(G, pos)
	pl.show()
	
#plot_graph(G)
