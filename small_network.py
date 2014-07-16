#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import itertools
import random
import math
from networkx.generators.classic import empty_graph, path_graph, complete_graph
import matplotlib.pyplot as pl
from matplotlib.pyplot import *

def ring_graph(N , k) :
	# contstructing a ring graph 
	# N : number of nodes
	# k : (even) number of nearest neighbors of any node
	
	if k>=N :
		raise nx.NetworkXError("k must be smaller than N")

	G_ring = nx.Graph()
	
	Nodes = list(range(N)) 
		
	for node_i in range(0, k/2 ):

		# listing available nodes, node_i is at last index 
		# merge nodes with available nodes in pairs
		# construct edges between paired nodes
		
		available_Nodes = Nodes[node_i+1:] + Nodes[0:node_i+1]
		paired_Nodes = zip(Nodes, available_Nodes)
		G_ring.add_edges_from(paired_Nodes)

	# check if the graph does not have self loops			
	for node_i in Nodes:
		if G_ring.has_edge(node_i , node_i):
			raise nx.NetworkXError('This graph has self loops!!')

	return G_ring	
	

def rewiring_edges(G , p ) :
	# returns a graph, does not guarantee connectedness
	# method a : rewiring the existing edges
	# method b : adding new edges instead of rewiring
	# G : a given graph to be manupulated
	# p : probability of rewiring of edges or adding new edges 
	
	Nodes = G.nodes()
	N     = G.number_of_nodes()
	
	values = []
	for node in G:
		values.append(G.degree(node))
	
	ave_degree = float(sum(values)/float(N))	
	
	for i in range(0, int(ave_degree)/2): 
		available_nodes = Nodes[i+1:] + Nodes[0:i+1] 
        
        for node_j, node_k in zip(Nodes,available_nodes):
			if G.has_edge(node_j ,node_k):
				if random.random() < p:		
					node_X = random.choice(Nodes)
					
					# avoid the choice of node_j again
					# avoid multiple edges 
					while node_X == node_j or G.has_edge(node_j, node_X):
						node_X = random.choice(Nodes)
						if G.degree(node_j) >= N-1:
							break
					# choose method a : uncomment G.remove_edge...
					# choose method b : comment G.remove_edge 		 
					else:
						#G.remove_edge(node_j,node_k)
						G.add_edge(node_j,node_X)
						
	# ensuring no self loops
	for node_i in Nodes:
		if G.has_edge(node_i , node_i):
			raise nx.NetworkXError('This graph has self loops!!')
							
	return G
	
def rewiring_edges_connected(G, p, tries=1000):
	# repeating rewiring_edges until it returns connected Graph
	G = rewiring_edges(G, p)	
	count = 1
	while not nx.is_connected(G):
		G = rewiring_edges(G, p)
		count = count + 1
		if count > tries:
			raise nx.NetworkXError('cannot generate connected graph')
	return G
	

def clustering_ring(k):
	# analytical clustering coefficient of ring lattice
	# k : number of nearest neighbors of any node in ring graph
	# [Barrat and Weigt, 2000]
	cc = (3*k-3)/float(4*k-2)
	return cc

def clustering_rewiring(k , p):
	# calculates the clustering coefficient of rewired graph
	# k : number of nearest neighbors of any node in ring graph
	
	# cc = float(3*k-3) * pow((1-p),3) /float(4*k-2)
	# [Barrat & Weigt, 2000] for rewiring (method a) 
	cc = (3*k-3) / float( 4*k-2 + 4*k*p*(p+2)  )
	# [Newman, 2002] without rewiring (method b)
	return cc

ADJ 	= {}	# an empty adjacency list for a node
triads  = {}	# an empty triangles list for a node

def degree_node(G,node_i):
	# finds the connected neigbors of a given node 
	# returns a list
	ADJ[node_i] = []
	for j in G.nodes():		
		if G.has_edge(j,node_i):			
			ADJ[node_i].append(j)
	return ADJ[node_i]

def triangle_node(G,nodes):
	# finds the number of triangles around a given node
	# returns an integer, must be divided by 2	
	
	ADJ[nodes] = []
	for j in G.nodes():
		if G.has_edge(j,nodes):
			ADJ[nodes].append(j) # list of neigbors	
	
	for node_i in ADJ:
		triads[node_i] = []		 
		adjacent_i= set(ADJ[node_i]) 
		count_tri = 0
		for node_j in adjacent_i:
			new_set = set(G[node_j])-set([node_j])
			count_tri +=len(adjacent_i.intersection(new_set))  		
	return  int(count_tri/2)

def cluster_coef_numer(G):
	# calculates average cluster coefficient of a graph
	# [Watts and Strogatz, 1998]
	clust_coef=0
	N = nx.number_of_nodes(G)
	for nodes in G:
		k_i = len(degree_node(G,nodes))   # number of degrees
		t_i = triangle_node(G,nodes)	  # number of triads	
		clust_coef += 2 * float(t_i) /(k_i * (k_i - 1))
	return clust_coef/float(N)


def path_node(G,node):
	# finds path lengths of given node to all other nodes
	# returns a dict, keys are other nodes, values distances
	node_paths = {}                 
	distance   = 0               
	temp_nodes = {node:0} 		
	while len(temp_nodes) != 0:
		new_nodes  = temp_nodes  
		temp_nodes = {}         
		for v in new_nodes:
			if v not in node_paths:
				node_paths[v] = distance 
				temp_nodes.update(G[v]) 	
		distance=distance+1	
	return node_paths

def path_ave(G):
	# returns average shortest pathway of a connected graph
	# [Barabási, A.-L., and R. Albert, 2002]
	N = nx.number_of_nodes(G)
	summ = 0
	for node_i in G:
		for keys in path_node(G , node_i):
			summ = summ + path_node(G, node_i)[keys]
	return summ / float(N*(N-1))

	
def f(x):
	# used to find the analytical solution of the shortest pathway
	# [Newman et al. , 1965]
    return 1.0/(2.0*math.sqrt(x*x + 2*x))*np.arctanh(math.sqrt(x/(x + 2.0))) 

def path_analy(L, k , p):
	path_length = L* f(L*k*p) / float(k)
	return path_length

def plot_graph(G):
	# plots a given graph
	pos = nx.shell_layout(G)
	nx.draw(G, pos)
	pl.show()



y=[]
L = 600
k = 4.0 


N 		 = 100
k 		 = 5
p_temp 	 = 0.1
G_1      = ring_graph(N, k) 		 # generate a ring graph
cc_analy = clustering_ring(k)        # analytical clustering coefficient
cc_numer = cluster_coef_numer(G_1)   # numerical clustering coefficient

G_2     = rewiring_edges(G_1, p_temp)    # generate a small world graph
G_2     = rewiring_edges_connected(G_1, p_temp, tries=100)
cc_A    = clustering_rewiring(k,p_temp)  # analytical clustering coeff.
cc_N    = cluster_coef_numer(G_2)        # numerical clustering coeff.

###plotting graphs####

#plot_graph(G_1)
#plot_graph(G_2)
#plot_graph(G_3)

### <measure>_<graph_type>_<algorithm> ###

d_sworld_numer	 = []			
d_ring_numer	 = []			

d_ring_analy	 = []
d_sworld_analy   = []

C_ring_numer     = []  
C_ring_analy     = []

C_sworld_analy   = []
C_sworld_numer   = []

G_ring   = ring_graph(N, k)
L_ring	 = nx.number_of_edges(G_ring)

temp_a   = clustering_ring(k)
temp_b   = cluster_coef_numer(G_ring)

d0_numer = path_ave(G_ring)
d0_analy = path_analy(L_ring, k , p=0.00001)


p_values = np.arange(0.0001,1.0001,0.01)


for p in p_values:
	
	G_sworld 			= rewiring_edges(G_ring, p)
	L                   = nx.number_of_edges(G_sworld)
	
	d_ring_analy        = np.append(d_ring_analy, d0_analy)
	d_ring_numer	    = np.append(d_ring_numer, d0_numer)

	d_sworld_analy 	    = np.append(d_sworld_analy, path_analy(L,k,p))	
	d_sworld_numer		= np.append(d_sworld_numer, path_ave(G_sworld))
	
	C_ring_analy        = np.append(C_ring_analy ,temp_a)
	C_ring_numer        = np.append(C_ring_numer, temp_b)

	C_sworld_analy	 	= np.append(C_sworld_analy , clustering_rewiring(k,p))
	C_sworld_numer      = np.append(C_sworld_numer, cluster_coef_numer(G_sworld))

	

	



fig = pl.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
fig.suptitle('Average Shortest Pathway of Small Network ' , fontsize=14, fontweight='bold')

fig = pl.figure(1)
ax  = fig.add_subplot(1,1,1)
ax.set_xscale('log')
pl.plot(p_values,(d_sworld_numer)/d_ring_numer, 'r', label = 'd - numerical')
pl.plot(p_values, d_sworld_analy/d_ring_analy, 'b' , label = 'd - analytical')
pl.ylim( 0 ,1.2 )
pl.xlim( ax.get_xlim() )
pl.xlabel('probability of rewiring, p')
pl.ylabel('$d/d_{max}$')
pl.title('N = ' +str(N)+  '   k = '+ str(k) ) 
lg = legend()
lg.draw_frame(False)
#pl.show()

figu = pl.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
figu.suptitle('Average Clustering Coefficient of Small Network ' , fontsize=14, fontweight='bold')

figu = pl.figure(2)
ax  = figu.add_subplot(1,1,1)
ax.set_xscale('log')
pl.plot(p_values, C_sworld_numer/C_ring_numer, 'r', label = 'cc - numerical')
pl.plot(p_values, C_sworld_analy/C_ring_analy, 'b', label = 'cc - analytical')
pl.ylim( 0 ,1.2 )
pl.xlim( ax.get_xlim() )
pl.xlabel('p')
pl.ylabel('$C/C_{max}$')
pl.title('N = ' +str(N)+  '   k = '+ str(k) )

pl.show()
