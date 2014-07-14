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
# N : number of nodes
# k : number of nearest neighbors of any node
# p : probability of rewiring each edge
# seed : seed for random number generation

def ring_graph(N , k, seed=None):
	
	if k>=N :
		raise nx.NetworkXError("k must be smaller than N")
	if seed is not None:
		random.seed(seed)
		
	G_ring = nx.Graph()
	
	Nodes = list(range(N)) 
		
	for node_i in range(0, k/2 ):

		# listing available nodes, node_i is at last index 
		# merge nodes with available nodes in pairs
		# construct edges between paired nodes
		
		available_Nodes = Nodes[node_i+1:] + Nodes[0:node_i+1]
		paired_Nodes = zip(Nodes, available_Nodes)
		G_ring.add_edges_from(paired_Nodes)
		
		#print "node_i , Nodes", node_i , Nodes
		#print "available_nodes", available_Nodes
		#print "paired_nodes" , paired_Nodes
		#print "\n"
	
	# check if the graph does not have self loops
			
	for node_i in Nodes:
		if G_ring.has_edge(node_i , node_i):
			raise nx.NetworkXError('This graph has self loops!!')

	return G_ring	
	

def rewiring_edges(G , p ) :
	
	Nodes = G.nodes()
	N     = G.number_of_nodes()
	
	values = []
	for node in G:
		values.append(G.degree(node))
	
	ave_degree = float(sum(values)/float(N))	
	
	#print "average degree of ring graph: " , ave_degree	
			
	for i in range(0, int(ave_degree)/2): 
		available_nodes = Nodes[i+1:] + Nodes[0:i+1] 
        
        for node_j, node_k in zip(Nodes,available_nodes):
			if G.has_edge(node_j ,node_k):
				if random.random() < p:
					node_X = random.choice(Nodes)
					
					# avoid the choice of node_j or already connected node_X
					while node_X == node_j or G.has_edge(node_j, node_X):
						node_X = random.choice(Nodes)
						if G.degree(node_j) >= N-1:
							break
					# swapping edges between paired nodes (rewiring) 		 
					else:
						#G.remove_edge(node_j,node_k)
						G.add_edge(node_j,node_X)
						
	# ensuring no self loops and no multiple edges
	for node_i in Nodes:
		if G.has_edge(node_i , node_i):
			raise nx.NetworkXError('This graph has self loops!!')

					
	return G
	
def rewiring_edges_connected(G, p, tries=1000):
	# repeating rewiring_edges function until it returns connected Graph
	G = rewiring_edges(G, p)	
	count = 1
	while not nx.is_connected(G):
		G = rewiring_edges(G, p)
		count = count + 1
		if count > tries:
			raise nx.NetworkXError('cannot generate connected graph')
	return G
	
def plot_graph(G):
	pos = nx.shell_layout(G)
	nx.draw(G, pos)
	pl.show()

def clustering_ring(k):
	# calculates the clustering coefficient of ring lattice
	# k : number of nearest neighbors of any node in ring graph
	cc = (3*k-3)/float(4*k-2)
	return cc

def clustering_rewiring(k , p):
	# calculates the clustering coefficient of rewired graph
	# k : number of nearest neighbors of any node in ring graph
	
	# cc = float(3*k-3) * pow((1-p),3) /float(4*k-2) ## Barrat & Weigt
	cc = (3*k-3) / float( 4*k-2 + 4*k*p*(p+2)  )    ## Newman
	return cc

ADJ = {}
triads ={}
def degree_node(G,node_i):
	# connected neigbors of a given node 
	# returns a list
	ADJ[node_i] = []
	for j in G.nodes():		
		if G.has_edge(j,node_i):			
			ADJ[node_i].append(j)
	return ADJ[node_i]

def triangle_node(G,nodes):
	# number of triangles around a given node
	# returns an integer	
	
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
	# Watts and Strogatz
	clust_coef=0
	N = nx.number_of_nodes(G)
	for nodes in G:
		k_i = len(degree_node(G,nodes))   # number of degrees
		t_i = triangle_node(G,nodes)	  # number of triads	
		clust_coef += 2 * float(t_i) /(k_i * (k_i - 1))
	return clust_coef/float(N)


def f(x):
	# master function required for shortest path analytical solution
    return 1.0/(2.0*math.sqrt(x*x + 2*x))*np.arctanh(math.sqrt(x/(x + 2.0))) 

y=[]
L = 600
k = 4.0 
#p = np.arange(0.0001,1.0001,0.01)
p = np.array(np.logspace(0.0001,1.0,100)) 
for i in range(0, len(p)):
	y = np.append(y, f(L*k*p[i]))
print y 

fig = pl.figure(1)
ax  = fig.add_subplot(1,1,1)
#ax.set_xscale('log')
pl.plot(p,L*y/k)
#xscale('log') 
pl.show()


N = 100
k = 5
#p = 0.1
G_1 = ring_graph(N, k, seed=None)
cc_ring = clustering_ring(k)


#G_2 = rewiring_edges(G_1, p)
#cc_rewired = clustering_rewiring(k,p)
#print "clustering coefficient of rewired graph is : " , cc_rewired

#G_3 = rewiring_edges_connected(G_1, p , tries=100)
#cc_con = clustering_rewiring(k,p)
#print "clustering coefficient of rewired connected graph is : " , cc_con


#plot_graph(G_1)
#plot_graph(G_2)
#plot_graph(G_3)


d_ave	 = []
d_max	 = []

p_values = np.arange(0.0001,1.0001,0.01)

C_max_numer = np.zeros_like(p_values)
C_max = np.zeros_like(p_values)

C = []
C_numer = []

G_RING = ring_graph(N , k , seed = None)
temp_a = clustering_ring(k)

#temp_b = cluster_coef_numer(G_RING)
temp_b = nx.average_clustering(G_RING)

for i in range(0, len(p_values)):
	C_max[i]       = temp_a
	C_max_numer[i] = temp_b

for p in p_values:
	C	 	    = np.append(C , clustering_rewiring(k,p))
	G_1 		= ring_graph(N, k, seed=None)
	G_2 		= rewiring_edges(G_1, p)
	C_numer     = np.append(C_numer, cluster_coef_numer(G_2))
	#C_numer	    = np.append(C_numer, nx.average_clustering(G_2))

C_to_plot = C/ C_max 
C_numer_to_plot = C_numer/C_max_numer

print C_numer_to_plot	

fig = pl.figure(1)
ax  = fig.add_subplot(1,1,1)
ax.set_xscale('log')
pl.plot(p_values , C_to_plot)
pl.plot(p_values, C_numer_to_plot, 'r')
pl.ylim( 0 ,1.2 )
pl.xlim( ax.get_xlim() )
pl.xlabel('p')
pl.ylabel('$C/C_{max}$')
pl.title('Clustering Coefficient Ration over Probability')

#pl.show()






#G_numerical =  nx.watts_strogatz_graph(10 , 2 , p=0.5 , seed=None)
#length_1 = nx.single_source_shortest_path_length(G_numerical, source =1)
#length_2 = nx.average_shortest_path_length(G_numerical)
#print "d_{ij}_1 for i = 1 : ", sum(length_1.values()) / float(len(length_1.values()))
#print "d_{ij}_2 for i = 1 : ", length_2  
#plot_graph(G_numerical)



