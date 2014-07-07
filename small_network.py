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
# k : number of nearest neighbots of any node
# p : probability of rewiring each edge
# seed : seed for random number generation

def ring_graph(N , k, seed=None):
	
	if k>=N :
		raise nx.NetworkXError("k must be smaller than N")
	if seed is not None:
		random.seed(seed)
	
	
	
	G_ring = nx.Graph()
	
	Nodes = list(range(N)) 
	
	#if k %2 != 0 : 
		#k = (k - 1)
	
	
	for node_i in range(0, k/2 ):

		# listing available nodes, node_i is at last index 
		# merge nodes with available nodes in pairs
		# construct edges between paired nodes
		
		available_Nodes = Nodes[node_i+1:] + Nodes[0:node_i+1]
		paired_Nodes = zip(Nodes, available_Nodes)
		G_ring.add_edges_from(paired_Nodes)
		
		print "Nodes", Nodes
		print "available_nodes", available_Nodes
		print "paired_nodes" , paired_Nodes
		print "\n"
	
	
	# check if the graph does not have self loops
		
	for node_i in Nodes:
		if G_ring.has_edge(node_i , node_i):
			raise nx.NetworkXError('This graph has self loops!!')

	return G_ring	
	

def random_graph(G , p ) :
	
	Nodes = G.nodes()
	N = G.number_of_nodes()
	
	values = []
	for node in G:
		values.append(G.degree(node))
	
	ave_degree = float(sum(values)/float(N))	
	
	print "averaga degree : " , ave_degree
	
	# swapping edges between paired nodes (rewiring) 
	# ensuring no self loops and no multiple edges
		
	for i in range(0, int(ave_degree)/2): 
		available_nodes = Nodes[i+1:] + Nodes[0:i+1] 
        
        for node_j, node_k in zip(Nodes,available_nodes):
			if random.random() < p:
				node_X = random.choice(Nodes)
				
				while node_X == node_j or G.has_edge(node_j, node_X):
					node_X = random.choice(Nodes)
					if G.degree(node_j) >= N-1:
						break 
				else:
					G.remove_edge(node_j,node_k)
					G.add_edge(node_j,node_X)
	return G
	


def plot_graph(G):
	pos = nx.shell_layout(G)
	nx.draw(G, pos)
	pl.show()


	
G_1 = ring_graph(10, 5, seed=None)
#G_2 = random_graph(G_1 ,  0.5)


plot_graph(G_1)


#G = nx.watts_strogatz_graph(10 , 5 , 1 , seed=None)

