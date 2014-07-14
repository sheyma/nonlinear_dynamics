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

N = nx.number_of_nodes(G)
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

print nx.average_clustering(G)
print cluster_coef_numer(G)



	
