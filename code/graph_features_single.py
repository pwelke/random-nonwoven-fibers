import sys, getopt

# Data related libraries
import numpy as np
import pandas as pd

# Network
import networkx as nx
from networkx.algorithms import bipartite

# Plotting
import matplotlib.pyplot as plt
#from ipywidgets import interact, fixed
#from IPython.display import display, HTML

# Convenient helpers
import csv
import math
import re
from copy import copy
from time import time
from datetime import date
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.preprocessing import minmax_scale

from os import listdir
from os.path import isfile, join

import fileutil

# Printing libraries and settings
# import warnings; warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format','{0:.2f}'.format)

#%matplotlib inline

def fileImportToNetworkX(file):
	return nx.read_graphml(file)
	
def getFaceNodes(G):
	# Node types
	upper_face_nodes, lower_face_nodes, front_face_nodes, back_face_nodes = [], [], [], []
	right_face_nodes, left_face_nodes, inner_nodes = [], [], []

	for node in list(G.nodes(data=True)):
		if node[1]['NodeType'] == 6:
			upper_face_nodes.append(node[0])
		elif node[1]['NodeType'] == 5:
			lower_face_nodes.append(node[0])
		elif node[1]['NodeType'] == 4:
			front_face_nodes.append(node[0])
		elif node[1]['NodeType'] == 3:
			back_face_nodes.append(node[0])
		elif node[1]['NodeType'] == 2:
			right_face_nodes.append(node[0])
		elif node[1]['NodeType'] == 1:
			left_face_nodes.append(node[0])
		elif node[1]['NodeType'] == 0:
			inner_nodes.append(node[0])

	
	return upper_face_nodes, lower_face_nodes, front_face_nodes, back_face_nodes, right_face_nodes, left_face_nodes, inner_nodes
	
def computeSumLength(G):
	summe = 0
	edge_list = list(G.edges(data = True))
	
	for i in range(0, len(edge_list)):
		summe += edge_list[i][2]['Length']
		
	return summe
	
def computeDiffEuclLength(G):
	"""
	Input: Graph G
	Output: Tuple of (Mean, Median, Sum) of the difference between length-attribute and euclidian distance between every two connected nodes
	"""
	diff_dist = []
	edges = list(G.edges(data = True))

	for i in range(0, len(edges)):
		edge = edges[i]
		u = edge[0]
		v = edge[1]

		x = (G.nodes[u]['x_Val'], G.nodes[u]['y_Val'], G.nodes[u]['z_Val']) 
		y = (G.nodes[v]['x_Val'], G.nodes[v]['y_Val'], G.nodes[v]['z_Val'])   
		euclid_dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
		length = edge[2]['Length']

		diff_dist.append(length - euclid_dist)
		
	return np.mean(diff_dist), np.median(diff_dist), np.sum(diff_dist)
	
def computeFaceDistance(G, upper_face_nodes, lower_face_nodes):
	"""
	Computes the number of edges along a shortest path (in terms of number of edges along it) 
	between the upper and lower face
	"""
	# Introduce two new nodes
	G.add_node('upper_fictive_node')
	G.add_node('lower_fictive_node')
	
	for node in upper_face_nodes:
		G.add_edge('upper_fictive_node', node)

	for node in lower_face_nodes:
		G.add_edge('lower_fictive_node', node)
	
	# Compute path length
	dist = nx.shortest_path_length(G, 'upper_fictive_node', 'lower_fictive_node')
	
	# Remove the introduced nodes
	G.remove_node('upper_fictive_node')
	G.remove_node('lower_fictive_node')
	
	# Here we have to substract the two (fictional) edges from the shortest path
	return dist - 2
	
def computeFaceWeightedDistance(G, upper_face_nodes, lower_face_nodes):
	"""
	Computes the length along a shortest weighted path 
	between upper and lower face nodes
	"""
	# Introduce two new nodes
	G.add_node('upper_fictive_node')
	G.add_node('lower_fictive_node')
	
	for node in upper_face_nodes:
		G.add_edge('upper_fictive_node', node)

	for node in lower_face_nodes:
		G.add_edge('lower_fictive_node', node)
	
	# Compute path length
	dist = nx.shortest_path_length(G, 'upper_fictive_node', 'lower_fictive_node', weight = 'Length')
	
	# Remove the introduced nodes
	G.remove_node('upper_fictive_node')
	G.remove_node('lower_fictive_node')
	
	# Here we have to substract the two (fictional) edges from the shortest path
	# THIS MIGHT BE WRONG COPIED FROM ABOVE? We should NOT subtract 2, because the newly introduced edges to not have any length specified (i.e. length = 0)
	return dist - 2
	
def computeFaceEuclidianDistance(G, upper_face_nodes, lower_face_nodes):
	"""
	Computes the sum of euclidian distances between successive nodes inside 
	a shortest weighted path between top and bottom nodes
	"""
	
	# Introduce two new nodes
	G.add_node('upper_fictive_node')
	G.add_node('lower_fictive_node')
	
	for node in upper_face_nodes:
		G.add_edge('upper_fictive_node', node)

	for node in lower_face_nodes:
		G.add_edge('lower_fictive_node', node)
		
	# Compute path
	path = nx.shortest_path(G, 'upper_fictive_node', 'lower_fictive_node', weight = 'Length')
	
	# Remove the introduced nodes
	G.remove_node('upper_fictive_node')
	G.remove_node('lower_fictive_node')
	
	# Remove virtual nodes from path
	path = path[1:len(path)-1]
	
	# Calculate distance
	path_dist_sum = 0
	for i in range(0, len(path)-1):
		u = path[i]
		v = path[i+1]
		#print(f"{G.node[u]['x_Val']}, {G.node[u]['y_Val']}, {G.node[u]['z_Val']}")

		x = (G.nodes[u]['x_Val'], G.nodes[u]['y_Val'], G.nodes[u]['z_Val']) 
		y = (G.nodes[v]['x_Val'], G.nodes[v]['y_Val'], G.nodes[v]['z_Val'])   

		path_dist_sum += math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
	
	return path_dist_sum
	
def computeMinCut(G, upper_face_nodes, lower_face_nodes):
	"""
	Computes the number of edges inside a minimum edge cut
	TODO: At this moment virtual edges on top or bottom can be part of the cut. 
	They need to be removed from the final result
	"""
	# Introduce two new nodes
	G.add_node('upper_fictive_node')
	G.add_node('lower_fictive_node')

	for node in upper_face_nodes:
		G.add_edge('upper_fictive_node', node)

	for node in lower_face_nodes:
		G.add_edge('lower_fictive_node', node)

	cut = nx.minimum_edge_cut(G, s = 'upper_fictive_node', t = 'lower_fictive_node')
	#print(f"Anzahl Kanten in einem min Schnitt, der oben von unten trennt: {len(cut)}")

	# Remove the introduced nodes
	G.remove_node('upper_fictive_node')
	G.remove_node('lower_fictive_node')
	
	return len(cut)
	
def compute_standard_graph_features(filename, file=None):
	
	results = {}

	if file is None:
		file = filename

	""" Import graph """
	G = fileImportToNetworkX(file)
	#print(f"Graph imported from {filename}")

	#""" Extract parameters from filename """
	m = re.search('Kappa0p[0-9]+_', filename)
	kappa = m.group(0).replace("Kappa","").replace("p",".").replace("_","")

	m = re.search('SigRamp[0-9]p[0-9]+_', filename)
	sigRamp = m.group(0).replace("SigRamp","").replace("p",".").replace("_","")

	m = re.search('SigSde[0-9]p[0-9]+_', filename)
	sigSde = m.group(0).replace("SigSde","").replace("p",".").replace("_","")

	m = re.search('Sld[0-9]+_', filename)
	sld = m.group(0).replace("Sld","").replace("_","")

	""" Write results """
	results[filename] = {}
	results[filename]['kappa'] = kappa
	results[filename]['sigRamp'] = sigRamp
	results[filename]['sigSde'] =  sigSde
	results[filename]['sld'] =  sld

	# Calculate basic statistics
	results[filename]['n_nodes'] = len(list(G.nodes))
	results[filename]['n_edges'] = len(list(G.edges))
	results[filename]['max_degree'] = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][1]
	results[filename]['length_sum'] = computeSumLength(G)

	# Calculate face nodes
	upper, lower, front, back, right, left, inner = getFaceNodes(G)
	results[filename]['n_upper_face'] = len(upper)
	results[filename]['n_lower_face'] = len(lower)

	# Calculate face distances
	min_dist = computeFaceDistance(G, upper, lower)
	results[filename]['min_dist'] = min_dist
	min_weight_dist = computeFaceWeightedDistance(G, upper, lower)
	results[filename]['min_weight_dist'] = min_weight_dist
	min_eucl_dist = computeFaceEuclidianDistance(G, upper, lower)
	results[filename]['min_eucl_dist'] = min_eucl_dist

	# Calculate Mean difference between length and euclidian disctance of every edge
	diff_eucl_length_mean, diff_eucl_length_median, diff_eucl_length_sum = computeDiffEuclLength(G)
	results[filename]['diff_eucl_length_mean'] = diff_eucl_length_mean
	results[filename]['diff_eucl_length_median'] = diff_eucl_length_median
	results[filename]['diff_eucl_length_sum'] = diff_eucl_length_sum

	# Cuts
	results[filename]['min_cut_size'] = computeMinCut(G, upper, lower)

	#print(f"LOG: {filename} finished")

	""" Export to csv """
	(pd.DataFrame.from_dict(data=results, orient='index')
	#    .to_csv(f'../results/features/{filename}_graph.csv', header=list(results[next(iter(results))].keys())))
	.to_csv(fileutil.featurefolder(f'{filename}_graph.csv'), header=list(results[next(iter(results))].keys())))

	## CodeOcean
	#(pd.DataFrame.from_dict(data=results, orient='index')
	#   .to_csv(f'../results/features/{filename}_graph.csv', header=list(results[next(iter(results))].keys())))
	   

if __name__ == "__main__":
   compute_standard_graph_features(sys.argv[1])
   print("Done!")