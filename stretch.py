import sys, getopt

import networkx as nx
import numpy as np
import math
import sys
import time

from dataclasses import dataclass, field
from typing import Any

from Heap import Heap

from os import listdir
from os.path import isfile, join
import pandas as pd
from tqdm import tqdm
from datetime import date

def load_graph(file: str, verbose: bool=True):
    ''' Load a graph, enforce that it is a multigraph and run some initial checks and statistics.
    Note that networkx has different edge access api depending on whether the graph is a multigraph or not.
    Not enforcing multigraph results in strange errors for graphs that happen to have at most one edge / fiber
    between all vertices.'''

    G = nx.read_graphml(file, force_multigraph=True)

    if verbose:
        print('n', G.number_of_nodes())
        print('m', G.number_of_edges())

        # if this returns true, the algorithm implemented below should terminate in finite time.
        print('is connected?', nx.algorithms.is_connected(G))

    # this is the central data storage that is processed and altered, together with G
    nodedata = dict(G.nodes.data())
    
    return G, nodedata

def parameter_validation(nodedata, 
                         sort_dim: str='z_Val', 
                         glued_node_type: int=5, 
                         target_node_type: int=6,
                         verbose: bool=True):
    
    # ranges from which to select parameters
    NodeTypes = {nodedata[n]['NodeType'] for n in nodedata}
    if glued_node_type not in NodeTypes:
        raise ValueError('glued vertex type ' + str(glued_node_type) + ' must be in ' + str(NodeTypes))
    if target_node_type not in NodeTypes:
        raise ValueError('target vertex type ' + str(target_node_type) + ' must be in ' + str(NodeTypes))

    # if your data format changes, you may have to change something here
    dims = {'x_Val', 'y_Val', 'z_Val'}
    if sort_dim not in dims:
        raise ValueError('stretch dimension ' + str(sort_dim) + ' must be in ' + str(dims))

    other_dims = dims.copy()
    other_dims.remove(sort_dim)
    return other_dims

def get_positions(original_data, sort_dim, target_node_type):
    target_nodes = [v for v in original_data if original_data[v]['NodeType'] == target_node_type]
    poss = [original_data[v][sort_dim] for v in target_nodes]
    return np.array(poss)

def init_queue(G, nodedata, glued_node_type, sort_dim, other_dims, eps):
    '''we put vertex objects in a priority queue that gives us the vertices, from smallest to largest'''
    vertex_queue = Heap()

    # mark all fixed vertices
    for v in nodedata:
        vv = nodedata[v]
        vv['v'] = v
        # don't add nodes that are in the glued class
        if vv['NodeType'] == glued_node_type:
            vv['fixed'] = True
        else:
            vv['fixed'] = False

    # add neighbors of fixed vertices to queue
    # we need the first pass above to label vertices correctly
    for v in nodedata:
        vv = nodedata[v]
        vv['v'] = v
        # add nodes that are neighbors of glued vertices
        if vv['NodeType'] == glued_node_type:
            for w in G.adj[v]:
                ww = nodedata[w]
                if ww['fixed'] == False:
                    vertex_queue.add(w, priority=max_move(G, nodedata, sort_dim, other_dims, eps, ww))

    return vertex_queue

def max_move(G, nodedata, sort_dim, other_dims, eps, vv):
    '''compute, for vv, the largest possible shift in sort_dim, 
    considering only constraints given by neighbors that are already fixed.
    '''
        
    v = vv['v']
    neighbors = G.adj[v]

    # if v == 'n12' or v == 'n50':
    #     print('we are here')

    # here, we will store the possible new positions of v
    shifts = list()

    for w in neighbors:
        ww = nodedata[w]
        if ww['fixed'] == True:
            for e in neighbors[w]:
                l = neighbors[w][e]['Length']

                # what's the largest sort_dim value that is achievable without breaking e?
                candidate_shift = 0.
                for other_dim in other_dims:
                    candidate_shift -= (ww[other_dim] - vv[other_dim]) ** 2
                candidate_shift += l ** 2 # do it after the above to avoid numerical issues
                
                # due to inexactness, candidate_shift might be slightly negative. 
                if candidate_shift > eps:
                    candidate_shift = math.sqrt(candidate_shift)
                else:
                    candidate_shift = 0.

                shifts.append(candidate_shift + ww[sort_dim])

    # now, find the largest sort_dim value that does not violate any edge constraint
    # and compute the largest possible shift of v by substracting the current sort_dim coordinate of v
    shift = min(shifts) - vv[sort_dim]

    # print(f'we may move {v} by {shift}')
    return shift

def update_max_move(G, vv, ww, sort_dim, other_dims, eps):
    '''compute, for ww, the largest possible shift in sort_dim, 
    considering only constraints given by edges to vv.
    We can use this to update the priority of ww after processing vv.
    '''

    v = vv['v']
    w = ww['v']
    adjacent_edges = G.adj[v][w]

    # here, we will store the possible new positions of v
    shifts = list()

    for e in adjacent_edges:
        l = adjacent_edges[e]['Length']

        # what's the largest sort_dim value that is achievable without breaking e?
        candidate_shift = 0.
        for other_dim in other_dims:
            candidate_shift -= (ww[other_dim] - vv[other_dim]) ** 2
        candidate_shift += l ** 2 # do it after the above to avoid numerical issues
        
        # due to inexactness, candidate_shift might be slightly negative. 
        if candidate_shift > eps:
            candidate_shift = math.sqrt(candidate_shift)
        else:
            candidate_shift = 0.

        shifts.append(candidate_shift + ww[sort_dim])

    # now, find the largest sort_dim value that does not violate any edge constraint
    # and compute the largest possible shift of v by substracting the current sort_dim coordinate of v
    shift = min(shifts) - vv[sort_dim]

    # print(f'we may move {v} by {shift}')
    return shift

def stretch_graph(G, nodedata, vertex_queue, sort_dim, other_dims, eps, no_unstretch, play_safe=True):
    '''we go through the queue, each time pulling the vertex object with the smalles possible shift when considering only fixed neighbor constraints.
    We shift it, fix it, and update its unfixed neighbors shift values accordingly.
    contents of nodedata and vertex_queue are altered by this method.
    :param:no_unstretch = True tells the algorithm to never move vertices 'down' in sort_dim, even if the input instance is invalid and would require this to be valid
    :param:play_safe = True raises a Warning whenever the invariant "moves can never decrease over the runtime of the algorithm" is broken. This *can* happen if the input instance is invalid
    '''

    # number of iterations and number of moves 
    niter = 0
    nmoves = 0

    minmove = -1.

    # while (not vertex_queue.empty()):
    try:
        while (True):
            v, move = vertex_queue.pop()

            if play_safe and minmove > move + eps:
                raise Warning('previous move was larger than the current one. this should not happen if the instance is valid')
            minmove = move

            vv = nodedata[v]
            # print(f'move {v} by {move}')
            vv[sort_dim] += move
            vv['fixed'] = True

            neighbors = G.adj[v]
            for w in neighbors:
                ww = nodedata[w]
                if ww['fixed'] == False:
                    # maxmove = update_max_move(G, vv, ww, sort_dim, other_dims, eps)
                    # if vertex_queue.exists(w):
                    #     # if the vertex is already in the queue, we have to respect constraints 
                    #     # from other fixed neighbors. however, like this it does not work...
                    #     maxmove = min(maxmove, vertex_queue.peek(w))

                    maxmove = max_move(G, nodedata, sort_dim, other_dims, eps, ww) # too slow
                    # print(f'maxmove for ({v}, {w}) is {maxmove}')

                    if no_unstretch:
                        maxmove = max(0., maxmove)
                    vertex_queue.add(w, maxmove)
                    # print(f'add {w}, {maxmove} to queue')

                    nmoves += 1

            niter += 1
    except KeyError:
        # the KeyError occurs when vertex_queue is empty, which can only happen if every vertex 
        # in the connected input graph has been processed
        pass
            
    return niter, nmoves

def l2dist(vv, ww, dims=['x_Val', 'y_Val', 'z_Val']):
    dist = 0.
    for dim in dims:
        dist += (vv[dim] - ww[dim]) ** 2
    return math.sqrt(dist)


def is_valid(G, nodedata, eps):
    invalid = 0
    invalidlist = list()
    for v in nodedata:
        vv = nodedata[v]
        neighbors = G.adj[v]
        for w in neighbors:
            ww = nodedata[w]
            for e in neighbors[w]:
                l = neighbors[w][e]['Length']
                d = l2dist(vv, ww)
                if l + eps < d:
                    invalid += 1
                    invalidlist.append((v, w, l, d, vv['z_Val'], ww['z_Val']))
    print(f'there are {invalid / 2} edges with dissatisfied constraints')
    if invalid > 0:
        print('here are the first ten or so:')
        parsed = [
            f'({x[0]}, {x[1]}) l:{x[2]} < d:{x[3]} zs: {x[4]} {x[5]}' for x in invalidlist if x[0] < x[1]]
        print('\n'.join(parsed[:min(invalid/2, 10)]))
    return invalid == 0

def get_features(processed_positions, original_positions):
    diffs = processed_positions - original_positions
    return (np.mean(diffs), np.std(diffs), np.max(diffs), np.median(diffs), np.sum(diffs))

def change_edge_length(G, factor):
    for u, v, keys, weight in G.edges(data="Length", keys=True):
        if weight is not None:
            G[u][v][keys]["Length"] = weight * factor

def main(file: str='2020_08_12_KAPPA0p0003_A0p15_B0p25_Sample001_RedFibStruc.graphml', 
         sort_dim: str='z_Val', 
         glued_node_type: int=5, 
         target_node_type: int=6, 
         eps: float=1e-6, 
         verbose: bool=False, 
         no_unstretch=True,
         factor = 1):
    '''
    This function assumes a graph whose vertices are embedded in three-dimensional space and 
    whose edges each have a maximum length. Initially, we expect the vertex positions to 
    respect the lengths of edges, i.e., d(v,w) <= l((v,w)) if v,w are vertices connected by an 
    edge (v,w) of length l((v,w)) and d(v,w) is the Euclidean distance of the embeddings of v,w
    in three-dimensional space.
    
    It loads a graph from file and stretches it along a given axis sort_dim, without changing the 
    coordinates of vertices in other axes. That is, we fix all vertices of glued_node_type. For all
    other vertices v, we start from smallest values along sort_dim and iteratively push the sort_dim
    value of the current vertex as far as possible, without breaking any d(v,w) <= l((v,w)) for all
    w incident to v.
    
    If we can move v by more than eps, we do so and add v again to the set of processed vertices. 
    If not, we are done with v and can remove it from our list of vertices that need to be processed.
    
    ## Known Issues
    - If the graph contains a connected component that does not contain any vertices of glued_node_type,
      then the algorithm breaks (we can move this component infinitely along sort_dim).
    - Otherwise, the algorithm terminates, as each vertex is popped from vertex_queue only once and then fixed.
    - The runtime should be similar to Dijkstra's algorithm for shortest paths, but there is still an inefficiency
      when updating the shift values of the unfixed neighbors of v
    - the algorithm can also handle invalid instances
    
    # parameters to set (= default value)
    # graph file
    file = '2020_08_12_KAPPA0p0003_A0p15_B0p25_Sample001_RedFibStruc.graphml'
    # the dimension among which we want to pull.
    # must be in dims
    sort_dim = 'z_Val'
    # the nodes that are fixed and the nodes that are used to compute the stretch feature
    # must be in NodeTypes 
    glued_node_type = 5
    target_node_type = 6
    # a numerical accuracy (below which we do not move vertices)
    eps = 0.0001
    
    # a flag whether to keep the input sort_dim coordinate as minimum value of the stretched graph
    # (if set to false, stretching might actually result in negative average shift of the target_nodes if the input instance is not valid)
    no_unstretch = True
    '''
    
    G, nodedata = load_graph(file, verbose)
    other_dims = parameter_validation(nodedata, sort_dim, glued_node_type, target_node_type, verbose)

    if verbose:
        print(f'eps is {eps}')
        is_valid(G, nodedata, eps)
    
    original_positions = get_positions(nodedata, sort_dim, target_node_type)
    
    # Stretch factor
    if factor != 1:
        change_edge_length(G, factor)
    
    tic = time.time()
    vertex_queue = init_queue(G, nodedata, glued_node_type, sort_dim, other_dims, eps) 
    niter, nmoves = stretch_graph(G, nodedata, vertex_queue, sort_dim, other_dims, eps, no_unstretch, play_safe=False)
    toc = time.time()

    if verbose:
        print('Number of iterations', niter)
        print('Number of stretch operations', nmoves)
        print('Time:', toc - tic)
        is_valid(G, nodedata, eps)

    processed_positions = get_positions(nodedata, sort_dim, target_node_type)
    return get_features(processed_positions, original_positions)


def run(file):
    results = {}
    #factors = [1.05, 1.1]
    factors = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]

    # No multiplier (c=1.0)
    normal_stretch = main(file, verbose = False)

    # Multipliers    
    result_factors = []
    for factor in factors:
        factor_stretch = main(file, verbose = False, factor = factor)
        result_factors.append(factor_stretch)

    result_tuple = normal_stretch

    for factor_result in result_factors:
        result_tuple = result_tuple + factor_result

    results[file] = result_tuple

    # Export
    result_df = pd.DataFrame.from_dict(data=results, orient='index')
    columns = ["stretch_diff_mean", "stretch_diff_std", "stretch_diff_max", "stretch_diff_median", "stretch_diff_sum"]
    for factor in factors:
        for pre in ["diff", "std", "max", "median", "sum"]:
            columns.append(f"stretch_diff_{pre}_{factor}")

    result_df.columns = columns
    result_df.to_csv(f"features/{file}_stretch.csv")


if __name__ == "__main__":
   run(sys.argv[1])
   print("Done!")