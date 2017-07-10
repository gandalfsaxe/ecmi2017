# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:33:20 2017

@author: Albert
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
    
def stochastic_decision_tree(graph, initial_node):
    ''' Finds a solution of the street plowing problem given a network with edges.
    At each intersection it selects a path based on weighted probabilities '''
    #Method 1: each edge has a "visited" flag
    for u,v,d in graph.edges_iter(data=True):
        d['visited'] = False
    
    node = initial_node
    if node not in graph:
        print 'Wrong initial node'
        return
    
    visited_edges = 0
    total_edges = graph.number_of_edges()
    
    total_distance = 0
    node_path = [initial_node]
    
    iteration = 0
    max_iters = 1000
    while visited_edges < total_edges:
        neighbor_list = graph[node].keys()
        neighbor_weights = np.zeros(len(neighbor_list))
        for ind,neighbor in enumerate(neighbor_list):
            #Each neighbor gets a probability based on a weight
            weight = 1
#            distance = graph[node][neighbor]['length']
            visited = graph[node][neighbor]['visited']
            if visited:
                weight *= 0.1

            neighbor_weights[ind] = weight
        neighbor_weights /= sum(neighbor_weights)
        chosen_node = np.random.choice(neighbor_list, p=neighbor_weights)

        if graph[node][chosen_node]['visited'] == False:
            graph[node][chosen_node]['visited'] = True
            visited_edges += 1
            
        total_distance += graph[node][chosen_node]['length']
        node_path.append(chosen_node)
        
        node = chosen_node
        
        iteration += 1
        if iteration > max_iters:
            print 'Out of time'
            break
        
    return node_path, total_distance
            
def test_stochastic_decision_tree(graph, repetitions):
    ''' Simulate a stochastic decision tree "repetitions" amount of time.
    Starts every time at a random node. Returns shortest path '''
    shortest_distance = np.inf

    for i in range(repetitions):
        initial_node = np.random.choice(graph.nodes())
        node_path, total_distance = stochastic_decision_tree(graph, initial_node)
        if total_distance < shortest_distance:
            shortest_distance = total_distance
            shortest_path = node_path
    return shortest_path, shortest_distance


def main():
#    graph = create_test_array()
    graph = create_test_array_with_dead_end()
    draw_network_with_delay(graph)
#    print stochastic_decision_tree(graph, 1)
    print test_stochastic_decision_tree(graph, 1000)

if __name__ == '__main__':
    tt = time.time()
    main()
    print 'Ellapsed time: %.4f seconds' %(time.time() - tt)
