# -*- coding: utf-8 -*-
"""
Classical algorithm that solves the chinese postman problem. Conditions:
- Each edge has to be visited once
- Any edge can be travelled more than once
- The objective is to find the solution that travels the least total distance
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import example_graphs
import pandas as pd

def add_node_order_to_graph(graph):
    ''' Adds a label to each node specifying its order '''
    node_order = create_node_order_dict(graph)
    nx.set_node_attributes(graph, 'order', node_order)
    return node_order
        
def create_node_order_dict(graph):
    ''' Creates a dictionary with the order of each node '''
    node_order = {}
    for node in graph.nodes_iter():
        order = len(nx.neighbors(graph, node))
        node_order[node] = order
    return node_order
    
def find_all_possible_list_of_pairs(l):
    ''' Given a list of nodes, this function groups them in pairs and returns all possible
    combinations of pairs without node repetition.
    EX: [a,b,c,d] -> [(ab,cd),(ac,bd),(ad,bc)] '''
    final_pair_list = []    
    possible_pairs = list(itertools.combinations(l,2))
    for pair_list in itertools.combinations(possible_pairs, len(l)/2):
        node_list = zip(*pair_list)
        node_list = node_list[0] + node_list[1]
        total_nodes = len(node_list)
        unique_nodes = len(set(node_list))
        if total_nodes == unique_nodes:
            final_pair_list.append(pair_list)
    
    return final_pair_list


def find_least_total_distance(graph, initial_node, end_node):
    '''  Finds the optimum total distance in the chinese postman problem. 
    Needs initial and end node specified '''
    node_order = add_node_order_to_graph(graph)
    odd_node_list = [node for node,order in node_order.items() if order%2 != 0]
    odd_node_pairing_distance = pd.DataFrame(0, index=odd_node_list, columns=odd_node_list)
    for node1, node2 in itertools.combinations(odd_node_list,2):
        min_distance = nx.shortest_path_length(graph, node1, node2, weight='length')
        odd_node_pairing_distance[node1][node2] = min_distance
        odd_node_pairing_distance[node2][node1] = min_distance
        
    possible_pair_lists = find_all_possible_list_of_pairs(odd_node_list)
    best_pair_total_distance = np.inf
    for pair_list in possible_pair_lists:
        total_distance = 0
        for node1, node2 in pair_list:
            total_distance += odd_node_pairing_distance[node1][node2]
        if total_distance < best_pair_total_distance:
            best_pair_list = pair_list
            best_pair_total_distance = total_distance
            
    dict_of_graph_distances = nx.get_edge_attributes(graph, 'length')
    total_graph_distance = sum(dict_of_graph_distances.values())
    least_total_distance = total_graph_distance + best_pair_total_distance
    return least_total_distance, best_pair_list
    

            
            
    
        
#    for ind1,node1 in enumerate(odd_node_list):
#        for ind2,node2 in enumerate(odd_node_list):
#            if ind2 <= ind1:
#                continue
#            min_distance = nx.shortest_path_length(graph, node1, node2, weight='length')
#            odd_node_pairing_distance[node1][node2] = min_distance
#            odd_node_pairing_distance[node2][node1] = min_distance
    

def main():
    graph = example_graphs.create_test_array_notebook()
#    graph = example_graphs.create_test_array_worked_example_33()
    example_graphs.draw_network_with_labels(graph)
#    node_order = add_node_order_to_graph(graph)
    print find_least_total_distance(graph, 'A', 'E')
    

if __name__ == '__main__':
    tt = time.time()
    main()
    print 'Ellapsed time: %.4f seconds' %(time.time() - tt)
