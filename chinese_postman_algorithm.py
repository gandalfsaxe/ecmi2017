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
import copy
from collections import OrderedDict
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
    
def analyze_odd_nodes(graph):
    ''' Returns list of nodes with odd number of neighbors and the distance
    between each pair '''
    
    node_order = add_node_order_to_graph(graph)
    odd_node_list = [node for node,order in node_order.items() if order%2 != 0]
    odd_node_pairing_distance = pd.DataFrame(0, index=odd_node_list, columns=odd_node_list)
    for node1, node2 in itertools.combinations(odd_node_list,2):
        min_distance = nx.shortest_path_length(graph, node1, node2, weight='length')
        odd_node_pairing_distance[node1][node2] = min_distance
        odd_node_pairing_distance[node2][node1] = min_distance
        
    return odd_node_list, odd_node_pairing_distance


def find_least_total_distance(graph, odd_node_list, odd_node_pairing_distance):
    '''  Finds the optimum total distance in the chinese postman problem and the edges that must be backtracked '''        
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
    
    backtracked_edges = []
    for node1, node2 in best_pair_list:
        backtracked_nodes = nx.shortest_path(graph, source=node1, target=node2, weight='length')
        for ind in range(len(backtracked_nodes)-1):
            backtracked_edges.append((backtracked_nodes[ind], backtracked_nodes[ind+1]))
            
    return least_total_distance, backtracked_edges
    
def get_all_possible_pairings(node_list, remove1, remove2):
    ''' Returns all possible pairings in "node list" excluding remove1 and 2. 
    Either or both must be None '''
    if (remove1 != None) != (remove2 != None): #True when either remove1 or 2 is None (XOR operator)
        if remove1 != None:
            node_to_fix = remove1
        else:
            node_to_fix = remove2
        
        if node_to_fix not in node_list:
            #If the fixed node is even, ignore it and search optimal start/end
            remove1 = remove2 = None
            print 'WARNING: start and/or end are not odd, searching for optimal odd pair'
        else:
            return [(node_to_fix,n) for n in node_list if n!= node_to_fix]
                    
    if remove1 == None and remove2 == None:
        return list(itertools.combinations(node_list,2))

    
def solve_chinese_postman_problem(graph, start=None, end=None):
    ''' Returns the path and total distance that solves the chinese postman problem.
    If "start" and/or "end" are None, then it finds the most optimal starting and/or ending points.
    "Start" and "End" can be the same number'''
    
    #Deals with start/end nodes and computes least total distance and backtracked nodes
    odd_node_list, odd_node_pairing_distance = analyze_odd_nodes(graph)
    
    if start == None or end == None:
        possible_startend_pairs= get_all_possible_pairings(odd_node_list, start, end)
        startend_dict = {} #{(start,end) : (least_total_distance, backtracked_edges)}
        for startend_pair in possible_startend_pairs:
            current_odd_node_list = [n for n in odd_node_list if n not in startend_pair]
            least_total_distance, backtracked_edges = find_least_total_distance(graph, current_odd_node_list, odd_node_pairing_distance)
            startend_dict[startend_pair] = (least_total_distance, backtracked_edges)
        optimal_pair = min(startend_dict.keys(), key = lambda x:startend_dict[x][0])
        
        least_total_distance, backtracked_edges = startend_dict[optimal_pair]
        start, end = optimal_pair
            
    else:
        if start not in odd_node_list or end not in odd_node_list:
            print 'WARNING: start and/or end are not odd, ending point might be unexpected'
        #If both start and end are specified, check if they are odd nodes
        current_odd_node_list = copy.deepcopy(odd_node_list)
        
        if start != end:
            try:
                current_odd_node_list.remove(start)
            except ValueError:
                pass
            try:
                current_odd_node_list.remove(end)
            except ValueError:
                pass
            least_total_distance, backtracked_edges = find_least_total_distance(graph, current_odd_node_list, odd_node_pairing_distance)

        else:
            least_total_distance, backtracked_edges = find_least_total_distance(graph, odd_node_list, odd_node_pairing_distance)

    # Count how many times each edge must be visited
    edge_visit_count = {edge:1 for edge in graph.edges()}
    for edge in backtracked_edges:
        try:
            edge_visit_count[edge] += 1
        except:
            edge_visit_count[edge[::-1]]
    nx.set_edge_attributes(graph, 'visits', edge_visit_count)
    
    # Deterministic decision tree to get the path
    path = [start]
    node = start
    temporal_graph = copy.deepcopy(graph)
    while True:
        neighbor_list = sorted(temporal_graph.neighbors(node))
        neighbor_visits = {n:temporal_graph[node][n]['visits'] for n in neighbor_list}
        next_non_visited_neighbor = next((n for n,visits in neighbor_visits.items() if visits != 0), None)
        
        found_ending = False
        while True:
            print node, neighbor_visits
            #Loop until an appropriate (or None) neighbors are found
            next_non_visited_neighbor = next((n for n,visits in neighbor_visits.items() if visits != 0), None)
            if next_non_visited_neighbor == None:
                if found_ending == False:
                    #If no neighbor or ending is found break
                    break
                else:
                    next_non_visited_neighbor = end
            
            elif next_non_visited_neighbor == end and temporal_graph[node][end]['visits'] == 1:
                #Avoid the last visit to the ending node as long as other alternatives exist
                del neighbor_visits[next_non_visited_neighbor]
                found_ending = True
                continue

            if temporal_graph[node][next_non_visited_neighbor]['visits'] == 1:
                #Delete edges that won't be visited, unless they leave the graph unconnected
                temporal_graph.remove_edge(node, next_non_visited_neighbor)
                if len(temporal_graph.neighbors(node)) == 0:
                    temporal_graph.remove_node(node)
                if nx.is_connected(temporal_graph) == False:
                    original_length = graph[node][next_non_visited_neighbor]['length']
                    temporal_graph.add_edge(node, next_non_visited_neighbor, attr_dict = {'visits':1, 
                    'length':original_length})
                    del neighbor_visits[next_non_visited_neighbor]
                    continue
            else:
                temporal_graph.edge[node][next_non_visited_neighbor]['visits'] -= 1
            break


        if next_non_visited_neighbor == None:
            #No more options -> cleared
            break

        node = next_non_visited_neighbor
        path.append(node)
        
    check_distance = 0
    prev_node = start
    for node in path[1:]:
        check_distance += graph.edge[prev_node][node]['length']
        prev_node = node

    if least_total_distance < check_distance:
        print 'WARNING: PATH OF FINAL DISTANCE IS NOT OPTIMAL'
        
    if sum(nx.get_edge_attributes(temporal_graph,'visits').values()) != 0:
        print 'WARNING: NOT ALL EDGES HAVE BEEN VISITED'

    return least_total_distance, path
    

def main():
    graph = example_graphs.create_test_array()
#    graph = example_graphs.create_test_array_with_dead_end()
#    print solve_chinese_postman_problem(graph, start=None, end=None)

#    graph = example_graphs.create_test_array_notebook()
    graph = example_graphs.create_test_array_worked_example_33()
    example_graphs.draw_network_with_labels(graph)
#    example_graphs.draw_network_with_labels(graph)
#    node_order = add_node_order_to_graph(graph)
    print solve_chinese_postman_problem(graph, start='E', end='E')


    

if __name__ == '__main__':
    tt = time.time()
    main()
    print 'Ellapsed time: %.4f seconds' %(time.time() - tt)
