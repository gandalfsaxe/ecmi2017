# -*- coding: utf-8 -*-
"""
Classical algorithm that solves the chinese postman problem. Conditions:
- Each edge has to be visited once
- Any edge can be travelled more than once
- The objective is to find the solution that travels the least total distance
"""

import time
import numpy as np
import networkx as nx
import itertools
import copy
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
    
def calculate_distance(graph, path):
    ''' Computes distance of travelling a path through a weighted graph. 
    Path is given as [node1, node2, node3, ...] '''
    total_distance = 0
    for ind in range(len(path[:-1])):
        node1 = path[ind]
        node2 = path[ind+1]
        distance = graph[node1][node2]['length']
        total_distance += distance
    return total_distance
        
    
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
    
def Get_Min_Weight_Matching(G):
    "Get_Min_Weight_Matching takes as input a complete weighted undirected networkx graph G, and returns a dictionary called matching, such that matching[v] == w if and only if the node v of G is matched with node w."
    nodes = G.nodes()
    # Get negative of distance matrix
    NegDistMat = -nx.to_numpy_matrix(G, weight='length')
    # Form new graph to obtain the matching
    Gnew = nx.from_numpy_matrix(NegDistMat, create_using=None)
    # Get min-weight matching
    matching = nx.max_weight_matching(Gnew, maxcardinality=True)
    # Return the matching
    matching_with_correct_nodes = {nodes[i]:nodes[matching[i]] for i in matching.keys()}    
    
    return matching_with_correct_nodes

    
def analyze_odd_nodes(graph):
    ''' Returns list of nodes with odd number of neighbors and the distance
    between each pair '''
    
    node_order = add_node_order_to_graph(graph)
    odd_node_list = [node for node,order in node_order.items() if order%2 != 0]
    odd_node_pairing_distance = pd.DataFrame(0., index=odd_node_list, columns=odd_node_list)
    for node1, node2 in itertools.combinations(odd_node_list,2):
        min_distance = nx.shortest_path_length(graph, node1, node2, weight='length')
        odd_node_pairing_distance[node1][node2] = min_distance
        odd_node_pairing_distance[node2][node1] = min_distance
        
    return odd_node_list, odd_node_pairing_distance
    
        
def find_eulerian_path_same_startend(graph, start):
    ''' Finds an eulerian path in the graph that returns to the starting point'''
    if nx.is_eulerian(graph) == False:
        print 'Graph is not eulerian, aborting'
        return
        
    eulerian_edge_path = nx.eulerian_circuit(graph, source=start)
    path = []
    for edge in eulerian_edge_path:
        path.append(edge[0])
    path.append(edge[1])
    return path
        
def find_eulerian_path_different_startend(graph, start, end):
    ''' Finds an eulerian path in the graph that (tries to) start and end at different points.'''
    # Deterministic decision tree to get the path
    path = [start]
    node = start
    temporal_graph = copy.deepcopy(graph)
    while True:
        #Order neighbors by alphabet/number and remaining visits
        neighbor_list = sorted(temporal_graph.neighbors(node))
        neighbor_visits = {n:len(temporal_graph[node][n]) for n in neighbor_list}
        neighbor_list = sorted(neighbor_visits.keys(), key=lambda x:neighbor_visits[x], reverse=True)
        while True:
#            print 'change',node, neighbor_list, neighbor_visits
            #Loop until an appropriate (or None) neighbors are found
            next_non_visited_neighbor = next(iter(neighbor_list), None)

            if next_non_visited_neighbor == None:
                #If no neighbor or ending is found break
                break
            
            elif next_non_visited_neighbor == end and len(neighbor_list)>1 and neighbor_visits[end] == 1:
                #Avoid the last visit to the ending node as long as other alternatives exist
                neighbor_list = neighbor_list[1:] + [end]
                continue

            #Delete crossed edges, unless they leave the graph unconnected
            temporal_graph.remove_edge(node, next_non_visited_neighbor)
            if len(temporal_graph.neighbors(node)) == 0:
                temporal_graph.remove_node(node)
            if nx.is_connected(temporal_graph) == False:
                original_length = graph[node][next_non_visited_neighbor][0]['length']
                temporal_graph.add_edge(node, next_non_visited_neighbor, attr_dict = {'length':original_length})
                del neighbor_visits[next_non_visited_neighbor]
                continue
            break

        if next_non_visited_neighbor == None:
            #No more options -> cleared
            break

        node = next_non_visited_neighbor
        path.append(node)
    return path

def create_dense_graph(nodes, distance_matrix):
    ''' Returns a graph in which all nodes are connected, with weights according to their distance matrix '''
    dense_graph = nx.Graph()
    for n1 in nodes:
        for n2 in nodes:
            if n1 == n2:
                continue
            dense_graph.add_edge(n1, n2, attr_dict = {'length':distance_matrix[n1][n2]})
    return dense_graph
    
def matching_algorithm_efficient(odd_node_list, odd_node_pairing_distance):
    ''' Finds the edges that must be backtracked to cover the entire graph while
    minimizing the weights. Uses original formulation from Edmund and Johnson's '''
    dense_odd_node_graph = create_dense_graph(odd_node_list, odd_node_pairing_distance)
    
#    ####### SAVE ODD NODE DISTANCE MATRIX #######
#
#    dense_distance_matrix = nx.to_numpy_matrix(dense_odd_node_graph, weight='length')
#    np.savetxt("dense_graph.csv", dense_distance_matrix, delimiter=",")    
#
#    ####### SAVE ODD NODE DISTANCE MATRIX #######

    
    optimal_odd_node_matching = Get_Min_Weight_Matching(dense_odd_node_graph)

    node_list = optimal_odd_node_matching.keys()
    optimal_odd_node_pairs = []
    added_distance = 0
    while True:
        node1 = next(iter(node_list), None)
        if node1 == None:
            break
        
        node2 = optimal_odd_node_matching[node1]
        node_list.remove(node1)
        node_list.remove(node2)
        optimal_odd_node_pairs.append((node1,node2))
        added_distance += odd_node_pairing_distance[node1][node2]
        
    return added_distance, optimal_odd_node_pairs
    
def matching_algorithm_original(odd_node_list, odd_node_pairing_distance):
    ''' Finds the optimal matching of odd nodes by iterating over all possibilities
    and finding the one with minimum added distance'''    
    possible_pair_lists = find_all_possible_list_of_pairs(odd_node_list)
    best_pair_total_distance = np.inf
    for pair_list in possible_pair_lists:
        total_distance = 0
        for node1, node2 in pair_list:
            total_distance += odd_node_pairing_distance[node1][node2]
        if total_distance < best_pair_total_distance:
            best_pair_list = pair_list
            best_pair_total_distance = total_distance
    return best_pair_total_distance, best_pair_list
    
def find_optimal_graph_extension(graph, odd_node_list, odd_node_pairing_distance, matching_algorithm = 'efficient'):
    ''' Finds the edges that must be added to the graph to make it Eulerian while
    minimizing the extra distance '''
    
    if len(odd_node_list) < 2:
        dict_of_graph_distances = nx.get_edge_attributes(graph, 'length')
        least_total_distance = sum(dict_of_graph_distances.values())
        backtracked_edges = []
        return least_total_distance, backtracked_edges
        
    if matching_algorithm == 'original':
        best_pair_total_distance, best_pair_list = matching_algorithm_original(odd_node_list, odd_node_pairing_distance)
        
    elif matching_algorithm == 'efficient':
        best_pair_total_distance, best_pair_list = matching_algorithm_efficient(odd_node_list, odd_node_pairing_distance)

    dict_of_graph_distances = nx.get_edge_attributes(graph, 'length')
    total_graph_distance = sum(dict_of_graph_distances.values())
    least_total_distance = total_graph_distance + best_pair_total_distance
    
#    print total_graph_distance, best_pair_total_distance
    
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

    
def solve_chinese_postman_problem(graph, start=None, end=None, matching_algorithm = 'efficient', is_multigraph = False):
    ''' Solves the chinese postman problem using the classical solution.
    'Efficient' algorithm uses Edmund and Johnson's solution (1976) to find optimal edges to be added.
    'Original' tries all possible edge combinations and selects the smallest one (slow) '''
    
    #Deals with start/end nodes and computes least total distance and backtracked nodes
    odd_node_list, odd_node_pairing_distance = analyze_odd_nodes(graph)
    if start == None or end == None:
        possible_startend_pairs= get_all_possible_pairings(odd_node_list, start, end)
        startend_dict = {} #{(start,end) : (least_total_distance, backtracked_edges)}
        for startend_pair in possible_startend_pairs:
            current_odd_node_list = [n for n in odd_node_list if n not in startend_pair]
            least_total_distance, backtracked_edges = find_optimal_graph_extension(graph, current_odd_node_list, odd_node_pairing_distance, matching_algorithm = matching_algorithm)
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
            least_total_distance, backtracked_edges = find_optimal_graph_extension(graph, current_odd_node_list, odd_node_pairing_distance, matching_algorithm = matching_algorithm)

        else:
            least_total_distance, backtracked_edges = find_optimal_graph_extension(graph, odd_node_list, odd_node_pairing_distance, matching_algorithm = matching_algorithm)
    
    #Create enhanced eulerian graph in which backtracked edges are repeated
    graph_extended = nx.MultiGraph(graph)
    for node1,node2 in backtracked_edges:
        graph_extended.add_edge(node1, node2, attr_dict = {'length': graph[node1][node2]['length']})
     
    if start == end:
        path = find_eulerian_path_same_startend(graph_extended, start)
    else:
        path = find_eulerian_path_different_startend(graph_extended, start, end)
            
    check_distance = 0
    prev_node = start
    for node in path[1:]:
        check_distance += graph.edge[prev_node][node]['length']
        prev_node = node

    if least_total_distance < check_distance:
        diff = check_distance - least_total_distance
        #To avoid floating point errors, only consider it a problem if the difference is more than 1% the least distance
        if float(diff)/least_total_distance > 0.01:
            print 'WARNING: PATH OF FINAL DISTANCE IS NOT OPTIMAL, TRAVELLED %.3f MORE' %diff
    
    check_path = check_solution(graph, path)
    if check_path == False:
        print 'WARNING: NOT ALL EDGES HAVE BEEN VISITED'
    
    if is_multigraph == True:
        # Modify duplicated nodes
        node_type = type(path[0])
        new_path = []
        for node in path:
            if type(node) == str:
                if "\'" in node:
                    node = node.replace("\'", "")
                    node = node_type(node)
            if len(new_path) > 0 and node == new_path[-1]:
                continue
            new_path.append(node)
        path = new_path

    return least_total_distance, path
    
def check_solution(graph, path):
    ''' Check if the path is a solution of the graph '''
    edges = graph.edges()
    for ind in range(len(path[:-1])):
        edge = (path[ind], path[ind+1])
        
        try:
            edges.remove(edge)
        except ValueError:
            pass
        
        try:
            edges.remove(edge[::-1])
        except ValueError:
            pass
        
    is_correct = False
    if len(edges) == 0:
        is_correct = True
    else:
        print 'missed edges', edges
    return is_correct
        

def check_solution_two_way(graph, path):
    ''' Check if a normal graph has every edge crossed twice '''
    edges = graph.edges()*2
    for ind in range(len(path[:-1])):
        edge = (path[ind], path[ind+1])
        
        
        try:
            edges.remove(edge)
        except ValueError:
            pass
        
        try:
            edges.remove(edge[::-1])
        except ValueError:
            pass
        
    is_correct = False
    if len(edges) == 0:
        is_correct = True
    else:
        print 'missed edges', edges
    return is_correct
            
    

def main():
    is_multigraph = False
    
#    graph = example_graphs.create_test_array(); start = 1; end = 1
#    graph = example_graphs.create_test_array_with_dead_end(); start = 1; end = 1

#    graph = example_graphs.create_test_array_notebook(); start = 'A'; end = 'H'
    graph = example_graphs.create_test_array_worked_example_33(); start = 'A'; end = 'A'

    graph = example_graphs.manual_map_one(); start = 0; end = 0
#    print calculate_distance(graph, [0, 1, 6, 8, 7, 11, 14, 18, 21, 20, 13, 12, 10, 12, 17, 20, 13, 14, 18, 15, 19, 22, 23, 21, 18, 15, 11, 7, 5, 6, 8, 9, 16, 19, 22, 16, 9, 3, 4, 3, 2, 1, 0])
#    
#    graph = example_graphs.create_test_array_multigraph(); start='A'; end = 'A'; is_multigraph=True
#    graph = example_graphs.manual_map_two_way(); start=0; end = 0; is_multigraph=True

    least_total_distance, path = solve_chinese_postman_problem(graph, start=start, end=end, matching_algorithm='efficient', is_multigraph=is_multigraph)
    print path, least_total_distance
    example_graphs.draw_network_with_labels(graph)
    
#    print check_solution_two_way(example_graphs.manual_map_one(), path)
    
#    print calculate_distance(example_graphs.manual_map_one(), [0, 1, 2, 1, 6, 8, 7, 11, 14, 13, 12, 17, 20, 17, 12, 13, 14, 18, 21, 20, 13, 20, 21, 23, 22, 19, 15, 18, 14, 13, 12, 10, 12, 13, 12, 17, 20, 13, 14, 11, 15, 19, 15, 11, 7, 8, 9, 3, 4, 3, 2, 3, 9, 16, 19, 22, 23, 21, 18, 15, 19, 16, 22, 16, 9, 8, 6, 5, 7, 5, 6, 1, 0])
#    print check_solution_two_way(example_graphs.manual_map_one(), [0, 1, 2, 1, 6, 8, 7, 11, 14, 13, 12, 17, 20, 17, 12, 13, 14, 18, 21, 20, 13, 20, 21, 23, 22, 19, 15, 18, 14, 13, 12, 10, 12, 13, 12, 17, 20, 13, 14, 11, 15, 19, 15, 11, 7, 8, 9, 3, 4, 3, 2, 3, 9, 16, 19, 22, 23, 21, 18, 15, 19, 16, 22, 16, 9, 8, 6, 5, 7, 5, 6, 1, 0])

    


    

if __name__ == '__main__':
    tt = time.time()
    main()
    print 'Ellapsed time: %.4f seconds' %(time.time() - tt)
    
    
    
    
#    g = nx.MultiGraph()
#    g.add_edges_from([(1,2),(2,3),(2,4),(3,5),(4,5),(5,6),(2,3),(3,5)])
##    g.add_edges_from([(2,3),(3,5)])
#    example_graphs.draw_network_with_labels(g)
#    nx.set_edge_attributes(g,'length',1)
#    print find_eulerian_path_different_startend(g, 1, 6)
