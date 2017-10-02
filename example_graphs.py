# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:33:20 2017

@author: Albert
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns


def draw_network_nice(G, labels = None, repeat_edges=[], pos=None):
    ''' Draws network with:
        - Node labels (big)
        - Spring layout '''
    
    nodesize = 350
    fontsize = 12
    fontsize_edge = 8
    w = 3
    
    nodesize = 900
    fontsize = 17
    fontsize_edge = 14
    w = 3
    
    if type(labels) != type(None):
        nx.relabel_nodes(G, labels)
        
    if type(pos) == type(None):
        pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=nodesize)
    repeat_edge_positions = {} #{repeated edge : }
    pos_up, pos_down = {},{}
    separation = .015
    for n1,n2 in repeat_edges:
        pos_up = {u:v for u,v in pos.items()}
        pos_down = {u:v for u,v in pos.items()}
        vec = pos[n2] - pos[n1]
        vec_perp = [-vec[1], vec[0]]
        vec_perp = vec_perp/np.sqrt(vec_perp[0]**2 + vec_perp[1]**2)
        pos_up[n1] = pos_up[n1] + separation*vec_perp
        pos_up[n2] = pos_up[n2] + separation*vec_perp
        pos_down[n1] = pos_down[n1] - separation*vec_perp
        pos_down[n2] = pos_down[n2] - separation*vec_perp
        repeat_edge_positions[(n1,n2)] = (pos_up, pos_down)
        
#    pos_up = {u:v+[0.01,0.01] for u,v in pos.items()}
#    pos_down = {u:v-[0.01, 0.01] for u,v in pos.items()}
        
        
    edge_labels = nx.get_edge_attributes(G, 'length')
    for edge in G.edges_iter():
        if edge not in repeat_edges:
            nx.draw_networkx_edges(G, pos, edgelist = [edge], width=w, alpha=0.75, edge_color='indianred')
            nx.draw_networkx_edge_labels(G, pos, edgelist = [edge], edge_labels={edge:edge_labels[edge]}, font_size=fontsize_edge, font_color='b')
        else:
            pos_up, pos_down = repeat_edge_positions[edge]
            nx.draw_networkx_edges(G, pos_up, edgelist = [edge], width=w, alpha=0.75, edge_color='indianred')
            nx.draw_networkx_edges(G, pos_down, edgelist = [edge], width=w, alpha=0.75, edge_color='indianred')
            nx.draw_networkx_edge_labels(G, pos, edgelist = [edge], edge_labels={edge:edge_labels[edge]}, font_size=fontsize_edge, font_color='b')

    labels={}
    for ind, node in enumerate(G.nodes_iter()):
        labels[node] = node
    nx.draw_networkx_labels(G, pos, labels, font_size=fontsize)

    plt.axis('off')
    
    return pos
    
def draw_network_with_labels(graph, layout = nx.spring_layout):
    pos = layout(graph)
    
    labels = nx.get_edge_attributes(graph, 'length')
    nx.draw_networkx(graph, pos=pos)
    plt.axis('off')

    if type(graph) == nx.classes.multigraph.MultiGraph:
        return

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)


def create_graph_from_length_edges(edges, graph_type = nx.Graph):
    ''' Creates graph from edges, format must be (node1, node2, length) '''
    graph = graph_type()
    for start, end, length in edges:
        graph.add_edge(start, end, length=length)
    return graph
    
def create_graph_from_distance_matrix(distance_matrix):
    ''' Creates graph from a matrix in which the element i,j indicates the distance between i and j '''
    node_num = distance_matrix.shape[0]
    node_list = range(node_num)
    graph = nx.Graph()
    graph.add_nodes_from(node_list)
    for i in node_list:
        for j in node_list:
            length = distance_matrix[i][j]
            if length > 0:
                graph.add_edge(i,j, length = length)
    return graph
    
def multigraph_to_graph(multigraph):
    ''' Transforms multigraph to graph by adding new nodes for every repeated edge.
    New nodes are "replicas" of the old ones, and are connected to the old by weights of zero.
    EXAMPLE:
    
    *Multigraph            A == 4,2 == B -- 2 -- C
    
                                    ||
                                    ||
                                   \  /
                                    \/  
    
    *Graph                 A -- 4 -- B -- 2 -- C
                           |         |
                           0         0
                           |         |
                          A' -- 2 -- B'         
    '''
    
    graph = nx.Graph(multigraph)
    all_edges_without_repetition = set(multigraph.edges())
    repeated_edges = multigraph.edges()
    for edge in all_edges_without_repetition:
        repeated_edges.remove(edge)

#    node_type = type(multigraph.nodes()[0])
#    if type(multigraph.nodes()[0]) in [int, float]:
        
    def generate_new_node_name(node):
        return str(node) + '\''
    
    for node1,node2 in graph.edges():
        repetitions = len(multigraph[node1][node2]) - 1
        new_node1, new_node2 = node1, node2
        for i in range(repetitions):
            distance = multigraph[node1][node2][repetitions - i - 1]['length']
            #Add a prime to the node name for each repetition of the edge
            new_node1 = generate_new_node_name(new_node1) 
            new_node2 = generate_new_node_name(new_node2)
            graph.add_edge(node1, new_node1, attr_dict = {'length':0.0001})
            graph.add_edge(node2, new_node2, attr_dict = {'length':0.0001})
            graph.add_edge(new_node1, new_node2, attr_dict = {'length':distance})
        
    return graph     

def make_graph_two_way(graph):
    ''' Transforms a normal graph into a multigraph with all edges duplicated '''
    multigraph = nx.MultiGraph(graph)
    for n1, n2 in graph.edges():
        multigraph.add_edge(n1, n2, attr_dict={'length':graph[n1][n2]['length']})
    return multigraph        
            

def create_test_array():               
    ''' Test array '''                
    edges = [(1,2,2),(2,3,1),(2,4,5),(3,4,2),(3,8,1),(8,6,4),(4,6,3)]
    return create_graph_from_length_edges(edges)
    
def create_test_array_with_dead_end():
    ''' Original array with an added dead-end node '''
    edges = [(1,2,2),(2,3,1),(2,4,5),(3,4,2),(3,8,1),(8,6,4),(4,6,3),(6,7,1)]
    return create_graph_from_length_edges(edges)
    
def create_test_array_notebook():
    ''' Test array from the "chinese postman problem" notebook '''
    edges = [('A','B',50),('A','C',50),('A','D',50),('B','D',50),('C','D',70),('C','G',70),('C','H',120),('B','E',70),('B','F',50),('D','F',60),('E','F',70),('F','H',60),('G','H',70)]
    return create_graph_from_length_edges(edges)
    
def create_test_array_worked_example_33():
    ''' Worked example 3.3 from the notebook '''
    edges = [('A','B',6),('A','C',6),('A','D',7),('B','D',5),('B','E',10),('C','D',8),('C','F',7),('D','E',6),('D','G',9),('D','F',11),('E','G',8),('E','H',7),('F','G',10),('F','H',9),('G','H',5)]
    return create_graph_from_length_edges(edges)
    
def create_test_array_multigraph(return_multigraph=False):
    ''' Crates array with a two-way edge in MultiGraph format '''
    edges = [('A','B',13),('A','C',11),('A','C',11),('A','F',10),('B','C',14),('C','D',9),('D','F',9),('D','E',7),('E','F',6)]
    multigraph = create_graph_from_length_edges(edges, graph_type = nx.MultiGraph)
    if return_multigraph == False:
        graph = multigraph_to_graph(multigraph)
        return graph
    else:
        return multigraph
    
def manual_map_one():
    ''' Creates the manual map '''
    path = os.getcwd() + os.sep + 'Data' + os.sep + 'Distance matrix for manual graph' + os.sep
    filename = 'DistanceMatrix.csv'
    distance_matrix = np.loadtxt(open(path+filename, "rb"), delimiter=",", skiprows=0)
    graph = create_graph_from_distance_matrix(distance_matrix)
    return graph
    
def manual_map_two_way():
    ''' Creates the manual map with two way streets'''
    graph = manual_map_one()
    multigraph = make_graph_two_way(graph)
    return multigraph_to_graph(multigraph)
    

    
#def generate_random_graph(size, min_length = 0.0, max_length = 1.0, graph_type = 'barabasi_albert'):
#    ''' Creates a random graph with weights within the specified numbers '''
#    if graph_type == 'barabasi_albert':
#        return nx.barabasi_albert_graph(size, 6)
    
def figure_multigraph():
    
    g = create_test_array_multigraph(return_multigraph = True)
    g = nx.Graph(g)
    plt.figure(1)
    pos = draw_network_nice(g, repeat_edges=[('A', 'C')])
    inc = 0.99
    vecC = pos['C'] - pos['D']
    vecA = pos['A'] - pos['F']    
    
    pos['C\''] = pos['C'] + inc*vecC
    pos['A\''] = pos['A'] + inc*vecA
    
    plt.figure(2)
    g = create_test_array_multigraph()
    g = multigraph_to_graph(g)
    pos = draw_network_nice(g, pos=pos)
    ax = plt.gca()
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    
    plt.figure(1)
    plt.axis([xmin, xmax, ymin, ymax])



    

    
def main():
    
    figure_multigraph()
    plt.show()
    return
    
#    g = create_test_array_worked_example_33()
#    g = create_test_array_multigraph()
#    g = manual_map_one()
    g = create_test_array_with_dead_end()
    
#    print g.edges()   
    
#    draw_network_with_labels(g, layout = nx.spring_layout)
    plt.figure(1)
    pos = draw_network_nice(g, repeat_edges=[(1,2),(3,4),(6,7)])
    plt.figure(2)
    pos = draw_network_nice(g, pos=pos)
    plt.show()
    
if __name__ == '__main__':

    main()