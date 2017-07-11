# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:33:20 2017

@author: Albert
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def create_graph_from_length_edges(edges):
    ''' Creates graph from edges, format must be (node1, node2, length) '''
    graph = nx.Graph()
    for start, end, length in edges:
        graph.add_edge(start, end, length=length)
    return graph

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
    
def draw_network_with_labels(graph):
    pos = nx.spring_layout(graph)
    labels = nx.get_edge_attributes(graph, 'length')
    nx.draw_networkx(graph, pos=pos)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.axis('off')
    
if __name__ == '__main__':
    g = create_test_array_worked_example_33()
    draw_network_with_labels(g)
