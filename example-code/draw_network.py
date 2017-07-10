# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:33:20 2017

@author: Albert
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def create_test_array():                               
    G = nx.Graph()
    edges = [(1,2,1),(2,3,1),(2,4,5),(3,4,2),(3,8,1),(8,6,4),(4,6,3)]
    for start, end, length in edges:
        G.add_edge(start, end, length=length)
        
    return G

def draw_network_with_delay(G):
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'length')
    nx.draw_networkx(G, pos=pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.axis('off')



def main():
    G = create_test_array()
    draw_network_with_delay(G)

if __name__ == '__main__':
    main()
