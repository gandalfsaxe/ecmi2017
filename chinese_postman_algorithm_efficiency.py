# -*- coding: utf-8 -*-
"""
Script to check and plot the efficiency of the postman algorithm
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sklearn as skl
from sklearn import linear_model
from sklearn import pipeline
import example_graphs
import chinese_postman_algorithm as cpa
import seaborn as sns 
sns.set_style("darkgrid", {'axes.grid' : False})

def PolynomialRegression(model,degree=15,**kwargs):
    return pipeline.make_pipeline(skl.preprocessing.PolynomialFeatures(degree), model(**kwargs))

def create_polyreg_matrix(x, order = 15):
    if len(x.shape)>1:
        x.ravel()
    polyreg_mat = x
    for i in range(2, order+1):
        polyreg_mat = np.vstack((polyreg_mat, x**i))
    return polyreg_mat.T


def generate_random_graph(size, min_l = 0.01, max_l = 10.):
    graph = nx.barabasi_albert_graph(size, 3)
    length_dict = {}
    for edge in graph.edges_iter():
        length = np.random.uniform(low=min_l, high=max_l)
        length = np.around(length, decimals = 3)
        length_dict[edge] = length
    nx.set_edge_attributes(graph, 'length', length_dict)
    
    return graph
    
def main():

    graph = generate_random_graph(5)   
    
    sizes = 10.**np.linspace(1.,2.,10)
    sizes = [125]
#    sizes = [10,20,30,40,50]
    reps = 1 #Times each size is repeated
    times = []
    for size in sizes:
        
        av_time = 0.    
        
        for i in range(reps):
            graph = generate_random_graph(size)
            #Make the first node odd to start with it
            first_node_order = len(graph.neighbors(0))
            graph[0]
            if first_node_order %2 == 0:
                first_neighbor = graph.neighbors(0)[0]
                graph.remove_edge(0, first_neighbor)
    
            tt = time.time()
            cpa.solve_chinese_postman_problem(graph, start = 0, end=0)
            ellapsed_time = time.time() - tt
            av_time += ellapsed_time
        av_time /= reps
        times.append(av_time)
        
#        example_graphs.draw_network_with_labels(graph)
#        plt.figure(2)
#        nx.draw_circular(graph)
#        plt.figure(1)
    
#    xx = np.linspace(sizes[0], sizes[-1], 1000)
#    yy = xx**3   
#    plt.plot(xx, yy, 'r--')

    
    plt.plot(sizes, times, label = 'Computational times')

    #Polynomial fitting    
    sizes = np.array(sizes)
    sizes = sizes[:, np.newaxis]
    times = np.array(times)
    times = times[:, np.newaxis]
    xx = np.linspace(sizes[0], sizes[-1],100)
    xx = xx[:, np.newaxis]
    regr = PolynomialRegression(linear_model.LinearRegression, degree = 3, normalize = True)
    regr.fit(sizes, times)
    yy = regr.predict(xx)
    r2 = skl.metrics.r2_score(times, regr.predict(sizes))
    print 'r2 score', r2
    print regr.named_steps['linearregression'].coef_

    plt.plot(xx, yy, 'r--', label = '3d degree polyfit, R2 = %.3f' %r2)
    plt.legend(loc='best')
    
    plt.xlabel('Number of nodes')
    plt.ylabel('Time (s)')


    
    
    
    plt.show()
        
        
    
    
    
    





if __name__ == '__main__':
    ttt = time.time()
    main()
    print 'Ellapsed time: %.2f seconds' %(time.time() - ttt)
