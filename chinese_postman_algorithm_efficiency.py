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
    
def plot_efficient():
    sizes = 10.**np.linspace(1.,2.,10)
    sizes = np.linspace(5,30,10)
#    sizes = [125]
#    sizes = [10,20,30,40,50]
    reps = 10 #Times each size is repeated
    times = []
    for size in sizes:
        print size
        
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
            cpa.solve_chinese_postman_problem(graph, start = 0, end=0, matching_algorithm = 'efficient')
            ellapsed_time = time.time() - tt
            av_time += ellapsed_time
        av_time /= reps
        times.append(av_time)
        
    fsize = 22
    
    plt.plot(sizes, times, 'o', ms= 12, label = 'Averaged computational time')

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

    plt.plot(xx, yy, 'r--', lw = 4, label = '3d order polynomial fit, R2 = %.3f' %r2)
    plt.legend(loc=2, fontsize = fsize/1.2)
    
    plt.tick_params(labelsize=fsize)
    plt.xlabel('Number of nodes',fontsize = fsize)
    plt.ylabel('Time (s)', fontsize = fsize)

def plot_original():
    sizes = 10.**np.linspace(1.,2.,10)
    sizes = np.linspace(5,16,3)
    sizes = [10,11,12,13,14,15,16]
#    sizes = [125]
#    sizes = [10,20,30,40,50]
    reps = 10 #Times each size is repeated
    times = []
    for size in sizes:
        print size
        
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
            cpa.solve_chinese_postman_problem(graph, start = 0, end=0, matching_algorithm = 'original')
            ellapsed_time = time.time() - tt
            av_time += ellapsed_time
            
            if size > 15: #Takes too long to make repetitions
                av_time = av_time * reps
                break
        av_time /= reps
        times.append(av_time)
        
        fsize = 22
    
    plt.plot(sizes, times, 'o', ms= 12, label = 'Averaged computational time')

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

    plt.plot(xx, yy, 'r--', lw = 4, label = '3d order polynomial fit, R2 = %.3f' %r2)
    plt.legend(loc=2, fontsize = fsize/1.2)
    
    plt.tick_params(labelsize=fsize)
    plt.xlabel('Number of nodes',fontsize = fsize)
    plt.ylabel('Time (s)', fontsize = fsize)
    
    
def main():

    plot_efficient()
    
#    plot_original()
    



    
    
    
        
        
    
    
    
    





if __name__ == '__main__':
    ttt = time.time()
    main()
    print 'Ellapsed time: %.2f seconds' %(time.time() - ttt)
    plt.show()

