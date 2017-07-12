# python 2
# idea here will be to compute efficient but half random paths to find an efficient one

import networkx as nx
import random
import os
import numpy as np

"""
G = nx.Graph()

edges = [('A', 'B', 6), ('A', 'C', 6), ('A', 'D', 7), ('B', 'D', 5), ('B', 'E', 10), ('C', 'D', 8), ('C', 'F', 7),
         ('D', 'E', 6), ('D', 'G', 9), ('D', 'F', 11), ('E', 'G', 8), ('E', 'H', 7), ('F', 'G', 10), ('F', 'H', 9),
         ('G', 'H', 5)]

for start, end, length in edges:
    G.add_edge(start, end, attr_dict={'length' : length, 'visits' : 0})
"""
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
                graph.add_edge(i,j, attr_dict={'length' : length, 'visits' : 0})
    return graph

def manual_map_one():
    ''' Creates the manual map '''
    path = "/Users/Carl/Documents/GitHub/ecmi2017/data/Distance matrix for manual graph/"
    filename = 'DistanceMatrix.csv'
    distance_matrix = np.loadtxt(open(path+filename, "rb"), delimiter=",", skiprows=0)
    graph = create_graph_from_distance_matrix(distance_matrix)
    return graph

# every walk should start and end at point A

G = manual_map_one()
W = [0]


def valuevertex(A, Walk):
    """return value of vertex"""
    quality_A = 0

    # first check if A or B cause a U-turn
    if len(Walk) > 1:
        if A == Walk[-2]:
            quality_A += 10

    # length of the path
    now = Walk[-1]
    # quality_A += G[now][A]['length']

    # number of visits of the path
    quality_A += 50*G[now][A]['visits']

    return quality_A


def nextpoint(walk, graph):
        """this function delivers a decision for the next point to go, looking at the graph and the walk"""
        nodenow = walk[-1]
        points = graph.neighbors(nodenow)
        # sort the points to have best vertex first
        if len(points)>1:
            i = 0
            # connect value to every point
            while i < len(points):
                points[i] = [points[i], valuevertex(points[i], walk)]
                i += 1
            # sort the list of points by the value connected
            points.sort(key=lambda x: x[1])

        else:
            return points[0]

        # now decide which point should be next, with highest chance for best point and lowest chance for worst point
        borders = range(1, len(points) + 1)[::-1] # e.g. for 3 points gives [3,2,1]

        borders[0] *= 5
        # s = pd.Series(borders)
        # borders = ((s * 10).tolist())

        # generate a random number to decide which point to go to
        r_n = random.uniform(0, 1) * sum(borders)
        # decide which point to go next
        for i in range(1, len(points) + 1):

                if r_n < sum(borders[:i]):
                    result = points[i-1][0]
                    break

        return result


def my_endwhile(walk, list):
    """returns true, if all edges where visited and we reached final position"""
    # first check if current position is the start position
    if len(walk) == 1 or walk[0] != walk[-1]:
        return False
    # only calculate if there is a non visited edge, if we reached end position
    elif len(walk) > 1 and walk[0] == walk[-1]:
        if len(list)==0:
            return True
        else: return False

def generate_walk(start, graph):

    """this function generates a walk through the graph with given start point
    start point is also ending point

    output is the generated walk and the fitness as a list"""
    g = graph
    walk = start
    fitness = 0
    listofedges = graph.edges()

    # repeat the iteration until the graph is fully covered and current position is starting position
    while not my_endwhile(walk, listofedges):
        now = walk[-1]
        next_vert = nextpoint(walk, g)
        walk.append(next_vert)
        try:
            g.edge[now][next_vert]['visits'] += 1
            fitness += g.edge[now][next_vert]['length']
        except:
            g.edge[next_vert][now]['visits'] += 1
            fitness += g.edge[next_vert][now]['length']
        try:
            listofedges.remove((next_vert, now))
        except:
            fitness += 0
        try:
            listofedges.remove((now, next_vert))
        except:
            fitness += 0

    return [walk, fitness]


if __name__ == '__main__':


    candidate_old = generate_walk(W, G)
    iteration = 0
    N = 1000
    for i in range(N):
        candidate_new = generate_walk(W, G)
        if candidate_new[1] < candidate_old[1]:
            candidate_old = candidate_new
            iteration = i
    print('END OF PROCESS, possible optimum is:')
    print(candidate_old[1])
    print('at iteration number:')
    print(iteration)








