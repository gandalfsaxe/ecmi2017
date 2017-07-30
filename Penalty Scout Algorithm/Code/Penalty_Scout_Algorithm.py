# python 2
# use smart heuristics to compute shortest path

import networkx as nx
import random
import numpy as np
import csv
import time


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
    ''' Creates the manual map using a CSV file containing the distance matrix. '''

    # Change the path to get the right file with your distance matrix!
    path = "/Users/Carl/Documents/GitHub/ecmi2017/data/Distance matrix for manual graph/"
    filename = 'DistanceMatrix.csv'

    distance_matrix = np.loadtxt(open(path+filename, "rb"), delimiter=",", skiprows=0)
    graph = create_graph_from_distance_matrix(distance_matrix)

    return graph


def valuevertex(A, Walk, graph, penalty):
    """Returns the penaltyvalue of the vertex. High penalty -> low possibilty to be choosen as next point"""

    quality_a = 0

    # first check if A or B cause a U-turn
    if len(Walk) > 1:
        if A == Walk[-2]:
            quality_a += 10

    # length of the path
    now = Walk[-1]
    quality_a += G[now][A]['length']# * graph[now][A]['visits']

    # number of visits of the path
    quality_a += 2 * penalty * graph[now][A]['visits']

    return quality_a


def nextpoint(walk, graph, penalty):
        """this function delivers a decision for the next point to go, looking at the graph and the walk"""
        nodenow = walk[-1]

        points = graph.neighbors(nodenow)

        # sort the points to have best vertex (with the lowest penalty) first
        if len(points)>1:
            i = 0
            # connect value to every point
            while i < len(points):
                points[i] = [points[i], valuevertex(points[i], walk, graph, penalty)]
                i += 1

            # sort the list of points by the value connected
            points.sort(key=lambda x: x[1])

        else:
            return points[0]

        # now decide which point should be next, with highest chance for best point and lowest chance for worst point
        borders = range(1, len(points) + 1)[::-1] # e.g. for 3 points gives [3,2,1]

        # here the first value gets increased, which results in a higher possibility for the first point
        # we found that this change improves the quality of generated walks
        borders[0] *= 10

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


def generate_walk(start, graph, expectedfitness):

    """this function generates a walk through the graph with given start point
    start point is also ending point

    output is the generated walk and the fitness as a list"""
    g = graph.copy()
    walk = start[:]

    finished = [0]

    # fitness is the length of the path, a lower fitness is better
    fitness = 0

    listofedges = graph.edges()
    fraud_candidate = True
    final_walk = []
    # repeat the iteration until the graph is fully covered and current position is starting position
    while len(finished) != 0:

        # here we compute our next step
        now = walk[-1]
        # where to go next
        next_vert = nextpoint(walk, g, len(listofedges))
        # append next point to the walk
        walk.append(next_vert)

        # increase number of visits for choosen edge and increase fitness
        try:
            g.edge[now][next_vert]['visits'] += 1
            fitness += g.edge[now][next_vert]['length']
        except:
            g.edge[next_vert][now]['visits'] += 1
            fitness += g.edge[now][next_vert]['length']

        try:
            listofedges.remove((next_vert, now))
        except:
            fitness += 0
        try:
            listofedges.remove((now, next_vert))
        except:
            fitness += 0

        # if fitness is higher than expected fitness, there is no need to continue walking
        if fitness > expectedfitness:
            fraud_candidate = False
            break

        # Here we decide, if the while loops needs to end using the my_endwhile function
        if my_endwhile(walk, listofedges):
            final_walk = walk

            finished = []

            break

    return [final_walk, fitness, fraud_candidate]


def start_penalty_scout(start, graph, N):

    G = graph.copy()
    # first generate initial guess of length,
    temp = G.edges(None, True)
    totallength = 0
    for i in temp:
        totallength += i[2]['length']

    # we expect the optimum route to be lower than 2 times the total length of the path, this increases computing time
    expectedfitness = totallength * 2

    W = start[:]

    candidate_old = generate_walk(W, G, expectedfitness)
    iteration = 0

    table = [['iteration', 'fitness', 'walkingpath']]

    for i in range(N):

        candidate_new = generate_walk(W, G, expectedfitness)
        if candidate_new[1] < candidate_old[1]:
            candidate_old = candidate_new
            expectedfitness = candidate_old[1]
            iteration = i
            if candidate_old[2]:
                table.append([iteration, expectedfitness])

    with open('penaltyscoutloglength.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in table]

    return [iteration, candidate_old]


if __name__ == '__main__':

    # generate graph
    G = manual_map_one()
    # Choose start and end point
    W = [0]
    # take the time
    tt = time.time()
    # set the number of iterations
    N = 300
    # calculate the candidate
    candidate = start_penalty_scout(W, G, N)

    print('Possible optimum found with length:')
    print(candidate[1])
    print('After')
    print(candidate[0])
    print('iterations')
    print('time needed:')
    print(str(time.time()-tt))
