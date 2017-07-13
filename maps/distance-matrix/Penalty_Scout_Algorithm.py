# python 2
# idea here will be to compute efficient but half random paths to find an efficient one

import networkx as nx
import random
import os
import numpy as np
import csv

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


def valuevertex(A, Walk, graph, penalty):
    """return value of vertex"""
    quality_A = 0

    # first check if A or B cause a U-turn
    if len(Walk) > 1:
        if A == Walk[-2]:
            quality_A += 10

    # length of the path
    now = Walk[-1]
    quality_A += G[now][A]['length']# * graph[now][A]['visits']

    # number of visits of the path
    quality_A += 2 * penalty * graph[now][A]['visits']

    return quality_A


def nextpoint(walk, graph, penalty):
        """this function delivers a decision for the next point to go, looking at the graph and the walk"""
        nodenow = walk[-1]
        points = graph.neighbors(nodenow)
        # sort the points to have best vertex first
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


def generate_walk(start, graph, expectedfitness, machines = 1 ):

    """this function generates a walk through the graph with given start point
    start point is also ending point

    output is the generated walk and the fitness as a list"""
    g = graph.copy()
    walk = [start] * machines
    finished = [False] * machines
    fitness = 0
    listofedges = graph.edges()
    fraud_candidate = True
    final_walk = []
    # repeat the iteration until the graph is fully covered and current position is starting position
    while len(finished) != 0:

        for i in range(len(walk)):
            walk_machine = walk[i][:]
            now = walk_machine[-1]
            next_vert = nextpoint(walk_machine, g, len(listofedges))
            walk_machine.append(next_vert)
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
            walk[i] = walk_machine
            if my_endwhile(walk_machine, listofedges):
                final_walk.append(walk_machine)
                walk.remove(walk[i])
                finished = finished[1:]
                break
        if fitness > expectedfitness:
            # we get here only if the fitness of current walk is worse than the expected one
            # the walk is not going to be saved
            fraud_candidate = False
            break

    return [final_walk, fitness, machines, fraud_candidate]


def start_penalty_scout(start, graph, N, machinenumber):

    G = graph.copy()
    # first generate initial guess of length,
    temp = G.edges(None, True)
    totallength = 0
    for i in temp:
        totallength += i[2]['length']

    # fitness in the end should be lower than the following value
    expectedfitness = totallength * 2 * machinenumber

    W = start[:]

    candidate_old = generate_walk(W, G, expectedfitness, machinenumber)
    iteration = 0

    table = [['iteration', 'fitness', 'machines running', 'walkingpath']]

    for i in range(N):

        candidate_new = generate_walk(W, G, expectedfitness, machinenumber)
        if candidate_new[1] < candidate_old[1]:
            candidate_old = candidate_new
            expectedfitness = candidate_old[1]
            iteration = i
            if candidate_old[3]:
                table.append([iteration, expectedfitness, machinenumber, candidate_old[0]])

    with open('penaltyscoutlog.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in table]

    return [iteration, candidate_old]

if __name__ == '__main__':

    G = manual_map_one()
    W = [0]

    temp = G.edges(None, True)
    totallength = 0

    for i in temp:
        totallength += i[2]['length']

    print(totallength)

    """with multiple machines it is important to look at the covered streets every step,
     if they are covered we should send them both home along the shortest path """

    machinenumber = 1

    N = 100000

    candidate = start_penalty_scout(W, G, N, machinenumber)
    iteration = candidate[0]

    print('END OF PROCESS, possible optimum is:')
    print(candidate[1])
    print('at iteration number:')
    print(iteration)

    file = open('solution.txt', 'w')
    file.truncate()
    file.write('Number of iterations tested: ' + str(N) + "\n")
    file.write('Fitness of solution found: ' + str(candidate[1][1]) + "\n")
    file.write('Path to follow is: ' + "\n")
    file.write(str(candidate[0]))

    file.close()









