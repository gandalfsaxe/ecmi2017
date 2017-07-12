# python 2
# idea here will be to compute efficient but half random paths to find an efficient one

import networkx as nx
import random

G = nx.Graph()

edges = [('A', 'B', 6), ('A', 'C', 6), ('A', 'D', 7), ('B', 'D', 5), ('B', 'E', 10), ('C', 'D', 8), ('C', 'F', 7),
         ('D', 'E', 6), ('D', 'G', 9), ('D', 'F', 11), ('E', 'G', 8), ('E', 'H', 7), ('F', 'G', 10), ('F', 'H', 9),
         ('G', 'H', 5)]

for start, end, length in edges:
    G.add_edge(start, end, attr_dict={'length' : length, 'visits' : 0})

# every walk should start and end at point A

W = ['A']


def valuevertex(A, Walk):
    """return value of vertex"""
    quality_A = 0

    # first check if A or B cause a U-turn
    if A == Walk[-2]:
        quality_A += 10

    # now number of visits for this point
    quality_A += Walk.count(A)

    # length of the path
    now = Walk[-1]

    quality_A += G[now][A]['length']

    # number of visits of the path
    quality_A += 10*G[now][A]['visits']

    return quality_A


def nextpoint(walk, graph):
        """this function delivers a decision for the next point to go, looking at the graph and the walk"""
        nodenow = walk[-1]
        points = nodenow.neighbours(walk[-1])
        # sort the points to have best vertex first
        if len(points)>1:
            i = 0
            # connect value to every point
            while i < len(points):
                points[i].append(valuevertex(points[i], walk))
                i += 1
            # sort the list of points by the value connected
            points.sort(key=lambda x: x[1])

        else:
            return points[0]
        # now decide which point should be next, with highest chance for best point and lowest chance for worst point
        borders = range(1, len(points) + 1)[::-1]  # e.g. for 3 points gives [3,2,1]
        # generate a random number to decide which point to go to
        r_n = random.uniform(0, 1) * sum(borders)
        # decide which point to go next
        for i in range(0, len(points)):
                if r_n < sum(borders[:i]):
                    result = points[i]
                    break
                else:
                    raise Exception('unexpected occurance in nextpoint function')
        return result


def my_endwhile(walk, graph):
    """returns true, if all edges where visited and we reached final position"""
    # first check if current position is the start position
    if len(walk) == 1 or walk[0] != walk[-1]:
        return False
    # only calculate if there is a non visited edge, if we reached end position
    elif len(walk) > 1 and walk[0] == walk[-1]:

        nonvisited_edges = list(n for n in graph if graph.edge[n]['visits'] == 0)
        if len(nonvisited_edges) == 0:
            return True
        else: return False

def generate_walk(start, graph):
    """this function generates a walk through the graph with given start point
    start point is also ending point

    output is the generated walk and the fitness as a list"""
    g = graph
    walk = start
    fitness = 0

    # repeat the iteration until the graph is fully covered and current position is starting position
    while not my_endwhile(walk, g):
        now = walk[-1]
        next_vert = nextpoint(walk, g)
        walk.append(next_vert)
        try:
            g.edge[now][next_vert]['visits'] += 1
            fitness += g.edge[now][next_vert]['length']
        except:
            g.edge[next_vert][now]['visits'] += 1
            fitness += g.edge[next_vert][now]['length']

    return [walk, fitness]


if __name__ == '__main__':
    Value = generate_walk(W, G)









