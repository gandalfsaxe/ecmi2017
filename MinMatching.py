import networkx as nx

def Get_Min_Weight_Matching(G):
    "Get_Min_Weight_Matching takes as input a complete weighted undirected networkx graph G, and returns a dictionary called matching, such that matching[v] == w if and only if the node v of G is matched with node w."
    # Get negative of distance matrix
    NegDistMat = -nx.to_numpy_matrix(G, weight='weight')
    # Form new graph to obtain the matching
    Gnew = nx.from_numpy_matrix(NegDistMat, create_using=None)
    # Get min-weight matching
    matching = nx.max_weight_matching(Gnew, maxcardinality=True)
    # Return the matching
    return matching
