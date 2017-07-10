import networkx as nx
from matplotlib import pyplot

G = nx.Graph()

G.add_node('Hamburg', pos=(53.5672, 10.0285))
G.add_node('Berlin', pos=(52.51704, 13.38792))

pyplot.gca().invert_yaxis()
pyplot.gca().invert_xaxis()

nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, node_size=0)