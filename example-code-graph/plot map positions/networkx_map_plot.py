import mplleaflet
import matplotlib.pyplot as plt
import networkx as nx

dic_pos = {u'0': [-77.51885109, 39.18193382],
 u'1': [-76.6688633, 39.18],
 u'2': [-77.2617, 39.1791792],
 u'3': [-77.1927, 39.1782]}

fig, ax = plt.subplots()

GG = nx.Graph()

nx.draw_networkx_nodes(GG,pos=dic_pos,node_size=10,node_color='red',edge_color='k',alpha=.5, with_labels=True)
nx.draw_networkx_edges(GG,pos=dic_pos,edge_color='gray', alpha=.1)
nx.draw_networkx_labels(GG,pos=dic_pos, label_pos =10.3)

mplleaflet.display(fig=ax.figure) 
mplleaflet.show(fig=ax.figure)