import mplleaflet
import matplotlib.pyplot as plt
import networkx as nx

GG=nx.Graph()

pos = {u'Afghanistan': [66.00473365578554, 33.83523072784668],
 u'Aland': [19.944009818523348, 60.23133494165451],
 u'Albania': [20.04983396108883, 41.14244989474517],
 u'Algeria': [2.617323009197829, 28.158938494487625]}

fig, ax = plt.subplots()

nx.draw_networkx_nodes(GG,pos=pos,node_size=10,node_color='red',edge_color='k',alpha=.5, with_labels=True)
nx.draw_networkx_edges(GG,pos=pos,edge_color='gray', alpha=.1)
nx.draw_networkx_labels(GG,pos, label_pos =10.3)

mplleaflet.display(fig=ax.figure)

#%%

import mplleaflet
import matplotlib.pyplot as plt
import networkx as nx

# Load longitude, latitude data
plt.hold(True)
# Plot the data as a blue line with red squares on top
# Just plot longitude vs. latitude
plt.plot(longitude, latitude, 'b') # Draw blue line
plt.plot(longitude, latitude, 'rs') # Draw red squares