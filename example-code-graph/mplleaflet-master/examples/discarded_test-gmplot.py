import json
import os

import matplotlib.pyplot as plt
import numpy as np

import mplleaflet
import pandas as pd

# Load up the geojson data
filename = os.path.join(os.path.dirname(__file__), 'data', 'track.geojson')
with open(filename) as f:
    gj = json.load(f)

# Grab the coordinates (longitude, latitude) from the features, which we
# know are Points
xy = np.array([feat['geometry']['coordinates'] for feat in gj['features'][::10]])

# NEW DATA
filename = os.path.join(os.path.dirname(__file__), '../../data/TEMP', 'priority1-reversed.csv')
df = pd.read_csv(filename, names=['Intersecting Streets', 'Latitude', 'Longitude'])

xy= np.array(df.iloc[:,1:])

# # Plot the path as red dots connected by a blue line
# plt.hold(True)
# plt.plot(xy[:,0], xy[:,1], 'r.')
# #plt.plot(xy[:,0], xy[:,1], 'b')

# root, ext = os.path.splitext(__file__)
# mapfile = root  + '.html'
# # Create the map. Save the file to basic_plot.html. _map.html is the default
# # if 'path' is not specified

# mplleaflet.show(path=mapfile)

longitudes = xy[:,0]
latitudes = xy[:,1]

import gmplot

# gmap = gmplot.GoogleMapPlotter(37.428, -122.145, 16)
gmap = gmplot.GoogleMapPlotter(61.04871, 28.13871, 13)


# gmap.plot(latitudes, longitudes, 'cornflowerblue', edge_width=10)
gmap.scatter(latitudes, longitudes, '#3B0B39', size=40, marker=False)
# gmap.scatter(latitudes, longitudes, 'k', marker=True)
# gmap.heatmap(latitudes, longitudes)

gmap.draw("mymap.html")
