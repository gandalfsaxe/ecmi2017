#%%
import pandas as pd
import numpy as np
import os

os.chdir('/Users/gandalf/Library/Mobile Documents/3L68KQB4HG~com~readdle~CommonDocuments/Documents/ecmi2017/map-plotting') # gandalfs computer
#os.chdir(os.path.dirname(__file__)) # general, but does not work in spyder

data = pd.read_csv('route.csv')
data = np.array(data)
data = data.tolist()

file = open('route-lat-lon.txt', 'w')

for line in data:
  file.write("%s,\n" % line)