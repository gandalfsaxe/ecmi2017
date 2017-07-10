#!python2
# Python 2 script for extracting Google Distance Matrix
# INPUT: name of text file containing list of locations
# OUTPUT: distance matrix as text file

import urllib
import json
import csv

# Google API service URL:
serviceurl = 'https://maps.googleapis.com/maps/api/geocode/json?'

# Names of streets
address1 = 'Lavolantie'
address2 = 'Salpausselankatu'


# String name of intersection
address = address1 + " Lappeenranta, Finland " + " & " + address2 + " Lappeenranta, Finland "


# Open a csv file to save the results
with open('Intersection Coordinates.csv', 'a') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Intersection'] + ['Latitude'] + ['Longitude'] + ['Intersection type'])


# Encode information in url format
url = serviceurl + urllib.urlencode({'address':address,'components':'country:FI','language':'English'})


print
print 'Retrieving', url
connection = urllib.urlopen(url)
data = connection.read()
print 'Retrieved',len(data),'characters'


# Parse JSON to get python dictionary
try: js = json.loads(str(data))
except: js = None
if 'status' not in js or js['status'] != 'OK':
    print '==== Failure To Retrieve ===='
    print data
    exit()

# Extract the relevant intersection information
Int_lat = js['results'][0]['geometry']['location']['lat']
Int_lng = js['results'][0]['geometry']['location']['lng']
LocType = js['results'][0]['geometry']['location_type']


# Print the coorinates
print 'Latitude and Longitude: '
print Int_lat
print Int_lng
print LocType

# Save the data in .csv format
with open('Intersection Coordinates.csv', 'a') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow([address1 + " & " + address2] + [Int_lat] + [Int_lng] + [LocType])



# Uncomment the next two lines to print all of the retrieved JSON
    print "Full GoogleMaps info:"
    print json.dumps(js, indent=4)
