#!python2
# Python 2 script for extracting Google Distance Matrix
# INPUT: name of text file containing list of locations
# OUTPUT: distance matrix as text file

import urllib
import json
import csv

def getdistancematrix(tablepath, MatrixName, APIKEY):


    """
    :param tablepath:   String, containing the path leading to the table (csv) of intersections with coordinates. Following
                        format is expected (columnwise) A: Street A ; B: Street B; C: Latitude of intersection; D:
                        Longitude of intersection
    :param MatrixName:  String, containing path to file which is going to be edited to be the csv table as distance
                        matrix
    :param APIKEY:      String, API key which allows the distance API requests to google

    :return:            No specific output
    """

    # Google API service URL and API key:
    serviceurl = 'https://maps.googleapis.com/maps/api/distancematrix/json?'
    key = APIKEY

    datafile = open(tablepath, 'r')
    datareader = csv.reader(datafile, delimiter=';')
    data = []
    for row in datareader:
        data.append(row[0].split(","))
    distancelist = []
    # now every i, j is an element of the list, containing rows of the csv file
    for i in data:
        for j in data:
            # are intersections neighbours?
            # add more sophisticated neighbour function here
            # Name of intersections
            intersec1_name = str(j[0]) + ',' + str(j[1])
            intersec2_name = str(i[0]) + ',' + str(i[1])
            if (j[0] in i or j[1] in i) and intersec1_name != intersec2_name:
                # Coordinates of intersections
                intersec1 = str(j[2]) + ',' + str(j[3])
                intersec2 = str(i[2]) + ',' + str(i[3])



                # origin
                origins = intersec1

                # units
                units = 'metric'

                # destinations
                destinations = intersec2


                # encode information in url format
                url = serviceurl + urllib.urlencode({'units':units, 'origins':origins, 'destinations':destinations, 'key':key})

                # print 'Retrieving', url
                connection = urllib.urlopen(url)
                data2 = connection.read()
                # print 'Retrieved',len(data2),'characters'

                # Parse JSON to get python dictionary
                try: js = json.loads(str(data2))
                except: js = None
                if 'status' not in js or js['status'] != 'OK':
                    print '==== Failure To Retrieve ===='
                    print(data2)
                    exit()

                # Extract the relevant intersection information
                distance = js['rows'][0]['elements'][0]['distance']['value']

                # Save the data in list format
                distancelist += [[[str(j[0]), str(j[1])], [str(i[0]),str(i[1])], distance]]

    # convert the txt file into distance matrix in csv format plus vector describing order of intersections
    intersectionlist = []

    for i in distancelist:
        intersectionlist += [sorted([i[0][0], i[0][1]])]

    intersectionlist = sorted(intersectionlist)
    sortedlist = [intersectionlist[i] for i in range(len(intersectionlist)) if i == 0 or intersectionlist[i] != intersectionlist[i - 1]]

    intersectionlist = sortedlist

    # define size of distance matrix
    N = len(intersectionlist)

    # create a list as matrix
    mylist = [[0] * N] * N

    for i in distancelist:
        # find coordinates of intersectionpairs
        x = intersectionlist.index(i[0])
        y = intersectionlist.index(i[1])
        mylist[x][y] = i[2]
        mylist[y][x] = i[2]


    with open(MatrixName, 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(mylist)

# call the function with path

pathnow = '/Users/Carl/Google Drive/Team Lambda/Data/Intersection coordinates/I LK JK+PP Intersections.csv'
matrixpath = '/Users/Carl/Google Drive/Team Lambda/Data/Intersection coordinates/Distancematrix.csv'
key = 'AIzaSyDKdoYG4dPg8JwBkTZ-ppCTpD8rSRMGLNk'

getdistancematrix(pathnow, matrixpath, key)