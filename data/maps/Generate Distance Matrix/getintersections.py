import csv
import os

path = '/Users/Carl/Desktop/'

os.chdir(path)

with open(path + 'SNP1.csv', 'rb') as csvfile:
    textfile = open('STF1.txt', 'w')

    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    data = list(spamreader)

    for i in data:
        for j in data:
            if i != j:
                streetA = str(i)[2:]
                streetA = streetA[:-2]
                streetB = str(j)[2:]
                streetB = streetB[:-2]

                # replacing the unneccesary letters

                streetA = streetA.replace("', '", " ")
                streetB = streetB.replace("', '", " ")

                string = streetA + ' & ' + streetB
                textfile.write(string + "\n")
            #print(string)
    textfile.close()
    textfile2 = open('STF1_streets.txt', 'w')

    for i in data:
        street = str(i)[2:]
        street = street[:-2]
        street = street.replace("', '", " ")
        textfile2.write(street + "\n")

    textfile2.close()
