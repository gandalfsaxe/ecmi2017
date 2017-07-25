# ecmi2017

[ECMI Modelling Week 2017](http://www.mafy.lut.fi/ECMIMW2017/) was a mathematics modelling week in Lappeenranta, Finland on July 9th - July 16th 2017. There were 9 teams, one for each of the [problems](http://www.mafy.lut.fi/ECMIMW2017/index.php?page=problems).

Project Lambda was about optimization of snow plowing operations in Skinnarila area near Lappeenranta. We wanted to minimize the distance travelled by the snow plowing vehicles, and if possible, prioritize roads differently.

This README serves two functions:
1. Show our results
2. Document how to use the code.

The second point is especially crucial because we had just 5 days to work this problem; there was no time to make the code structured or pretty.

# Introduction

The problem turned out to be a variation of the [Chinese Postman Problem](https://en.wikipedia.org/wiki/Route_inspection_problem) (CPP) (a.k.a route inspection problem) on a graph having road intersections as nodes and intersection distances as edges. With this basic model, the problem basically had three parts:

1. **Data acquisition**
2. **Solution**
3. **Visualizing solution**

And in summary we found:

1. **Data acquisition:** We opted for the Google Maps API to create our graph, despite the free API-key limitations, since it was very easy to work with. We ended up making a graph of the limited area of priority 1 roads, in a very manual fashion.
2. **Solution**: We basically worked on two solutions: 1. Optimal solution of basic CPP problem using by finding optimal pairings of odd-degree vertices (aided by [Blossom algorithm](https://en.wikipedia.org/wiki/Blossom_algorithm)) in order to restructure the graph into an Eulerian graph, for which the CPP can then solved in polynomial time, and 2. A stochastic algorithm with a cost function that penalize undesirable moves such as traveling the same edge multiple times, making needless U-turns etc.
3. **Visualizing solution**: We used the [Google Maps Javascript API](https://developers.google.com/maps/documentation/javascript/) to animate a symbol on the maps (luckily Google had provided a some [nice sample code](https://developers.google.com/maps/documentation/javascript/examples/overlay-symbol-animate)). This could be done after the solution was converted from a list of nodes to a list of (latitude, longitude) coordinates, using the Google Maps Directions API

# 1. Problem and results
The total length of all the edges in our (one-way) graph is 24.2355 km.
However, since it's not Eulerian, this distance is not achieveable and the snowplower would have to travel a longer distance. We found the theoretical lower bound of a solution to be 30.5275 km, which is exactly what both algorithms found.

## Optimal solver
![Alberithm solution](https://github.com/GandalfSaxe/ecmi2017/blob/master/map-plotting/animation/animation-videos/final-animation-videos/alberithm.gif?raw=true)

Total travel distance: 30.5275 km

## Stochastic solver
![Carlgorithm solution](https://github.com/GandalfSaxe/ecmi2017/blob/master/map-plotting/animation/animation-videos/final-animation-videos/carlgorithm.gif?raw=true)

Total travel distance: 30.5275 km


# 2. Code documentation (HOW-TO guides)
Due to the time constraints, the following is a series of not-so-pretty, but functional workflows of obtaining road graph data, obtaining route solutions and visualizing them using Google Maps API.

## How to obtain graph of road network

## How to obtain solutions for efficient routes

## How to convert list of nodes into list of (latitude, longitude) coordinates

## How to visualize solutions using Google Maps API


# The Team
Team Lambda:
Albert Miguel López

Carl Assmann

Edyta Kabat

Gandalf Saxe:

Matthew Geleta

Sara Battiston


