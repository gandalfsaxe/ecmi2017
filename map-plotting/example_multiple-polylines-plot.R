library(googleway)

api_key <- "AIzaSyC1tadiJ1aZlp7PfZbXcvxJM-MenS2o_ko"

directions <- google_directions(origin = "St Kilda, Melbourne, Australia",
                                destination = "Geelong, Victoria, Australia",
                                key = api_key)

## the results of the API give you distance in metres, and time in seconds
## so we need to calculate teh speed
spd <- (directions$routes$legs[[1]]$steps[[1]]$distance$value / 1000) / (directions$routes$legs[[1]]$steps[[1]]$duration$value/ 60 / 60)

## then we can start to build the object to use in the plot
## and as we are staying within Google's API, we can use the encoded polyline to plot the routes
## rather than extracting the coordinates
df <- data.frame(speed = spd,
                 polyline = directions$routes$legs[[1]]$steps[[1]]$polyline)



df$floorSpeed <- floor(df$speed)
colours <- seq(1, floor(max(df$speed)))
colours <- colorRampPalette(c("red", "yellow","green"))(length(colours))

df <- merge(df, 
            data.frame(speed = 1:length(colours), 
                       colour = colours), 
            by.x = "floorSpeed", 
            by.y = "speed")

map_key <- "AIzaSyCl_U7ctGUodLktDtw-o4Q1tUA3g8c5e70"

google_map(key = map_key) %>%
  add_polylines(data = df, polyline = "points", stroke_colour = "colour",
                stroke_weight = 5, stroke_opacity = 0.9)