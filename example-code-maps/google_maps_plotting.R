library(googleway)

key = "AIzaSyC1tadiJ1aZlp7PfZbXcvxJM-MenS2o_ko"

df <- google_directions(origin = "Melbourne, Australia",
                        destination = "Sydney, Australia",
                        key = key,
                        mode = "driving",
                        simplify = TRUE)

pl <- df$routes$overview_polyline$points