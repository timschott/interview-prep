### DS-SCRAP###

library(dplyr)
data("airquality")
date("mtcars")
missing_subset <- airquality[is.na(airquality$Ozone) | is.na(airquality$Solar.R), ]
mean(missing_subset$Temp)

for (i in 1: nrow(airquality)) {
  # since[i] <- paste0(abs(as.double(difftime(as.Date(airbnb$host_since[i]), d, units="days"))), " days")
  print("hi")
}

t <- as.factor(airquality$Temp)
levels(t)
length(levels(t))

find_highest_correlated = function(df, column) {
  column_index <- which(colnames(df) == column)
  highest <- 0
  for (i in 1:ncol(df)) {
    val <- abs(cor(df[, i], df[, column_index], use = "complete"))
    if (val > highest & i!= column_index) {
      highest <- val
    }
  }
  return(highest)
}

find_highest_correlated(airquality, 'Temp')

test <- airquality %>%
  mutate(new = Temp * 2)


