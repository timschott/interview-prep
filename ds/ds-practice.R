## R studying

## Load Data

bike_data <- read.csv("data/2017Q1-capitalbikeshare-tripdata.csv") # df

## Explore Data

## Count NAs per column

for (i in seq(1:length(colnames(bike_data)))) {
  print(sum(is.na(bike_data[, i])))
}

## Attach a column called "ride cost" that is a decimal bt 0 and 10
## Rounded to 2 decimal places, like dollars

### round(val, number_of_places)

### runif(length, min, max)

bike_data$ride_cost <- c(round(runif(length(bike_data$Duration), 0, 10), 2))

## How large is your data set now?

dim(bike_data)

## Change the date columns to have underscores, not periods in their name
colnames(bike_data)[2] <- "start_date"
colnames(bike_data)[3] <- "end_date"

## Convert the date columns to a useable date format

### R's base Date class does not store time information.
base_date <- as.Date(bike_data$start_date) # 2017-01-01

### If you need to retain time information, use POSIXct
full_timestamp <- as.POSIXct(bike_data$start_date) # "2017-01-01 00:00:41 PST"

### If you need specific information about a date, it's just like SQL
weekdays(base_date[1]) # Sunday

## Okay, so let's say I want to keep the time. 
## It's easiest to do that by setting the column to be POSIXct and then
## converting back to date when you need just the year part.

bike_data$start_date <- as.POSIXct(bike_data$start_date)
bike_data$end_date <- as.POSIXct(bike_data$end_date)

### inspect beginning of data set

head(bike_data)

## Filtering

### (rows)
### rmv data that has rides less than 5.00 and Member type is Casual
### syntax is: data_frame[data_frame$column_name = val,]
### the comma is very important! as we are selecting entire rows...
### comparison in R - only need one &
filtered_bike_data <- bike_data[c(bike_data$ride_cost < 5.00 & bike_data$Member.type == "Casual"),]

### (columns)
### only give me back the start and end stations
### opposite syntax, comma first
start_and_end_stations <- bike_data[, c("Start.station", "End.station")]
### you can also just do it directly like this
### bike_data[c("Duration", "ID")] 
### all of the duration and all of the ID column, returned as a dataframe

### and of course you could put this all together, by plugging in a row-level constraint
### and then subsetting the cols you want back.

### add and then remove a dummy column
bike_data$dummy <- c(seq(1, length(bike_data$Duration)))
bike_data <- bike_data[, -11] # remove the 11th column

### add an ID column at the beginning of the DF
bike_data$ID <- paste0(c(seq(1, length(bike_data$Duration))), "_DC_BS")

### (numerical)
bike_data[1, 2] # 1st row, 2nd column
bike_data[, 2] # entire 2nd column
bike_data[c(1,2,3), c(4,5,6)] # 1st 3 rows and their 4th, 5th and 6th cols
### note that R does not have negative indexing.
### if you run my_vector[-2] you are removing the second element! 

### subset the data set into 2 halves.

### first half
bike_data_first_half <- bike_data[seq(1:ceiling(length(bike_data$Duration) / 2)), ] # top part
### second half - look at what's in the first half, take the opposite
bike_data_second_half <- bike_data[!(bike_data$ID %in% bike_data_first_half$ID), ] # bottom part

### rejoin the two halves, row wise
bike_data_back_together <- as.data.frame(rbind(bike_data_first_half, bike_data_second_half))

### join the two halves, column wise
bike_data_column_bound <- as.data.frame(cbind(bike_data_first_half, bike_data_second_half))

### rearrange
### there is not a super clean way to do this. you basically have to reinput all the column names,
### just in the order you would like them
### i think dplyr let's you swap these more cleanly.
desired_col_names <- replace(colnames(bike_data), 1, "ID")
desired_col_names <- replace(desired_col_names, 11, "Duration")
bike_data <- bike_data[desired_col_names]

### Sorting
sorted_bike_data <- bike_data[order(bike_data$ride_cost, -bike_data$End.station.number), ]
### can also use dplyr arrange
# newdata <- mtcars[order(mpg, -cyl),]

### Aggregations... 
### count the number of times a ride cost 4.00
### verbose
length(bike_data[bike_data$ride_cost == 4.00,]$ID) # 662
### neater
sum(bike_data$ride_cost == 4.00) # 662

### count the number of unique station destinations
length(unique(bike_data$End.station)) # 452 

### now, use dplyr for aggregation
library(dplyr)
library(nycflights13)
### average ride cost
summarize(bike_data, mean_cost = mean(ride_cost))

### okay... what about average ride cost per end station, sorted highest to low.
bike_data %>% # take bike data
  group_by(End.station) %>% # hit it with a group by
  summarize(mean_cost = mean(ride_cost)) %>% # calculate your metric of interest
  arrange(desc(mean_cost)) # order results

## More dplyr practice

# count of delays, and their average distance to go, and their average delay
# grouped by destination
# and removing a noisy honololu entry
# and only including those w/ more than 20 delays.
delays <- flights %>%
  group_by(dest) %>%
  summarize(
    count = n(), 
    dist = mean(distance, na.rm = TRUE),
    delay = mean(arr_delay, na.rm = TRUE)
  ) %>%
  filter(count > 20, dest != "HNL")

## There are multiple commands/functions for the purposes of summarizing grouped data.  They include: 
## first(), last(), quantile(x, 0.25), max(x), min(x), sd(x), . . .

## number of carriers per destination
carrier_count <- flights %>%
  group_by(dest) %>%
  summarize(carriers = n_distinct(carrier)) %>% 
  arrange(desc(carriers))

## as you can see, dplyr has a pretty fluid api 

