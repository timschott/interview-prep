## R studying

## Load Data

set.seed(101)
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
## https://www.neonscience.org/resources/learning-hub/tutorials/dc-convert-date-time-posix-r

bike_data$start_date <- as.POSIXct(bike_data$start_date)
bike_data$end_date <- as.POSIXct(bike_data$end_date)

## inspect beginning of data set

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
### https://www.guru99.com/r-aggregate-function.html
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
summarize(bike_data, mean_cost = mean(bike_data$ride_cost))

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

### Stats ###

### first, lets add another column.. lets make a "distance" column.
bike_data$distance <- runif(length(bike_data$ID), 0, 25)
### now, standardize 
### use scale!
bike_data$distance <- scale(bike_data$distance) 

mean(bike_data$distance) # 0
sd(bike_data$distance) # 1

### Let's load in the mtcars dataset for the next part, because it has more numeric fields.
data("mtcars")

### correlation, for everything
c <- as.data.frame(cor(mtcars))

### just between mpg and cyl
  mtcars %>%
    select(mpg, cyl, hp) %>%
    summarize(r = cor(mpg, cyl))

### inspect data that is too highly correlated.
### youd typically do this with a much higher threshold.
for (i in 1:nrow(c)) {
  for (j in 1:ncol(c)) {
    if (upper.tri(c)[i,j] & abs(c[i, j]) > .80) {
      cat(sprintf("%s %s %s \n", row.names(c)[i], row.names(c)[j], round(abs(c[i, j]), 2)))
    }
  }
}
  
### 5 number summary, function call
fivenum(mtcars$mpg)

### distance is being weird... gonna drop it
bike_data <- bike_data[, -12]

### use dplyr, count rides for each day in the bike set...
bike_data %>%
  group_by(as.Date(start_date)) %>%
  summarize(
    n = n()
  )

### what is the 5 number summary?
bike_data %>% 
  group_by(as.Date(start_date)) %>%
  summarize(
    n = n(),
    min = fivenum(ride_cost)[1],
    Q1 = fivenum(ride_cost)[2],
    med = median(ride_cost),
    Q3 = fivenum(ride_cost)[4],
    max = fivenum(ride_cost)[5]
  )

### note how you can essentially load up the summarize call with all the things you need
### it's kind of the opposite of sql. it's not going to get mad at you after you do group by
### like i don't have to also group by the ride_cost here (that would make the set bigger in fact)

### what are the top 5 destinations per day?
t <- bike_data %>%
  group_by(as.Date(start_date), End.station) %>%
  summarize(
    n = n()
  ) %>%
  filter(row_number() <= 5) %>%
  arrange(`as.Date(start_date)`, desc(n))

### pretty cool, here, you can directly filter off of row number w/o actually
### attaching it as a column

### make a scatter plot with mtcars.
plot(mtcars$wt, mtcars$mpg)
### prediction
### first, double check weight and mpg are correlated
cor(mtcars$wt, mtcars$mpg)
### now, make a linear regression against these variables
### to start, split training and testing data
# Now Selecting 80% of data as sample from total 'n' rows of the data  
sample <- sample.int(n = nrow(mtcars), size = floor(.80*nrow(mtcars)), replace = F)
train <- mtcars[sample, ]
test  <- mtcars[-sample, ]

### "mpg [IS EXPLAINED BY] weight"
### weight is the predictor.
mpg_reg <- lm(mpg ~ wt, data = train)

### let's explore the model.
summary(mpg_reg)

# anova(mpg_reg)

### make a prediction
predict(mpg_reg, test)

### now let's check for "multi colinearity"
library(car)

#add diagonal line for estimated regression line
abline(a=0, b=1)



      