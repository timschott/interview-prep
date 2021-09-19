## Brainstorm

* just going to list out topics that are relevant...
  
## General-Dev

Should be able to do in Python or R (base R with tiny bit of dplyr, and then pandas for python)

* loading data sets
* handling missing values
  * removing NA's
  * imputing missing values 
* adding a new column
* removing a column
* move a column from the back to the front
* generate a column with 10 random values
* round numbers to desired decimal
* getting all data that meet a certain criteria
  * filter
* access data at a certain spot
  * subsetting / slicing
* name rows/columns
  * index / name management
* sort data sets
* aggregating data sets
  * 5 number summary
  * number of orders per day, etc
* handling dates
  * how many days have passed since df$order_date? 
* join two data sets
  * equiv on sql join
* stack two data sets
  * equiv of sql union
* standardize a column
  * where standardize = subtract mean and divide by std.dev

## Stats-Dev

* looking at relationships between variables
  * correlation overall
  * correlation given specific conditions
* creating a scatter plot for two variables
* using data to predict an outcome of interest
  * linear regression
  * logistic regression
* interpret the results of a prediction
  * inspecting coefficients
  * inspecting confidence intervals
  * error calculating 
    * residuals
* using data to determine if two variables are independent
  * chi squared
* use data to determine if the difference between 2 groups is significant
  * t-test
    * two samples, compares means for 2 groups
      * the groups must be statistically independent
      * normally distributed, same variances
    * paired sample, compares the same group at different times
      * data is in the form of matched pairs
    * one sample, test the mean of a group against a known mean
* use data to determine if there is a statistical difference between 3 or more indep. groups
  * ANOVA
* adding and subtracting variance / std.dev
* multiplying / dividing variance / std.dev