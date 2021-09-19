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
  * number of orders per day, etc
* handling dates
  * how many days have passed since df$order_date? 
* join two data sets
  * equiv on sql join
* stack two data sets
  * equiv of sql union

## Stats-Dev

* standardize a column
  * where standardize = subtract mean and divide by std.dev
* looking at relationships between variables
  * correlation overall
  * correlation given specific conditions
* 5 number summary for a column
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

## Stats Review

* Interpreting the output of a linear regression...
* Summary:

```
> summary(mpg_reg)

Call:
lm(formula = mpg ~ wt, data = train)

Residuals:
    Min      1Q  Median      3Q     Max 
-4.8351 -2.6410 -0.2644  1.2663  6.4635 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)  37.9602     2.3431  16.201 4.51e-14 ***
wt           -5.4654     0.7453  -7.333 1.85e-07 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 3.22 on 23 degrees of freedom
Multiple R-squared:  0.7004,	Adjusted R-squared:  0.6874 
F-statistic: 53.78 on 1 and 23 DF,  p-value: 1.846e-07
```

* `Residuals` are the difference between the actual value of the car's MPG and the value predicted by the model.
  * They can be accessed with `$residuals`
  * should be roughly symm and median close to 0
* `Coefficients` represent what would be attached if you were doing ordinary least squares by hand.
  * in 1d case, just a line
  * in 2d case, a plane...
  * `yˆ= f(x) = θ0 +θ1x`
  * in thi case, the first term is 37.9602 (intercept) and the second is -5.4654
  * The estimate is in fact the estimate of the mean of the dist of the variable
  * and the error is the square root of the variance of that dist!
* `t statistics` are the estimates divided by their standard errors.
  * `Pr(>|t|)` represents the probability of achieving a `t` value 
  * greater than the absolute values of the observed `t`.
* `Residual standard error` is an estimate of the std dev parameter
* `Adjusted r^2)` is r^2 adjusted for the number of params in the model
* `F statistic` is the ratio of two variances, SSR/SSE
  * sum of squares of regression (SSR) = the variance explained by parameters in the model
  * sum of squares of error (SSE) = the unexplained variance
* generally, the higher the `R^2` the better the model performs.
* when p is small, the more significant the factor is.

* anova
  * ANOVA is used to test the null hypothesis that the population mean is the same for all groups
  * Only applicable when you're testing across groups.
* 