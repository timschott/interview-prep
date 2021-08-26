# SQL NOTES #

SQL - Structured Query Language. Used to construct and maintain databases at scale.

A database is a group of tables.

# Grabbing Data

## SELECT and FROM

Two most basic SQL ingredients. 

`SELECT` indicates which columns you'd like to view.

`FROM` identifies the `table` they live in.

```
SELECT year, artist, album FROM spotify.playlist
```

You'll need to prefix the database name at the end 

```
SELECT year, month, month_name, south, west, midwest, northeast FROM tutorial.us_housing_units
```

## AS - Aliases

Used for renaming results in the display view.

Everything that needs to be contiguous should use underscores. 

If you really need a separate string, like "West Region" you need to explicitly add quotation marks. 

Same goes for if you want capital letters, otherwise everything will be lowercase. (This is usually fine)

```
SELECT west AS "West Region"
  FROM tutorial.us_housing_units
```

Without this alias, the following query would have a very verbose label as its column name.

```
SELECT
   CONCAT_WS(', ', lastName, firstname) AS "Full Name"
FROM
   employees;
```

Note that the `as` keyword itself is actually optional in that if you don't put it MySQL will still do the cleanup for you. Works for column names and the table.

Sometimes you're going to have to have an alias because 2 tables could have the same column name.

i.e. 

```
SELECT
	customerName,
	COUNT(o.orderNumber) total
FROM customers c INNER JOIN orders o ON c.customerNumber = o.customerNumber
GROUP BY
	customerName
ORDER BY
	total DESC;
```    

One catch with aliases is that you can't used an alias column again in the same SELECT statement.

```
SELECT year, month, south / (south + west + midwest + northeast) AS south_pct FROM tutorial.us_housing_units
```
For instance here, i can't just save the bottom portion as `total` and then use it again and again, it has to be continuously recalculated (at least without the use of a sub query).

## LIMIT 

Placed at the end of a query to cull the number of results returned.

## WHERE - Filtering Data

Used to filter the particular rows handed back by the query. 

Ordering... `SELECT`, then `FROM`, then `WHERE`. Entire rows of data are filtered out, together, so if you add a condition and a row doesn't meet it, all of its columns are removed from the result set. 

### Comparison Operators

Equals: `=`

Not Equal: `<>` / `!=`

Greater Than + equal to: `>=`

Less Than + equal to: `<=`

```
SELECT *
  FROM tutorial.us_housing_units
 WHERE west > 30
```

For comparisons on nun-numerical data, you can use single quotes.

```
SELECT * from tutorial.us_housing_units
WHERE month != 'January'
```

### Simple Math

You can do direct, across column operations with operators like `+` and `-`.

```
SELECT west, midwest, northeast, midwest + northeast AS midwest_and_northeast FROM tutorial.us_housing_units
LIMIT 10
```

You can even do this sort of combining directly in the where clause:

```
SELECT west, midwest + northeast as midwest_and_northeast
FROM tutorial.us_housing_units
WHERE west > midwest + northeast
```

More advanced aggregation methods to come (ie to calculate averages and things like that).

### Logical Operators - allow you to use multiple comparisons in a single query

#### `LIKE`

Let's you match on character similarity using string format notation. 

#### `Wilcards`:
*   The percent (%) matches any string of zero or more characters.
    *   `s%` matches sa, six, sunk.
*   The underscore (_) matches any *single* character. 
    *   `se_` matches sea and set because they have an `e` in position 2 and then something else in position 3. 
* The underscore is essentially used to enforce a minimum length. 
* If you just use the percent sign, you could get a match back for just what's to the left of the pct in `LIKE t%` (i.e. someone is just named 't' in the DB.).

For a case insenstive `LIKE`, use `ILIKE`. (Not sure why you'd want this).

Return all data for years where one of the groups had Ludacris in it.

```
SELECT * FROM tutorial.billboard_top_100_year_end
WHERE "group" LIKE '%Ludacris%'
```

Return all data for years where a "DJ" is at the lead of the group

```
SELECT * FROM tutorial.billboard_top_100_year_end WHERE "group" like 'DJ_%'
```
(ie, must be DJ SOMETHING not just an artist named DJ).

Select all records where the second letter of the City is an "a".


```
SELECT * FROM Customers
WHERE City LIKE '_a%'
```

There are some other reserved classes, like 

[abc] is a contigous group so LIKE '[ldn]%' means starting with l or d or n.

This can also be achieved with `IN`....

#### `IN`

Specify a list of values you'd like to include in the results.

i.e. 

`year_digit IN (1, 2, 3)`

or 

```
"artist" in ('M.C. Hammer', 'Hammer', 'Elvis Presley')
```

#### `BETWEEN`

Between works to select rows in a specific range. It can also be replaced by using `AND` - its a little more compact. 

Note: the command is *Inclusive* at its bounds ie `BETWEEN 5 and 10` will contain rows at the margins.


#### `IS NULl`

Used to exclude rows that contain missing entries.

`SELECT * from houses where tax_status IS NULL`

#### `AND`

Used to combine operations - rows must satisfy the conditions on both sides of the clause.

A chain of `AND`'s:
```
SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year = 2012
   AND year_rank <= 10
   AND "group" ILIKE '%feat%'
```

Top ranked song of 1990, 2000, 2010
```
SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year_rank = 1
   AND year IN (1990, 2000, 2010)
```   

#### `OR`

Select rows that satisfy either of two conditions. Much like AND, it can be chained with a group of conditions.

Combine via parenthesis.

Write a query that returns all rows for top-10 songs that featured either Katy Perry or Bon Jovi.

```
SELECT * from tutorial.billboard_top_100_year_end
WHERE 
(year_rank BETWEEN 1 and 10)
AND ("group" ilike '%katy perry%' OR "group" ilike '%bon jovi%')
```

Note how pushing those wildcards on the outside lets us hit both a song directly by Katy Perry or a song she's featured in that contains other artists.

And with the combination, the OR needs to be satisfied as its own separate statement, so that's going to exclude ALL data that don't have katy perry or bon jovi, regardless of if they have that high ranking (because of the AND).

Write a query that returns all songs with titles that contain the word "California" in either the 1970s or 1990s.

```
SELECT * from tutorial.billboard_top_100_year_end
WHERE 
"song_name" ilike '%california%'
AND (year BETWEEN 1970 and 1979 OR year BETWEEN 1990 and 1999)
```

#### `NOT`

Used to select rows for which that condition is false. You could imagine using this with `LIKE` if you don't want to bend over backwards with a regex (ie not like a rather than like [bcdef...])

Special syntax if you want to exclude null values --

```
SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year = 2013
   AND artist IS NOT NULL
```

#### `ORDER BY`

Used to order results alphabetically/numerically based on the inputted sort-column(s).

SQL defaults to ascending (a-z, 1-10 etc) order

If you want to flip this, use `DESC`

```
SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year = 2013
 ORDER BY year_rank DESC
```

ORDER BY should go before LIMIT.

Write a query that returns all rows from 2010 ordered by rank, with artists ordered alphabetically for each song.

SELECT *
  FROM tutorial.billboard_top_100_year_end
 WHERE year = 2010
 ORDER BY year_rank, artist

#### Comments 

Single line comments --are preceded w/ 2 dashes
Multi line comments /* use this syntax

*/

--- Interlude: What's in a SQL Interview? https://www.youtube.com/watch?v=pJeGiUTWi1s&list=PLY1Fi4XflWStFs6tLQ3Gey2Aaq_U4-Xnc&index=9&ab_channel=DataInterviewPro ---

* Likely will be asked SQL questions at multiple stages in interview process
* Will range from easy to hard
  * Differences between window functions
  * Why did you use this type of join?
  * Do Game Play Analysis on LeetCode
  * Why did your structure your results that way?
* Practice converting SQL statements to code, i.e. looking for keywords
* You'll be evaluated on your descriptive ability, and your thread of logic to solve it
  * Clearly communicate thought process before jumping into `SELECT...`
  * Evaluated on: 
    * executable and free of syntax errors
    * clean and concise
    * consider edge cases
    * efficiency, at least somewhat (1 join vs 3
  * Ask clear questions when getting stuck (what format, what timestamp, etc.)
* To practice: set a timer! (5, 10 and 15 minutes)

--- Approaching Data Science SQL Question https://www.youtube.com/watch?v=tNXliLTlrV8&list=PLv6MQO1ZzdmpDYL3eZRs0Z_PqqYGn2iGR&index=4&ab_channel=NateatStrataScratch ---

* First, "to confirm..." and *restate the question*
* Second, explore data schema and go through *assumptions* 
  * "by session_type, do you mean this?"
  * establish uniqueness (no aggregation ie no group by)
  * establish non-null (if not, more edge cases)
  * establish generally clean (start + end date)
* Third, verify your assumptions - look at the data!
* Fourth, *write down your logic.*
  * Take advantage of commenting
  * First step is...
  * Second step is...
* Then, instead of coding right away, go through it with the interviewer to double check that you are on the right track
* Code, code, code....while explaining!
* Next, "can you optimize this code?"
  * sometimes, the answer is no - you might just need those sub queries
  * other times, you can show off your knowledge of "nice to have" implementations

## "Intermediate SQL" 

## Aggregate Functions

*   `COUNT`
    *   Count the number of rows in a particular column.
    *   SELECT COUNT (*) outputs the number of rows in your whole **table**
    *   SELECT COUNT(column) outputs the number of non null entries in that particular **column** so it could different than the above.
    *   works for non-numerical columns
*   `SUM`
    *   Totals the values in a column. Can only be used on numerical columns.
    *   `SELECT SUM(volume)
  FROM tutorial.aapl_historical_stock_price`
    * Treats null as 0.
    * "Write a Query for AVG opening price w/o using AVG function
    * `SELECT sum(open) / count(open) FROM tutorial.aapl_historical_stock_price`
    * 
*   `MIN` 
    *   returns lowest value in a column (can work on non numeric)
*   `MAX`
    *   returns highest value in a column
    *   "Find highest single day increase for the stock"
    *   `SELECT max(close - open) from tutorial.aapl_historical_stock_price`
*   `AVG`
    *   Compiles the average of a column. 
    *   Numerical only.
    *   Ignores nulls - as in, doesn't treat them as 0, so can throw off your calculation.
*   `MOD`
    *   Returns the remainder of two values
    *   `SELECT MOD(price, tax)`

## GROUP BY 

Allows you to separate data into groups. Groups rows that have the same values into summary rows, ie "find the number of customers in each country."

keyword is *per*

Apple stock activity PER YEAR:

```
SELECT year,
       COUNT(*) AS count
  FROM tutorial.aapl_historical_stock_price
 GROUP BY year
```

You can group by more than 1 column; you have to separate column names with columns (just as with `ORDER BY`).

Calculate the average daily price change in stock, per year.

```
SELECT AVG(close - open), year 
FROM tutorial.aapl_historical_stock_price
GROUP BY year
ORDER BY year
```

The big thing to remember is `GROUP BY` is going to smush your data.
You'll have AT MOST n number of rows based on having n unique buckets ie if your dataset spans 2000-2014 and you group by year you'll have a max of 15 before any other filtering is considered.

Lowest and highest apple prices achieved per month.

-- two different ways of approaching this, do they mean
each month in history? as in jan 2020, feb 2020... or
just each month overall? jan, feb, etc...

each unique month:

```
SELECT MIN(low), MAX(high), year, month from tutorial.aapl_historical_stock_price
GROUP BY year, month
```

each month overall:

```
SELECT MIN(low), MAX(high), month from tutorial.aapl_historical_stock_price
GROUP BY month
```

Note how when we `GROUP BY` 2 columns rather than 1, our result set is larger (there are more columns) because we are requesting a more granular data set.

Average high for each month:

```
SELECT month, avg(high)
FROM tutorial.aapl_historical_stock_price
GROUP BY month
ORDER BY month
```

## HAVING 

The `WHERE` keyword cannot be used, *directly*, with aggregate functions ;-(

`HAVING` is the cleanest way to filter a query that has been aggregated. IOW, to curate the output of a query that has been `GROUP BY`'d.

```
SELECT column_name(s)
FROM table_name
WHERE condition
GROUP BY column_name(s)
HAVING condition
ORDER BY column_name(s);
```

keyword *only include / exclude* and it references what you had to `GROUP BY`

List the number of customers in each country, but only include countries that have more than 5 customers. 

```
SELECT COUNT(CustomerID), Country
FROM Customers
GROUP BY Country
HAVING COUNT(CustomerID) > 5;
```

Notice how we can place that condition directly into the HAVING.

Difference between `WHERE` and `HAVING`:

*WHERE filters before producing result set, HAVING filters after*.

*   WHERE: applied to entirety of selected rows. used to select data in the original tables being processed
    *   executed *before* `GROUP BY`.
        *   filters prior to grouping.
    *   called a 'pre filter'
*   HAVING: applied to aggregate rows that are grouped by conditions. used to filter data in the result set that was produced by the query.
      *   executed *after* `GROUP BY`.
      *   applied to the grouped results
      *   it can invoke aggregate values and aliases from the `SELECT` clause.

```
SELECT t1.val - t2.val diff
FROM t1 JOIN t2 ON (some expression)
HAVING diff > 10
```
is not possible with `WHERE` because `diff` is an alias.

Convert an Integer to Character?

`CAST(Column AS CHAR)`

## CASE statements

SQL's way of handling if/then control statements.

* Every `CASE` statement must contain a corresponding `END` statement.
* The `ELSE` statement is optional.
  * provides a mechanism to capture values exclude from the `WHEN` and `THEN satements.


```
SELECT player_name,
       year,
       CASE WHEN year = 'SR' THEN 'yes'
            ELSE NULL END AS is_a_senior
  FROM benn.college_football_players

Dionne Thrweatt-Vassar	SR	yes
Jordan Luallen	SR	yes
Deven Drane	SR	yes
Brendon Kay	FR
```
1. The `CASE` statement inspects each row to see if the initial conditional - `year = 'SR'` - is true
2. For each row, if it is true, the word "yes" gets printed in a column named `is_a_senior`
3. If it is false, `NULL` gets place in that column
4. The `player_name` and `year` data is pulled for every row w/ no modifications.

Write a query that includes a column named "CALI" that contains "yes" when a player is from California, and sort the results to include those players first.

```
SELECT player_name,
CASE WHEN state = 'CA' THEN 'yes'
ELSE NULL END as cali
from benn.college_football_players
ORDER BY cali
```

The utility of using `CASE` is that you can produce an additional column of data. This differs from the `WHERE` filter because that just reduces what's output in the select statement.

Without tacking on `END as cali` to close the statement, the new column name is just case. So, add the alias.

The Syntax for expanding the case statements is simple, just load it up with `WHEN`'s:

```
CASE
    WHEN condition1 THEN result1
    WHEN condition2 THEN result2
    WHEN conditionN THEN resultN
    ELSE result
END;

SELECT OrderID, Quantity,
CASE
    WHEN Quantity > 30 THEN "The quantity is greater than 30"
    WHEN Quantity = 30 THEN "The quantity is 30"
    ELSE "The quantity is under 30"
END AS QuantityText
FROM OrderDetails;
```

Note that the individual `WHEN`'s can contain multiple conditional statements (`AND`s and `OR`s).

`CASE WHEN year = 'FR' AND position = 'WR' THEN 'frosh_wr'

Write a query that includes players' names and a column that classifies them into four categories based on height.

```
SELECT player_name, height,
CASE 
WHEN height <= 65 THEN 'TINY'
WHEN 66 <= height AND height <= 70 THEN 'SMALL'
WHEN 71 <= height AND height <= 74 THEN 'FOOTBALLER'
WHEN height >= 75 THEN 'TALLGUY'
END as subjective_height
FROM benn.college_football_players

Ralph Abernathy	67	SMALL
Mekale McKay	78	TALLGUY
Trenier Orr	71	FOOTBALLER
Bennie Coney	75	TALLGUY
Johnny Holton	75	TALLGUY
```

Remember that if you're including a conjunctive, you have to include the `AND`. Also, you need a comma before the initial case.

With aggregate functions...

```
SELECT CASE WHEN year = 'FR' THEN 'FR'
            WHEN year = 'SO' THEN 'SO'
            WHEN year = 'JR' THEN 'JR'
            WHEN year = 'SR' THEN 'SR'
            ELSE 'No Year Data' END AS year_group,
            COUNT(1)
  FROM benn.college_football_players
 GROUP BY 1
```

This easily allows us to report aggregate numbers over a document, for instance you could imagine taking advantage of this to quickly count the number of A students, B students, etc. in a classroom dataset.

Write a query that counts the number of 300lb+ players for each of the following regions: West Coast (CA, OR, WA), Texas, and Other (Everywhere else).

```
SELECT * FROM
  (
    SELECT CASE
        WHEN state IN ('CA', 'OR', 'WA') AND weight >= 300 THEN 'WestCoast'
        WHEN state = 'TX AND weight >= 300 THEN 'Texas'
        WHEN weight >= 300 THEN 'Other'
      END AS big_boys,
      COUNT(1)
    FROM benn.college_football_players
    GROUP BY 1
  ) AS subquery
WHERE big_boys IS NOT NULL
```

This approach removes the empty column that gets automatically calculated (and not labeled) for the results that don't meet the criteria (of being 300+ pounds). But it is unnecessarily complicated. Compare to the below query:

```
SELECT CASE WHEN state IN ('CA', 'OR', 'WA') THEN 'West Coast'
            WHEN state = 'TX' THEN 'Texas'
            ELSE 'Other' END AS arbitrary_regional_designation,
            COUNT(1) AS players
  FROM benn.college_football_players
 WHERE weight >= 300
 GROUP BY 1
 ```

 See how we directly filter off of the weight since all of our buckets share this criteria. Should have been tipped off when i was repeatedly shoving that >=300 condition into each statement..

 *The COUNT (1) / GROUP BY 1 seems to be a very important paradigm.*

Write a query that calculates the combined weight of all underclass players (FR/SO) in California as well as the combined weight of all upperclass players (JR/SR) in California.

```
SELECT sum(weight),
CASE 
WHEN year IN ('FR', 'SO') THEN 'Underclass'
WHEN year IN ('JR', 'SR') THEN 'Upperclass'
END as year_description
FROM benn.college_football_players
WHERE STATE = 'CA'
GROUP BY year_description
```

Note that we need to smush the data into just the buckets we made, so we use the closing `GROUP BY`.

## DISTINCT

Used to return only different values. Use to explore the data in a column.

If you use it it once in a `SELECT` it gets applied to each of the column pairings. 

i.e. `SELECT DISTINCT month` will give you at most 12 results

i.e. `SELECT DISTINCT year, month` will give you all the pairings of these 2 values.

Write a query that returns the unique values in the year column, in chronological order.

```
SELECT DISTINCT year from tutorial.aapl_historical_stock_price 
ORDER BY year
```

It can be combined with the `COUNT` aggregator to determine the number of unique values within a column.

```
SELECT COUNT(DISTINCT month) AS unique_months
  FROM tutorial.aapl_historical_stock_price
```

Note that `DISTINCT` goes inside the aggregate function. You probably shouldn't use it with the other functions like `MAX` (that's redundant).

Overall, the `DISTINCT` function is quite slow. Think about the back-end operation... something to point out during a performance question.

Write a query that counts the number of unique values in the month column for each year.

```
SELECT year, COUNT(DISTINCT(month))
from tutorial.aapl_historical_stock_price
GROUP BY year
```

The parens around `month` there are optional but for clarity I included them.

Write a query that separately counts the number of unique values in the month column and the number of unique values in the `year` column.

```
SELECT COUNT(DISTINCT(year)), COUNT(DISTINCT(month))
from tutorial.aapl_historical_stock_price
```

No aggregation needed. The number of uniques in the year column can be tabulated irrespective of the number of uniques in the month column.

## JOINS

Conceptual baseline...

The advantage to distributing data across multiple tables is that updates to small data points don't need to be persisted across a bunch of rows.

For example, if twitter stored your bio in the same place it stored all of your tweets, and you changed your bio, it would need to update as many rows as however many tweets you have (ie if there are 6k tweets stored, update 6k records). That is expensive just for updating 1 piece of data.

Similarly, it wouldn't make sense to store every data point about the UVA football team in the `players` table. That should go in a `teams` table!.

To compile that info... use a join!

Average weight, per player, per conference.

```
SELECT teams.conference AS conference,
       AVG(players.weight) AS average_weight
  FROM benn.college_football_players players
  JOIN benn.college_football_teams teams
    ON teams.school_name = players.school_name
 GROUP BY teams.conference
 ORDER BY AVG(players.weight) DESC
``` 

* When performing joins, it makes sense to assign the table names an alias.
  * Here we see that `_football_players` got assigned `players` and `_football_teams` got assinged `teams`
  * Now, we can invoke that alias in the `SELECT` clause
* `JOIN` is followed by a table name
* `ON` is followed by column names separated by an equals sign
  * Indicates how the tables are alike - where their data is equivalent
  * In this case, the school name is the foreign key.

You don't have to join tables based on equality between two column values.
You can use any of the comparison conditions.
As in, join when e1.employee_name < LENGTH(e2.employee_name)

Write a query that selects the school name, player name, position, and weight for every player in Georgia, ordered by weight (heaviest to lightest). Be sure to make an alias for the table, and to reference all column names in relation to the alias.

```
SELECT players.school_name,
       players.player_name,
       players.position,
       players.weight
  FROM benn.college_football_players players
 WHERE players.state = 'GA'
 ORDER BY players.weight DESC
```

Example of using that alias directly after invoking the name of the table.

Get the average weight of every team in the ACC.

```
SELECT
players.school_name, avg(players.weight)
FROM benn.college_football_players players
JOIN benn.college_football_teams teams
ON players.school_name = teams.school_name
WHERE teams.conference = 'ACC'
GROUP BY 1
ORDER BY 2 DESC
```

Using the 1/2 strategy is a great way to compile / sort your data and increase your readability.

Write a query that displays player names, school names and conferences for schools in the "FBS (Division I-A Teams)" division.

```
SELECT
players.school_name, players.player_name, teams.conference
FROM benn.college_football_players players
JOIN benn.college_football_teams teams
ON players.school_name = teams.school_name
WHERE teams.division = 'FBS (Division I-A Teams)'
```


## Types of Joins

* `(INNER) JOIN`: Returns records that have matching values in both tables
  * The *intersection* of the two tables.
  * Example - if a player goes to ITT tech, and they aren't in the teams table, they won't be included in 
  ```FROM benn.college_football_players players
  JOIN benn.college_football_teams teams
    ON teams.school_name = players.school_name```

* `LEFT (OUTER) JOIN`: Returns records from the left table, and the matched records from the right table.
  * You'll get back `NULL` in rows when there is no matching right side
* `RIGHT (OUTER) JOIN`: Returns records from the right table, and the matched records from the left table
  * You'll get back `NULL` in rows when there is no matching left side.
* `FULL (OUTER) JOIN`: Returns all records when there is a match in either left or right table.
  * When neither row has a match, you'll get back `NULL` 
* `CROSS JOIN`: Produces the Cartesian product of the two tables. Will be m x n for t1 w/ m rows and t2 w/ n rows.
* `NATURAL JOIN`: implicitly uses a shared (same name) column to perform an (inner) join. 
  * De-duplicates columns with shared names.
  * In math, "equi-join".
  * All `NATURAL` joins can be written as `INNER` joins. but the converse is not true.
  * natural = a specific kind of join.

You can join a table to itself.

```
SELECT
e.employee_name AS 'Employee',
m.employee_name AS 'Manager'
FROM employee e
INNER JOIN employee m ON m.id = e.manager_id
```

