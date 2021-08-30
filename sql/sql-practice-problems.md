This file is intended to keep track of general resources for sql practice problems as well as list links to problems that were difficult / have worthwhile solutions to reconsider.

Great resource, to debug intermediate leetcode queries:

https://github.com/jjjchens235/leetcode-sql-unlocked/blob/master/db_fiddle_public_urls.md

## LeetCode 176. Second Highest Salary

https://leetcode.com/problems/second-highest-salary/

Write a SQL query to get the second highest salary from the Employee table.

```
+----+--------+
| Id | Salary |
+----+--------+
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |
+----+--------+
```

For example, given the above Employee table, the query should return 200 as the second highest salary. If there is no second highest salary, then the query should return null.

```
+---------------------+
| SecondHighestSalary |
+---------------------+
| 200                 |
+---------------------+
```

-- first, get the highest salary
-- then, requery, getting the highest salary that is not the max
-- constraints, will not work if there is only 1 salary in the table.

```
SELECT MAX(Salary) as SecondHighestSalary
from Employee
WHERE Salary 
NOT IN (
    SELECT 
    MAX(Salary) 
    from Employee
    )
```
Another approach uses `OFFSET`, which is basically when you apply a slider after a `LIMIT`.

`LIMIT 1 OFFSET 1`

## Leetcode 181: Employees Earning More Than Their Managers

https://leetcode.com/problems/employees-earning-more-than-their-managers/

https://www.db-fiddle.com/f/skWkHvm6Fazs1zTbVn46o3/0

The Employee table holds all employees including their managers. Every employee has an Id, and there is also a column for the manager Id.

```
+----+-------+--------+-----------+
| Id | Name  | Salary | ManagerId |
+----+-------+--------+-----------+
| 1  | Joe   | 70000  | 3         |
| 2  | Henry | 80000  | 4         |
| 3  | Sam   | 60000  | NULL      |
| 4  | Max   | 90000  | NULL      |
+----+-------+--------+-----------+
```

Given the Employee table, write a SQL query that finds out employees who earn more than their managers. For the above table, Joe is the only employee who earns more than his manager.

```
+----------+
| Employee |
+----------+
| Joe      |
+----------+
```

The key here is that you need to reference multiple rows of this data at the same time in order to come up with your result set.
This means that some sort of join needs to be performed.

You can taper down this join to join employees to their managers...

```
SELECT * FROM Employee as columnA
INNER JOIN 
Employee as columnB
ON columnA.ManagerId = columnB.Id
```

Okay, so that outputs all of an employee's data as well as their managers.

Next, just taper it by limiting:
`WHERE columnA.salary > columnB.salary`

Full answer:

```
SELECT columnA.Name FROM Employee as columnA
INNER JOIN 
Employee as columnB
ON columnA.ManagerId = columnB.Id
WHERE
columnA.salary > columnB.salary
```
In general, db-fiddle.com is a useful resource for plug and play sql work. 

Select All https://www.hackerrank.com/challenges/select-all-sql/problem

`SELECT * FROM CITY` (just get everything).

Pop density difference https://www.hackerrank.com/challenges/population-density-difference/problem

Query the difference between the maximum and minimum populations in CITY.

`SELECT MAX(POPULATION) - MIN(POPULATION) FROM CITY;` 

(diff of max and min)

Weather Station 5 https://www.hackerrank.com/challenges/weather-observation-station-5/problem?isFullScreen=true

Query the two cities in STATION with the shortest and longest CITY names, as well as their respective lengths (i.e.: number of characters in the name). If there is more than one smallest or largest city, choose the one that comes first when ordered alphabetically. ]you can use 2 queries]

```
SELECT CITY, LENGTH(CITY) // desired output
FROM STATION 
ORDER BY LENGTH(CITY), CITY ASC // order first by the length to get the shortest value, then alphabetize the cities that have that value
LIMIT 1;

SELECT CITY, LENGTH(CITY) 
FROM STATION 
ORDER BY LENGTH(CITY) DESC, CITY ASC // opposite. we want the longest 
LIMIT 1;

```

https://www.hackerrank.com/challenges/the-blunder/problem?isFullScreen=true

Samantha was tasked with calculating the average monthly salaries for all employees in the EMPLOYEES table, but did not realize her keyboard's  key was broken until after completing the calculation. She wants your help finding the difference between her miscalculation (using salaries with any zeros removed), and the actual average salary.

Write a query calculating the amount of error (i.e.:  average monthly salaries), and round it up to the next integer.

Interesting operations happening here.

Remember that we're just after a single number here, so using our aggregation functions like `AVG` is a good approach.

```
SELECT 
CEILING // we want our result rounded up to the nearest whole integer
    (AVG(Salary) - // we want the original value

    AVG(
        CAST( 
            REPLACE(
                CAST(Salary AS CHAR), 
            '0', '')
        AS UNSIGNED)
    )) FROM EMPLOYEES;
```

1. first, cast salary to a character
2. then replace all of its 0's
3. cast it back to an (unsigned) integer
4. next, compute the differences between the correct and incorrect values
5. lastly, round up to the nearest whole integer.

https://www.hackerrank.com/challenges/earnings-of-employees/problem

We define an employee's total earnings to be their monthly  worked, and the maximum total earnings to be the maximum total earnings for any employee in the Employee table. Write a query to find the maximum total earnings for all employees as well as the total number of employees who have maximum total earnings. Then print these values as  space-separated integers.

```
select (salary * months)as earnings, count(*) from employee 
group by 1
order by earnings desc 
limit 1;
```

1. we calculate the earnings
2. we group by employee_id to satisfy mysql syntax
   1. when you use an aggregate function like `COUNT`
      1. its best practice to also group your data by a corresponding column
      2. so in this case we'll group by the earnings column even though it's a little bit redundant
      3. c.f. https://stackoverflow.com/a/43482023
3. sort by the highest to lowest
4. limit of 1 grabs the max

this pattern - order by something desc and limit to 1 is a great way to get a maximum value without having to use the `MAX` function.

## Leetcode 184: Department Highest Salary
https://leetcode.com/problems/department-highest-salary/problem

What stood out in this problem is that you might be tempted to use `MAX` at some point. But you don't really care what the actual maximum value is. You want to see who earns it. That's why using `RANK` is correct. This problem added slight complexity by making you perform a `JOIN` prior to being able to tabulate your results.

I used a window operation, `RANK() OVER (Partition By d.Name Order By e.Salary DESC) as r` and sub-query / where to see who has that rank of 1. 


