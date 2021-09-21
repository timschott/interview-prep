## Core

## Sequences: List, Tuple, Range

* Essentially, everything that the immutable types can do mutable types can do as well
  * plus tons of other extra stuff

### Lists

```
>>> digits = [1, 8, 2, 8] # declaring a new list
>>> len(digits) # find its length
4
>>> digits[3] # accessing the element at index 3
8
```

* mutable sequence with arbitrary length
* construct them...
  * `[]`
  * `[a], [b,c,d]`
  * `list()` or `list(iterable)`
* any values can be included
  * including another list
* API highlights
* `len(x)` - length of list
* `min(x)` - smallest item in list
* `list.append(x)` - add item to end of list
* `list.insert(i, x)` - add item at the given position
  * the first argument is the index *before* which to insert
  * `a.insert(0, x)` inserts directly at the front.
* `list.remove(x)` - drop the first instance of `x` in the list.
* `list.pop([i])` - remove the item at a given position in the list, and return it.
  * if no index is specific, `a.pop()` removes and returns the last item in the list
* `list.clear()` - remove everything from list.
* In general, all mutable data structures do not have `insert`, `remove`, `sort` etc. methods that return anything
  * intentional
* How to use a list as a stack? 
  * add an item to the top with `append()`
  * remove it from the top with `pop()`
* How to use a list as a queue?
  * in general, its inefficient to use lists as a queue because the appends and pops from the end of the list cause every element in the list to have to shift by one
    * think about, for instance, if you have 10 people lined up, and the second person gets out of line. everyone after them has to shift up a spot 

### Tuples

* immutable structures, typically used to store collections of heterogenous data
* construct them...
  * `()`
  * `(a,)` -> singleton tuple
  * `(a,b,c)`
  * `tuple()`
* the comma is what sets it apart as a tuple.


### Ranges

* A `range` is another standard sequence in Python, representing a range of integers.
* `range(1,10)` # Includes 1, but not 10.
* you can put a range directly inside the list constructor
* `list(range(5,8))` -> `[5,6,7]`
* ranges are commonly used in a `for` header expression to specific how many times it should be executed
* shared an api with list since they are both mutable sequences.

```
for _ in range(3):
    print('Go Bears!') # will print Go Bears! 3 times.
```

### Iteration

* `while` loop:

```
>>> def count(s, value):
        """Count the number of occurrences of value in sequence s."""
        total, index = 0, 0
        while index < len(s):
            if s[index] == value:
                total = total + 1
            index = index + 1
        return total
>>> count(digits, 8)
2
```

* traditional `for` loop:

```
>>> def count(s, value):
        """Count the number of occurrences of value in sequence s."""
        total = 0
        for elem in s:
            if elem == value:
                total = total + 1
        return total
>>> count(digits, 8)
```

* iterating through a list while looking at its pairs

```
>>> pairs = [[1, 2], [2, 2], [2, 3], [4, 4]]

>>> for x, y in pairs:
        if x == y:
            same_count = same_count + 1
>>> same_count
2
```

### Sorting

* the default `sort()` method is guaranteed to be stable
* A sort is stable if it guarantees not to change the relative order of elements with the same value
  

### List Comprehension 

* many times, you can handle iterations through a list in one compact line.

```
odds = [1, 3, 5, 7, 9]
[x + 1 for x in odds]
# everything in odds has now gotten increased by 1
```

* the `for` keyword is part of a list comprehension because it is contained within square brackets.
* list comprehensions can also include control statements
```
>>> [x for x in odds if 25 % x == 0]
[1, 5]
```

## Pandas

### Overview

#### Series

* 1-d array that can store a mixture of data types
* `pd.Series({list, tuple or dictionary})`

```
d = {'name' : 'IBM', 'date' : '2010-09-08', 'shares' : 100, 'price' : 10.2}
ds = pd.Series(d)

print(ds) # print the whole thing

name IBM
date 2010-09-08
price 10.2
shares 100

print(ds[0]) # access the value at index 0

IBM
```

#### Dataframe

* for 2d arrays
  * column-index and row-index
* simplest way to do this is provide a *dictionary* of *equal-length lists*

```
data = { 'name' : ['AA', 'IBM', 'GOOG'],
  'date' : ['2001-12-01', '2012-02-10', '2010-04-09'],
  'shares' : [100, 30, 90],
  'price' : [12.3, 10.3, 32.2]
```

* add a column to a dataframe like:
  * `df['newColumn'] = 'Unknown'`

* the row index has a default value of 0, 1, ... number of rows
* you can change the **row** index with `index` attr
  * `df.index = ['one', 'two', 'three']`

* automatically, your df is going to get read in with column indices because of how the dictionary is set up
  * so for instance if you want to access the company names you can invoke `df['name']`

* to rename a column, use `df.rename` and supply a mapping of old to new names:
  * `df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)`
  * `inplace=True` is important, don't forget

* accessing data
  * use column index
    * `df['name']`
    * `df.loc[:,'name']`
  * use row level index
    * `df.loc[0]`
    * entire first row.

* deleting data:
  * `df.drop('col_name', axis = 1)`

#### Reading files

* use `read_csv('filename, 'index_col = None)`
  * `index_col` should be set to true if there is no dummy data in the very first column.
  * `casts = pd.read_csv('cast.csv', index_col = None)`
  * if you really don't want the arbitrary leading index, you can do `index_col=False` otherwise they're going to automatically make one for you.
* inspect first 3 rows
  * `df.head(3)`
* total number of rows:
  * `len(df)`

#### Data Access

* when you pull a row or column from a DF it is a 1d `series`
* i think the big point of contrast with an R data frame is its less about accessing data through a numeric index (its loc in the matrix) and is instead heavily tied to the actual column / row names ('index') of the data
* you can automatically pull a column out just by invoking its name
* like a java getter
* `mtcars.mpg` -> series.
  
#### Filtering

* like R, you can filter by injecting a boolean expression to the dataframe
* `after85 = titles[titles['year'] > 1985]`
* select row w/ mpg < 20 and hp > 200
* `mtcars[(mtcars['mpg'] < 20) & (mtcars['hp'] > 200)]`
  * note how you have to encapsulate the conditions inside their own parens
  * also note how its a single `&` rather than the pythonic `and`
  
#### Sorting

* by default, the cars will sort by the index.
* you can change this w/ `sort_values('col')`
  * `ascending = False` for z-a

#### Null Values

* you can use `pd.isnull()` and `pd.notnull()`
* ex
  * `ozone_good = [v for v in airquality['Ozone'] if pd.notnull(v)]`
* to quickly reconcile these issues, you can go use `.fillna('new val')`

#### String Operations

* they afford a special call to search series based on string pattern
* like sql `like`
* `t[t['title'].str.startswith("Maa ")].head(3)`
* however - in order to work with its particular api, you have to keep casting back the value at hand to `.str` the pandas string type
* `mtcars[mtcars['mnm'].str.lower().str.contains("p")]`
* for instance here, i lowercase the string, and then i also have to do a `.str` before i check for the contains.
* work in a regex...
* `mtcars[mtcars['mnm'].str.lower().str.match('[a-z]+\s[a-z]+\s[a-z]+')]`
  * woof lol
  * hit the lower, and then check the match.

#### Basic Plot

* https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

```
plt.scatter(mtcars.mpg, mtcars.wt, c="g")
plt.xlabel("mpg")
plt.ylabel("wt")
plt.show()
```

### Aggregation / Group by.

* you can group by column-headers
* as a rule ... this can be pretty slow if you try and do it on more than 1 col
  * `c.groupby(['year']).size()`
  * `bike_data.groupby(['Member type']).size()`
* `groupby()` is *itself* a function so you need to wrap it in parens, and then you are able to access the outputs
* something to note is that with this group by call, its just like in dplyr where effectively the output is a select * from that aggregated pool
  * different from sql where if you don't explicitly say what you want, you get an error

* w/ two columns:
  * `b_agg = pd.DataFrame(bike_data.groupby(['Start date', 'Start station']))`

* we have access to some standard functions like
  * `count`
  * `sum`
  * `mean`
  * `mad` (mean absolute deviation)
  * `min`
  * `max`
  * `abs`

* another, generally, helpful tool:
* `describe()` which takes a series and outputs 5 number summary as well as mean, stddev, count and max.
* `df['column'].describe()['std']` -> standard deviation for the col

* another handy call
* `data.groupby(['column']).groups.keys()`
  * gets you the (unique) group keys its using

* pipeline time...
* how to apply transformations and other functions to aggregated data
* ie dplyr / sql like group by "pipeline"
* https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

* example, sum durations per start station
* `bike_data.groupby(['Start station']).sum()`
* 