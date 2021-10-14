## Core

## Strings

* Strings in python are immutable. 
  * you can't replace a char at will.
* There is no char data type - those are just strings w/ length 1.
* Note that with these methods, you need to reassign them on the left side of the operation or the effect won't take place because that spot in memory isn't actually getting updated
* all strings are uniformly unicode, so no need to worry about char issues


### Cleanup and Data Information

* `.strip()`
  * remove whitespace and newlines
* `x in y`
  * same thing as `.contains()`
* `.join`
  * merge characters
  * `-.join("elvis") -> e-l-v-i-s`
  * `''.join(sorted(aaron)) -> aanor`
* `.split(delimiter)`
* `.startswith()`
* `.endswith()`
* `.isdigit()`
* `.isalpha()`
* `.replace()` [all]
  * return a copy of the string where *all* occurrences of a substring are replaced
* substrings are carried out with slices.
* `.count(s)`
  * # of occurrences
* `find(s)`
  * returns the *lowest* index of s in a string
  * "zucchini".find("c") returns 2.
  * so if you wanted the "last" c, you could hand off a reversed version of zucchini...
  * `len(vegetable) - vegetable[::-1].find("c") - 1`
    * hacky version of the above
* 

### Casing

* `.upper()`
* `.lower()`
* `.capitalize()`
  * uppercases the first letter
  * `"tim schott".capitalize()` -> Tim schott.
* `min` is the earliest alphabetical letter in a string
* for a rough comparison of alphabetical ordering, you can directly compare letters
  * `"apple"[0] < "zucchini"[0]`
  * true!
* 

### Slicing

* can slice just like lists
* again, this is how you would "substring"
* `test = "Timothy Schott"`
* `test[2:6]`
* -> you get `moth`
* last letter:
  * `string[-1]`
* everything before the last letter
  * `string[:-1]
* reverse with a slice
  * `test[::-1]` turns `Jupiter` into `'retipuJ'`
  * so test for palindrome:
    * x == y[::-1]
    * test == backwards[::-1]
* you can't get penalized for an out of bounds error with slicing:
  * instead you'll get back the maximum amount of data that *does* exist, but it'll fail in graceful way
  * this is a cool way to work through a window problem because instead of doing
    * `stuff[i] + stuff[i + 1]` and hoping you don't blow up on that second part,
    * you could do `stuff[i: i+2]` and check from there.

```python

"tim"[3] ## Out of Bounds!
"tim"[1:3] ## im
"XIV"[1:3] ## IV
```

## Data Structures 

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
  * i.e. it's searching for the *value* of x
  * not by index!
  * this means that if you want to remove something and you have its index, you have to feed it to remove like list.remove(list[index])
  * this of course could cause an issue with an unsorted list.
  * its safer in this case to use `pop`.
* `list.pop([i])` - remove the item at a given position in the list, and return it.
  * if no index is specific, `a.pop()` removes and returns the **last** item in the list
  * so for a stack/queue, you'd be using `struct.pop(0)`
* `list.clear()` - remove everything from list.
* In general, all mutable data structures do not have `insert`, `remove`, `sort` etc. methods that return anything
  * intentional
* combine two lists
  * `list_1 + list_2`
* How to use a list as a stack? 
  * add an item to the top with `append()`
  * remove it from the top with `pop(0)`
* How to use a list as a queue?
  * in general, its inefficient to use lists as a queue because the appends and pops from the end of the list cause every element in the list to have to shift by one
    * think about, for instance, if you have 10 people lined up, and the second person gets out of line. everyone after them has to shift up a spot 
* what item is at spot 8?
  * `nums.index(8)`
* does my list contain the number 12?
  * `12 in nums`
* list intersection (what in l1 is in l2)
  * `return [x for x in l1 if x in l2]`
* smallest element of a list
  * typically, you would use the `min()` function for this
  * but if you want to do something like, 
  * what is the *shortest* element of a list, you can add an extra argument for that criteria
  * `min(my_list, key = len)` will use that function as its basis for sorting
* note that to update a list "in place" you need to actually touch the elements
  * `list[:] = [mess w/ it]` will do that
  * but `list[] = [mess w/ it]` won't since you arent actually touching the stuff inside

```python
my_list = [1,2,3,4,5]
def test_func(a_list):
    a_list = [num * 2 for num in a_list]
    return a_list

test_func(my_list)
# [2, 4, 6, 8, 10]

my_list # note how it has note actually been updated.
# [1, 2, 3, 4, 5]
def test_func_touching_elements(a_list):
    a_list[:] = [num * 2 for num in a_list]
    return a_list

test_func_touching_elements(my_list)
### [2, 4, 6, 8, 10]
my_list # now it has been edited
### [2, 4, 6, 8, 10]
```

### Tuples

* immutable structures, typically used to store collections of heterogenous data
* construct them...
  * `()`
  * `(a,)` -> singleton tuple
  * `(a,b,c)`
  * `tuple()`
* the comma is what sets it apart as a tuple.
* variant, `namedtuple` that can associate label with the fields
* 

### Dictionaries

* associates keys with items
  * `my_dict = {"tim": "schott"}`
* values are access with bracket notation, inputting the key
* check for existence w/ `in`
  * `"Italy" in capitals`
* merge w/ `update`
  * in place
  * `capitals.update(morecapitals)`
  * this modifies `capitals` directly, doesn't return anything though.
* but what if you want to merge and have the result returned?
* `new = {**capitals, **morecapitals}`
  * kind of makes sense 
  * - handing off dict1 and dict2 as kwargs!
* delete w/ `del`
  * `del capitals['United Kingdom']`
* declare empty with `{}`
* get a tuple pair of its contents w/ `dict.items()`
* flip its keys and pairs:
  * `rev = {v: k for k, v in dict.items()}`
  * using a dictionary comprehension!

#### Looping over a dict

* over the keys:
  * `for key in dictionary:`
* over the values:
  * `for val in dictionary.values()`
* over both:
  * `for key, val in dictionary.items()`
* when you call `.keys()` you are handed back an "iterator" object that you can then cast w/ list
* `list(mydict.keys())`
* order of insertion is maintained in python now.

### Sets

* bag of mixed type of items
* no duplicates
* write w/ braces and no colons
  * `continents = {'tim', 'james', 'michael'}`

## Looping and Manipulating Structures

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

```python
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

* you can use this to iterate through a list until its empty:
* `while l1:`

* traditional `for` loop:

```python
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

```python
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
* for dictionaries, use the `sorted()` method
  * `sorted_bigrams = sorted(bigrams.items(), key=operator.itemgetter(1), reverse = True)`
  * another option would be using key = length to sort by the longest "items" val

### Comprehension 

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

* loop through 2 lists at once
  * `[x + y for x,y in zip(l1, l2)]`

* for a dictionary:
* you loop through key, column value pairs!

```python
countries_by_capital = {capital: country for country, capital in capitals_by_country.items()}
```

* apply a function to a comprehension
* surround the inner body of a comprehension w/ the function call:
  * `sum(num ** 2 for num in list)`

* nested loop comprehension
* let's transform a double loop into a single comprehension

```python

counting = []
# change this into a comprehension
for i in range(1, 11):
    for j in range(1, i+1):
        counting.append(j)

# becomes...

counting = [j for i in range(1, 11) for j in range(1, i+1)]
```

* if we're talking steps, literally collapse the loop and then add the additional `j` at the beginning.
* inner var + for outer var in outer range + for inner var in inner range (that should reference outer var)

* another use case - row wise loop across a numpy matrix
* `[a[i,j] for i in range(a.shape[0]) for j in range(a.shape[1])]`


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

* make a list of random integers.
* `randoms = np.random.randint(0, 10 ,size=(len(bike_data), 1))`
* make a list of random decimals
* `randoms = np.random.uniform(low=0.0, high=10.0, size=(len(bike_data), 1))`

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

* swap position of two columns

```
# this gets you every column name
col_list = list(bike_data)

# now you can move around your columns of interest
col_list[7], col_list[8] = col_list[8], col_list[7]

# and reassign to the df
bike_data.columns = col_list
```

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
* `bike_data.groupby(['Start station'])['Duration'].sum()`

* another thing to note is that you get a `DataFrameGroupBy` object
  * w/ key and value
  * it associates the grouping w/ the associated data for it
* to access the mini grouping...
* `df.groupby['city].get_group('mumbai')`

* maximum duration per city
* `bike_data.groupby(['Start station'])['Duration'].max().sort_values(ascending = False)`
  * notice how you have to plug in the pieces you want from left to right
  * first we group
  * then i grab only the Duration column from the mini df
    * which note will have an arbitrary listing of everything
  * then i grab the max

* one of the args you can pass to `groupby()` is `,dropna = True` to not consider na's
  
* note that after the aggregation, and you have 1 column, you have a `Series` which is that first pandas data structure. it has an index, still
* to just get the values, hit it with a `.values` at the end, then you can work w/ it like any other array
* `bike_data.groupby('End station')['ride_cost'].mean().sort_values(ascending = False).values[0:5]`




## Numpy

* N dimensions all of same size
* the size is the "shape"
* `(N, M)` ie `(2, 3)` for a matr. w/ 2 rows and 3 cols

### Numpy data types (dtypes)

* `numpy.int8`, `numpy.int16` ...
  * different types of integers
  * unsigned int, etc
* numpy needs to be more precise to be memory efficient
  * same w/ `numpy.float32`...
* `str`
  * fixed length

### Basic API

* where `arr` = a numpy object
* `.dim` 
  * number of array dimensions
* `.shape` 
* - lengths of the corresponding dimensions 
  * (rows by columns)
* `.dtype`
  * data type
* `.array`
  * creating a numpy array from a vanilla list
  * `np.array([[1,2,3], [4,5,6], [7,8,9]])`
* `.zeros`
  * creating an array full of zeros
  * `np.zeros((8,8), 'i')` 
  * where the `i` are is for int, `d` for float
* `.zeros_like(np_arr)`
  * make an array of zeros w/ the same shape of a given array
* `.identity(n)`
  * identity matr. of shape (n,n)`
* `.empty([2,2])`
  * "empty" array of given size
* `.arange(x, y, step_val)`
  * space values between x and y (exclusive) w/ step size
* `.random.random([x, y])`
  * generate random array of doubles w/ shape (x, y)
* for an integer, api is dumb
  * `np.random.randint(100, size=(10, 10))`
  * you enter the upper bound, then the size as a kwarg
* `.vstack([np_one_d_arrays])`
  * will stack them into a 2d array
* `.resize(array, new_shape)`
  * return a new array with the specified shape

* sum all of a 2d numpy array....
* `np.sum(np.resize(a, (1, a.shape[0] * a.shape[1])))`

### Indexing and Slicing

* indexing works just like if you were pulling from a matrix in a for loop
* `nparray[2, 3]` is 3rd row, 4th column
  * everything is 0 indexed
* negative indexing works the same way as with a list.
* note that they are comma separated in the brackets tho

* slicing....
* you can slice at will in each space in the commas
* `random_2d[0:5, 1]` 
  * first 5 rows, 2nd column
* remember the double colon notation means "step size"
  * `l = [1, 2, 3, 4 ,5 ,6 ,7, 8, 9, 10]`
  * `l[::2]` -> 1, 3, 5 ,7, 9
* you can inject this into numpy access
  * `monalisa[::20, ::20, :]`
    * every 20th row, every 20th column, all of that 3rd dim.
* you can use this to reassign vals in the array by setting them equal to a scalar for instance

* differences between lists and numpy arrays
  * when you slice a list, you make a copy of it
  * numpy slices point back to the original array
    * this means if you modify a slice, you'll modify those vals in the original array
  * so.... to *copy* a numpy array:
    * `monacopy = monalist.copy()`
* you access 2dimensional values differently
  * lists you need more than one statement in brackets
  * np arrays you put everything in one bracket statement

### Math

* `.sin`
* `.cos`
* `.log`
* `.dot`
* cool matrix math multiplication for easy dot product

```python

a = np.array([0, 1, 2])
b = np.array([-1, -2, -3])

a @ b # -8
np.dot(a, b) # -8
```

### For Data Analysis

* `.mean` `.nanmean`
* `.min` `.nanmin`
* `.max` `.nanmax`
* `.isnan`
* `.sum()`
* "how many nas?"
  * `np.sum(np.isna(np_array['col']))`
* `.var`
* so std dev is `.sqrt(np.var)`
* `.correlate`
  * Cross-correlation of two 1-dimensional sequences.
  * lets you find similarity between two series
  * "sliding dot product"
* what is **smoothing**
  * replacing time series data w/ an average of its neighbors
  * correlate your data against a dummy mask
  * this is how you can have a clear line vs just noise

* what is **axis = 0**
  * across all rows

* numpy vs pandas
  * numpy is geared more towards scientific computing
  * like an improved matlab
  * arrays
  * strictly numerical
* pandas is geared more towards higher level 'data science'
  * more visual
  * data frame / series
  * all data types

## Functions

### Exception handling (try/except)

```python

def find_anagram(word):
  sig = signature_dict(word)

  try:
    return anagrams_by_sig(sig)
  except KeyError:
    return set{}
```

### Arguments

* the two different types of arguments are **positional** and **keyword**
* let's look at a function: `def save_ranking(first, second, third=None, fourth=None):`
* this function features both kinds
* `first` is a positional argument. 
  * it cannot be omitted
* `third` is a keyword argument
  * you can use the function w/o supplying a value
  * the corresponding default is used
* positional arguments come first in function declarations.

### Packing

* in some cases, you don't know how many items you are going to hand off to a function
* in this case, you want to use **packing**
* with packing, a function accepts an arbitrary number of arguments
* `*args` = accept arbitrary # of *positional* arguments
* `*kwargs` = accept arbitrary # of *keyword* arguments
* for example, you could write a function to square an arbitrary group of positional arguments

```python
def my_square(*args):
  for arg in args:
    print(arg ** 2)

def my_sum(*args):
  print(sum(args))
```

* if you want to receive a dictionary, use 2 stars

```python
def func_kwargs(**kwargs):
  print('kwargs: kwargs)
```

```python
dict = {'arg1': 'Schott', 'arg2': 'Smith', 'Test': 'Man'}

def func_kwargs_positional(arg1, arg2, **kwargs):
    print('d1: ', arg1)
    print('d2: ', arg2)
    print('kwargs: ', kwargs)

func_kwargs_positional(**dict)

d1:  Schott
d2:  Smith
kwargs:  {'Test': 'Man'}
```

* this is a great way to write functions that only care about certain inputs but can still do stuff w/ other values.

### Unpacking

* this is a convenient way to satisfy a function contract if you know it accepts a certain number of arguments and you don't want to write them out
* if i have a constructor `person(first, last, dob)` and you have a tuple with those 3 items, i can just "unpack" them
* `person(*tuple_list[0])`

```python
def func(arg1, arg2, arg3):
    print('arg1 =', arg1)
    print('arg2 =', arg2)
    print('arg3 =', arg3)

l = ['one', 'two', 'three']

func(*l)
# arg1 = one
# arg2 = two
# arg3 = three
```

* likewise, to pack a dict, use 2 stars.

## Classes

### Dataclass

* a newer way to make an object in python
* very similar looking to a POJO

```python
@dataclass
class Robot:
  ID: int
  title: str = 'unknown'
  ability: int = 0
  weight: float = 0.0
```

* automatically lets you pull out fields (with dot notation)
* once you make an object like
* `terminator = Robot(2, 'Terminator', 7, 42.0)`
* you can convert it to a dictionary w/ `asdict()`

## Various

#### normalize the entire dataframe
* `normalized_df=(df-df.mean())/df.std()`

#### split data (row wise)

```
airquality1 = airquality.loc[:math.ceil(len(airquality)/2), :]
airquality2 = airquality.loc[math.ceil(len(airquality)/2):, :]
```

* ceiling function in python is `ceil()`
* You can accomplish a `UNION ALL` like operation w/
  * `pd.concat([df1, df2...])`

#### clean up a pandas column

```
def clean_price_column(df, column):
	
	price_good = [p if pd.notnull(p) else 0 for p in df['price']]
	
	price_good = [float(re.sub('[\$,\s]','', p)) for p in price_good]

```

#### data wrangling 
* after you subtract two dates, get the value with
  * https://stackoverflow.com/questions/25646200/python-convert-timedelta-to-int-in-a-dataframe
* and to make it a float, 
  * `pd.to_numeric(df['tdColumn'].dt.days, downcast='float')`

#### flatten list of lists
* to merge a list of lists into a single flattened list:
  * `sum(x, [])`

#### remove punctuation from string
* routine for removing punctuation characters from a string

```python
# remove punctuation here.
punc_list = [p for p in string.punctuation]

text = ''.join([char for char in text if char not in punc_list])
```
#### nice print formatting

* cool formatting trick
* `"\n".join([....list comprehension.....]`
  * will output each val on a new line

#### f strings

* very neat new construction
* `print(f'End is now {end}, Max length is now {maxLength}')`
* you can directly put variables into print statements

#### why are main methods in python so weird looking

* "explain why its `if __name__ == '__main__':`"
* this prevents a main method from A.py from being invoked in B.py imports A.py
* in writing this, we ensure that this block of code only executes when we are directly running the particular file its included in.

#### null checks

* in a function, to test your params are valid, it's O.k. to do
* `if not s: return`
* however, once you get to equality in a loop or something, that's where you'll want to use an equals sign for primitives and then the `not in` style notation for any object comparisons

#### all permutations of list

```python
import itertools
list(itertools.permutations([1, 2, 3]))
```

* you can also use `itertools.combinations` for combinations (order doesn't matter)

### Object oriented in java

#### Interface vs Abstract Class

* classes can extend one super class but can implement multiple interfaces
* abstract classes can have access modifiers on their fields
* interface fields are by default public static final
* when you extend an abstract class and update its method that is "overriding" and shows how java "polymorphism" works
* when you implement an interface, you have to implement all of its methods (unless they have a default implementation). with an abstract class, you can choose to just adopt whatever its already defined
* is vs has a

#### Reflection

* reflection just means java can do things like getClass() so you can determine this information at run time. 
  * useful to know if you can call certain methods given a vanilla "object"

#### Runtime polymorphism

* this just means that when you extend a class, you can override its methods and java figures out which method to call at run time

#### method overload

* same method name, different number of parameters

#### access modifiers

* public, private, protected

#### types of constructors

* default constructor, parameterized constructor, copy constructor

### Object Oriented in Python

#### simple class

```java
public class LoginPage extends ParentClass implements Interface1, Interface2 {

    private String name;

    @Override
    public String interfaceOneMethod() {
        return "Interface One Made Me Do This";
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
```

#### writing classes

```python
class Node:
  def __init__(self, data):
    self.data = data
    self.children = []
```

* provide the class name
* provide a parameterized constructor
  * can contain positional and functional parameters
  * note that in this constructor you can create default fields
  * this constructor encapsulates *instance* variables
* access the "attributes" (fields) w/ the dot notation
* when you add a function inside a class, you can invoke it with normal function syntax
* also you can add Object level "class" variables that apply to all instances of your class
  * so here, every `Person` will have `.species = Human`.
* within a class, `self` is basically the same thing as `this`


```python
class Person:

  species = 'Human' # shared by all Person objects

  def __init__(self, name, age):
    self.name = name
    self.age = age
    self.wallet = []
  
  def f(self):
    return 'hello world'

x = Person("timothy", 12)
x.age # 12

x.f() # 'hello world'
```

#### is python pass by value or pass by reference?

* mutable objects are automatically passed by reference
* high level, people call it "pass by assignment

### Odds and Ends, Java 

#### error vs exception

* we try to account for exceptions w/ try-catch, errors are unexpected