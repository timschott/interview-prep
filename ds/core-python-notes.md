# Python

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

### Slicing