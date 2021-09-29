# Algorithms

* This set of notes is designed to cover material commonly encountered during the general programming/software engineering portion of an interview
* Code wise, we'll use python3 throughout this set of notes.

### Asymptotic Analysis (Big O)

* Start with a video:
  * [BackToBackSWE - Asymptoptic Analysis](https://www.youtube.com/watch?v=myZKhztFhzE&ab_channel=BackToBackSWE)
* Recall that Big O is our upper bound
* And its concern is the asymptotic behavior when inputs are very large
  * i.e. no real diff bt y = x and y = 3x
* Algorithms grow at different rates
  * we measure their speed by this growth rate
* Big O notation describes the **number of operations**
* Establishes the **worst case** run time
* `O(*n*)` -> for a list of size n, requires n operations
  * Naive Search
* `log *n*` -> for a list of size n, requires log(n) operations
  * halves the work each iteration
  * Binary Search
* `O(*n* log *n*)`
  * an efficient sorting algorithm like quick sort
* `O(*n* <sup>2</sup>)`
  * a slow sorting algorithm like selection sort
* `O(*n*!)` 
  * a very slow algorithm like traveling salesman
* Often, it isn't until inputs are very large that you can see the advantages of a more efficient algorithm.

### Searches

#### Binary Search

* Eliminate half the numbers every time
* Move based on which direction your guess is (too high or too low)
* Run time: `log(n)`


### Practice Problems - explanations

### Two Sum

* essentially, two sum hands you two linked lists in reverse order and you need to return the sum of its values and place that in a linked list....
  * input: l1 = [2,4,3] l2 = [5,6,4]
  * output: [7,0,8]
* strategy:
  * it's not efficient just to literally REVERSE what is inside.
  * take advantage of the properties of math!

```

243
564
[ what im writing]
  7, carry the 1
  0, carry the 1
  8, carry the 1
]
```

* so, if you think about how you add numbers, we actually add them from right to left
* so it's helpful that our linked list is backwards. 

### Fibonacci

* simple. the idea is to return the nth fib. number
* where those are 0, 1, 1, 2, 3, 5, 8, 13 ...

```python
def fib(n):
    
    if (n < 0):
        return 'bad input'
    # first base case
    elif (n == 0):
        return 0
    # second base case
    elif (n == 1 or n == 2):
        return 1
    # else recurse
    else:
        return Solution.fib(n - 1) + Solution.fib(n - 2)
```

### Reverse a String by hand

* with slices its just
* `stringname[::-1]`
* with a while loop you would do

```python

s = "Test"
reversedString = []
i = length(s)

while (i > 0):
  reversedString += s[i - 1]
  i = i - 1

print(reversedString)
```

* with a for loop you would do 

```python
    s1 = ''
    for c in s:
        s1 = c + s1  # appending chars in reverse order
        print('s1 is ' + s1)
    return s1
```

### Fizzbuzz variant

* Print numbers from 1 to 100, but:
* if the number is even, print "a" instead of the number
* if the number is divisible by three, print "b" instead of the number
* if the number is even AND divisible by three, print "ab" instead of the number

```python
def fizzbuzzish():
    ans = []
    for i in range(1, 101):
        
        if (i % 6 == 0):
            print(str(i) + " ab")
        elif (i % 2 == 0):
            print(str(i) + " a")
        elif (i % 3 == 0):
            print(str(i) + " b")
        else:
            print(i)
```