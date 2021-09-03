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