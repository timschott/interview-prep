# Algorithms and Data Structures

* This set of notes is designed to cover material commonly encountered during the general programming/software engineering portion of an interview
* Code wise, we'll use python3 throughout this set of notes.

## Asymptotic Analysis 

### (Big O)

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

### Run Time Chart conceptual

![runtimes](runtimes.jpg)

* lists are the only structure with best case linear insert/delete
  * because stacks, queues, and linkedlist always insert/delete at the end
  * while lists need to dynamically expand after each operation
* however, lists do provide constant access time when you know the index
* if you're constantly searching, a hash map is the best solution
* the perf of binary search trees depends on how balanced your data is
  * a balanced tree will perform well - `O(log(n))` because you reduce the amt to search through by half at each step
  * but a skewed tree is going to be essentially a singly linked list, so you get an inefficient linear time for each operation

## Data Structures

### Hash Table

#### General

* maps keys to values for efficient lookup
* where a hash function is a deterministic function to map data of arbitrary size to fixed size values
  * compute hash code
  * map hash code to an index in array
  * insert - if theres a collision, use your strategy of choice
* worse case runtime for retrieval is O(N), but it's typically O(1)

#### Python

* in python, the `Dictionary` is the implementation of hash table
* the keys of the dictionary are hashable

### Lists (Resizable Arrays)

#### General

* dynamically allocates / resizes
* think Java `ArrayList`

#### Python 

* in python, `List`

### Linked Lists

#### General 
* a linked lists is a sequence of items connected via links
* they either point forwards or point forwards and backwards
* if you were to make it by hand, you would make a `Node` class and then a head node class.
* the head node points to a normal node, and so on.

#### Python 

* in python, we can use the `collections` module and the `deque` ("deck") class
  * **deque** = double-ended queue

### Stacks and Queues

#### General

* **stacks** store data Last In, First Out 
  * (LIFO)
  * use case: back-button on your browser
* **queues** store data like traditional ticket lines: First In, First Out 
  * (FIFO)
  * use case: real-world line to get into a concert

* the `stack` methods are:
  * pop - remove and return top element
  * push - add to stack, at the top!
  * peek - what's first?
* the `queue` methods are:
  * add - add entry, to the end
    * "back of the line..."
  * remove - remove first item in list
  * peek - what's first?

#### Python 

* *we can use the `deque` class to make our own stacks or queues!.* 
* just depends what methods we use for insert / retrieval...
* for a `stack`:
  * `q.appendleft("item")` # when someone gets added, they go on top
  * `q.popleft()`
* for a `queue`"
  * `q.append("item")` # when someone gets added, go to the back of the line
  * `q.popleft()`

```python
d = deque(['a','b','c']) 
d.popleft()
# 'a'
d.appendleft("z")
d
# deque(['z', 'b', 'c'])
```

* in practice, queues are very similar to linked list so theyre a good option for the deque class
* stacks, it's a little hackier, you have other options

### Trees and Graphs (Overview)

* a **tree** is a data structure composed of nodes ...
  * each tree has a *root node*
  * the root node has 0 or more children
  * each child has 0 or more children
  * when a node has no kids, it's a "leaf" node.
* really, from a code perspective, all you need to implement it, structure wise, is just:

```python
class Node:
  def __init__(self, data):
    self.data = data
    self.children = []
```

### Binary Tree

* a **binary tree** is a special tree where each node can have at *most* 2 children.

#### Binary Search Tree

* a special binary tree is a **binary search tree** in which every node fits the following rule:
  * `all left descendents <= node.value <= all right descendants`
* during an interview, make sure any "binary" tree you're working with is a BST.
* example of a BST node:

```python
# Definition for a binary tree node.
class TreeNode:
  def __init__(self, val=0, left=None, right=None):
  self.val = val
  self.left = left
  self.right = right
```

#### Other varieties of Binary Trees

* "complete"
  * every level is fully filled (except the last)
* "full"
  * every node has either 0 or 2 children
    * no only children!
* perfect binary tree?
  * complete + full
  * all leaf nodes are at same level
  * this level has max number of nodes
  * size?
  * 2^k - 1 where K = # of levels

### How do we find data? the two types of traversal strategies...

#### Depth first and Breadth first

* a **depth first** search exhausts all options down a branch (vertical path) before proceeding to the next branch
  * go deeeeeep
* these are heavily recursive
* intuitively, the break down into ....
* in-order
  * visit the left branch, then the current node, then the right branch
  * when you do this with a BST, you get the nodes in sorted order
* pre-order
  * visit the current node, then the left branch, then the right branch
* post-order
  * visit the left branch, then the right branch, the the current node

* code for in order traversal

```python
def inOrderTraversal(TreeNode node):
  if node is not None:
    inOrderTraversal(node.left)
    visit(node)
    inOrderTraversal(node.right)
```

* code for pre order traversal

```python
def preOrderTraversal(TreeNode node):
  if node is Not None:
    visit(node)
    preOrderTraversal(node.left)
    preOrderTraversal(node.right)
```

* a **breadth first search** explores each neighbor, on the same level as the root, before proceeding to children
  * go wiiiiiide
* overview:
  * start at the root node
  * travel through every child node *at the current level* before moving to the next level

* the actual implementation code is pretty confusing

```python
def bfs(self, root=None):
    if root is None:
        return
    queue = [root]

    while len(queue) > 0:
        cur_node = queue.pop(0)

        if cur_node.left is not None:
            queue.append(cur_node.left)

        if cur_node.right is not None:
            queue.append(cur_node.right)

        print(queue)
```


* when would you use depth first?
  * if there is important info at the leaves
    * like the Trie data structure
  * data structure - stack
  * (recursion = stack)
* when would you use breadth first?
  * if you know that the important information is very close to the root
  * 'nearest neighbor'
  * family tree example
    * i'm level one, my kids are level two, my grandkids are level three
  * data structure - queue

#### Min and Max Heaps are also a thing, but skipping

#### Tries (they use these at yext)

* a **trie** is a prefix tree that helps store words.
* it stores the lexicon, basically
* very fast - can tell if "how do i make" is a valid prefix in O(K) time

### Graphs 

* all trees are graphs
* not all graphs are trees
* vocab to know
  * "directed" if there is only one way bt nodes
  * "undirected" if its a 2 way street
  * "connected" if there is a path between every pair of vertices
  * "cycle" is 2 nodes pointing at one another
  * "vertex" another name for a node
* because trees are graphs, we actually search a graph with dfs/bfs as well
* just a little trickier.
* you might represent one in python with a dict:

```python
graph = {
  'A' : ['B','C'],
  'B' : ['D', 'E'],
  'C' : ['F'],
  'D' : [],
  'E' : ['F'],
  'F' : []
}
```

## Searches

### Binary Search

* Eliminate half the numbers every time
* Move based on which direction your guess is (too high or too low)
* Run time: `log(n)`

## Practice Problems - explanations

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

## Recursion and Dynamic Programming

* recursive solutions use previously computed work to solve a problem
  * ie, use f(6) to solve f(7)
* dynamic programming takes recursive approach and caches the old ones for future use
* example of using recursion for in order traversal
* note here i have a helper method because i want to keep track of stuff in a list
* you might not need that if you just have a "visit" method that prints the item

```python
class Solution:
    
  def traverse(self, node, visit_list):
      
      if node: 
          if node.left:
              self.traverse(node.left, visit_list)

          visit_list.append(node.val)

          if node.right:
              self.traverse(node.right, visit_list)    
  
  def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
      visit_list = []
      
      self.traverse(root, visit_list)
      
      return visit_list
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
    elif (n == 1):
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

* insight here is to try and catch the most specific condition at the top.
* they give it to you in that order because they want you to check for 2 then 3 then 6.. which isn't the correct ordering.
* also insight here is, if a number is divisible by a and b it's divisible by a * b (essentially the definition of factoring)

### First Unique Character in a String

* my first naive solution, use a list to keep track of what you have seen. 
* if you've run into it before, delete all instances.

```python
    seen = []

    for char in string:
        if (char not in seen):
            seen.append(char)
        else:
            string = string.replace(char, "")

    if string != "":
        return string[0]
    else:
        return "all chars unique"
```

* you could also do a dictionary w/ freq counts per character, but then you have to loop through the dictionary after you loop through the target string
* i guess if they say, no other methods

### merge sorted lists

* off the dome:

```python
i = 0
j = 0

sorted_merge = []
while i < len(l1) and j < len(l2):
    l1_temp = l1[i]
    l2_temp = l2[j]

    if (l1_temp <= l2_temp):
        sorted_merge.append(l1_temp)
        i += 1
    elif (l1_temp >= l2_temp):
        sorted_merge.append(l2_temp)
        j += 1

if (i < len(l1)):
    print('hi')
    sorted_merge.extend(l1[i:])

if (j < len(l2)):
    print('low')
    sorted_merge.extend(l2[j:])

return sorted_merge
```

* it works
* if you were just using methods you could obviously just do
* `l1.extend(l2)` and then hit that with a `sorted()`
* a useful way to improve my solution is pop.
  * `pop(n)` returns the element at n and then removes it
  * so you could do what i did but pop
  * and that way you don't even need an index, you just do the first
  * because they are already sorted

```python
while (l1 and l2):
    if (l1[0] <= l2[0]): # Compare both heads
        item = l1.pop(0) # Pop from the head
        sorted_list.append(item)
    else:
        item = l2.pop(0)
        sorted_list.append(item)

# Add the remaining of the lists
sorted_list.extend(l1 if l1 else l2)
```

### biggest area from a list

```python
def biggest_area(height):
    
  if not height:
      return 0
  if len(height) == 1:
      return numbers[0]
  
  largest = 0
  for i in range(len(height)):
      window = 1
      while i + window < len(height):
          # step 1: "normalize height metric" by getting min of current vs. inspect
          h = min(height[i], height[i + window])
          # step 2: calculate area - which is height * width where width = window size
          area = h * window
          # step 3: set this to largest if it's bigger!
          largest = max(largest, area)
          # step 4: keep the party going. move window over. 
          # print(f'{numbers[i]}, {numbers[i + window]}')
          window += 1
  
  return largest
```

### reverse a linked list recursively

* general strategy: our 2 cases are
  * normal nodes
  * establishing the head node
* okay, so like other recursive work, create a helper function and then get started
* the helper function first accounts for making the new head node
  * the new head node is the old tail node
  * so we know where this is when we have `curr.next is None`
  * we "fix" our class by setting `self.head = old tail`
* otherwise, keep track of `next` for the subsequent call
* and set the current node's `next` field to be the previously visited node
  * ie in 1 -> 2 -> 3
  * when we get to 2, next = 3, prev is 1 so we set 2's "next" to be 1 (reversing)
* keep calling this method, using next as current and current as previous
* note that on the very first invocation, `prev` is none. 
  * so curr.next = prev doesn't do anything for the prior head element.
* 

```python
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    
    if head is None:
        return
    self.reverseUtil(head, None)
    
    return self.head
    
def reverseUtil(self, curr, prev):
    
    # establish end of list
    if curr.next is None:
        self.head = curr

        # fix the head by making it point "backwards"
        curr.next = prev
        return

    # otherwise, keep next for subsequent call
    next = curr.next

    # and do the flip
    curr.next = prev

    self.reverseUtil(next, curr)
```

### recursive power of two 

```python
def isPowerOfTwo(self, n: int) -> bool:
    # power of two ....
    if not n:
        return False
    return self.helper(n)

def helper(self, i):
    if i == 1 or i == 2:
        ## success
        return True
    elif i % 2 == 0: 
        ## recurse
        return self.helper(i / 2)
    else:
        ## fail
        return False
```

### reverse a string, recursively

* the leetcode version of this was weird because it made you do it in the confines of a list
* in general, though, some approach should check for null, or else return method applied to the [1:] index of the string + the first char

```python
if len(s) == 0:
    return s
else:
    return reverse(s[1:]) + s[0]
```

### "Validate" parens / brackets

* a naive way is to use a dict of dicts to keep track of the counts
* but the issue with doing that is you aren't able to keep track of when in a string different brackets are used
* so you lose the idea that `"([)]"` needs to be false
* because you can't keep track of context
* strategy:
  * 1 - validate open and close counts matches
  * 2 - validate ordering matches
    * what is the definition of proper ordering?
    * an even number of chars between same tags?
    * only pairs
* i was able to get it given the condition that we only go from outside in
* ie `{([])}`
  * just compare if outer chars match, recurse while we still have.
* but that's not robust enough to an example like `{}[]()`.
* *much easier is just to use replace all*... 
  * because as we iteratively replace what's inside, the outer things will resolve
  * their lookup table is *closed* which is a big difference from my approach
  * a closed lookup table breaks down the problem to its smallest area
  * and helps us with both `{}()` and `{()}`
    * break down problem into its simplest smallest status

```
brackets = ['()', '{}', '[]']
while any(x in my_string for x in brackets):
    for br in brackets:
        my_string = my_string.replace(br, '')
return not my_string
```

### Sliding Window problems

* this sort of problem is used when you need to inspect sub-lists that are inside a collection
* "largest sum of 5 consecutive elements"
* "length of longest substring no repeated chars"
* sliding window is helpful because it can reduce our time complexity from something like a 2 loop solution to a linear O(n)

### Problem Approach, in general

* what is the easiest solution?
* what would a non-computer do to solve this?
* only try to generalize from a simple solution
  * *not* just off the cuff
* does the problem repeat itself, in a way
  * this is a tip off something like recursion may come in handy
  * or at the very least, iterate until it's solved (with a while loop)

### Edge Cases, in general

* possible things to think about...
  * empty
  * all same
  * off by one (car/cat) or (car/cor)
  * length one (you're looping, but someone hands you a length one list)