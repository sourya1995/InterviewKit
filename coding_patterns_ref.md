# Comprehensive Coding Interview Patterns Reference

## Table of Contents
1. [Array & String Patterns](#array--string-patterns)
2. [Two Pointers & Sliding Window](#two-pointers--sliding-window)
3. [Linked List Patterns](#linked-list-patterns)
4. [Stack & Queue Patterns](#stack--queue-patterns)
5. [Hash-Based Patterns](#hash-based-patterns)
6. [Heap & Priority Queue](#heap--priority-queue)
7. [Tree Patterns](#tree-patterns)
8. [Graph Patterns](#graph-patterns)
9. [Binary Search Variations](#binary-search-variations)
10. [Dynamic Programming](#dynamic-programming)
11. [Greedy Algorithms](#greedy-algorithms)
12. [Backtracking](#backtracking)
13. [Bit Manipulation](#bit-manipulation)
14. [Mathematical & Geometric](#mathematical--geometric)
15. [Advanced Data Structures](#advanced-data-structures)
16. [Specialized Techniques](#specialized-techniques)

---

## Array & String Patterns

### 1. Prefix Sum / Cumulative Sum
**Concept:** Pre-compute cumulative sums to answer range sum queries in O(1).

**When to Use:**
- Multiple range sum queries
- Subarray sum problems
- 2D matrix range sums

**Key Insight:** `sum(i, j) = prefix[j] - prefix[i-1]`

**Common Problems:**
- Range sum query
- Subarray sum equals K
- Continuous subarray sum

**Template:**
```python
# 1D Prefix Sum
prefix = [0] * (n + 1)
for i in range(n):
    prefix[i + 1] = prefix[i] + arr[i]
range_sum = prefix[j + 1] - prefix[i]

# 2D Prefix Sum
prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
for i in range(rows):
    for j in range(cols):
        prefix[i+1][j+1] = matrix[i][j] + prefix[i][j+1] + prefix[i+1][j] - prefix[i][j]
```

---

### 2. Difference Array
**Concept:** Efficiently perform multiple range updates using difference encoding.

**When to Use:**
- Multiple range increment/decrement operations
- Range update, point query scenarios
- Building final array after many range modifications

**Key Insight:** Update `diff[L] += val` and `diff[R+1] -= val`, then compute prefix sum.

**Common Problems:**
- Range addition
- Corporate flight bookings
- Car pooling

**Template:**
```python
diff = [0] * (n + 1)
# For each range update [L, R] with value val:
diff[L] += val
diff[R + 1] -= val

# Reconstruct array
result = [0] * n
result[0] = diff[0]
for i in range(1, n):
    result[i] = result[i-1] + diff[i]
```

---

### 3. Kadane's Algorithm
**Concept:** Find maximum sum of a contiguous subarray in O(n).

**When to Use:**
- Maximum subarray sum
- Maximum product subarray
- Maximum sum circular subarray

**Key Insight:** At each position, decide whether to extend current subarray or start new one.

**Common Problems:**
- Maximum subarray
- Maximum product subarray
- Maximum sum circular subarray

**Template:**
```python
max_sum = float('-inf')
current_sum = 0
for num in arr:
    current_sum = max(num, current_sum + num)
    max_sum = max(max_sum, current_sum)
```

---

### 4. Dutch National Flag (Three-Way Partitioning)
**Concept:** Partition array into three sections in one pass.

**When to Use:**
- Sort array with 3 distinct values
- Partition around pivot
- Move elements to specific positions

**Key Insight:** Maintain three pointers (low, mid, high) for three regions.

**Common Problems:**
- Sort colors
- Partition array
- Move zeros to end

**Template:**
```python
low, mid, high = 0, 0, len(arr) - 1
while mid <= high:
    if arr[mid] == 0:
        arr[low], arr[mid] = arr[mid], arr[low]
        low += 1
        mid += 1
    elif arr[mid] == 1:
        mid += 1
    else:
        arr[mid], arr[high] = arr[high], arr[mid]
        high -= 1
```

---

### 5. Cyclic Sort
**Concept:** Sort array where elements are in range [1, n] by placing each element at its correct index.

**When to Use:**
- Array elements in range [1, n] or [0, n-1]
- Finding missing/duplicate numbers
- In-place sorting with O(n) time

**Key Insight:** Place element `x` at index `x-1`.

**Common Problems:**
- Find missing number
- Find all duplicates
- First missing positive

**Template:**
```python
i = 0
while i < n:
    correct_pos = arr[i] - 1
    if arr[i] != arr[correct_pos]:
        arr[i], arr[correct_pos] = arr[correct_pos], arr[i]
    else:
        i += 1
```

---

### 6. Coordinate Compression
**Concept:** Map large range of values to smaller range while preserving relative order.

**When to Use:**
- Large coordinate ranges but few distinct values
- Range queries with sparse data
- Discretization needed

**Key Insight:** Sort unique values and map to indices.

**Common Problems:**
- Ranking elements
- Range module problems
- Skyline problem

**Template:**
```python
coords = sorted(set(arr))
compressed = {v: i for i, v in enumerate(coords)}
compressed_arr = [compressed[x] for x in arr]
```

---

### 7. Simulation
**Concept:** Follow problem rules exactly, step by step.

**When to Use:**
- Matrix manipulation (rotation, spiral)
- Game state transitions
- Process simulation

**Key Insight:** Carefully implement given rules, watch for edge cases.

**Common Problems:**
- Spiral matrix
- Rotate image
- Game of life

---

## Two Pointers & Sliding Window

### 8. Two Pointers (Classic)
**Concept:** Use two pointers moving toward each other or in same direction.

**When to Use:**
- Sorted array problems
- Finding pairs with target sum
- Removing duplicates

**Key Insight:** Move pointers based on current comparison.

**Common Problems:**
- Two sum (sorted)
- Container with most water
- Trapping rain water

**Template:**
```python
left, right = 0, len(arr) - 1
while left < right:
    if condition:
        # Process
        left += 1
    else:
        right -= 1
```

---

### 9. Fast & Slow Pointers
**Concept:** Two pointers moving at different speeds.

**When to Use:**
- Cycle detection in linked list
- Finding middle element
- Cycle start detection

**Key Insight:** If cycle exists, fast pointer will eventually meet slow.

**Common Problems:**
- Linked list cycle
- Find duplicate number
- Happy number

**Template:**
```python
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        return True  # Cycle detected
```

---

### 10. Sliding Window (Fixed Size)
**Concept:** Maintain window of fixed size k, slide through array.

**When to Use:**
- Fixed-size subarray problems
- Maximum/minimum in fixed window
- Average of subarrays

**Key Insight:** Add new element, remove leftmost element.

**Common Problems:**
- Maximum sum subarray of size k
- Average of subarrays
- First negative in every window

**Template:**
```python
window_sum = sum(arr[:k])
max_sum = window_sum
for i in range(k, n):
    window_sum += arr[i] - arr[i - k]
    max_sum = max(max_sum, window_sum)
```

---

### 11. Sliding Window (Variable Size)
**Concept:** Expand/contract window based on condition.

**When to Use:**
- Longest/shortest subarray with condition
- Substring problems
- Dynamic window size

**Key Insight:** Expand right, contract left when condition violated.

**Common Problems:**
- Longest substring without repeating characters
- Minimum window substring
- Longest subarray with sum ≤ k

**Template:**
```python
left = 0
for right in range(n):
    # Add arr[right] to window
    while window_invalid():
        # Remove arr[left] from window
        left += 1
    # Update result with valid window
```

---

### 12. Sliding Window with Deque
**Concept:** Use deque to maintain min/max in sliding window.

**When to Use:**
- Sliding window maximum/minimum
- Maintaining order in window
- O(n) solution needed

**Key Insight:** Deque stores indices, maintains monotonic property.

**Common Problems:**
- Sliding window maximum
- Shortest subarray with sum ≥ k
- Longest continuous subarray with absolute diff ≤ limit

**Template:**
```python
from collections import deque
dq = deque()
for i in range(n):
    # Remove elements outside window
    while dq and dq[0] < i - k + 1:
        dq.popleft()
    # Maintain monotonic property
    while dq and arr[dq[-1]] < arr[i]:
        dq.pop()
    dq.append(i)
    # arr[dq[0]] is min/max in current window
```

---

## Linked List Patterns

### 13. In-Place Linked List Manipulation
**Concept:** Reverse, rearrange linked list without extra space.

**When to Use:**
- Reverse list or portions
- Reorder list
- Palindrome checking

**Key Insight:** Careful pointer manipulation, often reverse in place.

**Common Problems:**
- Reverse linked list
- Reverse nodes in k-group
- Reorder list

**Template:**
```python
def reverse(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

---

### 14. Floyd's Cycle Detection (Extended)
**Concept:** Not just detect cycle, but find cycle start and length.

**When to Use:**
- Find where cycle begins
- Determine cycle length
- Find duplicate in array (treat as linked list)

**Key Insight:** After meeting, reset one pointer to head, move both at same speed.

**Common Problems:**
- Linked list cycle II
- Find duplicate number
- Happy number

**Template:**
```python
# Find meeting point
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        break

# Find cycle start
if not fast or not fast.next:
    return None
slow = head
while slow != fast:
    slow = slow.next
    fast = fast.next
return slow  # Cycle start
```

---

## Stack & Queue Patterns

### 15. Monotonic Stack
**Concept:** Stack maintaining monotonic (increasing/decreasing) property.

**When to Use:**
- Next greater/smaller element
- Largest rectangle in histogram
- Temperature problems

**Key Insight:** Pop elements that violate monotonic property.

**Common Problems:**
- Next greater element
- Largest rectangle in histogram
- Daily temperatures

**Template:**
```python
# Next greater element (decreasing stack)
stack = []
result = [-1] * n
for i in range(n):
    while stack and arr[stack[-1]] < arr[i]:
        idx = stack.pop()
        result[idx] = arr[i]
    stack.append(i)
```

---

### 16. Monotonic Queue
**Concept:** Queue maintaining monotonic property, typically with deque.

**When to Use:**
- Sliding window maximum/minimum
- Optimal ordering in queue
- Range queries with moving window

**Key Insight:** Remove elements from back that are worse than current.

**Common Problems:**
- Sliding window maximum
- Shortest subarray with sum ≥ k
- Jump game VI

**Template:**
```python
from collections import deque
dq = deque()
for i in range(n):
    # Remove worse elements from back
    while dq and arr[dq[-1]] <= arr[i]:
        dq.pop()
    dq.append(i)
    # dq[0] is optimal in current range
```

---

### 17. Stack for Expression Evaluation
**Concept:** Use stack(s) to evaluate mathematical expressions.

**When to Use:**
- Calculator problems
- Infix to postfix conversion
- Evaluate reverse polish notation

**Key Insight:** Separate stacks for numbers and operators, handle precedence.

**Common Problems:**
- Basic calculator
- Evaluate reverse polish notation
- Decode string

**Template:**
```python
def calculate(s):
    stack = []
    num = 0
    sign = '+'
    for i, char in enumerate(s):
        if char.isdigit():
            num = num * 10 + int(char)
        if char in '+-*/' or i == len(s) - 1:
            if sign == '+':
                stack.append(num)
            elif sign == '-':
                stack.append(-num)
            elif sign == '*':
                stack.append(stack.pop() * num)
            elif sign == '/':
                stack.append(int(stack.pop() / num))
            sign = char
            num = 0
    return sum(stack)
```

---

### 18. Parentheses/Bracket Matching
**Concept:** Validate and generate valid bracket sequences.

**When to Use:**
- Valid parentheses checking
- Generate valid combinations
- Longest valid parentheses

**Key Insight:** Stack for validation, backtracking for generation.

**Common Problems:**
- Valid parentheses
- Generate parentheses
- Longest valid parentheses

**Template:**
```python
# Validation
def isValid(s):
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    for char in s:
        if char in pairs:
            stack.append(char)
        elif not stack or pairs[stack.pop()] != char:
            return False
    return len(stack) == 0

# Generation (backtracking)
def generate(n, open_count=0, close_count=0, path=''):
    if len(path) == 2 * n:
        result.append(path)
        return
    if open_count < n:
        generate(n, open_count + 1, close_count, path + '(')
    if close_count < open_count:
        generate(n, open_count, close_count + 1, path + ')')
```

---

## Hash-Based Patterns

### 19. Hash Map (Frequency Counting)
**Concept:** Track element frequencies for quick lookup.

**When to Use:**
- Counting occurrences
- Finding duplicates
- Anagram problems

**Key Insight:** O(1) lookup and update.

**Common Problems:**
- Two sum
- Group anagrams
- Top K frequent elements

**Template:**
```python
from collections import Counter
freq = Counter(arr)
# or
freq = {}
for x in arr:
    freq[x] = freq.get(x, 0) + 1
```

---

### 20. Hash Map (State Tracking)
**Concept:** Track complex state using hash map.

**When to Use:**
- Tracking seen patterns
- Memoization
- Prefix sum with hash map

**Key Insight:** Map state to value/index for quick access.

**Common Problems:**
- Continuous subarray sum
- Subarray sum equals K
- Longest consecutive sequence

**Template:**
```python
seen = {}
for i, x in enumerate(arr):
    state = compute_state(x)
    if state in seen:
        # Found match
        result = process(seen[state], i)
    seen[state] = i
```

---

## Heap & Priority Queue

### 21. Min/Max Heap
**Concept:** Maintain smallest/largest element at top.

**When to Use:**
- Kth largest/smallest
- Merge operations
- Continuous median

**Key Insight:** O(log n) insert/delete, O(1) peek.

**Common Problems:**
- Kth largest element
- Top K frequent elements
- Merge K sorted lists

**Template:**
```python
import heapq
min_heap = []
heapq.heappush(min_heap, item)
smallest = heapq.heappop(min_heap)

# Max heap (negate values)
max_heap = []
heapq.heappush(max_heap, -item)
largest = -heapq.heappop(max_heap)
```

---

### 22. Two Heaps
**Concept:** Use min and max heap together.

**When to Use:**
- Median of stream
- Sliding window median
- Balance between two sets

**Key Insight:** Max heap for smaller half, min heap for larger half.

**Common Problems:**
- Find median from data stream
- Sliding window median
- IPO problem

**Template:**
```python
import heapq
max_heap = []  # smaller half (negated)
min_heap = []  # larger half

def add_num(num):
    heapq.heappush(max_heap, -num)
    heapq.heappush(min_heap, -heapq.heappop(max_heap))
    if len(min_heap) > len(max_heap):
        heapq.heappush(max_heap, -heapq.heappop(min_heap))

def find_median():
    if len(max_heap) > len(min_heap):
        return -max_heap[0]
    return (-max_heap[0] + min_heap[0]) / 2
```

---

### 23. K-Way Merge
**Concept:** Merge K sorted sequences using heap.

**When to Use:**
- Merge K sorted lists/arrays
- Find smallest range in K lists
- Kth smallest in sorted matrix

**Key Insight:** Heap stores one element from each list.

**Common Problems:**
- Merge K sorted lists
- Kth smallest element in sorted matrix
- Smallest range covering elements from K lists

**Template:**
```python
import heapq
heap = []
for i, lst in enumerate(lists):
    if lst:
        heapq.heappush(heap, (lst[0], i, 0))

result = []
while heap:
    val, list_idx, elem_idx = heapq.heappop(heap)
    result.append(val)
    if elem_idx + 1 < len(lists[list_idx]):
        next_val = lists[list_idx][elem_idx + 1]
        heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
```

---

### 24. Top K Elements
**Concept:** Maintain heap of size K to track top/bottom K elements.

**When to Use:**
- K largest/smallest elements
- K closest points
- K frequent elements

**Key Insight:** Use min heap of size K for K largest (and vice versa).

**Common Problems:**
- Kth largest element
- Top K frequent words
- K closest points to origin

**Template:**
```python
import heapq
heap = []
for x in arr:
    heapq.heappush(heap, x)
    if len(heap) > k:
        heapq.heappop(heap)
# heap contains k largest elements
```

---

## Tree Patterns

### 25. Tree DFS (Depth-First Search)
**Concept:** Traverse tree deeply before backtracking.

**When to Use:**
- Path problems
- Tree property checking
- Subtree problems

**Key Insight:** Recursive or stack-based traversal.

**Common Problems:**
- Maximum depth
- Path sum
- Lowest common ancestor

**Template:**
```python
def dfs(node):
    if not node:
        return base_case
    left = dfs(node.left)
    right = dfs(node.right)
    return process(node.val, left, right)
```

---

### 26. Tree BFS (Breadth-First Search)
**Concept:** Traverse tree level by level.

**When to Use:**
- Level-order traversal
- Minimum depth/steps
- Level-based problems

**Key Insight:** Use queue for level-by-level processing.

**Common Problems:**
- Level order traversal
- Binary tree right side view
- Minimum depth

**Template:**
```python
from collections import deque
queue = deque([root])
while queue:
    level_size = len(queue)
    for _ in range(level_size):
        node = queue.popleft()
        # Process node
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
```

---

### 27. Binary Search Tree Properties
**Concept:** Exploit BST ordering property (left < root < right).

**When to Use:**
- Validate BST
- Find/insert/delete in BST
- Kth smallest in BST

**Key Insight:** In-order traversal gives sorted order.

**Common Problems:**
- Validate BST
- Kth smallest element
- Lowest common ancestor in BST

**Template:**
```python
def inorder(node):
    if not node:
        return []
    return inorder(node.left) + [node.val] + inorder(node.right)

def validate_bst(node, min_val, max_val):
    if not node:
        return True
    if not (min_val < node.val < max_val):
        return False
    return (validate_bst(node.left, min_val, node.val) and
            validate_bst(node.right, node.val, max_val))
```

---

### 28. Trie (Prefix Tree)
**Concept:** Tree for storing strings, sharing common prefixes.

**When to Use:**
- Prefix matching
- Word search
- Auto-complete

**Key Insight:** Each node represents a character, paths represent words.

**Common Problems:**
- Implement trie
- Word search II
- Design search autocomplete

**Template:**
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
```

---

## Graph Patterns

### 29. Graph DFS
**Concept:** Explore as far as possible before backtracking.

**When to Use:**
- Path finding
- Connected components
- Cycle detection

**Key Insight:** Use recursion or stack, mark visited.

**Common Problems:**
- Number of islands
- Clone graph
- All paths from source to target

**Template:**
```python
def dfs(node, visited):
    if node in visited:
        return
    visited.add(node)
    for neighbor in graph[node]:
        dfs(neighbor, visited)
```

---

### 30. Graph BFS
**Concept:** Explore neighbors level by level.

**When to Use:**
- Shortest path in unweighted graph
- Level-based problems
- Minimum steps

**Key Insight:** Use queue, guarantees shortest path first.

**Common Problems:**
- Shortest path in binary matrix
- Word ladder
- Rotting oranges

**Template:**
```python
from collections import deque
def bfs(start):
    queue = deque([start])
    visited = {start}
    level = 0
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        level += 1
```

---

### 31. Topological Sort
**Concept:** Linear ordering of directed acyclic graph (DAG).

**When to Use:**
- Task scheduling with dependencies
- Course prerequisites
- Build order

**Key Insight:** Kahn's algorithm (BFS) or DFS with reverse postorder.

**Common Problems:**
- Course schedule
- Alien dictionary
- Sequence reconstruction

**Template:**
```python
# Kahn's Algorithm
from collections import deque
in_degree = {i: 0 for i in range(n)}
for u, v in edges:
    in_degree[v] += 1

queue = deque([i for i in range(n) if in_degree[i] == 0])
result = []
while queue:
    node = queue.popleft()
    result.append(node)
    for neighbor in graph[node]:
        in_degree[neighbor] -= 1
        if in_degree[neighbor] == 0:
            queue.append(neighbor)
return result if len(result) == n else []
```

---

### 32. Union Find (Disjoint Set)
**Concept:** Track connected components, support union and find operations.

**When to Use:**
- Dynamic connectivity
- Cycle detection in undirected graph
- Minimum spanning tree

**Key Insight:** Path compression and union by rank for efficiency.

**Common Problems:**
- Number of connected components
- Redundant connection
- Accounts merge

**Template:**
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

---

### 33. Dijkstra's Algorithm
**Concept:** Shortest path in weighted graph with non-negative weights.

**When to Use:**
- Single source shortest path
- Network delay time
- Cheapest flights

**Key Insight:** Use min heap, greedily pick shortest unvisited node.

**Common Problems:**
- Network delay time
- Path with minimum effort
- Cheapest flights within K stops

**Template:**
```python
import heapq
def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    heap = [(0, start)]
    
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = d + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    return dist
```

---

### 34. Bellman-Ford Algorithm
**Concept:** Shortest path allowing negative weights, detects negative cycles.

**When to Use:**
- Negative edge weights
- Detect negative cycles
- Relax edges V-1 times

**Key Insight:** Relax all edges repeatedly.

**Common Problems:**
- Cheapest flights within K stops
- Network delay with negative edges
- Currency arbitrage

**Template:**
```python
def bellman_ford(edges, n, start):
    dist = [float('inf')] * n
    dist[start] = 0
    
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    
    # Check for negative cycle
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return None  # Negative cycle exists
    return dist
```

---

### 35. Floyd-Warshall Algorithm
**Concept:** All-pairs shortest path using dynamic programming.

**When to Use:**
- Shortest paths between all pairs
- Dense graphs
- Transitive closure

**Key Insight:** Consider all intermediate vertices.

**Common Problems:**
- Find the city with smallest number of neighbors
- Check reachability
- Shortest path in weighted graph

**Template:**
```python
def floyd_warshall(graph, n):
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    
    for u, v, w in edges:
        dist[u][v] = w
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist
```

---

### 36. Minimum Spanning Tree (Kruskal's/Prim's)
**Concept:** Connect all vertices with minimum total edge weight.

**When to Use:**
- Minimum cost to connect all nodes
- Network design
- Clustering

**Key Insight:** Kruskal's uses union-find, Prim's uses heap.

**Common Problems:**
- Min cost to connect all points
- Optimize water distribution

**Template:**
```python
# Kruskal's Algorithm
def kruskal(edges, n):
    edges.sort(key=lambda x: x[2])  # Sort by weight
    uf = UnionFind(n)
    mst_weight = 0
    mst_edges = 0
    
    for u, v, w in edges:
        if uf.union(u, v):
            mst_weight += w
            mst_edges += 1
            if mst_edges == n - 1:
                break
    return mst_weight

# Prim's Algorithm
def prim(graph, n):
    visited = set([0])
    heap = [(w, 0, v) for v, w in graph[0]]
    heapq.heapify(heap)
    mst_weight = 0
    
    while heap and len(visited) < n:
        w, u, v = heapq.heappop(heap)
        if v in visited:
            continue
        visited.add(v)
        mst_weight += w
        for next_v, next_w in graph[v]:
            if next_v not in visited:
                heapq.heappush(heap, (next_w, v, next_v))
    return mst_weight
```

---

## Binary Search Variations

### 37. Classic Binary Search
**Concept:** Search in sorted array in O(log n).

**When to Use:**
- Find element in sorted array
- Find insertion position

**Key Insight:** Eliminate half of search space each iteration.

**Common Problems:**
- Binary search
- First/last position of element
- Search insert position

**Template:**
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

---

### 38. Modified Binary Search
**Concept:** Apply binary search on non-traditional search spaces.

**When to Use:**
- Search in rotated array
- Search in 2D matrix
- Peak element finding

**Key Insight:** Identify which half is sorted or which side to search.

**Common Problems:**
- Search in rotated sorted array
- Find minimum in rotated sorted array
- Find peak element

**Template:**
```python
# Rotated sorted array
def search_rotated(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        
        # Left half is sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

---

### 39. Binary Search on Answer
**Concept:** Binary search on possible answer range, not array indices.

**When to Use:**
- Minimize/maximize something
- "Find minimum X such that condition holds"
- Allocation problems

**Key Insight:** Search space is range of possible answers.

**Common Problems:**
- Koko eating bananas
- Capacity to ship packages
- Split array largest sum

**Template:**
```python
def binary_search_answer(arr, target):
    def is_valid(mid):
        # Check if mid satisfies the condition
        pass
    
    left, right = min_possible, max_possible
    result = right
    while left <= right:
        mid = left + (right - left) // 2
        if is_valid(mid):
            result = mid
            right = mid - 1  # Try to minimize
        else:
            left = mid + 1
    return result
```

---

## Dynamic Programming

### 40. 1D DP (Linear)
**Concept:** State depends on previous states in sequence.

**When to Use:**
- Sequences with one variable
- House robber, climbing stairs
- Maximum/minimum in sequence

**Key Insight:** `dp[i]` represents optimal solution up to index i.

**Common Problems:**
- Climbing stairs
- House robber
- Decode ways

**Template:**
```python
dp = [0] * (n + 1)
dp[0] = base_case
for i in range(1, n + 1):
    dp[i] = transition(dp[i-1], dp[i-2], ...)
return dp[n]
```

---

### 41. 2D DP (Grid/Matrix)
**Concept:** State depends on two dimensions.

**When to Use:**
- Grid path problems
- Two sequences (LCS, edit distance)
- Matrix problems

**Key Insight:** `dp[i][j]` represents solution for subproblem (i, j).

**Common Problems:**
- Unique paths
- Longest common subsequence
- Edit distance

**Template:**
```python
dp = [[0] * (n + 1) for _ in range(m + 1)]
# Initialize base cases
for i in range(1, m + 1):
    for j in range(1, n + 1):
        dp[i][j] = transition(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
return dp[m][n]
```

---

### 42. Knapsack Pattern
**Concept:** Choose items to maximize value with weight constraint.

**When to Use:**
- Subset sum
- Partition problems
- Target sum

**Key Insight:** For each item, choose to include or exclude.

**Common Problems:**
- 0/1 Knapsack
- Partition equal subset sum
- Target sum

**Template:**
```python
# 0/1 Knapsack
dp = [[0] * (capacity + 1) for _ in range(n + 1)]
for i in range(1, n + 1):
    for w in range(capacity + 1):
        if weight[i-1] <= w:
            dp[i][w] = max(dp[i-1][w], 
                          dp[i-1][w - weight[i-1]] + value[i-1])
        else:
            dp[i][w] = dp[i-1][w]
return dp[n][capacity]
```

---

### 43. Unbounded Knapsack
**Concept:** Can use items multiple times.

**When to Use:**
- Coin change
- Rod cutting
- Unlimited supply problems

**Key Insight:** Can reuse current item after selecting it.

**Common Problems:**
- Coin change
- Coin change 2
- Perfect squares

**Template:**
```python
dp = [float('inf')] * (target + 1)
dp[0] = 0
for i in range(1, target + 1):
    for coin in coins:
        if i >= coin:
            dp[i] = min(dp[i], dp[i - coin] + 1)
return dp[target] if dp[target] != float('inf') else -1
```

---

### 44. Longest Increasing Subsequence (LIS)
**Concept:** Find longest strictly increasing subsequence.

**When to Use:**
- LIS variants
- Number of increasing subsequences
- Russian doll envelopes

**Key Insight:** DP O(n²) or binary search O(n log n).

**Common Problems:**
- Longest increasing subsequence
- Number of LIS
- Russian doll envelopes

**Template:**
```python
# O(n²) approach
dp = [1] * n
for i in range(1, n):
    for j in range(i):
        if arr[j] < arr[i]:
            dp[i] = max(dp[i], dp[j] + 1)
return max(dp)

# O(n log n) approach with binary search
import bisect
tails = []
for num in arr:
    pos = bisect.bisect_left(tails, num)
    if pos == len(tails):
        tails.append(num)
    else:
        tails[pos] = num
return len(tails)
```

---

### 45. Palindrome DP
**Concept:** Check/count palindromic substrings or subsequences.

**When to Use:**
- Longest palindromic substring/subsequence
- Palindrome partitioning
- Count palindromes

**Key Insight:** `dp[i][j]` represents if substring [i, j] is palindrome.

**Common Problems:**
- Longest palindromic substring
- Longest palindromic subsequence
- Palindrome partitioning

**Template:**
```python
# Longest palindromic substring
n = len(s)
dp = [[False] * n for _ in range(n)]
start, max_len = 0, 1

for i in range(n):
    dp[i][i] = True

for length in range(2, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        if s[i] == s[j]:
            dp[i][j] = (length == 2) or dp[i+1][j-1]
            if dp[i][j] and length > max_len:
                start, max_len = i, length
return s[start:start + max_len]
```

---

### 46. DP on Strings
**Concept:** Dynamic programming with string operations.

**When to Use:**
- Edit distance
- Wildcard matching
- Regex matching

**Key Insight:** Compare characters, track operations.

**Common Problems:**
- Edit distance
- Wildcard matching
- Regular expression matching

**Template:**
```python
# Edit distance
dp = [[0] * (n + 1) for _ in range(m + 1)]
for i in range(m + 1):
    dp[i][0] = i
for j in range(n + 1):
    dp[0][j] = j

for i in range(1, m + 1):
    for j in range(1, n + 1):
        if s1[i-1] == s2[j-1]:
            dp[i][j] = dp[i-1][j-1]
        else:
            dp[i][j] = 1 + min(dp[i-1][j],    # delete
                              dp[i][j-1],      # insert
                              dp[i-1][j-1])    # replace
return dp[m][n]
```

---

### 47. DP with Bitmask
**Concept:** Use bitmask to represent state/subset.

**When to Use:**
- Subset enumeration
- TSP (Traveling Salesman)
- Assignment problems with small n

**Key Insight:** Each bit represents inclusion of element.

**Common Problems:**
- Traveling salesman problem
- Partition to K equal sum subsets
- Find shortest superstring

**Template:**
```python
# TSP with bitmask
n = len(graph)
dp = [[float('inf')] * n for _ in range(1 << n)]
dp[1][0] = 0  # Start at city 0

for mask in range(1 << n):
    for u in range(n):
        if mask & (1 << u):
            for v in range(n):
                if not (mask & (1 << v)):
                    new_mask = mask | (1 << v)
                    dp[new_mask][v] = min(dp[new_mask][v], 
                                         dp[mask][u] + graph[u][v])
return min(dp[(1 << n) - 1][i] + graph[i][0] for i in range(n))
```

---

### 48. State Machine DP
**Concept:** Model problem as state transitions.

**When to Use:**
- Stock buy/sell problems
- Problems with limited states
- Transition-based optimization

**Key Insight:** Each state has transitions to other states.

**Common Problems:**
- Best time to buy and sell stock
- Best time to buy and sell stock with cooldown
- Best time to buy and sell stock with transaction fee

**Template:**
```python
# Stock with cooldown
hold = -prices[0]  # Holding stock
sold = 0           # Just sold
rest = 0           # Resting (can buy)

for i in range(1, len(prices)):
    prev_hold, prev_sold, prev_rest = hold, sold, rest
    hold = max(prev_hold, prev_rest - prices[i])
    sold = prev_hold + prices[i]
    rest = max(prev_rest, prev_sold)
return max(sold, rest)
```

---

### 49. Interval DP
**Concept:** DP on intervals/ranges, build from smaller to larger.

**When to Use:**
- Burst balloons
- Matrix chain multiplication
- Merge stones

**Key Insight:** `dp[i][j]` represents answer for interval [i, j].

**Common Problems:**
- Burst balloons
- Minimum cost tree from leaf values
- Remove boxes

**Template:**
```python
n = len(arr)
dp = [[0] * n for _ in range(n)]

for length in range(1, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        dp[i][j] = float('inf')
        for k in range(i, j + 1):
            dp[i][j] = min(dp[i][j], 
                          dp[i][k-1] + dp[k+1][j] + cost(i, j, k))
return dp[0][n-1]
```

---

### 50. Matrix Chain Multiplication Pattern
**Concept:** Find optimal way to parenthesize matrix multiplications.

**When to Use:**
- Parenthesization problems
- Optimal binary search tree
- Polygon triangulation

**Key Insight:** Try all possible split points.

**Common Problems:**
- Matrix chain multiplication
- Burst balloons
- Minimum score triangulation

**Template:**
```python
def matrix_chain(dims):
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = (dp[i][k] + dp[k+1][j] + 
                       dims[i] * dims[k+1] * dims[j+1])
                dp[i][j] = min(dp[i][j], cost)
    return dp[0][n-1]
```

---

## Greedy Algorithms

### 51. Greedy (Activity Selection)
**Concept:** Select activities to maximize count without overlap.

**When to Use:**
- Interval scheduling
- Meeting rooms
- Non-overlapping intervals

**Key Insight:** Sort by end time, greedily select earliest ending.

**Common Problems:**
- Non-overlapping intervals
- Meeting rooms II
- Minimum arrows to burst balloons

**Template:**
```python
intervals.sort(key=lambda x: x[1])  # Sort by end time
count = 0
end = float('-inf')
for start, curr_end in intervals:
    if start >= end:
        count += 1
        end = curr_end
return count
```

---

### 52. Greedy (Two Pointers/Sorting)
**Concept:** Sort and use greedy choice with pointers.

**When to Use:**
- Assign cookies
- Boats to save people
- Two city scheduling

**Key Insight:** Sort and match optimally.

**Common Problems:**
- Assign cookies
- Boats to save people
- Queue reconstruction by height

**Template:**
```python
arr1.sort()
arr2.sort()
i, j = 0, 0
while i < len(arr1) and j < len(arr2):
    if condition:
        # Match found
        i += 1
    j += 1
```

---

### 53. Greedy (Priority Queue)
**Concept:** Use heap to make locally optimal choices.

**When to Use:**
- Task scheduling
- CPU scheduling
- Maximize profit

**Key Insight:** Process in priority order.

**Common Problems:**
- Task scheduler
- Reorganize string
- IPO

**Template:**
```python
import heapq
heap = [-freq for freq in frequencies]
heapq.heapify(heap)
result = []
while heap:
    most_freq = -heapq.heappop(heap)
    result.append(process(most_freq))
    if most_freq - 1 > 0:
        heapq.heappush(heap, -(most_freq - 1))
```

---

## Backtracking

### 54. Backtracking (Combinations)
**Concept:** Generate all combinations of elements.

**When to Use:**
- Subset generation
- Combination sum
- Letter combinations

**Key Insight:** Choose/not choose, backtrack.

**Common Problems:**
- Subsets
- Combination sum
- Letter combinations of phone number

**Template:**
```python
def backtrack(start, path):
    result.append(path[:])
    for i in range(start, n):
        path.append(arr[i])
        backtrack(i + 1, path)
        path.pop()

result = []
backtrack(0, [])
```

---

### 55. Backtracking (Permutations)
**Concept:** Generate all permutations of elements.

**When to Use:**
- Permutations
- N-Queens
- Sudoku solver

**Key Insight:** Try all positions, backtrack if invalid.

**Common Problems:**
- Permutations
- N-Queens
- Sudoku solver

**Template:**
```python
def backtrack(path):
    if len(path) == n:
        result.append(path[:])
        return
    for i in range(n):
        if arr[i] not in path:
            path.append(arr[i])
            backtrack(path)
            path.pop()

result = []
backtrack([])
```

---

### 56. Backtracking (Board/Grid)
**Concept:** Explore grid with backtracking.

**When to Use:**
- Word search
- N-Queens
- Rat in maze

**Key Insight:** Mark visited, explore, unmark.

**Common Problems:**
- Word search
- N-Queens
- Sudoku solver

**Template:**
```python
def backtrack(row, col, index):
    if index == len(word):
        return True
    if (row < 0 or row >= m or col < 0 or col >= n or 
        board[row][col] != word[index] or visited[row][col]):
        return False
    
    visited[row][col] = True
    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
        if backtrack(row + dr, col + dc, index + 1):
            return True
    visited[row][col] = False
    return False
```

---

## Bit Manipulation

### 57. Basic Bit Operations
**Concept:** Use bitwise operators for efficient operations.

**When to Use:**
- Check/set/clear bits
- Count set bits
- Power of two checking

**Key Insight:** Bits represent binary state efficiently.

**Common Problems:**
- Number of 1 bits
- Power of two
- Single number

**Template:**
```python
# Check if bit at position i is set
is_set = (num >> i) & 1

# Set bit at position i
num |= (1 << i)

# Clear bit at position i
num &= ~(1 << i)

# Toggle bit at position i
num ^= (1 << i)

# Clear rightmost set bit
num &= (num - 1)

# Get rightmost set bit
rightmost = num & -num

# Count set bits
count = bin(num).count('1')
```

---

### 58. XOR Tricks
**Concept:** XOR properties for finding unique elements.

**When to Use:**
- Find single/missing number
- Swap without temp variable
- Detect differences

**Key Insight:** `a ^ a = 0`, `a ^ 0 = a`, XOR is commutative.

**Common Problems:**
- Single number
- Single number II
- Missing number

**Template:**
```python
# Find single number (all others appear twice)
result = 0
for num in arr:
    result ^= num
return result

# Find two unique numbers (all others appear twice)
xor = 0
for num in arr:
    xor ^= num
rightmost_bit = xor & -xor
num1 = num2 = 0
for num in arr:
    if num & rightmost_bit:
        num1 ^= num
    else:
        num2 ^= num
return [num1, num2]
```

---

### 59. Bit Masking
**Concept:** Use bits to represent sets/states.

**When to Use:**
- Subset representation
- State compression
- Permissions/flags

**Key Insight:** Each bit represents presence/absence.

**Common Problems:**
- Maximum product of word lengths
- Repeated DNA sequences
- Find all duplicates

**Template:**
```python
# Check if subset represented by mask contains element i
if mask & (1 << i):
    # Element i is in subset
    pass

# Add element i to subset
mask |= (1 << i)

# Remove element i from subset
mask &= ~(1 << i)

# Iterate through all subsets
for mask in range(1 << n):
    # Process subset represented by mask
    pass
```

---

## Mathematical & Geometric

### 60. Math (Prime Numbers)
**Concept:** Prime generation, factorization.

**When to Use:**
- Count primes
- Prime factorization
- GCD/LCM problems

**Key Insight:** Sieve of Eratosthenes for multiple primes.

**Common Problems:**
- Count primes
- Ugly number II
- Perfect squares

**Template:**
```python
# Sieve of Eratosthenes
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(n + 1) if is_prime[i]]

# Check if prime
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

---

### 61. Math (GCD/LCM)
**Concept:** Greatest common divisor and least common multiple.

**When to Use:**
- Simplify fractions
- Find common factors
- Modular arithmetic

**Key Insight:** Euclidean algorithm for GCD.

**Common Problems:**
- Greatest common divisor
- Fraction addition
- X of a kind in deck of cards

**Template:**
```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return (a * b) // gcd(a, b)

# GCD of array
from functools import reduce
result = reduce(gcd, arr)
```

---

### 62. Math (Combinatorics)
**Concept:** Count combinations, permutations.

**When to Use:**
- Pascal's triangle
- Binomial coefficients
- Counting problems

**Key Insight:** Use formulas or DP to compute.

**Common Problems:**
- Pascal's triangle
- Unique paths
- Climbing stairs

**Template:**
```python
# Combinations C(n, k)
def combinations(n, k):
    if k > n - k:
        k = n - k
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

# Permutations P(n, k)
def permutations(n, k):
    result = 1
    for i in range(k):
        result *= (n - i)
    return result
```

---

### 63. Catalan Numbers
**Concept:** Sequence counting various combinatorial structures.

**When to Use:**
- Valid parentheses combinations
- Unique BSTs
- Path counting with restrictions

**Key Insight:** `C(n) = sum(C(i) * C(n-1-i))` or `C(n) = (2n)! / ((n+1)! * n!)`

**Common Problems:**
- Unique binary search trees
- Generate parentheses (counting)
- Number of ways to triangulate polygon

**Template:**
```python
# Compute nth Catalan number
def catalan(n):
    if n <= 1:
        return 1
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i-1-j]
    return dp[n]
```

---

### 64. Geometry (Line Sweep)
**Concept:** Process events at specific coordinates.

**When to Use:**
- Overlapping intervals
- Skyline problem
- Meeting rooms

**Key Insight:** Create events, sort, process sequentially.

**Common Problems:**
- Skyline problem
- Meeting rooms II
- My calendar problems

**Template:**
```python
events = []
for start, end in intervals:
    events.append((start, 1))    # Start event
    events.append((end, -1))     # End event
events.sort()

active = 0
max_active = 0
for time, delta in events:
    active += delta
    max_active = max(max_active, active)
```

---

### 65. Geometry (Manhattan Distance)
**Concept:** Distance in grid: `|x1 - x2| + |y1 - y2|`

**When to Use:**
- Grid distance problems
- Best meeting point
- K closest points (grid)

**Key Insight:** Can separate x and y coordinates.

**Common Problems:**
- Best meeting point
- Minimum cost to connect all points
- K closest points

**Template:**
```python
# Meeting point (minimize total Manhattan distance)
def meeting_point(points):
    x_coords = sorted([p[0] for p in points])
    y_coords = sorted([p[1] for p in points])
    median_x = x_coords[len(x_coords) // 2]
    median_y = y_coords[len(y_coords) // 2]
    return (median_x, median_y)
```

---

### 66. Geometry (Euclidean Distance)
**Concept:** Straight-line distance: `sqrt((x1-x2)² + (y1-y2)²)`

**When to Use:**
- K closest points to origin
- Minimum distance problems
- Circle/radius problems

**Key Insight:** Can compare squared distances to avoid sqrt.

**Common Problems:**
- K closest points to origin
- Valid square
- Minimum area rectangle

**Template:**
```python
import math

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Or squared distance (for comparison)
def squared_distance(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
```

---

## Advanced Data Structures

### 67. Binary Indexed Tree (Fenwick Tree)
**Concept:** Efficient range sum queries and point updates.

**When to Use:**
- Range sum with updates
- Inversion count
- Dynamic cumulative frequency

**Key Insight:** Store partial sums in tree structure.

**Common Problems:**
- Range sum query mutable
- Count smaller numbers after self
- Reverse pairs

**Template:**
```python
class BIT:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)
    
    def query(self, i):
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s
    
    def range_query(self, l, r):
        return self.query(r) - self.query(l - 1)
```

---

### 68. Segment Tree
**Concept:** Tree for range queries (min, max, sum, GCD) with updates.

**When to Use:**
- Range minimum/maximum query
- Range sum with updates
- Lazy propagation for range updates

**Key Insight:** Each node stores aggregate of its range.

**Common Problems:**
- Range sum query mutable
- Range minimum query
- Count of range sum

**Template:**
```python
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.build(arr, 0, 0, self.n - 1)
    
    def build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self.build(arr, 2*node+1, start, mid)
            self.build(arr, 2*node+2, mid+1, end)
            self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]
    
    def update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self.update(2*node+1, start, mid, idx, val)
            else:
                self.update(2*node+2, mid+1, end, idx, val)
            self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]
    
    def query(self, node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        return (self.query(2*node+1, start, mid, l, r) +
                self.query(2*node+2, mid+1, end, l, r))
```

---

### 69. Sparse Table
**Concept:** Static range queries in O(1) after O(n log n) preprocessing.

**When to Use:**
- Static array (no updates)
- Range min/max queries
- GCD queries

**Key Insight:** Precompute all power-of-2 length ranges.

**Common Problems:**
- Range minimum query (static)
- Range GCD query
- Range maximum query

**Template:**
```python
import math

class SparseTable:
    def __init__(self, arr):
        n = len(arr)
        k = int(math.log2(n)) + 1
        self.st = [[0] * k for _ in range(n)]
        
        # Initialize for intervals of length 1
        for i in range(n):
            self.st[i][0] = arr[i]
        
        # Build sparse table
        j = 1
        while (1 << j) <= n:
            i = 0
            while (i + (1 << j) - 1) < n:
                self.st[i][j] = min(self.st[i][j-1], 
                                   self.st[i + (1 << (j-1))][j-1])
                i += 1
            j += 1
    
    def query(self, l, r):
        j = int(math.log2(r - l + 1))
        return min(self.st[l][j], self.st[r - (1 << j) + 1][j])
```

---

### 70. Sqrt Decomposition
**Concept:** Divide array into sqrt(n) blocks for efficient queries.

**When to Use:**
- Range queries with updates
- Simpler alternative to segment tree
- Block-based processing

**Key Insight:** Balance between brute force and complex data structures.

**Common Problems:**
- Range sum query mutable
- Range minimum query
- Mo's algorithm problems

**Template:**
```python
import math

class SqrtDecomposition:
    def __init__(self, arr):
        self.n = len(arr)
        self.block_size = int(math.sqrt(self.n))
        self.num_blocks = (self.n + self.block_size - 1) // self.block_size
        self.blocks = [0] * self.num_blocks
        self.arr = arr[:]
        
        for i in range(self.n):
            self.blocks[i // self.block_size] += arr[i]
    
    def update(self, idx, val):
        block_idx = idx // self.block_size
        self.blocks[block_idx] += val - self.arr[idx]
        self.arr[idx] = val
    
    def query(self, l, r):
        result = 0
        while l <= r:
            if l % self.block_size == 0 and l + self.block_size - 1 <= r:
                result += self.blocks[l // self.block_size]
                l += self.block_size
            else:
                result += self.arr[l]
                l += 1
        return result
```

---

## Specialized Techniques

### 71. Rolling Hash (Rabin-Karp)
**Concept:** Hash function that can be efficiently updated for sliding window.

**When to Use:**
- String pattern matching
- Substring comparison
- Duplicate detection

**Key Insight:** Hash(s[i+1..j+1]) can be computed from Hash(s[i..j]) in O(1).

**Common Problems:**
- Longest duplicate substring
- Repeated DNA sequences
- Find all anagrams in string (with sorting)

**Template:**
```python
class RollingHash:
    def __init__(self, s, base=26, mod=10**9 + 7):
        self.base = base
        self.mod = mod
        self.n = len(s)
        self.hash = [0] * (self.n + 1)
        self.pow = [1] * (self.n + 1)
        
        for i in range(self.n):
            self.hash[i + 1] = (self.hash[i] * base + ord(s[i])) % mod
            self.pow[i + 1] = (self.pow[i] * base) % mod
    
    def get_hash(self, l, r):
        # Hash of s[l:r+1]
        return (self.hash[r + 1] - self.hash[l] * self.pow[r - l + 1]) % self.mod

# Simple rolling hash for pattern matching
def rabin_karp(text, pattern):
    base, mod = 256, 10**9 + 7
    m, n = len(pattern), len(text)
    
    pattern_hash = 0
    text_hash = 0
    h = pow(base, m - 1, mod)
    
    # Calculate initial hash
    for i in range(m):
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % mod
        text_hash = (base * text_hash + ord(text[i])) % mod
    
    matches = []
    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            if text[i:i+m] == pattern:
                matches.append(i)
        
        if i < n - m:
            text_hash = (base * (text_hash - ord(text[i]) * h) + 
                        ord(text[i + m])) % mod
            if text_hash < 0:
                text_hash += mod
    
    return matches
```

---

### 72. KMP Algorithm
**Concept:** Pattern matching using failure function to avoid redundant comparisons.

**When to Use:**
- String pattern matching
- Find all occurrences of pattern
- Prefix-suffix matching

**Key Insight:** Precompute longest proper prefix which is also suffix.

**Common Problems:**
- Implement strStr()
- Shortest palindrome
- Repeated substring pattern

**Template:**
```python
def kmp_search(text, pattern):
    def compute_lps(pattern):
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps
    
    n, m = len(text), len(pattern)
    lps = compute_lps(pattern)
    matches = []
    
    i = j = 0
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches
```

---

### 73. Z-Algorithm
**Concept:** Find all occurrences of pattern in linear time using Z-array.

**When to Use:**
- Pattern matching
- Prefix matching
- String analysis

**Key Insight:** Z[i] = length of longest substring starting at i which is also prefix.

**Common Problems:**
- Pattern matching
- Shortest palindrome
- String matching with wildcards

**Template:**
```python
def z_algorithm(s):
    n = len(s)
    z = [0] * n
    l, r = 0, 0
    
    for i in range(1, n):
        if i > r:
            l, r = i, i
            while r < n and s[r - l] == s[r]:
                r += 1
            z[i] = r - l
            r -= 1
        else:
            k = i - l
            if z[k] < r - i + 1:
                z[i] = z[k]
            else:
                l = i
                while r < n and s[r - l] == s[r]:
                    r += 1
                z[i] = r - l
                r -= 1
    
    return z

def pattern_search(text, pattern):
    s = pattern + ' + text
    z = z_algorithm(s)
    matches = []
    
    for i in range(len(pattern) + 1, len(s)):
        if z[i] == len(pattern):
            matches.append(i - len(pattern) - 1)
    
    return matches
```

---

### 74. Manacher's Algorithm
**Concept:** Find all palindromic substrings in linear time.

**When to Use:**
- Longest palindromic substring
- Count all palindromes
- Palindrome optimization

**Key Insight:** Expand around center with mirroring to avoid redundant checks.

**Common Problems:**
- Longest palindromic substring
- Count palindromic substrings
- Palindrome partitioning optimization

**Template:**
```python
def manacher(s):
    # Preprocess: insert '#' between characters
    t = '#'.join('^{}.format(s))
    n = len(t)
    p = [0] * n
    center = right = 0
    
    for i in range(1, n - 1):
        # Mirror of i
        mirror = 2 * center - i
        
        if i < right:
            p[i] = min(right - i, p[mirror])
        
        # Expand around center i
        while t[i + p[i] + 1] == t[i - p[i] - 1]:
            p[i] += 1
        
        # Update center and right boundary
        if i + p[i] > right:
            center, right = i, i + p[i]
    
    # Find longest palindrome
    max_len, center_idx = max((n, i) for i, n in enumerate(p))
    start = (center_idx - max_len) // 2
    return s[start:start + max_len]
```

---

### 75. Reservoir Sampling
**Concept:** Random sampling from stream of unknown length.

**When to Use:**
- Sample from large dataset
- Random selection from stream
- Fair random selection

**Key Insight:** Each element has equal probability of being selected.

**Common Problems:**
- Random pick index
- Linked list random node
- Random pick with weight

**Template:**
```python
import random

# Sample k items from stream
def reservoir_sample(stream, k):
    reservoir = []
    
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    
    return reservoir

# Sample one item
class RandomPick:
    def __init__(self, nums):
        self.nums = nums
    
    def pick(self, target):
        count = 0
        result = -1
        for i, num in enumerate(self.nums):
            if num == target:
                count += 1
                if random.randint(1, count) == 1:
                    result = i
        return result
```

---

### 76. Boyer-Moore Voting Algorithm
**Concept:** Find majority element (appears > n/2 times) in O(n) time, O(1) space.

**When to Use:**
- Majority element
- Candidate selection
- Finding frequent elements

**Key Insight:** Majority element survives pair-wise cancellation.

**Common Problems:**
- Majority element
- Majority element II (> n/3)
- Online majority element detection

**Template:**
```python
# Find majority element (> n/2)
def majority_element(nums):
    candidate = None
    count = 0
    
    for num in nums:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    
    return candidate

# Find elements appearing > n/3 times
def majority_element_ii(nums):
    candidate1, candidate2 = None, None
    count1, count2 = 0, 0
    
    for num in nums:
        if num == candidate1:
            count1 += 1
        elif num == candidate2:
            count2 += 1
        elif count1 == 0:
            candidate1, count1 = num, 1
        elif count2 == 0:
            candidate2, count2 = num, 1
        else:
            count1 -= 1
            count2 -= 1
    
    # Verify candidates
    return [c for c in [candidate1, candidate2] 
            if nums.count(c) > len(nums) // 3]
```

---

### 77. Game Theory / Minimax
**Concept:** Optimal strategy in two-player games.

**When to Use:**
- Stone games
- Nim games
- Turn-based games

**Key Insight:** Player chooses move that maximizes their advantage.

**Common Problems:**
- Stone game
- Predict the winner
- Nim game

**Template:**
```python
# Minimax with memoization
def minimax(state, memo={}):
    if is_terminal(state):
        return evaluate(state)
    
    if state in memo:
        return memo[state]
    
    if is_max_turn(state):
        value = float('-inf')
        for move in get_moves(state):
            value = max(value, minimax(make_move(state, move), memo))
    else:
        value = float('inf')
        for move in get_moves(state):
            value = min(value, minimax(make_move(state, move), memo))
    
    memo[state] = value
    return value

# Simple stone game
def stone_game(piles):
    n = len(piles)
    dp = [[0] * n for _ in range(n)]
    
    for i in range(n):
        dp[i][i] = piles[i]
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(piles[i] - dp[i+1][j], 
                          piles[j] - dp[i][j-1])
    
    return dp[0][n-1] > 0
```

---

### 78. Matrix Exponentiation
**Concept:** Compute matrix power efficiently for recurrence relations.

**When to Use:**
- Fibonacci in O(log n)
- Linear recurrence optimization
- Graph path counting

**Key Insight:** Use binary exponentiation on matrices.

**Common Problems:**
- Fibonacci number
- Climbing stairs (large n)
- Count paths in graph

**Template:**
```python
def matrix_mult(A, B, mod=10**9+7):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % mod
    return C

def matrix_pow(M, n, mod=10**9+7):
    size = len(M)
    result = [[1 if i == j else 0 for j in range(size)] 
              for i in range(size)]
    
    while n > 0:
        if n % 2 == 1:
            result = matrix_mult(result, M, mod)
        M = matrix_mult(M, M, mod)
        n //= 2
    
    return result

# Fibonacci using matrix exponentiation
def fibonacci(n):
    if n <= 1:
        return n
    M = [[1, 1], [1, 0]]
    result = matrix_pow(M, n - 1)
    return result[0][0]
```

---

### 79. Meet in the Middle
**Concept:** Split problem into two halves, solve separately, combine.

**When to Use:**
- Subset sum with smaller space
- Two-set problems
- Optimization with large n

**Key Insight:** Reduces O(2^n) to O(2^(n/2)).

**Common Problems:**
- Partition equal subset sum (optimization)
- Closest subsequence sum
- 4Sum II

**Template:**
```python
def meet_in_middle(arr, target):
    n = len(arr)
    mid = n // 2
    
    # Generate all sums from first half
    left_sums = {}
    for mask in range(1 << mid):
        s = sum(arr[i] for i in range(mid) if mask & (1 << i))
        left_sums[s] = left_sums.get(s, 0) + 1
    
    # Generate all sums from second half and check
    count = 0
    for mask in range(1 << (n - mid)):
        s = sum(arr[mid + i] for i in range(n - mid) if mask & (1 << i))
        if target - s in left_sums:
            count += left_sums[target - s]
    
    return count
```

---

### 80. Eulerian Path/Circuit
**Concept:** Path/circuit visiting every edge exactly once.

**When to Use:**
- Reconstruct itinerary
- Valid arrangement of pairs
- Chinese postman problem

**Key Insight:** Eulerian path exists if graph has 0 or 2 odd-degree vertices.

**Common Problems:**
- Reconstruct itinerary
- Valid arrangement of pairs
- Cracking the safe

**Template:**
```python
from collections import defaultdict, deque

def find_eulerian_path(graph):
    # Check in-degree and out-degree
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    
    for u in graph:
        out_degree[u] = len(graph[u])
        for v in graph[u]:
            in_degree[v] += 1
    
    # Find start vertex
    start = list(graph.keys())[0]
    for node in graph:
        if out_degree[node] - in_degree[node] == 1:
            start = node
            break
    
    # Hierholzer's algorithm
    stack = [start]
    path = []
    
    while stack:
        curr = stack[-1]
        if graph[curr]:
            next_node = graph[curr].pop()
            stack.append(next_node)
        else:
            path.append(stack.pop())
    
    return path[::-1]
```

---

### 81. Line Sweep with Events
**Concept:** Process geometric events in sorted order.

**When to Use:**
- Interval problems
- Rectangle area/perimeter
- Skyline problem

**Key Insight:** Convert to events, process in sorted order.

**Common Problems:**
- Skyline problem
- Rectangle area
- Employee free time

**Template:**
```python
# Skyline problem
def get_skyline(buildings):
    events = []
    for left, right, height in buildings:
        events.append((left, -height, 0))   # Start: negative height
        events.append((right, height, 1))    # End: positive height
    
    events.sort()
    
    from collections import defaultdict
    import heapq
    
    result = []
    heights = [0]
    
    for x, h, typ in events:
        if typ == 0:  # Building starts
            heapq.heappush(heights, h)
        else:  # Building ends
            heights.remove(-h)
            heapq.heapify(heights)
        
        max_h = -heights[0]
        
        if not result or result[-1][1] != max_h:
            result.append([x, max_h])
    
    return result
```

---

### 82. Intervals (Merge & Operations)
**Concept:** Merge, intersect, or operate on intervals.

**When to Use:**
- Merge intervals
- Insert interval
- Interval intersection

**Key Insight:** Sort intervals, then process sequentially.

**Common Problems:**
- Merge intervals
- Insert interval
- Interval list intersections

**Template:**
```python
# Merge overlapping intervals
def merge_intervals(intervals):
    if not intervals:
        return []
    
    intervals.sort()
    merged = [intervals[0]]
    
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    
    return merged

# Interval intersection
def interval_intersection(A, B):
    i = j = 0
    result = []
    
    while i < len(A) and j < len(B):
        start = max(A[i][0], B[j][0])
        end = min(A[i][1], B[j][1])
        
        if start <= end:
            result.append([start, end])
        
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    
    return result
```

---

### 83. Custom Data Structures
**Concept:** Design data structure for specific operations.

**When to Use:**
- LRU Cache
- LFU Cache
- Design custom structures

**Key Insight:** Combine multiple data structures for optimal performance.

**Common Problems:**
- LRU Cache
- LFU Cache
- Design HashMap

**Template:**
```python
# LRU Cache
class LRUCache:
    class Node:
        def __init__(self, key=0, val=0):
            self.key = key
            self.val = val
            self.prev = None
            self.next = None
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_head(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def get(self, key):
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_to_head(node)
        return node.val
    
    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        node = self.Node(key, value)
        self._add_to_head(node)
        self.cache[key] = node
        
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
```

---

### 84. Expand Around Center
**Concept:** Expand from center to find palindromes or patterns.

**When to Use:**
- Longest palindromic substring
- Count palindromic substrings
- Pattern expansion

**Key Insight:** Check both odd and even length palindromes.

**Common Problems:**
- Longest palindromic substring
- Palindromic substrings
- Longest palindrome by concatenating

**Template:**
```python
def expand_around_center(s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return right - left - 1

def longest_palindrome(s):
    if not s:
        return ""
    
    start, max_len = 0, 0
    
    for i in range(len(s)):
        # Odd length palindrome
        len1 = expand_around_center(s, i, i)
        # Even length palindrome
        len2 = expand_around_center(s, i, i + 1)
        
        curr_len = max(len1, len2)
        if curr_len > max_len:
            max_len = curr_len
            start = i - (curr_len - 1) // 2
    
    return s[start:start + max_len]
```

---

### 85. Subsets Generation
**Concept:** Generate all subsets using iteration or backtracking.

**When to Use:**
- Power set generation
- Subset sum
- Combination problems

**Key Insight:** Use bitmask or backtracking.

**Common Problems:**
- Subsets
- Subsets II (with duplicates)
- Combination sum

**Template:**
```python
# Iterative approach with bitmask
def subsets_iterative(nums):
    n = len(nums)
    result = []
    for mask in range(1 << n):
        subset = [nums[i] for i in range(n) if mask & (1 << i)]
        result.append(subset)
    return result

# Backtracking approach
def subsets_backtrack(nums):
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result

# With duplicates
def subsets_with_dup(nums):
    nums.sort()
    result = []
    
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result
```

---

## Quick Reference: Pattern Selection Guide

### By Problem Characteristic

**Array/String Problems:**
- Fixed/variable sliding window
- Two pointers
- Prefix sum
- Kadane's algorithm

**Counting/Frequency:**
- Hash map
- Bucket sort
- Counting sort

**Optimization Problems:**
- Dynamic programming
- Greedy algorithms
- Binary search on answer

**Graph/Tree Problems:**
- DFS/BFS
- Union Find
- Topological sort
- Shortest path algorithms

**Range Queries:**
- Segment tree
- Binary indexed tree
- Sparse table

**String Matching:**
- KMP
- Rabin-Karp
- Z-algorithm

**Subset/Combination:**
- Backtracking
- Bitmask DP
- Meet in the middle

---

## Complexity Analysis Summary

| Pattern | Time | Space | Use Case |
|---------|------|-------|----------|
| Two Pointers | O(n) | O(1) | Sorted array, pairs |
| Sliding Window | O(n) | O(k) | Subarray problems |
| Binary Search | O(log n) | O(1) | Sorted search |
| DFS/BFS | O(V+E) | O(V) | Graph traversal |
| Union Find | O(α(n)) | O(n) | Connectivity |
| Dijkstra | O((V+E)log V) | O(V) | Shortest path |
| Segment Tree | O(log n) | O(n) | Range queries |
| Trie | O(m) | O(ALPHABET*n*m) | Prefix matching |
| DP | Varies | Varies | Optimization |
| Backtracking | O(2^n) or O(n!) | O(n) | Generate all solutions |

---

## Tips for Pattern Recognition

1. **Keywords to patterns:**
   - "Maximum/minimum subarray" → Sliding window, Kadane's, DP
   - "All combinations/permutations" → Backtracking
   - "Shortest path" → BFS, Dijkstra
   - "Longest increasing" → DP, Binary search
   - "Range query" → Segment tree, Prefix sum
   - "Connected components" → Union Find, DFS

2. **Constraints hint at approach:**
   - n ≤ 20 → Bitmask, backtracking, meet in middle
   - n ≤ 100 → O(n³) DP
   - n ≤ 1000 → O(n²) DP
   - n ≤ 10^6 → O(n log n) or O(n)
   - Array is sorted → Binary search, two pointers

3. **Problem type to pattern:**
   - Optimization → DP, Greedy
   - Counting → DP, Combinatorics, Hash map
   - Decision (Yes/No) → Greedy, Binary search on answer
   - Construction → Backtracking, Greedy

---

*This reference covers 85+ coding patterns commonly seen in technical interviews. Master these patterns and you'll be well-equipped to tackle most coding interview problems!*