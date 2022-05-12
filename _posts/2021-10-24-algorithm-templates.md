---
title: "Algorithm templates"
date: 2021-10-24 9:08:22 +0800
layout: post
toc: true
toc_sticky: true
tags: [algorithm, templates]
---

# Sorting

## 3-way quick sort

[LC 75.Sort Colors](https://leetcode.com/problems/sort-colors/)<br>
[LC 1356.Sort Integers by The Number of 1 Bits](https://leetcode.com/problems/sort-integers-by-the-number-of-1-bits/)<br>
[LC 1636.Sort Array by Increasing Frequency](https://leetcode.com/problems/sort-array-by-increasing-frequency/)<br>
[LC 1122.Relative Sort Array](https://leetcode.com/problems/relative-sort-array/)<br>
[LC 2191.Sort the Jumbled Numbers](https://leetcode.com/problems/sort-the-jumbled-numbers)<br>
[LC 905.Sort Array By Parity](https://leetcode.com/problems/sort-array-by-parity/)<br>

```java
// [l, r)
void sort(int[] arr, int l, int r) {
  if (l > r) return;

  int i = l, j = r;
  int t = l;
  int pivot = arr[l + (r - l) / 2] // or arr[l + random.nextInt(r - l)];

  while (t <= j) {
    if (arr[t] < pivot) {
      swap(arr, t++, i++);
    } else if (arr[t] > pivot) {
      swap(arr, t, j--);
    } else {
      t++;
    }
  }

  //
  // After t > j we can partition arr to [L L L L L L L L | E E E E E E E | G G G G G G G G G ]
  //                                      l                 i           j                   r
  // L: Number that less than pivot
  // E: Number that equals to pivot
  // G: Number that greater than pivot
  //

  sort(arr, l, i - 1);
  sort(arr, j + 1, r);
}
```

### Quick Select

[LC 1387.Sort Integers by The Power Value](https://leetcode.com/problems/sort-integers-by-the-power-value/)<br>
[LC 973.K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)<br>

```java
private int[] kth(int[][] arr, int l, int r, int k) {
  if (l == r) return arr[l];

  int i = l, j = r;
  int t = l;
  int[] pivot = arr[(l + r) >> 1];

  while (t <= j) {
    if (compareTo(arr[t], pivot) < 0) {
      swap(arr, i++, t++);
    } else if (compareTo(arr[t], pivot) > 0) {
      swap(arr, j--, t);
    } else {
      t++;
    }
  }

  if (i - l >= k) {
    //         k
    // L L L L L L E E E E E G G G G G G G
    //             i       j
    return kth(arr, l, i - 1, k);
  } else if (j - l + 1 < k) {
    //                         k
    // L L L L L L E E E E E G G G G G G G
    //             i       j
    return kth(arr, j + 1, r, k - (j - l + 1));
  } else {
    return pivot;
  }
}
```

# Dynamic Programming

## Choose items with presum

[LC 2209.Minimum White Tiles After Covering With Carpets](https://leetcode.com/problems/minimum-white-tiles-after-covering-with-carpets/)<br>
[LC 2218.Maximum Value of K Coins From Piles](https://leetcode.com/problems/maximum-value-of-k-coins-from-piles/)<br>
[LC 410.Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/)<br>

```java
class Solution {
  private List<List<Integer>> piles;
  private int[][] memo;
  
  public int maxValueOfCoins(List<List<Integer>> piles, int k) {
    this.piles = piles;
    this.memo = new int[piles.size()][k + 1];

    for (int i = 0; i < piles.size(); i++) {
      Arrays.fill(memo[i], -1);
    }

    return dp(piles.size() - 1, k);
  }

  private int dp(int i, int k) {
    if (k == 0) return 0;
    if (i < 0) return Integer.MIN_VALUE >> 1;
    if (memo[i][k] != -1) return memo[i][k];
    
    List<Integer> pile = piles.get(i);
    int n = pile.size();
    
    // int[] presum = new int[n + 1];
    // for (int j = 1; j <= n; j++) {
    //   presum[j] = presum[j - 1] + pile.get(j - 1);
    // }

    int ans = Integer.MIN_VALUE >> 1;
    int v = 0;
    for (int j = 0; j <= Math.min(n, k); j++) {
      // int v = presum[j];
      
      ans = Math.max(ans, dp(i - 1, k - j) + v);
      
      if (j < n) v += pile.get(j);
    }
    
    return memo[i][k] = ans;
  }
}
```

## knapsack

### 0-1 knapsack

[LQ 186.糖果](https://www.lanqiao.cn/problems/186/learning/)<br>
[LC 416.Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)<br>
[LC 2212.Maximum Points in an Archery Competition](https://leetcode.com/problems/maximum-points-in-an-archery-competition/)<br>

```java
// dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - W[i]] + V[i])
int knapsack(int[] W, int[] V, int n, int capacity) {
  int[] dp = new int[capacity + 1];

  for (int i = 0; i < n; i++) {
    for (int j = capacity; j >= W[i]; j--) {
      dp[j] = Math.max(dp[j], dp[j - W[i]] + V[i]);
    }
  }

  return dp[capacity];
}
```

### Bounded knapsack

```java
// dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - W[i] * k] + V[i] * k)
int knapsack(int[] W, int[] V, int[] S, int n, int capacity) {
  int[] dp = new int[capacity + 1];

  for (int i = 0; i < n; i++) {
    for (int j = capacity; j >= W[i]; j--) {
      for (int k = 1; k <= S[i]; k++) {
        if (W[i] * k > j) break;
        dp[j] = Math.max(dp[j], dp[j - W[i] * k] + V[i] * k);
      }
    }
  }

  return dp[capacity];
}
```

### Unbounded knapsack

![unbounded_knapsack](/assets/img/algorithm_templates/unbounded_knapsack.png)

[ACWING 3.完全背包问题](https://www.acwing.com/problem/content/3/)<br>
[LC 322.Coin Change](https://leetcode.com/problems/coin-change/)<br>
[LC 518.Coin Change 2](https://leetcode.com/problems/coin-change-2/)<br>

```java
int knapsack(int[] W, int[] V, int n, int capacity) {
  int[] dp = new int[capacity + 1];

  for (int i = 0; i < n; i++) {
    for (int j = W[i]; j <= capacity; j++) {
      dp[j] = Math.max(dp[j], dp[j - W[i]] + V[i]);
    }
  }

  return dp[capacity];
}
```

## String

### LCS (Longest Common Subsequence)

[LC 1143.Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)<br>
[LC 516.Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)<br>
[LC 392.Is Subsequence](https://leetcode.com/problems/is-subsequence/)<br>

```java
int LCS(String s, String t) {
  int n = s.length(), m = t.length();

  // dp[i][j] := the longest common subsequence length of s[0:i] and t[0:j]
  int[][] dp = new int[n + 1][m + 1];

  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= m; j++) {
      if (s.charAt(i - 1) == t.charAt(j - 1)) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }

  return dp[n][m];
}
```

### Edit Distance

![edit_distance](/assets/img/algorithm_templates/edit_distance.png)

[LC 72.Edit Distance](https://leetcode.com/problems/edit-distance/submissions/)<br>

```java
int minDistance(String s, String t) {
  int n = s.length(), m = t.length();
  int[][] dp = new int[n + 1][m + 1];

  // s is empty
  for (int j = 0; j <= m; j++) {
    dp[0][j] = j;
  }

  // t is empty
  for (int i = 0; i <= n; i++) {
    dp[i][0] = i;
  }

  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= m; j++) {
      if (s.charAt(i - 1) == t.charAt(j - 1)) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        //
        // s [xxxxx] A
        //
        // t [xxxxx] B
        //
        dp[i][j] = Math.min(dp[i - 1][j] + 1,        // delete(A)
                   Math.min(dp[i][j - 1] + 1,        // insert(B)
                            dp[i - 1][j - 1] + 1));  // replace(A, B)
      }
    }
  }

  return dp[n][m];
}
```

### LIS (Longest Increasing Subsequence)

[LC 300.Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)<br>
[LC 1964.Find the Longest Valid Obstacle Course at Each Position](https://leetcode.com/problems/find-the-longest-valid-obstacle-course-at-each-position/)<br>
[LC 1671.Minimum Number of Removals to Make Mountain Array](https://leetcode.com/problems/minimum-number-of-removals-to-make-mountain-array/)<br>
[LC 334.Increasing Triplet Subsequence](https://leetcode.com/problems/increasing-triplet-subsequence/)<br>

```java
class Solution {
  public int lengthOfLIS(int[] nums) {
    List<Integer> list = new ArrayList<>();

    for (var num : nums) {
      int m = lowerBound(list, num);

      if (m == list.size()) {
        list.add(num);
      } else {
        list.set(m, num);
      }
    }

    return list.size();
  }

  private int lowerBound(List<Integer> list, int x) {
    int l = 0, r = list.size();

    while (l < r) {
      int m = l + (r - l) / 2;
      if (list.get(m) < x) {
        l = m + 1;
      } else {
        r = m;
      }
    }

    return l;
  }
}
```

## 2D Grid Path

[LC 741.Cherry Pickup](https://leetcode.com/problems/cherry-pickup/)<br>
[LC 1463.Cherry Pickup II](https://leetcode.com/problems/cherry-pickup-ii/)<br>
[LQ 3000.拿金币](http://lx.lanqiao.cn/problem.page?gpid=T3000)<br>
[LQ 79.方格取数](http://lx.lanqiao.cn/problem.page?gpid=T79)<br>

```java
class Solution {
  private int m, n;
  private int[][] grid;
  private Integer[][][] memo;
  private int[][] moves = {
  // x1 y1 x2
    {1, 0, 1, 0},
    {1, 0, 0, 1},
    {0, 1, 1, 0},
    {0, 1, 0, 1}
  //         y2(useless)
  };

  public int cherryPickup(int[][] grid) {
    this.m = grid.length;
    this.n = grid[0].length;
    this.grid = grid;
    this.memo = new Integer[n][m][n];

    // robot 1: (0, 0)
    // robot 2: (0, 0)
    //              ^
    //             don't need
    //
    //
    // x1 + y1 = x2 + y2
    // y2      = x1 + y1 - x2
    return Math.max(0, dp(0, 0, 0));
  }

  private int dp(int x1, int y1, int x2) {
    int y2 = x1 + y1 - x2;

    if (x1 >= m || x2 >= m ||
        y1 >= n || y2 >= n) return Integer.MIN_VALUE >> 1;
    if (grid[y1][x1] == -1 ||
        grid[y2][x2] == -1) return Integer.MIN_VALUE >> 1;
    if (x1 == n - 1 && y1 == m - 1) return grid[y1][x1];
    if (memo[x1][y1][x2] != null) return memo[x1][y1][x2];

    int res = Integer.MIN_VALUE >> 1;

    for (var move : moves) {
      res = Math.max(res, dp(x1 + move[0], y1 + move[1], x2 + move[2]));
    }

    res += grid[y1][x1];
    if (y1 != y2 && x1 != x2) res += grid[y2][x2];

    return memo[x1][y1][x2] = res;
  }
}
```

## Sparse Table

[ACWING 1270.数列区间最大值](https://www.acwing.com/problem/content/description/1272/)<br>

It can be used for (min, max, gcd) range query, below is the MinSparseTable implementation.

```java
class SparseTable {
  private int[] log2;

  //
  // dp[p][i] := minimum number in arr[i: i + 2**p - 1]
  //
  private int[][] dp;

  public SparseTable(int[] arr) {
    computeLog2(arr.length);

    int n = arr.length;
    this.dp = new int[log2[n] + 1][n];

    for (int i = 0; i < n; i++) {
      dp[0][i] = arr[i];
    }

    for (int p = 1; p < dp.length; p++) {
      for (int i = 0; i + (1 << p - 1) < n; i++) {
        dp[p][i] = Math.max(dp[p - 1][i], dp[p - 1][i + (1 << p - 1)]);
      }
    }
  }

  private void computeLog2(int n) {
    this.log2 = new int[n + 1];
    for (int i = 2; i <= n; i++) {
      log2[i] = log2[i / 2] + 1;
    }
  }

  public int query(int l, int r) {
    int p = log2[r - l + 1];
    return Math.max(dp[p][l], dp[p][r - (1 << p) + 1]);
  }
}
```

# Graph

## Shortest Path

### Dijkstra

[LC 45.Jump Game II](https://leetcode.com/problems/jump-game-ii)<br>
[LC 743.Network Delay Time](https://leetcode.com/problems/network-delay-time/)<br>
[LC 1368.Minimum Cost to Make at Least One Valid Path in a Grid](https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/)<br>
[LC 1879.Minimum XOR Sum of Two Arrays](https://leetcode.com/problems/minimum-xor-sum-of-two-arrays/)<br>
[LC 2203.Minimum Weighted Subgraph With the Required Paths](https://leetcode.com/problems/minimum-weighted-subgraph-with-the-required-paths/)<br>
[LC 1514.Path with Maximum Probability](https://leetcode.com/problems/path-with-maximum-probability/)<br>
[LC 1976.Number of Ways to Arrive at Destination](https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/)<br>
[LC 1786.Number of Restricted Paths From First to Last Node](https://leetcode.com/problems/number-of-restricted-paths-from-first-to-last-node/)<br>
[LC 1631.Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/)<br>

```java
// vertex are 0-indexed
//
// n: number of vertexes
// s: the start vertex
// edge: [u, v, w]
int[] dijkstra(int n, int s, int[][] edges) {
  // build graph
  List<int[]>[] graph = new ArrayList[n];
  for (int i = 0; i < n; i++) {
    graph[i] = new ArrayList<>();
  }
  for (int[] edge : edges) {
    int u = edge[0], v = edge[1], w = edge[2];
    graph[u].add(new int[]{v, w});
  }

  // init
  // {node, distance}
  Queue<int[]> queue = new PriorityQueue<>((a, b) -> Integer.compare(a[1], b[1]));
  int[] dist = new int[n];
  int[] prev = new int[n];

  for (int i = 0; i < n; i++) {
    dist[i] = Integer.MAX_VALUE >> 1;
    prev[i] = -1;
  }
  dist[s] = 0;

  queue.offer(new int[] {s, 0});

  while (!queue.isEmpty()) {
    int[] curr = queue.poll();
    int u = curr[0], d = curr[1];

    if (dist[u] != d) continue;

    for (int[] next : graph[u]) {
      int v = next[0], w = next[1];
      
      if (d + w < dist[v]) {
        dist[v] = d + w;
        prev[v] = u;
        queue.offer(new int[]{v, dist[v]});
      }
    }
  }

  return dist;
}
```

### Bellman-Ford

[LQ 609.最短路](https://www.lanqiao.cn/problems/609/learning/)<br>
[LC 743.Network Delay Time](https://leetcode.com/problems/network-delay-time/)<br>

```java
int[] bellmanFord(int n, int s, int[][] edges) {
  int[] dist = new int[n];
  int[] prev = new int[n];

  for (int i = 0; i < n; i++) {
    dist[i] = Integer.MAX_VALUE >> 1;
    prev[i] = -1;
  }
  dist[s] = 0;

  for (int i = 0; i < n - 1; i++) {
    for (int[] edge : edges) {
      int u = edge[0], v = edge[1], w = edge[2];

      // dist[v] = Math.min(dist[v], dist[u] + w);
      if (dist[u] + w < dist[v]) {
        dist[v] = dist[u] + w;
        prev[v] = u;
      }
    }
  }

  // check for negative-weight cycles
  for (int[] edge : edges) {
      int u = edge[0], v = edge[1], w = edge[2];

      // dist[v] = Math.min(dist[v], dist[u] + w);
      if (dist[u] + w < dist[v]) {
        System.out.println("Graph contains a negative-weight cycle.");
      }
  }

  return dist;
}
```

### Floyd-Warshall

```java
class Solution {
  // Floyd
  public int networkDelayTime(int[][] times, int n, int k) {
    int[][] dp = new int[n + 1][n + 1];
    for (int i = 1; i <= n; i++) {
      for (int j = 1; j <= n; j++) {
        if (i == j) dp[i][j] = 0;
        else dp[i][j] = Integer.MAX_VALUE >> 1;
      }
    }

    for (var time : times) {
      int u = time[0], v = time[1], w = time[2];
      dp[u][v] = w;
    }

    for (int l = 1; l <= n; l++) {
      for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
          dp[i][j] = Math.min(dp[i][j], dp[i][l] + dp[l][j]);
        }
      }
    }

    int maxTime = 0;
    for (int j = 1; j <= n; j++) {
      maxTime = Math.max(maxTime, dp[k][j]);
    }

    if (maxTime == Integer.MAX_VALUE >> 1) return -1;

    return maxTime;
  }
}
```

## Toplogical sort

[LC 207.Course Schedule](https://leetcode.com/problems/course-schedule/)<br>
[lC 210.Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)<br>
[LC 310.Minimum Height Trees](https://leetcode.com/problems/minimum-height-trees/)<br>
[LC 2115.Find All Possible Recipes from Given Supplies](https://leetcode.com/problems/find-all-possible-recipes-from-given-supplies/)<br>
[LC 1857.Largest Color Value in a Directed Graph](https://leetcode.com/problems/largest-color-value-in-a-directed-graph/)<br>
[LC 1591.Strange Printer II](https://leetcode.com/problems/strange-printer-ii/)<br>
[LC 2127.Maximum Employees to Be Invited to a Meeting](https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting/)<br>
[LC 2192.All Ancestors of a Node in a Directed Acyclic Graph](https://leetcode.com/problems/all-ancestors-of-a-node-in-a-directed-acyclic-graph/)<br>

```java
void toplogicalSort(int n, int[][] edges) {
  // build graph
  List<Integer>[] graph = new ArrayList[n];
  for (int i = 0; i < n; i++) {
    graph[i] = new ArrayList<>();
  }
  int[] inDegree = new int[n];
  for (var edge : edges) {
    int u = edge[0], v = edge[1];
    // u -> v
    graph[u].add(v);
    inDegree[v]++;
  }

  Queue<Integer> queue = new LinkedList<>();
  for (int i = 0; i < n; i++) {
    // leaf node don't have pre requirements, add it to the queue
    if (inDegree[i] == 0) {
      queue.offer(i);
    }
  }

  while (!queue.isEmpty()) {
    int size = queue.size();
    for (int i = 0; i < size; i++) {
      int u = queue.poll();

      for (var v : graph[u]) {
        inDegree[v]--;

        // if node v don't have pre requirements anymore, add it to the queue
        if (inDegree[v] == 0) {
          queue.offer(v);
        }
      }
    }
  }
}
```

## Union Find

[LC 547.Number of Provinces](https://leetcode.com/problems/number-of-provinces/)<br>
[LC 1584.Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points)<br>
[LC 684.Redundant Connection](https://leetcode.com/problems/redundant-connection/)<br>
[LC 1971.Find if Path Exists in Graph](https://leetcode.com/problems/find-if-path-exists-in-graph/)<br>
[LC 959.Regions Cut By Slashes](https://leetcode.com/problems/regions-cut-by-slashes/)<br>
[LQ 110.合根植物](https://www.lanqiao.cn/problems/110/learning/)<br>
[LC 130.Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)<br>
[LC 952.Largest Component Size by Common Factor](https://leetcode.com/problems/largest-component-size-by-common-factor/)<br>
[LQ 185.修改数组](https://www.lanqiao.cn/problems/185/learning/)<br>
[LC 2092.Find All People With Secret](https://leetcode.com/problems/find-all-people-with-secret/)<br>
[LQ 1505.剪邮票](https://www.lanqiao.cn/problems/1505/learning/)<br>
[LC 2076.Process Restricted Friend Requests](https://leetcode.com/problems/process-restricted-friend-requests/)<br>
[LC 1722.Minimize Hamming Distance After Swap Operations](https://leetcode.com/problems/minimize-hamming-distance-after-swap-operations/)<br>
[LC 2157.Groups of Strings](https://leetcode.com/problems/groups-of-strings/)<br>
[LC 1202.Smallest String With Swaps](https://leetcode.com/problems/smallest-string-with-swaps/)<br>

```java
class UnionFind {
  private int[] parent;
  private int[] rank;

  public UnionFind(int n) {
    this.parent = new int[n];
    this.rank = new int[n];

    for (int i = 0; i < n; i++) {
      parent[i] = i;
      /* rank[i] = 0; */
    }
  }

  public boolean union(int u, int v) {
    int pu = find(u);
    int pv = find(v);

    if (pu == pv) return false;

    if (rank[pu] < rank[pv]) {
      parent[pu] = pv;
    } else {
      parent[pv] = pu;
    }

    if (rank[pu] == rank[pv]) {
      rank[pu]++;
    }

    return true;
  }

  public int find(int u) {
    if (parent[u] == u) return u;

    return parent[u] = find(parent[u]);
  }
}
```

# Tree

## Segment tree

[LC 307.Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/)<br>

```java
class SegmentTree {
  class Node {
    int val, l, r;
    Node left, right;

    Node(int l, int r) {
      this(0, l, r);
    }

    Node(int val, int l, int r) {
      this.val = val;
      this.l = l;
      this.r = r;
    }
  }

  private static final int INVALID_VALUE = 0;
  private Node root;

  public SegmentTree(int[] nums) {
    root = build(nums, 0, nums.length - 1);
  }

  private Node build(int[] nums, int l, int r) {
    if (l == r) {
      return new Node(nums[l], l, r);
    }

    int m = (l + r) / 2;
    Node root = new Node(l, r);
    root.left = build(nums, l, m);
    root.right = build(nums, m + 1, r);
    root.val = combine(root.left.val, root.right.val);

    return root;
  }

  public void update(int index, int val) {
    update(root, index, val);
  }

  private void update(Node root, int index, int val) {
    if (root.l == index && root.r == index) {
      root.val = val;
      return;
    }

    int m = (root.l + root.r) / 2;

    if (index <= m) {
      update(root.left, index, val);
    } else {
      update(root.right, index, val);
    }

    root.val = combine(root.left.val, root.right.val);
  }

  public int query(int left, int right) {
    return query(root, left, right);
  }

  private int query(Node root, int l, int r) {
    if (l > r) return INVALID_VALUE;
    if (root.l == l && root.r == r) {
      return root.val;
    }

    int m = (root.l + root.r) / 2;

    return combine(query(root.left, l, Math.min(m, r)),
                   query(root.right, Math.max(m + 1, l), r));
  }

  private int combine(int left, int right) {
    return left + right;
  }
}
```

```java
class SegmentTree {
  private static final int INVALID_VALUE = Integer.MIN_VALUE;
  private int[] tree;
  private int[] leaf;
  private int n;

  public SegmentTree(int[] arr) {
    this.n = arr.length;
    this.leaf = arr;
    this.tree = new int[(n << 2) + 1];

    build(0, 0, n - 1);
  }

  private void build(int root, int l, int r) {
    if (l == r) {
      tree[root] = leaf[l];
      return;
    }

    int m = (l + r) >> 1;
    int left = left(root), right = right(root);
    build(left, l, m);
    build(right, m + 1, r);

    tree[root] = combine(tree[left], tree[right]);
  }

  public int query(int l, int r) {
    return query(0, 0, n - 1, l, r);
  }

  private int query(int root, int l, int r, int L, int R) {
    if (L > R) return INVALID_VALUE;
    if (l == L && r == R) {
      return tree[root];
    }

    int m = (l + r) >> 1;
    int left = left(root), right = right(root);

    return combine(query(left, l, m, L, Math.min(m, R)),
                   query(right, m + 1, r, Math.max(m + 1, L), R));
  }

  public void update(int i, int val) {
    update(0, 0, n - 1, i, val);
  }

  private void update(int root, int l, int r, int i, int val) {
    if (l == i && r == i) {
      leaf[i] = val;
      tree[root] = val;
      return;
    }

    int m = (l + r) >> 1;
    int left = left(root), right = right(root);

    if (i <= m) {
      update(left, l, m, i, val);
    } else {
      update(right, m + 1, r, i, val);
    }

    tree[root] = combine(tree[left], tree[right]);
  }

  private int combine(int left, int right) {
    return Math.max(left, right);
  }

  private int left(int root) {
    return (root << 1) + 1;
  }

  private int right(int root) {
    return (root << 1) + 2;
  }
}
```

## Fenwick tree

[LC 307.Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/)<br>
[LC 1409.Queries on a Permutation With Key](https://leetcode.com/problems/queries-on-a-permutation-with-key/)<br>
[LC 1395.Count Number of Teams](https://leetcode.com/problems/count-number-of-teams/)<br>
[LC 1375.Number of Times Binary String Is Prefix-Aligned](https://leetcode.com/problems/number-of-times-binary-string-is-prefix-aligned/)<br>
[LC 2179.Count Good Triplets in an Array](https://leetcode.com/problems/count-good-triplets-in-an-array/)<br>

```java
class FenwickTree {
  // 1-indexed
  private int[] tree;

  public FenwickTree(int[] nums) {
    int n = nums.length;
    this.tree = new int[n + 1];

    // deep copy nums[0:] to tree[1:]
    System.arraycopy(nums, 0, tree, 1, n);

    for (int i = 1; i <= n; i++) {
      int j = i + lsb(i);
      if (j <= n) tree[j] += tree[i];
    }
  }

  //
  // 1-indexed
  //
  // delta = newVal - oldVal
  //
  public void update(int i, int delta) {
    while (i < tree.length) {
      tree[i] += delta;
      i += lsb(i);
    }
  }

  //
  // 1-indexed
  //
  // sumOfRange(l, r) == presum(r) - presum(l - 1)
  //
  public int presum(int i) {
    int sum = 0;

    while (i > 0) {
      sum += tree[i];
      i -= lsb(i);
    }

    return sum;
  }

  //
  // lsb stands for least significant bit
  //
  //  x: 10111101000
  //
  // -x: (Negative numbers are stored in the form of Twos' complement in the computer)
  //     01000010111 (Ones' complement)
  //     01000011000 (Twos' complement)
  //
  //
  //     10111101000
  //   & 01000011000
  //   -------------
  //     00000001000
  //
  private int lsb(int i) {
    return i & -i;
  }
}
```

## Prefix Tree (Trie)

[LC 208.Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)<br>
[LC 1032.Stream of Characters](https://leetcode.com/problems/stream-of-characters/)<br>
[LC 421.Maximum XOR of Two Numbers in an Array](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/)<br>
[LC 211.Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)<br>

```java
class Trie {
  class Node {
    boolean isWord;
    Node[] children = new Node[26];
  }

  Node root;

  public Trie() {
    root = new Node();
  }

  public void insert(String word) {
    Node node = root;

    for (int i = 0; i < word.length(); i++) {
      int index = word.charAt(i) - 'a';

      if (node.children[index] == null) {
        node.children[index] = new Node();
      }

      node = node.children[index];
    }

    node.isWord = true;
  }

  public boolean search(String word) {
    Node node = root;

    for (int i = 0; i < word.length(); i++) {
      int index = word.charAt(i) - 'a';

      if (node.children[index] == null) {
        return false;
      }

      node = node.children[index];
    }

    return node.isWord; // !!!!
  }

  public boolean startsWith(String prefix) {
    Node node = root;

    for (int i = 0; i < prefix.length(); i++) {
      int index = prefix.charAt(i) - 'a';

      if (node.children[index] == null) {
        return false;
      }

      node = node.children[index];
    }

    return true; // !!!!
  }
}
```

# DFS

## Backtrack

### Combination

[LC 77.Combinations](https://leetcode.com/problems/combinations)<br>
[LC 39.Combination Sum](https://leetcode.com/problems/combination-sum)<br>
[LC 40.Combination Sum II](https://leetcode.com/problems/combination-sum-ii)<br>
[LC 216.Combination Sum III](https://leetcode.com/problems/combination-sum-iii)<br>
[LC 17.Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)<br>

```java
class Combination {
  public List<List<Integer>> C(int[] nums, int k) {
    List<List<Integer>> res = new ArrayList<>();

    C(nums, k, 0, new ArrayList<>(), res);

    return res;
  }

  private void C(int[] nums, int k, int s, List<Integer> temp, List<List<Integer>> res) {
    if (temp.size() == k) {
      res.add(new ArrayList<>(temp));
      return;
    }

    for (int i = s; i < nums.length; i++) {
      // this line use to remove duplicates,
      // but `nums` needs to be sorted first
      // if (i > s && nums[i] == nums[i - 1]) continue;

      temp.add(nums[i]);

      C(nums, k, i + 1, temp, res);

      temp.remove(temp.size() - 1);
    }
  }
}
```

### Permutation

[LC 46.Permutations](https://leetcode.com/problems/permutations)<br>
[LC 47.Permutations II](https://leetcode.com/problems/permutations-ii)<br>
[LQ 2995.数字游戏](http://lx.lanqiao.cn/problem.page?gpid=T2995)<br>

```java
class Permutation {
  public List<List<Integer>> P(int[] nums, int k) {
    List<List<Integer>> res = new ArrayList<>();

    P(nums, k, new boolean[nums.length], new ArrayList<>(), res);

    return res;
  }

  private void P(int[] nums, int k, boolean[] used, List<Integer> temp, List<List<Integer>> res) {
    if (temp.size() == k) {
      res.add(new ArrayList<>(temp));
      return;
    }

    for (int i = 0; i < nums.length; i++) {
      if (used[i]) continue;

      // this line use to remove duplicates,
      // but `nums` needs to be sorted first
      // if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) continue;

      temp.add(nums[i]);
      used[i] = true;

      P(nums, k, used, temp, res);

      used[i] = false;
      temp.remove(temp.size() - 1);
    }
  }
}
```

# Binary Exponentiation

## Basic

[LC 50.Pow(x, n)](https://leetcode.com/problems/powx-n/)<br>
[ACWING 89.a^b](https://www.acwing.com/problem/content/91/)<br>
[ACWING 90.64 位整数乘法](https://www.acwing.com/problem/content/92/)<br>

```java
long pow(int a, int n) {
  // a %= MOD;
  long res = 1;

  while (n > 0) {
    if ((n & 1) == 1) {
      // res = res * a % MOD;
      res *= a;
    }

    // a = a * a % MOD;
    a *= a;
    n >>= 1;
  }

  return res;
}
```

## Matrix

[LC 509.Fibonacci Number](https://leetcode.com/problems/fibonacci-number/)<br>
[LC 1137.N-th Tribonacci Number](https://leetcode.com/problems/n-th-tribonacci-number/)<br>
[LC 1220.Count Vowels Permutation](https://leetcode.com/problems/count-vowels-permutation/submissions/)<br>

```java
class MatrixFastPow {
  public int[][] pow(int[][] A, int n) {
    int[][] res = new int[A.length][A[0].length];

    // identity matrix
    for (int i = 0; i < A.length; i++) {
      res[i][i] = 1;
    }

    while (n > 0) {
      if ((n & 1) == 1) {
        // res *= A;
        res = dot(res, A);
      }

      // A *= A;
      A = dot(A, A);
      n >>= 1;
    }

    return res;
  }

  private int[][] dot(int[][] A, int[][] B) {
    int[][] C = new int[A.length][B[0].length];

    for (int i = 0; i < C.length; i++) {
      for (int j = 0; j < C[0].length; j++) {
        for (int k = 0; k < B.length; k++) {
          // modular if needed
          // C[i][j] += A[i][k] % MOD * B[k][j] % MOD;
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }

    return C;
  }
}
```

# Stack

## Monotonic Stack

[LC 316.Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/)<br>
[LC 402.Remove K Digits](https://leetcode.com/problems/remove-k-digits/)<br>
[LC 496.Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)<br>
[LC 496.Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)<br>
[LC 739.Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)<br>
[LC 1673.Find the Most Competitive Subsequence](https://leetcode.com/problems/find-the-most-competitive-subsequence/)<br>
[LC 2030.Smallest K-Length Subsequence With Occurrences of a Letter](https://leetcode.com/problems/smallest-k-length-subsequence-with-occurrences-of-a-letter/)<br>
[LC 84.Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)<br>
[LC 85.Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)<br>
[LC 1475.Final Prices With a Special Discount in a Shop](https://leetcode.com/problems/final-prices-with-a-special-discount-in-a-shop/)<br>
[LC 907.Sum of Subarray Minimums](https://leetcode.com/problems/sum-of-subarray-minimums/)<br>
[LC 2104.Sum of Subarray Ranges](https://leetcode.com/problems/sum-of-subarray-ranges/)<br>
[LC 1856.Maximum Subarray Min-Product](https://leetcode.com/problems/maximum-subarray-min-product/)<br>
[LC 849.Maximize Distance to Closest Person](https://leetcode.com/problems/maximize-distance-to-closest-person/)<br>

```java
void monotonicStack(int[] nums) {
  Stack<Integer> stack = new Stack<>();

  for (int num : nums) {
    // We want to keep a monotonously increasing sequence, so when `num` is
    // less than the top element of the stack, that means the balance is
    // broken, and we need pop the top element in order to fix the its
    // monotonicity.
    //
    // HINT: If you want to keep a monotonously decreasing sequence, just
    // change `num < stack.peek()` to `num > stack.peek()`
    while (!stack.isEmpty() && num < stack.peek()) {
      stack.pop();
    }

    // As we can go here, means that `num` is greater or equal than
    // `stack.peek()`, just push it into the stack.
    stack.push(num);
  }
}
```

# Binary Search

## Find a exact value

### Single target

```java
// [l, r]
int bSearch(int l, int r, int x) {
  while (l <= r) {
    int m = l + (r - l) / 2;
    if (f(m) < x) {
      l = m + 1;
    } else if (f(m) > x){
      r = m - 1;
    } else {
      return m;
    }
  }

  return -1;
}
```

### Range target

[LC 1385.Find the Distance Value Between Two Arrays](https://leetcode.com/problems/find-the-distance-value-between-two-arrays/)<br>

This is use to check there is a value between `lower` and `upper`.

```java
// [l, r]
boolean rangeQuery(int l, int r, int lower, int upper) {
  while (l <= r) {
    int m = l + (r - l) / 2;

    if (f(m) < lower) {
      l = m + 1;
    } else if (f(m) > upper) {
      r = m - 1;
    } else {
      return true;
    }
  }

  return false;
}
```

## Approach a value

[LC 2080.Range Frequency Queries](https://leetcode.com/problems/range-frequency-queries/)<br>

### Lower Bound

[LC 35.Search Insert Position](https://leetcode.com/problems/search-insert-position/)<br>
[LC 540.Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/)<br>
[LC 878.Nth Magical Number](https://leetcode.com/problems/nth-magical-number/)<br>
[LC 1201.Ugly Number III](https://leetcode.com/problems/ugly-number-iii/)<br>
[LC 875.Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)<br>
[LC 2141.Maximum Running Time of N Computers](https://leetcode.com/problems/maximum-running-time-of-n-computers/)<br>
[LC 2226.Maximum Candies Allocated to K Children](https://leetcode.com/problems/maximum-candies-allocated-to-k-children/)<br>

```java
//
// [l, r)
//
//    l                     r
// A: 1 2 3 4 5 5 5 5 5 6 7
// x: 5
//
//            r
//            l
// A: 1 2 3 4 5 5 5 5 5 6 7
// x: 5
//
int lowerBound(int l, int r, int x) {
  while (l < r) {
    int m = l + (r - l) / 2;
    if (f(m) < x) {
      l = m + 1;
    } else {
      r = m;
    }
  }

  return l;
}
```

### Upper Bound

[LC 441.Arranging Coins](https://leetcode.com/problems/arranging-coins/)<br>

```java
//
// [l, r)
//
//    l                     r
// A: 1 2 3 4 5 5 5 5 5 6 7
// x: 5
//
//                      r
//                      l
// A: 1 2 3 4 5 5 5 5 5 6 7
// x: 5
//
int upperBound(int l, int r, int x) {
  while (l < r) {
    int m = l + (r - l) / 2;
    if (f(m) <= x) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  return l;
}
```

# Bit Manipulation

## Least significant bits

```java
//
//  x: 10111101000
//
// -x: (Negative numbers are stored in the form of Twos' complement in the computer)
//     01000010111 (Ones' complement)
//     01000011000 (Twos' complement)
//
//
//     10111101000
//   & 01000011000
//   -------------
//     00000001000
//
int lsb(x) {
  return x & -x;
}
```

## Submask Enumeration

[LC 1178.Number of Valid Words for Each Puzzle](https://leetcode.com/problems/number-of-valid-words-for-each-puzzle/)<br>

I took this template and explination from [here](https://leetcode.com/problems/number-of-valid-words-for-each-puzzle/solution/).

```java
for (int subset = bitmask; subset > 0; subset = (subset - 1) & bitmask) {
  // do what you want with the current subset...
}
```

Why does this work? The subsets must be included in range `[0, bitmask]`, and if
we iterate from `bitmask` to `0` one by one, we are guaranteed to visit the `bitmask`
of every subset along the way.

But we can also meet those that are not a subset of `bitmask`. Fortunately,
instead of decrementing `subset` by one at each iteration, we can use `subset = (subset - 1) & bitmask` to ensure that each `subset` only contains characters that
exist in `bitmask`.

Also, we will not miss any subset because `subset - 1` turns at most one `1` into
`0`.

## Gosper's Hack

```java
int limit = 1 << n;
for (int k = n; k >= 1; k--) {
  int state = (1 << k) - 1;

  while (state < limit) {
    // do what you want with the current state...

    int c = state & -state;
    int r = state + c;
    state = (((r ^ state) >> 2) / c) | r;
  }
}
```

# Sliding Window

## Window without repeat

[LC 3.Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)<br>
[LC 1876.Substrings of Size Three with Distinct Characters](https://leetcode.com/problems/substrings-of-size-three-with-distinct-characters/)<br>

```java
class Solution {
  public int lengthOfLongestSubstring(String s) {
    int[] next = new int[128];
    Arrays.fill(next, -1);

    int ans = 0;

    for (int i = 0, j = 0; j < s.length(); j++) {
      int index = s.charAt(j);
      i = Math.max(i, next[index] + 1);

      //
      // window [i, j] has no repeating elements
      //
      ans = Math.max(ans, j - i + 1);

      next[index] = j;
    }

    return ans;
  }
}
```

## Window size have relation with its sum

[LC 485.Max Consecutive Ones](https://leetcode.com/problems/max-consecutive-ones/)<br>
[LC 1004.Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/)<br>
[LC 1343.Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold](https://leetcode.com/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/)<br>
[LC 2134.Minimum Swaps to Group All 1's Together II](https://leetcode.com/problems/minimum-swaps-to-group-all-1s-together-ii/)<br>

```java
class Solution {

  public int findMaxConsecutiveOnes(int[] nums) {
    int ans = 0;
    int save = 0;

    for (int i = 0, j = 0; j < nums.length; j++) {
      save += nums[j];

      // make sure the window is valid
      if (j - i + 1 > save) {
        i = j + 1;
        save = 0;
      }

      // update
      ans = Math.max(ans, j - i + 1);

      // other things
    }

    return ans;
  }
}
```

# Diff Array & Sweep Line

## 1D

[LC 1109.Corporate Flight Bookings](https://leetcode.com/problems/corporate-flight-bookings/)<br>
[LC 732.My Calendar III](https://leetcode.com/problems/my-calendar-iii/)<br>
[LC 56.Merge Intervals](https://leetcode.com/problems/merge-intervals/)<br>
[LQ 1276.小明的彩灯](https://www.lanqiao.cn/problems/1276/learning/)<br>
[LC 1094.Car Pooling](https://leetcode.com/problems/car-pooling/)<br>
[LC 2251.Number of Flowers in Full Bloom](https://leetcode.com/problems/number-of-flowers-in-full-bloom/)<br>

```java
// updates[i] = [x1, x2, delta]
//
//   x1      x2
//    +------+
//
void f(int[][] updates) {
  int[] diff = new int[N];

  for (var update : updates) {
    int x1 = update[0], x2 = update[1];
    int delta = update[2];

    diff[x1    ] += delta;
    diff[x2 + 1] -= delta;
  }

  // convert diff to presum
  for (int i = 0; i < N - 1; i++) {
    diff[i + 1] += diff[i];
  }
}
```

## 2D

[LC 2132.Stamping the Grid](https://leetcode.com/problems/stamping-the-grid/)<br>
[LC 850.Rectangle Area II](https://leetcode.com/problems/rectangle-area-ii/)<br>
[LC 2250.Count Number of Rectangles Containing Each Point](https://leetcode.com/problems/count-number-of-rectangles-containing-each-point/)<br>

```java
// updates[i] = [x1, y1, x2, y2, delta]
//
// (x1, y1)
//    +------+
//    |      |
//    |      |
//    +------+
//        (x2, y2)
//
void f(int[][] updates) {
  int[][] diff = new int[N][N];

  for (var update : updates) {
    int x1 = update[0], y1 = update[1];
    int x2 = update[2], y2 = update[3];
    int delta = update[4];

    diff[x1    ][y1    ] += delta;
    diff[x1    ][y2 + 1] -= delta;
    diff[x2 + 1][y1    ] -= delta;
    diff[x2 + 1][y2 + 1] += delta;
  }

  // convert diff to presum
  for (int i = 0; i < N - 1; i++) {
    for (int j = 0; j < N; j++) {
      diff[i + 1][j] += diff[i][j];
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N - 1; j++) {
      diff[i][j + 1] += diff[i][j];
    }
  }
}
```

## 3D

[LQ 180.三体攻击](https://www.lanqiao.cn/problems/180/learning/)<br>

```java
// updates[i] = [x1, y1, z1, x2, y2, z2, delta]
void f(int[][] updates) {
  int[][][] diff = new int[N][N][N];

  for (var update : updates) {
    int x1 = update[0], y1 = update[1], z1 = update[2];
    int x2 = update[3], y2 = update[4], z2 = update[5];
    int delta = update[6];

    diff[x1    ][y1    ][z1    ] += delta; // 000
    diff[x1    ][y1    ][z2 + 1] -= delta; // 001
    diff[x1    ][y2 + 1][z1    ] -= delta; // 010
    diff[x1    ][y2 + 1][z2 + 1] += delta; // 011
    diff[x2 + 1][y1    ][z1    ] -= delta; // 100
    diff[x2 + 1][y1    ][z2 + 1] += delta; // 101
    diff[x2 + 1][y2 + 1][z1    ] += delta; // 110
    diff[x2 + 1][y2 + 1][z2 + 1] -= delta; // 111

    // Apply `+` when there is even bit ones
    // otherwise apply `-`
  }

  // convert diff to presum
  for (int i = 0; i < N - 1; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        diff[i + 1][j][k] += diff[i][j][k];
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N - 1; j++) {
      for (int k = 0; k < N; k++) {
        diff[i][j + 1][k] += diff[i][j][k];
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N - 1; k++) {
        diff[i][j][k + 1] += diff[i][j][k];
      }
    }
  }
}
```

# Check From The Middle

## Count smaller/greater before/after self (Merge sort or [FenwickTree](#fenwick-tree))

[LC 315.Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)<br>
[LC 2179.Count Good Triplets in an Array](https://leetcode.com/problems/count-good-triplets-in-an-array/)<br>
[LC 493.Reverse Pairs](https://leetcode.com/problems/reverse-pairs/)<br>

```java
int[] nums;
int[] temp;
int[] smallerAfterSelf;

// l: inclusive
// r: exclusive
void sort(int[] sorted, int l, int r) {
  if (l >= r - 1) return;

  int m = l + (r - l) / 2;
  sort(sorted, l, m);
  sort(sorted, m, r);

  for (int i = l; i < m; i++) {
    smallerAfterSelf[i] += lowerBound(sorted, m, r, nums[i]) - m;
  }

  // merge
  int i = l, j = m, k = 0;
  while (i < m && j < r) {
    if (sorted[i] < sorted[j]) {
      temp[k++] = sorted[i++];
    } else {
      temp[k++] = sorted[j++];
    }
  }
  while (i < m) temp[k++] = sorted[i++];
  while (j < r) temp[k++] = sorted[j++];

  for (i = l, j = 0; i < r; i++, j++) {
    sorted[i] = temp[j];
  }
}
```

```java
void f(int[] nums) {
  int[] map = new int[n];

  for (int i = 0; i < n; i++) {
    map[nums[i]] = i;
  }

  int[] prevSmaller = new int[n];

  FenwickTree bit = new FenwickTree(n);
  for (int i = 0; i < n; i++) {
    int a = map[nums[i]];
    bit.update(a, 1);
    prevSmaller[i] = bit.sumOfRange(0, a - 1);
  }
}
```

## Find the first index that smaller/greater before/after self (Monotonic stack)

[Monotonic Stack](#monotonic-stack)

## Min/Max sum of n-th elements before/after self (PriorityQueue)

[LC 2163.Minimum Difference in Sums After Removal of Elements](https://leetcode.com/problems/minimum-difference-in-sums-after-removal-of-elements/)

```java
void f(int[] nums, int n) {
  Queue<Integer> leftQueue = new PriorityQueue<>(Collections.reverseOrder());
  long[] leftMin = new long[nums.length];
  long sum = 0;
  for (int i = 0; i < n << 1; i++) {
    leftQueue.offer(nums[i]);
    sum += nums[i];
    if (leftQueue.size() > n) {
      sum -= leftQueue.poll();
    }
    leftMin[i] = sum;
  }
}
```

# FastScanner

```java
class FastScanner {
  private static final int NULL = '\0';
  private static final int BUFFER_CAPACITY = 1 << 18;
  private InputStream in;
  private byte[] buffer = new byte[BUFFER_CAPACITY];
  private int bufferPtr, bufferSize;

  public FastScanner(InputStream in) {
    this.in = in;
    this.bufferPtr = 0;
    this.bufferSize = 0;
  }

  public String next() throws IOException {
    byte[] res = new byte[1 << 16];
    int len = 0;

    byte b = read();
    while (b != ' ' && b != '\n' && b != NULL) {
      res[len++] = b;
      b = read();
    }

    return new String(res, 0, len);
  }

  public int nextInt() throws IOException {
    int res = 0;

    // skip leading non-printable characters
    byte b = read();
    while (b <= ' ') b = read();

    boolean neg = b == '-';
    if (neg) b = read();

    while (b >= '0' && b <= '9') {
      res = res * 10 + b - '0';
      b = read();
    }

    return neg ? res * -1 : res;
  }

  public long nextLong() throws IOException {
    long res = 0;

    // skip leading non-printable characters
    byte b = read();
    while (b <= ' ') b = read();

    boolean neg = b == '-';
    if (neg) b = read();

    while (b >= '0' && b <= '9') {
      res = res * 10 + b - '0';
      b = read();
    }

    return neg ? res * -1 : res;
  }

  public double nextDouble() throws IOException {
    double res = 0;

    // skip leading non-printable characters
    byte b = read();
    while (b <= ' ') b = read();

    boolean neg = b == '-';
    if (neg) b = read();

    while (b >= '0' && b <= '9') {
      res = res * 10 + b - '0';
      b = read();
    }

    // read decimal part
    if (b == '.') {
      double w = 10;
      b = read();
      while (b >= '0' && b <= '9') {
        res += (b - '0') / (w);
        b = read();
        w *= 10;
      }
    }

    return neg ? res * -1 : res;
  }

  private byte read() throws IOException {
    if (bufferPtr >= bufferSize) {
      fillBuffer();
    }

    return buffer[bufferPtr++];
  }

  private void fillBuffer() throws IOException {
    bufferSize = in.read(buffer, 0, BUFFER_CAPACITY);
    bufferPtr = 0;
    if (bufferSize == -1) {
      buffer[0] = NULL;
    }
  }
}
```

# Parentheses

[LC 20.Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)<br>
[LC 32.Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/)<br>
[LC 856.Score of Parentheses](https://leetcode.com/problems/score-of-parentheses/)<br>
[LC 921.Minimum Add to Make Parentheses Valid](https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/)<br>
[LC 946.Validate Stack Sequences](https://leetcode.com/problems/validate-stack-sequences/)<br>
[LC 1249.Minimum Remove to Make Valid Parentheses](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/)<br>
[LC 1541.Minimum Insertions to Balance a Parentheses String](https://leetcode.com/problems/minimum-insertions-to-balance-a-parentheses-string/)<br>
[LC 1963.Minimum Number of Swaps to Make the String Balanced](https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/)<br>

```java
class Solution {
  public int minAddToMakeValid(String s) {
    int stack = 0;
    int cnt = 0;

    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);

      if (c == '(') {
        stack++; // push '('
      } else {
        if (stack > 0) {
          stack--; // pop '('
        } else {
          cnt++; // need insert a '(' to make s valid
        }
      }
    }

    //
    //           if stack = 3, means we need to insert 3 ')' to make s valid
    //
    //             v
    return cnt + stack;
  }
}
```

[LC 678.Valid Parenthesis String](https://leetcode.com/problems/valid-parenthesis-string/)<br>
[LC 2116.Check if a Parentheses String Can Be Valid](https://leetcode.com/problems/check-if-a-parentheses-string-can-be-valid/)<br>

```java
class Solution {
  public boolean checkValidString(String s) {
    int min = 0, max = 0;
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      
      if (c == '(') {
        min++;
        max++;
      } else if (c == ')') {
        min--;
        max--;
      } else /* if (c == '*') */ {
        min--; // treat * as )
        max++; // treat * as (
      }

      if (max < 0) break;
      if (min < 0) min = 0; // * allow empty
    }

    return min == 0;
  }
}
```
