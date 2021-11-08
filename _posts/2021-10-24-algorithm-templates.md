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

```java
void sort(int[] arr, int l, int r) {
  if (l > r) return;

  int i = l, j = r;
  int t = l;
  int pivot = arr[l + (r - l) / 2] // or arr[l + random.nextInt(r - l)];

  while (t < j) {
    if (arr[t] < pivot) {
      swap(arr, t++, i++);
    } else if (arr[t] > pivot) {
      swap(arr, t, j--);
    } else {
      t++;
    }
  }

  sort(arr, l, i - 1);
  sort(arr, j + 1, r);
}
```

# Dynamic Programming

## knapsack

### 0-1 knapsack

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

[ACWING 3.完全背包问题](https://www.acwing.com/problem/content/3/)<br>
[LC 322.Coin Change](https://leetcode.com/problems/coin-change/)<br>
[LC 518.Coin Change 2](https://leetcode.com/problems/coin-change-2/)<br>

![unbounded_knapsack](/assets/img/algorithm_templates/unbounded_knapsack.png)

```java
// dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - W[i]] + V[i])
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

# Graph

## Shortest Path

### Dijkstra

```java
// vertex are 0-indexed
//
// n: number of vertexes
// s: the start vertex
// edge: [u, v, w]
int[] dijkstra(int n, int s, int[][] edges) {
  List<int[]>[] graph = toAdjacentList(edges);
  // Initial
  Queue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0]);
  int[] dist = new int[n];
  int[] prev = new int[n];

  for (int i = 0; i < n; i++) {
    dist[i] = Integer.MAX_VALUE;
    prev[i] = -1;
  }
  dist[s] = 0;

  // {vertex, dist}
  pq.offer(new int[]{s, 0});

  while (!pq.isEmpty()) {
    int[] curr = pq.poll();
    int u = curr[0], d = curr[1];

    if (dist[u] != d) continue;

    for (int[] nei : graph[u]) {
      int v = nei[0], w = nei[1];
      if (d + w < dist[v]) {
        dist[v] = d + w;
        prev[v] = u;
        pq.offer(new int[]{v, dist[v]});
      }
    }
  }

  return dist;
}
```

### Bellman-Ford

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

### Help functions

```java
List<int[]>[] toAdjacentList(int[][] edges) {
  List<int[]>[] graph = new ArrayList[n];

  for (int i = 0; i < n; i++) {
    graph[i] = new ArrayList<>();
  }

  for (int[] edge : edges) {
    int u = edge[0], v = edge[1], w = edge[2];
    graph[u].add(new int[]{v, w});
  }

  return graph;
}
```

## Union Find

[LC 547.Number of Provinces](https://leetcode.com/problems/number-of-provinces/)<br>
[LC 1584.Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points)<br>
[LC 684.Redundant Connection](https://leetcode.com/problems/redundant-connection/)<br>
[LC 1971.Find if Path Exists in Graph](https://leetcode.com/problems/find-if-path-exists-in-graph/)<br>
[LC 959.Regions Cut By Slashes](https://leetcode.com/problems/regions-cut-by-slashes/)<br>
[LQ 110.合根植物](https://www.lanqiao.cn/problems/110/learning/)<br>

```java
class UnionFind {
  private int[] parent;
  private int[] rank;

  public UnionFind(int n) {
    this.parent = new int[n];
    this.rank = new int[n];

    for (int i = 0; i < n; i++) {
      parent[i] = i;
      rank[i] = 0;
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

  private Node root;

  public SegmentTree(int[] nums) {
    root = build(nums, 0, nums.length - 1);
  }

  private Node build(int[] nums, int l, int r) {
    if (l == r)
      return new Node(nums[l], l, r);

    int m = l + (r - l) / 2;
    Node root = new Node(l, r);
    root.left = build(nums, l, m);
    root.right = build(nums, m + 1, r);
    root.val = root.left.val + root.right.val;

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

    int m = root.l + (root.r - root.l) / 2;

    if (index <= m) {
      update(root.left, index, val);
    } else {
      update(root.right, index, val);
    }

    root.val = root.left.val + root.right.val;
  }

  public int query(int left, int right) {
    return query(root, left, right);
  }

  private int query(Node root, int l, int r) {
    if (root.l == l && root.r == r)
      return root.val;

    int m = root.l + (root.r - root.l) / 2;

    if (r <= m) {
      return query(root.left, l, r);
    } else if (l > m) {
      return query(root.right, l, r);
    } else {
      return query(root.left, l, m) + query(root.right, m + 1, r);
    }
  }
}
```

## Fenwick tree

[LC 307.Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/)<br>

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

# DFS

## Backtrack

### Combination

[LC 77.Combinations](https://leetcode.com/problems/combinations)<br>
[LC 39.Combination Sum](https://leetcode.com/problems/combination-sum)<br>
[LC 40.Combination Sum II](https://leetcode.com/problems/combination-sum-ii)<br>
[LC 216.Combination Sum III](https://leetcode.com/problems/combination-sum-iii)<br>

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
      // if (i > 0 && nums[i] == nums[i - 1]) continue;

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
[ACWING 90.64位整数乘法](https://www.acwing.com/problem/content/92/)<br>

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
