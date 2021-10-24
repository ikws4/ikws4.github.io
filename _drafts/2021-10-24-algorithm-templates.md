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
