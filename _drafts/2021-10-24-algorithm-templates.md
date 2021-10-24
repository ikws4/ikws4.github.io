---
title: "Algorithm templates"
date: 2021-10-24 9:08:22 +0800
layout: post
tags: [algorithm, templates]
---


# Dynamic Programming



# Shortest Path
#### dijkstra
vertex are 0-indexed

n: number of vertexes

s: the start vertex 

edge: [u, v, w]
```java
int[] dijkstra(int n, int s, int[][] edges) {
  // Convert edges to adjacent list
  List<int[]>[] graph = new ArrayList[n];
  for (int i = 0; i < n; i++) {
    graph[i] = new ArrayList<>();
  }
  for (int[] edge : edges) {
    int u = edge[0], v = edge[1], w = edge[2];
    graph[u].add(new int[]{v, w});
  }

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
