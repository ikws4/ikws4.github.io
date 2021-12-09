---
title: "lanqiao-solutions"
date: 2021-12-9 11:35:32 +0800
layout: post
toc: true
toc_sticky: true
tags: [algorithm]
---

# BFS

## 跳马

[problem Link](http://lx.lanqiao.cn/problem.page?gpid=T2987)

```java
import java.util.*;

public class Main {
  private final Scanner in = new Scanner(System.in);
  private final int[][] dirs = { {-2, 1}, {-1, 2}, {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1} };
  private final int n = 8;

  public void main() {
    int[][] grid = new int[n][n];
    int a = in.nextInt() - 1;
    int b = in.nextInt() - 1;
    int c = in.nextInt() - 1;
    int d = in.nextInt() - 1;

    Queue<Integer> queue = new LinkedList<>();
    queue.offer(a);
    queue.offer(b);
    grid[a][b] = 1;

    int step = 0;
    while (!queue.isEmpty()) {
      int size = queue.size();
      for (int k = 0; k < size / 2; k++) {
        int i = queue.poll();
        int j = queue.poll();

        if (i == c && j == d) {
          System.out.println(step);
          return;
        }

        for (int[] dir : dirs) {
          int _i = i + dir[0];
          int _j = j + dir[1];

          if (_i < 0 || _i >= n ||
              _j < 0 || _j >= n ||
              grid[_i][_j] == 1) continue;

          grid[_i][_j] = 1;
          queue.offer(_i);
          queue.offer(_j);
        }
      }
      step++;
    }

    System.out.println(-1);
  }

  public static void main(String[] args) throws IOException {
    (new Main()).main();
  }
}
```

# DFS

## 无聊的逗

[problem Link](http://lx.lanqiao.cn/problem.page?gpid=T2992)

```java
import java.util.*;

public class Main {
  private final Scanner in = new Scanner(System.in);
  private int ans = 0;
  private int total = 0;

  public void main() {
    int n = in.nextInt();
    int[] arr = new int[n];
    for (int i = 0; i < n; i++) {
      arr[i] = in.nextInt();
      total += arr[i];
    }
    Arrays.sort(arr);

    dfs(arr, 0, 0, new boolean[n]);

    System.out.println(ans);
  }

  private void dfs(int[] arr, int i, int a, boolean[] used) {
    if (i >= arr.length) return;
    if (isValid(arr, a, used)) ans = Math.max(ans, a);

    used[i] = true;
    dfs(arr, i + 1, a + arr[i], used);
    used[i] = false;

    dfs(arr, i + 1, a, used);
  }

  private boolean isValid(int[] arr, int a, boolean[] used) {
    int diff = total - 2 * a;
    int sum = 0;
    for (int i = 0; i < used.length; i++) {
      if (used[i]) continue;

      if (diff == sum || arr[i] == diff) return true;
      sum += arr[i];
    }

    return false;
  }

  public static void main(String[] args) throws IOException {
    (new Main()).main();
  }
}
```
