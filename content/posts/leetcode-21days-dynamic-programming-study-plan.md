+++
title = "leetcode 21 days dynamic programming study plan"
date = "2021-10-24T9:08:22+08:00"
author = "ikws4"
cover = ""
tags = ["leetcode", "dynamic programming", "algorithm"]
mathjax = true
Toc = true
+++

<!--more-->

# day 1

## Fibonacci Number

### 基本版本

```java
class Solution {
  public int fib(int n) {
    if (n == 0) return n;

    int[] dp = new int[n + 1];
    dp[0] = 0;
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
      dp[i] = dp[i - 1] + dp[i - 2];
    }

    return dp[n];
  }
}
```

### 空间优化版本

```java
class Solution {
  public int fib(int n) {
    if (n == 0) return 0;

    // int[] dp = new int[n + 1];
    int a = 0; // dp[0] = 0;
    int b = 1; // dp[1] = 1;

    for (int i = 2; i <= n; i++) {
      //
      // dp[i]     = dp[i - 1] + dp[i - 2];
      //               b         a
      //
      // dp[i + 1] = dp[i] + dp[i - 1];
      //               b         a
      //
      // 观察可以发现, 第 i + 1 项:
      //   b = dp[i]     = b + a
      //   a = dp[i - 1] = b
      //

      int b_ = b; // 保存上一轮 b 的值, 用于更新 a
      b = b + a;
      a = b_;
    }

    return b;
  }
}
```

### 记忆化递归版本

```java
class Solution {
  private int[] memo;

  public int fib(int n) {
    memo = new int[n + 1];
    return f(n);
  }

  private int f(int n) {
    if (n <= 1) return n;
    if (memo[n] != 0) return memo[n];

    return memo[n] = f(n - 1) + f(n - 2);
  }
}
```

### 矩阵快速幂

```java
class Solution {
  public int fib(int n) {
    if (n == 0) return 0;

    int[][] A = new int[][] {
      {1, 0}
    };

    int[][] B = new int[][] {
      {1, 1},
      {1, 0}
    };

    return dot(A, pow(B, n - 1))[0][0];
  }

  // 快速幂运算 O(log(n))
  private int[][] pow(int[][] A, int n) {
    int[][] res = new int[A.length][A[0].length];

    // 把 res 初始化成单位矩阵(E)
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

  // 计算两个矩阵的点积
  private int[][] dot(int[][] A, int[][] B) {
    int[][] C = new int[A.length][B[0].length];

    for (int i = 0; i < C.length; i++) {
      for (int j = 0; j < C[0].length; j++) {
        for (int k = 0; k < B.length; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }

    return C;
  }
}
```

## N-th Tribonacci Number

### 矩阵快速幂

```java
class Solution {
  public int tribonacci(int n) {
    if (n == 0) return 0;

    int[][] A = new int[][] {
      {1, 1, 0}
    };

    int[][] B = new int[][] {
      {1, 1, 0},
      {1, 0, 1},
      {1, 0, 0},
    };

    return dot(A, pow(B, n - 2))[0][0];
  }

  private int[][] pow(int[][] A, int n) {
    int[][] res = new int[A.length][A[0].length];

    // 把 res 初始化成单位矩阵(E)
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
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }

    return C;
  }
}
```

# day 2

## Climbing Stairs

### 空间优化版本

```java
class Solution {
  public int climbStairs(int n) {
    if (n == 1) return 1;

    // int[] dp = new int[n + 1];
    int a = 1;
    int b = 2;

    for (int i = 3; i <= n; i++) {
      // dp[i]     = dp[i - 1] + dp[i - 2];
      //   b           b         a
      //
      // dp[i + 1] = dp[i] + dp[i - 1];
      //   b           b         a
      //

      int b_ = b;
      b = b + a;
      a = b_;
    }

    return b;
  }
}
```

## Min Cost Climbing Stairs

### 空间优化版本

```java
class Solution {
  public int minCostClimbingStairs(int[] cost) {
    int n = cost.length;

    // int[] dp = new int[n]; // dp[i]: 从 i 层台阶跳到下一(二)层台阶的最小 cost
    int a = cost[0]; // dp[0] = cost[0];
    int b = cost[1]; // dp[1] = cost[1];

    for (int i = 2; i < n; i++) {
      // dp[i]     = Math.min(dp[i - 1], dp[i - 2]) + cost[i];
      //   b                   b           a
      //
      // dp[i + 1] = Math.min(dp[i], dp[i - 1]) + cost[i + 1];
      //   b                   b           a
      //


      int b_ = b;
      b = Math.min(b, a) + cost[i];
      a = b_;
    }

    // 可以在 n - 2 (a) 层两步上去或
    //       n - 1 (b) 层一步上去
    return Math.min(a, b);
  }
}
```

### 记忆化递归版本

```java
class Solution {
  private int[] cost;
  private int[] memo;

  public int minCostClimbingStairs(int[] cost) {
    int n = cost.length;
    this.cost = cost;
    this.memo = new int[n + 1];

    return dp(n);
  }

  private int dp(int n) {
    if(n < 2) return 0; // 如果楼梯小于2，没办法爬上去
    if(memo[n] != 0) return memo[n];

    // 我们需要从 n - 1 或者 n - 2 层爬上来
    //   1. 从 n - 1 层爬上来的 cost 是 cost[n - 1]
    //   2. 从 n - 2 层爬上来的 cost 是 cost[n - 2]
    //
    // 因为我们想要付出 minimum cost 所以在 1 和 2 中去一个最小值作为 n 的结果
    return memo[n] = Math.min(dp(n - 1) + cost[n - 1],
                              dp(n - 2) + cost[n - 2]);
  }
}
```
