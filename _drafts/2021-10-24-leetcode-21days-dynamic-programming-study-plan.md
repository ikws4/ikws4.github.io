---
title: "leetcode 21 days dynamic programming study plan"
date: 2021-10-24 9:08:22 +0800
layout: post
toc: true
toc_sticky: true
tags: [leetcode, dynamic programming, algorithm]
---

# day 1

## Fibonacci Number

斐波那契数列 `f[n] = f[n - 1] + f[n - 2]`

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

在观察这个公式之后，可以发现第 `i` 项的值
只与前两项的值有关，那么可以使用 `a` 和 `b` 两个临时变量滚动更新把空间复杂度从 `O(n)` 优化到 `O(1)` 

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
      // dp[i] = dp[i - 1] + dp[i - 2];
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

```text
[f(n), f(n - 1)] = [f(n - 1), f(n - 2)] dot [1 1]
                                            [1 0]

[f(n), f(n - 1)] = [f(1), f(0)] dot [1 1]^ (n - 1)
                                    [1 0]
```

```java
class Solution {
  public int fib(int n) {
    if (n == 0) return 0;
    
    // A: [
    //      [1, 0]
    //    ]
    int[][] A = new int[][] {
      {1, 0}
    };

    // B: [1 1]
    //    [1 0]
    int[][] B = new int[][] {
      {1, 1},
      {1, 0}
    };

    return dot(A, pow(B, n - 1))[0][0];
  }

  private int[][] pow(int[][] A, int n) {
    int[][] res = new int[A.length][A[0].length];
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
