---
title: "Oct LeetCoding Challenge Rust Solution"
date: 2022-10-1 10:00:00 +0800
layout: post
toc: true
toc_sticky: true
tags: [leetcode, algorithm]
---

### 91. Decode Ways

```rust
impl Solution {
    pub fn num_decodings(s: String) -> i32 {
        Solution::dp(s.as_bytes(), 0, &mut vec![0; s.len()])
    }

    fn dp(s: &[u8], i: usize, memo: &mut [i32]) -> i32 {
        if i >= s.len() {
            return 1;
        }
        if memo[i] != 0 {
            return memo[i];
        }

        let mut ret = 0;
        let mut num = 0;
        for j in i..s.len() {
            num = num * 10 + (s[j] - b'0') as i32;
            if num == 0 || num > 26 {
                break;
            }
            ret += Solution::dp(s, j + 1, memo);
        }
        memo[i] = ret;

        ret
    }
}
```

### 1155. Number of Dice Rolls With Target Sum

```rust
struct Env {
    k: i32,
    memo: Vec<Vec<i32>>,
}

impl Solution {
    pub fn num_rolls_to_target(n: i32, k: i32, target: i32) -> i32 {
        fn dp(env: &mut Env, n: i32, t: i32) -> i32 {
            if n == 0 && t == 0 {
                return 1;
            }
            if n < 0 || t < 0 {
                return 0;
            }

            let un = n as usize;
            let ut = t as usize;
            if env.memo[un][ut] != -1 {
                return env.memo[un][ut];
            }

            let mut ret = 0;
            for i in 1..=env.k {
                ret += dp(env, n - 1, t - i);
                ret %= 1_000_000_007;
            }

            env.memo[un][ut] = ret;
            ret
        }

        let memo = vec![vec![-1; (target + 1) as usize]; (n + 1) as usize];
        dp(&mut Env { k, memo }, n, target)
    }
}
```

### 1578. Minimum Time to Make Rope Colorful

```rust
impl Solution {
    pub fn min_cost(colors: String, needed_time: Vec<i32>) -> i32 {
        let colors = colors.as_bytes();
        let n = needed_time.len();

        let mut sum = 0;
        let mut max = 0;
        let mut i = 0;
        let mut j = 0;

        let mut ret = 0;
        while j <= n {
            if j == n || colors[i] != colors[j] {
                ret += sum - max;
                i = j;
                sum = 0;
                max = 0;
            }

            if j < n {
                sum += needed_time[j];
                max = max.max(needed_time[j]);
            }

            j += 1;
        }

        ret
    }
}
```

### 112. Path Sum

```rust
use std::cell::RefCell;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn has_path_sum(root: Node, target_sum: i32) -> bool {
        fn dfs(root: &Node, n: i32) -> bool {
            if let Some(root) = root {
                let root = root.borrow();

                if root.left.is_none() && root.right.is_none() {
                    return root.val == n;
                }

                return dfs(&root.left, n - root.val) || dfs(&root.right, n - root.val);
            }

            false
        }

        dfs(&root, target_sum)
    }
}
```

### 623. Add One Row to Tree

```rust
use std::cell::RefCell;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn add_one_row(root: Node, val: i32, depth: i32) -> Node {
        fn internal(root: Node, val: i32, depth: i32, is_left: bool) -> Node {
            if depth == 1 {
                let mut node = TreeNode::new(val);
                if is_left {
                    node.left = root;
                } else {
                    node.right = root;
                }
                return Some(Rc::new(RefCell::new(node)));
            }

            if let Some(root) = root.as_ref() {
                let mut root = root.borrow_mut();
                root.left = internal(root.left.take(), val, depth - 1, true);
                root.right = internal(root.right.take(), val, depth - 1, false);
            }

            root
        }

        internal(root, val, depth, true)
    }
}
```