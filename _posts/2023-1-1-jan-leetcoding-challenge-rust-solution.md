---
title: "Jan LeetCoding Challenge Rust Solution"
date: 2023-1-1 10:00:00 +0800
layout: post
toc: true
toc_sticky: true
tags: [leetcode, algorithm]
---

### 290. Word Pattern

```rust
use std::collections::HashMap;

impl Solution {
    pub fn word_pattern(pattern: String, s: String) -> bool {
        if pattern.len() != s.matches(' ').count() + 1 {
            return false;
        }

        let mut char_to_string = HashMap::new();
        let mut string_to_char = HashMap::new();

        for (c, w) in pattern.chars().zip(s.split_whitespace()) {
            if char_to_string.contains_key(&c) && char_to_string[&c] != w {
                return false;
            }

            if string_to_char.contains_key(w) && string_to_char[w] != c {
                return false;
            }

            char_to_string.insert(c, w);
            string_to_char.insert(w, c);
        }

        true
    }
}
```

### 520. Detect Capital

```rust
impl Solution {
    pub fn detect_capital_use(word: String) -> bool {
        let mut all_cap = word.chars().filter(|c| c.is_lowercase()).count() == 0;
        let mut all_lower = word.chars().filter(|c| c.is_uppercase()).count() == 0;
        let mut title = word
            .char_indices()
            .filter(|&(i, c)| i == 0 && c.is_uppercase() || c.is_lowercase())
            .count() == word.len();

        all_cap || all_lower || title
    }
}
```

### 944. Delete Columns to Make Sorted

```rust
impl Solution {
    pub fn min_deletion_size(mut strs: Vec<String>) -> i32 {
        let mut ret = 0;
        for j in 0..strs[0].len() {
            for i in 1..strs.len() {
                if strs[i - 1].as_bytes()[j] > strs[i].as_bytes()[j] {
                    ret += 1;
                    break;
                }
            }
        }

        ret
    }
}
```

### 2244. Minimum Rounds to Complete All Tasks

```rust
impl Solution {
    pub fn minimum_rounds(mut tasks: Vec<i32>) -> i32 {
        let n = tasks.len();
        tasks.sort();
        
        let mut dp = vec![i32::MAX >> 1; n + 5];
        dp[0] = 0;
        dp[2] = 1;
        for i in 3..=n {
            dp[i] = dp[i - 2].min(dp[i - 3]) + 1;
        }

        let mut ret = 0;
        let mut cnt = 1;
        for i in 1..=n {
            if i == n || tasks[i - 1] != tasks[i] {
                let r = dp[cnt];
                if r >= i32::MAX >> 1 {
                    return -1;
                }

                ret += r;
                cnt = 1;
            } else {
                cnt += 1;
            }
        }

        ret
    }
}
```

### 452. Minimum Number of Arrows to Burst Balloons

```rust
impl Solution {
    pub fn find_min_arrow_shots(mut points: Vec<Vec<i32>>) -> i32 {
        points.sort_by(|a, b| a[0].cmp(&b[0]));

        let mut ret = 0;
        let mut right = points[0][1];
        for i in 1..points.len() {
            if points[i][0] > right {
                ret += 1;
                right = points[i][1];
            } else {
                right = right.min(points[i][1]);
            }
        }

        ret + 1
    }
}
```

### 1833. Maximum Ice Cream Bars

```rust
impl Solution {
    pub fn max_ice_cream(mut costs: Vec<i32>, mut coins: i32) -> i32 {
        costs.sort();

        let n = costs.len();
        for i in 0..=n {
            if i == n || coins < costs[i] {
                return i as i32;
            }
            coins -= costs[i];
        }

        -1
    }
}
```

### 134. Gas Station

```rust
impl Solution {
    pub fn can_complete_circuit(gas: Vec<i32>, cost: Vec<i32>) -> i32 {
        let total_gain = gas.iter().zip(cost.iter()).map(|(g, c)| g - c).sum::<i32>();
        if total_gain < 0 {
            return -1;
        }

        let (mut g, mut ret) = (0, 0);
        for i in 0..gas.len() {
            g += gas[i] - cost[i];
            if g < 0 {
                g = 0;
                ret = i + 1;
            }
        }

        ret as i32
    }
}
```

### 149. Max Points on a Line

```rust
use std::collections::HashMap;

impl Solution {
    pub fn max_points(points: Vec<Vec<i32>>) -> i32 {
        let n = points.len();
        let mut ret = 0;
        let mut slope_map = HashMap::new();
        for i in 0..n {
            let mut max = 1;
            for j in 0..n {
                if i != j {
                    let s = slope(&points[i], &points[j]);
                    let v = slope_map.entry(s).or_insert(1);
                    *v += 1;
                    max = max.max(*v);
                }
            }
            slope_map.clear();
            ret = ret.max(max);
        }

        ret
    }
}

fn slope(p0: &[i32], p1: &[i32]) -> String {
    let mut dy = p0[1] - p1[1];
    let mut dx = p0[0] - p1[0];

    if dy == 0 {
        return "0".into();
    }
    if dx == 0 {
        return "inf".into();
    }

    let gcd = gcd(dy, dx);
    dy /= gcd;
    dx /= gcd;

    format!("{}/{}", dy, dx)
}

fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 {
        return a;
    }
    gcd(b, a % b)
}
```

### 144. Binary Tree Preorder Traversal

```rust
use std::rc::Rc;
use std::cell::RefCell;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn preorder_traversal(root: Node) -> Vec<i32> {
        let mut ret = vec![];
        f(&root, &mut ret);
        ret
    }
}

fn f(root: &Node, ret: &mut Vec<i32>) {
    if let Some(root) = root {
        let root = root.borrow();
        ret.push(root.val);
        f(&root.left, ret);
        f(&root.right, ret);
    }
}
```

### 100. Same Tree

```rust
use std::rc::Rc;
use std::cell::RefCell;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn is_same_tree(p: Node, q: Node) -> bool {
        fn f(p: &Node, q: &Node) -> bool {
            if p.is_none() && q.is_none() {
                return true;
            }
            if p.is_none() || q.is_none() {
                return false;
            }
            let p = p.as_ref().unwrap().borrow();
            let q = q.as_ref().unwrap().borrow();
            
            p.val == q.val && f(&p.left, &q.left) && f(&p.right, &q.right)
        }

        f(&p, &q)
    }
}
```