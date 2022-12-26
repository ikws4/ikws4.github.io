---
title: "Dec LeetCoding Challenge Rust Solution"
date: 2022-12-1 10:00:00 +0800
layout: post
toc: true
toc_sticky: true
tags: [leetcode, algorithm]
---

### 1704. Determine if String Halves Are Alike

```rust
impl Solution {
    pub fn halves_are_alike(s: String) -> bool {
        let s = s.chars().collect::<Vec<char>>();
        let n = s.len();
        let m = n / 2;
        Solution::count(&s, 0, m) == Solution::count(&s, m, n)
    }

    fn count(s: &[char], l: usize, r: usize) -> i32 {
        let mut ret = 0;
        for i in l..r {
            let c = s[i].to_ascii_lowercase();
            if c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' {
                ret += 1;
            }
        }
        ret
    }
}
```

### 1657. Determine if Two Strings Are Close

```rust
impl Solution {
    pub fn close_strings(word1: String, word2: String) -> bool {
        let n = 26;
        let mut cnt1 = vec![0; n];
        let mut cnt2 = vec![0; n];
        for &c in word1.as_bytes() {
            cnt1[(c - b'a') as usize] += 1;
        }
        for &c in word2.as_bytes() {
            cnt2[(c - b'a') as usize] += 1;
        }

        for c in 0..n {
            if cnt1[c] != 0 && cnt2[c] == 0 || cnt2[c] != 0 && cnt1[c] == 0 {
                return false;
            }
        }

        cnt1.sort();
        cnt2.sort();

        for c in 0..n {
            if cnt1[c] != cnt2[c] {
                return false;
            }
        }

        true
    }
}
```

### 451. Sort Characters By Frequency

```rust
impl Solution {
    pub fn frequency_sort(s: String) -> String {
        let n = 128;
        let mut freq = vec![0; n];
        let mut chars = (0..n).collect::<Vec<usize>>();
        for &c in s.as_bytes() {
            freq[c as usize] += 1;
        }
        chars.sort_by(|&a, &b| freq[b].cmp(&freq[a]));

        let mut ret = "".to_string();
        for c in chars {
            let f = freq[c];
            let c = c as u8;
            for _ in 0..f {
                ret.push(c.into());
            }
        }

        ret
    }
}
```

### 2256. Minimum Average Difference

```rust
impl Solution {
    pub fn minimum_average_difference(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        let mut left_sum = vec![0 as i64; n + 1];
        for i in 1..=n {
            left_sum[i] = left_sum[i - 1] + nums[i - 1] as i64;
        }

        let mut ret = 0;
        let mut min_abs = i64::MAX;
        let mut right_sum = 0;
        for i in (0..n).rev() {
            let left_avg = left_sum[i + 1] / (i - 0 + 1) as i64;
            let right_avg = right_sum / ((n - 1) - (i + 1) + 1).max(1) as i64;

            let abs = (left_avg - right_avg).abs();
            if abs <= min_abs {
                ret = i;
                min_abs = abs;
            }

            right_sum += nums[i] as i64;
        }

        ret as i32
    }
}
```

### 876. Middle of the Linked List

```rust
impl Solution {
    pub fn middle_node(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut slow = head.clone();
        let mut fast = head;

        while fast.is_some() && fast.as_ref().unwrap().next.is_some() {
            slow = slow.unwrap().next;
            fast = fast.unwrap().next.unwrap().next;
        }

        slow
    }
}
```

### 328. Odd Even Linked List

```rust
impl Solution {
    pub fn odd_even_list(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut odd = None;
        let mut even = None;
        let mut odd_iter = &mut odd;
        let mut even_iter = &mut even;
        
        let mut is_odd = true;
        while let Some(mut node) = head {
            head = node.next;
            node.next = None;

            if is_odd {
                odd_iter = &mut odd_iter.insert(node).next;
            } else {
                even_iter = &mut even_iter.insert(node).next;
            }
            is_odd ^= true;
        }

        if let Some(node) = even {
            odd_iter.insert(node);
        }
        
        odd
    }
}
```

### 938. Range Sum of BST

```rust
use std::cell::RefCell;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn range_sum_bst(root: Node, low: i32, high: i32) -> i32 {
        fn f(root: &Node, low: i32, high: i32) -> i32 {
            if let Some(root) = root {
                let root = root.borrow();

                if root.val < low {
                    return f(&root.right, low, high);
                }

                if root.val > high {
                    return f(&root.left, low, high);
                }

                return f(&root.left, low, high) + f(&root.right, low, high) + root.val;
            }

            0
        }

        f(&root, low, high)
    }
}
```

### 872. Leaf-Similar Trees

```rust
use std::cell::RefCell;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn leaf_similar(root1: Node, root2: Node) -> bool {
        fn collect_leaf(root: &Node, to: &mut Vec<i32>) {
            if let Some(root) = root {
                let root = root.borrow();
                if root.left.is_none() && root.right.is_none() {
                    to.push(root.val);
                    return;
                }

                collect_leaf(&root.left, to);
                collect_leaf(&root.right, to);
            }
        }

        let mut leaf1 = vec![];
        let mut leaf2 = vec![];
        collect_leaf(&root1, &mut leaf1);
        collect_leaf(&root2, &mut leaf2);

        leaf1.cmp(&leaf2).is_eq()
    }
}
```

### 1026. Maximum Difference Between Node and Ancestor

```rust
use std::cell::RefCell;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn max_ancestor_diff(root: Node) -> i32 {
        fn f(root: &Node, mut min: i32, mut max: i32) -> i32 {
            if let Some(root) = root {
                let root = root.borrow();
                min = min.min(root.val);
                max = max.max(root.val);

                return f(&root.left, min, max).max(f(&root.right, min, max));
            }

            max - min
        }

        let v = root.as_ref().unwrap().borrow().val;
        f(&root, v, v)
    }
}
```

### 1339. Maximum Product of Splitted Binary Tree

```rust
use std::cell::RefCell;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn max_product(root: Node) -> i32 {
        fn sum(root: &Node) -> i32 {
            if let Some(root) = root {
                let root = root.borrow();

                return sum(&root.left) + sum(&root.right) + root.val;
            }

            0
        }

        fn product(root: &Node, total_sum: i32) -> (i32, i64) {
            if let Some(root) = root {
                let root = root.borrow();
                let left = product(&root.left, total_sum);
                let right = product(&root.right, total_sum);

                let s = (left.0 + right.0) + root.val;
                let p = s as i64 * (total_sum - s) as i64;

                return (s, left.1.max(right.1).max(p));
            }

            (0, 0)
        }

        let total_sum = sum(&root);

        (product(&root, total_sum).1 % (1_000_000_007)) as i32
    }
}
```

### 124. Binary Tree Maximum Path Sum

```rust
use std::cell::RefCell;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn max_path_sum(root: Node) -> i32 {
        fn f(root: &Node, ret: &mut i32) -> i32 {
            if let Some(root) = root {
                let root = root.borrow();

                let l = f(&root.left, ret);
                let r = f(&root.right, ret);

                *ret = (*ret)
                    .max(root.val)
                    .max(l + root.val)
                    .max(r + root.val)
                    .max(l + r + root.val);

                return root.val.max(root.val + l).max(root.val + r);
            }

            0
        }

        let mut ret = i32::MIN >> 1;
        f(&root, &mut ret);

        ret
    }
}
```

### 70. Climbing Stairs

```rust
impl Solution {
    pub fn climb_stairs(n: i32) -> i32 {
        let n = n as usize;
        let mut dp = vec![1; n + 1];
        
        for i in 2..=n {
            dp[i] = dp[i - 1] + dp[i - 2];
        }

        dp[n]
    }
}
```

### 931. Minimum Falling Path Sum

```rust
struct Env {
    matrix: Vec<Vec<i32>>,
    memo: Vec<Vec<i32>>,
}

impl Solution {
    pub fn min_falling_path_sum(matrix: Vec<Vec<i32>>) -> i32 {
        fn f(env: &mut Env, i: i32, j: i32) -> i32 {
            if i >= env.matrix.len() as i32 {
                return 0;
            }
            if j < 0 || j >= env.matrix.len() as i32 {
                return i32::MAX >> 1;
            }
            let (iu, ju) = (i as usize, j as usize);
            if env.memo[iu][ju] != -1 {
                return env.memo[iu][ju];
            }

            env.memo[iu][ju] = f(env, i + 1, j - 1).min(f(env, i + 1, j)).min(f(env, i + 1, j + 1)) + env.matrix[iu][ju];
            env.memo[iu][ju]
        }

        let (m, n) = (matrix.len(), matrix[0].len());
        let mut ret = i32::MAX >> 1;
        let mut env = Env {
            matrix,
            memo: vec![vec![-1; n]; m]
        };
        
        for j in 0..n {
            ret = ret.min(f(&mut env, 0, j as i32));
        }

        ret
    }
}
```

### 198. House Robber

```rust
impl Solution {
    pub fn rob(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        let mut dp = vec![0; n];
        dp[0] = nums[0];
        if n > 1 {
            dp[1] = nums[0].max(nums[1]);
        }

        for i in 2..n {
            dp[i] = dp[i - 1].max(dp[i - 2] + nums[i]);
        }

        dp[n - 1]
    }
}
```

### 1143. Longest Common Subsequence

```rust
impl Solution {
    pub fn longest_common_subsequence(text1: String, text2: String) -> i32 {
        let (s1, s2) = (text1.as_bytes(), text2.as_bytes());
        let (m, n) = (s1.len(), s2.len());
        let mut dp = vec![vec![0; n + 1]; m + 1];

        for i in 1..=m {
            for j in 1..=n {
                if s1[i - 1] == s2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }

        dp[m][n]
    }
}
```

### 232. Implement Queue using Stacks

```rust
struct MyQueue {
    stack1: Vec<i32>,
    stack2: Vec<i32>
}

impl MyQueue {

    fn new() -> Self {
        Self {
            stack1: vec![],
            stack2: vec![],
        }
    }
    
    fn push(&mut self, x: i32) {
        self.stack1.push(x);
    }
    
    fn pop(&mut self) -> i32 {
        if self.peek() != -1 {
            return self.stack2.pop().unwrap();
        }

        -1
    }
    
    fn peek(&mut self) -> i32 {
        if self.stack2.is_empty() {
            while let Some(v) = self.stack1.pop() {
                self.stack2.push(v);
            }
        }

        if self.stack2.is_empty() {
            return -1
        }

        self.stack2[self.stack2.len() - 1]
    }
    
    fn empty(&self) -> bool {
        self.stack1.is_empty() && self.stack2.is_empty()
    }
}
```

### 150. Evaluate Reverse Polish Notation

```rust
use std::ops::{Add, Div, Mul, Sub};

impl Solution {
    pub fn eval_rpn(tokens: Vec<String>) -> i32 {
        let mut stack = vec![];

        fn eval(stack: &mut Vec<i32>, op: impl Fn(i32, i32) -> i32) {
            let b = stack.pop().unwrap();
            let a = stack.pop().unwrap();
            stack.push(op(a, b));
        }

        for token in tokens {
            match token.as_str() {
                "+" => eval(&mut stack, Add::add),
                "-" => eval(&mut stack, Sub::sub),
                "*" => eval(&mut stack, Mul::mul),
                "/" => eval(&mut stack, Div::div),
                _ => stack.push(token.parse().unwrap())
            }
        }

        stack.pop().unwrap()
    }
}
```

### 739. Daily Temperatures

```rust
impl Solution {
    pub fn daily_temperatures(temperatures: Vec<i32>) -> Vec<i32> {
        let n = temperatures.len();
        let mut ret = vec![0; n];
        let mut stack = vec![];

        for (i, &t) in temperatures.iter().enumerate() {
            while let Some(&j) = stack.last() {
                if t > temperatures[j] {
                    stack.pop();
                    ret[j] = (i - j) as i32;
                } else {
                    break;
                }
            }
            stack.push(i);
        }

        ret
    }
}
```

### 1971. Find if Path Exists in Graph

```rust
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect::<Vec<usize>>(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, u: usize) -> usize {
        if self.parent[u] == u {
            return u;
        }
        self.parent[u] = self.find(self.parent[u]);
        self.parent[u]
    }

    fn union(&mut self, u: usize, v: usize) -> bool {
        let pu = self.find(u);
        let pv = self.find(v);
        if pu == pv {
            return false;
        }

        if self.rank[pu] < self.rank[pv] {
            self.parent[pu] = pv;
        } else {
            self.parent[pv] = pu;
            if self.rank[pu] == self.rank[pv] {
                self.rank[pu] += 1;
            }
        }

        true
    }
}

impl Solution {
    pub fn valid_path(n: i32, edges: Vec<Vec<i32>>, source: i32, destination: i32) -> bool {
        let mut uf = UnionFind::new(n as usize);

        for edge in edges {
            let (u, v) = (edge[0] as usize, edge[1] as usize);
            uf.union(u, v);
        }

        uf.find(source as usize) == uf.find(destination as usize)
    }
}
```

### 841. Keys and Rooms

```rust
use std::collections::HashSet;

impl Solution {
    pub fn can_visit_all_rooms(rooms: Vec<Vec<i32>>) -> bool {
        fn f(u: i32, rooms: &Vec<Vec<i32>>, visited: &mut HashSet<i32>) {
            if visited.contains(&u) {
                return;
            }

            visited.insert(u);
            for &v in &rooms[u as usize] {
                f(v, rooms, visited);
            }
        }

        let mut visited = HashSet::new();
        f(0, &rooms, &mut visited);
        visited.len() == rooms.len()
    }
}
```

### 886. Possible Bipartition

```rust
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect::<Vec<usize>>(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, u: usize) -> usize {
        if self.parent[u] == u {
            return u;
        }
        self.parent[u] = self.find(self.parent[u]);
        self.parent[u]
    }

    fn union(&mut self, u: usize, v: usize) -> bool {
        let pu = self.find(u);
        let pv = self.find(v);
        if pu == pv {
            return false;
        }

        if self.rank[pu] < self.rank[pv] {
            self.parent[pu] = pv;
        } else {
            self.parent[pv] = pu;
            if self.rank[pu] == self.rank[pv] {
                self.rank[pu] += 1;
            }
        }

        true
    }
}

impl Solution {
    pub fn possible_bipartition(n: i32, dislikes: Vec<Vec<i32>>) -> bool {
        let n = n as usize;
        let mut graph = vec![vec![]; n];
        for d in dislikes {
            let (u, v) = (d[0] as usize - 1, d[1] as usize - 1);
            graph[u].push(v);
            graph[v].push(u);
        }

        let mut uf = UnionFind::new(n);
        for i in 0..n {
            let next = &graph[i];
            if next.is_empty() {
                continue;
            }

            let u = next[0];
            for &v in next {
                if uf.find(i) == uf.find(v) {
                    return false;
                }
                uf.union(u, v);
            }
        }

        true
    }
}
```

### 834. Sum of Distances in Tree

```rust
struct Env {
    count: Vec<i32>,
    ret: Vec<i32>,
}

impl Solution {
    pub fn sum_of_distances_in_tree(n: i32, edges: Vec<Vec<i32>>) -> Vec<i32> {
        let n = n as usize;
        let mut graph = vec![vec![]; n];
        for edge in edges {
            let (u, v) = (edge[0] as usize, edge[1] as usize);
            graph[u].push(v);
            graph[v].push(u);
        }
        let mut env = Env {
            count: vec![1; n],
            ret: vec![0; n]
        };
        
        fn f(env: &mut Env, graph: &Vec<Vec<usize>>, root: usize, parent: usize) {
            for &child in &graph[root] {
                if child == parent {
                    continue;
                }

                f(env, graph, child, root);
                env.count[root] += env.count[child];
                env.ret[root] += env.ret[child] + env.count[child];
            }
        }

        fn g(env: &mut Env, graph: &Vec<Vec<usize>>, root: usize, parent: usize) {
            for &child in &graph[root] {
                if child == parent {
                    continue;
                }

                env.ret[child] = env.ret[root] + (env.count.len() as i32 - env.count[child]) - env.count[child];
                g(env, graph, child, root);
            }
        }

        f(&mut env, &graph, 0, n + 1);
        g(&mut env, &graph, 0, n + 1);

        env.ret
    }
}
```

### 309. Best Time to Buy and Sell Stock with Cooldown

```rust
impl Solution {
    pub fn max_profit(prices: Vec<i32>) -> i32 {
        let mut n = prices.len();
        let mut dp = vec![vec![0; 3]; n];
        dp[0][0] = -prices[0];

        for i in 1..n {
            dp[i][0] = dp[i - 1][0].max(dp[i - 1][2] - prices[i]);
            dp[i][1] = dp[i - 1][1].max(dp[i - 1][0] + prices[i]);
            dp[i][2] = dp[i - 1][2].max(dp[i - 1][1]);
        }

        dp[n - 1][1].max(dp[n - 1][2])
    }
}
```

### 790. Domino and Tromino Tiling

```rust
impl Solution {
    pub fn num_tilings(n: i32) -> i32 {
        let n = n as usize;
        let mut dp: Vec<Vec<i64>> = vec![vec![0; 3]; n + 1];
        let m = 1_000_000_007;
        dp[0][0] = 1;
        dp[1][0] = 1;

        for i in 2..=n {
            dp[i][0] = (dp[i - 1][0] + dp[i - 2][0] + dp[i - 1][1] + dp[i - 1][2]) % m;
            dp[i][1] = (dp[i - 2][0] + dp[i - 1][2]) % m;
            dp[i][2] = (dp[i - 2][0] + dp[i - 1][1]) % m;
        }

        dp[n][0] as i32
    }
}
```

### 2389. Longest Subsequence With Limited Sum

```rust
impl Solution {
    pub fn answer_queries(mut nums: Vec<i32>, queries: Vec<i32>) -> Vec<i32> {
        nums.sort();
        let (n, m) = (nums.len(), queries.len());
        let mut presum = vec![0; n + 1];
        for i in 1..=n {
            presum[i] = presum[i - 1] + nums[i - 1];
        }

        fn bsearch(arr: &Vec<i32>, a: i32) -> usize {
            let (mut l, mut r) = (0, arr.len());
            while l < r {
                let m = l + (r - l) / 2;
                if arr[m] <= a {
                    l = m + 1;
                } else {
                    r = m;
                }
            }
            l
        }

        let mut ret = vec![0; m];
        for i in 0..m {
            ret[i] = (bsearch(&presum, queries[i]) - 1) as i32;
        }

        ret
    }
}
```

### 55. Jump Game

```rust
impl Solution {
    pub fn can_jump(nums: Vec<i32>) -> bool {
        let n = nums.len();
        let mut dp = vec![false; n];
        dp[0] = true;
        
        for i in 1..n {
            for j in (0..i).rev() {
                if nums[j] as usize >= i - j && dp[j] {
                    dp[i] = true;
                    break;
                }
            }
        }

        dp[n - 1]
    }
}
```
