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

### 1443. Minimum Time to Collect All Apples in a Tree

```rust
struct Env {
    graph: Vec<Vec<usize>>,
    has_apple: Vec<bool>
}

impl Solution {
    pub fn min_time(n: i32, edges: Vec<Vec<i32>>, has_apple: Vec<bool>) -> i32 {
        let n = n as usize;
        let mut graph = vec![vec![]; n];
        for edge in &edges {
            let (u, v) = (edge[0] as usize, edge[1] as usize);
            graph[u].push(v);
            graph[v].push(u);
        }
        
        fn f(env: &Env, u: usize, parent: usize) -> (i32, i32) {
            let mut seconds = 0;
            let mut apples = env.has_apple[u as usize] as i32;

            for &v in &env.graph[u] {
                if v == parent { continue; }

                let r = f(env, v, u);

                if r.1 > 0 {
                    seconds += r.0 + 2;
                }

                apples += r.1;
            }

            (seconds, apples)
        }

        let env = Env { graph, has_apple };
        f(&env, 0, n).0
    }
}
```

### 1519. Number of Nodes in the Sub-Tree With the Same Label

```rust
struct Env<'a> {
    graph: Vec<Vec<usize>>,
    labels: &'a [u8],
}

impl Solution {
    pub fn count_sub_trees(n: i32, edges: Vec<Vec<i32>>, labels: String) -> Vec<i32> {
        let n = n as usize;
        let mut graph = vec![vec![]; n];
        for edge in &edges {
            let (u, v) = (edge[0] as usize, edge[1] as usize);
            graph[u].push(v);
            graph[v].push(u);
        }

        let mut ret = vec![0; n];
        fn f(env: &Env, u: usize, p: usize, ret: &mut Vec<i32>) -> Vec<i32> {
            let mut count = vec![0; 26];

            for &v in &env.graph[u] {
                if v != p {
                    let r = f(env, v, u, ret);

                    for i in 0..count.len() {
                        count[i] += r[i];
                    }
                }
            }

            let lable = (env.labels[u] - b'a') as usize;
            count[lable] += 1;
            ret[u] = count[lable];

            count
        }

        let env = Env {
            graph,
            labels: labels.as_bytes(),
        };
        f(&env, 0, n, &mut ret);

        ret
    }
}
```

### 2246. Longest Path With Different Adjacent Characters

```rust
impl Solution {
    pub fn longest_path(parent: Vec<i32>, s: String) -> i32 {
        let n = parent.len();
        let mut graph = vec![vec![]; n];
        for (u, &p) in parent.iter().enumerate().skip(1) {
            graph[p as usize].push(u);
        }

        fn f(graph: &Vec<Vec<usize>>, s: &[u8], u: usize, ret: &mut i32) -> i32 {
            let mut child1 = 0;
            let mut child2 = 0;

            for &v in &graph[u] {
                let r = f(graph, s, v, ret);
                if s[u] != s[v] {
                    if r >= child1 {
                        child2 = child1;
                        child1 = r;
                    } else if r >= child2 {
                        child2 = r;
                    }
                }
            }

            *ret = (*ret).max(child1 + child2 + 1);

            child1 + 1
        }

        let mut ret = 0;
        f(&graph, s.as_bytes(), 0, &mut ret);

        ret
    }
}
```

### 1061. Lexicographically Smallest Equivalent String

```rust
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn with_capacity(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
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
            return true;
        }

        if self.rank[pu] < self.rank[pv] {
            self.parent[pu] = pv;
        } else {
            self.parent[pv] = pu;
            if self.rank[pu] == self.rank[pv] {
                self.rank[pu] += 1;
            }
        }

        false
    }
}

impl Solution {
    pub fn smallest_equivalent_string(s1: String, s2: String, base_str: String) -> String {
        let mut uf = UnionFind::with_capacity(128);
        s1.as_bytes()
            .iter()
            .zip(s2.as_bytes().iter())
            .for_each(|(&a, &b)| {
                uf.union(a as usize, b as usize);
            });

        let mut group = vec![vec![]; 128];
        (b'a'..=b'z').for_each(|c| {
            let root = uf.find(c as usize);
            group[root].push(c);
        });
        group.iter_mut().for_each(|g| g.sort());
        

        let mut ret = String::with_capacity(s1.len());
        base_str.as_bytes().iter().for_each(|&c| {
            let root = uf.find(c as usize);
            ret.push(group[root][0] as char);
        });

        ret
    }
}
```

### 2421. Number of Good Paths

```rust
use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashMap},
};

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn with_capacity(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
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
            return true;
        }

        if self.rank[pu] < self.rank[pv] {
            self.parent[pu] = pv;
        } else {
            self.parent[pv] = pu;
            if self.rank[pu] == self.rank[pv] {
                self.rank[pu] += 1;
            }
        }

        false
    }
}

impl Solution {
    pub fn number_of_good_paths(vals: Vec<i32>, edges: Vec<Vec<i32>>) -> i32 {
        let n = vals.len();
        let mut graph = vec![vec![]; n];
        let mut same_val_nodes = BTreeMap::new();
        for i in 0..n {
            same_val_nodes
                .entry(vals[i])
                .or_insert_with(Vec::new)
                .push(i);
        }
        for edge in edges {
            let u = edge[0] as usize;
            let v = edge[1] as usize;
            if vals[v] <= vals[u] {
                graph[u].push(v);
            }
            if vals[u] <= vals[v] {
                graph[v].push(u);
            }
        }

        let mut ret = n;
        let mut uf = UnionFind::with_capacity(n);
        let mut groups: HashMap<usize, usize> = HashMap::new();
        for (_, nodes) in same_val_nodes {
            for &u in &nodes {
                for &v in &graph[u] {
                    uf.union(u, v);
                }
            }

            groups.clear();
            for u in nodes {
                let e = groups.entry(uf.find(u)).or_default();
                *e += 1;
            }

            for &cnt in groups.values() {
                if cnt > 1 {
                    ret += (cnt * (cnt - 1)) / 2;
                }
            }
        }

        ret as i32
    }
}
```
