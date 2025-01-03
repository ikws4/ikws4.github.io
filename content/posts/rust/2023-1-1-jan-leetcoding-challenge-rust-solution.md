+++
title = "Jan LeetCoding Challenge Rust Solution"
date = "2023-01-01T08:53:56+08:00"
cover = ""
tags = ["leetcode", "algorithm"]
Toc = true
+++

<!--more-->

# 290. Word Pattern

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

# 520. Detect Capital

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

# 944. Delete Columns to Make Sorted

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

# 2244. Minimum Rounds to Complete All Tasks

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

# 452. Minimum Number of Arrows to Burst Balloons

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

# 1833. Maximum Ice Cream Bars

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

# 134. Gas Station

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

# 149. Max Points on a Line

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

# 144. Binary Tree Preorder Traversal

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

# 100. Same Tree

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

# 1443. Minimum Time to Collect All Apples in a Tree

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

# 1519. Number of Nodes in the Sub-Tree With the Same Label

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

# 2246. Longest Path With Different Adjacent Characters

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

# 1061. Lexicographically Smallest Equivalent String

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

# 2421. Number of Good Paths

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

# 57. Insert Interval

```rust
impl Solution {
    pub fn insert(intervals: Vec<Vec<i32>>, mut new_interval: Vec<i32>) -> Vec<Vec<i32>> {
        let n = intervals.len();
        let mut ret = vec![];

        let mut i = 0;
        while i < n && intervals[i][1] < new_interval[0] {
            ret.push(intervals[i].clone());
            i += 1;
        }

        while i < n && new_interval[1] >= intervals[i][0] {
            new_interval[0] = new_interval[0].min(intervals[i][0]);
            new_interval[1] = new_interval[1].max(intervals[i][1]);
            i += 1;
        }
        ret.push(new_interval);

        while i < n {
            ret.push(intervals[i].clone());
            i += 1;
        }

        ret
    }
}
```

# 926. Flip String to Monotone Increasing

```rust
impl Solution {
    pub fn min_flips_mono_incr(s: String) -> i32 {
        let n = s.len();
        let s = s.as_bytes();
        let mut total_zeros = 0;
        for i in 0..n {
            if s[i] == b'0' {
                total_zeros += 1;
            }
        }

        let mut ret = i32::MAX >> 1;
        let mut left_ones = 0;
        let mut left_zeros = 0;
        for i in 0..=n {
            ret = ret.min(left_ones + (total_zeros - left_zeros));

            if i < n {
                if s[i] == b'0' {
                    left_zeros += 1;
                } else {
                    left_ones += 1;
                }
            }
        }

        ret
    }
}
```

# 918. Maximum Sum Circular Subarray

```rust
impl Solution {
    pub fn max_subarray_sum_circular(nums: Vec<i32>) -> i32 {
        let mut dp_min = nums[0];
        let mut dp_max = nums[0];
        let mut sum = nums[0];
        let mut min = dp_min;
        let mut max = dp_max;

        for &num in nums.iter().skip(1) {
            dp_min = (dp_min + num).min(num);
            dp_max = (dp_max + num).max(num);
            sum += num;
            min = min.min(dp_min);
            max = max.max(dp_max);
        }

        if max < 0 {
            max
        } else {
            max.max(sum - min)
        }
    }
}
```

# 974. Subarray Sums Divisible by K

```rust
use std::collections::HashMap;

impl Solution {
    pub fn subarrays_div_by_k(mut nums: Vec<i32>, k: i32) -> i32 {
        let min = nums.iter().min().unwrap();
        let m = (min.abs() / k) + 1;
        for num in nums.iter_mut() {
            *num += m * k;
        }

        let mut ret = 0;
        let mut sum = 0;
        let mut map = HashMap::new();
        map.insert(0, 1);

        for num in nums {
            sum += num;

            let e = map.entry(sum % k).or_default();
            ret += *e;
            *e += 1;
        }

        ret
    }
}
```

# 491. Non-decreasing Subsequences

```rust
impl Solution {
    pub fn find_subsequences(nums: Vec<i32>) -> Vec<Vec<i32>> {
        fn f(i: usize, last: i32, nums: &Vec<i32>, sub: &mut Vec<i32>, ret: &mut Vec<Vec<i32>>) {
            if i >= nums.len() {
                if sub.len() >= 2 {
                    ret.push(sub.clone());
                }
                return;
            }

            if sub.len() == 0 || nums[i] >= last {
                sub.push(nums[i]);
                f(i + 1, nums[i], nums, sub, ret);
                sub.pop();
            }

            if nums[i] != last {
                f(i + 1, last, nums, sub, ret);
            }
        }

        let mut ret = vec![];
        let mut sub = vec![];
        f(0, i32::MIN >> 1, &nums, &mut sub, &mut ret);
        ret
    }
}
```

# 131. Palindrome Partitioning

```rust
impl Solution {
    pub fn partition(s: String) -> Vec<Vec<String>> {
        fn check(s: &String, mut l: usize, mut r: usize) -> bool {
            while l < r {
                if s.as_bytes()[l] != s.as_bytes()[r] {
                    return false;
                }
                l += 1;
                r -= 1;
            }
            true
        }

        fn f(s: &String, i: usize, parts: &mut Vec<String>, ret: &mut Vec<Vec<String>>) {
            if i >= s.len() {
                ret.push(parts.clone());
                return;
            }

            for j in i..s.len() {
                if check(s, i, j) {
                    parts.push(s[i..=j].into());
                    f(s, j + 1, parts, ret);
                    parts.pop();
                }
            }
        }

        let mut parts = vec![];
        let mut ret = vec![];
        f(&s, 0, &mut parts, &mut ret);

        ret
    }
}
```

# 997. Find the Town Judge

```rust
impl Solution {
    pub fn find_judge(n: i32, trust: Vec<Vec<i32>>) -> i32 {
        let n = n as usize;
        let mut in_degree = vec![0; n + 1];
        let mut out_degree = vec![0; n + 1];
        for edge in trust {
            let (u, v) = (edge[0], edge[1]);
            out_degree[u as usize] += 1;
            in_degree[v as usize] += 1;
        }

        let mut ret = -1;
        for i in 1..=n {
            if out_degree[i] == 0 && in_degree[i] == n - 1 {
                if ret != -1 {
                    return -1;
                }
                ret = i as i32;
            }
        }

        ret
    }
}
```

# 909. Snakes and Ladders

```rust
impl Solution {
    pub fn snakes_and_ladders(board: Vec<Vec<i32>>) -> i32 {
        let n = board.len();
        let mut queue = vec![];
        let mut visited = vec![vec![false; n]; n];
        queue.push(1);
        visited[n - 1][0] = true;

        let mut level = 0;
        while !queue.is_empty() {
            let mut next_queue = vec![];

            for _ in 0..queue.len() {
                let curr = queue.pop().unwrap();
                if curr == n * n {
                    return level;
                }

                for next in (curr + 1)..=(curr + 6).min(n * n) {
                    let i = (next - 1) / n;
                    let j = (next - 1) % n;
                    let revi = n - 1 - i;
                    let revj = n - 1 - j;

                    let i = revi;
                    let j = if revi & 1 == n & 1 { revj } else { j };

                    if visited[i][j] {
                        continue;
                    }
                    visited[i][j] = true;

                    if board[i][j] == -1 {
                        next_queue.push(next);
                    } else {
                        next_queue.push(board[i][j] as usize);
                    }
                }
            }

            queue = next_queue;
            level += 1;
        }

        -1
    }
}
```

# 2359. Find Closest Node to Given Two Nodes

```rust
use std::collections::VecDeque;

impl Solution {
    pub fn closest_meeting_node(edges: Vec<i32>, node1: i32, node2: i32) -> i32 {
        let n = edges.len();
        let mut graph = vec![vec![]; n];
        for i in 0..n {
            if edges[i] != -1 {
                graph[i].push(edges[i] as usize);
            }
        }

        fn dijkstra(graph: &Vec<Vec<usize>>, s: usize) -> Vec<i32> {
            // because all the weight is 1, we could using normal queue to
            // reduce the time complexity
            let mut queue = VecDeque::new();
            let mut dist = vec![i32::MAX >> 1; graph.len()];

            dist[s] = 0;
            queue.push_back((dist[s], s));

            while !queue.is_empty() {
                let node = queue.pop_front().unwrap();
                let (d, u) = (node.0, node.1);

                if d != dist[u] {
                    continue;
                }

                for &v in &graph[u] {
                    if d + 1 < dist[v] {
                        dist[v] = d + 1;
                        queue.push_back((dist[v], v));
                    }
                }
            }

            dist
        }

        let d1 = dijkstra(&graph, node1 as usize);
        let d2 = dijkstra(&graph, node2 as usize);

        let mut ret = -1;
        let mut min = i32::MAX >> 1;
        for i in 0..n {
            let d = d1[i].max(d2[i]);
            if d < min {
                min = d;
                ret = i as i32;
            }
        }

        ret
    }
}
```

# 787. Cheapest Flights Within K Stops

```rust
impl Solution {
    pub fn find_cheapest_price(n: i32, flights: Vec<Vec<i32>>, src: i32, dst: i32, k: i32) -> i32 {
        let mut dist = vec![i32::MAX >> 1; n as usize];
        dist[src as usize] = 0;

        for _ in 0..=k {
            let prev_dist = dist.clone();
            for edge in &flights {
                let (u, v, w) = (edge[0] as usize, edge[1] as usize, edge[2]);
                dist[v] = dist[v].min(prev_dist[u] + w);
            }
        }

        let ret = dist[dst as usize];
        if ret >= i32::MAX >> 1 {
            -1
        } else {
            ret
        }
    }
}
```

# 472. Concatenated Words

```rust
use std::collections::HashSet;

impl Solution {
    pub fn find_all_concatenated_words_in_a_dict(words: Vec<String>) -> Vec<String> {
        let mut word_set = HashSet::new();
        for word in &words {
            word_set.insert(word.as_str());
        }

        let mut ret = vec![];
        for &word in &word_set {
            if Solution::f(word, 0, 0, &word_set) {
                ret.push(word.into());
            }
        }
        ret
    }

    fn f(word: &str, cnt: usize, i: usize, words: &HashSet<&str>) -> bool {
        if i >= word.len() {
            return cnt >= 2;
        }

        for j in i..word.len() {
            let sub = &word[i..=j];
            if words.contains(sub) && Solution::f(word, cnt + 1, j + 1, words) {
                return true;
            }
        }

        false
    }
}
```

# 352. Data Stream as Disjoint Intervals

```rust
struct SummaryRanges {
    data: Vec<Vec<i32>>,
}

impl SummaryRanges {
    fn new() -> Self {
        Self {
            data: vec![
                vec![i32::MIN >> 1, i32::MIN >> 1],
                vec![i32::MAX >> 1, i32::MAX >> 1],
            ],
        }
    }

    fn add_num(&mut self, value: i32) {
        let l = self.data.partition_point(|range| range[1] < value);
        let left = &self.data[l - 1];
        let right = &self.data[l];
        let mid = value;

        fn contains(range: &[i32], v: i32) -> bool {
            range[0] <= v && v <= range[1]
        }

        if left[1] + 1 == mid && mid == right[0] - 1 {
            self.data[l - 1][1] = right[1];
            self.data.remove(l);
        } else if left[1] + 1 == mid || contains(left, mid) {
            self.data[l - 1][1] = left[1].max(mid);
        } else if mid == right[0] - 1 || contains(right, mid) {
            self.data[l][0] = right[0].min(mid);
        } else {
            self.data.insert(l, vec![mid, mid]);
        }
    }

    fn get_intervals(&self) -> Vec<Vec<i32>> {
        self.data[1..self.data.len() - 1].to_vec()
    }
}
```

# 460. LFU Cache

```rust
use std::collections::{BTreeSet, HashMap};

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
struct CacheEntry {
    count: usize,
    used_time: usize,
    key: i32,
    value: i32,
}

struct LFUCache {
    queue: BTreeSet<CacheEntry>,
    cache: HashMap<i32, CacheEntry>,
    capacity: usize,
    time: usize,
}

impl LFUCache {
    fn new(capacity: i32) -> Self {
        Self {
            queue: BTreeSet::new(),
            cache: HashMap::new(),
            capacity: capacity as usize,
            time: 0,
        }
    }

    fn get(&mut self, key: i32) -> i32 {
        self.time += 1;
        if let Some(entry) = self.cache.get_mut(&key) {
            self.queue.remove(entry);
            entry.used_time = self.time;
            entry.count += 1;
            self.queue.insert(entry.clone());
            return entry.value;
        }
        -1
    }

    fn put(&mut self, key: i32, value: i32) {
        self.time += 1;
        if self.capacity == 0 {
            return;
        }
        if let Some(entry) = self.cache.get_mut(&key) {
            self.queue.remove(entry);
            entry.used_time = self.time;
            entry.count += 1;
            entry.value = value;
            self.queue.insert(entry.clone());
        } else {
            if self.cache.len() >= self.capacity {
                let entry = self.queue.iter().next().unwrap().clone();
                self.cache.remove(&entry.key);
                self.queue.remove(&entry);
            }
            let entry = CacheEntry {
                key,
                value,
                count: 1,
                used_time: self.time,
            };
            self.cache.entry(key).or_insert(entry.clone());
            self.queue.insert(entry);
        }
    }
}
```

# 1137. N-th Tribonacci Number

```rust
impl Solution {
    pub fn tribonacci(n: i32) -> i32 {
        let n = n as usize;
        let mut memo = vec![-1; n + 1];

        fn f(memo: &mut [i32], n: usize) -> i32 {
            if n <= 2 {
                return (n as i32).min(1);
            }
            if memo[n] != -1 {
                return memo[n];
            }

            memo[n] = f(memo, n - 1) + f(memo, n - 2) + f(memo, n - 3);
            memo[n]
        }

        f(&mut memo, n)
    }
}
```

# 1626. Best Team With No Conflicts

```rust
use std::cmp::Ordering;

impl Solution {
    pub fn best_team_score(scores: Vec<i32>, ages: Vec<i32>) -> i32 {
        let n = scores.len();
        let mut index: Vec<usize> = (0..n).collect();
        let mut memo = vec![vec![-1; n + 1]; n];
        index.sort_by(|&a, &b| {
            let r = ages[a].cmp(&ages[b]);
            if r == Ordering::Equal {
                return scores[a].cmp(&scores[b]);
            }
            r
        });

        fn f(
            scores: &Vec<i32>,
            index: &Vec<usize>,
            memo: &mut Vec<Vec<i32>>,
            i: usize,
            p: usize,
        ) -> i32 {
            if i >= scores.len() {
                return 0;
            }
            if memo[i][p] != -1 {
                return memo[i][p];
            }

            let mut ret = f(scores, index, memo, i + 1, p);
            if p == scores.len() || scores[index[i]] >= scores[index[p]] {
                ret = ret.max(f(scores, index, memo, i + 1, i) + scores[index[i]]);
            }

            memo[i][p] = ret;
            memo[i][p]
        }

        f(&scores, &index, &mut memo, 0, n)
    }
}
```
