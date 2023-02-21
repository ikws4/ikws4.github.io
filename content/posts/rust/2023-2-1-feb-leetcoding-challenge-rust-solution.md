+++
title = "Feb LeetCoding Challenge Rust Solution"
date = "2023-02-01T10:00:00+08:00"
cover = ""
tags = ["leetcode", "algorithm"]
Toc = true
+++

<!--more-->

# 1071. Greatest Common Divisor of Strings

```rust
impl Solution {
    pub fn gcd_of_strings(str1: String, str2: String) -> String {
        if str2.is_empty() {
            return str1;
        }
        if !str1.starts_with(&str2) && !str2.starts_with(&str1) {
            return "".into();
        }
        let remainder = str1.replace(&str2, "");
        Self::gcd_of_strings(str2, remainder)
    }
}
```

# 953. Verifying an Alien Dictionary

```rust
use std::{cmp::Ordering, collections::HashMap};

impl Solution {
    pub fn is_alien_sorted(words: Vec<String>, order: String) -> bool {
        let char_order =
            order
                .char_indices()
                .fold(HashMap::with_capacity(26), |mut char_order, (i, c)| {
                    char_order.insert(c, i);
                    char_order
                });

        words
            .iter()
            .map(|s| s.chars().map(|c| char_order[&c]).collect::<Vec<_>>())
            .collect::<Vec<_>>()
            .windows(2)
            .all(|w| w[0] <= w[1])
    }
}
```

# 6. Zigzag Conversion

```rust
impl Solution {
    pub fn convert(s: String, num_rows: i32) -> String {
        let mut rows = vec![String::new(); num_rows as usize];

        let mut row = 0;
        let mut delta = -1;
        for c in s.chars() {
            rows[row as usize].push(c);
            if row == 0 || row == num_rows - 1 {
                delta *= -1;
            }
            row += delta;
            row = row.clamp(0, num_rows - 1);
        }

        rows.iter().fold(String::new(), |mut ret, row| {
            ret.push_str(row);
            ret
        })
    }
}
```

# 567. Permutation in String

```rust
use std::collections::HashMap;

impl Solution {
    pub fn check_inclusion(s1: String, s2: String) -> bool {
        fn default_hash_map() -> HashMap<char, i32> {
            let mut map = HashMap::new();
            for c in 'a'..='z' {
                map.entry(c).or_default();
            }
            map
        }

        let cnt1 = s1.chars().fold(default_hash_map(), |mut cnt, c| {
            cnt.entry(c).and_modify(|e| *e += 1);
            cnt
        });
        let mut cnt2 = default_hash_map();

        let s2 = s2.chars().collect::<Vec<_>>();
        let mut i = 0;
        for j in 0..s2.len() {
            let c = s2[j];
            cnt2.entry(c).and_modify(|e| *e += 1);

            while i < j && (j - i + 1 > s1.len() || cnt2[&c] > cnt1[&c]) {
                cnt2.entry(s2[i]).and_modify(|e| *e -= 1);
                i += 1;
            }

            if j - i + 1 == s1.len() && cnt1 == cnt2 {
                return true;
            }
        }

        false
    }
}
```

# 438. Find All Anagrams in a String

```rust
use std::collections::HashMap;

impl Solution {
    pub fn find_anagrams(s: String, p: String) -> Vec<i32> {
        fn default_hash_map() -> HashMap<char, i32> {
            let mut map = HashMap::new();
            for c in 'a'..='z' {
                map.entry(c).or_default();
            }
            map
        }

        let cnt1 = p.chars().fold(default_hash_map(), |mut cnt, c| {
            cnt.entry(c).and_modify(|e| *e += 1);
            cnt
        });
        let mut cnt2 = default_hash_map();

        let mut ret = vec![];
        let s = s.chars().collect::<Vec<_>>();
        let mut i = 0;
        for j in 0..s.len() {
            let c = s[j];
            cnt2.entry(c).and_modify(|e| *e += 1);

            while i < j && (j - i + 1 > p.len() || cnt2[&c] > cnt1[&c]) {
                cnt2.entry(s[i]).and_modify(|e| *e -= 1);
                i += 1;
            }

            if j - i + 1 == p.len() && cnt1 == cnt2 {
                ret.push(i as i32);
            }
        }

        ret
    }
}
```

# 1470. Shuffle the Array

```rust
impl Solution {
    pub fn shuffle(mut nums: Vec<i32>, n: i32) -> Vec<i32> {
        let n = n as usize;
        let right_half = nums.drain(n..).collect::<Vec<_>>();

        nums.iter()
            .zip(right_half.iter())
            .fold(Vec::with_capacity(2 * n), |mut ret, (&a, &b)| {
                ret.push(a);
                ret.push(b);
                ret
            })
    }
}
```

# 904. Fruit Into Baskets

```rust
use std::collections::HashMap;

impl Solution {
    pub fn total_fruit(fruits: Vec<i32>) -> i32 {
        let mut map: HashMap<i32, i32> = HashMap::new();

        let mut i = 0;
        let mut ret = 0;
        for j in 0..fruits.len() {
            map.entry(fruits[j]).and_modify(|e| *e += 1).or_insert(1);

            while map.len() > 2 {
                if map[&fruits[i]] <= 1 {
                    map.remove(&fruits[i]);
                } else {
                    map.entry(fruits[i]).and_modify(|e| *e -= 1);
                }
                i += 1;
            }

            ret = ret.max(j - i + 1);
        }

        ret as i32
    }
}
```

# 45. Jump Game II

```rust
impl Solution {
    pub fn jump(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        let mut memo = vec![-1; n];

        fn f(memo: &mut [i32], nums: &[i32], i: usize) -> i32 {
            if i >= memo.len() {
                return i32::MAX >> 1;
            }
            if i == memo.len() - 1 {
                return 0;
            }
            if memo[i] != -1 {
                return memo[i];
            }

            let mut ret = i32::MAX >> 1;
            for j in 1..=nums[i] as usize {
                ret = ret.min(f(memo, nums, i + j) + 1);
            }

            memo[i] = ret;
            memo[i]
        }

        f(&mut memo, &nums, 0)
    }
}
```

# 2306. Naming a Company

```rust
use std::collections::HashSet;

impl Solution {
    pub fn distinct_names(ideas: Vec<String>) -> i64 {
        let mut set = vec![HashSet::new(); 26];
        for idea in &ideas {
            let k = (idea.as_bytes()[0] - b'a') as usize;
            set[k].insert(&idea[1..]);
        }

        let mut ret = 0;
        for i in 0..26 {
            for j in (i + 1)..26 {
                let mut a = 0;
                let mut b = 0;

                for &v in &set[i] {
                    if !set[j].contains(v) {
                        a += 1;
                    }
                }

                for &v in &set[j] {
                    if !set[i].contains(v) {
                        b += 1;
                    }
                }

                ret += a * b;
            }
        }

        2 * ret
    }
}
```

# 1162. As Far from Land as Possible

```rust
use std::collections::VecDeque;

impl Solution {
    pub fn max_distance(mut grid: Vec<Vec<i32>>) -> i32 {
        let m = grid.len();
        let n = grid[0].len();
        let mut queue = VecDeque::new();

        for i in 0..m {
            for j in 0..n {
                if grid[i][j] == 1 {
                    queue.push_back((i as i32, j as i32));
                }
            }
        }

        if queue.is_empty() || queue.len() == m * n {
            return -1;
        }

        let mut ret = 0;
        let dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]];
        while !queue.is_empty() {
            for k in 0..queue.len() {
                if let Some((i, j)) = queue.pop_front() {
                    for &dir in &dirs {
                        let _i = i as i32 + dir[0];
                        let _j = j as i32 + dir[1];

                        if _i < 0 || _i >= m as i32 || _j < 0 || _j >= n as i32 || grid[_i as usize][_j as usize] == -1 {
                            continue;
                        }

                        grid[_i as usize][_j as usize] = -1;
                        queue.push_back((_i, _j));
                    }
                }
            }
            ret += 1;
        }

        ret - 1
    }
}
```

# 1129. Shortest Path with Alternating Colors

```rust
use std::collections::VecDeque;

impl Solution {
    pub fn shortest_alternating_paths(n: i32, red_edges: Vec<Vec<i32>>, blue_edges: Vec<Vec<i32>>) -> Vec<i32> {
        let n = n as usize;
        let mut graph = vec![vec![]; n];
        for edge in &red_edges {
            graph[edge[0] as usize].push((edge[1] as usize, 0));
        }
        for edge in &blue_edges {
            graph[edge[0] as usize].push((edge[1] as usize, 1));
        }

        let mut queue = VecDeque::new();
        let mut visited = vec![vec![false; n]; 3];
        queue.push_back((2, 0));
        visited[2][0] = true;

        let mut ret = vec![-1; n];
        let mut dist = 0;
        while !queue.is_empty() {
            for _ in 0..queue.len() {
                let (uc, u) = queue.pop_front().unwrap();

                if ret[u] == -1 {
                    ret[u] = dist;
                }

                for &(v, vc) in &graph[u] {
                    if uc != vc && !visited[vc][v] {
                        queue.push_back((vc, v));
                        visited[vc][v] = true;
                    }
                }
            }
            dist += 1;
        }

        ret
    }
}
```

# 2477. Minimum Fuel Cost to Report to the Capital

```rust
struct Env {
    graph: Vec<Vec<usize>>,
    seats: i32,
}

impl Solution {
    pub fn minimum_fuel_cost(roads: Vec<Vec<i32>>, seats: i32) -> i64 {
        let n = roads.len() + 1;
        let mut graph = vec![vec![]; n];
        for road in roads {
            let u = road[0] as usize;
            let v = road[1] as usize;
            graph[u].push(v);
            graph[v].push(u);
        }

        fn f(env: &Env, people: &mut Vec<i32>, u: usize, p: usize) -> i64 {
            let mut ret = 0;

            for &v in &env.graph[u] {
                if v != p {
                    ret += f(env, people, v, u) + (people[v] as f32 / env.seats as f32).ceil() as i64;
                    people[u] += people[v];
                }
            }

            ret
        }

        f(&Env { graph, seats }, &mut vec![1; n], 0, n)
    }
}
```

# 1523. Count Odd Numbers in an Interval Range

```rust
impl Solution {
    pub fn count_odds(low: i32, high: i32) -> i32 {
        let n = high - low + 1;
        if low & 1 == 0 {
            n / 2
        } else {
            (n - 1) / 2 + 1
        }
    }
}
```

# 67. Add Binary

```rust
impl Solution {
    pub fn add_binary(a: String, b: String) -> String {
        let mut c = 0;
        let mut i = (a.len() - 1) as i32;
        let mut j = (b.len() - 1) as i32;
        let mut ret = String::new();

        while i >= 0 || j >= 0 || c > 0 {
            let _a = if i >= 0 { a.as_bytes()[i as usize] - b'0' } else { 0 };
            let _b = if j >= 0 { b.as_bytes()[j as usize] - b'0' } else { 0 };
            let v = _a + _b + c;
            ret.push_str(&(v % 2).to_string());
            c = v / 2;
            i -= 1;
            j -= 1;
        }

        ret.chars().rev().collect()
    }
}
```

# 989. Add to Array-Form of Integer

```rust
impl Solution {
    pub fn add_to_array_form(mut num: Vec<i32>, mut k: i32) -> Vec<i32> {
        let mut ret = vec![];
        num.insert(0, 0);

        let mut i = (num.len() - 1) as i32;
        let mut c = 0;

        while i >= 1 || k > 0 || c > 0 {
            let v = num[i.max(0) as usize] + k % 10 + c;
            ret.push(v % 10);
            k /= 10;
            c = v / 10;
            i -= 1;
        }

        ret.into_iter().rev().collect::<Vec<i32>>()
    }
}
```

# 104. Maximum Depth of Binary Tree

```rust
use std::rc::Rc;
use std::cell::RefCell;

impl Solution {
    pub fn max_depth(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        if let Some(root) = root {
            let mut root = root.borrow_mut();
            let l = Self::max_depth(root.left.take());
            let r = Self::max_depth(root.right.take());
            l.max(r) + 1
        } else {
            0
        }
    }
}
```

# 783. Minimum Distance Between BST Nodes

```rust
use std::rc::Rc;
use std::cell::RefCell;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn min_diff_in_bst(root: Node) -> i32 {
        fn f(p: &mut i32, root: &Node) -> i32 {
            if let Some(root) = root {
                let root = root.borrow();
                let l = f(p, &root.left);
                let m = (*p - root.val).abs();
                *p = root.val;
                let r = f(p, &root.right);
                return l.min(m).min(r);
            }
            i32::MAX >> 1
        }

        f(&mut (i32::MAX >> 1), &root)
    }
}
```

# 226. Invert Binary Tree

```rust
use std::cell::RefCell;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn invert_tree(root: Node) -> Node {
        if let Some(root) = root {
            let _root = root.clone();
            let mut root = root.borrow_mut();
            let l = Self::invert_tree(root.left.take());
            let r = Self::invert_tree(root.right.take());
            root.left = r;
            root.right = l;
            return Some(_root);
        }
        None
    }
}
```

# 103. Binary Tree Zigzag Level Order Traversal

```rust
use std::rc::Rc;
use std::cell::RefCell;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn zigzag_level_order(root: Node) -> Vec<Vec<i32>> {
        fn f(root: &Node, depth: usize, ret: &mut Vec<Vec<i32>>) {
            if let Some(root) = root {
                let root = root.borrow();

                if ret.len() <= depth {
                    ret.push(vec![]);
                }
                ret[depth].push(root.val);
                
                f(&root.left, depth + 1, ret);
                f(&root.right, depth + 1, ret);
            }
        }

        let mut ret = vec![];
        f(&root, 0,&mut ret);
        for (d, e) in ret.iter_mut().enumerate() {
            if d & 1 == 1 {
                e.reverse();
            }
        }
        ret
    }
}
```

# 35. Search Insert Position 

```rust
impl Solution {
    pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
        nums.partition_point(|num| num < &target) as i32
    }
}
```

# 540. Single Element in a Sorted Array

```rust
impl Solution {
    pub fn single_non_duplicate(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        let mut l = 0;
        let mut r = n;
        while l < r {
            let m = l + (r - l) / 2;
            // 1 1 2 2 3 3 
            //       ^
            // 1 1 2 2 3 3 
            //         ^
            // -----------
            // 0 1 2 3 4 5 (index)
            if (m as i32 - 1) >= 0 && m & 1 == 1 && nums[m] == nums[m - 1] ||
                m + 1          < n && m & 1 == 0 && nums[m] == nums[m + 1] {
                l = m + 1;
            } else {
                r = m;
            }
        }
        nums[l]
    }
}
```
