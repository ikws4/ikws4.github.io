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

### 981. Time Based Key-Value Store

```rust
use std::collections::{BTreeMap, HashMap};

struct TimeMap {
    map: HashMap<String, BTreeMap<i32, String>>,
}

impl TimeMap {
    fn new() -> Self {
        TimeMap {
            map: HashMap::new(),
        }
    }

    fn set(&mut self, key: String, value: String, timestamp: i32) {
        self.map.entry(key).or_default().insert(timestamp, value);
    }

    fn get(&self, key: String, timestamp: i32) -> String {
        self.map
            .get(&key)
            .and_then(|m| m.range(..=timestamp).next_back().map(|(_, v)| v.clone()))
            .unwrap_or_default()
    }
}
```

### 732. My Calendar III

```rust
use std::collections::BTreeMap;

struct MyCalendarThree {
    diff: BTreeMap<i32, i32>,
}

impl MyCalendarThree {
    fn new() -> Self {
        MyCalendarThree {
            diff: BTreeMap::new(),
        }
    }

    fn book(&mut self, start: i32, end: i32) -> i32 {
        *self.diff.entry(start).or_default() += 1;
        *self.diff.entry(end).or_default() -= 1;

        let mut s = 0;
        let mut ret = 0;
        for d in self.diff.values() {
            s += d;
            ret = ret.max(s);
        }

        ret
    }
}
```

### 16. 3Sum Closest

```rust
impl Solution {
    pub fn three_sum_closest(mut nums: Vec<i32>, target: i32) -> i32 {
        nums.sort();

        let mut ret = 0;
        let mut diff = i32::MAX >> 1;
        for i in 0..nums.len() {
            let mut j = i + 1;
            let mut k = nums.len() - 1;
            while j < k {
                let sum = nums[i] + nums[j] + nums[k];
                let d = (sum - target).abs();
                if d < diff {
                    diff = d;
                    ret = sum;
                    if d == 0 {
                        break;
                    }
                }

                if sum < target {
                    j += 1;
                } else {
                    k -= 1;
                }
            }
        }

        ret
    }
}
```

### 653. Two Sum IV - Input is a BST

```rust
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn find_target(root: Node, k: i32) -> bool {
        fn internal(root: &Node, k: i32, set: &mut HashSet<i32>) -> bool {
            if let Some(root) = root {
                let root = root.borrow();
                if set.contains(&(k - root.val)) {
                    return true;
                }
                set.insert(root.val);
                return internal(&root.left, k, set) || internal(&root.right, k, set);
            }
            false
        }

        internal(&root, k, &mut HashSet::new())
    }
}
```

### 1328. Break a Palindrome

```rust
impl Solution {
    pub fn break_palindrome(mut palindrome: String) -> String {
        let n = palindrome.len();
        if n == 1 {
            return "".into();
        }

        let mut p = palindrome.into_bytes();
        for i in 0..n / 2 {
            if p[i] > b'a' {
                p[i] = b'a';
                return String::from_utf8(p).unwrap();
            }
        }
        p[n - 1] = b'b';

        String::from_utf8(p).unwrap()
    }
}
```

### 334. Increasing Triplet Subsequence

```rust
impl Solution {
    pub fn increasing_triplet(nums: Vec<i32>) -> bool {
        let mut m1 = i32::MAX;
        let mut m2 = i32::MAX;

        for &num in &nums {
            if num <= m1 {
                m1 = num;
            } else if num <= m2 {
                m2 = num;
            } else {
                return true;
            }
        }

        false
    }
}
```

### 976. Largest Perimeter Triangle

```rust
impl Solution {
    pub fn largest_perimeter(mut nums: Vec<i32>) -> i32 {
        nums.sort();
        for w in nums.windows(3).rev() {
            if let &[a, b, c] = w {
                if a + b > c {
                    return a + b + c;
                }
            }
        }
        0
    }
}
```

### 237. Delete Node in a Linked List

```rust
impl Solution {
    pub fn delete_node(node: Option<Box<ListNode>>) {
        if let Some(mut node) = node {
            let next = node.next.unwrap();
            node.val = next.val;
            node.next = next.next;
        }
    }
}
```

### 2095. Delete the Middle Node of a Linked List

```rust
type Node = Option<Box<ListNode>>;

impl Solution {
    pub fn delete_middle(head: Node) -> Node {
        fn node_len(head: &Node) -> usize {
            if let Some(head) = head {
                return node_len(&head.next) + 1;
            }
            0
        }

        fn node_delete(head: Node, n: usize) -> Node {
            if let Some(mut head) = head {
                if n == 0 {
                    return head.next;
                } else {
                    head.next = node_delete(head.next, n - 1);
                    return Some(head);
                }
            }
            None
        }

        let len = node_len(&head);
        node_delete(head, len / 2)
    }
}
```

### 1531. String Compression II

```rust
struct Env<'a> {
    s: &'a [u8],
    memo: Vec<Vec<i32>>,
}

impl Solution {
    pub fn get_length_of_optimal_compression(s: String, k: i32) -> i32 {
        fn run(c: usize) -> i32 {
            if c == 1 {
                return 1;
            }
            if c <= 9 {
                return 2;
            }
            if c <= 99 {
                return 3;
            }
            4
        }

        fn internal(env: &mut Env, i: usize, k: usize) -> i32 {
            if env.s.len() - i <= k {
                return 0;
            }
            if env.memo[i][k] != -1 {
                return env.memo[i][k];
            }

            let mut ret = i32::MAX >> 1;
            let mut keep_count = 0;
            let mut counter = vec![0; 26];
            for j in i..env.s.len() {
                let at = (env.s[j] - b'a') as usize;
                counter[at] += 1;

                keep_count = keep_count.max(counter[at]);
                let remove_count = j - i + 1 - keep_count;
                if remove_count <= k {
                    ret = ret.min(internal(env, j + 1, k - remove_count) + run(keep_count));
                }
            }

            env.memo[i][k] = ret;
            ret
        }

        internal(&mut Env {
            s: s.as_bytes(),
            memo: vec![vec![-1; 101]; 101],
        }, 0, k as usize)
    }
}
```

### 1335. Minimum Difficulty of a Job Schedule

```rust
struct Env {
    n: usize,
    max: Vec<Vec<i32>>,
    memo: Vec<Vec<i32>>,
}

impl Solution {
    pub fn min_difficulty(job_difficulty: Vec<i32>, d: i32) -> i32 {
        fn internal(env: &mut Env, i: usize, p: usize) -> i32 {
            if i >= env.n {
                return i32::MAX >> 1;
            }
            if p == 0 {
                return env.max[i][env.n - 1];
            }
            if env.memo[i][p] != -1 {
                return env.memo[i][p];
            }

            let mut ret = i32::MAX >> 1;
            for j in i..env.n {
                ret = ret.min(env.max[i][j] + internal(env, j + 1, p - 1));
            }

            env.memo[i][p] = ret;
            ret
        }

        // build max array
        let n = job_difficulty.len();
        let mut max = vec![vec![0; n]; n];
        for i in 0..n {
            let mut m = job_difficulty[i];
            for j in i..n {
                m = m.max(job_difficulty[j]);
                max[i][j] = m;
            }
        }

        // compute the result
        let d = d as usize;
        let ret = internal(
            &mut Env {
                n,
                max,
                memo: vec![vec![-1; d]; n],
            },
            0,
            d - 1,
        );

        // check invalid solution
        if ret == i32::MAX >> 1 {
            -1
        } else {
            ret
        }
    }
}
```

### 1832. Check if the Sentence Is Pangram

```rust
impl Solution {
    pub fn check_if_pangram(sentence: String) -> bool {
        let mut has = vec![false; 26];
        let sentence = sentence.as_bytes();
      
        for c in sentence {
            has[(c - b'a') as usize] = true;
        }

        for c in 0..26 {
            if !has[c] {
                return false;
            }
        }

        true
    }
}
```

### 38. Count and Say

```rust
impl Solution {
    pub fn count_and_say(n: i32) -> String {
        if n == 1 {
            return "1".to_string();
        }

        let r = Solution::count_and_say(n - 1);
        let r = r.as_bytes();

        let mut ret = String::new();
        let mut cnt = 1;
        for i in 1..=r.len() {
            if i == r.len() || r[i] != r[i - 1] {
                ret.push_str(&cnt.to_string());
                ret.push(r[i - 1] as char);
                cnt = 0;
            }
            cnt += 1;
        }

        ret
    }
}
```

### 692. Top K Frequent Words

```rust
use std::collections::HashMap;

impl Solution {
    pub fn top_k_frequent(words: Vec<String>, k: i32) -> Vec<String> {
        let mut map: HashMap<String, i32> = HashMap::new();
        for word in words {
            *map.entry(word).or_default() += 1;
        }

        let mut entries = vec![];
        for entry in map {
            entries.push(entry);
        }

        entries.sort_by(|a, b| {
            use std::cmp::Ordering;
            
            let ret = b.1.cmp(&a.1);
            match ret {
                Ordering::Equal => a.0.cmp(&b.0),
                _ => ret
            }
        });

        let mut ret = vec![];
        for entry in entries {
            if ret.len() >= k as usize {
                break;
            }
            ret.push(entry.0);
        }
        
        ret
    }
}
```
