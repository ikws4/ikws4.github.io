---
title: "Nov LeetCoding Challenge Rust Solution"
date: 2022-11-1 10:00:00 +0800
layout: post
toc: true
toc_sticky: true
tags: [leetcode, algorithm]
---

### 1706. Where Will the Ball Fall

```rust
impl Solution {
    pub fn find_ball(grid: Vec<Vec<i32>>) -> Vec<i32> {
        let m = grid[0].len();
        let mut ret = vec![0; m];

        fn internal(grid: &Vec<Vec<i32>>, i: usize, j: usize) -> i32 {
            if i == grid.len() {
                return j as i32;
            }
            if grid[i][j] == 1 && (j + 1 >= grid[0].len() || grid[i][j + 1] == -1) {
                return -1;
            }
            if grid[i][j] == -1 && (j as i32 - 1 < 0 || grid[i][j - 1] == 1) {
                return -1;
            }

            if grid[i][j] == 1 {
                internal(grid, i + 1, j + 1)
            } else {
                internal(grid, i + 1, j - 1)
            }
        }

        for j in 0..m {
            ret[j] = internal(&grid, 0, j);
        }

        ret
    }
}
```

### 433. Minimum Genetic Mutation

```rust
use std::collections::{HashSet, VecDeque};

impl Solution {
    pub fn min_mutation(start: String, end: String, bank: Vec<String>) -> i32 {
        let choices = ["A", "C", "G", "T"];

        let mut bank_set: HashSet<String> = bank.into_iter().collect();
        let mut queue = VecDeque::new();
        queue.push_back(start);

        let mut ret = 0;
        while !queue.is_empty() {
            let size = queue.len();
            for _ in 0..size {
                let curr = queue.pop_front().unwrap();
                if curr == end {
                    return ret;
                }

                for i in 0..curr.len() {
                    let mut next = curr.clone();
                    for choice in choices {
                        next.replace_range(i..=i, choice);

                        if !bank_set.contains(&next) {
                            continue;
                        }

                        bank_set.remove(&next);
                        queue.push_back(next.clone());
                    }
                }
            }
            ret += 1;
        }

        -1
    }
}
```

### 2131. Longest Palindrome by Concatenating Two Letter Words

```rust
use std::collections::HashMap;

impl Solution {
    pub fn longest_palindrome(words: Vec<String>) -> i32 {
        let mut counter = HashMap::new();

        let mut ret = 0;
        for word in words.iter() {
            let rev_word = word.chars().rev().collect::<String>();
            
            let rev_cnt = counter.entry(rev_word).or_insert(0);
            if *rev_cnt > 0 {
                ret += 4;
                *rev_cnt -= 1;
            } else {
                *counter.entry(word.clone()).or_insert(0) += 1;
            }
        }
        
        for word in words.iter() {
            let bytes = word.as_bytes();
            if bytes[0] == bytes[1] && counter[word] > 0 {
                ret += 2;
                break;
            }
        }

        ret
    }
}
```

### 212. Word Search II

```rust
use std::collections::{HashMap, HashSet};

const DIRS: [[i32; 2]; 4] = [[0, 1], [1, 0], [0, -1], [-1, 0]];

struct Trie {
    is_word: bool,
    children: HashMap<char, Trie>,
}

impl Trie {
    pub fn new() -> Self {
        Trie {
            is_word: false,
            children: HashMap::new(),
        }
    }

    pub fn insert(&mut self, word: &str) {
        let mut node = self;

        for c in word.chars() {
            let child = node.children.entry(c).or_insert_with(Trie::new);
            node = child;
        }

        node.is_word = true;
    }
}

impl Solution {
    pub fn find_words(mut board: Vec<Vec<char>>, words: Vec<String>) -> Vec<String> {
        let mut trie = Trie::new();
        for word in &words {
            trie.insert(word);
        }

        fn internal(
            board: &mut Vec<Vec<char>>,
            i: i32,
            j: i32,
            mut node: &Trie,
            path: &mut String,
            ret: &mut HashSet<String>,
        ) {
            if i < 0 || i >= board.len() as i32 || j < 0 || j >= board[0].len() as i32 {
                return;
            }
            let iu = i as usize;
            let ju = j as usize;

            if board[iu][ju] == '#' {
                return;
            }

            if let Some(child) = node.children.get(&board[iu][ju]) {
                node = child;
            } else {
                return;
            }

            let v = board[iu][ju];
            board[iu][ju] = '#';
            path.push(v);

            if node.is_word {
                ret.insert(path.clone());
            }

            for dir in DIRS {
                internal(board, i + dir[0], j + dir[1], node, path, ret);
            }

            path.pop();
            board[iu][ju] = v;
        }

        let mut ret = HashSet::new();
        let mut path = "".to_string();
        for i in 0..board.len() as i32 {
            for j in 0..board[0].len() as i32 {
                internal(&mut board, i, j, &trie, &mut path, &mut ret);
            }
        }

        ret.into_iter().collect()
    }
}
```

### 899. Orderly Queue

```rust
impl Solution {
    pub fn orderly_queue(s: String, k: i32) -> String {
        if k == 1 {
            let mut ret = s.clone();
            for i in 1..s.len() {
                let l = &s[0..i];
                let r = &s[i..];
                let t = format!("{}{}", r, l);
                if t < ret {
                    ret = t;
                }
            }
            ret
        } else {
            let mut s: Vec<char> = s.chars().collect();
            s.sort();
            s.into_iter().collect()
        }
    }
}
```

### 1323. Maximum 69 Number

```rust
impl Solution {
    pub fn maximum69_number (num: i32) -> i32 {
        let mut s = num.to_string();
        if let Some(index) = s.find('6') {
            s.replace_range(index..=index, "9");
            return s.parse().unwrap();
        }
        num
    }
}
```

### 1544. Make The String Great

```rust
impl Solution {
    pub fn make_good(s: String) -> String {
        let mut ret = "".to_string();
        for c in s.chars() {
            let revc = if c.is_ascii_lowercase() {
                c.to_ascii_uppercase()
            } else {
                c.to_ascii_lowercase()
            };
            if !ret.is_empty() && ret.ends_with(revc) {
                ret.pop();
            } else {
                ret.push(c);
            }
        }
        ret
    }
}
```

### 901. Online Stock Span

```rust
struct StockSpanner {
    prices: Vec<i32>,
    stack: Vec<usize>,
    cache: Vec<i32>,
}

impl StockSpanner {
    fn new() -> Self {
        StockSpanner {
            prices: Vec::new(),
            stack: Vec::new(),
            cache: Vec::new(),
        }
    }

    fn next(&mut self, price: i32) -> i32 {
        self.prices.push(price);
        let last_index = self.prices.len() - 1;
        let mut index = last_index;
        while let Some(&peek) = self.stack.last() {
            if price >= self.prices[peek] {
                index = self.stack.pop().unwrap();
            } else {
                break;
            }
        }
        self.stack.push(last_index);

        let ret = (last_index - index + 1) as i32 + (self.cache.get(index).unwrap_or(&1) - 1);
        self.cache.push(ret);

        ret
    }
}
```

### 1047. Remove All Adjacent Duplicates In String

```rust
impl Solution {
    pub fn remove_duplicates(s: String) -> String {
        let mut ret = "".to_string();
        for c in s.chars() {
            if ret.ends_with(c) {
                ret.pop();
            } else {
                ret.push(c);
            }
        }
        ret
    }
}
```

### 26. Remove Duplicates from Sorted Array

```rust
impl Solution {
    pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
        let mut j = 0;
        for i in 0..nums.len() {
            if nums[i] != nums[j] {
                j += 1;
                nums[j] = nums[i];
            }
        }

        (j + 1) as i32
    }
}
```

### 295. Find Median from Data Stream

```rust
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

struct MedianFinder {
    left: BinaryHeap<Reverse<i32>>,
    right: BinaryHeap<i32>,
}

impl MedianFinder {
    fn new() -> Self {
        MedianFinder {
            left: BinaryHeap::new(),
            right: BinaryHeap::new(),
        }
    }

    fn add_num(&mut self, num: i32) {
        self.left.push(Reverse(num));
        self.right.push(self.left.pop().unwrap().0);

        if self.right.len() > self.left.len() {
            self.left.push(Reverse(self.right.pop().unwrap()));
        }
    }

    fn find_median(&self) -> f64 {
        match self.left.len().cmp(&self.right.len()) {
            Ordering::Less => *self.right.peek().unwrap() as f64,
            Ordering::Equal => (self.left.peek().unwrap().0 + self.right.peek().unwrap()) as f64 / 2.0,
            Ordering::Greater => self.left.peek().unwrap().0 as f64
        }
    }
}
```

### 151. Reverse Words in a String

```rust
impl Solution {
    pub fn reverse_words(s: String) -> String {
        s.split(' ')
            .filter(|word| word != &"")
            .rev()
            .collect::<Vec<&str>>()
            .join(" ")
    }
}
```

### 947. Most Stones Removed with Same Row or Column

```rust
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        let mut parent = vec![0; n];
        let rank = vec![0; n];

        for i in 0..parent.len() {
            parent[i] = i;
        }

        Self { parent, rank }
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
            if self.rank[pv] == self.rank[pu] {
                self.rank[pv] += 1;
            }
        }

        true
    }

    fn find(&mut self, u: usize) -> usize {
        if self.parent[u] == u {
            return u;
        }
        self.parent[u] = self.find(self.parent[u]);
        self.parent[u]
    }
}

impl Solution {
    pub fn remove_stones(stones: Vec<Vec<i32>>) -> i32 {
        let n = stones.len();
        let mut ret: i32 = 0;
        let mut uf = UnionFind::new(n);

        for i in 0..n {
            for j in (i + 1)..n {
                if stones[i][0] == stones[j][0] || stones[i][1] == stones[j][1] {
                    if uf.union(i, j) {
                        ret += 1;
                    }
                }
            }
        }

        ret
    }
}
```

### 222. Count Complete Tree Nodes

```rust
use std::rc::Rc;
use std::cell::RefCell;

impl Solution {
    pub fn count_nodes(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        if let Some(root) = root {
            let mut root = root.borrow_mut();
            return Solution::count_nodes(root.left.take()) +
                   Solution::count_nodes(root.right.take()) + 1;
        }
        0
    }
}
```

### 374. Guess Number Higher or Lower

```rust
impl Solution {
    unsafe fn guessNumber(n: i32) -> i32 {
        let mut l = 0;
        let mut r = n;
      
        while l < r {
            let m = l + (r - l) / 2;
            if guess(m) == 1 {
                l = m + 1;
            } else {
                r = m;
            }
        }
      
        l
    }
}
```

### 223. Rectangle Area

```rust
impl Solution {
    pub fn compute_area(
        ax1: i32,
        ay1: i32,
        ax2: i32,
        ay2: i32,
        bx1: i32,
        by1: i32,
        bx2: i32,
        by2: i32,
    ) -> i32 {
        let cx1 = ax1.max(bx1);
        let cy1 = ay1.max(by1);
        let cx2 = ax2.min(bx2);
        let cy2 = ay2.min(by2);

        Solution::area(ax1, ay1, ax2, ay2) +
        Solution::area(bx1, by1, bx2, by2) -
        Solution::area(cx1, cy1, cx2, cy2)
    }

    fn area(x1: i32, y1: i32, x2: i32, y2: i32) -> i32 {
        if x1 > x2 || y1 > y2 {
            return 0;
        }
        (x2 - x1) * (y2 - y1)
    }
}
```

### 263. Ugly Number

```rust
impl Solution {
    pub fn is_ugly(mut n: i32) -> bool {
        if n < 1 {
            return false;
        }

        while n & 1 == 0 {
            n >>= 1;
        }

        while n % 3 == 0 {
            n /= 3;
        }

        while n % 5 == 0 {
            n /= 5
        }

        n == 1
    }
}
```

### 587. Erect the Fence

```rust
impl Solution {
    pub fn outer_trees(mut points: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let p0 = &points
            .iter()
            .min_by(|a, b| a[1].cmp(&b[1]).then(a[0].cmp(&b[0])))
            .unwrap()
            .clone();

        points.sort_by(|a, b| {
            angle(p0, a)
                .partial_cmp(&angle(p0, b))
                .unwrap()
                .then(dist(p0, a).cmp(&dist(p0, b)))
        });

        let n = points.len();
        let mut i = (n - 1) as i32;
        while i >= 0 && orientation(p0, &points[n - 1], &points[i as usize]) == 0 {
            i -= 1;
        }
        reverse(&mut points, (i + 1) as usize, n - 1);

        let mut stack: Vec<Vec<i32>> = vec![];
        for p in points {
            while stack.len() >= 2
                && orientation(&stack[stack.len() - 2], &stack[stack.len() - 1], &p) < 0
            {
                stack.pop();
            }
            stack.push(p);
        }

        stack
    }
}

fn reverse(points: &mut Vec<Vec<i32>>, mut l: usize, mut r: usize) {
    while l < r {
        points.swap(l, r);
        l += 1;
        r -= 1;
    }
}

fn angle(p0: &[i32], p1: &[i32]) -> f32 {
    ((p1[1] - p0[1]) as f32).atan2((p1[0] - p0[0]) as f32)
}

fn dist(p0: &[i32], p1: &[i32]) -> i32 {
    (p1[0] - p0[0]).pow(2) + (p1[1] - p0[1]).pow(2)
}

fn orientation(p0: &[i32], p1: &[i32], p2: &[i32]) -> i32 {
    (p2[1] - p1[1]) * (p1[0] - p0[0]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
}
```

### 224. Basic Calculator

```rust
impl Solution {
    pub fn calculate(s: String) -> i32 {
        let expr = s
            .chars()
            .filter(|it| !it.is_ascii_whitespace())
            .collect::<Vec<char>>();
        Calculator::new(expr).eval()
    }
}

struct Calculator {
    expr: Vec<char>,
    index: usize,
}

// grammar
//   term: unary (('+' / '-') unary)*
//   unary: '-'? group
//   group: '(' term ')' | number
//   number: ('0' - '9')+
impl Calculator {
    pub fn new(expr: Vec<char>) -> Self {
        Self { expr, index: 0 }
    }

    pub fn eval(&mut self) -> i32 {
        self.term()
    }

    fn term(&mut self) -> i32 {
        let mut ret = self.unary();
        while self.is_match('+') || self.is_match('-') {
            if self.is_match('+') {
                self.advance();
                ret += self.unary();
            } else {
                self.advance();
                ret -= self.unary();
            }
        }
        ret
    }

    fn unary(&mut self) -> i32 {
        if self.is_match('-') {
            self.advance();
            return -self.group();
        }
        self.group()
    }

    fn group(&mut self) -> i32 {
        if self.is_match('(') {
            self.advance();
            let ret = self.term();
            self.advance();
            return ret;
        }
        self.number()
    }

    fn number(&mut self) -> i32 {
        let mut num: i32 = 0;
        while self.peek() >= '0' && self.peek() <= '9' {
            num = num * 10 + (self.peek() as u8 - b'0') as i32;
            self.advance();
        }
        num
    }

    fn advance(&mut self) {
        self.index += 1
    }

    fn is_match(&self, c: char) -> bool {
        self.peek() == c
    }

    fn peek(&self) -> char {
        if self.index >= self.expr.len() {
            return '\0';
        }
        self.expr[self.index]
    }
}
```

### 1926. Nearest Exit from Entrance in Maze

```rust
use std::collections::VecDeque;

impl Solution {
    pub fn nearest_exit(maze: Vec<Vec<char>>, entrance: Vec<i32>) -> i32 {
        let (m, n) = (maze.len(), maze[0].len());
        let mut queue = VecDeque::new();
        let mut visited = vec![vec![false; n]; m];
        queue.push_back((entrance[0], entrance[1]));
        visited[entrance[0] as usize][entrance[1] as usize] = true;

        let dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]];
        let mut ret = 0;
        while !queue.is_empty() {
            ret += 1;
            for _ in 0..queue.len() {
                let u = queue.pop_front().unwrap();

                for dir in dirs {
                    let i = u.0 + dir[0];
                    let j = u.1 + dir[1];

                    if i < 0 || i >= m as i32 || j < 0 || j >= n as i32 {
                        continue;
                    }

                    if visited[i as usize][j as usize] || maze[i as usize][j as usize] == '+' {
                        continue;
                    }
                    if i == 0 || i == (m - 1) as i32 || j == 0 || j == (n - 1) as i32 {
                        return ret;
                    }

                    queue.push_back((i, j));
                    visited[i as usize][j as usize] = true;
                }
            }
        }

        -1
    }
}
```

### 279. Perfect Squares

```rust
use std::collections::VecDeque;

impl Solution {
    pub fn num_squares(n: i32) -> i32 {
        let n = n as usize;
        let mut queue = VecDeque::new();
        let mut visited = vec![false; n + 1];
        queue.push_back(n);
        visited[n] = true;

        let mut ret = 0;
        while !queue.is_empty() {
            ret += 1;
            for _ in 0..queue.len() {
                let u = queue.pop_front().unwrap();
                for i in 1..=u {
                    if u < i * i {
                        break;
                    }
                    let v = u - i * i;
                    if v == 0 {
                        return ret;
                    }
                    if visited[v] {
                        continue;
                    }

                    queue.push_back(v);
                    visited[v] = true;
                }
            }
        }

        -1
    }
}
```

### 36. Valid Sudoku

```rust
impl Solution {
    pub fn is_valid_sudoku(board: Vec<Vec<char>>) -> bool {
        let n = board.len();
        let mut row = vec![0; n];
        let mut col = vec![0; n];
        let mut cel = vec![0; n];

        for i in 0..n {
            for j in 0..n {
                if board[i][j] != '.' {
                    let k = (i / 3) * 3 + (j / 3);
                    let mask = 1 << (board[i][j] as u8 - b'0');

                    if row[i] & mask != 0 || col[j] & mask != 0 || cel[k] & mask != 0 {
                        return false;
                    }

                    row[i] |= mask;
                    col[j] |= mask;
                    cel[k] |= mask;
                }
            }
        }

        true
    }
}
```

### 79. Word Search

```rust
const DIRS: [[i32; 2]; 4] = [[0, 1], [1, 0], [0, -1], [-1, 0]];

struct Env {
    board: Vec<Vec<char>>,
    word: Vec<char>,
    m: i32,
    n: i32,
}

impl Solution {
    pub fn exist(board: Vec<Vec<char>>, word: String) -> bool {
        fn internal(env: &mut Env, i: i32, j: i32, k: usize) -> bool {
            if k == env.word.len() {
                return true;
            }
            if i < 0 || i >= env.m || j < 0 || j >= env.n {
                return false;
            }

            let (ui, uj) = (i as usize, j as usize);
            let v = env.board[ui][uj];

            if v == '#' || v != env.word[k] {
                return false;
            }

            env.board[ui][uj] = '#';
            for dir in DIRS {
                if internal(env, i + dir[0], j + dir[1], k + 1) {
                    return true;
                }
            }
            env.board[ui][uj] = v;

            false
        }

        let (m, n) = (board.len() as i32, board[0].len() as i32);
        let word = word.chars().collect::<Vec<char>>();
        let mut env = Env { board, word, m, n };

        for i in 0..m {
            for j in 0..n {
                if internal(&mut env, i as i32, j as i32, 0) {
                    return true;
                }
            }
        }

        false
    }
}
```

### 1235. Maximum Profit in Job Scheduling

```rust
struct Job {
    start_time: i32,
    end_time: i32,
    profit: i32,
}

impl Job {
    fn new(start_time: i32, end_time: i32, profit: i32) -> Self {
        Self {
            start_time,
            end_time,
            profit,
        }
    }
}

impl Solution {
    pub fn job_scheduling(start_time: Vec<i32>, end_time: Vec<i32>, profit: Vec<i32>) -> i32 {
        let n = start_time.len();
        let mut jobs = vec![];
        for i in 0..n {
            jobs.push(Job::new(start_time[i], end_time[i], profit[i]));
        }
        jobs.sort_by(|a, b| a.end_time.cmp(&b.end_time));

        let mut dp = vec![0; n];
        dp[0] = jobs[0].profit;

        for i in 1..n {
            let job = &jobs[i];

            let (mut l, mut r) = (0, i);
            while l < r {
                let m = l + (r - l) / 2;
                if jobs[m].end_time <= job.start_time {
                    l = m + 1;
                } else {
                    r = m;
                }
            }

            let valid = l > 0 && l <= i;
            dp[i] = dp[i - 1].max(job.profit + if valid { dp[l - 1] } else { 0 });
        }

        dp[n - 1]
    }
}
```

### 446. Arithmetic Slices II - Subsequence

```rust
use std::collections::HashMap;

struct Env {
    n: usize,
    nums: Vec<i32>,
    memo: HashMap<i64, i32>,
}

impl Solution {
    pub fn number_of_arithmetic_slices(nums: Vec<i32>) -> i32 {
        fn dp(env: &mut Env, diff: i64, i: usize) -> i32 {
            let key = diff << 32 | i as i64;
            if let Some(&ret) = env.memo.get(&key) {
                return ret;
            }
            
            let mut ret = 0;
            for j in (i + 1)..env.n {
                if (env.nums[j] as i64 - env.nums[i] as i64) == diff as i64 {
                    ret += dp(env, diff, j) + 1;
                }
            }


            env.memo.entry(key).or_insert(ret);
            ret
        }

        
        let n = nums.len();
        let mut env = Env {
            n: nums.len(),
            nums: nums.clone(),
            memo: HashMap::new(),
        };

        let mut ret = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let diff = nums[j] as i64 - nums[i] as i64;
                ret += dp(&mut env, diff, j);
            }
        }

        ret
    }
}
```

### 2225. Find Players With Zero or One Losses

```rust
impl Solution {
    pub fn find_winners(matches: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let n = 100001;
        let mut in_degree = vec![0; n];
        let mut in_match = vec![false; n];
        for edge in matches {
            let (u, v) = (edge[0] as usize, edge[1] as usize);
            in_degree[v] += 1;
            in_match[u] = true;
            in_match[v] = true;
        }

        let mut ret = vec![vec![]; 2];
        for i in 0..n {
            if !in_match[i] {
                continue;
            }
            
            if in_degree[i] == 0 {
                ret[0].push(i as i32);
            } else if in_degree[i] == 1 {
                ret[1].push(i as i32);
            }
        }

        ret
    }
}
```

### 380. Insert Delete GetRandom O(1)

```rust
use std::collections::HashMap;
use rand::prelude::*;

struct RandomizedSet {
    index_map: HashMap<i32, usize>,
    nums: Vec<i32>,
}

impl RandomizedSet {
    fn new() -> Self {
        Self {
            index_map: HashMap::new(),
            nums: Vec::new(),
        }
    }

    fn insert(&mut self, val: i32) -> bool {
        if self.index_map.contains_key(&val) {
            return false;
        }

        self.nums.push(val);
        self.index_map.entry(val).or_insert(self.nums.len() - 1);

        true
    }

    fn remove(&mut self, val: i32) -> bool {
        if !self.index_map.contains_key(&val) {
            return false;
        }

        let at = self.index_map[&val];
        let last_index = self.nums.len() - 1;
        self.nums.swap(at, last_index);
        self.index_map.entry(self.nums[at]).and_modify(|v| *v = at);
        self.index_map.remove(&self.nums[last_index]);
        self.nums.remove(last_index);

        true
    }

    fn get_random(&self) -> i32 {
        let at = thread_rng().gen_range(0..self.nums.len());
        self.nums[at]
    }
}
```
