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
