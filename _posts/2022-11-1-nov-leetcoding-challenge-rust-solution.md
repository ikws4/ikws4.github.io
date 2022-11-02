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
