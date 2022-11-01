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
