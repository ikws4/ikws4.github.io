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
