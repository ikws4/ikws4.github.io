---
title: "Feb LeetCoding Challenge Rust Solution"
date: 2023-2-1 10:00:00 +0800
layout: post
toc: true
toc_sticky: true
tags: [leetcode, algorithm]
---

### 1071. Greatest Common Divisor of Strings

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

### 953. Verifying an Alien Dictionary

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
