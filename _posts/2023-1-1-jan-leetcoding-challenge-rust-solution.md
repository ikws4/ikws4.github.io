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
