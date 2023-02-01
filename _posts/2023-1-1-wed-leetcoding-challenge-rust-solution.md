---
title: "Wed LeetCoding Challenge Rust Solution"
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
