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

### 1657. Determine if Two Strings Are Close

```rust
impl Solution {
    pub fn close_strings(word1: String, word2: String) -> bool {
        let n = 26;
        let mut cnt1 = vec![0; n];
        let mut cnt2 = vec![0; n];
        for &c in word1.as_bytes() {
            cnt1[(c - b'a') as usize] += 1;
        }
        for &c in word2.as_bytes() {
            cnt2[(c - b'a') as usize] += 1;
        }

        for c in 0..n {
            if cnt1[c] != 0 && cnt2[c] == 0 || cnt2[c] != 0 && cnt1[c] == 0 {
                return false;
            }
        }

        cnt1.sort();
        cnt2.sort();

        for c in 0..n {
            if cnt1[c] != cnt2[c] {
                return false;
            }
        }

        true
    }
}
```

### 451. Sort Characters By Frequency

```rust
impl Solution {
    pub fn frequency_sort(s: String) -> String {
        let n = 128;
        let mut freq = vec![0; n];
        let mut chars = (0..n).collect::<Vec<usize>>();
        for &c in s.as_bytes() {
            freq[c as usize] += 1;
        }
        chars.sort_by(|&a, &b| freq[b].cmp(&freq[a]));

        let mut ret = "".to_string();
        for c in chars {
            let f = freq[c];
            let c = c as u8;
            for _ in 0..f {
                ret.push(c.into());
            }
        }

        ret
    }
}
```
