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

### 6. Zigzag Conversion

```rust
impl Solution {
    pub fn convert(s: String, num_rows: i32) -> String {
        let mut rows = vec![String::new(); num_rows as usize];

        let mut row = 0;
        let mut delta = -1;
        for c in s.chars() {
            rows[row as usize].push(c);
            if row == 0 || row == num_rows - 1 {
                delta *= -1;
            }
            row += delta;
            row = row.clamp(0, num_rows - 1);
        }

        rows.iter().fold(String::new(), |mut ret, row| {
            ret.push_str(row);
            ret
        })
    }
}
```

### 567. Permutation in String

```rust
use std::collections::HashMap;

impl Solution {
    pub fn check_inclusion(s1: String, s2: String) -> bool {
        fn default_hash_map() -> HashMap<char, i32> {
            let mut map = HashMap::new();
            for c in 'a'..='z' {
                map.entry(c).or_default();
            }
            map
        }

        let cnt1 = s1.chars().fold(default_hash_map(), |mut cnt, c| {
            cnt.entry(c).and_modify(|e| *e += 1);
            cnt
        });
        let mut cnt2 = default_hash_map();

        let s2 = s2.chars().collect::<Vec<_>>();
        let mut i = 0;
        for j in 0..s2.len() {
            let c = s2[j];
            cnt2.entry(c).and_modify(|e| *e += 1);

            while i < j && (j - i + 1 > s1.len() || cnt2[&c] > cnt1[&c]) {
                cnt2.entry(s2[i]).and_modify(|e| *e -= 1);
                i += 1;
            }

            if j - i + 1 == s1.len() && cnt1 == cnt2 {
                return true;
            }
        }

        false
    }
}
```

### 438. Find All Anagrams in a String

```rust
use std::collections::HashMap;

impl Solution {
    pub fn find_anagrams(s: String, p: String) -> Vec<i32> {
        fn default_hash_map() -> HashMap<char, i32> {
            let mut map = HashMap::new();
            for c in 'a'..='z' {
                map.entry(c).or_default();
            }
            map
        }

        let cnt1 = p.chars().fold(default_hash_map(), |mut cnt, c| {
            cnt.entry(c).and_modify(|e| *e += 1);
            cnt
        });
        let mut cnt2 = default_hash_map();

        let mut ret = vec![];
        let s = s.chars().collect::<Vec<_>>();
        let mut i = 0;
        for j in 0..s.len() {
            let c = s[j];
            cnt2.entry(c).and_modify(|e| *e += 1);

            while i < j && (j - i + 1 > p.len() || cnt2[&c] > cnt1[&c]) {
                cnt2.entry(s[i]).and_modify(|e| *e -= 1);
                i += 1;
            }

            if j - i + 1 == p.len() && cnt1 == cnt2 {
                ret.push(i as i32);
            }
        }

        ret
    }
}
```

### 1470. Shuffle the Array

```rust
impl Solution {
    pub fn shuffle(mut nums: Vec<i32>, n: i32) -> Vec<i32> {
        let n = n as usize;
        let right_half = nums.drain(n..).collect::<Vec<_>>();

        nums.iter()
            .zip(right_half.iter())
            .fold(Vec::with_capacity(2 * n), |mut ret, (&a, &b)| {
                ret.push(a);
                ret.push(b);
                ret
            })
    }
}
```

### 904. Fruit Into Baskets


```rust
use std::collections::HashMap;

impl Solution {
    pub fn total_fruit(fruits: Vec<i32>) -> i32 {
        let mut map: HashMap<i32, i32> = HashMap::new();
        
        let mut i = 0;
        let mut ret = 0;
        for j in 0..fruits.len() {
            map.entry(fruits[j]).and_modify(|e| *e += 1).or_insert(1);

            while map.len() > 2 {
                if map[&fruits[i]] <= 1 {
                    map.remove(&fruits[i]);
                } else {
                    map.entry(fruits[i]).and_modify(|e| *e -= 1);
                }
                i += 1;
            }

            ret = ret.max(j - i + 1);
        }

        ret as i32
    }
}
```

### 45. Jump Game II

```rust
impl Solution {
    pub fn jump(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        let mut memo = vec![-1; n];

        fn f(memo: &mut [i32], nums: &[i32], i: usize) -> i32 {
            if i >= memo.len() {
                return i32::MAX >> 1;
            }
            if i == memo.len() - 1 {
                return 0;
            }
            if memo[i] != -1 {
                return memo[i];
            }

            let mut ret = i32::MAX >> 1;
            for j in 1..=nums[i] as usize {
                ret = ret.min(f(memo, nums, i + j) + 1);
            }
            
            memo[i] = ret;
            memo[i]
        }

        f(&mut memo, &nums, 0)
    }
}
```

### 2306. Naming a Company

```rust
use std::collections::HashSet;

impl Solution {
    pub fn distinct_names(ideas: Vec<String>) -> i64 {
        let mut set = vec![HashSet::new(); 26];
        for idea in &ideas {
            let k = (idea.as_bytes()[0] - b'a') as usize;
            set[k].insert(&idea[1..]);
        }

        let mut ret = 0;
        for i in 0..26 {
            for j in (i + 1)..26 {
                let mut a = 0;
                let mut b = 0;

                for &v in &set[i] {
                    if !set[j].contains(v) {
                        a += 1;
                    }
                }

                for &v in &set[j] {
                    if !set[i].contains(v) {
                        b += 1;
                    }
                }

                ret += a * b;
            }
        }

        2 * ret
    }
}
```