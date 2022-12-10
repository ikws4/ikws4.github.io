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

### 2256. Minimum Average Difference

```rust
impl Solution {
    pub fn minimum_average_difference(nums: Vec<i32>) -> i32 {
        let n = nums.len();
        let mut left_sum = vec![0 as i64; n + 1];
        for i in 1..=n {
            left_sum[i] = left_sum[i - 1] + nums[i - 1] as i64;
        }

        let mut ret = 0;
        let mut min_abs = i64::MAX;
        let mut right_sum = 0;
        for i in (0..n).rev() {
            let left_avg = left_sum[i + 1] / (i - 0 + 1) as i64;
            let right_avg = right_sum / ((n - 1) - (i + 1) + 1).max(1) as i64;

            let abs = (left_avg - right_avg).abs();
            if abs <= min_abs {
                ret = i;
                min_abs = abs;
            }

            right_sum += nums[i] as i64;
        }

        ret as i32
    }
}
```

### 876. Middle of the Linked List

```rust
impl Solution {
    pub fn middle_node(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut slow = head.clone();
        let mut fast = head;

        while fast.is_some() && fast.as_ref().unwrap().next.is_some() {
            slow = slow.unwrap().next;
            fast = fast.unwrap().next.unwrap().next;
        }

        slow
    }
}
```

### 328. Odd Even Linked List

```rust
impl Solution {
    pub fn odd_even_list(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut odd = None;
        let mut even = None;
        let mut odd_iter = &mut odd;
        let mut even_iter = &mut even;
        
        let mut is_odd = true;
        while let Some(mut node) = head {
            head = node.next;
            node.next = None;

            if is_odd {
                odd_iter = &mut odd_iter.insert(node).next;
            } else {
                even_iter = &mut even_iter.insert(node).next;
            }
            is_odd ^= true;
        }

        if let Some(node) = even {
            odd_iter.insert(node);
        }
        
        odd
    }
}
```

### 938. Range Sum of BST

```rust
use std::cell::RefCell;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn range_sum_bst(root: Node, low: i32, high: i32) -> i32 {
        fn f(root: &Node, low: i32, high: i32) -> i32 {
            if let Some(root) = root {
                let root = root.borrow();

                if root.val < low {
                    return f(&root.right, low, high);
                }

                if root.val > high {
                    return f(&root.left, low, high);
                }

                return f(&root.left, low, high) + f(&root.right, low, high) + root.val;
            }

            0
        }

        f(&root, low, high)
    }
}
```

### 872. Leaf-Similar Trees

```rust
use std::cell::RefCell;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn leaf_similar(root1: Node, root2: Node) -> bool {
        fn collect_leaf(root: &Node, to: &mut Vec<i32>) {
            if let Some(root) = root {
                let root = root.borrow();
                if root.left.is_none() && root.right.is_none() {
                    to.push(root.val);
                    return;
                }

                collect_leaf(&root.left, to);
                collect_leaf(&root.right, to);
            }
        }

        let mut leaf1 = vec![];
        let mut leaf2 = vec![];
        collect_leaf(&root1, &mut leaf1);
        collect_leaf(&root2, &mut leaf2);

        leaf1.cmp(&leaf2).is_eq()
    }
}
```

### 1026. Maximum Difference Between Node and Ancestor

```rust
use std::cell::RefCell;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn max_ancestor_diff(root: Node) -> i32 {
        fn f(root: &Node, mut min: i32, mut max: i32) -> i32 {
            if let Some(root) = root {
                let root = root.borrow();
                min = min.min(root.val);
                max = max.max(root.val);

                return f(&root.left, min, max).max(f(&root.right, min, max));
            }

            max - min
        }

        let v = root.as_ref().unwrap().borrow().val;
        f(&root, v, v)
    }
}
```

### 1339. Maximum Product of Splitted Binary Tree

```rust
use std::cell::RefCell;
use std::rc::Rc;

type Node = Option<Rc<RefCell<TreeNode>>>;

impl Solution {
    pub fn max_product(root: Node) -> i32 {
        fn sum(root: &Node) -> i32 {
            if let Some(root) = root {
                let root = root.borrow();

                return sum(&root.left) + sum(&root.right) + root.val;
            }

            0
        }

        fn product(root: &Node, total_sum: i32) -> (i32, i64) {
            if let Some(root) = root {
                let root = root.borrow();
                let left = product(&root.left, total_sum);
                let right = product(&root.right, total_sum);

                let s = (left.0 + right.0) + root.val;
                let p = s as i64 * (total_sum - s) as i64;

                return (s, left.1.max(right.1).max(p));
            }

            (0, 0)
        }

        let total_sum = sum(&root);

        (product(&root, total_sum).1 % (1_000_000_007)) as i32
    }
}
```
