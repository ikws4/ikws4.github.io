+++
title = "How to randomly pick an element in an array base on weights?"
date = "2021-10-11T11:08:32+08:00"
cover = ""
tags = ["random algorithm", "algorithm"]
p5 = true
Toc = true
+++

<!--more-->

Lower the probability, higher the chance its probability will under the random
variable right? But what we want is that the higher the probability, the more
likely it is to be chosen. So when `arr[i] < r` we need pick another one.

```java
int pick(int[] arr) {
  int i = 0;
  int r = random(0, sum(arr));

  while (arr[i] < r) {
    r -= arr[i++];
  }

  return i;
}
```

Another version that use cumulative probability, it is equivalent to the one
above, but I think this one is easier to understand. Here is an example I took
from [this](https://stackoverflow.com/questions/17250568/randomly-choosing-from-a-list-with-weighted-probabilities) StackOverflow answer.

```text
Element    A B C D
Frequency  1 4 3 2
Cumulative 1 5 8 10

Pick A, if `r` is in the range of (0, 1]
Pick B, if `r` is in the range of (1, 5]
Pick C, if `r` is in the range of (5, 8]
Pick D, if `r` is in the range of (8, 10]
```

How often will `r` in (0, 1], oh well the answer is 1, right? So what about (1,
5]? You are right 4! That is what we want. We can use the interval that random
variable `r` will fall into as the measurement to randomly pick an element
base on their probability.

```java
int pick(int[] arr) {
  int i = 0;
  int r = random(0, sum(arr));
  int acc = arr[i];

  while (acc < r) {
    acc += arr[++i];
  }

  return i;
}
```

Below is a demo that use above algorithm to illustrate the process. Click
`PLAY` to run the algorithm, you can use the slider to control each
animals weights ( higher the weighs means the higher the chance it will be
pick.)

{{< p5sketch name="random_visualizer" >}}
