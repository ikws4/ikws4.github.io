---
title: "Convert https to ssh for git submodule config"
date: 2023-2-15 20:56:22 +0800
layout: post
toc: false
toc_sticky: false
tags: [git]
---

###

My company's self-hosting git server have some issues with the http
authentication, so I want to switch to ssh instead. Is there has any command
can help me convert those remote-urls to ssh? After searching the internet, I
found these snippets `perl -i -p -e 's|https://(.*?)/|git@\1:|g'`[^1] to
replace https url to ssh. The project that I'm working on has a lot of
submodules, so I use the `find` command to search through the `.git` directory,
and use `xargs` to pass config files into `perl` command to do the convertion, job done!

> One-liner

```base
cd .git && find . -name "config" | xargs perl -i -p -e 's|https://(.*?)/|git@\1:|g'
```

### Refs
[^1]: [One-liner to replace HTTPS into SSH url in .gitmodules](https://dhoeric.github.io/2017/https-to-ssh-in-gitmodules/)
