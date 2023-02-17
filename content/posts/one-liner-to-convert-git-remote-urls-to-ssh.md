+++
title = "One-liner to convert git remote urls to ssh"
date = "2023-02-16T08:37:36+08:00"
author = "ikws4"
cover = ""
tags = ["git"]
showFullContent = false
readingTime = false
Toc = false
+++

My company's self-hosting git server have some issues with the http
authentication, so I want to switch to ssh instead. Is there has any command
can help me convert those remote-urls to ssh?

<!--more-->

After searching the internet, I found the snippet [^1] to replace https url to ssh.

```bash
perl -i -p -e 's|https://(.*?)/|git@\1:|g'
```

But the project that I'm working on has a lot of submodules, so I use the `find` command to search
through the `.git` directory, and use `xargs` to pass config files into `perl`
command to do the convertion, job done!

Here is the command:

```bash
cd .git && find . -name "config" | xargs perl -i -p -e 's|https://(.*?)/|git@\1:|g'
```

### Refs

[^1]: [One-liner to replace HTTPS into SSH url in .gitmodules](https://dhoeric.github.io/2017/https-to-ssh-in-gitmodules/)
