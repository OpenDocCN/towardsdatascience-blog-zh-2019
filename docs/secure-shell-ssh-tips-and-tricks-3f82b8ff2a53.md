# 安全外壳(SSH)提示和技巧

> 原文：<https://towardsdatascience.com/secure-shell-ssh-tips-and-tricks-3f82b8ff2a53?source=collection_archive---------29----------------------->

![](img/50600e69a38f27b86796aa5064721bf3.png)

Photo by [Jefferson Santos](https://unsplash.com/@jefflssantos?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

作为一名程序员，我最喜欢的一点是我可以坐在沙滩上远程工作。但作为一名数据科学家，事情稍微有点棘手，因为我无法在本地笔记本电脑上执行任何大数据处理(我不会带着一台高规格但笨重的电脑去海滩……)。我仍然需要在拥有更多资源的远程服务器上运行我的下一个模型训练。尽管如此，不要担心，我们仍然有远程安全 shell 来拯救我们！

在这篇文章中，我将分享一些我在日常工作中使用过的 ssh 技巧。在做任何事情之前，让我们首先定义 bash 脚本的变量。

```
#!/bin/bash
user=usernameString
password=passwordString
serverAddr=IPAddressOfTheRemoteServer
serverPort=PortToBeTunnelledFromTheRemoteServer
sourceLoc=locationOfTheFolderToBeCopied
```

# 通过 SSH 仅复制更新的文件

有时候，我需要将大量文件从本地笔记本电脑复制到远程服务器。我在技术上可以使用 [SCP](https://www.ssh.com/ssh/scp/) 。然而，缺点是，如果我需要重复复制许多文件，其中大多数文件自上次操作以来没有更新过，SCP 仍然会盲目地将所有文件复制到服务器。为了减少复制所需的负载和时间，我们可以使用 rsync。Rsync 将只向服务器发送更新的文件。

```
rsync -avh -e ssh $sourceLoc $user@$serverAddr:$remoteLoc
```

# **远程运行脚本**

假设我有一个名为 *script.sh* 的 shell 脚本，我想在远程服务器上执行它。我可以把它和前缀“bash -s”一起传递给 ssh 命令。

```
ssh ${user}@${serverAddr} 'bash -s' < script.sh
```

# 建立端口隧道(仅推荐用于调试！)

有些情况下，您可能无法在服务器外部公开端口(很可能是由于安全问题)。例如，假设我在服务器的端口 5000 上运行一个 web 服务。不幸的是，这个端口不能从外部 ping 通。我可以通过 ssh 执行隧道操作，将我的本地计算机的端口 5000(技术上可以是任何选择的$localPort)链接到服务器的端口 5000(同样，任何$serverPort 都可以)。这样，只要我保持 SSH 连接，当我在本地浏览器中访问 localhost:5000 时，我就会看到来自远程服务器的 web 服务的内容。

```
ssh -N -L localhost:$localPort:localhost:$serverPort ${user}@${serverAddr}
```

# 在终端的变量中写入密码(不推荐，仅用于调试)

使用基于密码的 ssh 意味着每次我们执行 SSH 命令时，都会提示我们输入密码。如果您曾经希望有一种方法可以将密码存储在终端的临时变量中(很可能太懒了)，您会发现 [sshpass](https://linux.die.net/man/1/sshpass) 很有用。事实上，我们所需要做的就是将 *sshpass -p $password* 添加到任何普通的 ssh 命令中。下面是一个如何将 sshpass 与前面所有命令一起使用的示例。

```
sshpass -p $password rsync -avh -e ssh $sourceLoc $user@$serverAddr:$remoteLocsshpass -p $password ssh ${user}@${serverAddr} 'bash -s' < script.shsshpass -p $password ssh -N -L localhost:$localPort:localhost:$serverPort ${user}@${serverAddr}
```

# **结论**

请谨慎使用 SSH。必须认真对待 IT 安全，尤其是考虑到最近网络攻击越来越普遍。如果您在公共网络中，请确保您只连接到安全的接入点。实际上，我并不推荐远程工作，比如在海滩上(这是个玩笑！)和 ssh-ing 到工作服务器。最安全的方法是只从内部网络访问资源(或者至少激活 VPN 连接)。