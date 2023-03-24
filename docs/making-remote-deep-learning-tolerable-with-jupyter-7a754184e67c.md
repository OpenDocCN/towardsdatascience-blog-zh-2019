# 用 Jupyter 让远程服务器上的深度学习变得可容忍

> 原文：<https://towardsdatascience.com/making-remote-deep-learning-tolerable-with-jupyter-7a754184e67c?source=collection_archive---------22----------------------->

![](img/2b1201872223b77df57726adaea72980.png)

[Image credits](https://blog.denaeford.me/tag/frustration/)

TL；DR 如何用 Jupyter 和远程服务器设置一个基本的深度学习环境，而不会发疯。

我最近开始从事深度学习，每当我想改变一个超参数时，我几乎疯狂地使用 [vim](https://www.vim.org/) 打开我的 5000 多行代码库。在习惯了 jupyter 之后，我希望能够使用 Jupyter 处理远程服务器上的文件。

如果你不熟悉 Jupyter，这里有一个[的好帖子](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)让你开始。熟悉之后，您现在可以继续阅读，将 Jupyter 配置为从本地浏览器运行，并在服务器上处理文件。

1.  到远程服务器的 SSH
2.  使用终端管理器，它允许我们在终端内创建多个会话。如果与远程服务器的连接丢失，它还可以防止代码被截断。因此，如果我们有需要长时间运行的代码，并且您可能想让它一直运行，那么这是非常有用的。我个人用 [byobu](http://byobu.co/) 但是你可以用 [tumx](https://hackernoon.com/a-gentle-introduction-to-tmux-8d784c404340) 。在 Ubuntu 18 中，这些都是预装的。如果您运行的是旧版本，并且没有看到 byobu，您只需使用

```
sudo apt-get install byobu 
```

在那之后，你可以跑路了

```
 byobu
```

我们使用终端管理器的主要原因是，当我们运行 Jupyter 笔记本时，终端会被用完。通过 byobu，我们可以为 jupyter 创建一个会话，并在另一个选项卡上运行测试/培训。

3.接下来，我们需要将浏览器上的“localhost”链接到我们第一次启动 Jupyter 的终端上

```
jupyter notebook -- no-browser -- port=8889
```

这将迫使 Jupyter 不打开浏览器，而使用端口 8889。我们可以随心所欲地改变它。

接下来，我们将这个端口从我们的服务器链接到本地机器上的“localhost”。我们通过打电话

```
ssh -N -f -L localhost:8888:localhost:8889 username@remote-server
```

请用您的用户名替换用户名，用服务器地址替换远程服务器。我们应该会看到这样的提示

```
The Jupyter Notebook is running at: [http://localhost:8889/?token=57cba986153f10a08c0efafa91e91e3299358a287afefaafa](http://localhost:8889/?token=57cba986153f10a08c0ebb91e91e3299358a287a08a5fd61)
```

现在我们可以跑了

```
localhost:8888 
```

这将在浏览器中启动链接到远程服务器的 Jupyter 会话。

页（page 的缩写）启动本地主机时，系统可能会提示您输入代码或令牌。为此，只需从终端复制粘贴令牌 id (`/'后的字符串？token= `)

你已经准备好了！

感谢阅读！