# Jupyter 笔记本提示和技巧

> 原文：<https://towardsdatascience.com/jupyter-notebooks-tips-and-tricks-4e995e7b1fd0?source=collection_archive---------25----------------------->

![](img/145a801e31f492c83ea586d5ebad4505.png)

# 快捷指令

*   Shift + Enter 运行单元格(代码或降价)。
*   用于在当前单元格上方插入新单元格的。
*   b 在当前单元格下插入一个新单元格。
*   m 将当前单元格改为 Markdown
*   y 将当前单元格更改为代码。
*   D + D(两次)删除选中的单元格。

# 魔法命令

*   记录单行代码或整个单元的执行时间

```
# Run the code multiple times and find mean runtime
%timeit CODE_LINE
%%timeit CODE_CELL# Run once and report
%time CODE_LINE
%%time CODE_CELL
```

*   使用`!`前缀运行一个 bash 命令行
*   `%%bash`将当前代码单元改为 bash 模式运行，基本上就是在其中编写 bash 命令

```
%%bashecho "hello from $BASH"
```

*   `%%js`，`%% html`，`%%latex`，`%%python2`，`%%python3`，...以指定语言或格式运行和呈现代码单元格。
*   当您不想在执行新代码之前担心重新加载模块时，IPython 扩展非常有用。换句话说，当你改变当前笔记本使用的某个模块中的某些东西时，改变将在你运行新的代码单元时发生，而不必担心任何事情。

```
%load_ext autoreload
%autoreload 2
```

*   jupyter 笔记本中的嵌入式 tensorboard

```
%load_ext tensorboard%tensorboard --logdir logs/model_training_logs
```

*   最后，您可以通过运行`%lsmagic`列出所有可用的魔术，这将显示当前定义的线条和单元格魔术。

# 其他的

*   有时你会有这个内存饥渴的变量，你可以通过设置它为`NONE`然后强制 gc 运行来回收内存

```
some_var = None
gc.collect()
```

*   使用`sudo service jupyter restart`来重启 jupyter，因为每隔一段时间 jupyter 就会抛出一个 fit，重启内核不足以让它回到响应状态。
*   在几乎任何函数、变量、...并运行代码单元来访问它的文档。
*   `[tqdm](https://tqdm.github.io/)`在阿拉伯语(塔卡杜姆，تقدّم)中的意思是“进步”,它与 jupyter 笔记本并不相关，但可以用来显示智能进度表。只要用`tqdm(iterable)`包住任何一个`iterable`

```
from tqdm import tqdmfor i in tqdm(range(10000)):
    pass
```

*   当您想要计算目录中文件的数量时，可以运行以下命令

```
!ls DIR_NAME | wc -l
```

# 经典笔记本扩展

在[jupyter _ contrib _ nb extensions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions.html)中有很多很棒的扩展。不过你应该用 **Jupyter lab** 代替。

首先你需要改为`jupyter_contrib_nbextensions`，然后你可以安装各种有用的扩展。

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
```

这些是我喜欢的:

*   *code _ pretify*由 *autopep8* 支持，非常适合根据 [PEP 8 风格指南](https://www.python.org/dev/peps/pep-0008/)重新格式化笔记本代码单元格中的代码

```
pip install autopep8
jupyter nbextension enable code_prettify/autopep8
```

*   拼写检查器突出显示减价单元格中拼写错误的单词，这让我避免了一些令人尴尬的错别字。

```
jupyter nbextension enable spellchecker/main
```

*   *toggle_all_line_numbers* 顾名思义，它增加了一个工具栏按钮来切换是否显示行号

```
jupyter nbextension enable toggle_all_line_numbers/main
```

*   *varInspector* 非常适合调试 python 和 R 内核。它在一个浮动窗口中显示所有已定义变量的值

```
jupyter nbextension enable varInspector/main
```

# 主题

*   dunovank/jupyter-themes 有一个我遇到的最好的主题。我试过，然后我停止使用它，因为我一直在转换环境，所以对我来说习惯股票主题是最好的。

```
pip install jupyterthemes# dark
jt -t onedork -fs 95 -altp -tfs 11 -nfs 115 -cellw 88% -T# light
jt -t grade3 -fs 95 -altp -tfs 11 -nfs 115 -cellw 88% -T# Restore default theme
jt -r
```

# Jupyter 实验室扩展

目前我只使用两个扩展

*   [*krassowski/jupyterlab-go-to-definition*](https://github.com/krassowski/jupyterlab-go-to-definition)它允许我使用 Alt + click 通过鼠标跳转到定义，或者使用 Ctrl + Alt + B 键盘。

```
jupyter labextension install @krassowski/jupyterlab_go_to_definition
```

*   [*krassowski/jupyterlab-LSP*](https://github.com/krassowski/jupyterlab-lsp)增加代码导航+悬停建议+ linters +自动补全的支持。查看他们的[支持的语言服务器的完整列表](https://github.com/krassowski/jupyterlab-lsp/blob/master/LANGUAGESERVERS.md)

```
pip install --pre jupyter-lsp
jupyter labextension install @krassowski/jupyterlab-lspconda install -c conda-forge python-language-server
```

最后，您需要重新构建 jupyter 实验室应用程序

```
jupyter lab build
```

# 主题

有很多主题，但是我列表中的第一个定制插件不是主题。这是一个顶栏扩展，可以在亮暗主题之间快速切换

```
jupyter labextension install jupyterlab-topbar-extension jupyterlab-theme-toggle
```

以下是我最近使用的一些主题

```
jupyter labextension install @telamonian/theme-darcula
jupyter labextension install @rahlir/theme-gruvbox
jupyter labextension install @kenshohara/theme-nord-extension
```

分享一下，联系一下，我的 [Twitter DMs](https://twitter.com/mgazar_) 开了！