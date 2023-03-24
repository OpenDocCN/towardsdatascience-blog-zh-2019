# GC 笔记本故障排除

> 原文：<https://towardsdatascience.com/troubleshooting-gc-notebooks-f3531a4478e1?source=collection_archive---------26----------------------->

## 我最近花了相当多的时间搞清楚谷歌云提供的 Jupyter 笔记本服务。这是一个解决我在设置服务时不得不处理的一些烦人问题的指南。

![](img/60c8c4208a2540a0cc4a2e9f94fae2c5.png)

I have no infra guys, I have none.

如果一切顺利，创建笔记本应该是一项简单的任务。然而，我无法只使用谷歌云控制台来启动和运行它。笔记本实例一直在初始化，显然在`setting up of proxy.`失败了

如果这个简单的过程(New Instance——Open Jupyter)适合您，那么就没有必要阅读指南。然而，如果你在这里，你可能会寻找更多。

## 解决设置代理错误

一本正经的指南:

*   创建一个具有所需配置的新笔记本。
*   在完成之前，点击`Customise`并向下滚动到`Networking`并禁用`Allow proxy access when it's available`。
*   许可下，默认`Compute Engine default service account`有效。如果您稍后遇到权限错误，也可以考虑其他两个选项(`Other service account`、`Single user only`)。
*   创建笔记本实例。
*   单击实例的名称将打开配置编辑器页面。在进行以下任何配置更改之前，您需要停止该实例。
*   故障排除文档提到正确设置`Custom metadata`到`product_owners`下的`proxy-mode`。文档可以在这里[找到。然而，对我来说，工作配置是 `**proxy_mode**` **设置为** `**none**`。当您执行步骤 2 时，这是默认模式。](https://cloud.google.com/ai-platform/notebooks/docs/ssh-access#your_jupyterlab_instances_proxy-mode_metadata_setting_is_incorrect)
*   在`Network Interfaces`下，确保`External IP`不是`None`。
*   在`IAM and Admin`中，确保当前用户拥有计算访问所需的相关权限。(E.x. compute.instances.get)
*   接下来，打开`jupyter-lab`的选项将被替换为 SSH 到实例。点击`connect`获取需要在本地机器上执行的预编写的 shell 命令。
*   当尝试使用 SSH 连接时，您的实例需要正在运行。
*   如果您在尝试 SSH 时得到一个`Permission denied (publickey)`，或者类似这样的消息:

```
ssh: connect to host IP port 22: Connection refusedERROR: (gcloud.compute.ssh) [/usr/bin/ssh] exited with return code [255].
```

启动文本编辑器，用超级用户权限编辑这个文件。(以下位置适用于 MacOS)

```
sudo nano /private/etc/ssh/sshd_config
```

将现有的两行更新为:

```
Port 22
PermitRootLogin yes
```

请再次尝试关闭该实例。您现在应该能够成功地做到这一点。您可以使用下面的命令来验证`jupyter-lab`是否真的作为服务在实例`ps -ef | grep jupyter`上运行

如果您仍然得到相同的错误，运行`gcloud auth login`并基于网络完成认证过程。您也可以设置默认的项目详细信息。

前往`[http://localhost:8080](http://localhost:8080)`访问笔记本！

## 附加注释

如果您使用 GPU，请确保在配置笔记本实例时选中自动安装必要的驱动程序。

一个新启动的实例需要一些时间来安装驱动程序，所以请耐心等待。使用下面的代码片段来验证是否一切顺利，以及是否列出了 GPU。([来源](https://www.tensorflow.org/guide/gpu#setup))

```
from __future__ import absolute_import, division, print_function, unicode_literalsimport tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

如果您已经打开了日志记录(`tf.debugging.set_log_device_placement(True)`)，那么您可以在 SSHed 实例中使用下面的命令来查看日志:`journalctl -f -u jupyter`。这些日志将有助于在出现 GPU 相关问题时进行额外的故障排除。这些日志也可在控制台上的监控-串行端口#1(控制台)中找到。

再见。