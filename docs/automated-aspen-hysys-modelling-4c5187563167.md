# 自动化 Aspen HYSYS 建模

> 原文：<https://towardsdatascience.com/automated-aspen-hysys-modelling-4c5187563167?source=collection_archive---------24----------------------->

## 以及为什么所有过程工程师都应该担心这个问题

![](img/005368445df5b2bbe3d5d61f300776ef.png)

Aspen HYSYS snapshot of an oil and gas plant

Aspen HYSYS 是最著名的过程建模软件之一，大多数过程(化学)工程师都知道并使用它。在这篇博客中，我将回顾我们如何利用现实生活中的业务问题实现 it 自动化的 3 种方法。在我开始解释这些方法之前，让我详细解释一下为什么我认为这很重要。

![](img/84481f9051f1f2e704a01766dcfaf438.png)

Image from freepik.com

> ***重要性:*** 随着人工智能的到来和计算机技术的进步，你可能会听到这样的说法:在未来 10 年内，人类将几乎没有工作。这听起来很可怕，而且目前与你无关。让我们想想它会怎样发生。

从自动化任务开始，这在过去几年中在所有工作方面都有显著扩展。我观察到无论是大公司还是小公司都鼓励他们的员工将所有重复性的工作投入自动化模式。工程软件开始从基于电子表格转变为基于专有工具，自动将 HYSYS 直接链接到软件。作为一名经验丰富的工艺工程师，我们在设备选型和工艺计算方面投入的精力越来越少。我无法想象在不久的将来，年轻的工程师怎么可能会有我过去的那种“脏手”经历，因为他们可能一无所知，而软件将为他/她做几乎所有的事情。迟早你会发现这一点，所以让我们作为第一步开始学习如何自动化它吧！

本文的目标是记录和分享我在项目中尝试自动化 HYSYS 建模的内容。也许，这可能对和我有类似问题的人有益。下表是我在 HYSYS 的一个设备处理方案中运行各种案例的目的。需要该汇总表来确定选择哪种情况来建立设备选型的 H&MB 和操作范围。

![](img/2756eea0c9ea6ceb0602d2994aa74e10.png)

Various cases of HYSYS simulation required to complete this work

***背景:*** 在我的过程工程职业生涯中，我经常发现的一个问题是需要考虑的操作案例数量非常多。这通常来自地下储层预测的不确定性、生产混合策略的变化以及许多其他原因。例如，低输入温度(低海底温度)的情况将导致最高的原油热交换器负荷，另一方面，高海底温度的情况通常决定压缩机功率大小。作为主要的工厂设计工程师，我们将需要仔细确定所有的操作参数，以涵盖这些情况，以确保我们不会遇到任何情况，我们将不得不操作和生活，但设备不是为其设计的。

***业务案例:*** 具体到我的项目，从探井数据和自然界中各种各样的储层来看，不同气体成分的混合、凝析油 API、来自天然气平台的凝析油气比(CGR)、石油成分、来自石油平台的各种气油比、CO2 含量和海底温度的变化都会以某种方式影响设备设计。这些因素的综合结果是，如下表所示，总共需要考虑 64 个 HYSYS 模拟案例。

![](img/d8f5d956c39ccdbbb29c9a154a324562.png)

在过去，我们通常试图通过使用一些工程判断来减少案例的数量。例如，人们可能会假设高 API 凝析油情况会与高 API 油情况同时发生。然而，我认为减少案例数量可能不是正确的方法，因为成分混合的结果不容易预见，并且可能导致忽略一些重要操作案例的问题。另一方面，如果这些耗时的任务可以自动化，就可以建立一个更健壮的系统方法。

***自动化选项:*** 我在这部作品中考虑了三个选项:
1。使用石油专家解析软件
2。使用 Python GUI 控件包
3。使用 Excel Visual Basic for Applications 和 HYSYS 类型库外接程序

我选择实施的最终解决方案是#2 和#3 的组合。

对于第一个选项，PetEx RESOLVE 软件提供了 HYSYS 与其他软件(如 Excel、Pipesim 或 GAP)之间的良好交互。探索了一段时间后，我觉得这个选项看起来不适合我的问题。主要是，该软件更侧重于全局优化(从油井到加工设施的建模)和基于给定时间段预测各种生产率的结果。用 RESOLVE 构建的交互模块需要在不同的时间输入案例。虽然我可以在不同的时间设置不同的案例名称，但看起来软件内置的曲线绘制或软件之间的自动导入/导出功能对我的案例没有用。相互作用主要集中在流量、压力、温度和简单的油/水/气重力。对于除此之外的其他属性，它需要额外的编码，如下图所示。

![](img/079147c32a2287c6eefc202d6620e2f2.png)

Example of Petroleum Expert RESOLVE snapshot

应注意，在 RESOLVE 中，为了让 HYSYS 显示正确的入口和出口流，子系统中的所有流都应显示在主页上。

![](img/79501b10225d3f38b42f86365714aab6.png)

Example of Petroleum Expert RESOLVE snapshot (2)

如上所述，这个输入解算器和预测并不完全符合我的需求，所以我尝试了其他方法。

对于第二个选项，我尝试使用 Python GUI 包来控制和自动化任务。Python 是一种免费的开源且广泛使用的编程语言。我们可以安装和使用大量的免费软件包。如果你不知道如何开始使用 Python，我推荐你阅读我的第一篇博客。我发现的一个很酷的功能是可以控制鼠标和键盘到屏幕上你想要的任何地方或者输入任何东西。它像人一样直接在软件上交互控制。该选项需要安装 pyautogui 或 pywinauto 之类的软件包。安装后，我们可以使用“Ctrl”+“F”功能打开 HYSYS 中的搜索菜单，在 HYSYS 中导航并更改选定的流。然后，我们可以使用“Tab”和其他键盘快捷命令来定位并更改我们想要的案例属性。但是，我发现仅使用键盘功能来调整 HYSYS stream 并不完全符合我的需求。我需要一些鼠标控制，每次打开软件时，当我用笔记本电脑屏幕和显示器打开软件或重新启动 HYSYS 时，HYSYS 中的设备操作项目不会出现在屏幕上的相同位置。因此，我意识到这可能不是自动化 HYSYS 建模案例的最佳方法。虽然它有一些有用的功能，比如截图保存，或者在案例完成运行并收敛后保存 HYSYS 文件的能力，但是调整流组成和属性以匹配我的问题案例才是真正的问题。

![](img/d7502baf72d049375d47b26caa15ba30.png)

Snapshot of coding in python to define some custom function for this work

![](img/84ffb5bed1acbfc3be440f81708784ae.png)

HYSYS Object Navigator command screen after pressing ‘CTRL’ + ‘F’

然后，我尝试了第三个选项—使用 Excel Visual Basic for Application 添加 HYSYS 类型库。在这种方法中，我可以直接导航到每个 HYSYS 工艺流程、设备操作，并使用内置功能调整操作。它的工作效率很高，因为可以在 Excel 中清楚地定义案例，一旦提供了编码，就可以执行导入/导出功能。我发现的唯一一个遗留问题是当我们改变入口流条件时 HYSYS 模拟的收敛状态。在写这篇文章的时候，除了手动检查，我仍然找不到任何更好的解决方案。

![](img/84868eb6e4d187af9788215b6e8bb085.png)

HYSYS type 10 library that can be added in Excel >> VBA >> Tools >> References

![](img/9676befe3ebc9987fe0bde2e9020da7e.png)

Some VBA code used in this project

最后，我得出的结论是，最好的办法可能是通过以下步骤将方案 2 和方案 3 结合起来:
1 .运行 Python 代码打开 HYSYS 文件和 Excel 文件
2。在每种情况下运行 Excel 宏来改变 HYSYS 入口条件
3。手动检查仿真结果是否收敛，并且没有出现奇怪的现象
4。运行 Excel 宏将 HYSYS 流属性导出到 Excel 表
5。继续下一个 Python 代码以保存 HYSYS 截图，将 HYSYS 文件保存为新案例，并为下一个案例重复步骤#2。

![](img/0dfea0354440cd486c00220fe43913f0.png)

Final program interface that I used

在尝试了 3 种方法后，我总结了它们的利弊…

![](img/f4da440b6c584654d6dd1ba0b7cba957.png)

我已经把我所有的代码放在这个 [GIT](https://github.com/SuradechKKPB/AutomatedHYSYS) 里了。请随意阅读/使用，如果您觉得有用，请给我反馈您的意见。下次见。

参考资料:

[](https://www.chemengonline.com/using-excel-vba-process-simulator-data-extraction/) [## 使用 Excel VBA 进行流程模拟数据提取

### 加工和处理工程师可以更好地利用过程模拟的结果，通过自动导出…

www.chemengonline.com](https://www.chemengonline.com/using-excel-vba-process-simulator-data-extraction/) [](http://hysyssimulations.blogspot.com/) [## Hysys Excel 自动化

### 打开如何将 Hysys 仿真链接到 Excel？Example1.HSC。该文件在 Aspen Hysys V7.1 中生成，可能无法在…

hysyssimulations.blogspot.com](http://hysyssimulations.blogspot.com/)