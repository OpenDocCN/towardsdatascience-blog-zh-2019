# 已解决:一个 DbContext 不能将多个 DbSet 指向同一个对象

> 原文：<https://towardsdatascience.com/solved-one-dbcontext-cant-point-to-multiple-dbset-to-the-same-object-a7231a72ff1d?source=collection_archive---------17----------------------->

## 在本指南中，我们将解决实体框架不允许我们解决的一个主要问题

![](img/da038eab6d9d9c57070a58261cb483bb.png)

Photo by [Olav Ahrens Røtne](https://unsplash.com/@olav_ahrens?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/collections?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# 问题是

最近我在做一个项目，需要我创建多个表，其中每个表包含相同数量的相同数据类型的列，即每个表必须遵循相同的结构。众所周知，**实体框架 6** 期望我们用不同的名字创建单独的模型(类),所以从开发人员的角度来看，用不同的名字多次创建相同的模型并不是一个好方法，当我们同时考虑维护时，这是如此的无聊和复杂。

让我们来看看下面的代码摘录

从给定的代码中可以看出，我创建了一个名为 ***CommanEntity*** 的类，它包含两个属性，即 ***id*** 和 ***name*** ，然后我在 MyDbContext 中继承了 DbContext 类，最后创建了两个显然名称不同的表对象。所以当你运行上面的代码时，你会得到异常。现在我们来看一下解决方案。

我想我们大多数人会认为这是一个非常简单和常见的问题，我们希望他们能提供简单的解决方案，但他们提供的解决方案非常复杂和冗长，所以我们不想深入讨论。我们将用魔法解决它。

# 神奇的戏法

首先，我创建了名为 ***GetTable()*** 的用户定义方法，并将表名作为我们之前创建的参数传递，然后创建了 *DbSet* 对象作为通用模型。最后我调用了 ***First()*** 方法，用简单的 ***Sql*** 命令作为参数，在 ***MyDbContext*** 对象上使用表名，就这样。

# **结论**

我们已经解决了 entity framework 6 的问题，其中在 DbSet 对象上用不同的名称创建多个表是不受官方支持的。