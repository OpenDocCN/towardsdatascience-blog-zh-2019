# 我如何在 5 分钟内向数据科学家解释 OOP

> 原文：<https://towardsdatascience.com/how-i-explain-oop-to-a-data-scientist-in-5-minutes-44faf72ecca7?source=collection_archive---------12----------------------->

## 每次你使用熊猫，你都在使用一个物体…

![](img/2b7dda0fe7dde48f74fa206badd54dad.png)

当我说数据科学家时，我实际上是指那些将大部分时间花在统计和分析上的人，那些建立各种模型并使用它来解决业务问题的人。

经常从身边的数据科学家那里听到的几个关键词——SQL、R、Python。许多与大数据主题相关的术语，如 Hadoop、Spark、Pig 和 Hive，被抛入日常对话中，但我几乎没有听到他们中的任何一个谈论面向对象编程(oop)。

尽管对于数据科学家来说，知道 OOP 是什么并不是必须的，但我认为如果他们至少对 OOP 有一个粗略的概念，那还是很好的。

所以几天前，在我午休的时候，我决定花 5 分钟的时间向我的一个同事解释 OOP 的概念。

由于 Python 一直是数据科学家最喜欢的语言，所以我选择用 Python 向 LH 解释 OOP。我是这样解释的。

> 我:今天我们用 Python 来做一个只有 1 个角色的角色扮演游戏吧！LH:好吧…？
> 
> 我:你能给我写一个有名字，有生命，有法力，有等级的角色吗？LH:当然，小菜一碟。

```
name = "Jason"
health = 100
mana = 80
level = 3
```

> 我:现在我想给游戏添加一个怪物:D
> LH:嗯……好吧……

```
hero_name = "Jason"
hero_health = 100
hero_mana = 80
hero_level = 3
monster_name = "Techies"
monster_health = 20
monster_mana = 0 
monster_level = 1
```

> 我:如果这次我要 10 个怪物呢？
> LH: …？

```
hero_name = "Jason"
hero_health = 100
hero_mana = 80
hero_level = 3
monster_1_name = "Techies"
monster_1_health = 20
monster_1_mana = 0 
monster_1_level = 1
monster_2_name = "Sand King"
monster_2_health = 120
monster_2_mana = 20 
monster_2_level = 3... monster_10_name = "Chaos Knight"
monster_10_health = 150
monster_10_mana = 50
monster_10_level = 10
```

> LH:这没有意义……
> 我:让我给你看看 OOP 是如何解决这个问题的！

解决这个问题的一个面向对象的方法是使用一个**对象**——把一切都当作一个对象。注意英雄和怪物都有相同的属性。我们可以有一个普通的**职业**叫做**生物**，由英雄和怪物共享:

```
class Creature():

    def __init__(self, name, health, mana, level):
        self.name = name
        self.health = health
        self.mana = mana
        self.level = level
```

什么是课？一个类就像一个对象的蓝图。

现在，每当我们想要一个新的怪物，英雄或任何其他生物，我们不必重新命名我们的变量或保持创建多个属性。使用我们刚刚声明的**生物类**作为蓝图，我们可以轻松地创建新对象:

```
hero = Creature("Jason", 100, 80, 3)
monster_1 = Creature("Techies", 20, 0, 1)
monster_2 = Creature("Sand King", 120, 20, 3)... monster_10 = Creature("Chaos Knight", 150, 20, 3)
```

要访问对象的属性，我们可以简单地这样做:

```
hero.name = "James"
monster_1.health = 0
monster_2.level += 1
```

> LH:酷哥！这就是 OOP 的全部内容吗？我:这只是 OOP 提供的一部分，实际上还有更多。
> 
> 我:下周我再给你讲讲 OOP 吧！LH:当然可以！期待啊！