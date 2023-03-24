# 科特林统计学简介

> 原文：<https://towardsdatascience.com/introduction-to-kotlin-statistics-cdad3be88b5?source=collection_archive---------8----------------------->

![](img/b79f6c371e3db13f22f36929412fe642.png)

Kotlin Island outside of Saint Petersburg, Russia (SOURCE: [Wikimedia](https://commons.wikimedia.org/wiki/File:Kotlin_Island_west_side.jpg#filelinks))

## 使用 Kotlin 的流畅数据科学运算符

在过去的几年里，我一直是 Kotlin 的狂热用户。但是我对 Kotlin 的癖好不仅仅是因为对语言的厌倦或者对 [JetBrains 产品](https://www.jetbrains.com/)(包括 [PyCharm，伟大的 Python IDE](https://www.jetbrains.com/pycharm/) )的热情。Kotlin 是一个更加实用的 Scala，或者我听到有人这样描述它:“傻瓜的 Scala”。它的独特之处在于，它试图不这样做，专注于实用性和工业，而不是学术实验。它吸收了迄今为止编程语言(包括 Java、Groovy、Scala、C#和 Python)的许多最有用的特性，并将它们集成到一种语言中。

我对 Kotlin 的使用是出于需要，这是我在 2017 年 KotlinConf 演讲中谈到的事情:

你可能会说“Python 是实用的”，当然，我仍然使用 Python，尤其是当我需要某些库的时候。但是，在快速发展的生产应用程序中管理 10，000 行 Python 代码可能会很困难。虽然有些人能够成功地做到这一点，并在 Python 上运行整个公司，但像 Khan Academy 这样的一些公司正在发现 Kotlin 及其静态类型的现代方法的好处。Khan Academy 写了他们从 Python 生态系统转换到 Python/Kotlin 生态系统的经历:

[](https://engineering.khanacademy.org/posts/kotlin-adoption.htm) [## 可汗学院服务器上的科特林

### 在 Khan Academy，我们使用 Python 2.7 在 Google 的应用引擎标准上运行我们的 web 应用程序。我们非常喜欢 Python

engineering.khanacademy.org](https://engineering.khanacademy.org/posts/kotlin-adoption.htm) 

Khan 还为希望学习 Kotlin 的 Python 开发人员写了一个文档:

 [## 面向 Python 开发者的 Kotlin

### 对 Kotlin 的全面介绍，面向具有 Python 或其他动态语言背景的开发人员。

khan.github.io](https://khan.github.io/kotlin-for-python-developers/) ![](img/869b3ed7f8de1aec5074e88519dd0e63.png)

但是我跑题了。在这篇文章中我想介绍的是一个我已经工作了一段时间的库，叫做 [Kotlin-Statistics](https://github.com/thomasnield/kotlin-statistics) 。它最初是一个实验，用函数式和面向对象的编程来表达有意义的统计和数据分析，同时使代码清晰直观。换句话说，我想证明在不求助于数据框架和其他数据科学结构的情况下分析 OOP/功能数据是可能的。

[](https://github.com/thomasnield/kotlin-statistics) [## 托马斯尼尔德/科特林-统计

### 科特林惯用的统计运算符。为托马斯尼尔德/科特林统计发展作出贡献

github.com](https://github.com/thomasnield/kotlin-statistics) 

以下面这段 Kotlin 代码为例，我声明了一个`Patient`类型，并包含了名字、姓氏、生日和白细胞计数。我还有一个名为`Gender`的`enum`来反映男性/女性类别。当然，我可以从文本文件、数据库或其他来源导入这些数据，但是现在我打算用 Kotlin 代码来声明它们:

```
**import** java.time.LocalDate

**data class** Patient(**val firstName**: String,
                   **val lastName**: String,
                   **val gender**: Gender,
                   **val birthday**: LocalDate,
                   **val whiteBloodCellCount**: Int) { 

    **val age get**() = 
         ChronoUnit.**YEARS**.between(**birthday**, LocalDate.now())
}**val** *patients* = *listOf*(
        Patient(
                **"John"**,
                **"Simone"**,
                Gender.**MALE**,
                LocalDate.of(1989, 1, 7),
                4500
        ),
        Patient(
                **"Sarah"**,
                **"Marley"**,
                Gender.**FEMALE**,
                LocalDate.of(1970, 2, 5),
                6700
        ),
        Patient(
                **"Jessica"**,
                **"Arnold"**,
                Gender.**FEMALE**,
                LocalDate.of(1980, 3, 9),
                3400
        ),
        Patient(
                **"Sam"**,
                **"Beasley"**,
                Gender.**MALE**,
                LocalDate.of(1981, 4, 17),
                8800
        ),
        Patient(
                **"Dan"**,
                **"Forney"**,
                Gender.**MALE**,
                LocalDate.of(1985, 9, 13),
                5400
        ),
        Patient(
                **"Lauren"**,
                **"Michaels"**,
                Gender.**FEMALE**,
                LocalDate.of(1975, 8, 21),
                5000
        ),
        Patient(
                **"Michael"**,
                **"Erlich"**,
                Gender.**MALE**,
                LocalDate.of(1985, 12, 17),
                4100
        ),
        Patient(
                **"Jason"**,
                **"Miles"**,
                Gender.**MALE**,
                LocalDate.of(1991, 11, 1),
                3900
        ),
        Patient(
                **"Rebekah"**,
                **"Earley"**,
                Gender.**FEMALE**,
                LocalDate.of(1985, 2, 18),
                4600
        ),
        Patient(
                **"James"**,
                **"Larson"**,
                Gender.**MALE**,
                LocalDate.of(1974, 4, 10),
                5100
        ),
        Patient(
                **"Dan"**,
                **"Ulrech"**,
                Gender.**MALE**,
                LocalDate.of(1991, 7, 11),
                6000
        ),
        Patient(
                **"Heather"**,
                **"Eisner"**,
                Gender.**FEMALE**,
                LocalDate.of(1994, 3, 6),
                6000
        ),
        Patient(
                **"Jasper"**,
                **"Martin"**,
                Gender.**MALE**,
                LocalDate.of(1971, 7, 1),
                6000
        )
)

**enum class** Gender {
    **MALE**,
    **FEMALE** }
```

先说一些基本的分析:所有患者的`whiteBloodCellCount`的平均值和标准差是多少？我们可以利用 Kotlin 统计数据中的一些扩展函数来快速找到这一点:

```
**fun** main() {

    **val** averageWbcc =
            *patients*.*map* **{ it**.**whiteBloodCellCount }**.*average*()

    **val** standardDevWbcc = *patients*.*map* **{ it**.**whiteBloodCellCount }** .*standardDeviation*()

    *println*(**"Average WBCC: $**averageWbcc**, 
               Std Dev WBCC: $**standardDevWbcc**"**)

    *// PRINTS: 
    // Average WBCC: 5346.153846153846, 
         Std Dev WBCC: 1412.2177503341948* }
```

我们还可以从一组项目中创建一个`DescriptiveStatistics`对象:

```
**fun** main() {

    **val** descriptives = *patients* .*map* **{ it**.**whiteBloodCellCount }** .*descriptiveStatistics

    println*(**"Average: ${**descriptives.**mean} 
         STD DEV: ${**descriptives.**standardDeviation}"**)

    */* PRINTS
      Average: 5346.153846153846   STD DEV: 1412.2177503341948
     */* }
```

然而，我们有时需要对数据进行切片，不仅是为了更详细的了解，也是为了判断我们的样本。例如，我们是否获得了男性和女性患者的代表性样本？我们可以使用 Kotlin Statistics 中的`countBy()`操作符，通过一个`keySelector`来计数一个`Collection`或`Sequence`项目，如下所示:

```
**fun** main() {

    **val** genderCounts = *patients*.*countBy* **{ it**.**gender }** *println*(genderCounts)

    *// PRINTS
    // {MALE=8, FEMALE=5}* }
```

这将返回一个`Map<Gender,Int>`，反映打印时显示`{MALE=8, FEMALE=5}` 的按性别分类的患者计数。

好吧，我们的样本有点男性化，但让我们继续。我们还可以使用`averageBy()`通过`gender`找到平均白细胞数。这不仅接受一个`keySelector` lambda，还接受一个`intSelector`来从每个`Patient`中选择一个整数(我们也可以使用`doubleSelector`、`bigDecimalSelector`等)。在这种情况下，我们从每个`Patient`中选择`whiteBloodCellCount`并通过`Gender`进行平均，如下所示。有两种方法可以做到这一点:

***方法 1:***

```
**fun** main() {

    **val** averageWbccByGender = *patients* .*groupBy* **{ it**.**gender }** .*averageByInt* **{ it**.**whiteBloodCellCount }** *println*(averageWbccByGender)

    *// PRINTS
    // {MALE=5475.0, FEMALE=5140.0}* }
```

***方法二:***

```
**fun** main() {

    **val** averageWbccByGender = *patients*.*averageBy*(
            keySelector = **{ it**.**gender }**,
            intSelector = **{ it**.**whiteBloodCellCount }** )

    *println*(averageWbccByGender)

    *// PRINTS
    // {MALE=5475.0, FEMALE=5140.0}* }
```

所以男性的平均 WBCC 是 5475，女性是 5140。

年龄呢？我们对年轻和年长的患者进行了很好的取样吗？如果你看看我们的`Patient`类，我们只有一个`birthday`可以使用，那就是 Java 8 `LocalDate`。但是使用 Java 8 的日期和时间工具，我们可以像这样导出`keySelector`中的年龄:

```
**fun** main() {

    **val** patientCountByAge = *patients*.*countBy*(
        keySelector = **{ it.age }** )

    patientCountByAge.forEach **{** age, count **->** *println*(**"AGE: $**age **COUNT: $**count**"**)
    **}** */* PRINTS: 
    AGE: 30 COUNT: 1
    AGE: 48 COUNT: 1
    AGE: 38 COUNT: 1
    AGE: 37 COUNT: 1
    AGE: 33 COUNT: 3
    AGE: 43 COUNT: 1
    AGE: 27 COUNT: 2
    AGE: 44 COUNT: 1
    AGE: 24 COUNT: 1
    AGE: 47 COUNT: 1
    */* }
```

如果您查看我们的代码输出，按年龄计数并没有太大的意义。如果我们能够按照年龄范围来计算，比如 20-29 岁、30-39 岁和 40-49 岁，那就更好了。我们可以使用`binByXXX()`操作符来做到这一点。如果我们想要通过一个`Int`值(比如年龄)来进行分类，我们可以定义一个从 20 开始的`BinModel`，并且将每个`binSize`递增 10。我们还使用`valueSelector`提供了我们是宁滨的值，即患者的年龄，如下所示:

```
**fun** main() {

    **val** binnedPatients = *patients*.*binByInt*(
            valueSelector = **{ it.age }**,
            binSize = 10,
            rangeStart = 20
    )

    binnedPatients.*forEach* **{** bin **->** *println*(bin.**range**)
        bin.**value**.*forEach* **{** patient **->** *println*(**"    $**patient**"**)
        **}
    }** }/* PRINTS:[20..29]
    Patient(firstName=Jason, lastName=Miles, gender=MALE... 
    Patient(firstName=Dan, lastName=Ulrech, gender=MALE...
    Patient(firstName=Heather, lastName=Eisner, gender=FEMALE...
[30..39]
    Patient(firstName=John, lastName=Simone, gender=MALE...
    Patient(firstName=Jessica, lastName=Arnold, gender=FEMALE...
    Patient(firstName=Sam, lastName=Beasley, gender=MALE...
    Patient(firstName=Dan, lastName=Forney, gender=MALE...
    Patient(firstName=Michael, lastName=Erlich, gender=MALE...
    Patient(firstName=Rebekah, lastName=Earley, gender=FEMALE...
[40..49]
    Patient(firstName=Sarah, lastName=Marley, gender=FEMALE...
    Patient(firstName=Lauren, lastName=Michaels, gender=FEMALE...
    Patient(firstName=James, lastName=Larson, gender=MALE...
    Patient(firstName=Jasper, lastName=Martin, gender=MALE...*/
```

我们可以使用 getter 语法查找给定年龄的 bin。例如，我们可以像这样检索年龄为 25 的`Bin`，它将返回 20-29 的 bin:

```
**fun** main() {

    **val** binnedPatients = *patients*.*binByInt*(
            valueSelector = **{ it.age }**,
            binSize = 10,
            rangeStart = 20
    )

    *println*(binnedPatients[25])
}
```

如果我们不想将项目收集到 bin 中，而是对每个项目执行聚合，我们也可以通过提供一个`groupOp`参数来实现。这允许您使用 lambda 来指定如何为每个`Bin`减少每个`List<Patient>`。以下是按年龄范围划分的平均白细胞数:

```
**val** avgWbccByAgeRange = *patients*.*binByInt*(
        valueSelector = **{ it.age }**,
        binSize = 10,
        rangeStart = 20,
        groupOp = **{ it**.*map* **{ it**.**whiteBloodCellCount }**.*average*() **}** )

*println*(avgWbccByAgeRange)/* PRINTS:
BinModel(bins=[Bin(range=[20..29], value=5300.0), 
    Bin(range=[30..39], value=5133.333333333333), 
    Bin(range=[40..49], value=5700.0)]
)
*/
```

有时，您可能希望执行多个聚合来创建各种指标的报告。这通常可以使用 Kotlin 的 let()操作符来实现。假设您想按性别找出第 1、25、50、75 和 100 个百分点。我们可以有策略地使用一个名为`wbccPercentileByGender()`的 Kotlin 扩展函数，它将选取一组患者，并按性别进行百分位数计算。然后我们可以为五个期望的百分点调用它，并将它们打包在一个`Map<Double,Map<Gender,Double>>`中，如下所示:

```
**fun** main() {

  **fun** Collection<Patient>.wbccPercentileByGender(
        percentile: Double) =
            *percentileBy*(
                percentile = percentile,
                keySelector = **{ it**.**gender }**,
                valueSelector = **{ 
                    it**.**whiteBloodCellCount**.toDouble() 
                **}** )

    **val** percentileQuadrantsByGender = *patients*.*let* **{** *mapOf*(1.0 *to* **it**.*wbccPercentileByGender*(1.0),
                25.0 *to* **it**.*wbccPercentileByGender*(25.0),
                50.0 *to* **it**.*wbccPercentileByGender*(50.0),
                75.0 *to* **it**.*wbccPercentileByGender*(75.0),
                100.0 *to* **it**.*wbccPercentileByGender*(100.0)
        )
    **}** percentileQuadrantsByGender.*forEach*(::println)
}/* PRINTS:
1.0={MALE=3900.0, FEMALE=3400.0}
25.0={MALE=4200.0, FEMALE=4000.0}
50.0={MALE=5250.0, FEMALE=5000.0}
75.0={MALE=6000.0, FEMALE=6350.0}
100.0={MALE=8800.0, FEMALE=6700.0}
*/
```

这是对科特林统计学的简单介绍。请务必阅读该项目的[自述文件](https://github.com/thomasnield/kotlin-statistics/blob/master/README.md)以查看库中更全面的可用操作符集合(它也有一些不同的工具，如[朴素贝叶斯分类器](https://github.com/thomasnield/kotlin-statistics#naive-bayes-classifier)和[随机操作符](https://github.com/thomasnield/kotlin-statistics#random-selection))。

我希望这能证明 Kotlin 在战术上的有效性，但也很强大。Kotlin 能够快速周转以进行快速的特别分析，但是您可以使用静态类型的代码，并通过许多编译时检查对其进行改进。虽然您可能认为 Kotlin 没有 Python 或 R 所拥有的生态系统，但它实际上在 JVM 上已经有了很多库和功能。随着 [Kotlin/Native](https://kotlinlang.org/docs/reference/native-overview.html) 获得牵引力，看看什么样的[数字库](https://github.com/altavir/kmath)会从 Kotlin 生态系统中崛起将会很有趣。

为了获得一些关于将 Kotlin 用于数据科学目的的资源，我在这里整理了一个列表:

[](https://github.com/thomasnield/kotlin-data-science-resources/blob/master/README.md) [## 托马斯尼尔德/科特林-数据-科学-资源

### 管理图书馆、媒体、链接和其他资源，将 Kotlin 用于数据科学…

github.com](https://github.com/thomasnield/kotlin-data-science-resources/blob/master/README.md) 

以下是我为演示 Kotlin 进行数学建模而撰写的一些其他文章:

[](/animating-the-traveling-salesman-problem-56da20b95b2f) [## 制作旅行推销员问题的动画

### 关于制作模型动画的经验教训

towardsdatascience.com](/animating-the-traveling-salesman-problem-56da20b95b2f) [](/sudokus-and-schedules-60f3de5dfe0d) [## 数独和时间表

### 用树搜索解决调度问题

towardsdatascience.com](/sudokus-and-schedules-60f3de5dfe0d)