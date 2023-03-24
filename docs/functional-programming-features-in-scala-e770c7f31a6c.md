# Scala 中的函数式编程特性

> 原文：<https://towardsdatascience.com/functional-programming-features-in-scala-e770c7f31a6c?source=collection_archive---------21----------------------->

## 数据工程

## Scala 语言的特性和模式

在过去的几个月里，我一直在探索使用 Scala 及其 eco 系统进行函数式编程。

在这篇文章中，我将重点介绍该语言的一些特性，这些特性支持为分布式系统和数据操作直观地创建功能代码。

![](img/c87a2b5e8aec59f80247e8489b7ee961.png)

Photo by [apoorv mittal](https://unsplash.com/@aprvm?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 高阶函数

根据[官方文档](https://docs.scala-lang.org/tour/higher-order-functions.html)，函数是 Scala 中的第一类对象，这意味着它们可以-

*   把另一个函数作为参数，或者…
*   返回一个函数

一个函数将另一个函数作为参数的例子是 Scala 的标准集合库中的`map()`函数。

```
val examplelist: List[Int] = List(2,9,8,14)examplelist.map(x=> x * 2) // anonymous function as argument
```

当使用标准的 Scala 集合时，链操作符也非常直观，尤其是使用中缀符号时。在下面的小代码示例中，我定义了一个从 1 到 20 的数字列表，过滤偶数，然后对它们求和。

```
(1 to 20).toList filter (_%2 == 0) reduce (_ + _)
```

`_`是通配符——在地图和过滤器的情况下，它指的是集合中的值。

# 递归

对集合中的所有项目进行操作的推荐方式是使用操作符`map`、`flatMap`或`reduce`。

如果这些操作符不能满足用例的需求，那么写一个尾递归函数来操作集合中的所有条目是非常有用的。

下面的代码示例显示了计算一个数的阶乘的尾递归函数定义。

```
import scala.annotation.tailrec@tailrec
// Factorial Function Implementation that uses Tail Recursion
def factorial(in_x: Double, prodsofar: Double = 1.0): Double = {
    if (in_x==0) prodsofar
    else factorial(in_x-1, prodsofar*in_x)
}factorial(5)
```

在 Scala 中，一个尾部递归函数，如上所述，可以被编译器优化(使用上面的`@tailrec`注释),只占用一个堆栈帧——所以即使是多层递归也不会出现 stackoverflow 错误。这是可能的开箱即用，不需要任何框架或插件。

如上所述，推荐的方式是使用集合运算符(如`reduce`等)。).作为集合 API 简单性的演示，上面的阶乘函数也可以由下面的 1-liner 实现

`(1 to 5).toList reduce (_*_)`

为了从概念上理解`reduce`，看看这个[伟大的链接](http://allaboutscala.com/tutorials/chapter-8-beginner-tutorial-using-scala-collection-functions/scala-reduce-example/)！

(还要做查看`foldLeft`、`foldRight`、`map`、`flatMap`的解释，了解一些常用的数据操作！)

![](img/0a1bcf04c17951838da0c796d1c1e388.png)

Photo by [Patrick Baum](https://unsplash.com/@gecko81de?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 案例类别

Case 类可以非常容易地实例化，不需要 boiler plate 代码，如下例所示。

```
case class BusinessTransaction(sourceaccountid: Long, targetaccountid: Long, amount: Long)// create some transactions now to demo case classes// I lend my friend
val 1_xaction = BusinessTransaction(112333L, 998882L, 20L)
// My friend pays me back 
val 2_xaction = BusinessTransaction(998882L, 112333L, 20L)
```

仅仅上面的 1 `case class ..`行做了以下有用的事情——

*   定义了 3 个不可变的值`sourceaccountid`、`targetaccountid`和`amount`
*   定义 get 方法来访问构造函数参数(例如:`1_xaction.amount`)

虽然易用性很好，但 case 类是在 Scala 中存储不可变数据实例的推荐方式。例如，在大数据应用程序中，大数据文件的每一行都可以通过 case 类建模并存储。

使用 case 类存储数据的一个例子是这里的。

在链接的示例中，函数`rawPostings`将数据文件的每一行建模为 case 类`Posting`的一个实例。最终，它返回一个类型为`RDD[Posting]`的数据集。

# 模式匹配

在 Scala 中，case 类、常规类和集合等对象可以通过模式匹配进行分解。

您可以使用模式匹配来-

*   分解对象的类型(如下例)
*   获取集合的头(例如一个`List`或一个`Seq`)

下面的代码示例展示了如何使用模式匹配来分解一个`Seq`。

```
val seq1: Seq[Int] = Seq(1,3,4,5,5)seq1 match {
    case x::y => println(s"The first element in the sequence is ${x}")
    case Nil => println("The sequence is empty")
}
```

cons 操作符(`::`)创建一个由头部(`x`)和列表其余部分(称为尾部，`y`)组成的列表。

![](img/8124602059279230e6354cae1c570a5e.png)

# 伴随物体

在 OOP 中，一个静态变量有时被用在一个类中来存储多个实例化对象的状态或属性。

但是 Scala 中没有`static`关键字。相反，我们使用的是伴随对象，也称为单例对象。伴随对象是使用`object`关键字定义的，并且与伴随对象的类具有完全相同的名称。

伴随对象可以定义不可变的值，这些值可以被类中的方法引用。

在 Scala 中有两种使用伴随对象的常见模式

*   作为工厂方法
*   提供类共有的功能(即 Java 中的静态函数)

```
// The 'val specie' straightaway defines an immmutable parameter abstract class Animal(val specie: String) {
    import Animal._ // Common Behaviour to be mixed-in to Canine/Feline classes
    def getConnectionParameters: String = Animal.connectionParameter }object Animal { // .apply() is the factory method
    def apply(specie: String): Animal = specie match {
        case "dog" => new Canine(specie)
        case "cat" => new Feline(specie)
    } val connectionParameter:String = System.getProperty("user.dir") }class Canine(override val specie: String) extends Animal(specie) {   
    override def toString: String = s"Canine of specie ${specie}"
}class Feline(override val specie: String) extends Animal(specie) {
    override def toString: String = s"Feline of specie ${specie}"
} // syntactic sugar, where we don't have to say new Animal
val doggy = Animal("dog")
val kitty = Animal("cat")doggy.getConnectionParameters
```

# 选择

大多数应用程序代码检查 Null/None 类型。Scala 对空类型的处理略有不同——使用的构造称为`Option`。最好用一个例子来说明这一点。

```
val customermap: Map[Int, String] = Map(
    11-> "CustomerA", 22->"CustomerB", 33->"CustomerC"
)customermap.get(11)         // Map's get() returns an Option[String]
customermap.get(11).get     // Option's get returns the String
customermap.get(999).get    // Will throw a NoSuchElementException
customermap.get(999).getOrElse(0) // Will return a 0 instead of throwing an exception
```

在 Python 这样的语言中，`if None:`检查在整个代码库中很常见。在 Java 中，会有 try-catch 块来处理抛出的异常。`Option` s 允许关注逻辑流程，对类型或异常检查进行最小的转移。

在 Scala 中使用`Option` s 的一个标准方式是让你的自定义函数返回`Option[String]`(或者`Int`、`Long`等等。).让我们看看`Map`结构的`get()`函数签名-

`def get(key: A): Option[B]`

使用它的一个(直观的)方法是用如下所示的`getOrElse()`函数链接它

```
// Map of IDs vs Namesval customermap: Map[Int, String] = Map(
    11-> "CustomerA", 22->"CustomerB", 33->"CustomerC"
)customermap.get(11).getOrElse("No customer found for the provided ID")
```

使用`Option` s 的一个非常有用的方法是使用像`flatMap`这样的集合操作符，它直接透明地为你处理类型。

```
// Map of IDs vs Namesval customermap: Map[Int, String] = Map(
    11-> "CustomerA", 22->"CustomerB", 33->"CustomerC"
)val listofids: List[Int] = List(11,22,33,99)listofids flatMap (id=> customermap.get(id)) //flatMap magic
```

这就是我这篇文章的全部内容！我正在研究 Akka 和 Actor 模型的并发系统。在以后的文章中，我将分享我在这个主题上的心得(以及它与 Scala 函数式编程方法的关系)。

*原载于*[*http://github.com*](https://gist.github.com/kevvo83/05d2f6cca40d9a5336722c3d52a14873)*。*