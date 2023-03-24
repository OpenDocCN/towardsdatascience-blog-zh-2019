# 介绍谷歌番石榴

> 原文：<https://towardsdatascience.com/introducing-google-guava-f295d22319f7?source=collection_archive---------5----------------------->

![](img/a4e48d8e8e5d7a88f93fd82751d8b154.png)

## Guava 是一个开源的 Java“收藏库”库，由 Google 开发。

它提供了使用 Java 集合的实用程序。当您深入研究 Guava 时，您会注意到它是如何通过使代码简洁易读来减少编码错误、促进标准编码实践和提高生产率的。

我决定把番石榴教程分解成一系列的帖子。我们将涵盖许多 Guava 概念——*Guava 实用程序类*，*函数式编程，使用集合，以及事件总线*。

在这篇文章中，你将了解到:
1。 ***给你的 Java 项目添加番石榴。***
2。*Guava 中的基本实用程序—拆分器、地图拆分器、连接程序、地图连接程序和预处理类*

# 1.将番石榴添加到项目中

番石榴有两种口味

1.  一个用于 Java 8+ JavaRuntimeEnvironment。
2.  另一个用于 Java 7 或 Android 平台。

如果您使用的是 Maven，将下面的代码片段添加到 ***<依赖关系>…</依赖关系>*** 部分

```
<dependency>
    <groupId>com.google.guava</groupId>
    <artifactId>guava</artifactId>
    <version>**version**</version>
</dependency>
```

如果你正在使用 Gradle，添加***maven central()****到资源库*

```
*repositories {
    mavenCentral()
}*
```

*然后将下面的代码片段添加到 ***build.gradle*** 文件的依赖项部分。*

```
*dependencies {
    compile group:'com.google.guava’, name:'guava', version:**version**
}*
```

*查看[链接](https://mvnrepository.com/artifact/com.google.guava/guava)了解更多版本信息。*

*对于不使用任何项目管理工具(如 Maven 或 Gradle)的人来说—*

1.  *从 [**这里**](https://mvnrepository.com/artifact/com.google.guava/guava) **下载一罐谷歌番石榴。***
2.  *如果使用 IDE，将 JAR 文件作为外部库添加。*
3.  *如果您使用文本编辑器，将 JAR 文件添加到您的类路径中。*

# *2.基本的番石榴公用事业*

***2.1 Joiner 类**
它接受任意字符串，并用一些定界符将它们连接在一起。通过为每个元素调用 **Object.toString()** 来构建结果。*

*通常你会这样做*

```
*public String concatenateStringsWithDelimiter(List<String> strList, String delimiter) {
    StringBuilder builder = new StringBuilder();
    for (String str: strList) 
        if (str != null) 
            builder.append(str).append(delimiter);
    // To remove the delimiter from the end
    builder.setLength(builder.length() - delimiter.length());
    return builder.toString();
}*
```

*但是在 Joiner 类的帮助下，等价于上面的代码可以写成*

```
*Joiner.on(delimiter).skipNulls().join(strList);*
```

*如你所见，代码变得简洁且易于维护。同样，用番石榴找虫子也相对容易。现在，如果你想添加一个空字符串的替换，该怎么办呢？嗯，Joiner 类也处理这种情况。*

```
*Joiner.on(delimiter).useForNull(replacement).join(strList);*
```

*你可能会想，也许 Joiner 类仅限于处理字符串，但事实并非如此。因为它是作为泛型类实现的，所以也可以传递任何对象的数组、iterable 或 varargs。
Joiner 类一旦创建就不可改变。因此，它是线程安全的，可以用作静态最终变量。*

```
*public static final Joiner jnr = Joiner.on(delimiter).skipNulls();
String result = jnr.append(strList);*
```

***2.2 接合工。MapJoiner 类**
它与 Joiner 类的工作方式相同，唯一的区别是它用指定的键-值分隔符将给定的字符串连接成键-值对。*

```
*public void DemoMapJoiner() {
    // Initialising Guava LinkedHashMap Collection
    Map<String, String> myMap = Maps.newLinkedHashMap();
    myMap.put(“India”, “Hockey”);
    myMap.put(“England”, “Cricket”);
    String delimiter = “#”;
    String separator = “=”;
    String result = Joiner.on(delimiter).withKeyValueSeperator(separator).join(myMap);
    String expected = “India=Hocket#England=Cricket”;
    assertThat(result, expected);
}*
```

***2.3 Splitter 类**
Splitter 类的作用与 Joiner 类相反。它接受一个带分隔符的字符串(一个字符、一个字符串或者甚至是一个正则表达式模式),并在分隔符上分割该字符串，然后获得一个部分数组。*

*通常你会这样做*

```
*String test = “alpha,beta,gamma,,delta,,”;
String[] parts = test.split(“,”);
// parts = {“alpha”, “beta”, “gamma”, “”, “delta”, “”};*
```

*你能注意到这个问题吗？你不希望空字符串成为我的结果的一部分。所以，split()方法还有待改进。*

*但是在 Splitter 类的帮助下，与上面等价的代码可以写成*

```
*Splitter splitter = Splitter.on(“,”);
String[] parts = splitter.split(test);*
```

***split()** 方法返回一个 iterable 对象，该对象包含测试字符串中的各个字符串部分。 **trimResults()** 方法可用于删除结果中的前导和尾随空格。*

*就像 Joiner 类一样，Splitter 类一旦创建也是不可变的。因此，它是线程安全的，可以用作静态最终变量。*

***2.4 MapSplitter 类**
Splitter 类伴随着 MapSplitter。它接受一个字符串，该字符串中的键值对由某种分隔符(一个字符、一个字符串甚至是一个正则表达式模式)分隔，并以与原始字符串相同的顺序返回带有键值对的 Map 实例。*

```
*public void DemoMapSplitter() {
    String test = “India=Hocket#England=Cricket”; 
    // Initialising Guava LinkedHashMap Collection
    Map<String, String> myTestMap = Maps.newLinkedHashMap();
    myMap.put(“India”, “Hockey”);
    myMap.put(“England”, “Cricket”);
    String delimiter = “#”;
    String seperator = “=”;
    Splitter.MapSplitter mapSplitter = Splitter.on(delimiter).withKeyValueSeperator(seperator);
    Map<String, String> myExpectedMap = mapSplitter.split(test);
    assertThat(myTestMap, myExpectedMap);
}*
```

***2.5 前置条件类**
前置条件类提供了静态方法的集合来检查我们代码的状态。前提条件很重要，因为它们保证成功代码的期望得到满足。*

*比如:
1。检查空条件。你可以一直写*

```
*if (testObj == null)
        throw new IllegalArgumentException(“testObj is null”);*
```

*使用前置条件类使它更加简洁易用*

```
*checkNotNull(testObj, “testObj is null”);*
```

*2.检查有效参数。*

```
*public void demoPrecondition {
    private int age;
    public demoPrecondition(int age) {
    checkArgument(age > 0, “Invalid Age”);
        this.age = age;
    }
}*
```

*checkArgument(exp，msg)计算作为参数传递给方法的变量的状态。它计算一个布尔表达式 exp，如果表达式计算结果为 false，则抛出 IllegalArgumentException。*

*3.检查对象的状态*

```
*public void demoPrecondition {
        private String name;
        public demoPrecondition(String name) {
            this.name = checkNotNull(name, “Anonamous”);
        }

        public void Capitalize() {
            checkState(validate(), “Empty Name”);    
        }

        private bool validate() { this.name.length() > 0; }
    }*
```

*checkState(exp，msg)计算对象的状态，而不是传递给方法的参数。它计算一个布尔表达式 exp，如果表达式计算结果为 false，则抛出 IllegalArgumentException。*

*4.检查有效的元素索引*

```
*public void demoPrecondition {
        int size;
        private int [] price;

        public demoPrecondition(int size) {
            this.size = checkArgument(size > 0, “size must be greater than 0”);
           this.price = new int[this.size];
        }
        public void updateItem(int index, int value) {
            int indexToBeUpdated = checkElementIndex(index, this.size, “Illegal Index Access”);
        }
    }*
```

*谢谢你的阅读。在下一篇文章中，我们将讨论 Guava 中的函数式编程。*