# Python 列表从头开始！！！

> 原文：<https://towardsdatascience.com/python-lists-from-scratch-4b958eb956fc?source=collection_archive---------10----------------------->

## 让我们从基础到高级理解 Python 列表。

![](img/9f7fa66748669480fc3a8973c5f6ecc4.png)

[GangBoard](https://www.gangboard.com/blog/python-list/)

列表是 Python 中的一种数据结构，它充当同时保存或存储多个数据的容器。列表是可变的或可改变的有序元素序列。要了解更多关于 Python 列表的信息，你可以访问官方的 [Python 列表文档](https://docs.python.org/2/tutorial/datastructures.html)。你必须记住的一点是，我们经常必须使用[ ]括号来声明列表。[ ]中的元素是列表的值。例如:

```
*# Creating an empty list called "list"* list = [ ]
*# Adding the values inside the list* list = [ 1, 2, 3, 4, 5]
*# Printing the list* list**[1, 2, 3, 4, 5]**
```

这里列表的名称为“**列表**，列表值为 **1，2，3，4，5** 。列表的优点是列表中的值不必是相同的类型，这意味着:

```
*# Adding the values irrespective of their data type: Integer, String, float.*list = [1, 2, 3, “Tanu”, 4.0, 4/2]
list**[1, 2, 3, 'Tanu', 4.0, 2.0]**
```

为了检查变量是否为列表，使用如下的" **type** "方法:

```
*# Create a list as list* list = [1, 2, 3, 4, 5]*# Use type method by passing the name of the list as an arguement type(list)***list**
```

## **1)创建、访问和计算列表的长度。**

让我们创建一个名为“name”的列表，然后插入一些值，稍后我们可以在[ ]的帮助下使用索引访问列表中的元素，方法是将索引值放入列表中。然后，我们可以使用 len()方法计算列表的长度，只需将列表的名称作为参数传递给 len()方法，最后，我们还可以使用相同的 len()方法获得列表中各个元素的长度，但这次我们应该将索引位置指定为参数。

```
*# Creating a list called name* name = [“Tanu”, “Nanda”, “Prabhu”]
name**['Tanu', 'Nanda', 'Prabhu']***# Accessing the elements in the list* name[0] *# Tanu***‘Tanu’***# Calculating the length of the list.* len(name)**3***# Calculating the length of individual elements in a list* len(name[0])  * # length of "Tanu" is 4***4**
```

## **2)列表上的赋值运算符**

列表上带有 **"="** 的赋值不会产生副本。相反，赋值使两个变量指向内存中的一个链表。有时可以使用赋值操作符从一个列表复制到另一个列表。

```
*# Creating a list called name* name = [“Tanu”, “Nanda”, “Prabhu”]
name**[‘Tanu’, ‘Nanda’, ‘Prabhu’]***# Creating an empty list names* names = []
names**[]***# Using assignment operator on Lists doesn't create a copy.* names = name
*# Assigning the old list name to the new list names.* names**[‘Tanu’, ‘Nanda’, ‘Prabhu’]**
```

## 3)将两个列表添加到一个列表中。

通常追加两个列表可以使用 append()方法，但是追加也可以使用' **+** '(这里+不是加法的意思)而 **+** 可以用来将两个列表相加或者合并成一个列表，如下所示。

```
*# Creating a new list called Cars* cars = [“Mercedes”, “BMW”, “Audi”]
cars**['Mercedes', 'BMW', 'Audi']***# Creating a new list called bikes*
bikes = [“Honda”, “Yamaha”, “Aprilla”]
bikes**['Honda', 'Yamaha', 'Aprilla']***# Appending both the lists*
cars_bikes = cars + bikes
cars_bikes**['Mercedes', 'BMW', 'Audi', 'Honda', 'Yamaha', 'Aprilla']**
```

## 在列表中使用 FOR 和 IN

在 Python 中，中的**和**中的**被称为**构造**，这些构造很容易使用，当你需要迭代一个列表时就可以使用它们。**

```
*# Creating a list*
list = [1, 2, 3, 4, 5]
*# Assigning sum to 0*
sum = 0*# Using a for loop to iterate over the list*
for num in list:
  sum = sum + num
print(sum) *# 15***15**
```

## 在列表中使用 IF 和 IN。

“ **if** 和“ **in** ”构造本身是一种测试元素是否出现在列表(或其他集合)中的简单方法，它测试值是否在集合中，返回 True/False。这也适用于 Python 中的字符串字符。

```
*# Creating a list* list = [1, 2, 3, 4, 5]
if 1 in list:
  print(“True”)
else:
  print(“False”)**True**
```

## **6)Python 中的 Range 函数。**

Python 中的 range 函数用作循环结构的边界，Range 函数从 0 到 n-1 开始，不包括最后一个数字，如下所示。

```
*# Range starts 0 to n-1* for i in range(5):
 print(i)**0 
1 
2
3
4**
```

## 7)**python 中的 While 循环**

Python 有一个标准的 while 循环，它的工作方式类似于 For 循环，首先你必须初始化，然后插入一个条件，最后递增。

```
*# Initialisin i to 0* i = 0 
while i < 10:
 print(i) # 0–9
 i = i+ 1**0
1
2
3
4
5
6
7
8
9**
```

## 8)列举方法。

下面是一些最常见的列表方法，并附有示例。

**a)列表追加方法**

list.append()只是将一个元素追加到列表中，但从不返回任何内容，它只是将元素追加到现有列表中。

```
*# Creating a new list called list* list = [‘Tanu’, ‘Nanda’]
*# Before appending* list**[‘Tanu’, ‘Nanda’]***# Using the append operation on the current (“Prabhu”) is the value to be appended.* list.append(“Prabhu”)
*# After appending* list**[‘Tanu’, ‘Nanda’, ‘Prabhu’]**
```

**b)列表插入方法**

List.insert()操作用于将元素插入到指定索引位置的列表中。

```
*# Creating a new list called list* list = [‘Tanu’, ‘Nanda’]
# Before inserting
list**[‘Tanu’, ‘Nanda’]***# Insert Operation on the existing list, here the index position is 2.* list.insert(2, “Prabhu”)
# After Inserting
list**[‘Tanu’, ‘Nanda’, ‘Prabhu’]**
```

**c)列表扩展方法**

extend 方法将元素添加到新列表的末尾。它类似于 append，但是您必须将列表作为参数传递，而在 append 中不一定如此。

```
*# Creating a new list called list* list = [‘Tanu’, ‘Nanda’]
*# Before extending* list**[‘Tanu’, ‘Nanda’]***# Using extend but make sure that you put the elements in a []
# Without []* list.extend(“Prabhu”)
*# After extending* list**[‘Tanu’, ‘Nanda’, ‘P’, ‘r’, ‘a’, ‘b’, ‘h’, ‘u’]***# After []* list.extend([“Prabhu”])
list**[‘Tanu’, ‘Nanda’, ‘P’, ‘r’, ‘a’, ‘b’, ‘h’, ‘u’, ‘Prabhu’]**
```

**d)列表索引法**

list 中的 index 操作搜索列表中的元素，然后返回该元素的索引。如果该元素不在列表中，那么 index 方法返回一个值错误，告知该元素不在列表中。

```
*# Creating a new list called list* list = [‘Tanu’, ‘Nanda’]
*# Before indexing* list**[‘Tanu’, ‘Nanda’]***# After indexing, type the element that you want to index.* list.index(“Nanda”)**1***# If the element is not present then you get an error* list.index(“Mercedes”)**---------------------------------------------------------------------------****ValueErro Traceback (most recent call last)****<ipython-input-4-4f3441849c1e> in <module>()
----> 1 list.index("Mercedes")****ValueError: 'Mercedes' is not in list**
```

**e)列表删除方法**

list 中的 Remove 方法搜索元素，然后在匹配时移除列表中存在的元素。当元素不在列表中时，也会引发错误。

```
*# Creating a new list called list* list = [‘Tanu’, ‘Nanda’]
*# Before removing* list**[‘Tanu’, ‘Nanda’]***# After removing* list.remove(“Nanda”)
list**[‘Tanu’]**list.remove(‘Prabhu’)---------------------------------------------------------------------------**ValueError Traceback (most recent call last)****<ipython-input-8-9df87da8501a> in <module>()
----> 1 list.remove('Prabhu')****ValueError: list.remove(x): x not in list**
```

**f)列表排序方法**

顾名思义，sort 方法按升序对元素进行排序。这个方法不返回任何东西。

```
*# Creating a list called list*
list = [1, 4, 5, 2, 3]
*# Before sorting* list**[1, 4, 5, 2, 3]***# After sorting* list.sort()
list**[1, 2, 3, 4, 5]**
```

**g)列表反转方法**

顾名思义，reverse 方法反转整个列表。这个方法不返回任何东西。

```
*# Creating a list called list* list = [1, 2, 3, 4, 5]
*# Before Reversing* list**[1, 2, 3, 4, 5]**list.reverse()*# After the reverse* list**[5, 4, 3, 2, 1]**
```

**h)列表弹出方法**

Pop 方法移除并返回给定索引处的元素。如果省略了索引，则返回最右边的元素(大致与 append()相反)。

```
*# Creating a list called list* list = [1, 2, 3, 4, 5]
*# Before popping* list**[1, 2, 3, 4, 5]***# After popping* list.pop(1)**2**list**[1, 3, 4, 5]**
```

## **9)列出切片**

切片列表意味着减少列表元素，它也可以用来改变列表的子部分。

```
list = [‘a’, ‘b’, ‘c’, ‘d’]
list**[‘a’, ‘b’, ‘c’, ‘d’]**list[1:-1]     *## [‘b’, ‘c’]*
list**[‘a’, ‘b’, ‘c’, ‘d’]**list[0:2] = ‘z’    *## replace [‘a’, ‘b’] with [‘z’]*
list               *## [‘z’, ‘c’, ‘d’]***[‘z’, ‘c’, ‘d’]**
```

## 将字符串转换成列表

这是最重要的技术，在我们的项目中，我们经常需要将字符串转换成列表，这是一种简单的方法。下面这个例子来自 geeksforgeeks.com。这里我们使用了字符串的 split 方法，split 方法只是从字符串中分离元素，然后广播到一个列表中，如下所示。

```
*# Creating a String* String = “Tanu”
String**‘Tanu’***# Checking for the type* type(String)**str***# Using the split method to split the string into a list.* list = list(String.split(“ “)) 
list**[‘Tanu’]**type(list)**list**
```

有时当你试图将字符串转换为列表时，会出现如下所示的错误，要解决这个问题，你只需将列表的名称改为其他名称，在上面的例子中，我将列表的名称改为“list ”,因此这经常会与列表方法产生名称冲突，所以如果你使用 Google Colab 或 Jupyter notebook，就会经常出现这个错误，然后你必须重新启动内核并再次运行所有的单元格。因此，解决方案只是更改列表的名称，然后在出现此错误时重启内核。

```
**---------------------------------------------------------------------------****TypeError                                 Traceback (most recent call last)**[**<ipython-input-46-58e364821885>**](/<ipython-input-46-58e364821885>) **in <module>()
----> 1 list = list(String.split(" "))
      2 list****TypeError: 'list' object is not callable**
```

因此，以上是 Python 中非常重要的列表技术或方法。大多数例子都是从谷歌的 [Python 列表](https://developers.google.com/edu/python/lists)中引用的。我用一种简单的方式写了这个，这样每个人都能理解和掌握 Python 中的列表概念。如果你们对代码有什么疑问，评论区就是你们的了。

**谢谢。**