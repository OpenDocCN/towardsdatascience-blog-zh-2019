# 面向数据科学家的链表简单介绍

> 原文：<https://towardsdatascience.com/a-simple-introduction-of-linked-lists-for-data-scientists-a71f0eb31d87?source=collection_archive---------17----------------------->

![](img/b7908de5b50de0245ff3c886bcb258f8.png)

Image by [WILLGARD](https://pixabay.com/users/WILLGARD-4665627/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=4626641) from [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=4626641)

## [算法面试](https://towardsdatascience.com/tagged/algorithms-interview)

## 或者说，什么是链表，为什么我需要了解它们？

算法和数据结构是数据科学不可或缺的一部分。虽然我们大多数数据科学家在学习时没有上过适当的算法课程，但它们仍然很重要。

许多公司在招聘数据科学家的面试过程中会询问数据结构和算法。

现在，许多人在这里问的问题是，问一个数据科学家这样的问题有什么用。 ***我喜欢这样描述，一个数据结构问题可以被认为是一个编码能力测试。***

我们都在人生的不同阶段进行过能力倾向测试，虽然它们不是判断一个人的完美代理，但几乎没有什么是真的。那么，为什么没有一个标准的算法测试来判断人的编码能力。

但我们不要自欺欺人，他们需要像你的数据科学面试一样的热情，因此，你可能需要花一些时间来研究算法和数据结构问题。

***这篇文章是关于快速跟踪这项研究，并以一种容易理解的方式为数据科学家解释链表概念。***

# 什么是链表？

链表只是一个非常简单的数据结构，表示一系列节点。

![](img/c1be99b56e4dedf62f69ff8b4c7525f3.png)

每个节点只是一个包含值和指向下一个节点的指针的对象。例如，在这里的例子中，我们有一个包含数据 12 并指向下一个节点 99 的节点。然后 99 指向节点 37 等等，直到我们遇到一个空节点。

![](img/c205cd10db4b45026a9064a25258167a.png)

也有双向链表，其中每个节点包含下一个和前一个节点的地址。

# 但是我们为什么需要链表呢？

![](img/fb1b88779177dce37a92866608f3753b.png)

我们都使用过 Python 中的列表。 ***但是你想过列表数据结构的插入时间吗？***

假设我们需要在列表的开头插入一个元素。在 python 列表的开头插入或删除元素需要一个 *O(n)* 复制操作。

***如果我们面临这样的问题，有很多这样的插入，我们需要一个数据结构，实际上在常数 O(1)时间内做插入，怎么办？***

有很多你可以想到的链表的实际应用。人们可以使用双向链表来实现一个系统，其中只需要前一个和下一个节点的位置。 例如 chrome 浏览器中的上一页和下一页功能。或者照片编辑器中的上一张和下一张照片。

***使用链表的另一个好处是，我们不需要对链表有连续的空间要求，即节点可以驻留在内存中的任何地方，而对于像数组这样的数据结构，节点需要分配一个内存序列。***

# 我们如何在 Python 中创建一个链表？

我们首先定义一个可以用来创建单个节点的类。

```
class Node:
    def __init__(self,val):
        self.val = val
        self.next = None
```

然后，我们使用这个类对象创建多个节点，并将它们连接在一起形成一个链表。

```
head = Node(12)
a = Node(99)
b = Node(37)head.next = a
a.next = b
```

我们有链表，从`head`开始。在大多数情况下，我们只保留变量`head`来定义我们的链表，因为它包含了我们访问整个链表所需的所有信息。

# 链接列表的常见操作或面试问题

## 1.插入新节点

在开始，我们说我们可以在一个常数 O(1)时间内在链表的开始插入一个元素。让我们看看我们能做些什么。

```
def insert(head,val):
    new_head = Node(val)
    new_head.next = head
    return new_head
```

因此，给定节点的头，我们只需创建一个`new_head`对象，并将其指针设置为指向链表的前一个头。我们只需创建一个新节点，并更新一个指针。

## **2。打印**/遍历**链表**

打印链表的元素非常简单。我们只是以迭代的方式遍历链表，直到遇到 None 节点(或者末尾)。

```
def print(head):
    while head:
        print(head.val)
        head = head.next
```

## 3.反转单向链表

这更像是链表上一个非常常见的面试[问题](https://leetcode.com/problems/reverse-linked-list)。如果给你一个链表，你能在 O(n)时间内反转这个链表吗？

```
**For Example:
Input:** 1->2->3->4->5->NULL
**Output:** 5->4->3->2->1->NULL
```

***那么我们该如何应对呢？***

我们从遍历链表开始，在将指针移动到下一个节点时反转指针方向，直到有下一个节点。

```
def reverseList(head):
    newhead = None
    while head:
        tmp = head.next
        head.next = newhead
        newhead = head
        head = tmp
    return newhead
```

# 结论

***在这篇帖子里，我谈到了链表及其实现。***

链表是数据科学面试中一些最常见问题的基础，很好地理解这些问题可能会帮助你找到理想的工作。

虽然您可以在不学习它们的情况下在数据科学中走得更远，但您可以为了一点乐趣而学习它们，也许是为了提高您的编程技能。

这里有一个小的[笔记本](https://www.kaggle.com/mlwhiz/linked-list-code-sample)给你，我把所有这些小概念都放在里面了。

我把这个问题留给你自己解决— ***实现一个函数来检查一个链表是否是回文。***

如果你想学习算法和数据结构，也可以看看我在[系列](https://towardsdatascience.com/tagged/algorithms-interview)的其他帖子。

# 继续学习

如果你想了解更多关于算法和数据结构的知识，我强烈推荐 UCSanDiego**在 Coursera 上的 [**算法专门化。**](https://click.linksynergy.com/deeplink?id=lVarvwc5BD0&mid=40328&murl=https%3A%2F%2Fwww.coursera.org%2Fspecializations%2Fdata-structures-algorithms)**

**谢谢你的阅读。将来我也会写更多初学者友好的帖子。在 [**媒体**](https://medium.com/@rahul_agarwal?source=post_page---------------------------) 关注我，或者订阅我的 [**博客**](http://eepurl.com/dbQnuX?source=post_page---------------------------) 了解他们。一如既往，我欢迎反馈和建设性的批评，可以通过 Twitter [@mlwhiz](https://twitter.com/MLWhiz?source=post_page---------------------------) 联系。**

**此外，一个小小的免责声明——这篇文章中可能会有一些相关资源的附属链接，因为分享知识从来都不是一个坏主意。**