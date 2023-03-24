# 仅用空间查找最低公共祖先子树和最短依赖路径

> 原文：<https://towardsdatascience.com/find-lowest-common-ancestor-subtree-and-shortest-dependency-path-with-spacy-only-32da4d107d7a?source=collection_archive---------13----------------------->

## 使用空间作为一个解决所有问题的工具

![](img/9123c63e79450e92c3b94390d52cffe2.png)

Photo by [Paula May](https://unsplash.com/@paulamayphotography?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/path?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

在之前的帖子:[如何用 spaCy 和 StanfordNLP 寻找最短依赖路径](/how-to-find-shortest-dependency-path-with-spacy-and-stanfordnlp-539d45d28239?source=your_stories_page---------------------------)中，我讲过如何用 spaCy 和 NetworkX 提取最短依赖路径(SDP)。

但是使用 NetworkX 有个问题。我们无法获得头部实体或尾部实体的索引。例如，我们有下面的句子。三联是`Convulsions->(caused_by)->fever`。但是这句话里有两个`fever`。

```
Convulsions that occur after DTaP are caused by a fever, and fever may cause headache.
```

# NetworkX 解决方案

一个解决方案是为每个令牌添加一个索引，并指定要查找哪个`fever`。

```
import spacy
import networkx as nx
nlp = spacy.load("en_core_web_sm")
doc = nlp(u'Convulsions that occur after DTaP are caused by a fever, and fever may cause headache.')# Add pair to edges
edges = []
for token in doc:
    for child in token.children:
        **edges.append(('{0}-{1}'.format(token.text, token.i),
                      '{0}-{1}'.format(child.text, child.i)))**# Construct Graph with nextworkx
graph = nx.Graph(edges)# Get the length and path
**entity1 = 'Convulsions-0'
entity2 = 'fever-9'**print(nx.shortest_path_length(graph, source=entity1, target=entity2))
print(nx.shortest_path(graph, source=entity1, target=entity2))##### output #####
3
['Convulsions-0', 'caused-6', 'by-7', 'fever-9']
```

下面的`edges`看起来像。

```
In [6]: edges
Out[6]:
[('Convulsions-0', 'occur-2'),
 ('occur-2', 'that-1'),
 ('occur-2', 'after-3'),
 ('caused-6', 'Convulsions-0'),
 ('caused-6', 'DTaP-4'),
 ('caused-6', 'are-5'),
 ('caused-6', 'by-7'),
 ('caused-6', ',-10'),
 ('caused-6', 'and-11'),
 ('caused-6', 'cause-14'),
 ('by-7', 'fever-9'),
 ('fever-9', 'a-8'),
 ('cause-14', 'fever-12'),
 ('cause-14', 'may-13'),
 ('cause-14', 'headache-15'),
 ('cause-14', '.-16')]
```

这样，我们可以确保尾部实体令牌是`fever-9`而不是`fever-12`。

这个解决方案有点麻烦，因为 NetworkX 只接受字符串类型，我们必须在字符串中包含这样的信息。

# 只有空间的解决方案

sapCy 中的[令牌类](https://spacy.io/api/token)非常强大。它在每个标记中都有索引信息。但是怎么用 spaCy 找 SDP 呢？

经过一些研究，我发现我们可以利用 [get_lca_matrix](https://spacy.io/api/doc#get_lca_matrix) 函数。

```
In [11]: doc = nlp(u"This is a test")
    ...: lca_matrix = doc.get_lca_matrix()In [12]: lca_matrix
Out[12]:
array([[0, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 2, 3],
       [1, 1, 3, 3]], dtype=int32)
```

> Doc.get_lca_matrix:计算给定`*Doc*`的最低公共祖先(lca)矩阵。返回包含祖先整数索引的 LCA 矩阵，如果没有找到共同祖先，则返回`*-1*`。

我们可以使用这个函数来查找 SDP。

```
import spacy
nlp = spacy.load("en_core_web_sm")doc = nlp(u'Convulsions that occur after DTaP are caused by a fever, and fever may cause headache.')**def get_sdp_path(doc, subj, obj, lca_matrix):
**  lca = lca_matrix[subj, obj]

  current_node = doc[subj]
  subj_path = [current_node]
  if lca != -1: 
    if lca != subj: 
      while current_node.head.i != lca:
        current_node = current_node.head
        subj_path.append(current_node)
      subj_path.append(current_node.head)current_node = doc[obj]
  obj_path = [current_node]
  if lca != -1: 
    if lca != obj: 
      while current_node.head.i != lca:
        current_node = current_node.head
        obj_path.append(current_node)
      obj_path.append(current_node.head)

  return subj_path + obj_path[::-1][1:]

# set head entity index and tail entity index
**head = 0
tail = 9**

**sdp = get_sdp_path(doc, head, tail, doc.get_lca_matrix())** print(sdp)##### output #####
[Convulsions, caused, by, fever]
```

`get_sdp_path()`可以找到头实体和尾实体之间的 SDP。我们唯一需要的是输入头部实体索引和尾部实体索引。

在`get_sdp_path()`函数中，它实际上首先找到从头实体到 LCA 令牌的 **SDP 路径**，然后找到从尾实体到 LCA 令牌的 **SDP 路径**。最后，我们将两个子树组合在一起并返回结果。

> ***查看我的其他帖子*** [***中***](https://medium.com/@bramblexu) ***同*** [***一个分类查看***](https://bramblexu.com/posts/eb7bd472/) ***！
> GitHub:***[***bramble Xu***](https://github.com/BrambleXu) ***LinkedIn:***[***徐亮***](https://www.linkedin.com/in/xu-liang-99356891/) ***博客:***[***bramble Xu***](https://bramblexu.com)

# 参考

*   [如何用 spaCy 和 StanfordNLP 找到最短依赖路径](/how-to-find-shortest-dependency-path-with-spacy-and-stanfordnlp-539d45d28239?source=your_stories_page---------------------------)
*   [get_lca_matrix](https://spacy.io/api/doc#get_lca_matrix)