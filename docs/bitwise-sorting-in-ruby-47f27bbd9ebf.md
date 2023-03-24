# Ruby 中的按位排序

> 原文：<https://towardsdatascience.com/bitwise-sorting-in-ruby-47f27bbd9ebf?source=collection_archive---------12----------------------->

想象一下，你运营着一个 NBA 篮球招募网站。您希望根据以下三个属性对全国所有大学篮球运动员进行排名:

1.  玩家的个人评分(1-10)
2.  玩家的团队等级(1-10)
3.  玩家的会议等级(1-10)
4.  玩家的 ID(任意数字)

几个玩家的例子，排名是这样的:

```
Player 1: rating 10, team rating 10, conference rating 10
Player 2: rating 10, team rating 10, conference rating 5
Player 3: rating 10, team rating 5,  conference rating 10
Player 4: rating 10, team rating 5,  conference rating 5
Player 5: rating 5,  team rating 10, conference rating 10
Player 6: rating 5,  team rating 10, conference rating 5
```

如果你采用天真的方法，对所有球队的所有球员进行排序可能是一个昂贵的计算。举例来说，让我们假设有 1，000，000 个玩家需要评估。使用嵌套排序，我们可以构建如下内容:

```
Player = Struct.new(:id, :rating, :team_rating, :conf_rating)# setupid = 1.step
[@players](http://twitter.com/players) = 1_000_000.times.map do 
  Player.new(id.next, rand(0..10), rand(0..10), rand(0..10))
end# I want to sort by player rating descending,
# then team rating descending,
# then conference rating descending,
# then by id descending.def sort_naively
  [@players](http://twitter.com/players).group_by(&:rating).sort.reverse.map do |rb|
    rb[1].group_by(&:team_rating).sort.reverse.map do |tb|
      tb[1].group_by(&:conf_rating).sort.reverse.map do |cb|
        cb[1].group_by(&:id).sort.reverse.map do |ib|
          ib[1]
        end
      end.flatten
    end.flatten
  end.flatten
end
```

平均来说，使用这种方法对 1，000，000 条记录进行排序需要大约 2000 毫秒。考虑到记录集的大小，这还不错。然而，深度嵌套的数组结构使用了大量的内存，大约 219MB。这是 Ruby 应用程序中的一个常见问题，但是因为我们有一个 GC，所以我们通常是安全的。总而言之，对于较小的数据集来说，这是一个可用的算法，但我认为我们可以对它进行改进。

我们可以使用`Array.sort_by`进行排序，它看起来更漂亮，并传入一个值数组进行排序:

```
@players.sort_by do |p| 
  [p.rating, p.team_rating, p.conf_rating, p.id]
end.reverse
```

…但这更慢。在我的测试中，平均需要 10 秒钟。然而，它使用了 80MB 的内存，所以这是一个小的改进。在低内存环境中，这可能会派上用场，但让我们探索另一种排序方法，一种通过利用巧妙的位运算来节省时间和内存的方法。

如果你不熟悉按位运算，它们在 C 等低级语言中被广泛用于对位进行运算。在 Ruby 中，主要的按位运算符有:

```
&  -> AND
|   -> OR
^   -> XOR
~   -> NOT
<<  -> LEFT SHIFT
>>  -> RIGHT SHIFT
```

关于这些功能的详细解释可以在[这里](https://www.calleerlandsson.com/posts/rubys-bitwise-operators/)找到，这超出了本文的范围。对于这篇文章，我们只需要一个关于`| (OR)`和`<< (LEFT SHIFT)`的解释者

如果你已经了解了这么多，你可能对计算机的工作原理有了很好的了解，这些位就是 1 和 0。按位运算符对这些位进行运算。

Bitwise `| (OR)`基本上接受两个参数，第一个和第二个位，并返回一个新位，其中第一个位的值为 1，第二个位的值为 1，在缺少 1 的地方添加 1。例如:

```
integer  |  bit  | operator | other bit  |  new bit |  result
      0  | 0000  |    |     |      0001  |    0001  |       1
      1  | 0001  |    |     |      0001  |    0000  |       1
      2  | 0010  |    |     |      0001  |    0011  |       3
      3  | 0011  |    |     |      0001  |    0011  |       3
      4  | 0100  |    |     |      0001  |    0101  |       5
      5  | 0101  |    |     |      0010  |    0111  |       7
     10  | 1010  |    |     |      0101  |    1111  |      15
```

另一方面，按位`<< (LEFT SHIFT)`将位左移一定数量的位置，在该位的末尾添加 0。例如:

```
integer  |   bit  |  operator  |  number  |  result  | integer
      1  |     1  |     <<     |     1    |      10  |       2
      2  |    10  |     <<     |     1    |     100  |       4
      3  |    11  |     <<     |     1    |     110  |       6
     10  |  1010  |     <<     |     2    |  101000  |      40
```

我们可以利用这两个操作来使我们的排序算法变得更加智能。由于排序优先顺序总是`rating > team_rating > conf_rating > id`，因此，无论其他等级是什么，等级为 10 的玩家将总是排在等级为 9 的玩家之上，以此类推。在评级相同的玩家之间，具有较好团队评级的玩家将被整体评级更高，等等。

为了使用按位操作符实现这种排序，我们应该在 Player 结构中添加一个新参数`bit_rank`。新代码如下所示:

```
Player = Struct.new(
  :id, 
  :rating, 
  :team_rating, 
  :conf_rating, 
  :bit_rank  # <- new attribute
)# setupid = 1.step
[@players](http://twitter.com/players) = 1_000_000.times.map do 
  Player.new(id.next, rand(0..2), rand(0..2), rand(0..2))
end#now, calculate the bit rank for each player
@players.each do |p| 
  p.bit_rank = p.rating << 30 |
    p.team_rating << 25 |
    p.conf_rating << 20 |
    p.id
end
```

简而言之，这个新的`bit_rank`属性是一个数字(一个很大的数字),代表玩家的整体等级。我们在 30 个位置上移动评级，团队评级 25 个位置，会议评级 20 个位置，然后对所有三个位置加上 ID 运行按位`OR`。例如，一个 ID 为 1 并且在所有三个类别中都被评为 10 的玩家将拥有`11_083_448_321`的`bit_rank`。当查看该值的位表示时，这是很直观的，它是:

```
0101001010010100000000000000000000101010 01010 01010 0000000000000000000001
^     ^     ^     ^
|     |     |     |__player ID = 1
|     |     |
|     |     |__player conference rating (10 = 01010)
|     |
|     |__player team rating (10 = 01010)
|
|__player rating (10 = 01010)
```

同一个全是 5 的玩家会有一个`5_541_724_161`的`bit_rank`，当在 bit 镜头中看到时:

```
0010100101001010000000000000000000100101 00101 00101 00000000000000000001
^     ^     ^     ^                   
|     |     |     |__player ID = 1
|     |     |
|     |     |__player conference rating (5 = 00101)
|     |
|     |__player team rating (5 = 00101)
| 
|__player rating (5 = 00101)
```

…有道理。

`bit_rank`将排序优先级嵌入到自身中，其中玩家各自的评级被转移到位等级的适当区域，并且它们都被向左移动足够远，以在最后仍然按 ID 排序。

现在我们已经了解了`bit_rank`正在做什么，让我们看看运行这样一个复杂的操作需要什么代码:

```
@players.sort_by(&:bit_rank)
```

对，就是这样。因为`bit_rank`是 Struct 的一个属性，所以可以使用旧的`Symbol.to_proc` ruby magic 来调用它。如果我们在 Rails 领域，这可能是模型的一个属性，使得排序和排名非常容易。

但是，当您查看时间和内存使用情况时，使用这种方法确实很出色。

平均而言，将该属性添加到百万个玩家对象中的每一个会使设置阶段增加 200 毫秒，但是基于该属性的排序会将排序时间减少到 500 毫秒。总之，以前的`800ms build + 2000ms sort = 2800ms`操作现在变成了`1000ms build + 500ms sort = 1500ms`！我们将排序时间减少了 1300 毫秒，提高了 46%!我们添加的排序属性越多，效率也会成倍提高。

然而，内存的使用是难以置信的。为了刷新，最初的简单排序使用了 219MB 的内存，主要是因为它要排序到`10 + (10*10) + (10*10*10) + (10*10*10*10) = 11,110`个独立的、已排序的数组中。当使用`bit_rank`排序方法时，我们的操作只使用了 16MB，也就是少了 203MB，减少了 92%，因为我们没有任何嵌套数组。实际上，所有的内存占用都来自于构建 1，000，000 个玩家的数组。

使用这种方法非常适合大型数据集上的简单排序，提供了一种简单的方法来对多个值进行排序，并且在内存不足的环境中非常有用。

探索位操作符背后的想法来自参与代码的[问世，我强烈推荐参与其中。如果你还没有完成挑战，你也可以完成。](https://adventofcode.com/)

希望你喜欢这篇文章。你可以在[推特](https://twitter.com/ni3t)或 [github](http://github.com/ni3t) 上找到我。