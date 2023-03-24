# 编码面试问题-油藏取样

> 原文：<https://towardsdatascience.com/coding-interview-questions-reservoir-sampling-687ff5b6822b?source=collection_archive---------13----------------------->

储层采样是一种从数据流中采样元素的算法。假设您有一个非常大的数据元素流，例如:

*   6 月对 DuckDuckGo 搜索的查询
*   圣诞节期间在塞恩斯伯里购买的产品
*   白皮书指南中的姓名。

让我用这些简单的词语想象一下下面的“约会”游戏节目。参赛选手是一名单身女子，她坐在一张空椅子旁。主持人介绍第一个求婚者；未婚女子不得不邀请他和她坐在一起，成为她目前的“约会对象”。接下来，主持人介绍第二位求婚者。现在女孩可以选择是继续她现在的“约会对象”还是用新的追求者取代他。她可以使用各种手段来做出决定，比如提问或者让两个追求者以某种方式竞争。之后，主持人介绍第三个追求者，女孩可以再次选择保留或替换她现在的“约会对象”以这种方式展示了 *n* 个追求者后，游戏节目结束，女孩与她在最后保留的追求者，即节目的“赢家”进行真正的约会。

想象一下，一个参赛者仅仅通过抛硬币来决定是否交换她现在的“约会对象”。这对追求者“公平”吗，也就是说，获胜者的概率分布在所有追求者中是一致的吗？答案是否定的，因为最后几个追求者比最初几个追求者更有可能获胜。第一个求婚者是最不幸的，因为如果他想和女孩约会，他必须通过 *n-* 1 次抛硬币。最后一个求婚者的机会最大——他只需要赢得一次抛硬币的机会

```
import random

def reservoir_sampling(iterator, k):
    result = []
    n = 0
    for item in iterator:
        n = n + 1
        if len(result) < k:
            print(result)
            result.append(item)
        else:
            j = int(random.random() * n)
            if j < k:
                result[j] = item
                print('else:', result)

    return result

if __name__ == "__main__":
    stream = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    k = 5
    print(reservoir_sampling(stream, k))
```

干杯，
**扎伊德·阿里萨·阿尔马利基**

在 Udemy 上用 python 查看我们的免费课程 AWS。

> *感谢阅读。如果你喜欢这篇文章，请点击下面的按钮，这样我们就可以保持联系。*