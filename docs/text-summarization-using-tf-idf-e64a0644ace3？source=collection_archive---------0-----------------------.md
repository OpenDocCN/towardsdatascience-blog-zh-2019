# NLP —使用 NLTK: TF-IDF 算法的文本摘要

> 原文：<https://towardsdatascience.com/text-summarization-using-tf-idf-e64a0644ace3?source=collection_archive---------0----------------------->

## 使用 Python ft 轻松实现。简化应用程序

在文章 [**使用 NLTK**](https://becominghuman.ai/text-summarization-in-5-steps-using-nltk-65b21e352b65)**分 5 步进行文本摘要中，我们看到了如何使用**词频算法**对文本进行摘要。**

> ****边注:**测试版用户，注册我的新公司:[***https://lessentext.com***](https://yep.so/p/lessentextai?ref=medium)**

**额外奖励:使用 [**Streamlit App**](https://share.streamlit.io/akashp1712/streamlit-text-summarization/main/app.py) 查看实际操作**

**现在，我们将使用 **Tf-IDF 算法对文本进行总结。****

**![](img/3cef39b79cc67264fe06907024a18020.png)**

**Photo by [Romain Vignes](https://unsplash.com/@rvignes?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)**

****注意**，我们在这里实现实际的算法，不使用任何库来完成大部分任务，我们高度依赖数学。**

# **术语频率*逆文档频率**

**用简单的语言来说，TF-IDF 可以定义如下:**

> ***TF-IDF 中的高权重通过该术语在整个文档集合中的高术语频率(在给定文档中)和低文档频率来达到。***

**TF-IDF 算法由两种算法相乘而成。**

## **检索词频率**

**词频(TF)是一个词在文档中出现的频率，除以有多少个词。**

****TF(t) =(术语 t 在文档中出现的次数)/(文档中的总术语数)****

## **逆文档频率**

**词频是一个词有多常见，逆文档频率(IDF)是一个词有多独特或罕见。**

****IDF(t) = log_e(文档总数/包含术语 t 的文档数)****

****例如，**
考虑包含 100 个单词的文档，其中单词 ***苹果*** 出现 5 次。 ***苹果*** 的项频率(即 TF)则为(5 / 100) = 0.05。**

**现在，假设我们有 1000 万份文档，其中 1000 份文档中出现了单词 *apple* 。然后，逆文档频率(即 IDF)计算为 log(10，000，000 / 1，000) = 4。**

**因此，TF-IDF 重量是这些量的乘积:0.05 * 4 = 0.20。**

**很简单，对吧？我们将使用相同的公式来生成摘要。**

**哦，是的，我喜欢数学。**

# **9 步实施**

****额外条件** Python3，Python 的 NLTK 库，你最喜欢的文本编辑器或 IDE**

## **1.给句子做记号**

**我们将在这里标记句子而不是单词。我们会给这些句子加权。**

## **2.创建每个句子中单词的频率矩阵。**

**我们计算每个句子中的词频。**

**结果会是这样的:**

```
{'\nThose Who Are ': {'resili': 1, 'stay': 1, 'game': 1, 'longer': 1, '“': 1, 'mountain': 1}, 'However, I real': {'howev': 1, ',': 2, 'realis': 1, 'mani': 1, 'year': 1}, 'Have you experi': {'experienc': 1, 'thi': 1, 'befor': 1, '?': 1}, 'To be honest, I': {'honest': 1, ',': 1, '’': 1, 'answer': 1, '.': 1}, 'I can’t tell yo': {'’': 1, 'tell': 1, 'right': 1, 'cours': 1, 'action': 1, ';': 1, 'onli': 1, 'know': 1, '.': 1}...}
```

**在这里，每个**句子都是关键**,**值是词频字典。****

## **3.计算词频并生成矩阵**

**我们会找到段落中每个单词的词频。**

**现在，记住 **TF，**的定义**

****TF(t) =(术语 t 在文档中出现的次数)/(文档中的总术语数)****

**在这里，文档是一个段落，术语是一个段落中的一个词。**

**现在得到的矩阵看起来像这样:**

```
{'\nThose Who Are ': {'resili': 0.03225806451612903, 'stay': 0.03225806451612903, 'game': 0.03225806451612903, 'longer': 0.03225806451612903, '“': 0.03225806451612903, 'mountain': 0.03225806451612903}, 'However, I real': {'howev': 0.07142857142857142, ',': 0.14285714285714285, 'realis': 0.07142857142857142, 'mani': 0.07142857142857142, 'year': 0.07142857142857142}, 'Have you experi': {'experienc': 0.25, 'thi': 0.25, 'befor': 0.25, '?': 0.25}, 'To be honest, I': {'honest': 0.2, ',': 0.2, '’': 0.2, 'answer': 0.2, '.': 0.2}, 'I can’t tell yo': {'’': 0.1111111111111111, 'tell': 0.1111111111111111, 'right': 0.1111111111111111, 'cours': 0.1111111111111111, 'action': 0.1111111111111111, ';': 0.1111111111111111, 'onli': 0.1111111111111111, 'know': 0.1111111111111111, '.': 0.1111111111111111}}
```

**如果我们将这个表与我们在步骤 2 中生成的表进行比较，您将看到具有相同频率的单词具有相似的 TF 分数。**

## **4.为每个单词创建一个文档表**

**这也是一个简单的表格，有助于计算 IDF 矩阵。**

**我们计算，“**多少个句子包含一个单词**”，姑且称之为文档按单词矩阵。**

**这是我们现在得到的，**

```
{'resili': 2, 'stay': 2, 'game': 3, 'longer': 2, '“': 5, 'mountain': 1, 'truth': 1, 'never': 2, 'climb': 1, 'vain': 1, ':': 8, 'either': 1, 'reach': 1, 'point': 2, 'higher': 1, 'today': 1, ',': 22, 'train': 1, 'power': 4, 'abl': 1, 'tomorrow.': 1, '”': 5, '—': 3, 'friedrich': 1, 'nietzsch': 1, 'challeng': 2, 'setback': 2, 'meant': 1, 'defeat': 3, 'promot': 1, '.': 45, 'howev': 2, 'realis': 2, 'mani': 3, 'year': 4, 'crush': 1, 'spirit': 1, 'easier': 1, 'give': 4, 'risk': 1}
```

****即**单词`resili`出现在 2 个句子中，`power`出现在 4 个句子中。**

## **5.计算 IDF 并生成矩阵**

**我们将为段落中的每个单词找到 IDF。**

**现在，记住 **IDF，**的定义**

****IDF(t) = log_e(文档总数/包含术语 t 的文档数)****

**在这里，文档是一个段落，术语是一个段落中的一个词。**

**现在得到的矩阵看起来像这样:**

```
{'\nThose Who Are ': {'resili': 1.414973347970818, 'stay': 1.414973347970818, 'game': 1.2388820889151366, 'longer': 1.414973347970818, '“': 1.0170333392987803, 'mountain': 1.7160033436347992}, 'However, I real': {'howev': 1.414973347970818, ',': 0.37358066281259295, 'realis': 1.414973347970818, 'mani': 1.2388820889151366, 'year': 1.1139433523068367}, 'Have you experi': {'experienc': 1.7160033436347992, 'thi': 1.1139433523068367, 'befor': 1.414973347970818, '?': 0.9378520932511555}, 'To be honest, I': {'honest': 1.7160033436347992, ',': 0.37358066281259295, '’': 0.5118833609788743, 'answer': 1.414973347970818, '.': 0.06279082985945544}, 'I can’t tell yo': {'’': 0.5118833609788743, 'tell': 1.414973347970818, 'right': 1.1139433523068367, 'cours': 1.7160033436347992, 'action': 1.2388820889151366, ';': 1.7160033436347992, 'onli': 1.2388820889151366, 'know': 1.0170333392987803, '.': 0.06279082985945544}}
```

**将其与 **TF 矩阵**进行比较，看看有什么不同。**

## **6.计算 TF-IDF 并生成矩阵**

**现在我们有了矩阵，下一步就很容易了。**

****TF-IDF 算法由两种算法相乘而成。****

**简单地说，我们将矩阵中的值相乘并生成新的矩阵。**

```
{'\nThose Who Are ': {'resili': 0.04564430154744574, 'stay': 0.04564430154744574, 'game': 0.03996393835210118, 'longer': 0.04564430154744574, '“': 0.0328075270741542, 'mountain': 0.05535494656886449}, 'However, I real': {'howev': 0.10106952485505842, ',': 0.053368666116084706, 'realis': 0.10106952485505842, 'mani': 0.08849157777965261, 'year': 0.07956738230763119}, 'Have you experi': {'experienc': 0.4290008359086998, 'thi': 0.2784858380767092, 'befor': 0.3537433369927045, '?': 0.23446302331278887}, 'To be honest, I': {'honest': 0.34320066872695987, ',': 0.07471613256251859, '’': 0.10237667219577487, 'answer': 0.2829946695941636, '.': 0.01255816597189109}, 'I can’t tell yo': {'’': 0.0568759289976527, 'tell': 0.15721926088564644, 'right': 0.12377148358964851, 'cours': 0.19066703818164435, 'action': 0.13765356543501517, ';': 0.19066703818164435, 'onli': 0.13765356543501517, 'know': 0.11300370436653114, '.': 0.006976758873272827}}
```

## **7.给句子打分**

**不同的算法给句子打分是不同的。这里，我们使用 Tf-IDF 句子中的单词得分来给段落加权。**

**这给出了句子表及其相应的分数:**

```
{'\nThose Who Are ': 0.049494684794344025, 'However, I real': 0.09203831532832171, 'Have you experi': 0.3239232585727256, 'To be honest, I': 0.16316926181026162, 'I can’t tell yo': 0.12383203821623005}
```

## **8.找到门槛**

**与任何总结算法类似，可以有不同的方法来计算阈值。我们在计算平均句子得分。**

**我们得到以下平均分数:**

```
0.15611302409372044
```

## **9.生成摘要**

****算法:**如果句子得分大于平均得分，则选择一个句子进行摘要。**

## **#一切尽在一处:算法集合😆**

**对于阈值，我们使用平均分数的 1.3 倍**。您可以随意使用这些变量来生成摘要。****

# **试驾？**

****原文:****

```
Those Who Are Resilient Stay In The Game Longer
“On the mountains of truth you can never climb in vain: either you will reach a point higher up today, or you will be training your powers so that you will be able to climb higher tomorrow.” — Friedrich Nietzsche
Challenges and setbacks are not meant to defeat you, but promote you. However, I realise after many years of defeats, it can crush your spirit and it is easier to give up than risk further setbacks and disappointments. Have you experienced this before? To be honest, I don’t have the answers. I can’t tell you what the right course of action is; only you will know. However, it’s important not to be discouraged by failure when pursuing a goal or a dream, since failure itself means different things to different people. To a person with a Fixed Mindset failure is a blow to their self-esteem, yet to a person with a Growth Mindset, it’s an opportunity to improve and find new ways to overcome their obstacles. Same failure, yet different responses. Who is right and who is wrong? Neither. Each person has a different mindset that decides their outcome. Those who are resilient stay in the game longer and draw on their inner means to succeed.I’ve coached many clients who gave up after many years toiling away at their respective goal or dream. It was at that point their biggest breakthrough came. Perhaps all those years of perseverance finally paid off. It was the 19th Century’s minister Henry Ward Beecher who once said: “One’s best success comes after their greatest disappointments.” No one knows what the future holds, so your only guide is whether you can endure repeated defeats and disappointments and still pursue your dream. Consider the advice from the American academic and psychologist Angela Duckworth who writes in Grit: The Power of Passion and Perseverance: “Many of us, it seems, quit what we start far too early and far too often. Even more than the effort a gritty person puts in on a single day, what matters is that they wake up the next day, and the next, ready to get on that treadmill and keep going.”I know one thing for certain: don’t settle for less than what you’re capable of, but strive for something bigger. Some of you reading this might identify with this message because it resonates with you on a deeper level. For others, at the end of their tether the message might be nothing more than a trivial pep talk. What I wish to convey irrespective of where you are in your journey is: NEVER settle for less. If you settle for less, you will receive less than you deserve and convince yourself you are justified to receive it.“Two people on a precipice over Yosemite Valley” by Nathan Shipps on Unsplash
Develop A Powerful Vision Of What You Want
“Your problem is to bridge the gap which exists between where you are now and the goal you intend to reach.” — Earl Nightingale
I recall a passage my father often used growing up in 1990s: “Don’t tell me your problems unless you’ve spent weeks trying to solve them yourself.” That advice has echoed in my mind for decades and became my motivator. Don’t leave it to other people or outside circumstances to motivate you because you will be let down every time. It must come from within you. Gnaw away at your problems until you solve them or find a solution. Problems are not stop signs, they are advising you that more work is required to overcome them. Most times, problems help you gain a skill or develop the resources to succeed later. So embrace your challenges and develop the grit to push past them instead of retreat in resignation. Where are you settling in your life right now? Could you be you playing for bigger stakes than you are? Are you willing to play bigger even if it means repeated failures and setbacks? You should ask yourself these questions to decide whether you’re willing to put yourself on the line or settle for less. And that’s fine if you’re content to receive less, as long as you’re not regretful later.If you have not achieved the success you deserve and are considering giving up, will you regret it in a few years or decades from now? Only you can answer that, but you should carve out time to discover your motivation for pursuing your goals. It’s a fact, if you don’t know what you want you’ll get what life hands you and it may not be in your best interest, affirms author Larry Weidel: “Winners know that if you don’t figure out what you want, you’ll get whatever life hands you.” The key is to develop a powerful vision of what you want and hold that image in your mind. Nurture it daily and give it life by taking purposeful action towards it.Vision + desire + dedication + patience + daily action leads to astonishing success. Are you willing to commit to this way of life or jump ship at the first sign of failure? I’m amused when I read questions written by millennials on Quora who ask how they can become rich and famous or the next Elon Musk. Success is a fickle and long game with highs and lows. Similarly, there are no assurances even if you’re an overnight sensation, to sustain it for long, particularly if you don’t have the mental and emotional means to endure it. This means you must rely on the one true constant in your favour: your personal development. The more you grow, the more you gain in terms of financial resources, status, success — simple. If you leave it to outside conditions to dictate your circumstances, you are rolling the dice on your future.So become intentional on what you want out of life. Commit to it. Nurture your dreams. Focus on your development and if you want to give up, know what’s involved before you take the plunge. Because I assure you, someone out there right now is working harder than you, reading more books, sleeping less and sacrificing all they have to realise their dreams and it may contest with yours. Don’t leave your dreams to chance.
```

****总结正文:****

```
**Have you experienced this before? Who is right and who is wrong? Neither. It was at that point their biggest breakthrough came. Perhaps all those years of perseverance finally paid off. It must come from within you. Where are you settling in your life right now? Could you be you playing for bigger stakes than you are? So become intentional on what you want out of life. Commit to it. Nurture your dreams.**
```

**瞧啊。您刚刚使用臭名昭著的 **Tf-IDF** 算法总结了文本。现在向你的朋友炫耀吧。😎**

## **接下来呢？**

1.  ****玩玩:**试试改变阈值(1.5x 到 1.3x 或者 1.8x)，看看会出什么结果。**
2.  ****引申:**你也可以引申为用“你想要的若干行/句子”来概括一段文字。**

****注** : *这是一种* ***抽取*** *文本摘要技术。***

## **在这里找到完整的代码**

**[](https://github.com/akashp1712/nlp-akash/blob/master/text-summarization/TF_IDF_Summarization.py) [## akashp1712/nlp-akash

### 自然语言处理注释和实现— akashp1712/nlp-akash

github.com](https://github.com/akashp1712/nlp-akash/blob/master/text-summarization/TF_IDF_Summarization.py) 

> **一个小小的请求:**请报名参加我的新创业:[***【https://lessentext.com】***](https://yep.so/p/lessentextai?ref=medium)**并提前提供反馈！****