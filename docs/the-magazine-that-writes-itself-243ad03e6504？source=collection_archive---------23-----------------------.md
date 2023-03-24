# 自己写的杂志

> 原文：<https://towardsdatascience.com/the-magazine-that-writes-itself-243ad03e6504?source=collection_archive---------23----------------------->

## 我们让一个人工智能写下未来，这是它要说的话和我们建造它的步骤。

凯瑟琳·劳伦斯&安·基德

![](img/a17e3cd64b54185f5e071c0032a014d5.png)

A screenshot of MONTAG magazine’s homepage at montag.wtf

**TL；DR(或者如果你不关心它是如何建成的，只想看一些 AI 写的关于未来的文章):是的，杂志可以自己写，但它仍然需要一个人类出版商。在** [**查看 montag.xyz**](https://montag.xyz)

这个[拼写](https://spell.run)启用的项目的原始材料是 [MONTAG Magazine](https://www.montag.wtf) ，这是一个关于技术将如何改变我们生活方式的故事的在线和印刷杂志。MONTAG 成立于 2017 年，是 Grover 的创意，Grover 是一家总部位于柏林的初创公司，提供灵活的月租消费电子产品。我们启动了 MONTAG，让我们的目光超越今天可用的技术设备，更深入地思考技术对社会的影响。

人们经常会问，为什么我们以如此古老的格式发布如此具有前瞻性的内容——印刷媒体在数字时代有什么地位？我们经常问自己，未来我们将如何与媒体和创意产业中的技术合作？**与人工智能的互动将如何影响人类的创造力？**

这个项目旨在回答其中的一些问题，特别是随着人工智能能力的快速发展，如 OpenAI 的 [GPT-2](https://openai.com/blog/better-language-models/) 。在 MONTAG 第 3 期[“编码创意”](https://www.montag.wtf/issue3/)中讨论了神经网络生成的艺术和文本的主题后，我决定尝试一下，尝试使用神经网络来制作一本自己写作的杂志。

*   第一步，使用 shell 脚本和 Python 进行 web 抓取
*   第二步，使用[法术训练 Max Woolfe 的 Simple-gp T2 . run](https://spell.run)
*   第三步，提取文本样本

# 第一步:清理文章文本

大多数数据科学家会告诉你，他们花了更多的时间通过收集、组织和清理数据来创建良好的数据集，然后才能做任何有趣或有用的事情。不幸的是，我们训练神经网络所需的文本数据也是如此。

作为该杂志的作者之一，理论上我可以进入我们的内容管理工具(MONTAG 使用 [Ghost](https://ghost.org/) ，一个非常酷的开源发布平台)手动复制所有文本，但从网站上删除文本数据会更容易。此外，如果我想在新的文章写出来后重复这个过程，或者用它来写另一个在线博客或杂志，它将更容易被复制。

为了从网站上收集数据，我在编写 shell 脚本和 Python 脚本之间切换，前者要求我的计算机下载 HTML 页面，后者为我读取这些 HTML 页面并找到我需要的信息。使用这个过程，我只需要找到所有的文章链接，将每篇文章下载为一个 HTML 页面，然后将所有的 HTML 页面数据转换为一个文本文件。我将从一开始就为此使用 Spell，这样所有的文本数据都位于我们将进行处理的同一个位置，并且我的计算机不会充满 HTML 文件——如果我们正在抓取一个大得多的文本语料库，这可能是一个问题。

首先，我需要注册并登录一个[法术](https://spell.run)账户(更多关于如何做的[在这里](https://spell.run/docs/quickstart))。然后，我将安装拼写命令行界面，并使用我在注册时创建的用户名和密码登录:

```
$ pip install spell
$ spell login
```

然后，我将创建一个文件夹来存放我为项目编写的任何脚本。

```
$ mkdir montagproject
$ cd montagproject
```

我们暂时将该文件夹保留为空，稍后在本文中添加我们的脚本。

查看我想要抓取的站点，在本例中是 [www.montag.wtf](http://www.montag.wtf) ，我看到底部的分页显示“第 1 页，共 27 页”，这很好，现在我知道我需要抓取多少页了。在这里，我将编写一个脚本，它将遍历 montag.wtf 上的每一页，并将其存储在我的法术运行中。由于这个脚本不需要 GPU，我们可以在 Spell 的默认 CPU 机器上运行它。

```
$ spell run 'for p in `seq 1 27`; do curl[https://www.montag.wtf/page/$p/](https://www.montag.wtf/page/$p/) -o montagpage$p.html; done'
Could not find a git repository, so no user files will be available. Continue anyway? [Y/n]:y💫 Casting spell #1...
✨ Stop viewing logs with ^C
✨ Machine_Requested... done
✨ Building... done
✨ Run is running
% Total % Received % Xferd Average Speed Time Time Time Current
.
.
.
✨ Saving... done
✨ Pushing... done
🎉 Total run time: 30.39674s
🎉 Run 1 complete
```

瞧，几秒钟后我就有了 27 个 HTML 文件。我们可以使用`ls`命令查看它们。如果您想下载任何一个文件(或所有文件),您可以使用`spell cp`命令，该命令适用于目录路径或文件路径:

```
$ spell ls runs/1
35 May 21 18:48 montagpage1.html
12228 May 21 18:48 montagpage10.html
12166 May 21 18:48 montagpage11.html
12224 May 21 18:48 montagpage12.html$ spell cp runs/1/montagpage1.html
✔ Copied 1 files
```

这些文章的链接藏在这几页的某个地方。我可以查看这个文件夹中的第一个 HTML 页面，看看我将如何使用 [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) 提取文章的链接，并准备另一个脚本来抓取文章页面。

看起来每个文章链接都在一个名为`homepage-post-info`的 [div 元素](https://www.w3schools.com/tags/tag_div.asp)中，所以首先我们要求 BeautifulSoup 找到所有这些链接。然后，对于它找到的每个对象，我们要求它找到里面所有的锚标记，并为每个锚标记提取`href`，这是我们到文章的相对链接。下一部分为我们的新 shell 脚本打开文件以下载所有的文章 HTML 页面，并添加一行以将每个文章页面输出到一个新的 HTML 页面。

下面是提取文章链接并创建一个新的 shell 脚本来下载它们的 Python 脚本。让我们将代码复制到我们的`montagproject`文件夹中名为`montagarticlesoup.py`的文件中。

```
from urllib import urlopen
from bs4 import BeautifulSoup
import os
import syscount = 1
montagpages = os.listdir(sys.argv[1])
path_to_pages = sys.argv[1]for htmlFile in montagpages:
 textPage = urlopen(path_to_pages + "/" + str(htmlFile))
 text = textPage.read()
 soup = BeautifulSoup(text, features="html.parser")
 articles = soup.find_all(class_="homepage-post-info")
 for x in articles:
 links = x.find_all("a")
 for a in links:
  link = a['href']
  f = open("getmontagarticles.sh", "a")
  f.write("curl [https://www.montag.wtf](https://www.montag.wtf)" + link + " -o montagarticles" + str(count) + ".html\n") 
  count += 1
  f.close()
```

要运行这个脚本，我们首先需要将我们的文件`commit`到 git。如果您在本地运行，这是不必要的，但是因为我们要在 Spell 上运行，所以我们需要在 git 存储库中保存我们的文件。我们还需要`init`git repo，因为这是我们第一次使用它。

```
$ git init && git add * && git commit -m "add montagarticlesoup.py"
```

然后运行我们的脚本，使用下面的命令。您必须传入一个命令行参数，即我们的 Montag 页面的目录名。该参数紧跟在 python 文件的名称之后。

**别忘了把下面的** `**runs/1**` **换成上面你跑的次数！**

如果您忘记了运行编号，您可以在`web.spell.run/runs`或使用`spell ps`查找。

```
$ spell run -m runs/1:pages --pip urllib3 --pip beautifulsoup4 --python2 "python montagarticlesoup.py pages"...
✨ Saving... done
✨ Pushing... done
🎉 Total run time: 23.557325s
🎉 Run 2 complete
```

在 Spell 上运行整个项目的另一个好处是，我不需要在我的电脑上安装任何东西，也不需要使用虚拟环境来收集所有的依赖项。我使用法术`--pip`标志来添加我需要的包。

我还需要使用`-m`标志从前面的命令中挂载我的文件。如果您还记得的话，所有的 html 文件都在 run `1`的输出中(或者，如果您已经玩了一会儿法术，不管您执行第一步时的运行编号是什么)。然后，我表示希望在新的运行中将这些文件挂载到名为`pages`的目录中。最后，我将把目录名 pages 作为我的命令行参数传入，这样我的脚本就知道在哪里查找文件。

跑步结束后，我们会看到这些步骤。如果我们`ls`运行，我们将看到我们的`getmontagarticles.sh`文件——太好了！

```
$ spell ls runs/2
14542 May 22 14:50 getmontagarticles.sh
```

让我们运行新脚本来获取所有文章。在运行脚本之前，您可能需要更改它的权限，我已经用下面的`chmod`命令完成了。不要忘记将`2`替换为您跑步的号码。

```
$ spell run -m runs/2:script "chmod +x script/getmontagarticles.sh; script/getmontagarticles.sh"...🎉 Run 3 complete
```

我们应该会发现，我们的运行输出中现在有超过 100 个 HTML 页面。我们现在需要从大量的文章中获取甜蜜的文本数据来训练神经网络。

在我们带着所有的文本数据自由回家之前，我们还得做一碗汤。这一次，我们将浏览所有这些文章，提取所有文本内容，并将其写入一个名为`montagtext.txt`的文本文件——没有链接，没有图像，没有格式，没有问题。姑且称之为`montagtextsoup.py`:

```
from urllib import urlopen
from bs4 import BeautifulSoup
import os
import sysarticlepages = os.listdir(sys.argv[1])
path_to_pages = sys.argv[1]for htmlFile in articlepages:
 if ".html" in htmlFile:
  textUrl = path_to_pages + "/" + htmlFile
  textPage = urlopen(textUrl)
  text = textPage.read()
  soup = BeautifulSoup(text, features="html.parser")
  articletext = soup.find_all(class_="content-text")
  for element in articletext:
   words = element.get_text().encode('utf-8')
   f = open("montagtext.txt", "a")
   f.write(words)
  f.close()
```

将它添加到您的`montagproject`目录中，并且不要忘记在运行它之前`commit`它。

```
$ git add . && git commit -m "add montagtextsoup.py script"
```

然后运行:

```
$ spell run -m runs/3:articles "python montagtextsoup.py articles" --pip urllib3 --pip beautifulsoup4 --python2
.
.
.
🎉 Run 4 complete
```

Bingo bango，我们现在有一个 1.3 MB 的训练数据的纯文本文件！

因为我们会非常需要这个文件，所以让我们创建一个到它的链接，这样我们就不需要在挂载它的时候记住运行 id。

```
$ spell link montag_text_file runs/4/montagtext.txt
Successfully created symlink:
montag_text_file → runs/4/montagtext.txt Jun 11 10:59
```

# 第二步。将文本输入神经网络

现在我们可以开始玩 Spell 的 GPU 了。为了掌握这一点，我在 [Learn 上看了一些视频。Spell.run](https://learn.spell.run/) ，发现文章[在 Spell](https://medium.com/@spellrun/generating-rnn-text-on-spell-18a1ab8179b8) 上生成 RNN 文本非常有用。

我们将要使用的夏尔-RNN 代码在这个库中:[https://github.com/minimaxir/gpt-2-simple.git](https://github.com/minimaxir/gpt-2-simple.git)——但是你不必在本地克隆或添加它来使用它。

我将在名为`run.py`的`montagproject`文件夹中添加一个 Python 脚本，它将调用 GPT-2 的能力，并根据我们的 MONTAG 文本数据对 117M 模型进行微调:

```
import gpt_2_simple as gpt2
import sysmodel_name = "117M"
file_name = sys.argv[1]
gpt2.download_gpt2(model_name=model_name) # model is saved into current directory under /models/117M/sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
 file_name, # name of the file with our text data
 model_name=model_name,
 steps=1000) # steps is max number of training stepsgpt2.generate(sess)
```

我再次提交我的 git repo，注意到我添加了`run.py`脚本。

```
$ git add . && git commit -m "add run.py script"
```

关键时刻到了，该做饼干了。

这里的最后一个参数，`-t k80`，指定我们要使用 Spell 的 K80 GPUs，这将比使用 CPU 能力快得多——并且当你注册时，Spell 会给你 10 美元的免费积分，所以这将是快速的*和免费的*。

```
$ spell run --pip gpt_2_simple "python run.py montag.txt" -m montag_text_file:montag.txt -t k80
```

如果一切按计划进行，您将开始看到大量输出，这意味着它是有生命的，是有生命的！

这就是咒语的美妙之处——你不必担心看着你的电脑，以确保它不会着火或飞行。事实上，你可以在神经网络训练的整个过程中一直正常使用你的计算机。既然我们选择了一个`K80`，它应该比 CPU 快，但是仍然有足够的时间在它工作的时候起来跳一段庆祝的舞蹈。

# 第三步。对文本取样

1 小时 14 分 55 秒后，GPT 2 号完成了训练。看看这一过程中产生的一些输出，我们知道这是一条好的道路。以下是样本中最有说服力的两个片段:

```
101: "The idea of a currency is exciting, but not at all convincing. As a modern society, we often try to imagine how much food and drink we would need in order to live - which, no matter how much you drink, doesn't always seem as important as what someone else wants, either. So, in the past two millennia, we have tried to create things that are like any existing economy, but with fewer and fewer laws and regulations, and easier access to credit, buying and selling seems like simple economics.In reality, we're doing things the old-fashioned way, creating new currencies for buying and selling, but with a big emphasis on flexibility and anonymity."
501: "The future (with some exceptions) tends to bang itself around a little more, so sometimes the combination of tech and mindfulness that accompany truly being constantly alive gets you really high.For me, that combination of being constantly alive is extremely important to the success of my company. Back in 2014, I completed a study by Citi that found that mindfulness meditation led to lower prices for wine and a better experience, and that engaging with material things was linked to less dopamine release.And while a healthy and happy life is fun and hip, a study by Mindful reported an 80% decrease in anxiety and panic attacks among suitably excited customers.So while you're at it, join me as I meditate on a beautiful day: for a low, gain access to plenty of happiness in all you potential customers I can influence into spending less time on pointless things like Netflix, I can boost creativity, and my life will be more fulfilling.Open Happiness.Find it on Twitter.Youtube.And don't miss the chance to get involved!"
```

这是我见过的很好的内容营销材料。现在，我们想创建一个小脚本，让我们可以随时对文本进行采样，并调整温度等因素，温度(非常非常简单)控制输出与数据的匹配程度——温度越低，书写越真实，温度越高，书写越奇怪。

我们在使用 Max Woolfe 的 GPT-2 实现时遇到的一个问题是`load_gpt2`和 generate，这两个我们在[自述示例](https://github.com/minimaxir/gpt-2-simple/blob/master/README.md)中看到的命令，如果不做一点改动就无法与拼写一起工作。在将`run_name`作为一个参数之后，这个函数的下一部分试图找到默认情况下`checkpoint/run1`下的训练数据。相反，我们希望将加载和生成函数指向我们在 Spell 上的训练运行中创建的输出，所以我们必须这样做:

首先创建一个法术链接，从你训练的跑步中挂载检查点(在我的例子中，是跑`5`)。这并不是绝对必要的，但它使跟踪跑步变得更容易:

```
$ spell link montagmodel runs/5
```

接下来，我们将把从检查点生成新文本的脚本(可以在 repo 的 README.md 中找到)放入一个名为`generate.py`的文件中:

```
import gpt_2_simple as gpt2sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)gpt2.generate(sess)
```

当我们运行`generate.py`时，我们指定以这种方式从 run `5`中的训练模型加载检查点(不要忘记提交新文件！):

**编辑完** `**generate.py**` **文件后，不要忘记提交 git:**

```
$ git add . && git commit -m "add generate.py"
$ spell run --pip gpt_2_simple "python generate.py" -m montagmodel/checkpoint:checkpoint
```

现在我已经确保脚本可以找到训练好的模型，并可以从它们生成文本，让我们通过改变温度来稍微调整一下`generate.py`,并生成更多的样本，看看我们是否可以得到更真实的文章。这段代码为温度 0.5、0.7、1.0 和 1.5 分别创建了 20 个样本，并将它们分别输出到各自的文本文件中。

```
import gpt_2_simple as gpt2sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)
gpt2.generate(sess)gpt2.generate_to_file(sess, destination_path='montagarticlernn-temp0-5-20samples.txt', nsamples=20, temperature=0.5)
gpt2.generate_to_file(sess, destination_path='montagarticlernn-temp0-7-20samples.txt', nsamples=20, temperature=0.7)
gpt2.generate_to_file(sess, destination_path='montagarticlernn-temp1-20samples.txt', nsamples=20, temperature=1.0)
gpt2.generate_to_file(sess, destination_path='montagarticlernn-temp1-5-20samples.txt', nsamples=20, temperature=1.5)
```

为了运行，我们使用与之前相同的命令:

```
$ spell run --pip gpt_2_simple “python generate.py” -m montagmodel/checkpoint:checkpoint
```

这里有几篇文章样本来自最保守的温度，0.5:

这似乎是对一个新的 YouTube 趋势和现代艺术运动宣言的解释，它特别令人印象深刻，因为它包括一个 Twitter 图片链接和一个不存在的柏林艺术博物馆的说明:

```
"Youtube Video - "Bombing"Bombing is when an utterly stupid idea, quickly borne from parents' and peers' knowledge base, sets off alarm bells.Bombing is when a concept, idea or practice - known in some countries as "art" - breaks through the normal rules of normal human behaviour and convinces - or induces - us to act.Bombing movies, TV shows, games - many more are likely to be flaunted around the world than art.So if you really want to see how susceptible we are to the ravages of modern art-based violence, look no further than the Brutalist fortress city of Dusseldorf, where the city walls have been razed to the ground and graffiti-covered lampposts have been torn up and flung into the distance; where standing ovation is still the most effective form of resistance.But don't be alarmed if art-based paleo-ethno-blasting gets itself into a bit of a quandary. Berlin's Trafigura Museum has started a fund-raiser to pay for wall-removal, claiming $150,000 as their "reward" for completing and destroying a 60-foot-tall wall in the centre of the museum.pic.twitter.com/7xilOswu1h - The Trafigura Museum (@TheTrafigura) August 8, 2017The fact that this kind of art-based annihilation is so readily available to a fraction of the population is down to a stubborn refusal to take risks, a stubborn refusal that is partly understandable, partly reflects the fragility of local democratic culture, and partly reflects the desperation of the ruling class, who are determined that any resistance is met with force and violence.Anti-art, anti-future, anti-art, anti-future. The only thing standing between us and this kind of apocalypse is a willingness to take the risk.The day has come."
```

《人工智能》对人工智能未来的推测:

```
"OMFG, what is the future of artificial intelligence?Let's keep these two threads running for as long as possible. AI generally likes to think of itself as a race to be first in the race, and if it can be thought of as a, well, any race, then so be it. But human intelligence also likes to think of it as a race to be free. To be the person who is not afraid to die, who doesn't want to be drained of all sense of self, and who doesn't want to be anything but.And the more you think about it, the more you'll understand why automation is bad for our planet and why deep learning is bad for us.And it's not even just the jobs that are going to be automated: the people who are unaugmented by their jobs and are actively working to make their jobs obsolete. It's the automation of the cultural sphere, at least for a while.What about the other jobs too? Those that are situational unemployment insurance workers, who either can't or won't work due to the lack of cultural capital or cultural influence to live elsewhere, until they find a better life elsewhere.These people are going to do anything to survive and thrive in a technologically-driven world, and even if they survive, they're doing it in a way that makes them less than human. And they're not going to get a job or a raise any time soon."
```

来自温度为 1.0 的样品:

费德里科·赞尼尔(Federico Zannier，一名软件开发人员，2013 年在 Kickstarter 上出售了自己的数据——我猜他现在几乎是我们人工智能的自由职业者)署名的关于资本主义和自动化未来的思考！)，以及 CNN 政治记者、专家丹·梅里卡(Dan Merica)编造的一段话。然而，正如开头一样令人信服的是，这篇文章确实可以归结为一个处方:

```
"By Federico ZannierWhat does the future have in store for us after billions of Euro has been gambled on global capitalism? One company has taken advantage of this reality, opening factories in China, and setting up shop in one of the world's most sweatshops - cutting the work environment for Chinese workers, and for the global capitalist project. Workers in China start cashing in on their garment and real estate investments, and after years of being let off the hook for unpaid internships and part-time work, are finally being punished for their collective labor and working conditions. Workers in Costa Rica, chipping in for ethical sweatshows, and setting up shop in sweatshops all over the world. Open Capitalism Now!But is it really open for capitalism to go to a sweatshop and begin to innovate? Maybe it's not capitalism that takes the risk, it's open work, capitalism in action.Some are afraid that, if allowed to go ahead, the next few years will be one of the most dangerous for capitalism. Over at MONTAG, Dan Merica warns of the dangers of automation and digital wage manipulation:"Americans are going to be driving the decision to buy a car - and many are predicting that as more automation and digital labor labor comes to the U.S. they'll be the ones who are hardest hit. And their predictions are off, at least for the summer."And Mark Giustina at technobabylontips says something vaguely optimistic about the future of work: "The worst fears of automation workers - who want strict control over their jobs before they ever have a chance to build their own company - are unfounded."Return to text version: this article originally ran on MONTAGIn the second of my looks into tomorrow's tomorrow's sweatshop-like work, I want to see how robots may fare in the kitchen. This part is tricky. Knock down a tomato and bake it for about 500 minutes, then fry it in a fragrant butter sauce until golden brown. Repeat. Flavor it up."
```

以及对一种叫做迷你裤的加密货币的猜测，这是替代硬币的一个好名字:

```
"TL;DR: There's no point in hoarding currencies in the first placeOnly after discussing the likelihood with a crypto evangelist, would we not want to write a very detailed analysis of the individual reasons why, even with all of the giddiness and fanciness of the concept, it is hard to picture putting two and two (even though there are plenty of them in all of us).The death of currency would cease when most people no longer needed it, and cryptocurrencies would be the wrong entity to mint. We need instead a new coin that we can knock around and use to ferry messages between millions.There is already an utterly fascinating array of altcoins waiting in the wings, and none of them are nearly as obscure as the Minie-Pants. Oh, the lost alloys Tenka and The Bull.They too are trying to mint a new currency, and likelihood of success is as good: if the altcoin is truly effective at changing how people spend USD, it must also be persuasive enough to convince potential buyers that it is a good investment.And yet, the Minie-Pants seem more convinced than I.T. pros that they're still in the early days of crypto. So they seem to be genuinely interested in the coin, and can see clearly when to take it out of proportion. They also seem to be fundamentally wrong about taking money out of thin air.Why? The coins and concepts that make them tick all have a lot to do with money's freeze-and-reset cycle: everything from debit cards to US currency, including Bitcoin, has to be part of the normal economy to keep the system running smoothly.The Minie-Pants are fundamentally unfair: they want to make sure everyone has a USB stick to use Bitcoin at all times, and don't want governments, banks and central banks to be sending coins to those countries where there are no internet services or banks to be found.They want to make sure no-one has access to money, and neither do we. But to this we say: Shut the fuck up, girl!"
```

虽然这个神经网络在加密货币、人工智能和劳动力自动化等未来思维的技术主题上显然是一个相当称职的作家，但它还没有学会上传文章和点击发布按钮——但只要再多一点编程，这就可以轻松完成。

等等…我刚刚是不是自动丢掉了一份工作？

在 AI 决定我过时之前，你可以在 [montag.xyz](https://montag.xyz) 阅读一本由 AI 写的杂志，由人类出版。