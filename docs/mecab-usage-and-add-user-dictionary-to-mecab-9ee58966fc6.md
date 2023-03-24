# MeCab 用法并将用户词典添加到 MeCab

> 原文：<https://towardsdatascience.com/mecab-usage-and-add-user-dictionary-to-mecab-9ee58966fc6?source=collection_archive---------19----------------------->

## 这是一个英语 MeCab 教程，面向那些不在 - **说日语的人**的程序员

![](img/cdfe971b63638140f60ea4fea3089fff.png)

Photo by [David Emrich](https://unsplash.com/@otoriii?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/kanji?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

如果你是一个在日本工作的 NLP 工程师，你一定听说过文本分割库 MeCab。这个库在日本的学术界和工业界都被广泛使用。如果你的母语不是日语，使用这个工具可能会有困难，因为文档大部分是由日本人写的。所以我写这篇教程来减轻负担。

# 装置

```
Environment:
- macOS Mojave 10.14.6
- Xcode: 11.0 
- python 3
- pip 19.1
```

**安装 MeCab 和字典**。没有这些依赖项，mecab-python3 就无法工作。

```
$ brew install mecab
$ brew install mecab-ipadic
```

您可以运行`mecab`命令来尝试交互界面

```
$ mecab
おはようございます
おはよう    感動詞,*,*,*,*,*,おはよう,オハヨウ,オハヨー
ござい   助動詞,*,*,*,五段・ラ行特殊,連用形,ござる,ゴザイ,ゴザイ
ます  助動詞,*,*,*,特殊・マス,基本形,ます,マス,マス
EOS
```

**安装** [**SWIG**](https://www.wikiwand.com/en/SWIG) 。我们必须安装这个库，否则，安装 mecab-python3 时会出现错误。

```
$ brew install swig$ swig -versionSWIG Version 4.0.1Compiled with clang++ [x86_64-apple-darwin18.6.0]Configured options: +pcrePlease see [http://www.swig.org](http://www.swig.org) for reporting bugs and further information
```

**最后安装 mecab-python3。**

```
$ pip install mecab-python3Collecting mecab-python3
  Downloading [https://files.pythonhosted.org/packages/97/9f/3e5755e0488f608e3c2d18a0f3524434ebf36904b8fd4eec74a3e84416a9/mecab_python3-0.996.2-cp36-cp36m-macosx_10_6_intel.whl](https://files.pythonhosted.org/packages/97/9f/3e5755e0488f608e3c2d18a0f3524434ebf36904b8fd4eec74a3e84416a9/mecab_python3-0.996.2-cp36-cp36m-macosx_10_6_intel.whl) (14.1MB)
     |████████████████████████████████| 14.1MB 2.4MB/s 
Installing collected packages: mecab-python3
Successfully installed mecab-python3-0.996.2
```

安装程序已完成。我们可以写个剧本试试。

```
import MeCab
mecab = MeCab.Tagger("-Ochasen") # Create a MeCab object
malist = mecab.parse("ＮＥＣが二位、東芝がモトローラを抜いて二年ぶりに三位になる。") # morphological analysis
print(malist)# output
ＮＥＣ  エヌイーシー    ＮＥＣ  名詞-固有名詞-組織
が      ガ      が      助詞-格助詞-一般
二      ニ      二      名詞-数
位      イ      位      名詞-接尾-助数詞
、      、      、      記号-読点
東芝    トウシバ        東芝    名詞-固有名詞-組織
が      ガ      が      助詞-格助詞-一般
モトローラ      モトローラ      モトローラ      名詞-固有名詞-組織
を      ヲ      を      助詞-格助詞-一般
抜い    ヌイ    抜く    動詞-自立       五段・カ行イ音便        連用タ接続
て      テ      て      助詞-接続助詞
二      ニ      二      名詞-数
年      ネン    年      名詞-接尾-助数詞
ぶり    ブリ    ぶり    名詞-接尾-一般
に      ニ      に      助詞-格助詞-一般
三      サン    三      名詞-数
位      イ      位      名詞-接尾-助数詞
に      ニ      に      助詞-格助詞-一般
なる    ナル    なる    動詞-自立       五段・ラ行      基本形
。      。      。      記号-句点
EOS
```

# 添加用户词典

根据[官方文件](https://taku910.github.io/mecab/dic.html)显示，添加自定义词典有 2 种方式，更改“系统词典”和添加“用户词典”。如果我们改变“系统字典”，我们必须编译字典并重新安装。所以我们选择添加一个新的“用户词典”。

## 确定格式

首先，我们必须将自定义词典写入一个 CSV 文件，格式如下。

```
表層形,左文脈 ID,右文脈 ID,コスト,品詞,品詞細分類 1,品詞細分類 2,品詞細分類 3,活用型,活用形,原形,読み,発音Surface type, left context ID, right context ID, cost, part of speech, sub POS 1, sub-POS 2, sub-POS 3, conjugation type, conjugation form, original form, ruby, pronunciation
```

Each line contains 13 features. For example, if we want to add **noun words (活用しない語)**, like location, people’s name, we could write like below.

```
工藤,1223,1223,6058,名詞,固有名詞,人名,名,*,*,くどう,クドウ,クドウ
```

But if we want to add **adjectives or verb (活用する語)**, we have to add all their **conjugation type (活用型)** and **conjugation forms (活用形)**. See the below example for the adjective “いそがしい”.

```
いそがしい,120,120,6078,形容詞,自立,*,*,形容詞・イ段,基本形,いそがしい,イソガシイ,イソガシイ
いそがし,128,128,6080,形容詞,自立,*,*,形容詞・イ段,文語基本形,いそがしい,イソガシ,イソガシ
いそがしから,136,136,6079,形容詞,自立,*,*,形容詞・イ段,未然ヌ接続,いそがしい,イソガシカラ,イソガシカラ
いそがしかろ,132,132,6079,形容詞,自立,*,*,形容詞・イ段,未然ウ接続,いそがしい,イソガシカロ,イソガシカロ
いそがしかっ,148,148,6078,形容詞,自立,*,*,形容詞・イ段,連用タ接続,いそがしい,イソガシカッ,イソガシカッ
いそがしく,152,152,6078,形容詞,自立,*,*,形容詞・イ段,連用テ接続,いそがしい,イソガシク,イソガシク
いそがしくっ,152,152,6079,形容詞,自立,*,*,形容詞・イ段,連用テ接続,いそがしい,イソガシクッ,イソガシクッ
いそがしゅう,144,144,6079,形容詞,自立,*,*,形容詞・イ段,連用ゴザイ接続,いそがしい,イソガシュウ,イソガシュウ
いそがしゅぅ,144,144,6079,形容詞,自立,*,*,形容詞・イ段,連用ゴザイ接続,いそがしい,イソガシュゥ,イソガシュゥ
いそがしき,124,124,6079,形容詞,自立,*,*,形容詞・イ段,体言接続,いそがしい,イソガシキ,イソガシキ
いそがしけれ,108,108,6079,形容詞,自立,*,*,形容詞・イ段,仮定形,いそがしい,イソガシケレ,イソガシケレ
いそがしかれ,140,140,6079,形容詞,自立,*,*,形容詞・イ段,命令ｅ,いそがしい,イソガシカレ,イソガシカレ
いそがしけりゃ,112,112,6079,形容詞,自立,*,*,形容詞・イ段,仮定縮約１,いそがしい,イソガシケリャ,イソガシケリャ
いそがしきゃ,116,116,6079,形容詞,自立,*,*,形容詞・イ段,仮定縮約２,いそがしい,イソガシキャ,イソガシキャ
いそがし,104,104,6080,形容詞,自立,*,*,形容詞・イ段,ガル接続,いそがしい,イソガシ,イソガシ
```

## 长元音

下面是一个长元音现象的演示。

```
今日,,,10,名詞,副詞可能,*,*,*,*,今日,キョウ,キョー
労働,,,10,名詞,サ変接続,*,*,*,*,労働,ロウドウ,ロードー
```

“今日” and “労働” are the surface form. “10” is the cost. Lower cost means this word has a high frequency in the corpus. “名詞” is the POS tag. “一般” is the sub-POS tag, which can be ignored. The final three columns represent “original form”, “[ruby](https://www.wikiwand.com/en/Ruby_character)”, and “pronunciation”. The ruby character means how a word should be pronounced. But in real conversation, people usually ignore the vowel letters (a e i o u, アイウエオ in Japanse) to pronounce a word easier. This phenomenon usually occurs with [long vowels](https://www.wikiwand.com/en/Vowel_length#/Phonemic_vowel_length).

在下面的例子中，“ゥ”将被忽略。

```
キョウ -> キョー
ロウドウ -> ロードー
```

The Romanji of “今日” is “kyou”. The “o” and “u” are both vowels, so this is a long vowel. When we pronounce “今日”, “u” will be ignored. So the pronunciation becomes “kyo” and “o” produce as a long vowel, /uː/.

## 使用 CSV 文件扩充字典

对于下面的演示，我构建一个`foo.csv`文件并输入一些名词。

```
二位,,,10,名詞,一般,*,*,*,*,*,二位,ニイ,ニイ
三位,,,10,名詞,一般,*,*,*,*,*,三位,サンイ,サンイ
```

接下来我们需要编译这个 CSV 文件。

Mac(与自制软件一起安装)

```
$ cd /home/foo/bar # the path where the foo.csv exists
$ /usr/local/Cellar/mecab/0.996/libexec/mecab/mecab-dict-index \
-d /usr/local/lib/mecab/dic/ipadic \
-u foo.dic \
-f utf-8 \
-t utf-8 foo.csv reading foo.csv ... 2
emitting double-array: 100% |###########################################|
```

选项:

*   d:系统字典目录
*   u:输出用户词典的名称
*   f:CSV 文件的编码
*   t:输出用户字典文件的编码

确保用户字典创建成功。

```
$ ls 
foo.csv foo.dic
```

将 foo.dic 移动到一个新目录。

```
$ mkdir /usr/local/lib/mecab/dic/user_dict
$ mv foo.dic /usr/local/lib/mecab/dic/user_dict
```

接下来，我们需要添加路径到`mecabrc`文件。

```
vim /usr/local/etc/mecabrc
```

更改如下

```
; userdic = /home/foo/bar/user.dic
->
userdic = /usr/local/lib/mecab/dic/user_dict/foo.dic
```

如果您有多个用户字典文件，您可以将它们写在一行中。

```
userdic = /usr/local/lib/mecab/dic/user_dict/foo.dic, /usr/local/lib/mecab/dic/user_dict/foo2.dic, /usr/local/lib/mecab/dic/user_dict/foo3.dic
```

好了，就这些。

## 试试用户词典

```
$ echo "ＮＥＣが二位、東芝がモトローラを抜いて二年ぶりに三位になる。" | mecabＮＥＣ  名詞,固有名詞,組織,*,*,*,ＮＥＣ,エヌイーシー,エヌイーシー
が      助詞,格助詞,一般,*,*,*,が,ガ,ガ
二位    名詞,一般,*,*,*,*,*,二位,ニイ,ニイ
、      記号,読点,*,*,*,*,、,、,、
東芝    名詞,固有名詞,組織,*,*,*,東芝,トウシバ,トーシバ
が      助詞,格助詞,一般,*,*,*,が,ガ,ガ
モトローラ      名詞,固有名詞,組織,*,*,*,モトローラ,モトローラ,モトローラ
を      助詞,格助詞,一般,*,*,*,を,ヲ,ヲ
抜い    動詞,自立,*,*,五段・カ行イ音便,連用タ接続,抜く,ヌイ,ヌイ
て      助詞,接続助詞,*,*,*,*,て,テ,テ
二      名詞,数,*,*,*,*,二,ニ,ニ
年      名詞,接尾,助数詞,*,*,*,年,ネン,ネン
ぶり    名詞,接尾,一般,*,*,*,ぶり,ブリ,ブリ
に      助詞,格助詞,一般,*,*,*,に,ニ,ニ
三位    名詞,一般,*,*,*,*,*,三位,サンイ,サンイ
に      助詞,格助詞,一般,*,*,*,に,ニ,ニ
なる    動詞,自立,*,*,五段・ラ行,基本形,なる,ナル,ナル
。      記号,句点,*,*,*,*,。,。,。
EOS
```

We can see the “二位” and “三位” are recognized as a word.

如果我想知道成本，我们可以改变输出格式。`%m`是表层形式，`%c`是成本，`%H`是逗号分隔的词性、变位、阅读等列表。你可以在这些文件中找到更多细节， [EN 版](https://github.com/buruzaemon/natto/wiki/Output-Formatting)，和 [JP 版](https://taku910.github.io/mecab/format.html)。

```
$ echo "NECが二位、東芝がモトローラを抜いて二年ぶりに三位になる。" | mecab -F '%m %c %H\n'NEC 13835 名詞,固有名詞,組織,*,*,*,*
が 3866 助詞,格助詞,一般,*,*,*,が,ガ,ガ
二位 10 名詞,一般,*,*,*,*,*,二位,ニイ,ニイ
、 -2435 記号,読点,*,*,*,*,、,、,、
東芝 1426 名詞,固有名詞,組織,*,*,*,東芝,トウシバ,トーシバ
が 3866 助詞,格助詞,一般,*,*,*,が,ガ,ガ
モトローラ 4897 名詞,固有名詞,組織,*,*,*,モトローラ,モトローラ,モトローラ
を 4183 助詞,格助詞,一般,*,*,*,を,ヲ,ヲ
抜い 7346 動詞,自立,*,*,五段・カ行イ音便,連用タ接続,抜く,ヌイ,ヌイ
て 5170 助詞,接続助詞,*,*,*,*,て,テ,テ
二 2914 名詞,数,*,*,*,*,二,ニ,ニ
年 8465 名詞,接尾,助数詞,*,*,*,年,ネン,ネン
ぶり 7451 名詞,接尾,一般,*,*,*,ぶり,ブリ,ブリ
に 4304 助詞,格助詞,一般,*,*,*,に,ニ,ニ
三位 10 名詞,一般,*,*,*,*,*,三位,サンイ,サンイ
に 4304 助詞,格助詞,一般,*,*,*,に,ニ,ニ
なる 5063 動詞,自立,*,*,五段・ラ行,基本形,なる,ナル,ナル
。 215 記号,句点,*,*,*,*,。,。,。
EOS
```

> ***查看我的其他帖子*** [***中等***](https://medium.com/@bramblexu) ***同*** [***分类查看***](https://bramblexu.com/posts/eb7bd472/) ***！
> GitHub:***[***bramble Xu***](https://github.com/BrambleXu) ***LinkedIn:***[***徐亮***](https://www.linkedin.com/in/xu-liang-99356891/) ***博客:***[*bramble Xu*](https://bramblexu.com)

# 参考

*   [https://qiita.com/TomOse/items/90a6addda3b0419f40e3](https://qiita.com/TomOse/items/90a6addda3b0419f40e3)
*   [https://taku910.github.io/mecab/dic.html](https://taku910.github.io/mecab/dic.html)
*   [https://www.kanjifumi.jp/keyword/chi/](https://www.kanjifumi.jp/keyword/chi/)
*   [https://www . wiki wand . com/en/元音 _ 长度#/Phonemic _ 元音 _ 长度](https://www.wikiwand.com/en/Vowel_length#/Phonemic_vowel_length)
*   [https://rooter.jp/data-format/mecab_user_dictionary/](https://rooter.jp/data-format/mecab_user_dictionary/)
*   [https://github.com/buruzaemon/natto/wiki/Output-Formatting](https://github.com/buruzaemon/natto/wiki/Output-Formatting)
*   [https://taku910.github.io/mecab/format.html](https://taku910.github.io/mecab/format.html)