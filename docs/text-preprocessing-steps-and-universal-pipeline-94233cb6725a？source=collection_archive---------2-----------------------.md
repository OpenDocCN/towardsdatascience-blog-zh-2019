# 文本预处理步骤和通用可重用流水线

> 原文：<https://towardsdatascience.com/text-preprocessing-steps-and-universal-pipeline-94233cb6725a?source=collection_archive---------2----------------------->

![](img/adb99914a17e7a62c4673ebf1a3b77b6.png)

## 所有文本预处理步骤的描述和可重用文本预处理管道的创建

在向任何 ML 模型提供某种数据之前，必须对其进行适当预处理。你一定听过这个谚语:`Garbage in, garbage out` (GIGO)。文本是一种特殊的数据，不能直接输入到大多数 ML 模型中，所以在输入到模型中之前，你必须以某种方式从中提取数字特征，换句话说就是`vectorize`。矢量化不是本教程的主题，但您必须了解的主要内容是，GIGO 也适用于矢量化，您只能从定性预处理的文本中提取定性特征。

我们将要讨论的事情:

1.  标记化
2.  清洁
3.  正常化
4.  词汇化
5.  汽蒸

最后，我们将创建一个可重用的管道，您可以在您的应用程序中使用它。

Kaggle 内核:[https://www . ka ggle . com/balat mak/text-预处理-步骤-通用-管道](https://www.kaggle.com/balatmak/text-preprocessing-steps-and-universal-pipeline)

让我们假设这个示例文本:

```
An explosion targeting a tourist bus has injured at least 16 people near the Grand Egyptian Museum, 
next to the pyramids in Giza, security sources say E.U.

South African tourists are among the injured. Most of those hurt suffered minor injuries, 
while three were treated in hospital, N.A.T.O. say.

http://localhost:8888/notebooks/Text%20preprocessing.ipynb

@nickname of twitter user and his email is email@gmail.com . 

A device went off close to the museum fence as the bus was passing on 16/02/2012.
```

# 标记化

`Tokenization`——文本预处理步骤，假设将文本分割成`tokens`(单词、句子等)。)

看起来你可以使用某种简单的分隔符来实现它，但是你不要忘记，在许多不同的情况下，分隔符是不起作用的。例如，如果您使用带点的缩写，那么`.`用于将标记化成句子的分隔符将会失败。所以你必须有一个更复杂的模型来达到足够好的结果。通常这个问题可以通过使用`nltk`或`spacy` nlp 库来解决。

NLTK:

```
from nltk.tokenize import sent_tokenize, word_tokenize

nltk_words = word_tokenize(example_text)
display(f"Tokenized words: **{nltk_words}**")
```

输出:

```
Tokenized words: ['An', 'explosion', 'targeting', 'a', 'tourist', 'bus', 'has', 'injured', 'at', 'least', '16', 'people', 'near', 'the', 'Grand', 'Egyptian', 'Museum', ',', 'next', 'to', 'the', 'pyramids', 'in', 'Giza', ',', 'security', 'sources', 'say', 'E.U', '.', 'South', 'African', 'tourists', 'are', 'among', 'the', 'injured', '.', 'Most', 'of', 'those', 'hurt', 'suffered', 'minor', 'injuries', ',', 'while', 'three', 'were', 'treated', 'in', 'hospital', ',', 'N.A.T.O', '.', 'say', '.', 'http', ':', '//localhost:8888/notebooks/Text', '%', '20preprocessing.ipynb', '@', 'nickname', 'of', 'twitter', 'user', 'and', 'his', 'email', 'is', 'email', '@', 'gmail.com', '.', 'A', 'device', 'went', 'off', 'close', 'to', 'the', 'museum', 'fence', 'as', 'the', 'bus', 'was', 'passing', 'on', '16/02/2012', '.']
```

空间:

```
import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()

doc = nlp(example_text)
spacy_words = [token.text for token **in** doc]
display(f"Tokenized words: **{spacy_words}**")
```

输出:

```
Tokenized words: ['\\n', 'An', 'explosion', 'targeting', 'a', 'tourist', 'bus', 'has', 'injured', 'at', 'least', '16', 'people', 'near', 'the', 'Grand', 'Egyptian', 'Museum', ',', '\\n', 'next', 'to', 'the', 'pyramids', 'in', 'Giza', ',', 'security', 'sources', 'say', 'E.U.', '\\n\\n', 'South', 'African', 'tourists', 'are', 'among', 'the', 'injured', '.', 'Most', 'of', 'those', 'hurt', 'suffered', 'minor', 'injuries', ',', '\\n', 'while', 'three', 'were', 'treated', 'in', 'hospital', ',', 'N.A.T.O.', 'say', '.', '\\n\\n', 'http://localhost:8888/notebooks', '/', 'Text%20preprocessing.ipynb', '\\n\\n', '@nickname', 'of', 'twitter', 'user', 'and', 'his', 'email', 'is', 'email@gmail.com', '.', '\\n\\n', 'A', 'device', 'went', 'off', 'close', 'to', 'the', 'museum', 'fence', 'as', 'the', 'bus', 'was', 'passing', 'on', '16/02/2012', '.', '\\n']
```

在 spacy 输出标记化中，而不是在 nltk 中:

```
{'E.U.', '\\n', 'Text%20preprocessing.ipynb', 'email@gmail.com', '\\n\\n', 'N.A.T.O.', 'http://localhost:8888/notebooks', '@nickname', '/'}
```

在 nltk 中但不在 spacy 中:

```
{'nickname', '//localhost:8888/notebooks/Text', 'N.A.T.O', ':', '@', 'gmail.com', 'E.U', 'http', '20preprocessing.ipynb', '%'}
```

我们看到`spacy`标记了一些奇怪的东西，比如`\n`、`\n\n`，但是能够处理 URL、电子邮件和类似 Twitter 的提及。此外，我们看到`nltk`标记化的缩写没有最后的`.`

# 清洁

`Cleaning` is 步骤假设删除所有不需要的内容。

## 删除标点符号

当标点符号不能为文本矢量化带来附加值时，这可能是一个好的步骤。标点符号删除最好在标记化步骤之后进行，在此之前进行可能会导致不良影响。`TF-IDF`、`Count`、`Binary`矢量化的好选择。

让我们假设这一步的文本:

```
@nickname of twitter user, and his email is email@gmail.com .
```

在标记化之前:

```
text_without_punct = text_with_punct.translate(str.maketrans('', '', string.punctuation))
display(f"Text without punctuation: **{text_without_punct}**")
```

输出:

```
Text without punctuation: nickname of twitter user and his email is emailgmailcom
```

在这里，您可以看到用于正确标记化的重要符号已被删除。现在电子邮件无法正常检测。正如您在`Tokenization`步骤中提到的，标点符号被解析为单个符号，所以更好的方法是先进行符号化，然后删除标点符号。

```
import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()doc = nlp(text_with_punct)
tokens = [t.text for t **in** doc]*# python based removal*
tokens_without_punct_python = [t for t **in** tokens if t **not** **in** string.punctuation]
display(f"Python based removal: **{tokens_without_punct_python}**")# spacy based removal
tokens_without_punct_spacy = [t.text for t **in** doc if t.pos_ != 'PUNCT']
display(f"Spacy based removal: **{tokens_without_punct_spacy}**")
```

基于 Python 的移除结果:

```
['@nickname', 'of', 'twitter', 'user', 'and', 'his', 'email', 'is', 'email@gmail.com']
```

基于空间的移除:

```
['of', 'twitter', 'user', 'and', 'his', 'email', 'is', 'email@gmail.com']
```

这里你可以看到`python-based`移除比 spacy 更有效，因为 spacy 将`@nicname`标记为`PUNCT`词性。

## 停止单词删除

`Stop words`通常指一种语言中最常见的词，通常不会带来额外的意义。没有一个所有 nlp 工具都使用的通用停用词列表，因为这个术语的定义非常模糊。尽管实践已经表明，当准备用于索引的文本时，这一步骤是必须的，但是对于文本分类目的来说可能是棘手的。

空间停止字数:`312`

NLTK 停止字数:`179`

让我们假设这一步的文本:

```
This movie is just not good enough
```

空间:

```
import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()text_without_stop_words = [t.text for t **in** nlp(text) if **not** t.is_stop]
display(f"Spacy text without stop words: **{text_without_stop_words}**")
```

没有停用词的空白文本:

```
['movie', 'good']
```

NLTK:

```
import nltk

nltk_stop_words = nltk.corpus.stopwords.words('english')
text_without_stop_words = [t for t **in** word_tokenize(text) if t **not** **in** nltk_stop_words]
display(f"nltk text without stop words: **{text_without_stop_words}**")
```

无停用词的 NLTK 文本:

```
['This', 'movie', 'good', 'enough']
```

这里你看到 nltk 和 spacy 的词汇量不一样，所以过滤的结果也不一样。但我想强调的主要一点是，单词`not`被过滤了，这在大多数情况下是没问题的，但在你想确定这个句子的极性的情况下`not`会带来额外的含义。

对于这种情况，您可以在空间库中设置可以忽略的停用词。在 nltk 的情况下，您可以删除或添加自定义单词到`nltk_stop_words`，它只是一个列表。

```
import en_core_web_sm

nlp = en_core_web_sm.load()

customize_stop_words = [
    'not'
]

for w **in** customize_stop_words:
    nlp.vocab[w].is_stop = False

text_without_stop_words = [t.text for t **in** nlp(text) if **not** t.is_stop]
display(f"Spacy text without updated stop words: **{text_without_stop_words}**")
```

没有更新停用词的空白文本:

```
['movie', 'not', 'good']
```

# 正常化

像任何数据一样，文本也需要规范化。如果是文本，则为:

1.  将日期转换为文本
2.  数字到文本
3.  货币/百分比符号到文本
4.  缩写扩展(内容相关)NLP —自然语言处理、神经语言编程、非线性编程
5.  拼写错误纠正

总而言之，规范化是将任何非文本信息转换成文本等效信息。

为此，有一个很棒的库——[normalize](https://github.com/EFord36/normalise)。我将从这个库的自述文件中向您展示这个库的用法。这个库基于`nltk`包，所以它需要`nltk`单词标记。

让我们假设这一步的文本:

```
On the 13 Feb. 2007, Theresa May announced on MTV news that the rate of childhod obesity had risen from 7.3-9.6**% i**n just 3 years , costing the N.A.T.O £20m
```

代码:

```
from normalise import normalise

user_abbr = {
    "N.A.T.O": "North Atlantic Treaty Organization"
}

normalized_tokens = normalise(word_tokenize(text), user_abbrevs=user_abbr, verbose=False)
display(f"Normalized text: {' '.join(normalized_tokens)}")
```

输出:

```
On the thirteenth of February two thousand and seven , Theresa May announced on M T V news that the rate of childhood obesity had risen from seven point three to nine point six % in just three years , costing the North Atlantic Treaty Organization twenty million pounds
```

这个库中最糟糕的事情是，目前你不能禁用一些模块，如缩写扩展，它会导致像`MTV` - > `M T V`这样的事情。但是我已经在这个库上添加了一个适当的问题，也许过一会儿就可以修复了。

# 脱镁和汽蒸

`Stemming`是将单词的词形变化减少到其词根形式的过程，例如将一组单词映射到同一个词干，即使该词干本身在语言中不是有效单词。

`Lemmatization`与词干不同，适当减少词根变化，确保词根属于该语言。在引理化中，词根称为引理。一个词条(复数词条或词条)是一组单词的规范形式、词典形式或引用形式。

让我们假设这一步的文本:

```
On the thirteenth of February two thousand and seven , Theresa May announced on M T V news that the rate of childhood obesity had risen from seven point three to nine point six % in just three years , costing the North Atlantic Treaty Organization twenty million pounds
```

NLTK 词干分析器:

```
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenizetokens = word_tokenize(text)
porter=PorterStemmer()# vectorizing function to able to call on list of tokens
stem_words = np.vectorize(porter.stem)stemed_text = ' '.join(stem_words(tokens))
display(f"Stemed text: **{stemed_text}**")
```

带词干的文本:

```
On the thirteenth of februari two thousand and seven , theresa may announc on M T V news that the rate of childhood obes had risen from seven point three to nine point six % in just three year , cost the north atlant treati organ twenti million pound
```

NLTK 术语化:

```
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenizetokens = word_tokenize(text)wordnet_lemmatizer = WordNetLemmatizer()# vectorizing function to able to call on list of tokens
lemmatize_words = np.vectorize(wordnet_lemmatizer.lemmatize)lemmatized_text = ' '.join(lemmatize_words(tokens))
display(f"nltk lemmatized text: **{lemmatized_text}**")
```

NLTK 词条化文本:

```
On the thirteenth of February two thousand and seven , Theresa May announced on M T V news that the rate of childhood obesity had risen from seven point three to nine point six % in just three year , costing the North Atlantic Treaty Organization twenty million pound
```

空间引理化；

```
import en_core_web_sm

nlp = en_core_web_sm.load()lemmas = [t.lemma_ for t **in** nlp(text)]
display(f"Spacy lemmatized text: {' '.join(lemmas)}")
```

Spacy 词条化文本:

```
On the thirteenth of February two thousand and seven , Theresa May announce on M T v news that the rate of childhood obesity have rise from seven point three to nine point six % in just three year , cost the North Atlantic Treaty Organization twenty million pound
```

我们看到`spacy`比 nltk 好得多，其中一个例子`risen` - > `rise`，只有`spacy`处理了它。

# 可重复使用管道

现在是我最喜欢的部分！我们将创建一个可重用的管道，您可以在您的任何项目中使用它。

```
import numpy as np
import multiprocessing as mp

import string
import spacy 
import en_core_web_sm
from nltk.tokenize import word_tokenize
from sklearn.base import TransformerMixin, BaseEstimator
from normalise import normalise

nlp = en_core_web_sm.load()

class **TextPreprocessor**(BaseEstimator, TransformerMixin):
    def __init__(self,
                 variety="BrE",
                 user_abbrevs={},
                 n_jobs=1):
        *"""*
 *Text preprocessing transformer includes steps:*
 *1\. Text normalization*
 *2\. Punctuation removal*
 *3\. Stop words removal*
 *4\. Lemmatization*

 *variety - format of date (AmE - american type, BrE - british format)* 
 *user_abbrevs - dict of user abbreviations mappings (from normalise package)*
 *n_jobs - parallel jobs to run*
 *"""*
        self.variety = variety
        self.user_abbrevs = user_abbrevs
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        X_copy = X.copy()

        partitions = 1
        cores = mp.cpu_count()
        if self.n_jobs <= -1:
            partitions = cores
        elif self.n_jobs <= 0:
            return X_copy.apply(self._preprocess_text)
        else:
            partitions = min(self.n_jobs, cores)

        data_split = np.array_split(X_copy, partitions)
        pool = mp.Pool(cores)
        data = pd.concat(pool.map(self._preprocess_part, data_split))
        pool.close()
        pool.join()

        return data

    def _preprocess_part(self, part):
        return part.apply(self._preprocess_text)

    def _preprocess_text(self, text):
        normalized_text = self._normalize(text)
        doc = nlp(normalized_text)
        removed_punct = self._remove_punct(doc)
        removed_stop_words = self._remove_stop_words(removed_punct)
        return self._lemmatize(removed_stop_words)

    def _normalize(self, text):
        *# some issues in normalise package*
        try:
            return ' '.join(normalise(text, variety=self.variety, user_abbrevs=self.user_abbrevs, verbose=False))
        except:
            return text

    def _remove_punct(self, doc):
        return [t for t **in** doc if t.text **not** **in** string.punctuation]

    def _remove_stop_words(self, doc):
        return [t for t **in** doc if **not** t.is_stop]

    def _lemmatize(self, doc):
        return ' '.join([t.lemma_ for t **in** doc])
```

此代码可用于 sklearn 管道。

测量的性能:在 22 分钟内在 4 个进程上处理了 2225 个文本。甚至没有接近快！这导致了规范化部分，库没有充分优化，但产生了相当有趣的结果，并可以为进一步的矢量化带来额外的价值，所以是否使用它取决于您。

我希望你喜欢这篇文章，我期待你的反馈！

Kaggle 内核:[https://www . ka ggle . com/balat mak/text-预处理-步骤-通用-管道](https://www.kaggle.com/balatmak/text-preprocessing-steps-and-universal-pipeline)