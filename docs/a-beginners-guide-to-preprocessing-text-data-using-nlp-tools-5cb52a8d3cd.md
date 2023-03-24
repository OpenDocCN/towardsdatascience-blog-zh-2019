# 使用 NLP 预处理文本数据的初学者指南

> 原文：<https://towardsdatascience.com/a-beginners-guide-to-preprocessing-text-data-using-nlp-tools-5cb52a8d3cd?source=collection_archive---------17----------------------->

## 用于预处理从 Twitter 获得的推文的代码

![](img/80c661fc6f7d71ee547418fb7e1b0a99.png)

Source: [https://www.blumeglobal.com/learning/natural-language-processing/](https://www.blumeglobal.com/learning/natural-language-processing/)

下面我概述了我用来预处理自然语言处理项目的代码。这个代码主要用于分类者从 Twitter 上获得的 tweets。我希望本指南能够帮助有抱负的数据科学家和机器学习工程师，让他们熟悉一些非常有用的自然语言处理工具和步骤。

## 要导入的包

```
from nltk.stem import LancasterStemmer, SnowballStemmer, RegexpStemmer, WordNetLemmatizer 
#this was part of the NLP notebookimport nltk
nltk.download('punkt')#import sentence tokenizer
from nltk import sent_tokenize#import word tokenizer
from nltk import word_tokenize#list of stopwords
from nltk.corpus import stopwordsimport string
```

## 处理表情符号

```
import emoji#checking if a character is an emoji
def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI#does the text contain an emoji?
def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return True
    return False#remove the emoji
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')
```

## 删除标点符号

```
punct =[]
punct += list(string.punctuation)
punct += '’'
punct.remove("'")def remove_punctuations(text):
    for punctuation in punct:
        text = text.replace(punctuation, ' ')
    return text
```

下面的函数被用来为我工作的分类器项目对 tweets 进行大量的预处理。这应该同样适用于您可能正在从事的其他 NLP 项目。上面的功能会在下面辅助。

## 函数完成了大部分预处理工作，为了便于理解，已经将其注释掉了

```
def nlp(df):
    # lowercase everything
    # get rid of '\n' from whitespace
    # regex remove hyperlinks
    # removing '&gt;'
    # check for emojis
    # remove emojis
    # remove punctuation
    # remove ' s ' from removing punctuation # lowercase everything
    df['token'] = df['tweet'].apply(lambda x: x.lower())
    # get rid of '\n' from whitespace 
    df['token'] = df['token'].apply(lambda x: x.replace('\n', ' '))
    # regex remove hyperlinks
    df['token'] = df['token'].str.replace('http\S+|[www.\S+'](http://www.\S+'), '', case=False)
    # removing '&gt;'
    df['token'] = df['token'].apply(lambda x: x.replace('&gt;', ''))
    # Checking if emoji in tokens column, use for EDA purposes otherwise not necessary to keep this column
    df['emoji'] = df['token'].apply(lambda x: text_has_emoji(x))
    # Removing Emojis from tokens
    df['token'] = df['token'].apply(lambda x: deEmojify(x))
    # remove punctuations
    df['token'] = df['token'].apply(remove_punctuations)
    # remove ' s ' that was created after removing punctuations
    df['token'] = df['token'].apply(lambda x: str(x).replace(" s ", " ")) return df
```

# 使用空间的词汇化

```
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookupssp = spacy.load('en')
lookups = Lookups()
lemm = Lemmatizer(lookups)
```

## 创建和执行一个引理函数

```
def lemma_function(text):
    dummy = []
    #this is just a test to see if it works
    for word in sp(text):
        dummy.append(word.lemma_) return ' '.join(dummy)df['lemmatized'] = df['token'].apply(lambda x: lemma_function(x))
```

在对推文进行词汇化后，我发现“-PRON-”出现在我的文本中，这是在你使用 spacy 对代词进行词汇化后出现的“短语”。这对于通知我一条推文的内容并不重要，所以我也删除了这个“-PRON-”短语。

```
df['lemmatized'] = df['lemmatized'].apply(lambda x: x.replace('-PRON-', ' '))
```

## 标记我的数据

```
df['final_text'] = df['lemmatized'].apply(word_tokenize)
```

## 从我的令牌中删除停用字词

```
from nltk.corpus import stopwords
my_stopwords = set(stopwords.words('english'))df['final_text'] = df['final_text'].apply(lambda text_list: [x for x in text_list if x not in stopwords.words('english')])
```

## 从我的标记化文本数据中移除数字

```
df['final_text'] = df['final_text'].apply(lambda list_data: [x for x in list_data if x.isalpha()])
```

既然你的 tweet 数据已经被预处理了，你的数据现在应该更干净了。除了上述内容之外，您还有更广泛的考虑事项。在我的分类器项目中，我还使用了 **TextBlob** 来自动更正我的语料库中的任何拼写错误。但是请注意，TextBlob 的计算开销很大。此外，一旦您对数据进行了充分的预处理，并准备好创建单词包(计数矢量器)或 TF-IDF 矢量器，您就可以调整参数，以满足您对机器学习问题的要求。

我希望这篇指南能加速你的下一个 NLP 项目的文本数据的预处理。请随意留下任何想法和见解。