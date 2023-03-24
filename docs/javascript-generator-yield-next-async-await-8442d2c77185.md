# Javascript 生成器 Yield/Next 与 Async-Await 概述和比较

> 原文：<https://towardsdatascience.com/javascript-generator-yield-next-async-await-8442d2c77185?source=collection_archive---------4----------------------->

![](img/85ac86fb7c76e3be87aa94b7e00c3769.png)

## 发电机(ES6)

> 可以根据用户需求在不同的时间间隔返回多个值并可以管理其内部状态的函数是生成器函数。如果一个函数使用了`[*function**](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/function*)`语法，它就变成了 GeneratorFunction。

它们不同于普通函数，普通函数在一次执行中运行完成，而*生成器函数可以暂停并恢复*，因此它们确实运行完成，但触发器仍在开发人员手中。它们允许对异步功能进行更好的执行控制，但这并不意味着它们不能用作同步功能。

> 注意:当执行生成器函数时，它返回一个新的生成器对象。

暂停和恢复是使用`yield` & `next`完成的。所以让我们看看它们是什么，它们是做什么的。

## 产量/下一个

关键字`yield`暂停生成器函数的执行，关键字`yield`后面的表达式的值被返回给生成器的调用者。它可以被认为是基于生成器的`return`关键字版本。

`yield`关键字实际上返回一个具有两个属性`value`和`done`的`IteratorResult`对象。(如果你[不知道什么是迭代器和可迭代对象，那么请阅读这里的](https://codeburst.io/javascript-es6-iterables-and-iterators-de18b54f4d4))。

一旦在`yield`表达式上暂停，生成器的代码执行就会保持暂停，直到调用生成器的`next()`方法。每次调用生成器的`next()`方法时，生成器恢复执行并返回[迭代器](https://codeburst.io/javascript-es6-iterables-and-iterators-de18b54f4d4)的结果。

咳…理论讲够了，让我们来看一个例子

```
function* UUIDGenerator() {
    let d, r;
    while(true) {
        yield 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            r = (new Date().getTime() + Math.random()*16)%16 | 0;
            d = Math.floor(d/16);
            return (c=='x' ? r : (r&0x3|0x8)).toString(16);
        });
    }
};
```

这里，`UUIDGenerator`是一个生成器函数，它使用随机数的当前时间计算 UUID，并在每次执行时返回一个新的 UUID。

要运行上面的函数，我们需要创建一个生成器对象，我们可以在其上调用`next()`

```
const UUID = UUIDGenerator();
// UUID is our generator objectUUID.next() 
// return {value: 'e35834ae-8694-4e16-8352-6d2368b3ccbf', done: false}
```

`UUID.next()`将在`value`键下的每个`UUID.next()`上返回新的 UUID，而`done`将始终为假，因为我们处于一个无限循环中。

> 注意:我们暂停在无限循环之上，这有点酷，在一个生成器函数中的任何“停止点”，它们不仅可以向一个外部函数产生值，还可以从外部接收值。

上面有很多生成器的实际实现，也有很多大量使用它的库， [co](https://github.com/tj/co) 、 [koa](https://koajs.com/) 和 [redux-saga](https://github.com/redux-saga/redux-saga) 就是一些例子。

## 异步/等待(ES7)

![](img/d378ca65594459b9e9857629afa5fdd4.png)

传统上，当异步操作返回使用`Promise`处理的数据时，回调被传递和调用。

Async/Await 是一种特殊的语法，以更舒适的方式处理承诺，非常容易理解和使用。

`async` 关键字用于定义一个*异步函数*，返回一个`[AsyncFunction](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/AsyncFunction)`对象。

`await`关键字用于暂停异步函数的执行，直到`Promise`完成，即解决或拒绝，并在完成后继续执行`async`函数。当恢复时，`await`表达式的值就是已完成的`Promise`的值。

**要点:**

1.`await`只能在`async`函数中使用。

2.带有`async`关键字的函数将总是返回一个承诺。

3.在同一个函数下，多个等待将总是按顺序运行。

4.如果承诺正常解决，那么`await promise`返回结果。但是在拒绝的情况下，它抛出错误，就好像在那一行有一个`throw`语句一样。

5.一个`async`函数不能同时等待多个承诺。

6.如果将许多`await`作为一条语句使用而不依赖于前一条语句，可能会出现性能问题。

到目前为止一切顺利，现在让我们看一个简单的例子:

```
async function asyncFunction() { const promise = new Promise((resolve, reject) => {
    setTimeout(() => resolve("i am resolved!"), 1000)
  }); const result = await promise; 
  // wait till the promise resolves (*) console.log(result); // "i am resolved!"
}asyncFunction();
```

`asyncFunction`执行在第`await promise`行“暂停”,并在承诺完成时恢复，结果是`result`。所以上面的代码一秒钟显示了“`i am resolved!`”。

## 生成器和异步等待—比较

1.  *Generator functions/yield* 和 *Async functions/await* 都可以用来编写“等待”的异步代码，这意味着代码看起来好像是同步的，尽管它实际上是异步的。
2.  *一个生成器函数*一个产出**一个产出**地执行，即通过其迭代器(T4 方法)一次执行一个产出表达式，而*同步等待*，它们被依次执行**等待**。
3.  *Async/await* 使得实现*生成器*的特定用例更加容易。
4.  g *生成器*的返回值始终是`**{value: X, done: Boolean}**`，而对于*同步函数来说，*将始终是一个**承诺**，要么解析为值 X，要么抛出一个错误。
5.  一个`async`函数可以被分解成一个生成器和 promise 实现，这是一个很好的工具。

如果您想被添加到我的电子邮件列表中，请考虑在这里输入您的电子邮件地址[](https://goo.gl/forms/MOPINWoY7q1f1APu2)**和**关注我的** [**中的**](https://medium.com/@ideepak.jsd) **阅读更多关于 javascript 和**[**github**](https://github.com/dg92)**的文章，看看我的疯狂代码**。如果有什么不清楚或者你想指出什么，请在下面评论。**

**你可能也会喜欢我的其他文章**

1.  **[Javascript 执行上下文和提升](https://levelup.gitconnected.com/javascript-execution-context-and-hoisting-c2cc4993e37d)**
2.  **[理解 Javascript‘this’关键字(上下文)](https://medium.com/datadriveninvestor/javascript-context-this-keyword-9a78a19d5786)。**
3.  **[Javascript 数据结构与映射、归约、过滤](https://levelup.gitconnected.com/write-beautiful-javascript-with-%CE%BB-fp-es6-350cd64ab5bf)**
4.  **[Javascript- Currying VS 部分应用](https://medium.com/datadriveninvestor/javascript-currying-vs-partial-application-4db5b2442be8)**
5.  **[Javascript ES6 —可迭代程序和迭代器](https://medium.com/datadriveninvestor/javascript-es6-iterables-and-iterators-de18b54f4d4)**
6.  **[Javascript —代理](https://medium.com/datadriveninvestor/why-to-use-javascript-proxy-5cdc69d943e3)， [Javascript —作用域](https://medium.com/datadriveninvestor/still-confused-in-js-scopes-f7dae62c16ee)**

****如果您喜欢这篇文章，请随意分享并帮助他人找到它！****

****谢谢！****