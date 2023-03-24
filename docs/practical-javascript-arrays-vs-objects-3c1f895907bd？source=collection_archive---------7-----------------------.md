# 实用 JavaScript:数组与对象

> 原文：<https://towardsdatascience.com/practical-javascript-arrays-vs-objects-3c1f895907bd?source=collection_archive---------7----------------------->

![](img/befcf36a0fad3d5ad13110f94458d961.png)

Photo by [Meagan Carsience](https://unsplash.com/@mcarsience_photography?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

今天有人问我:“你怎么知道什么时候使用对象，什么时候使用数组？”我在网上找不到能给出我想要的答案的资源，所以…我会成为我想看到的改变。

# TL；博士简介

想想**你的特定数据代表什么**:如果它是一个具有命名属性的单一实体，你需要一个对象。如果是一组相同类型/形状的实体，或者顺序很重要，您可能需要一个数组。

如果还不清楚，想想**你将如何处理数据**:操纵单个属性？大概是反对。对整体数据进行操作，还是过滤和操作大块数据？我猜是一个数组。

此外，如果您正在处理现有数据，并且它已经是一个对象或数组，那么如果没有充分的理由，您可能不会将它转换为另一个对象或数组。

```
// A list of ordered strings is a good case for an array:
const sortedNames = ['Axl', 'Billie', 'Chuck'];// An item with named properties is a good case for an object:
const box = { height: 4, width: 3, color: 'blue' };
```

# 两种类型的集合

数组和对象是将数据收集到一个组中的两种方式。数据可以是原语(字符串、数字、布尔值):

```
const namesArr = ['Danny', 'Donny', 'Joey', 'Jordan', 'Jonathan'];
const userObj = { name: 'Jamie', age: 42 };
```

…或者它们可以由其他数组或对象组成:

```
const usersArr = [{ name: 'Jim', age: 4 }, { name: 'Al', age: 62 }];
const miscObj = { colors: ['orange', 'red'], numbers: [1, 2, 3] };
```

那么，你为什么要选择一个而不是另一个呢？冒着过于简化的风险，它归结为**易用性**和**性能**。

# 插入、删除、迭代、更新

我说的易用性是什么意思？当我们将数据分组在一起时，我们通常希望以某种方式使用它。具体来说，我们希望**添加**元素，**移除**元素，**访问/更新**元素，或者**迭代**元素。

*边注:提问的人正在使用 React，所以不变性是一个问题，这对易用性/可读性有影响。像* `*push(), pop(), splice()*` *等可变方法会使事情变得更简单，但是在这些例子中，我会不变地思考。也有一些不同的方法来实现这些示例中的每一个(例如，spread vs.* `*concat*` *)，但我将只坚持一种方法。*

## **插入**

假设我们有这样一组名字:

```
const names = ['Bob', 'Cate'];
```

我们有一个新的名字，我们想添加一个到两端。轻松点。

```
const namesPlusEnd = [...names, 'Deb'];
// ['Bob', 'Cate', 'Deb'];const namesPlusStart = ['Axl', ...names];
// ['Axl', 'Bob', 'Cate'];
```

但是当我们想在数组中间插入一个名字的时候，我们需要知道索引。我们不能插入东西，除非我们知道*它需要去哪里*，所以如果我们没有索引，我们需要使用`Array.findIndex`找到它，这需要时间来遍历数组。

```
const namesPlusMiddle = [
  ...names.slice(0, 1),
  'Bud',
  ...names.slice(1)
];// ['Bob', 'Bud', 'Cate']
```

另一方面，对象不跟踪顺序，所以在任何地方添加属性都很简单，因为没有开始/中间/结束的概念，并且快速，因为我们不需要迭代:

```
const box = { width: 4, height: 3, color: 'blue' };If we care about immutability:
const newBox = { ...box, id: 42 };Or, if we don't:
box.id = 42;// box/newBox are both now:
// { width: 4, height: 3, color: 'blue', id: 42 };
```

## 删除

移除物品呢？还是那句话，看情况！易于从数组的开头或结尾删除:

```
const colors = ['red', 'green', 'blue'];const colorsWithoutFirst = colors.slice(1);
// ['green', 'blue']const colorsWithoutLast = colors.slice(0, -1);
// ['red', 'green']
```

从中间开始也很容易，但是同样，您需要知道想要删除的索引(在本例中是 index `1`)，或者迭代过滤出值:

```
const colorsMinusMid = [...colors.slice(0, 1), ...colors.slice(2)];
// ['red', 'blue']const colorsMinusGreen = colors.filter(color => color !== 'green');
// ['red', 'blue']
```

就像给对象添加属性一样，无论对象在哪里，删除对象属性都很简单(因为没有什么东西在对象中“哪里”的概念)。

```
Immutably:
const { color, ...colorlessBox } = box;With mutation:
delete box.color;colorlessBox/box are both now:
// { height: 4, width: 3, id: 42 }
```

## 更新

更新-不是一个真正的词。当我们想要更新数组中的元素时，我们可以通过索引来完成，或者如果我们不知道索引，我们可以迭代它，根据元素的值(或者元素的属性)来查找元素。通过迭代进行更新是很常见的，因为我们经常在不知道索引的情况下处理大型数据集，或者在索引可能发生变化的情况下处理动态数据。

```
const fruits = ['apple', 'banana', 'clementine'];const newFruits = [
  ...fruits.slice(0, 1),
  'watermelon',
   ...fruits.slice(1)
];This is a little simpler, and leaves the fruits array unchanged:
const fruitsCopy = fruits.slice();
fruitsCopy[1] = 'watermelon';Or, if we don't know the index:
const newFruits = fruits.map(fruit => {
  if (fruit === 'banana') return 'watermelon';
  return fruit;
});// ['apple', 'watermelon', 'clementine'];
```

同样，更新一个对象要简单得多:

```
const box = { height: 4, width: 3, color: 'blue' };Immutably:
const redBox = { ...box, color: 'red' };Mutably:
box.color = 'red';// box/newBox are both now:
// { height: 4, width: 3, color: 'red' }
```

## 访问元素

如果您只需要获得数组中某个元素的值(不需要更新它)，如果您知道索引就很简单，如果您不知道索引就不会太难(但是您知道一些关于您要寻找的元素的信息):

```
const fruits = ['apple', 'banana', 'clementine'];const secondFruit = fruits[1];
// 'banana'const clementine = fruits.find(fruit => fruit === 'clementine');
// 'clementine'
```

访问对象属性也很容易:

```
const box = { width: 4, height: 3, color: 'blue' };const boxColor = box.color
// 'blue'
```

## 迭代和方法

到目前为止，与对象相比，数组是一种累赘。用单个数组元素做任何事情都需要知道索引，或者需要更多的代码。最后，随着迭代，是时候让数组发光了。当您想成批地对元素进行一些转换时，数组就是为此而设计的:

```
const fruits = ['apple', 'banana', 'clementine'];const capitalFruits = fruits.map(fruit => fruit.toUpperCase());
// ['APPLE', 'BANANA', 'CLEMENTINE']fruits.forEach(fruit => console.log(fruit));
// 'apple'
// 'banana'
// 'clementine'Iteration is common in React:
const FruitsList = props => (
  <ul>
    {props.fruits.map(fruit => <li>{fruit}</li>)}
  </ul>
);
// <ul>
//   <li>apple</li>
//   <li>banana</li>
//   <li>clementine</li>
// </ul>
```

要迭代一个对象，我们唯一真正的选择是一个`for...in`循环，但是(在我看来)通常更简单/更易读的方法是……将它转换成一个数组。`Object.keys/values/entries`遍历键、值或两者，并给我们一个数据数组:

```
const box = { height: 4, width: 3, color: 'blue' };const BoxProperties = ({ box }) => (
  <ul>
    Object.keys(box).map(prop => <li>{prop}: {box[prop]}</li>);
  </ul>
);
// <ul>
//   <li>height: 4</li>
//   <li>width: 3</li>
//   <li>color: blue</li>
// </ul>
```

数组还有其他方法允许您处理数据，而这些方法是对象所没有的:

```
const animalNames = ['ant', 'bird', 'centipede', 'demogorgon'];animalNames.reverse();
// ['demogorgon', 'centipede', 'bird', 'ant']const shortNames = animalNames.filter(name => name.length < 5);
// ['ant', 'bird'];const containsB = animalNames.some(name => name.includes('b'));
// trueconst allAreLong = animalNames.every(name => name.length > 3);
// falseconst totalLetters = animalNames.reduce((total, name) => {
  return total + name.length;
}, 0);
// 26
```

您可以用`for...in`很容易地实现其中的任何一个，但是数组有现成的。

## 表演

速度并不总是一个需要考虑的因素，但是当它是一个需要考虑的因素时，数组和对象之间会有很大的不同。互联网上有大量关于数组与对象性能的资源，但简单来说:当您不知道索引(*线性时间*或 O( *n* ))时，数组操作会更慢，因为您必须迭代每个元素，直到找到您想要使用的元素。如果您确实知道索引并且不变性不是问题，那么您不需要迭代，并且可以快速访问/更新该索引处的元素(*常量时间*，或者 O(1))。对象属性查找/更新/插入/删除发生得很快(也是*常数时间*)，因为属性名称给了你一个参考，所以你不必去寻找你想要的元素。

## **结论**

经验法则是:类似类型的数据组(您需要对其进行排序或者希望对其进行批处理操作)更适合于数组，而单个实体的分组属性更适合于对象。使用正确的数据类型并不总是一个明确的选择，但是你使用每种数据类型越多，在任何给定的情况下哪种数据类型更有意义就越明显。