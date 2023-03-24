# 使用 TensorFlow 2.0 和 Keras API

> 原文：<https://towardsdatascience.com/getting-your-hands-dirty-with-tensorflow-2-0-and-keras-api-cc8579eb0915?source=collection_archive---------12----------------------->

## 深入研究使用 TensorFlow 2.0 和 Keras API 创建回归模型的技术细节。在 TensorFlow 2.0 中，Keras 自带 TensorFlow 库。API 得到了简化，使用起来更加方便。

![](img/167756bc6cf5cb38086db06314d7c91a.png)

Source: Pixabay

TensorFlow 2.0 内部打包了 Keras，没有必要将 Keras 作为单独的模块导入(虽然如果需要也可以这样做)。TensorFlow 2.0 API 进行了简化和改进。这对我们——机器学习开发者来说是个好消息。

现在，您可以这样从 TensorFlow 导入 Keras:

```
from tensorflow import feature_column
from tensorflow import keras
from tensorflow.keras import layersprint(tf.__version__)
2.0.0
```

我正在使用 [tf.data](https://www.tensorflow.org/guide/data#consuming_csv_data) 输入管道来编码分类列，Keras API 与 tf.data 配合得很好。tf.data 的一个主要优势是它充当了数据和模型之间的桥梁。无需自己转换数据，只需定义转换规则，转换后的数据将自动应用于训练。

数据从 CSV 文件提取到 Pandas 数据帧:

```
column_names = ['report_id','report_params','day_part','exec_time']
raw_dataframe = pd.read_csv('report_exec_times.csv')
dataframe = raw_dataframe.copy()dataframe.head()
```

*report_params* 的列值各不相同，我们需要对该列进行规范化(使值处于相似的范围内):

```
eps=0.001 # 0 => 0.1¢
dataframe['report_params'] = np.log(dataframe.pop('report_params')+eps)
```

我使用一个实用的方法(这个方法取自 TensorFlow [教程](https://www.tensorflow.org/tutorials/structured_data/feature_columns))从熊猫数据帧创建 tf.data 数据集:

```
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('exec_time')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds
```

接下来，我们需要为分类列编码定义数据映射。我用的是 TensorFlow 词汇列表函数，包括所有唯一值的映射(如果值很多，用嵌入 API 会更好)。两列被编码— *报告 _id* 和*日 _ 部分*:

```
feature_columns = []feature_columns.append(feature_column.numeric_column('report_params'))report_id = feature_column.categorical_column_with_vocabulary_list('report_id', ['1', '2', '3', '4', '5'])
report_id_one_hot = feature_column.indicator_column(report_id)
feature_columns.append(report_id_one_hot)day_part = feature_column.categorical_column_with_vocabulary_list('day_part', ['1', '2', '3'])
day_part_one_hot = feature_column.indicator_column(day_part)
feature_columns.append(day_part_one_hot)
```

使用 TensorFlow 编码在数组外创建 Keras 密集要素图层。我们将在 Keras 模型构建期间使用该层来定义模型训练特征:

```
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
```

我们已经完成了功能。接下来，在实用函数的帮助下，将 Pandas dataframe 转换为 tf.data:

```
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
```

定义 Keras 顺序模型时使用密集特征层(无需将特征数组传递到 *fit* 函数中):

```
def build_model(feature_layer):
  model = keras.Sequential([
    feature_layer,
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
  ]) optimizer = keras.optimizers.RMSprop(0.001) model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
```

通过 *model.fit* 功能执行训练。我们使用 tf.data 输入管道来通过训练和验证集:

```
history = model.fit(train_ds,
              validation_data=val_ds,
              epochs=EPOCHS,
              callbacks=[early_stop])
```

令人惊叹的是，数据编码发生在幕后，基于为要素图层定义的规则。

如何用 tf.data 和 feature 图层运行 *model.predict* 函数？很简单。

用输入数据构建熊猫数据框架:

```
headers = ['report_id', 'report_params', 'day_part']
dataframe_input = pd.DataFrame([[1, 15, 3]],
                                columns=headers, 
                                dtype=float,
                                index=['input'])
```

将 *report_params* 值转换为与训练时相同的比例:

```
eps=0.001 # 0 => 0.1¢
dataframe_input['report_params'] = np.log(dataframe_input.pop('report_params')+eps)
```

从 Pandas 数据帧创建 tf.data 输入管道:

```
input_ds = tf.data.Dataset.from_tensor_slices(dict(dataframe_input))
input_ds = input_ds.batch(1)
```

运行*模型预测*功能:

```
res = model.predict(input_ds)
print(res)
```

资源:

*   带有示例数据的源代码可以在我的 [GitHub](https://github.com/abaranovskis-redsamurai/automation-repo/tree/master/tf2.0) repo 上获得

尽情享受吧！