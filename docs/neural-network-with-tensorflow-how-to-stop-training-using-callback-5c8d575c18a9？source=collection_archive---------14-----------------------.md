# 如何停止使用回调训练神经网络？

> 原文：<https://towardsdatascience.com/neural-network-with-tensorflow-how-to-stop-training-using-callback-5c8d575c18a9?source=collection_archive---------14----------------------->

![](img/dc9fc48f4ddb2fbc9bf7f6d77414369f.png)

Photo by [Samuel Zeller](https://unsplash.com/@samuelzeller?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

## Tensorflow 和 Keras 的有用工具

# 介绍

通常，当训练一个非常深的神经网络时，一旦训练精度达到某个期望的阈值，我们就想停止训练。因此，我们可以实现我们想要的(最优模型权重)并避免资源浪费(时间和计算能力)。

在这个简短的教程中，让我们学习如何在 Tensorflow 和 Keras 中实现这一点，使用**回调**方法，**4 个简单的步骤**。

# 深潜

```
# Import tensorflow
import tensorflow as tf
```

1.  首先，设置精度阈值，直到您想要训练您的模型。

```
ACCURACY_THRESHOLD = 0.95
```

2.现在实现回调类和函数，在精度达到 ACCURACY_THRESHOLD 时停止训练。

```
# Implement callback function to stop training
# when accuracy reaches e.g. ACCURACY_THRESHOLD = 0.95class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('acc') > ACCURACY_THRESHOLD):   
        print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))   
        self.model.stop_training = True
```

这到底是怎么回事？我们正在通过扩展*TF . keras . callbacks . callback*来创建新的类，并实现 *on_epoch_end()* 方法。这在每个时期结束时调用。接下来，我们获取该时期结束时的精度值，如果它大于我们的阈值，我们将模型的 stop_training 设置为 True。

3.实例化一个 *myCallback* 类的对象。

```
callbacks = myCallback()
```

接下来，按照 TensorFlow 或 Keras 的正常步骤建立一个 DNN 或 Conv 网络模型。当使用 *fit()* 方法训练模型时，将使用我们在上面构建的回调。

4.只需将一个参数作为**callbacks =[<my callback 类的新实例化对象> ]** 传递给 fit()方法。

```
model.fit(x_train, y_train, epochs=20, callbacks=[callbacks])
```

仅此而已！训练时，一旦精度达到在 ACCURACY_THRESHOLD 中设置的值，训练将会停止。

为了将所有这些联系在一起，这里有一个完整的代码片段。

# 结论

凭借我们的想象力，这种方法可以以各种创造性的方式使用，特别是当我们想要运行快速 POC 来测试和验证多个 DNN 架构时。你还能想到什么有趣的用法？请在下面的评论区分享你的想法。