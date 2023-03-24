# 实时视频上的神经类型转换(带有完整的可实现代码)

> 原文：<https://towardsdatascience.com/neural-style-transfer-on-real-time-video-with-full-implementable-code-ac2dbc0e9822?source=collection_archive---------14----------------------->

![](img/1ebc513b8ff91ed8274215fea5a5ec09.png)

Neural Style Transfer

执笔:2019 年 5 月 29 日作者:Sourish Dey

最近几年，我们几乎在生活的每个角落都体验到了计算机视觉的应用——这要归功于海量数据和超强 GPU 的可用性，这使得卷积神经网络(CNN)的训练和部署变得超级容易。在我的[上一篇文章](https://www.linkedin.com/pulse/cnn-application-structured-data-automated-feature-extraction-dey/)中，我讨论了一个这样的商业增值应用。今天围绕机器学习的最有趣的讨论之一是它可能如何影响和塑造我们未来几十年的文化和艺术作品。神经类型转移是[卷积神经网络](https://en.wikipedia.org/wiki/Convolutional_neural_network)最具创造性的应用之一。

![](img/84d05eba5ac91771a584b61b0dce7a30.png)

通过拍摄*内容*图像和*风格*图像，神经网络可以将内容和风格图像重新组合，以有效地创建*艺术(*重组 *)* 图像。虽然有许多像 Prisma 这样的应用程序可以为手机拍摄的图片生成艺术风格，但本文的目的是要立刻理解这个看似困难的概念背后的科学和艺术。这里分享实时可实现的代码。

# 背景-神经类型转移

神经风格转移(Neural Style Transfer)的概念最初是由 Gatys、Ecker 和 Bethge ( [艺术风格的神经算法](https://www.google.com/url?q=https://arxiv.org/abs/1508.06576&sa=D&ust=1544893057277000&usg=AFQjCNFJnIPsTIGXOX8u_iqJR8yCR7_kZQ)*于 2015 年)在开创性的论文[中提出的，演示了一种将一幅图像的艺术风格与另一幅图像的内容相结合的方法。基本思想是采用像 VGG 16(通常为图像分类或对象检测而训练)这样的预训练深度卷积神经网络学习的特征表示，以获得图像风格和内容的单独表示。一旦找到这些表示，我们就试图优化生成的图像，以重新组合不同目标图像的内容和风格。因此，这个概念随机化了纹理、对比度和颜色，同时保留了内容图像的形状和语义特征(中心方面)。虽然它有点类似于颜色转换，但它能够传递纹理(样式)和其他各种扭曲，这是经典颜色滤镜所无法实现的。](https://arxiv.org/abs/1508.06576)*

# 问题陈述——不是优化问题吗？

所以这里的问题陈述给定了一张内容照片 X 和风格照片 Y 我们如何将 Y 的风格转移到内容 X 上生成新的照片 Z 我们如何训练 CNN 处理和优化差异(X 和 Y 之间的差异)以达到一个最优的全局(Z)？

基本概述-神经风格转移([来源](https://www.google.co.in/search?q=neural+style+transfer+applications&tbm=isch&source=lnms&sa=X&ved=0ahUKEwiS2ZTew7PiAhXC4KQKHalkDQ0Q_AUICigB&biw=1536&bih=760&dpr=1.25#imgrc=Z6wgFpT7fEeuBM:))

![](img/1ec4780fdd0b7ab71113dac86d9b5894.png)

基本概述-神经类型转移([来源](https://www.google.co.in/search?q=neural+style+transfer+applications&tbm=isch&source=lnms&sa=X&ved=0ahUKEwiS2ZTew7PiAhXC4KQKHalkDQ0Q_AUICigB&biw=1536&bih=760&dpr=1.25#imgrc=Z6wgFpT7fEeuBM:))

# 优化问题概述

[Gatys](https://arxiv.org/abs/1508.06576) 在原论文([一种艺术风格的神经算法](https://www.google.com/url?q=https://arxiv.org/abs/1508.06576&sa=D&ust=1544893057277000&usg=AFQjCNFJnIPsTIGXOX8u_iqJR8yCR7_kZQ)*2015)中显示“将一幅图像的风格(纹理)转移到另一幅内容图像上，被提出为一个优化问题，可以通过训练一个深度神经网络来解决”。以下是这个拼图的组成部分:*

*   *内容损失:它表示样式转移网络的输出图像(样式化图像)的内容与输入图像或“内容目标”的内容的相似程度，如果输入图像(X)和样式化图像(Z)在内容方面彼此相似，则它趋于零，如果它们不同，则它增大。为了恰当地捕捉图像的内容，我们需要保留空间(形状/边界等。)/内容图像的高级特征。由于像 VGG16 这样的图像分类卷积神经网络被迫学习更深层的高级特征/抽象表示或图像的“内容”,因此对于内容比较，我们在输出(softmax)层之前的某个更深层(L)1 或 2 层使用激活/特征映射。可以根据需要选择 L(从哪个层提取内容特征),选择的层越深，输出图像看起来就越抽象。所以 L 是网络的超参数。*

*![](img/391fb76ed737406f4f43bd326e844cad.png)*

*   *Gram 矩阵和样式损失:虽然稍微复杂一些，但是原始样式图像(Y)和网络输出图像(Z)之间的样式损失也是作为从 VGG-16 的图层输出中提取的要素(激活图)之间的距离来计算的。这里的主要区别在于，不是直接比较来自 VGG-16 的层激活矩阵的特征表示，而是将那些特征表示转换成空间相关矩阵(在激活图内)，这是通过计算 Gram 矩阵来完成的。gram 矩阵包含风格图像层上每对特征图之间的相关性。因此，本质上，Gram matrix 捕获了在图像的不同部分同时出现的特征的趋势。它表示一组向量的内点积，这捕获了两个向量之间的相似性，因为如果两个向量彼此相似，那么它们的点积将会很大，因此 Gram 矩阵也会很大。*

*![](img/0064f2835d8db5323d191eed5869aab1.png)*

*[来源](https://www.google.co.in/search?q=gram+matrix+style+transfer&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjLsJmg1MbiAhXFyKQKHUvKBHEQ_AUIDigB&biw=1536&bih=760#imgrc=BOpxPKsI9KMrfM:)*

*   *在最初的论文中，Gatys 建议结合浅层和深层 conv 层来计算风格表现的风格损失。因此，风格损失是每个 conv 层激活矩阵的原始风格图像(Y)和生成的图像(Z)之间的风格特征的均方差(欧几里德距离)的总和。*

*![](img/e91cc059eb33d99161f7a47dabefd8ec.png)*

*   *总体损失:总体损失是内容损失和风格损失的加权和，如下所示。*

*![](img/7c113ec7d12b8b423254c37b0be68e18.png)*

*   *网络被训练成同时最小化内容损失和风格损失。α和β分别是内容损失和风格损失的权重，也是整个 CNN 的超参数。这些值的选择完全取决于生成的图像(Z)中需要保留多少内容或样式。这里，我们从随机(白噪声)图像矩阵开始，并在每次迭代中计算内容图像(内容损失)和样式图像(样式损失)之间的特征映射距离(总体损失)，以计算总体损失。因此，这种损失通过网络反向传播，我们需要通过适当的优化技术(相当于梯度下降)在迭代(几千次)中最小化总损失，以更新与内容和风格图像一样接近的随机图像矩阵。我们继续这个过程，直到我们得到最小的阈值损失值，以生成优化的混合图像(Z)，它看起来类似于内容(X)和样式图像(Y)。*

*![](img/77edcf341dfcfa6018a6fa07aeec9d3b.png)*

*因此，根据上述典型的 NST 网络，在更深的层计算内容损失，以捕获高级特征(空间)，而对于样式图像，则捕获详细的样式特征(纹理、颜色等)。)风格损失通过提取每个 conv 区块的浅层(conv-1)的网络输出(激活图)来计算。典型的预训练分类 CNN 如 [VGG16](https://neurohive.io/en/popular-networks/vgg16/) 由几个 conv 块组成，其具有 2 或 3 个卷积(Conv2D)层(conv1、conv2 等。)接着是 pooling(最大/平均)层。所以风格图像网络是一个多输出模型。在下一节中，我们将简要讨论这个概念在实时视频数据上的实现。详细代码以及所有输入(内容视频和风格图像)和输出(生成的图像帧)可在[这里](https://github.com/nitsourish/Neural-Style-Transfer-on-video-data)找到。*

# *实时视频上的神经类型转移；*

*我将解释图像的步骤，因为视频只不过是一组图像的集合。这些图像被称为帧，可以组合起来获得原始视频。因此，我们可以循环所有单个帧的步骤，重新组合并生成风格化的视频。*

***第一步:装载预先训练好的 VGG-16 CNN 模型***

*为 NST 应用从零开始构建(训练)CNN 需要大量时间和强大的计算基础设施，这对于个人来说是不容易获得的。*

*因此，我们将加载权重(从著名的' [ImageNet](https://en.wikipedia.org/wiki/ImageNet) 的图像中训练)挑战)一个预先训练好的 CNN- [VGG-16](https://neurohive.io/en/popular-networks/vgg16/) 实现神经风格转移。我们将使用 Keras 应用程序为 VGG-16 加载预先训练好的重量。VGG-16 可能不是 NST 的最佳 CNN 架构。对于这种应用，有更复杂(具有高级架构的更深层次)的网络，如 InceptionV4、VGG-19、Resnet-101 等，这将花费更多的时间来加载和运行。然而，作为实验，我们选择了 VGG-16(具有高分类准确度和对特征的良好内在理解)。*

```
*from keras.applications.vgg16 import VGG16
shape = (224,224)
vgg = VGG16(input_shape=shape,weights='imagenet',include_top=False)*
```

*形状在这里很重要，因为 VGG-16 网络采用形状为 224 x 224 x 3 的输入图像。*

```
*vgg.summary()
**_________________________________________________________________**
Layer (type)                 Output Shape              Param #   
=================================================================
input_21 (InputLayer)        (None, 224, 224, 3)       0         
**_________________________________________________________________**
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
**_________________________________________________________________**
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
**_________________________________________________________________**
average*_pooling2d_*101 (Avera (None, 112, 112, 64)      0         
**_________________________________________________________________**
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
**_________________________________________________________________**
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
**_________________________________________________________________**
average*_pooling2d_*102 (Avera (None, 56, 56, 128)       0         
**_________________________________________________________________**
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
**_________________________________________________________________**
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
**_________________________________________________________________**
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
**_________________________________________________________________**
average*_pooling2d_*103 (Avera (None, 28, 28, 256)       0         
**_________________________________________________________________**
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
**_________________________________________________________________**
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
**_________________________________________________________________**
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
**_________________________________________________________________**
average*_pooling2d_*104 (Avera (None, 14, 14, 512)       0         
**_________________________________________________________________**
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
**_________________________________________________________________**
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
**_________________________________________________________________**
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
**_________________________________________________________________**
average_pooling2d_105 (Avera (None, 7, 7, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0*
```

*VGG-16 建筑*

***第二步:定义内容模型和成本函数***

*对于高级内容特征，我们希望考虑整个图像的特征。所以我们将把 max-pool(可能会丢掉一些信息)换成 average pool。然后，我们将从总共 13 个卷积中选择任何更深的层作为“输出”,并定义该层的模型。然后，我们将在 n/w 中馈送我们的预处理内容图像(X ),以在输出层 wrt 计算(预测的)特征/激活图。该模型和模型输出 wrt 为定义形状(224 x 224 x 3)的任意随机(白噪声)矩阵。我们计算内容图像网络的 MSE 损失和梯度。这将有助于将输入图像(随机图像)更新到梯度的相反方向，并允许内容损失值降低，从而生成的图像将与输入图像的图像相匹配。详细的实现代码保存在我的 [GitHub 库](https://github.com/nitsourish/Neural-Style-Transfer-on-video-data)中。*

```
*content_model = vgg_cutoff(shape, 13) *#Can be experimented with other deep layers*
*# make the target*
target = K.variable(content_model.predict(x))
*# try to match the input image*
*# define loss in keras*
loss = K.mean(K.square(target - content_model.output))
*# gradients which are needed by the optimizer*
grads = K.gradients(loss, content_model.input)*
```

***第三步:定义风格模型和风格损失函数***

*两个图像的特征映射在给定层产生相同的 Gram 矩阵，我们期望两个图像具有相同的风格(但不一定是相同的内容)。因此，网络中早期层的激活图将捕获一些更精细的纹理(低级特征)，而更深层的激活图将捕获图像风格的更多高级元素。因此，为了获得最佳结果，我们将浅层和深层的组合作为输出，以比较图像的样式表示，并相应地定义多输出模型。*

*这里，我们首先计算每一层的 Gram 矩阵，并计算风格网络的总风格损失。我们对不同的层采用不同的权重来计算加权损失。然后，基于样式损失(样式分量的差异)和梯度，我们更新输入图像(随机图像)并减少样式损失值，使得生成的图像(Z)纹理看起来类似于样式图像(Y)的纹理。*

```
**#Define multi-output model*
symb_conv_outputs = [layer.get_output_at(1) for layer in vgg.layers\
if layer.name.endswith('conv1')]
multi_output_model = Model(vgg.input, symb_conv_outputs)
*#Style feature map(outputs) of style image*
symb_layer_out = [K.variable(y) for y in multi_output_model.predect(x)] *#Defining Style loss*
def gram_matrix(img):
    X = K.batch_flatten(K.permute_dimensions(img,(2,0,1)))
    gram_mat = K.dot(X,K.transpose(X))/img.get_shape().num_elements()
    return gram_mat def style_loss(y,t):
    return K.mean(K.square(gram_matrix(y)-gram_matrix(t))) *#Style loss calculation through out the network*
*#Defining layer weights for layers* 
weights = [0.2,0.4,0.3,0.5,0.2]
loss=0
for symb,actual,w in zip(symb_conv_outputs,symb_layer_out,weights):
    loss += w * style_loss(symb[0],actual[0])
grad = K.gradients(loss,multi_output_model.input)
get_loss_grad = K.Function(inputs=[multi_output_model.input], outputs=[loss] + grad)*
```

***第四步:定义总成本(总损失):***

*现在我们可以结合内容和风格损失来获得网络的整体损失。我们需要使用合适的优化算法在迭代过程中最小化这个数量。*

```
*#Content Loss
loss=K.mean(K.square(content_model.output-content_target)) * Wc #Wc is content loss weight(hyperparameter)#Defining layer weights of layers for style loss 
weights = [0.2,0.4,0.3,0.5,0.2]#Total loss and gradient
for symb,actual,w in zip(symb_conv_outputs,symb_layer_out,weights):
    loss += Ws * w * style_loss(symb[0],actual[0]) #Wc is content loss weight(hyperparameter)

grad = K.gradients(loss,vgg.input)
get_loss_grad = K.Function(inputs=[vgg.input], outputs=[loss] + grad)*
```

***第五步:求解优化问题和损失最小化函数***

*在定义了整个符号计算之后，图的优化算法是主要的组成部分，它将能够迭代地最小化整个网络的成本。这里不用使用 keras 标准的优化器函数(如 optimizer。Adam、optimizers.sgd 等。)，这可能需要更多时间，我们将使用[有限内存 BFGS(Broyden–Fletcher–gold farb–Shanno)，](https://en.wikipedia.org/wiki/Limited-memory_BFGS)这是一种使用有限计算机内存的近似数值优化算法。由于它的线性内存需求，这种方法非常适合于涉及大量变量(参数)的优化问题。像正常的 BFGS 它是一个标准的拟牛顿法，通过最大化正则化对数似然优化平滑函数。*

*Scipy 的最小化函数(fmin_l_bfgs_b)允许我们传回函数值 f(x)及其梯度 f'(x)，这是我们在前面的步骤中计算的。但是，我们需要以一维数组格式展开最小化函数的输入，并且损耗和梯度都必须是 np.float64。*

```
**#Wrapper Function to feed loss and gradient with proper format to L-BFGS* 
def get_loss_grad_wrapper(x_vec):
        l,g = get_loss_grad([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)
*#Function to minimize loss and iteratively generate the image*
def min_loss(fn,epochs,batch_shape):
    t0 = datetime.now()
    losses = []
    x = np.random.randn(np.prod(batch_shape))
    for i in range(epochs):
        x, l, _ = scipy.optimize.fmin_l_bfgs_b(func=fn,x0=x,maxfun=20)
    *# bounds=[[-127, 127]]*len(x.flatten())*
    *#x = np.clip(x, -127, 127)*
    *# print("min:", x.min(), "max:", x.max())*
        print("iter=%s, loss=%s" % (i, l))
        losses.append(l)
    print("duration:", datetime.now() - t0)
    plt.plot(losses)
    plt.show()
    newimg = x.reshape(*batch_shape)
    final_img = unpreprocess(newimg)
    return final_img[0]*
```

***步骤 6:对输入内容和样式图像运行优化器功能:***

*在输入内容帧和样式图像上运行优化器，并且按照定义的符号计算图，网络执行其预期的最小化总体损失的工作，并且生成看起来尽可能接近内容和样式图像的图像。*

*![](img/aec26760d72a9a6d29c89dc235d0e826.png)*

*输出图像仍然有噪声，因为我们仅运行网络 30 次迭代。理想的 NST 网络应该经过数千次迭代优化，以达到最小损耗阈值，从而产生清晰的混合输出。*

***步骤 7:对所有图像帧重复上述步骤:***

*从短视频中提取帧后，对每一帧进行网络推理，为每一帧生成风格化图像，并对风格化图像帧进行重组/拼接。*

```
**#Vedio Reading and extracting frames*cap = cv2.VideoCapture(path)
while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(224,224))
    X = preprocess_img(frame) *#Running the above optimization as per defined comutation graph and generate styled image frame#* 
    final_img = min_loss(fn=get_loss_grad_wrapper,epochs=30,batch_shape=batch_shape)    
    plt.imshow(scale(final_img))
    plt.show()
    cv2.imwrite(filename, final_img)*#Recombine styled image frames to form the video*
video = cv2.VideoWriter(video_name, 0, 1, (width,height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
cv2.destroyAllWindows()
video.release()*
```

*我们也可以使用设备相机胶卷来尝试视频，并尝试在线模式下的风格转换(实时视频)，只需调整视频捕捉模式。*

```
*cap = cv2.VideoCapture(0)
cap.release()*
```

# *商业应用*

*除了这种看似奇特的技术的个人和艺术用途，神经风格转移有可能改变人类创造力传统上占主导地位的任何行业，如美术、时装、建筑或新时尚汽车纹理设计等。以时装业为例，该行业需要对时尚的机制有深刻的理解:趋势的成因和传播，循环重复的原则和进化模式，以发展未来的时尚。*

*![](img/a14e7ca06a16a488de9d00d49039ef5f.png)*

*[图片提供](https://labs.eleks.com/2016/09/designing-apparel-neural-style-transfer.html)*

*然而，神经网络或 NST 可以通过自动为不同类型的服装分配形状、元素和创意纹理(风格)来帮助设计新的设计，并进一步将它们结合起来，以生产出未来的流行时尚。通过自动化 NST 的重要部分，有可能大大减少服装设计过程。为了进一步了解，我鼓励你阅读[这篇文章](https://labs.eleks.com/2016/09/designing-apparel-neural-style-transfer.html)。*

# *进一步的改进和实验:*

*以下是一些提高生成图像质量的策略:*

*1)更多次迭代:显然，运行网络更多次迭代(大约 1000 次)将减少总体损失，并将创建更清晰的混合图像。*

*2)高级 CNN 架构:对于 NST 应用，通常具有非常高级连接的更深的神经网络可以更准确地捕捉高级(空间)和详细的纹理特征。因此，值得尝试其他优秀的预训练网络，如 InceptionV4、GoogLeNet、Resnet-101 等。然而，对于具有数千次迭代 NST 应用，这些网络的运行时间非常高，且需要昂贵的计算基础设施，如强大的 GPU 栈。*

*3)调整内容和风格损失权重:作为一项实验，我尝试分别使用 4 和 0.03 作为内容和风格损失权重，主要是为了尽可能多地捕捉内容(因为我只运行了几次网络迭代)。但是，这可能不合适，找到最佳权重的理想方法是通过网格搜索*

*4)针对样式损失调整层权重:为了最大化样式特征捕获，我们需要调整各个 conv 层中的权重以控制样式损失计算，从而优化纹理的提取(早期层的较精细纹理和较深层的较高级特征)。同样，这些是超参数，网格搜索是理想的选择。*

*此外，我们可以使用层(L)来提取内容特征。l 也是网络的超参数。*