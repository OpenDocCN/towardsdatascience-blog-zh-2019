# 火炬视觉和迁移学习

> 原文：<https://towardsdatascience.com/torchvision-transfer-learning-1d4778b807cc?source=collection_archive---------23----------------------->

## 试图直接操纵预先训练好的火炬视觉模型

![](img/1faba7eeb90f59b46ecc5eae3cfbf2a7.png)

Photo by [Possessed Photography](https://unsplash.com/@possessedphotography?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

这篇文章可能对刚开始深度学习的人或相对来说不熟悉 PyTorch 的人最感兴趣。这是我最近尝试修改 torchvision 包的 CNN 的经验总结，这些 CNN 已经根据来自 Imagenet 的数据进行了预训练。目的是使多体系结构分类器更容易编程。

众所周知，机器学习实践者可以通过保留预训练模型的最后一层以外的所有层来利用预训练模型，冻结剩余层中的参数，然后将自定义分类器附加到模型的末尾，稍后使用用户数据进行训练。

我在 2019 年 3 月下旬完成了 Udacity 所谓的纳米学位“用 Python 进行人工智能编程”。对于本课程的期末项目，学生必须使用至少两种不同类型的 CNN 架构来正确分类不同类型的植物和野花的照片。

可供学生使用的 CNN 架构由 PyTorch 的 torchvision 模块提供，并在 Imagenet 的图像上进行了预处理。面临的挑战是采用这些不同的预训练 CNN 架构，然后利用迁移学习的概念，将我们自己的利用 PyTorch 的分类层附加到模型的末尾。然后，该分类器将根据互联网上某个来源提供的植物和野花照片进行训练。

为了检查 torchvision 中包含的预训练模型的架构，我使用了 Python 解释器中的以下过程:

```
>>> from torchvision import models                                                                                                                                                                   
>>> model = models.vgg13(pretrained=True)                                                                                                                                                            
>>> model                                                                                                                                                                                            
VGG(                                                                                                                                                                                                 
  (features): Sequential(                                                                                                                                                                            
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                                                                                                                            
    (1): ReLU(inplace=True)                                                                                                                                                                          
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                                                                                                                           
    (3): ReLU(inplace=True)                                                                                                                                                                          
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)                                                                                                                  
[…snip…]                                                                                                             
    (24): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)                                                                                                                 
  )                                                                                                                                                                                                  
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))                                                                                                                                                   
  (classifier): Sequential(                                                                                                                                                                          
    (0): Linear(in_features=25088, out_features=4096, bias=True)                                                                                                                                     
    (1): ReLU(inplace=True)                                                                                                                                                                          
    (2): Dropout(p=0.5, inplace=False)                                                                                                                                                               
    (3): Linear(in_features=4096, out_features=4096, bias=True)                                                                                                                                      
    (4): ReLU(inplace=True)                                                                                                                                                                          
    (5): Dropout(p=0.5, inplace=False)                                                                                                                                                               
    (6): Linear(in_features=4096, out_features=1000, bias=True)                                                                                                                                      
  )                                                                                                                                                                                                  
)
```

请注意，对于 torchvision vgg13 预训练实现，最后一个父模块，即被学生的分类器替换的模块，被命名为“分类器”。

由于任务的目标是使用至少两种不同的预训练 CNN 架构，我看了看 torchvision 模块提供的其他模型。我发现火炬视觉预训练 CNN 模型的分类部分至少有两个不同的名字。除了“分类器”，resnet101 预训练 CNN 使用“fc”(大概代表“完全连接”)作为其最终分类模块。

最终，我的自定义分类层需要使用类似下面的语句连接到预训练的 CNN:

```
model.classifier = my_custom_classifier
```

但是，如果用户从命令行指定一个 resnet101 架构，会发生什么呢？既然架构的名称与“分类器”不同，我该如何用上面的技术覆盖最后一个模块呢？

因为我想在项目中重用尽可能多的代码，所以我有几个选择。第一，我可以为分类模块选择具有相同名称的架构，第二，我可以看看是否可以找到一种方法来实现 ***任何通用架构*** ，并找到一种方法来处理不同架构对分类模块具有不同名称的事实。第三种选择是根据用户选择的模型将自定义分类器附加到用户选择的架构上。比如“如果”用户选择了 resnet101，“那么”我们使用“fc”，否则，我们使用“分类器”。

最终，我选择了第二个选项。我的决定是试图将“分类器”模块名等同于“fc”模块名。换句话说，我选择了做类似这样的事情:

```
model.classifier = model.fc = my_custom_classifier
```

目标是为我的自定义分类器提供多个名称，这样无论我选择什么架构，我都可以使用名称“fc”来引用分类层。这个 ***看起来*** 起作用了，这是我最后一个项目(通过了)用的。

在 2019 年 8 月完成“深度学习”纳米学位项目(我的第二个人工智能相关纳米学位)后，我想回到我最初的“用 Python 进行人工智能编程”项目，花多一点时间熟悉 PyTorch，因为大多数深度学习项目都使用 Tensorflow。我还想添加一个命令行参数，以便用户可以指定分类器可以区分的类别数量，而不是在代码库中固定数量。

很快，我开始研究几个月前完成的项目代码，准备修改它以接受分类器可以从命令行区分的类别数量。

没过多久，我就发现我的“诡计”根本没有达到我的预期目的。代码运行得很好，但不像我最初设计的那样。

代码没有让我用多个名称引用我的自定义分类器，而是在我选择的预训练架构的末尾添加了两个自定义分类器*。例如，我对 vgg13 架构的总结如下:*

```
*VGG(                                                                                                                                                                                                                                               
  (features): Sequential(                                                                                                                                                                                                                          
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                                                                                                                                                                          
    (1): ReLU(inplace=True)                                                                                                                                                                                                                        
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                                                                                                                                                                         
    (3): ReLU(inplace=True)                                                                                                                                                                                                                        
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)                                                                                                                                                                
[…snip…]                                                                                                                                                       
    (24): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)                                                                                                                                                               
  )                                                                                                                                                                                                                                                
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))                                                                                                                                                                                                 
  (classifier): Network(                                                                                                                                                                                                                           
    (hidden_layers): ModuleList(                                                                                                                                                                                                                   
      (0): Linear(in_features=25088, out_features=256, bias=True)                                                                                                                                                                                  
    )                                                                                                                                                                                                                                              
    (output): Linear(in_features=256, out_features=102, bias=True)                                                                                                                                                                                 
    (dropout): Dropout(p=0.2, inplace=False)                                                                                                                                                                                                       
  )                                                                                                                                                                                                                                                
  (fc): Network(                                                                                                                                                                                                                                   
    (hidden_layers): ModuleList(                                                                                                                                                                                                                   
      (0): Linear(in_features=25088, out_features=256, bias=True)                                                                                                                                                                                  
    )                                                                                                                                                                                                                                              
    (output): Linear(in_features=256, out_features=102, bias=True)                                                                                                                                                                                 
    (dropout): Dropout(p=0.2, inplace=False)                                                                                                                                                                                                       
  )                                                                                                                                                                                                                                                
)*
```

*正如您从名为' classifier '和' fc '的模块中看到的，我尝试为自定义分类器提供多个名称的最终结果失败了。它所做的只是将自定义分类模块的两个副本放在预训练的 vgg13 CNN 的末尾。更糟糕的是，我的辍学层出现在了错误的地方。我希望它在隐藏层之后显示出来。回到众所周知的绘图板。*

*我意识到我需要的是一种方法，可以用来操纵 torchvision 模块提供的预训练模型的架构。如果我能够操纵这个架构，我就可以对 torchvision 提供的任何预训练 CNN 的最后一个模块执行相当于“删除”(或者“重命名”)的操作。然后，我想，在预先训练的 CNN 中，分类模块的名称是什么并不重要，我可以随便叫它什么，而不必担心分类模块的名称。更好的是，我不必担心决策代码来处理不同架构的使用。*

*在这一点上，我决定看看 torchvision 模型到底有哪些可用的方法和属性。为了做到这一点，我再次回到解释器:*

```
*>>> from torchvision import models                                                                                                                                                                   
>>> x = models.vgg13(pretrained=True)                                                                                                                                                                
>>> x.<tab><tab>                                                                                                                                                                                               
x.add_module(                 x.cpu(                        x.features(                   x.named_buffers(              x.register_buffer(            x.state_dict(                                  
x.apply(                     x.cuda(                      x.float(                     x.named_children(            x.register_forward_hook(     x.to(                                          
x.avgpool(                   x.double(                    x.forward(                   x.named_modules(             x.register_forward_pre_hook( x.train(                                       
x.buffers(                   x.dump_patches               x.half(                      x.named_parameters(          x.register_parameter(        x.training                                     
x.children(                  x.eval(                      x.load_state_dict(           x.parameters(                x.requires_grad_(            x.type(                                        
x.classifier(                x.extra_repr(                x.modules(                   x.register_backward_hook(    x.share_memory(              x.zero_grad(*
```

*通过查看上面的输出，我学到的第一件事是，每个顶级模块都有对应的方法。例如，每个“特性”、“avgpool”和“分类器”都有相应的方法。其中名称根据火炬视觉模型而不同，因此方法也不同。例如，resnet101 模型没有“classifer”方法。它有一个用于分类器的“fc”方法。*

*我花了一些时间来试验不同的方法，并利用 Python 的帮助工具来探索它们，并提出了我的第一个(尽管是天真的)想法，即使用从 nn 继承的“add_module”方法。模块:*

```
*>>> help(model.add_module)                                                                                                                                                                           
Help on method add_module in module torch.nn.modules.module:                                                                                                                                         

add_module(name, module) method of torch.nn.modules.container.Sequential instance                                                                                                                    
    Adds a child module to the current module.                                                                                                                                                       

    The module can be accessed as an attribute using the given name.                                                                                                                                 

    Args:                                                                                                                                                                                            
        name (string): name of the child module. The child module can be                                                                                                                             
            accessed from this module using the given name                                                                                                                                           
        module (Module): child module to be added to the module.*
```

*根据上面的信息，这似乎正是我所需要的，我对实现进行了如下测试:*

```
*>>> from torchvision import models                                                                                                                                                                   
>>> from torch import nn                                                                                                                                                                             
>>> tv_model = models.vgg13(pretrained=True)                                                                                                                                                         
>>> model = nn.Sequential()                                                                                                                                                                          
>>> tv_model_children = list(tv_model.children())[:-1]                                                                                                                                               
>>> tv_model_children                                                                                                                                                                                
[Sequential(                                                                                                                                                                                         
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                                                                                                                              
  (1): ReLU(inplace=True)                                                                                                                                                                            
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                                                                                                                             
  (3): ReLU(inplace=True)                                                                                                                                                                            
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)                                                                                                                    
[…snip…]                                                                                                                                                                           
  (24): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)                                                                                                                   
), AdaptiveAvgPool2d(output_size=(7, 7))]                                                                                                                                                            
>>>*
```

*请注意，tv_model_children 列表中没有列出“分类器”模块。由于我们最终将把我们自己的分类器附加到上面创建的名为“model”的模型上，这正是我们想要的。*

*现在，让我们将 tv_model_children 列表中的模块添加到我们的新模型中:*

```
*>>> for i in range(len(tv_model_children)):                                                                                                                                                          
...     model.add_module(str(i), tv_model_children[i])                                                                                                                                               
...                                                                                                                                                                                                  
>>> model                                                                                                                                                                                            
Sequential(                                                                                                                                                                                          
  (0): Sequential(                                                                                                                                                                                   
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                                                                                                                            
    (1): ReLU(inplace=True)                                                                                                                                                                          
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                                                                                                                           
    (3): ReLU(inplace=True)                                                                                                                                                                          
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)                                                                                                                  
[…snip…]                        
    (24): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)                                                                                                                 
  )                                                                                                                                                                                                  
  (1): AdaptiveAvgPool2d(output_size=(7, 7))                                                                                                                                                         
)*
```

*此时，一切似乎都如我所愿。我已经有效地从 torchvision vgg13 预训练模型中“删除”了分类器模块。现在，我可以用我选择的任何名称将分类器附加到模型上。我的下一步是在我的程序中复制这种方法，看看它是如何工作的:*

```
*python train.py --arch vgg13 --dropout 0.2 --epochs 7 --gpu --hidden_units 1024 512 256 --learning_rate 0.003 /opt/data/flowersConstructing vgg13 pretrained neural network:                                                                                                                                                        
        Hidden units:  [1024, 512, 256]                                                                                                                                                              

Establishing training, validation and testing loaders...                                                                                                                                             

Training classification layer:                                                                                                                                                                       
        Epochs:  7                                                                                                                                                                                   
        Learning Rate:  0.003                                                                                                                                                                        
        Dropout Prob:  0.2                                                                                                                                                                           
        GPU:  TrueTraceback (most recent call last):                                                                                                                                                                   
  File "train.py", line 38, in <module>                                                                                                                                                              
    checkpoint = trainmodel(args, model, loader_dict)                                                                                                                                                
  File "/home/rsbrownjr/work/test/aikit.py", line 369, in trainmodel                                                                                                                                 
    output = model.forward(images)                                                                                                                                                                   
  File "/home/rsbrownjr/anaconda3/envs/imgclassifier/lib/python3.6/site-packages/torch/nn/modules/container.py", line 92, in forward                                                                 
    input = module(input)                                                                                                                                                                            
  File "/home/rsbrownjr/anaconda3/envs/imgclassifier/lib/python3.6/site-packages/torch/nn/modules/module.py", line 547, in __call__                                                                  
    result = self.forward(*input, **kwargs)                                                                                                                                                          
  File "/home/rsbrownjr/work/test/aikit.py", line 37, in forward                                                                                                                                     
    x = F.relu(linear(x))                                                                                                                                                                            
  File "/home/rsbrownjr/anaconda3/envs/imgclassifier/lib/python3.6/site-packages/torch/nn/modules/module.py", line 547, in __call__                                                                  
    result = self.forward(*input, **kwargs)                                                                                                                                                          
  File "/home/rsbrownjr/anaconda3/envs/imgclassifier/lib/python3.6/site-packages/torch/nn/modules/linear.py", line 87, in forward                                                                    
    return F.linear(input, self.weight, self.bias)                                                                                                                                                   
  File "/home/rsbrownjr/anaconda3/envs/imgclassifier/lib/python3.6/site-packages/torch/nn/functional.py", line 1371, in linear                                                                       
    output = input.matmul(weight.t())                                                                                                                                                                
RuntimeError: size mismatch, m1: [114688 x 7], m2: [25088 x 1024] at /pytorch/aten/src/THC/generic/THCTensorMathBlas.cu:273*
```

*好吧，这可不太管用，不是吗？让我们看看生成的模型，并将其与预训练的 vgg13 架构进行比较，因为异常输出提到了乘法步骤中涉及的张量大小问题。也许在创建模型的过程中有些东西被破坏了？*

```
*Sequential(                                                                                                                                                                                          
  (0): Sequential(                                                                                                                                                                                   
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                                                                                                                            
    (1): ReLU(inplace=True)                                                                                                                                                                          
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                                                                                                                           
    (3): ReLU(inplace=True)                                                                                                                                                                          
[…snip…]                                                                                                                                   
    (24): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)                                                                                                                 
  )                                                                                                                                                                                                  
  (1): AdaptiveAvgPool2d(output_size=(7, 7))                                                                                                                                                         
  (fc): Network(                                                                                                                                                                                     
    (hidden_layers): ModuleList(                                                                                                                                                                     
      (0): Linear(in_features=25088, out_features=1024, bias=True)(1): Dropout(p=0.2, inplace=False)                                                                                                                                   
      (2): Linear(in_features=1024, out_features=512, bias=True)                                                                                                                                     
      (3): Linear(in_features=512, out_features=256, bias=True)                                                                                                                                      
    )                                                                                                                                                                                                                                                                                                                                                          
    (output): Linear(in_features=256, out_features=34, bias=True)                                                                                                                                    
  )                                                                                                                                                                                                  
)*
```

*一切看起来井然有序。我的分类器只有一个名为“fc”的副本，dropout 层现在位于正确的位置，尽管我完成后会有更多的分类器。此时，我被难住了。我去了谷歌的搜索引擎。*

*搜索类似于“删除火炬视觉模型的最后一个模块”的字符串会提供大量线索。有人建议 Python 的“del”函数可以和我想要移除的图层的名称一起使用。这是行不通的，因为 PyTorch 的“Sequential”对象没有“del”方法。许多其他的线索包含了和我自己相似的问题，但是都没有答案。*

*还有其他线索包含一个共同主题的变化。这些线程依赖于使用“children()”方法，但是它们直接绕过了 add_module()方法。例如:*

```
*model = models.resnet152(pretrained=True)
newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
print(newmodel)*
```

*这看起来和我之前尝试的方法非常相似。但是，因为它没有直接使用 add_module 方法，所以我想在我的代码中尝试一下。再次执行程序后，我得到了完全相同的异常输出。这里也不走运。*

*此时，我开始考虑尺寸不匹配的潜在来源。由于我的新模型与预训练模型的架构几乎相同，我开始想知道这两个模型之间有什么不同。当我直接使用 torchvision vgg13 型号时，它工作正常。当我在试图删除分类模块时“复制”它时，它没有。*

*我又回头看了一遍回溯，发现提到了对“forward”的调用。这给了我一个想法。如果仅仅是将模块从一个架构复制到另一个架构，而不包括与模型相关联的 forward()方法的副本，那会怎么样呢？在偶然发现这个想法后，我回到我的代码，添加了下面一行:*

```
*model.forward = tv_model.forward*
```

*再次运行该程序产生了另一个异常，但这一次非常不同，它的内容帮助我认识到:*

```
*Traceback (most recent call last):                                                                                                                                                                   
  File "train.py", line 38, in <module>                                                                                                                                                              
    checkpoint = trainmodel(args, model, loader_dict)                                                                                                                                                
  File "/home/rsbrownjr/work/test/aikit.py", line 371, in trainmodel                                                                                                                                 
    output = model.forward(images)                                                                                                                                                                   
  File "/home/rsbrownjr/anaconda3/envs/imgclassifier/lib/python3.6/site-packages/torchvision/models/vgg.py", line 46, in forward                                                                     
    x = self.classifier(x)                                                                                                                                                                           
  File "/home/rsbrownjr/anaconda3/envs/imgclassifier/lib/python3.6/site-packages/torch/nn/modules/module.py", line 547, in __call__                                                                  
    result = self.forward(*input, **kwargs)                                                                                                                                                          
  File "/home/rsbrownjr/anaconda3/envs/imgclassifier/lib/python3.6/site-packages/torch/nn/modules/container.py", line 92, in forward                                                                 
    input = module(input)                                                                                                                                                                            
  File "/home/rsbrownjr/anaconda3/envs/imgclassifier/lib/python3.6/site-packages/torch/nn/modules/module.py", line 547, in __call__                                                                  
    result = self.forward(*input, **kwargs)                                                                                                                                                          
  File "/home/rsbrownjr/anaconda3/envs/imgclassifier/lib/python3.6/site-packages/torch/nn/modules/linear.py", line 87, in forward                                                                    
    return F.linear(input, self.weight, self.bias)                                                                                                                                                   
  File "/home/rsbrownjr/anaconda3/envs/imgclassifier/lib/python3.6/site-packages/torch/nn/functional.py", line 1369, in linear                                                                       
    ret = torch.addmm(bias, input, weight.t())                                                                                                                                                       
RuntimeError: Expected object of backend CPU but got backend CUDA for argument #4 'mat1'*
```

*请注意，这个异常不仅不同，而且它似乎是基于对一个名为“classifier”而不是“fc”的方法的调用！换句话说，将向前方法转移到我的新模型 ***中，解决了尺寸不匹配的问题*** 。但是 torchvision vgg13 模型内部的前向代码仍然在寻找一个叫做“分类器”而不是“fc”的方法。如果我想让这种方法工作，我就要编写我自己的 forward 方法，以便通过每个不同层的输入产生正确大小的输入，进入我的分类器 ***，其中*** 是分类器的正确名称。不值得。*

*我也开始想知道与火炬视觉 vgg13 模型相关的参数。很可能我也要照顾这些。同样，不值得。*

*作为我尝试操作 torchvision 预训练模型的层的研究结果，我得出的结论是，没有任何方法可以让我将我的方法推广到我的训练代码中的不同网络。*

*我必须考虑代码内部层名的差异，或者干脆选择与最终分类模块同名的 torchvision 架构。我至少可以说，通过这次练习，我学到了很多东西，并享受了这段旅程。*

*我希望你发现这是一份有价值的资料。*