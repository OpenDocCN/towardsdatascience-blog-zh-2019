# 这是伟大的泵，查理·布朗

> 原文：<https://towardsdatascience.com/its-the-great-pumpgan-charlie-brown-blog-exxact-bfd159d2c1fe?source=collection_archive---------22----------------------->

# 进入这个南瓜发电机教程甘斯的万圣节精神

![](img/f1794602ad1b0aeb4e663d6251ed14ea.png)

(生成性对抗性训练。蓝线表示输入流，绿线表示输出，红线表示错误信号。)

生成对抗网络，简称 GANs，是过去 10 年中出现的最令人兴奋的深度学习领域之一。这是根据 MNIST 的 Yann LeCun 和反向传播的名声得出的。自 2014 年伊恩·古德费勒(Ian Goodfellow)等人引入 GANs 以来，对抗训练取得了快速进展，这标志着对抗训练是一种突破性的想法，完全有可能以有益、邪恶和愚蠢的方式改变社会。甘的训练被用于各种场合，从可预测的[猫](https://github.com/AlexiaJM/Deep-learning-with-cats) [发电机](https://ajolicoeur.wordpress.com/cats/)到在艺术品拍卖会上以[六位数](https://www.artnome.com/news/2018/10/13/the-ai-art-at-christies-is-not-what-you-think)卖出的甘“画”出来的虚构肖像画家。所有的 GANs 都基于决斗网络的简单前提:一个生成某种输出数据(在我们的例子中是图像)的创造性网络和一个输出数据是真实的或生成的概率的怀疑性网络。这些被称为“生成器”和“鉴别器”网络，通过简单地试图阻挠对方，它们可以学习生成真实的数据。在本教程中，我们将基于流行的全卷积 DCGAN 架构构建一个 GAN，并训练它为万圣节制作南瓜。

我们将使用 [PyTorch](https://www.exxactcorp.com/PyTorch) ，但是您也可以使用 [TensorFlow](https://www.exxactcorp.com/TensorFlow) (如果您对它感到舒服的话)。使用任何一个主要深度学习库的体验都变得惊人地相似，鉴于今年 [2.0 版本](https://github.com/tensorflow/tensorflow/releases/tag/v2.0.0) [中对 TensorFlow 的](https://blog.exxactcorp.com/tensorflow-2-0-dynamic-readable-and-highly-extended/)[更改，我们已经看到这是流行框架与动态执行的 pythonic 代码更广泛融合的一部分，具有可选的优化图形编译以加速和部署。](https://blog.exxactcorp.com/tensorflow-2-0-dynamic-readable-and-highly-extended/)

要为基本 PyTorch 实验设置和激活虚拟环境:

```
virtualenv pytorch --python=python3 pytorch/bin/pip install numpy matplotlib torch torchvision source pytorch/bin/activate
```

如果您安装了 conda，并且喜欢使用它:

```
conda new -n pytorch numpy matplotlib torch torchvision conda activate pytorch
```

为了节省您猜测的时间，下面是我们需要的导入:

```
import random import time import numpy as np import matplotlib.pyplot as plt import torch import torch.nn as nn import torch.nn.parallel import torch.optim as optim import torch.nn.functional as F import torch.utils.data import torchvision.datasets as dset import torchvision.transforms as transforms import torchvision.utils as vutils
```

我们的 GAN 将基于 [DCGAN 架构](https://arxiv.org/abs/1511.06434)，并大量借鉴 [PyTorch 示例](https://github.com/pytorch/examples)中的官方实现。“DCGAN”中的“DC”代表“深度卷积”，DCGAN 架构扩展了 Ian Goodfellow 的原始 [GAN 论文](https://arxiv.org/abs/1406.2661)中描述的无监督对抗训练协议。这是一个相对简单易懂的网络架构，可以作为测试更复杂想法的起点。

与所有 GAN 一样，DCGAN 架构实际上由两个网络组成，即鉴别器和发生器。重要的是要保持它们在力量、训练速度等方面的平衡。以避免网络变得不匹配。众所周知，GAN 训练是不稳定的，可能需要相当多的调整才能使其在给定的数据集架构组合上工作。在这个 DCGAN 示例中，很容易陷入生成器输出黄色/橙色棋盘格乱码的困境，但是不要放弃！总的来说，我非常钦佩像这样的小说突破的作者，他们很容易因早期的糟糕结果而气馁，可能需要英雄般的耐心。话又说回来，有时这只是一个充分准备和一个好主意的问题，只需要[几个小时的额外工作和计算](https://youtu.be/pWAc9B2zJS4?t=214)事情就解决了。

生成器是一堆转置的卷积层，将一个细长的多通道张量潜在空间转换为全尺寸图像。DCGAN 论文中的下图对此进行了举例说明:

![](img/4a264beddc2ce1abb3cf44cc5bea9f2e.png)

来自拉德福德的全卷积生成器*等* 2016。

我们将实例化为`torch.nn.Module`类的子类。这是一种实现和开发模型的灵活方式。您可以在`forward`类函数中植入种子，允许合并像跳过连接这样的东西，这在简单的`torch.nn.Sequential`模型实例中是不可能的。

```
class Generator(nn.Module): def __init__(self, ngpu, dim_z, gen_features, num_channels): super(Generator, self).__init__() self.ngpu = ngpu self.block0 = nn.Sequential(\ nn.ConvTranspose2d(dim_z, gen_features*32, 4, 1, 0, bias=False),\ nn.BatchNorm2d(gen_features*32),\ nn.ReLU(True)) self.block1 = nn.Sequential(\ nn.ConvTranspose2d(gen_features*32,gen_features*16, 4, 2, 1, bias=False),\ nn.BatchNorm2d(gen_features*16),\ nn.ReLU(True)) self.block2 = nn.Sequential(\ nn.ConvTranspose2d(gen_features*16,gen_features*8, 4, 2, 1, bias=False),\ nn.BatchNorm2d(gen_features*8),\ nn.ReLU(True)) self.block3 = nn.Sequential(\ nn.ConvTranspose2d(gen_features*8, gen_features*4, 4, 2, 1, bias=False),\ nn.BatchNorm2d(gen_features*4),\ nn.ReLU(True)) self.block5 = nn.Sequential(\ nn.ConvTranspose2d(gen_features*4, num_channels, 4, 2, 1, bias=False))\ def forward(self, z): x = self.block0(z) x = self.block1(x) x = self.block2(x) x = self.block3(x) x = F.tanh(self.block5(x)) return x
```

是我们“甘多”中有创造力的那一半，而创造看似新奇的图像的习得能力是大多数人倾向于关注的。事实上，如果没有一个匹配良好的鉴别器，发电机是没有希望的。鉴别器架构对于那些过去构建过一些深度卷积图像分类器的人来说应该很熟悉。在这种情况下，它是一个二进制分类器，试图区分真假，所以我们在输出中使用 sigmoid 激活函数，而不是我们在多类问题中使用的 softmax。我们也去掉了所有完全连接的层，因为它们在这里是不必要的。

![](img/3d7792327121a7977ffb0b45e6bea5c0.png)

全卷积二进制分类器适合用作鉴别器***【D(x)***。

代码是:

```
class Discriminator(nn.Module): def __init__(self, ngpu, gen_features, num_channels): super(Discriminator, self).__init__() self.ngpu = ngpu self.block0 = nn.Sequential(\ nn.Conv2d(num_channels, gen_features, 4, 2, 1, bias=False),\ nn.LeakyReLU(0.2, True)) self.block1 = nn.Sequential(\ nn.Conv2d(gen_features, gen_features, 4, 2, 1, bias=False),\ nn.BatchNorm2d(gen_features),\ nn.LeakyReLU(0.2, True)) self.block2 = nn.Sequential(\ nn.Conv2d(gen_features, gen_features*2, 4, 2, 1, bias=False),\ nn.BatchNorm2d(gen_features*2),\ nn.LeakyReLU(0.2, True)) self.block3 = nn.Sequential(\ nn.Conv2d(gen_features*2, gen_features*4, 4, 2, 1, bias=False),\ nn.BatchNorm2d(gen_features*4),\ nn.LeakyReLU(0.2, True)) self.block_n = nn.Sequential( nn.Conv2d(gen_features*4, 1, 4, 1, 0, bias=False),\ nn.Sigmoid()) def forward(self, imgs): x = self.block0(imgs) x = self.block1(x) x = self.block2(x) x = self.block3(x) x = self.block_n(x) return x
```

我们还需要一些帮助函数来创建数据加载器，并根据 DCGAN 论文中的建议初始化模型权重。下面的函数返回一个 PyTorch 数据加载器，带有一些轻微的图像增强，只需将它指向包含您的图像的文件夹。我正在处理来自 [Pixabay](https://pixabay.com/photos/search/pumpkin/) 的相对较少的一批免费图像，因此图像放大对于从每张图像中获得更好的效果是很重要的。

```
def get_dataloader(root_path): dataset = dset.ImageFolder(root=root_path,\ transform=transforms.Compose([\ transforms.RandomHorizontalFlip(),\ transforms.RandomAffine(degrees=5, translate=(0.05,0.025), scale=(0.95,1.05), shear=0.025),\ transforms.Resize(image_size),\ transforms.CenterCrop(image_size),\ transforms.ToTensor(),\ transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),\ ])) dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,\ shuffle=True, num_workers=num_workers) return dataloader
```

并初始化权重:

```
def weights_init(my_model): classname = my_model.__class__.__name__ if classname.find("Conv") != -1: nn.init.normal_(my_model.weight.data, 0.0, 0.02) elif classname.find("BatchNorm") != -1: nn.init.normal_(my_model.weight.data, 1.0, 0.02) nn.init.constant_(my_model.bias.data, 0.0)
```

这就是函数和类的全部内容。现在剩下的就是用一些脚本将它们联系在一起(以及对超参数的无情迭代)。将超参数分组在脚本顶部附近是一个好主意(或者用 flags 或 argparse 将它们传入)，这样很容易更改值。

```
# ensure repeatability my_seed = 13 random.seed(my_seed) torch.manual_seed(my_seed) # parameters describing the input latent space and output images dataroot = "images/pumpkins/jacks" num_workers = 2 image_size = 64 num_channels = 3 dim_z = 64 # hyperparameters batch_size = 128 disc_features = 64 gen_features = 64 disc_lr = 1e-3 gen_lr = 2e-3 beta1 = 0.5 beta2 = 0.999 num_epochs = 5000 save_every = 100 disp_every = 100 # set this variable to 0 for cpu-only training. This model is lightweight enough to train on cpu in a few hours. ngpu = 2
```

接下来，我们实例化模型和数据加载器。我使用双 GPU 设置来快速评估一些不同的超参数迭代。在 PyTorch 中，通过将模型包装在`torch.nn.DataParallel`类中，在几个 GPU 上进行训练是微不足道的。如果您的所有 GPU 都被束缚在对人工通用智能的追求中，请不要担心，这种模型足够轻量级，可以在合理的时间内(几个小时)训练 CPU。

```
dataloader = get_dataloader(dataroot) device = torch.device("cuda:0" if ngpu > 0 and torch.cuda.is_available() else "cpu") gen_net = Generator(ngpu, dim_z, gen_features, \ num_channels).to(device) disc_net = Discriminator(ngpu, disc_features, num_channels).to(device) # add data parallel here for >= 2 gpus if (device.type == "cuda") and (ngpu > 1): disc_net = nn.DataParallel(disc_net, list(range(ngpu))) gen_net = nn.DataParallel(gen_net, list(range(ngpu))) gen_net.apply(weights_init) disc_net.apply(weights_init)
```

发生器和鉴别器网络在一个大循环中一起更新。在此之前，我们需要定义我们的损失标准(二进制交叉熵)，为每个网络定义优化器，并实例化一些我们将用来跟踪训练进度的列表。

```
criterion = nn.BCELoss() # a set sample from latent space so we can unambiguously monitor training progress fixed_noise = torch.randn(64, dim_z, 1, 1, device=device) real_label = 1 fake_label = 0 disc_optimizer = optim.Adam(disc_net.parameters(), lr=disc_lr, betas=(beta1, beta2)) gen_optimizer = optim.Adam(gen_net.parameters(), lr=gen_lr, betas=(beta1, beta2)) img_list = [] gen_losses = [] disc_losses = [] iters = 0
```

# 训练循环

训练循环在概念上是简单明了的，但是用一个片段来概括有点长，所以我们将把它分成几个部分。概括地说，我们首先基于对一组真实的和生成的图像的预测来更新鉴别器。然后，我们将生成的图像馈送到新更新的鉴别器，并使用来自***【D(G(z))***的分类输出作为生成器的训练信号，使用真实标签作为目标。

首先，我们将进入循环并执行鉴别器更新:

```
t0 = time.time() for epoch in range(num_epochs): for ii, data in enumerate(dataloader,0): # update the discriminator disc_net.zero_grad() # discriminator pass with real images real_cpu = data[0].to(device) batch_size= real_cpu.size(0) label = torch.full((batch_size,), real_label, device=device) output = disc_net(real_cpu).view(-1) disc_real_loss = criterion(output,label) disc_real_loss.backward() disc_x = output.mean().item() # discriminator pass with fake images noise = torch.randn(batch_size, dim_z, 1, 1, device=device) fake = gen_net(noise) label.fill_(fake_label) output = disc_net(fake.detach()).view(-1) disc_fake_loss = criterion(output, label) disc_fake_loss.backward() disc_gen_z1 = output.mean().item() disc_loss = disc_real_loss + disc_fake_loss disc_optimizer.step()
```

请注意，我们还记录了假批次和真批次的平均预测值。这将通过告诉我们每次更新后预测如何变化，给我们一个简单的方法来跟踪我们的训练有多平衡。

接下来，我们将使用真实标签和二进制交叉熵损失，基于鉴别器的预测来更新生成器。请注意，我们是基于鉴别器将假图像误分类为真实图像来更新生成器的。与最小化鉴别器直接检测假货的能力相比，该信号为训练产生更好的梯度。令人印象深刻的是，甘斯最终能够学会制作基于这种决斗损失信号的[真实感内容](https://arxiv.org/abs/1710.10196)。

```
# update the generator gen_net.zero_grad() label.fill_(real_label) output = disc_net(fake).view(-1) gen_loss = criterion(output, label) gen_loss.backward() disc_gen_z2 = output.mean().item() gen_optimizer.step()
```

最后，有一点内务管理来跟踪我们的训练。平衡 GAN 训练是一门艺术，仅仅从数字上并不总是显而易见你的网络是否有效地学习，所以偶尔检查图像质量是一个好主意。另一方面，如果 print 语句中的任何值变为 0.0 或 1.0，那么您的训练很可能已经失败，使用新的超参数进行迭代是一个好主意。

```
if ii % disp_every == 0: # discriminator pass with fake images, after updating G(z) noise = torch.randn(batch_size, dim_z, 1, 1, device=device) fake = gen_net(noise) output = disc_net(fake).view(-1) disc_gen_z3 = output.mean().item() print("{} {:.3f} s |Epoch {}/{}:\tdisc_loss: {:.3e}\tgen_loss: {:.3e}\tdisc(x): {:.3e}\tdisc(gen(z)): {:.3e}/{:.3e}/{:.3e}".format(iters,time.time()-t0, epoch, num_epochs, disc_loss.item(), gen_loss.item(), disc_x, disc_gen_z1, disc_gen_z2, disc_gen_z3)) disc_losses.append(disc_loss.item()) gen_losses.append(gen_loss.item()) if (iters % save_every == 0) or \ ((epoch == num_epochs-1) and (ii == len(dataloader)-1)): with torch.no_grad(): fake = gen_net(fixed_noise).detach().cpu() img_list.append(vutils.make_grid(fake, padding=2, normalize=True).numpy()) np.save("./gen_images.npy", img_list) np.save("./gen_losses.npy", gen_losses) np.save("./disc_losses.npy", disc_losses) torch.save(gen_net.state_dict(), "./generator.h5") torch.save(disc_net.state_dict(), "./discriminator.h5") iters += 1
```

可能需要一点努力才能得到可接受的结果，但幸运的是，现实中的小故障实际上更适合放大令人毛骨悚然的效果。这里描述的代码还可以改进，但是对于低分辨率下看似合理的南瓜灯应该是不错的。

![](img/d76418a7532bd120f8796345cb1060ae.png)

大约 5000 次更新后的训练进度。

希望上面的教程能激起你对长袍和/或万圣节工艺品的兴趣。掌握了我们在这里构建的基本 DCGAN 之后，尝试更复杂的架构和应用程序。训练 GANs 在很大程度上仍然是一门艺术，平衡训练是棘手的。使用来自 [ganhacks](https://github.com/soumith/ganhacks) 的提示，在获得适用于您的数据集/应用程序/想法的简化概念证明后，一次只添加一小块复杂性。祝你好运，训练愉快。

*原载于 2019 年 10 月 28 日*[*【https://blog.exxactcorp.com】*](https://blog.exxactcorp.com/its-the-great-pumpgan-charlie-brown/)*。*