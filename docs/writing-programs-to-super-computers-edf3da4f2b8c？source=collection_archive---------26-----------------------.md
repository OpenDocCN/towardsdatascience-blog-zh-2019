# 为超级计算机编写程序

> 原文：<https://towardsdatascience.com/writing-programs-to-super-computers-edf3da4f2b8c?source=collection_archive---------26----------------------->

## 何时、何事以及如何改变你的代码。

![](img/51ee27c3bf8abcf85c785d0232036d6c.png)

Server Racks ([Image Source](https://pixabay.com/photos/supercomputer-mainframe-mira-1781372/))

# 抽象而不精确的印象

为了给超级计算机一个概念，想象一整套强大的机器，你可以把它们作为一个整体来使用。够奇怪吗？它不一定是一台拥有大量内存或 CPU 能力的机器。事实上，它们的确拥有强大的计算能力，但不仅仅是通过提高速度或内存。让我们来看看 [NCI](https://nci.org.au/our-systems/hpc-systems) (我目前正好与之合作)。这些是在 [NCI](https://nci.org.au/our-systems/hpc-systems) 的规格。

*   4，500 个英特尔至强 Sandy Bridge、Broadwell 和 Skylake 节点中的 89，256 个内核
*   32 个 NVIDIA Tesla K80 和 P100 节点中的 128 个 GPU
*   32 个节点中的 32 个 64 核英特尔至强融核处理器
*   4 个 IBM Power8 节点中的 64 个内核
*   300 兆字节的内存
*   8pb 的操作磁盘存储
*   混合 FDR/EDR Mellanox Infiniband 全胖树互连(高达 100 Gb/秒)

由于这是一个完整的计算设施，澳大利亚各地成千上万的用户使用，程序被接受以作业的形式运行，在特定的项目拨款(通常由研究机构资助)下提交给队列。不同的队伍收费不同。但是基本上是按 CPU 小时收费，乘以队列成本(由队列规格和优先级决定)。如果你有 1 个 CPU 程序运行 1 小时；你消耗 1 个服务单位乘以排队成本([如果你喜欢](https://opus.nci.org.au/display/Help/Raijin+User+Guide#RaijinUserGuide-QueueStructure)可以阅读更多)。通常授权数量级为数千或数百万 CPU 小时。

# 规格的含义

提交工作时，您可以指定所需的资源。这里最重要的事情是，系统架构对于 CPU 总是 x64，对于 GPU 总是 CUDA。所以当你为 Linux 操作系统编译时，你可以在超级计算机上运行它们。您的职位提交将如下所示。

```
#!/bin/bash
#PBS -P ch35
#PBS -q biodev
#PBS -l ncpus=28
#PBS -l mem=512GB
#PBS -l jobfs=1GB
#PBS -l walltime=100:00:00
#PBS -l wdmodule load java/jdk1.8.0_60module use -a /short/qr59/aw5153/modules/
module load canu/1.8java_path=$(which java)canu -p meta -pacbio-raw cluster-0cluster-0.fastq -d canu-cluster-0cluster-0/ maxMemory=512 maxThreads=28 useGrid=false java=$java_path genomeSize=50m
```

这是我用来组装宏基因组样本的真实脚本(有数百万次读取。或者像一些程序一样思考)。我已经从 project ch35 申请了 28 个内核/CPU，内存为 512GB，运行时间为 100 小时。所以我会消耗`28 * 100 * 1.5 = 2800 * 1.5 = 4200`服务单位。1.5 是`biodev`作业队列的成本。通常内存是不收费的，因为当你要求 28 个 CPU 时，它们会分配一个完整的节点，所以我们基本上每个节点都有`576GB`的 RAM。

您可以看到，我们在 4500 个节点中有 89，256 个内核。这些节点可以被认为是通过 infiniband(超高速网络)连接的独立计算机。这些节点可以通过消息传递接口(MPI)共享数据，但它们没有共享内存。这意味着如果您要求 56 个核心，我将得到 2 个节点，每个节点 28 个核心，每个节点 576GB 最大 RAM(总共 1152GB 或 1TB RAM)。但是你不能加载一个超过 576GB 的程序。这是因为这些节点是 NUMA 节点，代表非统一内存访问。一个节点中的一个进程不能访问另一个节点的 RAM。所以如果你的程序更大，你就完了！还是你？

> 让我们用一个示例程序来看看这个场景

假设您想要计算一个巨大数据集(数百万个基因组读数——每个约 100k 个字符的字符串)的所有可能的 n 元文法。然后有一个每个 n 元文法计数的查找表。然后，对于每个字符串，您希望用它们在整个语料库中的 n-gram 计数来注释它们。最有效的方法是在内存中保存一个巨大的哈希表，并将其用作查找表，迭代每个 100k 字符串的 n 元语法并进行注释)。查找表将占用 25GB(假设)。按顺序做这件事需要很多时间，所以我们想并行处理它们。

# 什么时候改？

在我们的示例场景中，每个节点最多可以有 28 个线程(这个特定场景的 RAM 不是大问题)。但是仅用 28 个内核运行这个需要几天时间。现在，我们需要使用 28 个以上的内核，更有可能是 100 个甚至更多的内核。

**现实:**如果我只要求 1000 个(这会自动挑选 36 个节点)核心并提交我的作业，它会因[颠簸](https://en.wikipedia.org/wiki/Thrashing_(computer_science))而停止。这是因为尽管我要求 1000 个内核，但我的程序将在 1 个节点上运行，因为一个进程只能在一个节点上运行。

这时我们知道我们的程序必须改变以适应超级计算环境或网格系统。

# 要改什么？

如果我们能运行 1000 个并行进程，我们就能克服上述挑战。这是因为独立的进程可以在独立的节点上运行，因为操作系统(或网格系统)可以有效地调度它们。但是现在我们的 RAM 需求会立刻增加，需要我们`25*1000GB`，因为我们必须为每个进程准备单独的查找表。我们只需要 25TB 的内存。但是对于 36 个节点，我们只有`36*576GB = 20TB`。这就是我们需要多处理和多线程的地方。因此，我们需要改变线程和进程管理方面，以获得对服务单元的最佳利用。让我们看看该怎么做。

# 怎么改？

由于我们正在努力提高性能，我将从科学计算中经常使用的 C++和 Python 编程的角度进行解释。对于多线程，我们可以使用 [OpenMP](https://www.openmp.org/) (C++)或者 python 的多处理库([你必须使用共享内存架构](https://docs.python.org/3.9/library/multiprocessing.shared_memory.html))。否则，您将再次丢失内存)。对于多重处理，您可以使用[Open-MPI](https://www.open-mpi.org/doc/)(c++和 Python 变体都可用)。现在，我们可以简单地部署每个进程有大约 28 个线程的进程。现在，每个进程将只使用驻留在单个节点上的 25GB RAM。同样，我们可以一次将更多的 100k 字符串加载到 RAM 中，用 28 个线程进行处理(因为在加载查找表后，每个节点中还有大量 RAM)。这个程序运行得快多了。

您将需要一些预处理和后处理来将初始数据集分成 36 个部分，并收集所有 36 个输出，这些输出可以是作业队列之外的简单的`split and cat`命令。以下是来自 [MPI 教程](https://mpitutorial.com/tutorials/mpi-hello-world/)的修改代码。

```
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <string>int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Print off a hello world message
    vector<string> data; // Load data set of rank to run with 28 threads
    #pragma omp parallel for num_threads(28)
    for(string s: data)
        // process // Finalize the MPI environment.
    MPI_Finalize();
}
```

你真正的程序会比这复杂得多。如果您正在使用 python 工作，请确保您已经阅读了文档，以确保线程化时不会出现冗余内存共享(否则它将克隆所有共享变量，包括查找表)。

我打算分享完整的实现，但不是在这一点上，因为我打算介绍许多优化，你可以在未来的写作。

希望这是一本好书，至少对初学者来说是这样。

干杯！