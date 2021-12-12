# 基于FPGA的数字识别-实时视频处理的定点卷积神经网络实现

这是训练神经网络在浅色背景上检测深色数字的项目。 然后使用多种技术将神经网络转换为 Verilog HDL ，以减少 FPGA 上所需的资源并提高处理速度。 它可以很容易地扩展以用于检测具有不同神经网络结构的其他对象。



整篇文章主要内容如下所示：



![](https://files.mdnice.com/user/17442/c0b13a32-a5e7-42f9-9ccd-9c72484acb39.png)


效果：



![](https://files.mdnice.com/user/17442/12bea4fa-1a97-41c3-a298-68ce5a1755ba.png)

PS：图像模糊的原因是一个人拍摄相机不好聚焦。

电脑显示数字（手写也可以，要求是浅色背景上检测深色数字（要求是训练集的问题）），通过摄像头采集缓存到SDRAM后在显示屏上显示摄像头数据，然后右下角显示监测到的数字。

下面就简单介绍一下相关知识。

# 工具

Python 3.5, Tensorflow 1.4.0, Keras 2.1.3

# 怎么使用？

* python r01_train_neural_net_and_prepare_initial_weights.py
* python r02_rescale_weights_to_use_fixed_point_representation.py
* python r03_find_optimal_bit_for_weights.py
* python r04_verilog_generator_grayscale_file.py
* python r05_verilog_generator_neural_net_structure.py

FPGA工程位于 ''verilog''文件夹中. 包括摄像头和SPI 显示屏等相关的所有代码。关键的神经网络代码位于''verliog/code/neuroset'' 文件夹中.

# 数据集

手写数字识别的MNIST数据集（http://yann.lecun.com/exdb/mnist/）广泛应用于计算机视觉领域。然而，并不适合在我们的应用中训练神经网络，因为它与相机图像有很大的不同。主要区别包括：

- MNIST图像是深色背景上的浅色数字，与来自摄像头的图像相反（下图中A来自MINIST，B来自普通的相机）；

- 摄像头产生彩色图像，图像大小为320×240像素，而MNIST是灰度的MNIST图像大小为28×28像素；

- 与MNIST图像中居中放置的数字和相同背景（黑色）不同，数字可以在相机图像中移动和轻微旋转，有时背景中会有噪声；

- MNIST没有单独的数字图像类别。


![](https://files.mdnice.com/user/17442/698ae130-052e-4841-8b77-7db6390e7fdf.png)


鉴于MNIST数据集的识别性能非常高，我们将摄像机图像的大小减少到28×28像素，并将其转换为灰度。这有助于我们解决以下问题：

- 识别的准确度没有明显损失，因为即使在小图像中，数字仍然很容易被识别；

- 对于数字识别而言，颜色信息过多，所以转换成灰度图像刚刚好；

- 通过减少和平均相邻像素，可以清除来自摄像机的噪声图像。

由于图像变换也是在硬件级上执行的，因此必须预先考虑最小的一组算术函数，这些函数可以有效地将图像转换为所需的形式。修改摄像头图像的算法如下所示：

- 从320×240图像中裁剪出一个中心部分，该部分测量224×224像素，由于224=28×8，因此随后可以轻松过渡到所需的图像大小。

- 然后，将裁剪的图像部分转换为灰度图像。由于人类视觉感知的特殊性，我们采用加权平均，而不是简单的平均。为了便于在硬件级别进行转换，使用以下公式：


![](https://files.mdnice.com/user/17442/9a946638-da39-4abb-ad80-412b5f7aa59a.png)

即RGB的权重为5:8:3

为了便于FPGA编程实现，我们可以在FPGA中使用移位实现8的乘法和16的除法。

- 最后，将224×224图像分割成8×8块。我们计算每个块的平均值，在28×28图像中形成相应的像素。

由此产生的算法简单，适合FPGA实现并且速度非常快。

为了使用MNIST图像训练神经网络，需要把MINIST训练集进行修改：

- 颜色反转；
- 在两个方向上随机旋转10度；
- 图像随机扩展或缩小4像素；
- 图像强度的随机变化（从0到80）；
- 增加0%至10%的随机噪声。
- 将来自摄像头的图像混合到训练集中。

上诉操作可以使用MATLAB或者Python批量处理。

# CNN设计

CNN的体系一直在发展（也就是为什么ASIC没有批量生产，还用FPGA验证一些CNN最新的算法），但是本质仍然是一样，因为我们使用的FPGA是一个入门型的，所以我们也不用最新的CNN。

CNN的本质：输入大小从一层到另一层减小，而过滤器的数量增加。在网络的末端，形成一组特征，这些特征被馈送到分类层，并且输出层指示图像属于特定类别的可能性。

![](https://files.mdnice.com/user/17442/623470f9-bd12-4b25-9474-50895ad7b38a.png)

由于使用FPGA实现所以权重的总数对于设计来说是个瓶颈，所以需要最小化存储权重的总数（这对于移动系统至关重要），并促进向定点计算（FPGA只能进行定点计算）的转移：

- 尽可能减少完全连接层的数量，这些层消耗大量的权重；
- 在不降低分类性能的情况下，尽可能减少每个卷积层的滤波器数量；
- 不使用偏差，当从浮点转换为定点时，添加常数会妨碍值的监控范围，并且每层上的舍入偏差误差会累积；
- 使用简单类型的激活，如RELU（线性整流函数（Rectified Linear Unit, ReLU），又称修正线性单元），因为其他激活，如Sigmoid和Tahn，包含除法、求幂和其他难以在硬件中实现的运算；
- 尽量减少异构层的数量。

在将神经网络转换为硬件之前，在准备好的数据集上对其进行训练，并保留软件实现的方式以供测试。使用Keras和Tensorflow后端的软件实现。

架构如下：

![Neural Net Structure](https://github.com/suisuisi/FPGAandCNN/blob/main/images/Neural-Net-Structure.png?raw=true "Neural Net Structure")

# 浮点计算转向定点计算

在神经网络实现的方案中，在GPU（快速）或CPU（慢速）上使用浮点计算方案是最常见的方案，例如，使用float32类型。在使用FPGA实现时，浮点运算对于这个“硬疙瘩”实在是很难实现，所以我们需要将浮点运算转换成定点计算（在牺牲一点识别率的情况）。

考虑一下神经网络的第一卷积层，它是卷积结构的主要构造。

层输入是一个二维矩阵（原始图片）28×28，其值从[0；1]。当a∈[−1,1]和b∈[−1，1]时，a·b∈[−1, 1]. 

对于3×3卷积，第二层中特定像素（i，j）的值计算公式如下：


![](https://files.mdnice.com/user/17442/976a19d8-3685-4323-b118-6ce7f31befd3.png)

当使用卷积块的定点计算时，有几种不同的策略：

- 对所有可能的输入图像进行排序，并将注意力集中在潜在的最小值和最大值上，可以得到非常大的缩减系数；
- 对于有限的权重和中间结果的宽度的定点计算，舍入误差不可避免地出现，每次加法和乘法基本运算后进行舍入；
- 在卷积运算的最后进行精确计算和四舍五入（在内存开销和这种方案测试中，这种方案是最有利的）。

# 硬件测试

整个硬件架构如下：

![](https://files.mdnice.com/user/17442/546c11e4-28b7-47d8-8e73-a60e7264d96d.png)

相关的硬件如下：

* [OpenEP4](https://github.com/suisuisi/gamegirl/tree/master/Hardware/V1.0/CoreBoard) 
![OpenEP4](https://github.com/suisuisi/gamegirl/blob/master/Hardware/V1.0/CoreBoard/CORE%20PCB.png?raw=true  "OpenEP4")
* [Camera OV7670](https://www.voti.nl/docs/OV7670.pdf) 
![Camera OV7670](https://github.com/suisuisi/FPGAandCNN/blob/main/images/ov7670.png?raw=true  "Camera OV7670")
* [SPI TFT](https://cdn-shop.adafruit.com/datasheets/ILI9341.pdf)
* ![Display ILI9341](https://github.com/suisuisi/FPGAandCNN/blob/main/images/spitft.jpg?raw=true  "SPI TFT")
* [PMOD](https://github.com/suisuisi/FPGAandCNN/tree/main/hardware/PMOD_OV7670)
* ![PMOD](https://github.com/suisuisi/FPGAandCNN/blob/main/images/PMOD.png?raw=true  "PMOD")

硬件连接如下：

![TOP](https://github.com/suisuisi/FPGAandCNN/blob/main/images/Conn-Foto-top.jpg?raw=true  "TOP")

![BOTTOM](https://github.com/suisuisi/FPGAandCNN/blob/main/images/Conn-Foto-bot.jpg?raw=true  "BOTTOM")

PS：图中的飞线是为了验证其他项目飞的电源线，不影响本项目使用


整个硬件数据流：摄像头将图像以低频率写入FIFO，然后SDRAM控制器以高频率读取数据。然后FPGA将SDRAM中的数据写入屏幕FIFO。

来自摄像头的图片经过SDRAM后，按原样显示在屏幕上，并将图像转换为灰度并降低分辨率的图像输入到神经网络进行识别。当神经网络操作完成后，结果也直接输出到屏幕上。

# 注意

可以将r05_verilog_generator_neural_net_structure.py 中的常量num_conv = 2更改为 1、2 或 4 个并行工作的卷积块。更多的卷积块将使用更多的 FPGA逻辑资源，但是会提高整体运行速度。

下面是不同位权重和卷积块数量的比较表（红色行：由于FPGA 限制，无法合成）。

![使用的FPGA资源](https://github.com/suisuisi/FPGAandCNN/blob/main/images/Info-Table.png?raw=true "使用的FPGA资源")



# 视频演示

[![使用FPGA实现数字识别-基于定点神经网络（CNN）](https://github.com/suisuisi/FPGAandCNN/blob/main/images/Video-screen.png)](https://www.bilibili.com/video/BV1yY411p7Ju)



# 参考文献

[1] Huang, Gao, et al. "Densely connected convolutional networks." CVPR. Vol. 1. No. 2. 2017.

[2] Chen, Liang-Chieh, et al. "Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs." IEEE transactions on pattern analysis and machine intelligence 40.4 (2018): 834-848.

[3] A. Shvets, A. Rakhlin, A. A. Kalinin, and V. Iglovikov, “Automatic instrument segmentation in robot-assisted surgery using deep learning,” arXiv preprint arXiv:1803.01207, 2018.

[4] Sandler M. et al. “Inverted residuals and linear bottlenecks:Mobile networks for classification, detection and segmentation” arXiv preprint arXiv:1801.04381, 2018.

[5] Roman Solovyev, Fixed-Point Convolutional Neural Network for Real-Time Video Processing in FPGA, 2019

# 致谢

本人刚接触这方面的知识，很感谢在设计过程中帮忙的朋友，让我在磕磕碰碰中完成了该项目，非常感谢！