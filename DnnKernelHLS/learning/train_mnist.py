
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix


# 1.定义网络模型
class Net(nn.Module):
    def __init__(self, num_output_classes=10):
        super(Net, self).__init__()

        # 输入为28x28的灰度图像（通道数=1）
        # 进行输出为8通道的卷积
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)


        # 激活函数为ReLU
        self.relu1 = nn.ReLU(inplace=True)

        # 将图像从28x28缩小到14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 4ch -> 8ch, 14x14 -> 7x7
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        # 8ch的7x7图像被视为一个矢量，缩小到元素数32的矢量
        self.fc1 = nn.Linear(8 * 7 * 7, 32)
        self.relu3 = nn.ReLU(inplace=True)

        # 全连接层
        # 缩小到输出类数
        self.fc2 = nn.Linear(32, num_output_classes)

    def forward(self, x):
        # 第一层折叠
        # 激活函数是ReLU
        x = self.conv1(x)
        x = self.relu1(x)

        # 缩小
        x = self.pool1(x)

        # 第2层+缩小
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 格式转换(Batch, Ch, Height, Width) -> (Batch, Ch)
        x = x.view(x.shape[0], -1)

        # 全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


net = Net()

# 2. 数据集读取方法的定义
# 获取MNIST的学习测试数据
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())


# 数据读取方法的定义
# 1步骤的每个学习测试读取16张图像
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)


# 损失函数，定义优化器
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# 3. 学习
# 循环，直到使用数据集中的所有图像10次
for epoch in range(10):
    running_loss = 0

    # 在数据集中循环
    for i, data in enumerate(trainloader, 0):
        # 导入输入批（图像，正确标签）
        inputs, labels = data

        # 初始化优化程序
        optimizer.zero_grad()

        # 通过模型获取输入图像的输出标签
        outputs = net(inputs)

        # 与正确答案的误差计算+误差反向传播
        loss = loss_func(outputs, labels)
        loss.backward()

        # 使用误差优化模型
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

# 4. 测试
ans = []
pred = []
for i, data in enumerate(testloader, 0):
    inputs, labels = data

    outputs = net(inputs)

    ans += labels.tolist()
    pred += torch.argmax(outputs, 1).tolist()

print('accuracy:', accuracy_score(ans, pred))
print('confusion matrix:')
print(confusion_matrix(ans, pred))

# 5. 保存模型
# 用于从PyTorch正常读取的模型文件
torch.save(net.state_dict(), 'model.pt')

# 保存用于从libtorch（C++API）读取的Torch Script Module
example = torch.rand(1, 1, 28, 28)
traced_script_module = torch.jit.trace(net, example)
traced_script_module.save('traced_model.pt')
