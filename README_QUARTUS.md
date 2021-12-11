# 如何在Quartus中使用工程（方法一）

* 从 [官方网站](https://www.altera.com/downloads/download-center.html) 下载 Quartus。我们使用了 Quartus 13.1 Web* Edition
* 从 GitHub 下载 Quartus  II工程
* 进去到“verilog/imp”文件夹。使用 Quartus  II 打开“cam_proj.qpf”或双击（参见 img图 1）。

![Img 1](https://github.com/ZFTurbo/Verilog-Generator-of-Neural-Net-Digit-Detector-for-FPGA/blob/master/images/QV_01.png "Img 1")

* Run "Processing" -> "Start compilation". Wait while it finished
* Connect De0Nano device to USB. Make sure you have Altera USB drivers installed.
* Run "Tools" -> "Programmer". Click on "Hardware Setup" and select USB-Blaster (see img 2)

![Img 2](https://github.com/ZFTurbo/Verilog-Generator-of-Neural-Net-Digit-Detector-for-FPGA/blob/master/images/QV_02.png "Img 2")

* Push "Start" button. Sometimes it failed so push it until you see "100% Successful" in Progress field.
* After that device must work fine. Sometimes it has problem with clock synchronization, it can be fixed with "Reset button" on De0Nano.

# How to use project in Quartus (Method 2)

In first method project is loaded in energy dependent memory. So after you reconnect De0Nano it will reset to initial state. There is method to
store project in Flash memory. So after you reconnect device to any energy source it will be already initialized.

* After you compile project go to "File" -> "Convert Programming File".
* Choose Programming File Type: "JTAG Indirect Configuration File (.jic)"
* Choose "Configuration device": "EPCS16"
* Select "Flash Loader" and press "Add device..." from dialog choose "Cyclone IV E" and "EP4CE22"
* Select "SOF Data" and press "Add File...". From dialog choose "output_files/cam_proj.sof"
* And then press "Generate" (see img 3)

![Img 3](https://github.com/ZFTurbo/Verilog-Generator-of-Neural-Net-Digit-Detector-for-FPGA/blob/master/images/QV_03.png "Img 3")

* Then go to "Tools" -> "Programmer". Select "output_files/cam_proj.sof" and press "Delete". Then press "Add file..." and
in dialog choose "output_files/output_file.jic". Then press "Start" and wait for "100% Successful" in Progress field.
(see img 4)

![Img 4](https://github.com/ZFTurbo/Verilog-Generator-of-Neural-Net-Digit-Detector-for-FPGA/blob/master/images/QV_04.png "Img 4")

* Now you need to restart device to make it work (for example plug/unplug it from USB).

# 

* 
* 
* 提取存档并导航到“verilog/imp”文件夹。您可以使用 Quartus 打开“cam_proj.qpf”或双击它（参见 img 1）。

![图像 1](https://github.com/ZFTurbo/Verilog-Generator-of-Neural-Net-Digit-Detector-for-FPGA/blob/master/images/QV_01.png "图像 1")

* 运行“处理”->“开始编译”。等它完成
* 将 De0Nano 设备连接到 USB。确保已安装 Altera USB 驱动程序。
* 运行“工具”->“程序员”。单击“硬件设置”并选择 USB-Blaster（参见 img 2）

![图像 2](https://github.com/ZFTurbo/Verilog-Generator-of-Neural-Net-Digit-Detector-for-FPGA/blob/master/images/QV_02.png "图像 2")

*按“开始”按钮。有时它会失败，因此请推动它，直到您在“进度”字段中看到“100% 成功”。
* 之后该设备必须工作正常。有时会出现时钟同步问题，可以通过 De0Nano 上的“重置按钮”修复。

# 如何在Quartus中使用工程（方法二）

在第一种方法中，项目被加载到能源相关内存中。因此，在您重新连接 De0Nano 后，它将重置为初始状态。有方法可以
将项目存储在闪存中。因此，在您将设备重新连接到任何能源后，它将已经被初始化。

* 编译项目后，转到“文件”->“转换编程文件”。
* 选择编程文件类型：“JTAG 间接配置文件（.jic）”
* 选择“配置设备”：“EPCS16”
* 选择“Flash Loader”并按“Add device...”从对话框中选择“Cyclone IV E”和“EP4CE22”
* 选择“SOF 数据”并按“添加文件...”。从对话框中选择“output_files/cam_proj.sof”
* 然后按“生成”（见 img 3）

![Img 3](https://github.com/ZFTurbo/Verilog-Generator-of-Neural-Net-Digit-Detector-for-FPGA/blob/master/images/QV_03.png "Img 3")

* 然后转到“工具”->“程序员”。选择“output_files/cam_proj.sof”并按“删除”。然后按“添加文件...”和
  在对话框中选择“output_files/output_file.jic”。然后按“开始”并等待“进度”字段中的“100% 成功”。
  （见 img 4）

![Img 4](https://github.com/ZFTurbo/Verilog-Generator-of-Neural-Net-Digit-Detector-for-FPGA/blob/master/images/QV_04.png "Img 4")

* 现在您需要重新启动设备以使其工作（例如从 USB 插入/拔出它）。
