# 如何在Quartus中使用工程（方法一）

* 从 [官方网站](https://www.altera.com/downloads/download-center.html) 下载 Quartus。我们使用了 Quartus 13.1 Web* Edition
* 从 GitHub 下载 Quartus  II工程
* 进去到“verilog/imp”文件夹。使用 Quartus  II 打开“cam_proj.qpf”或双击（参见 img图 1）。

![图 1](https://github.com/suisuisi/FPGAandCNN/blob/main/images/QV_01.png?raw=true "图 1")

* 运行 "Processing" -> "Start compilation"，等待完成
* 将摄像头和显示屏连接到FPGA板卡
* 将FPGA板卡连接到 USB，然后插到PC USB。确保已安装 Altera USB 驱动程序。
* 运行“Tools" -> "Programmer”。单击“硬件设置”并选择 USB-Blaster（参见 图 2）

![图 2](https://github.com/suisuisi/FPGAandCNN/blob/main/images/QV_02.png?raw=true "图 2")

* 按 "Start" 按钮，就会在“进度条”中看到"100% Successful"

* 之后设备就会开始工作，将摄像头对准深色背景的数字上，就会在SPI显示屏左下角看到检测到的数字（参见 图 3）。有时会出现时钟同步问题，可以通过 FPGA板卡 上的“reset按钮”进行修复

  ![图 3](https://github.com/suisuisi/FPGAandCNN/blob/main/images/Video-screen.png?raw=true "图 3")

  

# 如何在Quartus中使用工程（方法二）

第一种方式是将二进制文件下载到FPGA的SRAM中，断电程序会丢失，第二种方式就是将二进制文件下载到外部FLASH中，这样即使断电也会从外部FLASH加载程序，这种方式不影响本项目使用，请自行查找方式。



PS：下载到FLASH后需要先断电再上电FPGA才会运行。
