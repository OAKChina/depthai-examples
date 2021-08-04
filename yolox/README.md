`YOLOX` 目标检测演示
====================

![detection](demo.png)

本演示展示了目标检测网络。您可以在演示中使用以下一组预先训练的模型：

-   [`yolox_nano`](models/yolox_nano_320x320_openvino_2021.4_6shave.blob) 是目标检测网络

安装依赖项
----------

依赖项

-   Python ( 3.6+ )
-   OpenCV (\>=3.4.0)
-   DepthAI (\>=2.8.0.0)

要安装所有必需的 Python 模块，您可以使用：

``` shell
pip3 install -r requirements.txt
```

运行演示
--------

使用 -h 选项运行应用程序会产生以下用法消息：

``` shell
usage: main.py [-h] [-nd] [-cam] [-vid VIDEO] [-db] [-n NAME]

optional arguments:
  -h, --help            show this help message and exit
  -nd, --no-debug       Prevent debug output
  -cam, --camera        Use DepthAI 4K RGB camera for inference (conflicts with -vid)
  -vid VIDEO, --video VIDEO
                        Path to output video file
  -img IMAGE, --image IMAGE
                        Path to image file


```

运行该应用程序的有效命令行示例：

``` shell
python3 main.py
```

或

``` shell
python3 main.py -vid <path_to_video>
```

该示例需要的模型已在 [models](./models) 文件夹中。
