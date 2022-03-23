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
-   DepthAI (\>=2.13.0.0)

要安装所有必需的 Python 模块，您可以使用：

``` shell
pip3 install -r requirements.txt
```

运行演示
--------

使用 -h 选项运行应用程序会产生以下用法消息：

``` shell
Usage: yolox.py [OPTIONS]

  基于 yolox 的目标检测器

Options:
  -vid, --video / -cam, --camera  使用 DepthAI 4K RGB 摄像头或视频文件进行推理  [default:
                                  cam]
  -m, --model_name [yolox|helmet]
                                  模型名称  [default: yolox]
  -size, --model_size INTEGER     模型输入大小  [default: 320]
  -p, --video_path PATH           指定用于推理的视频文件的路径
  -o, --output PATH               指定用于保存的视频文件的路径
  -fps, --fps INTEGER             保存视频的帧率  [default: 30]
  -s, --frame_size <INTEGER INTEGER>...
                                  保存视频的宽度，高度  [default: 1280, 720]
  -h, --help                      Show this message and exit.


```

运行该应用程序的有效命令行示例：

``` shell
python3 yolox.py
```

或

``` shell
python3 yolox.py -vid -p <path_to_video>
```

该示例需要的模型已在 [models](./models) 文件夹中。
