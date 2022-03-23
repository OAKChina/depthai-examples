面部信息识别
============

![](./facial_info_recognition.png)

本演示展示了面部信息识别。您可以在演示中使用以下一组预先训练的模型：

-   `face-detection-retail-0004` 是检测定位人脸的主要网络
-   `age-gender-recognition-retail-0013` 是识别年龄与性别的的主要网络
-   `emotions-recognition-retail-0003` 是识别面部表情的主要网络

本示例演示了如何使用 Gen2 Pipeline Builder 在 DepthAI 上运行两阶段推理。
首先，在图像上检测到脸部，然后将裁剪的脸部框架发送到年龄性别识别网络，该网络产生估计结果。

安装依赖项
----------

依赖项

-   Python ( 3.6+ )
-   OpenCV (\>=3.4.0)
-   DepthAI (\>=2.13.0.0)

要安装所有必需的 Python 模块，您可以使用：

``` {.shell}
pip3 install -r requirements.txt
```

运行演示
--------

使用 -h 选项运行应用程序会产生以下用法消息：


``` {.bash}
Usage: facial_info.py [OPTIONS]

    在视频流中实时检测和计数眨眼次数

Options:
  -vid, --video / -cam, --camera  使用 DepthAI 4K RGB 摄像头或视频文件进行推理  [default:
                                  cam]
  -p, --video_path PATH           指定用于推理的视频文件的路径
  -o, --output PATH               指定用于保存的视频文件的路径
  -fps, --fps INTEGER             保存视频的帧率  [default: 30]
  -s, --frame_size <INTEGER INTEGER>...
                                  保存视频的宽度，高度  [default: 1080, 1080]
  -h, --help                      Show this message and exit.
```

运行该应用程序的有效命令行示例：

``` {.shell}
python3 facial_info.py -cam
```

或

``` {.shell}
python3 facial_info.py -vid -p <path_to_video>
```

要运行该演示，可以使用公共或预训练的模型。要下载预训练的模型，请使用
`OpenVINO`
[模型下载器](https://docs.openvinotoolkit.org/latest/omz_tools_downloader_README.html)。

> 该示例需要的模型已在 [models](./models) 文件夹中。
