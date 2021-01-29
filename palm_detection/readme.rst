手掌检测演示
============

.. figure:: ./palm_detection.gif
   :alt: palm\_detection

   palm\_detection

本演示展示了手掌检测网络。您可以在演示中使用以下一组预先训练的模型：

-  ``palm_detection`` 是查找车辆和车牌的主要检测网络

安装依赖项
----------

依赖项

-  Python ( 3.6+ )
-  OpenCV (>=3.4.0)
-  DepthAI (0.0.2.1)

要安装所有必需的 Python 模块，您可以使用：

.. code:: shell

    pip3 install -r requirements.txt

运行演示
--------

使用 -h 选项运行应用程序会产生以下用法消息：

.. code:: bash

    usage: main.py [-h] [-nd] [-cam] [-vid VIDEO] [-db] [-n NAME]

    optional arguments:
      -h, --help            show this help message and exit
      -nd, --no-debug       阻止调试输出
      -cam, --camera        使用 DepthAI 4K RGB 摄像头进行推理 (与 -vid 冲突)
      -hd                                 输出高清视频
      -vid VIDEO, --video VIDEO
                            用于推理的视频文件的路径 (与 -cam 冲突)
      -db, --databases      保存数据（仅在运行识别网络时使用）
      -n NAME, --name NAME  数据名称（和 -db 一起使用）

    使用空的选项列表运行应用程序会产生一条错误消息

运行该应用程序的有效命令行示例：

.. code:: shell

    python3 main.py -cam

或

.. code:: shell

    python3 main.py -vid <path_to_video>

要运行该演示，可以使用公共或预训练的模型。要下载预训练的模型，请使用
``OpenVINO``
`模型下载器 <https://docs.openvinotoolkit.org/latest/omz_tools_downloader_README.html>`__\ 。

    该示例需要的模型已在 `models <./models>`__ 文件夹中。
