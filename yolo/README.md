# 基于设备上解码的 Yolo 检测

![Yolo-on-device](https://user-images.githubusercontent.com/56075061/144863222-a52be87e-b1f0-4a0a-b39b-f865bbb6e4a4.png)

代码：[depthai_yolo](https://gitcode.net/oakchina/depthai-examples/-/tree/master/depthai_yolo) （国内 gitcode.net），
[DepthAI_Yolo](https://github.com/richard-xx/DepthAI_Yolo) （github.com）

该存储库 (
修改自 [device-decoding](https://github.com/luxonis/depthai-experiments/tree/master/gen2-yolo/device-decoding))
包含直接使用 DepthAI SDK (`main_sdk.py`) 或 DepthAI API (`main_api.py`) 在设备上解码运行 Yolo 目标检测的代码。目前，支持的版本有：

* `YoloV3` & `YoloV3-tiny`,
* `YoloV4` & `YoloV4-tiny`,
* `YoloV5`,
* `YoloV6`,
* `YoloV7`.

我们在 `main_sdk.py` 和 `main_api.py` 中使用相同样式的 JSON 解析，但您也可以在代码中手动设置这两种情况下的值。

### 导出模型

由于模型必须以某种方式导出转换到 OpenVINO IR，我们提供了关于训练和导出的教程：

* `YoloV3`, `YoloV4`, 和它们的 `tiny` 版本：
    * 训练：
        * [YoloV3_V4_tiny_training.ipynb](https://github.com/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV3_V4_tiny_training.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV3_V4_tiny_training.ipynb)
        * [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
    * 导出转换：
        * [https://github.com/luxonis/yolo2openvino](https://github.com/luxonis/yolo2openvino)

* `YoloV5`, `YoloV6`, 和 `YoloV7` ：
    * 训练可参考原始仓库：
        * [YoloV5](https://github.com/ultralytics/yolov5),
        * [YoloV6](https://github.com/meituan/YOLOv6),
        * [YoloV7](https://github.com/WongKinYiu/yolov7)
    * 导出转换：
        * 可使用 [https://tools.luxonis.com/](https://tools.luxonis.com/)
          网页在线转换，
        * 或参考 [https://github.com/luxonis/tools/tree/master/yolo](https://github.com/luxonis/tools/tree/master/yolo)
          和 [https://github.com/luxonis/tools/tree/master/yolov7](https://github.com/luxonis/tools/tree/master/yolov7)
          进行本地转换

## 用法

1. 安装依赖
    ```shell
    python3 -m pip install -r requirements.txt
    ```
2. 运行脚本
    ```shell
    python3 main_sdk.py -m <model_name> -c <config_json>
    ```
   或者
    ```shell
    python3 main_api.py -m <model_name> -c <config_json>
    ```
   Tips：

    * `<model_name>` 是来自 DepthAI 模型库 (https：zoo.luxonis.com) 的模型名称或 blob 文件的相对路径。请查看我们的模型库以查看可用的预训练模型。
    * `<config_json>` 是带有 Yolo 模型元数据（输入形状、锚点、标签等）的 JSON 的相对路径。

## JSONs

我们已经为常见的 Yolo 版本提供了一些 JSON。您可以编辑它们并为您的模型设置它们，如上述教程中的后续步骤部分所述。如果您要更改教程中的某些参数，则应编辑相应的参数。一般来说，JSON
中的设置应该遵循模型的 CFG 中的设置。对于 YoloV5，默认设置应与 YoloV3 相同。

**Note**：值必须与训练期间在 CFG 中设置的值相匹配。如果您使用不同的输入宽度，您还应该将 `side32` 更改为 `sideX`
并将 `side16` 更改为 `sideY`，其中 `X = width16` 和 `Y = width32`。如果您使用的是非微型模型，则这些值为 `width8`、`width16`
和 `width32`。

您还可以更改 IOU 和置信度阈值。如果多次检测到同一个目标，则增加 IOU
阈值。如果没有检测到足够的目标，则降低置信度阈值。请注意，这不会神奇地改善您的目标检测器，但如果某些目标由于阈值太高而被过滤掉，则可能会有所帮助。

## Depth 信息

DepthAI 使您能够利用深度信息并获取检测到的对象的 `x`、`y` 和 `z` 坐标。

```shell
python3 main_sdk.py -m <model_name> -c <config_json> --spatial
```

或者

```shell
python3 main_api.py -m <model_name> -c <config_json> --spatial
```

如果您对使用 Yolo 检测器的深度信息感兴趣，
请查看我们的 [文档](https://docs.oakchina.cn/projects/api/samples/SpatialDetection/spatial_tiny_yolo.html)。
![SpatialObjectDetection](https://user-images.githubusercontent.com/56075061/144864639-4519699e-d3da-4172-b66b-0495ea11317e.png)