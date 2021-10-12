# coding=utf-8
import argparse
import time
from datetime import datetime, timedelta

import cv2
import depthai
import numpy as np


def timer(function):
    """
    装饰器函数 timer

    :param function: 想要计时的函数
    :return: 运行时间
    """

    def wrapper(*arg, **kwargs):
        time_start = time.time()
        res = function(*arg, **kwargs)
        cost_time = time.time() - time_start
        print(f"【 {function.__name__} 】运行时间：【 {cost_time} 】秒")
        return res

    return wrapper


parser = argparse.ArgumentParser()
parser.add_argument("-nd", "--no-debug", action="store_true", help="阻止调试输出")
parser.add_argument(
    "-cam",
    "--camera",
    action="store_true",
    help="使用 DepthAI 4K RGB 摄像头进行推理 (与 -vid 冲突)",
)

parser.add_argument(
    "-hd",
    action="store_true",
    help="输出高清视频",
)

parser.add_argument(
    "-vid",
    "--video",
    type=str,
    help="用于推理的视频文件的路径 (与 -cam 冲突)",
)

parser.add_argument(
    "-db",
    "--databases",
    action="store_true",
    help="保存数据（仅在运行识别网络时使用）",
)

parser.add_argument(
    "-n",
    "--name",
    type=str,
    help="数据名称（和 -db 一起使用）",
)

args = parser.parse_args()

debug = not args.no_debug

is_db = args.databases

if args.camera and args.video:
    raise ValueError("命令行参数错误！ “ -cam” 不能与 “ -vid” 一起使用！")
elif args.camera is False and args.video is None:
    args.camera = True
    print("缺少推理源！使用 “ -cam” 作为默认源")


def wait_for_results(queue):
    """
    如果 两次时间间隔超过 1 秒
    返回 False

    :param queue:
    :return: 是否超时
    """
    start = datetime.now()
    while not queue.has():
        if datetime.now() - start > timedelta(seconds=1):
            return False
    return True


def resize_padding(img: np.ndarray, fixed_h, fixed_w, bgr: list = None):
    """
    图像等比例缩放并填充

    :param img: cv2所读取的图像
    :param fixed_h: 期望的高
    :param fixed_w: 期望的宽
    :param bgr: 填充颜色，默认黑色
    :return: pad_img,top,left
    """
    if bgr is None:
        bgr = [0, 0, 0]

    h, w = img.shape[:2]
    scale = max(w / fixed_w, h / fixed_h)  # 获取缩放比例
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv2.resize(img, (new_w, new_h))  # 按比例缩放

    # 计算需要填充的像素长度
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (
            (fixed_h - new_h) // 2,
            (fixed_h - new_h) // 2,
            (fixed_w - new_w) // 2 + 1,
            (fixed_w - new_w) // 2,
        )
    elif new_w % 2 == 0 and new_h % 2 != 0:
        top, bottom, left, right = (
            (fixed_h - new_h) // 2 + 1,
            (fixed_h - new_h) // 2,
            (fixed_w - new_w) // 2,
            (fixed_w - new_w) // 2,
        )
    elif new_w % 2 == 0 and new_h % 2 == 0:
        top, bottom, left, right = (
            (fixed_h - new_h) // 2,
            (fixed_h - new_h) // 2,
            (fixed_w - new_w) // 2,
            (fixed_w - new_w) // 2,
        )
    else:
        top, bottom, left, right = (
            (fixed_h - new_h) // 2 + 1,
            (fixed_h - new_h) // 2,
            (fixed_w - new_w) // 2 + 1,
            (fixed_w - new_w) // 2,
        )

    # 填充图像
    pad_img = cv2.copyMakeBorder(
        resize_img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=bgr,
    )

    return (
        pad_img,
        scale,
        top,
        left,
    )


def restore_point(point, scale, top, left):

    return (np.array(point).reshape(-1, 2) - (left, top)) * scale

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    img = resize_padding(arr, shape[0], shape[1])

def to_planar(arr: np.ndarray, shape: tuple):
    img, scale, top, left = resize_padding(arr, shape[0], shape[1])

    return (
        [
            val
            for channel in img.transpose(2, 0, 1)
            for y_col in channel
            for val in y_col
        ],
        scale,
        top,
        left,
    )

    # return [
    #     val
    #     for channel in cv2.resize(arr, shape).transpose(2, 0, 1)
    #     for y_col in channel
    #     for val in y_col
    # ]


def to_nn_result(nn_data):
    """
    :param nn_data: 神经网络数据
    :return: 以 np 数组 形式返回 第一层网络
    """
    return np.array(nn_data.getFirstLayerFp16())


def to_tensor_result(packet):
    """

    :param packet: 数据包
    :return: 以字典形式 返回 网络层
    """
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }


# @timer
def to_bbox_result(nn_data):
    """

    :param nn_data:
    :return:
    """
    arr = to_nn_result(nn_data)
    if np.argwhere(arr == -1).size > 0:
        arr = arr[: np.argwhere(arr == -1)[0][0]]
    arr = arr.reshape((arr.size // 7, 7))
    return arr


def scale_bboxes(bboxes, scale=True, scale_size=1.5):
    bboxes = np.concatenate(bboxes).reshape(-1, 4)
    if scale:
        scale_size = np.sqrt(scale_size)
        bboxes[:, 3] = (bboxes[:, 3] + bboxes[:, 1]) / 2 + scale_size * (
            (bboxes[:, 3] - bboxes[:, 1]) / 2
        )
        bboxes[:, 1] = (bboxes[:, 3] + bboxes[:, 1]) / 2 - scale_size * (
            (bboxes[:, 3] - bboxes[:, 1]) / 2
        )
        bboxes[:, 2] = (bboxes[:, 2] + bboxes[:, 0]) / 2 + scale_size * (
            (bboxes[:, 2] - bboxes[:, 0]) / 2
        )
        bboxes[:, 0] = (bboxes[:, 2] + bboxes[:, 0]) / 2 - scale_size * (
            (bboxes[:, 2] - bboxes[:, 0]) / 2
        )
    return np.where(bboxes > 0, bboxes, 0)


# @timer
def run_nn(x_in, x_out, in_dict):
    """

    :param x_in: X_link_in
    :param x_out: X_link_out
    :param in_dict:
    :return:
    """
    nn_data = depthai.NNData()
    for key in in_dict:
        nn_data.setLayer(key, in_dict[key])
    x_in.send(nn_data)
    # has_results = wait_for_results(x_out)
    # if not has_results:
    #     raise RuntimeError("No data from nn!")
    return x_out.tryGet()


def frame_norm(frame, *xy_vals):
    """
    nn data, being the bounding box locations, are in <0..1> range -
    they need to be normalized with frame width/height

    :param frame: (h, w) or frame
    :param xy_vals: the bounding box locations
    :return:
    """
    if isinstance(frame, np.ndarray):
        return (
            np.clip(np.array(xy_vals), 0, 1)
            * np.array(frame.shape[:2] * (len(xy_vals) // 2))[::-1]
        ).astype(int)
    else:

        return (
            np.clip(np.array(xy_vals), 0, 1)
            * np.array(frame * (len(xy_vals) // 2))[::-1]
        ).astype(int)


def draw_3d_axis(image, head_pose, origin, size=50):
    roll = head_pose[0] * np.pi / 180
    pitch = head_pose[1] * np.pi / 180
    yaw = -(head_pose[2] * np.pi / 180)

    # X axis (red)
    x1 = size * (np.cos(yaw) * np.cos(roll)) + origin[0]
    y1 = (
        size
        * (
            np.cos(pitch) * np.sin(roll)
            + np.cos(roll) * np.sin(pitch) * np.sin(yaw)
        )
        + origin[1]
    )
    cv2.line(image, (origin[0], origin[1]), (int(x1), int(y1)), (0, 0, 255), 3)

    # Y axis (green)
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + origin[0]
    y2 = (
        size
        * (
            -np.cos(pitch) * np.cos(roll)
            - np.sin(pitch) * np.sin(yaw) * np.sin(roll)
        )
        + origin[1]
    )
    cv2.line(image, (origin[0], origin[1]), (int(x2), int(y2)), (0, 255, 0), 3)

    # Z axis (blue)
    x3 = size * (-np.sin(yaw)) + origin[0]
    y3 = size * (np.cos(yaw) * np.sin(pitch)) + origin[1]
    cv2.line(image, (origin[0], origin[1]), (int(x3), int(y3)), (255, 0, 0), 2)

    return image


def correction(frame, coords, invert=False):
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(angle + 90) if angle < -45 else -angle
    # print(f"倾斜角度为：{angle}度")
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1)
    affine = cv2.invertAffineTransform(mat).astype("float32")
    corr = cv2.warpAffine(
        frame,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if invert:
        return corr, affine
    return corr


def cosine_distance(a, b):
    """
    根据输入数据的不同，分为两种模式处理。

    输入数据为一维向量，计算单张图片或文本之间的相似度 （单张模式）

    输入数据为二维向量（矩阵），计算多张图片或文本之间的相似度 （批量模式）

    :param a: 图片向量
    :param b: 图片向量
    :return: 余弦相似度
    """
    if a.shape != b.shape:
        raise RuntimeError(
            "array {} shape not match {}".format(a.shape, b.shape)
        )
    if a.ndim == 1:
        # 操作是求向量的范式，默认是 L2 范式，等同于求向量的欧式距离
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim == 2:
        # 设置参数 axis=1 。对于归一化二维向量时，将数据按行向量处理，相当于单独对每张图片特征进行归一化处理。
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similarity = np.dot(a, b.T) / (a_norm * b_norm)
    # dist = 1.0 - similarity  # 余弦距离 = 1- 余弦相似度
    # return dist
    return similarity


def sigmoid(x):
    """sigmoid.

    :param x: Tensor.
    :return: numpy ndarray.

    (1. + np.tanh(.5 * x)) * .5 = 1. / (1. + np.exp(-x))
    sigmoid（x）==（1 + tanh（x / 2））/ 2
    """
    # return 1.0 / (1.0 + np.exp(-x))
    # return expit(x)
    return (1.0 + np.tanh(0.5 * x)) * 0.5


def decode_boxes(raw_boxes, anchors, shape, num_keypoints):
    """
    使用锚点框将预测转换为实际坐标。一次处理整个批次。

    Converts the predictions into actual coordinates using the anchor boxes.
    Processes the entire batch at once.
    """
    boxes = np.zeros_like(raw_boxes)
    x_scale, y_scale = shape

    x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / x_scale * anchors[:, 2]
    h = raw_boxes[..., 3] / y_scale * anchors[:, 3]

    boxes[..., 1] = y_center - h / 2.0  # xmin
    boxes[..., 0] = x_center - w / 2.0  # ymin
    boxes[..., 3] = y_center + h / 2.0  # xmax
    boxes[..., 2] = x_center + w / 2.0  # ymax

    for k in range(num_keypoints):
        offset = 4 + k * 2
        keypoint_x = (
            raw_boxes[..., offset] / x_scale * anchors[:, 2] + anchors[:, 0]
        )
        keypoint_y = (
            raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
        )
        boxes[..., offset] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes


def raw_to_detections(
    raw_box_tensor, raw_score_tensor, anchors_, shape, num_keypoints
):
    """

    This function converts these two "raw" tensors into proper detections.
    Returns a list of (num_detections, 17) tensors, one for each image in
    the batch.

    This is based on the source code from:
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto

    :param num_keypoints:
    :param shape:
    :param raw_box_tensor: 神经网络的输出是一个形状为（b，896，16）的张量，其中包含边界框回归预测
    :param raw_score_tensor: 具有分类置信度的形状张量（b，896，1）
    :param anchors_: 锚点框
    :return:
    """
    detection_boxes = decode_boxes(
        raw_box_tensor, anchors_, shape, num_keypoints
    )
    detection_scores = sigmoid(raw_score_tensor).squeeze(-1)
    output_detections = []
    for i in range(raw_box_tensor.shape[0]):
        boxes = detection_boxes[i]
        scores = np.expand_dims(detection_scores[i], -1)
        output_detections.append(np.concatenate((boxes, scores), -1))
    return output_detections


def non_max_suppression(boxes, probs=None, angles=None, overlapThresh=0.3):
    # 如果没有框，则返回一个空列表
    if len(boxes) == 0:
        return [], []

    # 如果边界框是整数，则将它们转换为浮点型 --
    # 这很重要，因为我们将进行很多划分
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # 初始化选择的索引列表
    pick = []

    # 抓取边界框的坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算边界框的面积并获取索引进行排序
    # （如果未提供任何概率，只需在左下角的y坐标上排序）
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # 如果提供了概率，请对其进行排序
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # 继续循环，而某些索引仍保留在索引列表中
    while len(idxs) > 0:
        # 获取索引列表中的最后一个索引，并将索引值添加到选择的索引列表中
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # 为边界框的起点找到最大（x，y）坐标，为边界框的终点找到最小（x，y）坐标
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # 计算边界框的宽度和高度
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # 计算重叠率
        overlap = (w * h) / area[idxs[:last]]

        # 从索引列表中删除所有重叠大于提供的重叠阈值的所有索引
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    # 仅返回已选择的边界框
    if angles is not None:
        return boxes[pick].astype("int"), angles[pick]
    return boxes[pick].astype("int")
