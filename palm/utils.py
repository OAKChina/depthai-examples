# coding=utf-8
import depthai as dai
import numpy as np
from Xlib.display import Display
from depthai_sdk import toPlanar
from pynput.mouse import Controller

mouse = Controller()
screen = Display().screen()
screen_size = (screen.width_in_pixels, screen.height_in_pixels)


# print("screen_size: ",screen_size)
def distance(pt1, pt2):
    """
    两点间距离
    """
    assert len(pt1) == len(pt2), f"两点维度要一致，pt1:{len(pt1)}维, pt2:{len(pt2)}维"
    return np.sqrt(np.float_power(np.array(pt1) - pt2, 2).sum())


def point_mapping(dot, center, original_side_length, target_side_length):
    """

    :param dot: 点座标
    :param center: 帧中心点座标
    :param original_side_length: 源边长
    :param target_side_length: 目标边长
    :return:
    """
    if isinstance(original_side_length, (int, float)):
        original_side_length = np.array((original_side_length, original_side_length))
    if isinstance(target_side_length, (int, float)):
        target_side_length = np.array((target_side_length, target_side_length))

    return center + (np.array(dot) - center) * (
        np.array(target_side_length) / original_side_length
    )


def move_mouse(dots, dot, frame_shape):
    if len(dots) > 0 and distance(dot, dots[-1]) < 5:
        dot = dots[-1]
    dots.append(dot)
    if len(dots) >= 10:
        dot = np.mean(dots, axis=0)
        dot_s = point_mapping(dot, (64, 64), 108, 128)
        dot_l = point_mapping(dot_s, (0, 0), frame_shape, screen_size)
        mouse.position = (dot_l[0], dot_l[1])
        dots.pop(0)
    return dots


def run_nn(img, input_queue, width, height):
    frameNn = dai.ImgFrame()
    frameNn.setType(dai.ImgFrame.Type.BGR888p)
    frameNn.setWidth(width)
    frameNn.setHeight(height)
    frameNn.setData(toPlanar(img, (height, width)))
    input_queue.send(frameNn)


def raw_to_detections(raw_box_tensor, raw_score_tensor, anchors_, shape, num_keypoints):
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
    detection_boxes = decode_boxes(raw_box_tensor, anchors_, shape, num_keypoints)
    detection_scores = sigmoid(raw_score_tensor).squeeze(-1)
    output_detections = []
    for i in range(raw_box_tensor.shape[0]):
        boxes = detection_boxes[i]
        scores = np.expand_dims(detection_scores[i], -1)
        output_detections.append(np.concatenate((boxes, scores), -1))
    return output_detections


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
        keypoint_x = raw_boxes[..., offset] / x_scale * anchors[:, 2] + anchors[:, 0]
        keypoint_y = (
            raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
        )
        boxes[..., offset] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes


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
        overlap = np.divide(w * h, area[idxs[:last]])

        # 从索引列表中删除所有重叠大于提供的重叠阈值的所有索引
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    # 仅返回已选择的边界框
    if angles is not None:
        return boxes[pick].astype("int"), angles[pick]
    return boxes[pick].astype("int")
