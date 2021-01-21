# coding=utf-8
import argparse
import time
from datetime import datetime, timedelta

import cv2
import depthai
import numpy as np
import pretty_errors
from scipy.special.cython_special import expit


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
    raise ValueError("缺少推理源！使用 “ -cam” 在 DepthAI 摄像机上运行，或使用 “ -vid <path>” 在视频文件上运行")


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


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [
        val
        for channel in cv2.resize(arr, shape).transpose(2, 0, 1)
        for y_col in channel
        for val in y_col
    ]


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
    arr = arr[: np.where(arr == -1)[0][0]]
    arr = arr.reshape((arr.size // 7, 7))
    return arr


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
    has_results = wait_for_results(x_out)
    if not has_results:
        raise RuntimeError("No data from nn!")
    return x_out.tryGet()


def frame_norm(frame, *xy_vals):
    height, width = frame.shape[:2]
    result = []
    for i, val in enumerate(xy_vals):
        if i % 2 == 0:
            result.append(max(0, min(width, int(val * width))))
        else:
            result.append(max(0, min(height, int(val * height))))
    return result


def draw_3d_axis(image, head_pose, origin, size=50):
    roll = head_pose[0] * np.pi / 180
    pitch = head_pose[1] * np.pi / 180
    yaw = -(head_pose[2] * np.pi / 180)

    # X axis (red)
    x1 = size * (np.cos(yaw) * np.cos(roll)) + origin[0]
    y1 = (
        size
        * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw))
        + origin[1]
    )
    cv2.line(image, (origin[0], origin[1]), (int(x1), int(y1)), (0, 0, 255), 3)

    # Y axis (green)
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + origin[0]
    y2 = (
        size
        * (-np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll))
        + origin[1]
    )
    cv2.line(image, (origin[0], origin[1]), (int(x2), int(y2)), (0, 255, 0), 3)

    # Z axis (blue)
    x3 = size * (-np.sin(yaw)) + origin[0]
    y3 = size * (np.cos(yaw) * np.sin(pitch)) + origin[1]
    cv2.line(image, (origin[0], origin[1]), (int(x3), int(y3)), (255, 0, 0), 2)

    return image


def correction(frame, coords):
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(angle + 90) if angle < -45 else -angle
    print(f"倾斜角度为：{angle}度")
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 0.8)
    corr = cv2.warpAffine(
        frame,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
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
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
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
    return 1.0 / (1.0 + np.exp(-x))
    # return expit(x)
    # return (1.0 + np.tanh(0.5 * x)) * 0.5


_conf_threshold = 0.5


def rotated_rectangle(bbox, angle):
    x0, y0, x1, y1 = bbox
    width = abs(x0 - x1)
    height = abs(y0 - y1)
    x = int(x0 + width * 0.5)
    y = int(y0 + height * 0.5)

    pt1_1 = (int(x + width / 2), int(y + height / 2))
    pt2_1 = (int(x + width / 2), int(y - height / 2))
    pt3_1 = (int(x - width / 2), int(y - height / 2))
    pt4_1 = (int(x - width / 2), int(y + height / 2))

    t = np.array(
        [
            [np.cos(angle), -np.sin(angle), x - x * np.cos(angle) + y * np.sin(angle)],
            [np.sin(angle), np.cos(angle), y - x * np.sin(angle) - y * np.cos(angle)],
            [0, 0, 1],
        ]
    )

    tmp_pt1_1 = np.array([[pt1_1[0]], [pt1_1[1]], [1]])
    tmp_pt1_2 = np.dot(t, tmp_pt1_1)
    pt1_2 = (int(tmp_pt1_2[0][0]), int(tmp_pt1_2[1][0]))

    tmp_pt2_1 = np.array([[pt2_1[0]], [pt2_1[1]], [1]])
    tmp_pt2_2 = np.dot(t, tmp_pt2_1)
    pt2_2 = (int(tmp_pt2_2[0][0]), int(tmp_pt2_2[1][0]))

    tmp_pt3_1 = np.array([[pt3_1[0]], [pt3_1[1]], [1]])
    tmp_pt3_2 = np.dot(t, tmp_pt3_1)
    pt3_2 = (int(tmp_pt3_2[0][0]), int(tmp_pt3_2[1][0]))

    tmp_pt4_1 = np.array([[pt4_1[0]], [pt4_1[1]], [1]])
    tmp_pt4_2 = np.dot(t, tmp_pt4_1)
    pt4_2 = (int(tmp_pt4_2[0][0]), int(tmp_pt4_2[1][0]))

    points = np.array([pt1_2, pt2_2, pt3_2, pt4_2])

    return points


def non_max_suppression(boxes, probs=None, angles=None, overlapThresh=0.3):
    """
    NMS用于消除多重检测

    :param boxes: boxes of objects. x1,y1,x2,y2
    :param probs: scores of objects
    :param angles:
    :param overlapThresh:
    :return: 返回已选择的边界框
    """
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
    return boxes[pick].astype("int"), angles[pick]


def decode_predictions(scores, geometry1, geometry2):
    # 从分数卷中获取行数和列数，然后初始化我们的边界框矩形集和相应的置信度分数
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    angles = []

    # 循环行数
    for y in range(0, numRows):
        # 提取分数（概率），然后提取几何数据，以得出围绕文本的潜在边界框坐标
        scores_data = scores[0, 0, y]
        x_data0 = geometry1[0, 0, y]
        x_data1 = geometry1[0, 1, y]
        x_data2 = geometry1[0, 2, y]
        x_data3 = geometry1[0, 3, y]
        angles_data = geometry2[0, 0, y]

        # 循环遍历列数
        for x in range(0, numCols):
            # 如果我们的分数没有足够的可能性，
            # 请忽略它
            if scores_data[x] < _conf_threshold:
                continue

            # 计算偏移因子，因为我们生成的特征图将比输入图像小4倍
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # 提取预测的旋转角度，然后计算正弦和余弦
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # 使用几何体体积导出边界框的宽度和高度
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            # 计算文本预测边界框的开始和结束（x，y）坐标
            end_x = int(offsetX + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offsetY - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            # 将边界框坐标和概率分数添加到我们各自的列表中
            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])
            angles.append(angle)

    # 返回边界框和相关置信度的元组
    return rects, confidences, angles


def decode_east(nnet_packet, **kwargs):
    scores = nnet_packet.get_tensor(0)
    geometry1 = nnet_packet.get_tensor(1)
    geometry2 = nnet_packet.get_tensor(2)
    bboxes, confs, angles = decode_predictions(scores, geometry1, geometry2)
    boxes, angles = non_max_suppression(
        np.array(bboxes), probs=confs, angles=np.array(angles)
    )
    boxesangles = (boxes, angles)
    return boxesangles


def show_east(boxes_angles, frame, **kwargs):
    bboxes = boxes_angles[0]
    angles = boxes_angles[1]
    for ((X0, Y0, X1, Y1), angle) in zip(bboxes, angles):
        width = abs(X0 - X1)
        height = abs(Y0 - Y1)
        c_x = int(X0 + width * 0.5)
        c_y = int(Y0 + height * 0.5)

        rot_rect = ((c_x, c_y), ((X1 - X0), (Y1 - Y0)), angle * (-1))
        points = rotated_rectangle(frame, rot_rect)
        cv2.polylines(
            frame,
            [points],
            isClosed=True,
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_8,
        )

    return frame


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    mat = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, mat, (max_width, max_height))

    return warped
