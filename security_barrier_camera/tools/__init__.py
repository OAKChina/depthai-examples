import argparse
from datetime import datetime, timedelta
from math import cos, sin

import cv2
import depthai
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-nd", "--no-debug", action="store_true", help="Prevent debug output"
)
parser.add_argument(
    "-cam",
    "--camera",
    action="store_true",
    help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)",
)
parser.add_argument(
    "-vid",
    "--video",
    type=str,
    help="Path to video file to be used for inference (conflicts with -cam)",
)

parser.add_argument(
    "-db",
    "--databases",
    action="store_true",
    help="保存数据库",
)

parser.add_argument(
    "-n",
    "--name",
    type=str,
    help="库名",
)

args = parser.parse_args()

debug = not args.no_debug

is_db = args.databases

if args.camera and args.video:
    raise ValueError(
        'Incorrect command line parameters! "-cam" cannot be used with "-vid"!'
    )
elif args.camera is False and args.video is None:
    raise ValueError(
        'Missing inference source! Either use "-cam" to run on DepthAI camera or "-vid <path>" to run on video file'
    )


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


def to_bbox_result(nn_data):
    """

    :param nn_data:
    :return:
    """
    arr = to_nn_result(nn_data)
    arr = arr[: np.where(arr == -1)[0][0]]
    arr = arr.reshape((arr.size // 7, 7))
    return arr


def run_nn(x_in, x_out, in_dict):
    """

    :param x_in: Xlinkin
    :param x_out: Xlinkout
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
    return x_out.get()


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
    x1 = size * (cos(yaw) * cos(roll)) + origin[0]
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + origin[1]
    cv2.line(image, (origin[0], origin[1]), (int(x1), int(y1)), (0, 0, 255), 3)

    # Y axis (green)
    x2 = size * (-cos(yaw) * sin(roll)) + origin[0]
    y2 = (
            size * (-cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + origin[1]
    )
    cv2.line(image, (origin[0], origin[1]), (int(x2), int(y2)), (0, 255, 0), 3)

    # Z axis (blue)
    x3 = size * (-sin(yaw)) + origin[0]
    y3 = size * (cos(yaw) * sin(pitch)) + origin[1]
    cv2.line(image, (origin[0], origin[1]), (int(x3), int(y3)), (255, 0, 0), 2)

    return image


def correction(frame, coords):
    frame = frame.copy()
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(angle + 90) if angle < -45 else -angle

    print(f"倾斜角度为：{angle}度")
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 0.8)
    corr = cv2.warpAffine(
        frame,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        # borderMode=cv2.BORDER_WRAP
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
    similiarity = np.dot(a, b.T) / (a_norm * b_norm)
    # dist = 1.0 - similiarity  # 余弦距离 = 1- 余弦相似度
    # return dist
    return similiarity