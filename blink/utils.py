# coding=utf-8
import cv2
import depthai as dai
import numpy as np
from depthai_sdk import toPlanar


def run_nn(img, input_queue, width, height):
    frameNn = dai.ImgFrame()
    frameNn.setType(dai.ImgFrame.Type.BGR888p)
    frameNn.setWidth(width)
    frameNn.setHeight(height)
    frameNn.setData(toPlanar(img, (height, width)))
    input_queue.send(frameNn)


def scale_bbox(bboxes, scale_size=1.5):
    box = np.zeros_like(bboxes)
    scale_size = np.sqrt(scale_size)

    xy = ((bboxes[2:] + bboxes[:2]) / 2,)
    wh = (bboxes[2:] - bboxes[:2]) / 2
    box[:2] = xy - wh * scale_size
    box[2:] = xy + wh * scale_size
    return np.clip(box, 0, None)


def order_points(pts):
    """
    https://www.pyimagesearch.com//2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    """

    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    (tl, bl) = leftMost[np.argsort(leftMost[:, 1]), :]
    (tr, br) = rightMost[np.argsort(rightMost[:, 1]), :]

    # return the coordinates in
    # top-left, top-right, bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def get_mini_box_frame(img, contour):
    """
    https://www.jianshu.com/p/90572b07e48f
    """

    # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    center, size, angle = cv2.minAreaRect((contour[:, :2]).astype(int))

    box = cv2.boxPoints([center, size, angle])
    # 返回四个点顺序：左上→右上→右下→左下
    box = order_points(box)

    center, size = tuple(map(int, center)), tuple(map(int, size))

    affine = False
    if affine:
        """
        仿射变换
        第一种裁剪旋转矩形的方法是通过仿射变换旋转图像的方式。"""

        height, width = img.shape[:2]

        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)
        return img_crop, np.int0(box)

    else:
        """
        透视变换
        第二种裁剪旋转矩形的方法是通过透视变换直接将旋转矩形的四个顶点映射到正矩形的四个顶点。
        """
        w, h = size
        if w < h:
            w, h = h, w

        # specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32"
        )

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(box, dst)
        warped = cv2.warpPerspective(img, M, (w, h))
        return warped, np.int0(box)


def drawText(frame, text, org, color="black",bgcolor="gray"):
    if isinstance(color, str):
        color = color_tables.get(color.lower(), "black")
    if isinstance(bgcolor, str):
        bgcolor = color_tables.get(bgcolor.lower(), "gray")
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgcolor, 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


color_tables = {
    "aliceblue": (255, 248, 240),
    "antiquewhite": (215, 235, 250),
    "aqua": (255, 255, 0),
    "aquamarine": (212, 255, 127),
    "azure": (255, 255, 240),
    "beige": (220, 245, 245),
    "bisque": (196, 228, 255),
    "black": (0, 0, 0),
    "blanchedalmond": (205, 235, 255),
    "blue": (255, 0, 0),
    "blueviolet": (226, 43, 138),
    "brown": (42, 42, 165),
    "burlywood": (135, 184, 222),
    "cadetblue": (160, 158, 95),
    "chartreuse": (0, 255, 127),
    "chocolate": (30, 105, 210),
    "coral": (80, 127, 255),
    "cornflowerblue": (237, 149, 100),
    "cornsilk": (220, 248, 255),
    "crimson": (60, 20, 220),
    "cyan": (255, 255, 0),
    "darkblue": (139, 0, 0),
    "darkcyan": (139, 139, 0),
    "darkgoldenrod": (11, 134, 184),
    "darkgray": (169, 169, 169),
    "darkgreen": (0, 100, 0),
    "darkgrey": (169, 169, 169),
    "darkkhaki": (107, 183, 189),
    "darkmagenta": (139, 0, 139),
    "darkolivegreen": (47, 107, 85),
    "darkorange": (0, 140, 255),
    "darkorchid": (204, 50, 153),
    "darkred": (0, 0, 139),
    "darksalmon": (122, 150, 233),
    "darkseagreen": (143, 188, 143),
    "darkslateblue": (139, 61, 72),
    "darkslategray": (79, 79, 47),
    "darkslategrey": (79, 79, 47),
    "darkturquoise": (209, 206, 0),
    "darkviolet": (211, 0, 148),
    "deeppink": (147, 20, 255),
    "deepskyblue": (255, 191, 0),
    "dimgray": (105, 105, 105),
    "dimgrey": (105, 105, 105),
    "dodgerblue": (255, 144, 30),
    "firebrick": (34, 34, 178),
    "floralwhite": (240, 250, 255),
    "forestgreen": (34, 139, 34),
    "fuchsia": (255, 0, 255),
    "gainsboro": (220, 220, 220),
    "ghostwhite": (255, 248, 248),
    "gold": (0, 215, 255),
    "goldenrod": (32, 165, 218),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "green": (0, 128, 0),
    "greenyellow": (47, 255, 173),
    "honeydew": (240, 255, 240),
    "hotpink": (180, 105, 255),
    "indianred": (92, 92, 205),
    "indigo": (130, 0, 75),
    "ivory": (240, 255, 255),
    "khaki": (140, 230, 240),
    "lavender": (250, 230, 230),
    "lavenderblush": (245, 240, 255),
    "lawngreen": (0, 252, 124),
    "lemonchiffon": (205, 250, 255),
    "lightblue": (230, 216, 173),
    "lightcoral": (128, 128, 240),
    "lightcyan": (255, 255, 224),
    "lightgoldenrodyellow": (210, 250, 250),
    "lightgray": (211, 211, 211),
    "lightgreen": (144, 238, 144),
    "lightgrey": (211, 211, 211),
    "lightpink": (193, 182, 255),
    "lightsalmon": (122, 160, 255),
    "lightseagreen": (170, 178, 32),
    "lightskyblue": (250, 206, 135),
    "lightslategray": (153, 136, 119),
    "lightslategrey": (153, 136, 119),
    "lightsteelblue": (222, 196, 176),
    "lightyellow": (224, 255, 255),
    "lime": (0, 255, 0),
    "limegreen": (50, 205, 50),
    "linen": (230, 240, 250),
    "magenta": (255, 0, 255),
    "maroon": (0, 0, 128),
    "mediumaquamarine": (170, 205, 102),
    "mediumblue": (205, 0, 0),
    "mediumorchid": (211, 85, 186),
    "mediumpurple": (219, 112, 147),
    "mediumseagreen": (113, 179, 60),
    "mediumslateblue": (238, 104, 123),
    "mediumspringgreen": (154, 250, 0),
    "mediumturquoise": (204, 209, 72),
    "mediumvioletred": (133, 21, 199),
    "midnightblue": (112, 25, 25),
    "mintcream": (250, 255, 245),
    "mistyrose": (225, 228, 255),
    "moccasin": (181, 228, 255),
    "navajowhite": (173, 222, 255),
    "navy": (128, 0, 0),
    "oldlace": (230, 245, 253),
    "olive": (0, 128, 128),
    "olivedrab": (35, 142, 107),
    "orange": (0, 165, 255),
    "orangered": (0, 69, 255),
    "orchid": (214, 112, 218),
    "palegoldenrod": (170, 232, 238),
    "palegreen": (152, 251, 152),
    "paleturquoise": (238, 238, 175),
    "palevioletred": (147, 112, 219),
    "papayawhip": (213, 239, 255),
    "peachpuff": (185, 218, 255),
    "peru": (63, 133, 205),
    "pink": (203, 192, 255),
    "plum": (221, 160, 221),
    "powderblue": (230, 224, 176),
    "purple": (128, 0, 128),
    "red": (0, 0, 255),
    "rosybrown": (143, 143, 188),
    "royalblue": (225, 105, 65),
    "saddlebrown": (19, 69, 139),
    "salmon": (114, 128, 250),
    "sandybrown": (96, 164, 244),
    "seagreen": (87, 139, 46),
    "seashell": (238, 245, 255),
    "sienna": (45, 82, 160),
    "silver": (192, 192, 192),
    "skyblue": (235, 206, 135),
    "slateblue": (205, 90, 106),
    "slategray": (144, 128, 112),
    "slategrey": (144, 128, 112),
    "snow": (250, 250, 255),
    "springgreen": (127, 255, 0),
    "steelblue": (180, 130, 70),
    "tan": (140, 180, 210),
    "teal": (128, 128, 0),
    "thistle": (216, 191, 216),
    "tomato": (71, 99, 255),
    "turquoise": (208, 224, 64),
    "violet": (238, 130, 238),
    "wheat": (179, 222, 245),
    "white": (255, 255, 255),
    "whitesmoke": (245, 245, 245),
    "yellow": (0, 255, 255),
    "yellowgreen": (50, 205, 154),

}
