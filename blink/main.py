# coding=utf-8
import queue
from pathlib import Path

import blobconverter
import cv2
import depthai as dai
import numpy as np
from depthai_sdk import getDeviceInfo, FPSHandler, toTensorResult, toPlanar
from loguru import logger

blobconverter.set_defaults(
    output_dir=Path(__file__).parent / Path("models"), optimizer_params=None
)

preview_size = (1080, 1080)


def create_pipeline():
    pipeline = dai.Pipeline()
    # pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.setInterleaved(False)
    # cam.setPreviewSize(300, 300)
    cam.setPreviewSize(preview_size)

    cam_out = pipeline.create(dai.node.XLinkOut)
    cam_out.setStreamName("rgb")
    cam.preview.link(cam_out.input)

    mesh = pipeline.create(dai.node.NeuralNetwork)
    mesh.setBlobPath(
        "models/face_landmark_openvino_2021.4_6shave.blob"
    )
    mesh.setNumInferenceThreads(2)
    mesh.input.setBlocking(False)
    mesh.input.setQueueSize(2)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(192, 192)
    cam.preview.link(manip.inputImage)
    manip.out.link(mesh.input)
    # cam.preview.link(face_det.input)

    # mesh_in = pipeline.create(dai.node.XLinkIn)
    # mesh_in.setStreamName("mesh_in")
    # mesh_in.setMaxDataSize(192 * 192 * 3)
    # mesh_in.out.link(mesh.input)

    mesh_out = pipeline.create(dai.node.XLinkOut)
    mesh_out.setStreamName("mesh_nn")
    mesh.out.link(mesh_out.input)

    eye = pipeline.create(dai.node.NeuralNetwork)
    eye.setBlobPath(
        blobconverter.from_zoo(
            "open-closed-eye-0001",
            shaves=6,
            # version=pipeline.getOpenVINOVersion()
        )
    )
    eye.setNumInferenceThreads(2)
    eye.input.setBlocking(False)
    eye.input.setQueueSize(2)

    eye_in = pipeline.create(dai.node.XLinkIn)
    eye_in.setStreamName("eye_in")
    eye_in.setMaxDataSize(32 * 32 * 3)
    eye_in.out.link(eye.input)

    eye_out = pipeline.create(dai.node.XLinkOut)
    eye_out.setStreamName("eye_nn")
    eye.out.link(eye_out.input)
    return pipeline


def blink():
    device_info = getDeviceInfo()  # type: dai.DeviceInfo
    with dai.Device(create_pipeline(), device_info) as device:
        fps_handler = FPSHandler()
        cam_out = device.getOutputQueue("rgb")
        # face_nn = device.getOutputQueue("face_nn")

        # mesh_in = device.getInputQueue("mesh_in")
        mesh_nn = device.getOutputQueue("mesh_nn")

        eye_in = device.getInputQueue("eye_in")
        eye_nn = device.getOutputQueue("eye_nn")
        left_eye_blink = []
        right_eye_blink = []

        left_number_of_blinks = 0
        right_number_of_blinks = 0
        while 1:
            frame = cam_out.get().getCvFrame()
            frame_debug = frame.copy()

            # run_nn(frame_debug, mesh_in, 192, 192)
            mesh_data = toTensorResult(mesh_nn.get())
            fps_handler.tick("Mesh")

            score = mesh_data.get("conv2d_31").reshape((1,))
            if score > 0.5:
                mesh = mesh_data.get("conv2d_21").reshape((468, 3))

                wh = np.array(frame_debug.shape[:2])[::-1]
                mesh *= np.array([*wh / 192, 1])

                for v in mesh.astype(int):
                    cv2.circle(frame_debug, v[:2], 1, (191, 255, 0))

                left_eye, left_box = get_mini_box_frame(
                    frame_debug,
                    np.array([mesh[71], mesh[107], mesh[116], mesh[197]]),
                )


                run_nn(left_eye, eye_in, 32, 32)
                left_eye_blink.append(
                    np.argmax(toTensorResult(eye_nn.get()).get("19"))
                )

                if (
                        len(left_eye_blink) > 5
                        and left_eye_blink[-1] not in left_eye_blink[-5:-1]
                ):
                    left_number_of_blinks += 1

                if len(left_eye_blink) > 20:
                    del left_eye_blink[0]

                right_eye, right_box = get_mini_box_frame(
                    frame_debug,
                    np.array([mesh[301], mesh[336], mesh[345], mesh[197]]),
                )

                run_nn(right_eye, eye_in, 32, 32)
                right_eye_blink.append(
                    np.argmax(toTensorResult(eye_nn.get()).get("19"))
                )
                if (
                        len(right_eye_blink) > 5
                        and right_eye_blink[-1] not in right_eye_blink[-5:-1]
                ):
                    right_number_of_blinks += 1

                if len(right_eye_blink) > 20:
                    del right_eye_blink[0]

                cv2.drawContours(frame_debug, [np.int0(right_box)], -1, (0, 139, 0), 2)
                cv2.drawContours(frame_debug, [np.int0(left_box)], -1, (0, 139, 0), 2)
                cv2.imshow("left", left_eye)
                cv2.imshow("right", right_eye)



            cv2.putText(
                frame_debug,
                f"NumberOfBlinksOfRightEye:{right_number_of_blinks}",
                (20, 50),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5,
                color=(34, 34, 178),
                lineType=1,
            )
            cv2.putText(
                frame_debug,
                f"NumberOfBlinksOfLeftEye:{left_number_of_blinks}",
                (20, 80),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5,
                color=(34, 34, 178),
                lineType=1,
            )

            cv2.imshow("", frame_debug)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            fps_handler.printStatus()


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
        cv2.imshow("img_rot", img_rot)
        img_crop = cv2.getRectSubPix(
            img_rot, size, center
        )
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
        dst = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(box, dst)
        warped = cv2.warpPerspective(img, M, (w, h))
        return warped, np.int0(box)


if __name__ == "__main__":
    with logger.catch():
        blink()
