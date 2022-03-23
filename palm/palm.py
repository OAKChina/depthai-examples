# coding=utf-8
import time
from pathlib import Path

import click
import cv2
import depthai as dai
import numpy as np
from depthai_sdk import toTensorResult, frameNorm, getDeviceInfo, FPSHandler
from loguru import logger

from utils import run_nn, raw_to_detections, non_max_suppression, move_mouse

preview_size = (640, 640)


def create_pipeline(video):
    pipeline = dai.Pipeline()
    palm = pipeline.create(dai.node.NeuralNetwork)
    palm.setBlobPath("models/palm.blob")
    palm.setNumInferenceThreads(2)
    palm.input.setBlocking(False)
    palm.input.setQueueSize(2)

    if video:
        palm_in = pipeline.create(dai.node.XLinkIn)
        palm_in.setStreamName("palm_in")
        palm_in.setMaxDataSize(128 * 128 * 3)
        palm_in.out.link(palm.input)
    else:
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setInterleaved(False)
        cam.setPreviewSize(preview_size)

        cam_out = pipeline.create(dai.node.XLinkOut)
        cam_out.setStreamName("rgb")
        cam.preview.link(cam_out.input)
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setResize(128, 128)
        cam.preview.link(manip.inputImage)
        manip.out.link(palm.input)
        # cam.preview.link(face_det.input)

    palm_out = pipeline.create(dai.node.XLinkOut)
    palm_out.setStreamName("palm_nn")
    palm.out.link(palm_out.input)

    return pipeline


@click.command(
    context_settings=dict(
        help_option_names=["-h", "--help"], token_normalize_func=lambda x: x.lower()
    )
)
@click.option(
    "-vid/-cam",
    "--video/--camera",
    "source",
    is_flag=True,
    help="使用 DepthAI 4K RGB 摄像头或视频文件进行推理",
    show_default=True,
)
@click.option(
    "-p",
    "--video_path",
    type=click.Path(exists=True),
    help="指定用于推理的视频文件的路径",
    show_default=True,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(resolve_path=True, path_type=Path),
    help="指定用于保存的视频文件的路径",
)
@click.option("-fps", "--fps", default=30, type=int, help="保存视频的帧率", show_default=True)
@click.option(
    "-s",
    "--frame_size",
    default=preview_size,
    type=(int, int),
    help="保存视频的宽度，高度",
    show_default=True,
)
def palm(source, video_path, output, fps, frame_size):
    """
    手掌检测,控制鼠标
    """
    # click.echo(click.get_current_context().params)
    device_info = getDeviceInfo()  # type: dai.DeviceInfo
    with dai.Device(create_pipeline(source), device_info) as device:
        fps_handler = FPSHandler()
        if source:
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_shape = [frame_height, frame_width]
            print("CAP_PROP_FRAME_SHAPE: %s" % frame_shape)
            cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
            print("CAP_PROP_FPS: %d" % cap_fps)

            palm_in = device.getInputQueue("palm_in")
        else:
            cam_out = device.getOutputQueue("rgb")

        palm_nn = device.getOutputQueue("palm_nn")

        dots = []

        def should_run():
            if source:
                return cap.isOpened()
            else:
                return True

        def get_frame():
            if source:
                return cap.read()
            else:
                return True, cam_out.get().getCvFrame()

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output), fourcc, fps, frame_size)

        while should_run():
            read_correctly, frame = get_frame()
            if not read_correctly:
                break
            frame_debug = frame.copy()
            if source:
                run_nn(frame_debug, palm_in, 128, 128)
            results = toTensorResult(palm_nn.get())
            fps_handler.tick("palm")

            num_keypoints = 7
            min_score_thresh = 0.7
            anchors = np.load("anchors_palm.npy")

            raw_box_tensor = results.get("regressors")  # regress
            raw_score_tensor = results.get("classificators")  # classification
            detections = raw_to_detections(
                raw_box_tensor, raw_score_tensor, anchors, (128, 128), num_keypoints
            )

            palm_coords = [
                frameNorm(frame, obj[:4])
                for det in detections
                for obj in det
                if obj[-1] > min_score_thresh
            ]

            palm_confs = [
                obj[-1]
                for det in detections
                for obj in det
                if obj[-1] > min_score_thresh
            ]

            if len(palm_coords) > 0:
                palm_coords = non_max_suppression(
                    boxes=np.concatenate(palm_coords).reshape(-1, 4),
                    probs=palm_confs,
                    overlapThresh=0.1,
                )

                for bbox in palm_coords:
                    cv2.rectangle(frame_debug, bbox[:2], bbox[2:], (10, 245, 10))
                    dot_x = (bbox[2] + bbox[0]) / 2
                    dot_y = (bbox[3] + bbox[1]) / 2
                    dots = move_mouse(dots, (dot_x, dot_y), frame.shape[:2])
            cv2.imshow("", frame_debug)
            if output:
                writer.write(cv2.resize(frame_debug, frame_size))

            key = cv2.waitKey(1)
            if key in [ord("q"), 27]:
                break
            elif key == ord("s"):
                cv2.imwrite(
                    "saved_%s.jpg" % time.strftime("%Y%m%d_%H%M%S", time.localtime()),
                    frame_debug,
                )
        fps_handler.printStatus()
        if source:
            cap.release()
        if output:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    with logger.catch():
        palm()
