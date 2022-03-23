# coding=utf-8
import time
from pathlib import Path

import blobconverter
import click
import cv2
import depthai as dai
import numpy as np
from depthai_sdk import (
    getDeviceInfo,
    FPSHandler,
    toTensorResult,
)
from loguru import logger

from utils import run_nn, drawText

blobconverter.set_defaults(
    output_dir=Path(__file__).parent / Path("models"), optimizer_params=None
)

preview_size = (1080, 1080)


def create_pipeline(video):
    pipeline = dai.Pipeline()
    # pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    fire = pipeline.create(dai.node.NeuralNetwork)
    fire.setBlobPath(
        blobconverter.from_tf(
            "models/fire_detection_mobilenet_v2_100_224.pb",
            optimizer_params=[
                "--input_shape=[1,224,224,3]",
                "--scale_values=[255,255,255]",
            ],
            data_type="FP32",
            shaves=6,
        )
    )
    fire.setNumInferenceThreads(2)
    fire.input.setBlocking(False)
    fire.input.setQueueSize(2)

    if video:
        fire_in = pipeline.create(dai.node.XLinkIn)
        fire_in.setStreamName("fire_in")
        fire_in.setMaxDataSize(224 * 224 * 3)
        fire_in.out.link(fire.input)
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
        manip.initialConfig.setResize(224, 224)
        cam.preview.link(manip.inputImage)
        manip.out.link(fire.input)
        # cam.preview.link(face_det.input)

    fire_out = pipeline.create(dai.node.XLinkOut)
    fire_out.setStreamName("fire_nn")
    fire.out.link(fire_out.input)

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
def fire_detection(source, video_path, output, fps, frame_size):
    """
    基于深度学习的烟火检测器
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

            fire_in = device.getInputQueue("fire_in")
        else:
            cam_out = device.getOutputQueue("rgb")

        fire_nn = device.getOutputQueue("fire_nn")

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

        label_map = ["fire", "normal", "smoke"]
        while should_run():
            read_correctly, frame = get_frame()
            if not read_correctly:
                break
            frame_debug = frame.copy()
            if source:
                run_nn(frame_debug, fire_in, 224, 224)
            fire_data = toTensorResult(fire_nn.get()).get("final_result")
            fps_handler.tick("fire")
            conf = np.max(fire_data)
            if conf > 0.5:
                label = label_map[np.argmax(fire_data)]

                drawText(frame_debug, f"{label}", (10, 30), "black")
                drawText(frame_debug, f"{conf:.2%}", (10, 50), "black")
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
        fire_detection()
