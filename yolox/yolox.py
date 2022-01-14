# coding=utf-8
import time
from pathlib import Path

import blobconverter
import click
import cv2
import depthai as dai
import numpy as np
from depthai_sdk import FPSHandler, toTensorResult, getDeviceInfo
from loguru import logger

from demo_utils import multiclass_nms, demo_postprocess, run_nn, to_planar
from label_classes import *
from visualize import vis

preview_size = (1280, 720)

# To use a different NN, change `size` and `nnPath` here:
parentDir = Path(__file__).parent
shaves = 6
blobconverter.set_defaults(output_dir=parentDir / Path("models"))
size = (320, 320)

MODELS = {
    "yolox": (parentDir / Path("models/yolox_nano_320x320.onnx")).as_posix(),
    "helmet": (parentDir / Path("models/helmet_detection_yolox.onnx")).as_posix()
}

LABELS = {
    "yolox": COCO_CLASSES,
    "helmet": HELMET_CLASSES
}


def create_pipeline(video, model_name, model_size):
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

    # NeuralNetwork
    yoloDet = pipeline.createNeuralNetwork()
    yoloDet.setBlobPath(
        blobconverter.from_onnx(
            model=MODELS.get(model_name),
            optimizer_params=[
                "--scale_values=[58.395, 57.12 , 57.375]",
                "--mean_values=[123.675, 116.28 , 103.53]",
            ],
            shaves=shaves,
        )
    )

    yolox_det_nn_xout = pipeline.createXLinkOut()
    yolox_det_nn_xout.setStreamName("yolox_det_nn")
    yoloDet.out.link(yolox_det_nn_xout.input)

    if video:
        yolox_det_in = pipeline.createXLinkIn()
        yolox_det_in.setStreamName("yolox_det_in")
        yolox_det_in.setMaxDataSize(model_size * model_size * 3)
        yolox_det_in.out.link(yoloDet.input)

    else:
        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(preview_size)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)

        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

        manip = pipeline.createImageManip()
        manip.initialConfig.setResizeThumbnail(model_size, model_size, 114, 114, 114)
        manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
        cam.preview.link(manip.inputImage)
        manip.out.link(yoloDet.input)
        # cam.preview.link(yoloDet.input)
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
    "-m",
    "--model_name",
    type=click.Choice([*MODELS.keys()], case_sensitive=False),
    default="yolox",
    help="模型名称",
    show_default=True,
)
@click.option(
    "-size",
    "--model_size",
    type=int,
    default=320,
    help="模型输入大小",
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
def yolox(source, model_name, model_size, video_path, output, fps, frame_size):
    """
    基于 yolox 的目标检测器
    """
    # click.echo(click.get_current_context().params)
    device_info = getDeviceInfo()  # type: dai.DeviceInfo
    with dai.Device(create_pipeline(source, model_name, model_size), device_info) as device:
        print("Starting pipeline...")
        fps_handler = FPSHandler()
        if source:
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_shape = [frame_height, frame_width]
            print("CAP_PROP_FRAME_SHAPE: %s" % frame_shape)
            cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
            print("CAP_PROP_FPS: %d" % cap_fps)

            yolox_det_in = device.getInputQueue("yolox_det_in")
        else:
            cam_out = device.getOutputQueue("cam_out", 1, True)
        yolox_det_nn = device.getOutputQueue("yolox_det_nn")

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
            fps_handler.tick("Frame")
            if not read_correctly:
                break

            frame_debug = frame.copy()
            if source:
                run_nn(yolox_det_in, to_planar(frame, (model_size, model_size)), model_size, model_size)
            yolox_det_data = yolox_det_nn.get()

            res = toTensorResult(yolox_det_data).get("output")
            fps_handler.tick("nn")
            predictions = demo_postprocess(res, (model_size, model_size), p6=False)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4, None] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

            input_shape = np.array([model_size, model_size])
            min_r = (input_shape / frame.shape[:2]).min()
            offset = (np.array(frame.shape[:2]) * min_r - input_shape) / 2
            offset = np.ravel([offset, offset])
            boxes_xyxy = (boxes_xyxy + offset[::-1]) / min_r

            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.2)

            if dets is not None:
                final_boxes = dets[:, :4]
                final_scores, final_cls_inds = dets[:, 4], dets[:, 5]
                frame_debug = vis(
                    frame_debug,
                    final_boxes,
                    final_scores,
                    final_cls_inds,
                    conf=0.5,
                    class_names=LABELS[model_name],
                )
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
        yolox()
