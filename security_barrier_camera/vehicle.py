# coding=utf-8
import time
from pathlib import Path

import blobconverter
import click
import cv2
import depthai as dai
import numpy as np
from depthai_sdk import getDeviceInfo, FPSHandler, toTensorResult, frameNorm
from loguru import logger

from utils import run_nn, drawText, pad_resize

blobconverter.set_defaults(
    output_dir=Path(__file__).parent / Path("models"), optimizer_params=None
)

preview_size = (1080, 1080)


def create_pipeline(video):
    pipeline = dai.Pipeline()
    # pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    vehicle = pipeline.create(dai.node.MobileNetDetectionNetwork)
    vehicle.setBlobPath(
        blobconverter.from_zoo("vehicle-license-plate-detection-barrier-0106", shaves=6)
    )
    vehicle.setConfidenceThreshold(0.8)
    vehicle.setNumInferenceThreads(2)
    vehicle.input.setBlocking(False)
    vehicle.input.setQueueSize(2)

    if video:
        vehicle_in = pipeline.create(dai.node.XLinkIn)
        vehicle_in.setStreamName("vehicle_in")
        vehicle_in.setMaxDataSize(300 * 300 * 3)
        vehicle_in.out.link(vehicle.input)
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
        manip.initialConfig.setResize(300, 300)
        cam.preview.link(manip.inputImage)
        manip.out.link(vehicle.input)
        # cam.preview.link(face_det.input)

    vehicle_out = pipeline.create(dai.node.XLinkOut)
    vehicle_out.setStreamName("vehicle_nn")
    vehicle.out.link(vehicle_out.input)

    attr = pipeline.create(dai.node.NeuralNetwork)
    attr.setBlobPath(
        blobconverter.from_zoo(
            "vehicle-attributes-recognition-barrier-0039",
            shaves=6,
            # version=pipeline.getOpenVINOVersion()
        )
    )
    attr.setNumInferenceThreads(2)
    attr.input.setBlocking(False)
    attr.input.setQueueSize(2)

    attr_in = pipeline.create(dai.node.XLinkIn)
    attr_in.setStreamName("attr_in")
    attr_in.setMaxDataSize(72 * 72 * 3)
    attr_in.out.link(attr.input)

    attr_out = pipeline.create(dai.node.XLinkOut)
    attr_out.setStreamName("attr_nn")
    attr.out.link(attr_out.input)

    license = pipeline.create(dai.node.NeuralNetwork)
    license.setBlobPath(
        blobconverter.from_zoo(
            "license-plate-recognition-barrier-0007",
            shaves=6,
            # version=pipeline.getOpenVINOVersion()
        )
    )
    license.setNumInferenceThreads(2)
    license.input.setBlocking(False)
    license.input.setQueueSize(2)

    license_in = pipeline.create(dai.node.XLinkIn)
    license_in.setStreamName("license_in")
    license_in.setMaxDataSize(24 * 94 * 3)
    license_in.out.link(license.input)

    license_out = pipeline.create(dai.node.XLinkOut)
    license_out.setStreamName("license_nn")
    license.out.link(license_out.input)

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
def vehicle(source, video_path, output, fps, frame_size):
    """
    车辆属性识别和车牌识别
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

            vehicle_in = device.getInputQueue("vehicle_in")
        else:
            cam_out = device.getOutputQueue("rgb")

        vehicle_nn = device.getOutputQueue("vehicle_nn")

        attr_in = device.getInputQueue("attr_in")
        attr_nn = device.getOutputQueue("attr_nn")
        license_in = device.getInputQueue("license_in")
        license_nn = device.getOutputQueue("license_nn")

        colors = ["white", "gray", "yellow", "red", "green", "blue", "black"]
        types = ["car", "bus", "truck", "van"]

        license_dict = [
            *map(chr, range(48, 58)),
            "<Anhui>",
            "<Beijing>",
            "<Chongqing>",
            "<Fujian>",
            "<Gansu>",
            "<Guangdong>",
            "<Guangxi>",
            "<Guizhou>",
            "<Hainan>",
            "<Hebei>",
            "<Heilongjiang>",
            "<Henan>",
            "<HongKong>",
            "<Hubei>",
            "<Hunan>",
            "<InnerMongolia>",
            "<Jiangsu>",
            "<Jiangxi>",
            "<Jilin>",
            "<Liaoning>",
            "<Macau>",
            "<Ningxia>",
            "<Qinghai>",
            "<Shaanxi>",
            "<Shandong>",
            "<Shanghai>",
            "<Shanxi>",
            "<Sichuan>",
            "<Tianjin>",
            "<Tibet>",
            "<Xinjiang>",
            "<Yunnan>",
            "<Zhejiang>",
            "<police>",
            *map(chr, range(65, 91)),
        ]

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
                run_nn(frame_debug, vehicle_in, 300, 300)
            vehicle_data = vehicle_nn.get().detections
            fps_handler.tick("vehicle")

            for bbox in vehicle_data:
                if bbox.label == 1:
                    vehicle_coord = frameNorm(
                        frame_debug, [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
                    )

                    cv2.rectangle(
                        frame_debug, vehicle_coord[:2], vehicle_coord[2:], (128, 128, 0)
                    )

                    vehicle_frame = frame[
                        vehicle_coord[1] : vehicle_coord[3],
                        vehicle_coord[0] : vehicle_coord[2],
                    ]
                    run_nn(vehicle_frame, attr_in, 72, 72)
                    attr_data = toTensorResult(attr_nn.get())
                    color_ = colors[attr_data.get("color").argmax()]
                    type_ = types[attr_data.get("type").argmax()]
                    drawText(
                        frame_debug,
                        color_,
                        (vehicle_coord[0] + 10, vehicle_coord[1] + 10),
                    )
                    drawText(
                        frame_debug,
                        type_,
                        (vehicle_coord[0] + 10, vehicle_coord[1] + 25),
                    )
                elif bbox.label == 2:
                    plate_coord = frameNorm(
                        frame_debug, [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
                    )
                    cv2.rectangle(
                        frame_debug, plate_coord[:2], plate_coord[2:], (128, 128, 0)
                    )

                    plate_frame = frame[
                        plate_coord[1] : plate_coord[3],
                        plate_coord[0] : plate_coord[2],
                    ]
                    plate_frame = pad_resize(plate_frame, (24, 94))

                    # cv2.imshow("pl",plate_frame.astype(np.uint8))
                    run_nn(plate_frame, license_in, 94, 24)
                    license_data = (
                        toTensorResult(license_nn.get())
                        .get("d_predictions.0")
                        .squeeze()
                    )
                    plate_str = ""
                    for j in license_data:
                        if j == -1:
                            break
                        plate_str += license_dict[j]
                    drawText(
                        frame_debug,
                        plate_str,
                        (plate_coord[0] - 10, plate_coord[1] - 10),
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
        vehicle()
