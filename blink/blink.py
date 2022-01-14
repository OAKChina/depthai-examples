# coding=utf-8
import time
from pathlib import Path

import blobconverter
import click
import cv2
import depthai as dai
import numpy as np
from depthai_sdk import getDeviceInfo, FPSHandler, toTensorResult
from loguru import logger

from utils import run_nn, get_mini_box_frame, drawText

blobconverter.set_defaults(
    output_dir=Path(__file__).parent / Path("models"), optimizer_params=None
)

preview_size = (1080, 1080)


def create_pipeline(video):
    pipeline = dai.Pipeline()
    # pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    mesh = pipeline.create(dai.node.NeuralNetwork)
    mesh.setBlobPath("models/face_landmark_openvino_2021.4_6shave.blob")
    mesh.setNumInferenceThreads(2)
    mesh.input.setBlocking(False)
    mesh.input.setQueueSize(2)

    if video:
        mesh_in = pipeline.create(dai.node.XLinkIn)
        mesh_in.setStreamName("mesh_in")
        mesh_in.setMaxDataSize(192 * 192 * 3)
        mesh_in.out.link(mesh.input)
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
        manip.initialConfig.setResize(192, 192)
        cam.preview.link(manip.inputImage)
        manip.out.link(mesh.input)
        # cam.preview.link(face_det.input)

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
def blink(source, video_path, output, fps, frame_size):
    """
    在视频流中实时检测和计数眨眼次数
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

            mesh_in = device.getInputQueue("mesh_in")
        else:
            cam_out = device.getOutputQueue("rgb")

        mesh_nn = device.getOutputQueue("mesh_nn")

        eye_in = device.getInputQueue("eye_in")
        eye_nn = device.getOutputQueue("eye_nn")
        left_eye_blink = []
        right_eye_blink = []

        left_number_of_blinks = 0
        right_number_of_blinks = 0

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
                run_nn(frame_debug, mesh_in, 192, 192)
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
                left_eye_blink.append(np.argmax(toTensorResult(eye_nn.get()).get("19")))

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

                cv2.drawContours(frame_debug, [right_box], -1, (0, 139, 0), 2)
                cv2.drawContours(frame_debug, [left_box], -1, (0, 139, 0), 2)
                cv2.imshow("left", left_eye)
                cv2.imshow("right", right_eye)

            drawText(
                frame_debug,
                f"NumberOfBlinksOfRightEye: {right_number_of_blinks}",
                (20, 50),
                color="red",
            )
            drawText(
                frame_debug,
                f"NumberOfBlinksOfLeftEye: {left_number_of_blinks}",
                (20, 80),
                color="red",
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
        blink()
