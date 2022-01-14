# coding=utf-8
import time
from pathlib import Path

import blobconverter
import click
import cv2
import depthai as dai
from depthai_sdk import frameNorm, toTensorResult, getDeviceInfo, FPSHandler
from loguru import logger

from utils import run_nn, rotate_frame, drawText

preview_size = (1280, 720)
shaves = 6
blobconverter.set_defaults(output_dir=Path("models"), optimizer_params=None)


def create_pipeline(video):
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

    face_nn = pipeline.createMobileNetDetectionNetwork()
    face_nn.setBlobPath(
        str(blobconverter.from_zoo("face-detection-retail-0004", shaves=shaves))
    )
    face_nn.setConfidenceThreshold(0.5)
    face_nn.input.setBlocking(False)

    if video:
        face_in = pipeline.createXLinkIn()
        face_in.setStreamName("face_in")
        face_in.out.link(face_nn.input)

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
        manip = pipeline.create(dai.node.ImageManip)  # type: dai.node.ImageManip
        manip.initialConfig.setResize(300, 300)
        manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
        manip.initialConfig.setKeepAspectRatio(False)
        cam.preview.link(manip.inputImage)
        manip.out.link(face_nn.input)

    face_nn_xout = pipeline.createXLinkOut()
    face_nn_xout.setStreamName("face_nn")
    face_nn.out.link(face_nn_xout.input)

    head_pose_nn = pipeline.create(dai.node.NeuralNetwork)
    head_pose_nn.setBlobPath(
        str(blobconverter.from_zoo("head-pose-estimation-adas-0001", shaves=shaves))
    )
    head_pose_nn.input.setBlocking(False)

    head_pose_in = pipeline.createXLinkIn()
    head_pose_in.setStreamName("head_pose_in")
    head_pose_in.out.link(head_pose_nn.input)

    head_pose_nn_xout = pipeline.createXLinkOut()
    head_pose_nn_xout.setStreamName("head_pose_nn")
    head_pose_nn.out.link(head_pose_nn_xout.input)

    age_nn = pipeline.create(dai.node.NeuralNetwork)
    age_nn.setBlobPath(
        str(blobconverter.from_zoo("age-gender-recognition-retail-0013", shaves=shaves))
    )
    age_nn.input.setBlocking(False)

    age_in = pipeline.createXLinkIn()
    age_in.setStreamName("age_in")
    age_in.out.link(age_nn.input)

    age_nn_xout = pipeline.createXLinkOut()
    age_nn_xout.setStreamName("age_nn")
    age_nn.out.link(age_nn_xout.input)

    emo_nn = pipeline.create(dai.node.NeuralNetwork)
    emo_nn.setBlobPath(
        str(blobconverter.from_zoo("emotions-recognition-retail-0003", shaves=shaves))
    )
    emo_nn.input.setBlocking(False)

    emo_in = pipeline.createXLinkIn()
    emo_in.setStreamName("emo_in")
    emo_in.out.link(emo_nn.input)

    emo_nn_xout = pipeline.createXLinkOut()
    emo_nn_xout.setStreamName("emo_nn")
    emo_nn.out.link(emo_nn_xout.input)

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
def facial_info(source, video_path, output, fps, frame_size):
    """
    面部信息识别
    """
    device_info = getDeviceInfo()  # type: dai.DeviceInfo
    with dai.Device(create_pipeline(source), device_info) as device:
        print("Starting pipeline...")
        # device.startPipeline()
        if source:
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_shape = [frame_height, frame_width]
            print("CAP_PROP_FRAME_SHAPE: %s" % frame_shape)
            cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
            print("CAP_PROP_FPS: %d" % cap_fps)

            face_in = device.getInputQueue("face_in")
        else:
            cam_out = device.getOutputQueue("cam_out", 1, True)
        face_nn = device.getOutputQueue("face_nn")
        head_pose_in = device.getInputQueue("head_pose_in")
        head_pose_nn = device.getOutputQueue("head_pose_nn")
        age_in = device.getInputQueue("age_in")
        age_nn = device.getOutputQueue("age_nn")
        emo_in = device.getInputQueue("emo_in")
        emo_nn = device.getOutputQueue("emo_nn")

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output), fourcc, fps, frame_size)

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

        fps_handler = FPSHandler()

        while should_run():
            read_correctly, frame = get_frame()
            if not read_correctly:
                break
            frame_debug = frame.copy()
            if source:
                run_nn(face_in, frame, 300, 300)
            face_nn_data = face_nn.get()
            fps_handler.tick("All")
            if face_nn_data is not None:
                bboxes = face_nn_data.detections

                for bbox in bboxes:
                    face_coord = frameNorm(
                        frame_debug, [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
                    )
                    face_frame = frame[
                        face_coord[1] : face_coord[3],
                        face_coord[0] : face_coord[2],
                    ]
                    cv2.rectangle(
                        frame_debug,
                        (face_coord[0], face_coord[1]),
                        (face_coord[2], face_coord[3]),
                        (0, 0, 0),
                    )

                    run_nn(head_pose_in, face_frame, 60, 60)
                    roll_degree = toTensorResult(head_pose_nn.get()).get("angle_r_fc")[
                        0
                    ][0]
                    center = (
                        (face_coord[2] + face_coord[0]) / 2,
                        (face_coord[3] + face_coord[1]) / 2,

                    )
                    size = (
                        (face_coord[2] - face_coord[0]),
                        (face_coord[3] - face_coord[1]),

                    )
                    face_frame_corr = rotate_frame(
                        frame,
                        center,size,
                        roll_degree,
                    )
                    cv2.imshow("face_frame_corr", face_frame_corr)

                    run_nn(age_in, face_frame_corr, 62, 62)
                    age_gender = toTensorResult(age_nn.get())
                    age = age_gender.get("age_conv3").squeeze() * 100
                    # 0 - female, 1 - male
                    gender = "Male" if age_gender.get("prob").argmax() else "Female"
                    drawText(
                        frame_debug,
                        f"Age: {age:0.0f}",
                        (face_coord[0]+10, face_coord[1] + 30),
                        color="greenyellow",
                    )
                    drawText(
                        frame_debug,
                        f"Gender: {gender}",
                        (face_coord[0]+10, face_coord[1] + 50),
                        "greenyellow",
                    )

                    run_nn(emo_in, face_frame_corr, 64, 64)
                    # 0 - 'neutral', 1 - 'happy', 2 - 'sad', 3 - 'surprise', 4 - 'anger'
                    emo = ["neutral", "happy", "sad", "surprise", "anger"]
                    emo = emo[toTensorResult(emo_nn.get()).get("prob_emotion").argmax()]
                    drawText(
                        frame_debug,
                        f"emo: {emo}",
                        (face_coord[0]+10, face_coord[1] + 70),
                        "greenyellow",
                    )

            if output:
                writer.write(cv2.resize(frame_debug, frame_size))

            cv2.imshow("debug", frame_debug)
            key = cv2.waitKey(1)
            if key in [ord("q"), 27]:
                break
            elif key == ord("s"):
                cv2.imwrite(
                    "saved_%s.jpg" % time.strftime("%Y%m%d_%H%M%S", time.localtime()),
                    frame_debug,
                )
        if source:
            cap.release()
        if output:
            writer.release()
        fps_handler.printStatus()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    with logger.catch():
        facial_info()
