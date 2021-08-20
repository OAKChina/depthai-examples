import argparse
import time
from pathlib import Path
from time import monotonic

import cv2
import depthai as dai
import numpy as np
from imutils.video import FPS
from rich import print

from demo_utils import multiclass_nms, demo_postprocess
from visualize import vis

VOC_CLASSES = ("helmet", "head")


parser = argparse.ArgumentParser()
parser.add_argument(
    "-nd", "--no-debug", action="store_true", help="Prevent debug output"
)
parser.add_argument(
    "-cam",
    "--camera",
    action="store_true",
    default=True,
    help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)",
)

parser.add_argument(
    "-vid",
    "--video",
    type=Path,
    help="The path of the video file",
)

parser.add_argument(
    "-o",
    "--output_dir",
    type=Path,
    default=None,
    help="The directory of the output video.",
)
parser.add_argument(
    "-is",
    "--input_shape",
    default=(720, 1280),
    nargs="+",
    type=int,
    help="The shape (h, w) of the input stream used only for the camera stream. "
    "\r\nDefault value: 720, 1280",
)

parser.add_argument(
    "-os",
    "--output_shape",
    default=None,
    nargs="+",
    type=int,
    help="The shape (h,w) of the video to be saved. "
    "\r\nThe default is the shape of the input stream",
)
parser.add_argument(
    "-f",
    "--fps",
    type=int,
    default=0,
    help="Set the FPS (only for the output video of the camera stream). "
    "\r\nDefault value: 30",
)
args = parser.parse_args()

if args.video:
    args.camera = False

if not args.camera and not args.video:
    raise RuntimeError(
        "No source selected. "
        'Please use either "-cam" to use RGB camera as a source , '
        '"-vid <path>" to run on video, '
    )

debug = not args.no_debug


def to_tensor_result(packet):
    data = {}
    for tensor in packet.getRaw().tensors:
        if tensor.dataType == dai.TensorInfo.DataType.INT:
            data[tensor.name] = np.array(packet.getLayerInt32(tensor.name)).reshape(
                tensor.dims  # [::-1]
            )
        elif tensor.dataType == dai.TensorInfo.DataType.FP16:
            data[tensor.name] = np.array(packet.getLayerFp16(tensor.name)).reshape(
                tensor.dims  # [::-1]
            )
        elif tensor.dataType == dai.TensorInfo.DataType.I8:
            data[tensor.name] = np.array(packet.getLayerUInt8(tensor.name)).reshape(
                tensor.dims  # [::-1]
            )
        else:
            print("Unsupported tensor layer type: {}".format(tensor.dataType))
    return data


def to_planar(arr: np.ndarray, input_size: tuple = None) -> np.ndarray:
    if input_size is None or tuple(arr.shape[:2]) == input_size:
        return arr.transpose((2, 0, 1))

    input_size = np.array(input_size)
    if len(arr.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(arr)
    r = min(input_size / img.shape[:2])
    resize_ = (np.array(img.shape[:2]) * r).astype(int)
    resized_img = cv2.resize(
        img,
        resize_[::-1],
        interpolation=cv2.INTER_LINEAR,
    )
    padding = (input_size - resize_) // 2
    padded_img[
        padding[0] : padding[0] + int(img.shape[0] * r),
        padding[1] : padding[1] + int(img.shape[1] * r),
    ] = resized_img
    image = padded_img.transpose(2, 0, 1)
    return image


def frame_norm(frame, bbox):
    return (
        np.clip(np.array(bbox), 0, 1)
        * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]
    ).astype(int)


def print_results(result, data=False):
    for i in result:
        print(i, result[i].shape)
        if data:
            print(result[i])


def run_nn(x_in, frame, w, h):
    nn_data = dai.ImgFrame()
    nn_data.setData(to_planar(frame, (w, h)))
    nn_data.setType(dai.RawImgFrame.Type.BGR888p)
    nn_data.setTimestamp(monotonic())
    nn_data.setWidth(w)
    nn_data.setHeight(h)
    x_in.send(nn_data)


def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

    if args.camera:
        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(*args.input_shape[::-1])
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setFps(30)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Pedestrian Detection Neural Network...")
    yoloDet = pipeline.createNeuralNetwork()
    if args.camera:
        yoloDet.setBlobPath(
            Path("models/helmet_detection_yolox1_openvino_2021.4_6shave.blob")
            .resolve()
            .absolute()
            .as_posix()
        )
    else:
        yoloDet.setBlobPath(
            Path("models/helmet_detection_yolox1_openvino_2021.4_8shave.blob")
            .resolve()
            .absolute()
            .as_posix()
        )
    yolox_det_nn_xout = pipeline.createXLinkOut()
    yolox_det_nn_xout.setStreamName("yolox_det_nn")
    yoloDet.out.link(yolox_det_nn_xout.input)

    if args.camera:
        manip = pipeline.createImageManip()
        manip.initialConfig.setResizeThumbnail(320, 320, 114, 114, 114)
        manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
        cam.preview.link(manip.inputImage)
        manip.out.link(yoloDet.input)
        # cam.preview.link(yoloDet.input)
    else:
        yolox_det_in = pipeline.createXLinkIn()
        yolox_det_in.setStreamName("yolox_det_in")
        yolox_det_in.out.link(yoloDet.input)

    return pipeline


with dai.Device(create_pipeline()) as device:
    print("Starting pipeline...")
    # device.startPipeline()
    if args.camera:
        cam_out = device.getOutputQueue("cam_out", 1, True)
    else:
        yolox_det_in = device.getInputQueue("yolox_det_in")
    yolox_det_nn = device.getOutputQueue("yolox_det_nn")

    frame = None
    cap_fps = 0

    if args.video:
        cap = cv2.VideoCapture(args.video.resolve().absolute().as_posix())

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("CAP_PROP_FRAME_COUNT: %d" % frame_count)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_shape = [frame_height, frame_width]
        print("CAP_PROP_FRAME_SHAPE: %s" % frame_shape)
        cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("CAP_PROP_FPS: %d" % cap_fps)

    if args.output_dir:
        output_shape = tuple(args.output_shape) if args.output_shape else frame_shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
        output_path = (
            args.output_dir
            / "saved_%s.mp4"
            % time.strftime("%Y%m%d_%H%M%S", time.localtime())
        )
        if cap_fps > 1:
            videoWriter = cv2.VideoWriter(
                output_path.as_posix(), fourcc, cap_fps, output_shape[::-1], 1
            )
        else:
            videoWriter = cv2.VideoWriter(
                output_path.as_posix(),
                fourcc,
                args.fps if args.fps else 30,
                output_shape[::-1],
                1,
            )

    def should_run():
        if args.video:
            return cap.isOpened()
        else:
            return True

    def get_frame():
        if args.video:
            if frame_count <= 1 and frame is not None:
                return True, frame
            return cap.read()
        else:
            return True, cam_out.get().getCvFrame()

    fps = FPS()
    fps.start()
    while should_run():
        read_correctly, frame = get_frame()
        if not read_correctly:
            break

        frame_debug = frame.copy()
        if not args.camera:
            run_nn(yolox_det_in, frame, 320, 320)
            yolox_det_data = yolox_det_nn.get()
        else:
            yolox_det_data = yolox_det_nn.tryGet()
        if yolox_det_data is not None:
            fps.update()
            res = to_tensor_result(yolox_det_data).get("output")
            predictions = demo_postprocess(res, (320, 320), p6=False)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4, None] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

            input_shape = np.array([320, 320])
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
                    conf=0.6,
                    class_names=VOC_CLASSES,
                )
        if args.output_dir:
            if args.output_shape:
                videoWriter.write(
                    to_planar(frame_debug, output_shape)
                    .transpose((1, 2, 0))
                    .astype(np.uint8)
                )
            else:
                videoWriter.write(frame_debug)

        r = (args.input_shape / np.array(frame.shape[:2])).max()
        if debug:
            cv2.imshow("debug", cv2.resize(frame_debug, (0, 0), fx=r, fy=r))
        else:
            cv2.imshow("preview", cv2.resize(frame, (0, 0), fx=r, fy=r))
        key = cv2.waitKey(1)
        if key in [ord("q"), 27]:
            break
        elif key == ord("s"):
            cv2.imwrite(
                "saved_%s.jpg" % time.strftime("%Y%m%d_%H%M%S", time.localtime()),
                frame_debug,
            )
fps.stop()
if args.video:
    cap.release()
if args.output_dir:
    videoWriter.release()
print("NN FPS: %d" % fps.fps())
cv2.destroyAllWindows()
