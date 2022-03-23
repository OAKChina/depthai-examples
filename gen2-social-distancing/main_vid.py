import argparse
import uuid
from pathlib import Path
from time import monotonic, time

import cv2
import depthai as dai
import numpy as np
from depthai_sdk import FPSHandler
from gen2Alerting import Bird
from gen2Distance import DistanceGuardianDebug

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

debug = not args.no_debug

if args.camera and args.video:
    raise ValueError("Incorrect command line parameters! \"-cam\" cannot be used with \"-vid\"!")
elif args.camera is False and args.video is None:
    raise ValueError("Missing inference source! Either use \"-cam\" to run on DepthAI camera or \"-vid <path>\" to run on video file")

def timer(function):
    """
    Decorator function timer
    :param function:The function you want to time
    :return:
    """

    def wrapper(*args, **kwargs):
        time_start = time()
        res = function(*args, **kwargs)
        cost_time = time() - time_start
        print("【 %s 】operation hours：【 %s 】second" % (function.__name__, cost_time))
        return res

    return wrapper

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

class DepthAI:
    distance_guardian_class = DistanceGuardianDebug
    distance_bird_class = Bird

    def __init__(self, file=None, camera=False):
        self.file = file
        self.camera = camera
        self.create_pipeline()
        self.start_pipeline()
        self.fps_handler = FPSHandler()
        self.distance_guardian = self.distance_guardian_class()
        self.distance_bird = self.distance_bird_class()
    
    def create_pipeline(self):
        self.pipeline = dai.Pipeline()
        if self.camera:
            cam = self.pipeline.createColorCamera()
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setPreviewSize(544, 320)
            cam.setInterleaved(False)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            self.cam_xout = self.pipeline.createXLinkOut()
            self.cam_xout.setStreamName("cam_out")
            # cam.preview.link(self.cam_xout.input)
            self.create_models("models/person-detection-retail-0013/person-detection-retail-0013_openvino_2021.3_13shave.blob","model", cam)
            self.model.passthrough.link(self.cam_xout.input)
        else:
            self.create_models("models/person-detection-retail-0013/person-detection-retail-0013_openvino_2021.3_13shave.blob","model")
        
        self.StereoDepthXLink()
        self.stereo.depth.link(self.model.inputDepth)
        self.model.passthroughDepth.link(self.stereo_out.input)

    def create_models(self, model_path, name, cam=None):
        print(f"正在创建{model_path}神经网络...")
        self.model = self.pipeline.createMobileNetSpatialDetectionNetwork()
        self.model.setBlobPath(str(Path(model_path).resolve().absolute()))
        self.model.setConfidenceThreshold(0.5)
        self.model.input.setBlocking(False)
        self.model.setBoundingBoxScaleFactor(0.5)
        self.model.setDepthLowerThreshold(100)
        self.model.setDepthUpperThreshold(5000)
        if self.camera:
            cam.preview.link(self.model.input)
        else:
            self.model_in = self.pipeline.createXLinkIn()
            self.model_in.setStreamName(f"{name}_in")
            self.model_in.out.link(self.model.input)
        self.model_out = self.pipeline.createXLinkOut()
        self.model_out.setStreamName(f"{name}_nn")
        self.model.out.link(self.model_out.input)
    
    def StereoDepthXLink(self):
        print(f"正在创建深度节点...")
        self.stereo = self.pipeline.createStereoDepth()
        self.stereo.setConfidenceThreshold(230)
        self.stereo_out = self.pipeline.createXLinkOut()
        self.stereo_out.setStreamName("depth")

        if self.camera:
            mono_left = self.pipeline.createMonoCamera()
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            mono_right = self.pipeline.createMonoCamera()
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            mono_left.out.link(self.stereo.left)
            mono_right.out.link(self.stereo.right)
        else:
            self.mono_left = self.pipeline.createXLinkIn()
            self.mono_left.setStreamName("vid_left")
            self.mono_right = self.pipeline.createXLinkIn()
            self.mono_right.setStreamName("vid_right")

            self.mono_left.out.link(self.stereo.left)
            self.mono_right.out.link(self.stereo.right)
        
            self.stereo.setEmptyCalibration()
            self.stereo.setInputResolution(1280, 720)


    def start_pipeline(self):
        print(f"启动设备管道...")
        self.device = dai.Device(self.pipeline)
        if self.camera:
            self.camRgb = self.device.getOutputQueue(self.cam_xout.getStreamName(), 4, False)
        else:
            self.person_in = self.device.getInputQueue(self.model_in.getStreamName())
            self.vid_left = self.device.getInputQueue(self.mono_left.getStreamName())
            self.vid_right = self.device.getInputQueue(self.mono_right.getStreamName())
        self.person_nn = self.device.getOutputQueue(self.model_out.getStreamName(), 4, False)
        self.depth = self.device.getOutputQueue(self.stereo_out.getStreamName(), 4, False)

    def draw_bbox(self, bbox, color, detection):
        cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(self.debug_frame, "x: {}m".format(round(detection.spatialCoordinates.x) / 1000), (bbox[0], bbox[1] + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        cv2.putText(self.debug_frame, "y: {}m".format(round(detection.spatialCoordinates.y) / 1000), (bbox[0], bbox[1] + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        cv2.putText(self.debug_frame, "z: {}m".format(round(detection.spatialCoordinates.z) / 1000), (bbox[0], bbox[1] + 70), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        cv2.putText(self.debug_frame, "conf: {}".format(round(detection.confidence * 100)), (bbox[0], bbox[1] + 90), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

    def run_model(self):
        width = self.frame.shape[1]
        height = self.frame.shape[0]
        if self.camera:
            nn_data = self.person_nn.get()
        else:
            # nn_data = run_nn(self.person_in, self.person_nn, self.frame)
            img = dai.ImgFrame()
            img.setData(to_planar(self.frame, (544, 320)))
            img.setType(dai.RawImgFrame.Type.BGRF16F16F16p)
            img.setTimestamp(monotonic())
            img.setWidth(544)
            img.setHeight(320)
            self.person_in.send(img)

            img_left = dai.ImgFrame()
            img_left.setData(to_planar(self.frame, (544, 320)))
            img_left.setType(dai.RawImgFrame.Type.GRAYF16)
            img_left.setTimestamp(monotonic())
            img_left.setWidth(544)
            img_left.setHeight(320)
            self.vid_left.send(img_left)

            img_right = dai.ImgFrame()
            img_right.setData(to_planar(self.frame, (544, 320)))
            img_right.setType(dai.RawImgFrame.Type.GRAYF16)
            img_right.setTimestamp(monotonic())
            img_right.setWidth(544)
            img_right.setHeight(320)
            self.vid_right.send(img_right)

            nn_data = self.person_nn.tryGet()
        print(nn_data)
        if nn_data is not None:
            boxse = []
            detections = nn_data.detections
            for detection in detections:
                x_min = int(detection.xmin * width)
                y_min = int(detection.ymin * height)
                x_max = int(detection.xmax * width)
                y_max = int(detection.ymax * height)
                bbox = (x_min, y_min, x_max, y_max)
                self.draw_bbox(bbox, (10, 245, 10), detection)
                boxse.append({
                    "id": uuid.uuid4(),
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "depth_x" : detection.spatialCoordinates.x / 1000,
                    "depth_y": detection.spatialCoordinates.y / 1000,
                    "depth_z": detection.spatialCoordinates.z / 1000
                })
                print(boxse)
            results = self.distance_guardian.parse_frame(self.debug_frame, boxse)
            self.bird_frame = self.distance_bird.parse_frame(self.debug_frame, results, boxse)

    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()
        self.run_model()
        self.fps_handler.tick("NN")
        if debug:
            numpy_horizontal = np.hstack((self.debug_frame, self.bird_frame))
            cv2.imshow("Frame", numpy_horizontal)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                raise StopIteration()

    def run_video(self):
        cap = cv2.VideoCapture(str(Path(self.file).resolve().absolute()))
        while cap.isOpened():
            read_correctly, self.frame = cap.read()
            if not read_correctly:
                break
            try:
                self.parse()
            except StopIteration:
                break
        cap.release()

    def run_camera(self):
        while True:
            rgb_in = self.camRgb.get()
            self.frame = rgb_in.getCvFrame()
            try:
                self.parse()
            except StopIteration:
                break
    
    def run(self):
        if self.file is not None:
            self.run_video()
        else:
            self.run_camera()
        self.fps_handler.printStatus()
    
    def __del__(self):
        del self.pipeline
        del self.device


if __name__ == "__main__":
    if args.video:
        DepthAI(file=args.video).run()
    else:
        DepthAI(camera=args.camera).run()