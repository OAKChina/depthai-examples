import argparse
import uuid
from pathlib import Path
from time import time

import cv2
import depthai as dai
import numpy as np
from depthai_sdk import FPSHandler

from gen2Alerting import Bird
from gen2Distance import DistanceGuardianDebug

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
args = parser.parse_args()

debug = not args.no_debug

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

class DepthAI:
    distance_guardian_class = DistanceGuardianDebug
    distance_bird_class = Bird

    def __init__(self):
        self.frame_size = (672, 384)
        self.create_pipeline()
        self.start_pipeline()
        self.fps_handler = FPSHandler()
        self.distance_guardian = self.distance_guardian_class()
        self.distance_bird = self.distance_bird_class(self.frame_size)
    
    def create_pipeline(self):
        self.pipeline = dai.Pipeline()
        cam = self.pipeline.createColorCamera()
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setPreviewSize(self.frame_size[0], self.frame_size[1])
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.cam_xout = self.pipeline.createXLinkOut()
        self.cam_xout.setStreamName("cam_out")
        self.create_models("models/person-detection-retail-0013/pedestrian-detection-adas-0002.blob","model", cam)
        self.model.passthrough.link(self.cam_xout.input)
        
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
        cam.preview.link(self.model.input)
        self.model_out = self.pipeline.createXLinkOut()
        self.model_out.setStreamName(f"{name}_nn")
        self.model.out.link(self.model_out.input)
    
    def StereoDepthXLink(self):
        print(f"正在创建深度节点...")
        self.stereo = self.pipeline.createStereoDepth()
        self.stereo.initialConfig.setConfidenceThreshold(230)
        self.stereo_out = self.pipeline.createXLinkOut()
        self.stereo_out.setStreamName("depth")

        mono_left = self.pipeline.createMonoCamera()
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right = self.pipeline.createMonoCamera()
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_left.out.link(self.stereo.left)
        mono_right.out.link(self.stereo.right)

    def start_pipeline(self):
        print(f"启动设备管道...")
        self.device = dai.Device(self.pipeline)
        self.camRgb = self.device.getOutputQueue(self.cam_xout.getStreamName(), 4, False)
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
        nn_data = self.person_nn.get()
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
            results = self.distance_guardian.parse_frame(self.debug_frame, boxse)
            self.bird_frame = self.distance_bird.parse_frame(self.debug_frame, results, boxse)

            depthFrame = self.depth.get().getFrame()
            self.depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            self.depthFrameColor = cv2.equalizeHist(self.depthFrameColor)
            self.depthFrameColor = cv2.applyColorMap(self.depthFrameColor, cv2.COLORMAP_JET)

    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()
        self.run_model()
        self.fps_handler.tick("NN")
        if debug:
            numpy_horizontal = np.hstack((self.debug_frame, self.bird_frame))
            cv2.imshow("Frame", numpy_horizontal)
            cv2.imshow("depth", self.depthFrameColor)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                raise StopIteration()

    def run_camera(self):
        while True:
            rgb_in = self.camRgb.get()
            self.frame = rgb_in.getCvFrame()
            try:
                self.parse()
            except StopIteration:
                break
    
    def run(self):
        self.run_camera()
        self.fps_handler.printStatus()
    
    def __del__(self):
        del self.pipeline
        del self.device


if __name__ == "__main__":
    DepthAI().run()