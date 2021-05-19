#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import time
import argparse
from Focuser import Focuser
from pid import PID

# labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
#             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

nnPathDefault = str((Path(__file__).parent / Path('models/mobilenet-ssd_openvino_2021.2_6shave.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-ff', '--full_frame', action="store_true", help="Perform tracking on full RGB frame", default=False)

args = parser.parse_args()


fullFrameTracking = args.full_frame

# Start defining a pipeline
pipeline = dai.Pipeline()

colorCam = pipeline.createColorCamera()
detectionNetwork = pipeline.createMobileNetDetectionNetwork()
objectTracker = pipeline.createObjectTracker()

xoutRgb = pipeline.createXLinkOut()
trackerOut = pipeline.createXLinkOut()

xoutRgb.setStreamName("preview")
trackerOut.setStreamName("tracklets")

colorCam.setPreviewSize(300, 300)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

detectionNetwork.setBlobPath(args.nnPath)
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.input.setBlocking(False)

# Link plugins CAM . NN . XLINK
colorCam.preview.link(detectionNetwork.input)
objectTracker.passthroughTrackerFrame.link(xoutRgb.input)


objectTracker.setDetectionLabelsToTrack([15])  # track only person
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssigmentPolicy(dai.TrackerIdAssigmentPolicy.SMALLEST_ID)

objectTracker.out.link(trackerOut.input)
if fullFrameTracking:
    colorCam.setPreviewKeepAspectRatio(False)
    colorCam.video.link(objectTracker.inputTrackerFrame)
    objectTracker.inputTrackerFrame.setBlocking(False)
    # do not block the pipeline if it's too slow on full frame
    objectTracker.inputTrackerFrame.setQueueSize(2)
else:
    detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
detectionNetwork.out.link(objectTracker.inputDetections)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    focuser = Focuser(1)
    pan_pid = PID(p=0.045, i=0.03, d=0.031, imax=90)
    tilt_pid = PID(p=0.04, i=0.04, d=0.031, imax=90)

    # Start the pipeline
    device.startPipeline()

    preview = device.getOutputQueue("preview", 4, False)
    tracklets = device.getOutputQueue("tracklets", 4, False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    frame = None
    key = None
    id = None
    pan = 0
    tilt = 0
    old_horizontal_deviation = 0
    old_vertical_deviation = 0
    count = 0
    while(True):
        imgFrame = preview.get()
        track = tracklets.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        color = (255, 0, 0)
        frame = imgFrame.getCvFrame()
        frame_center = (frame.shape[1] / 2, frame.shape[0] / 2)
        trackletsData = track.tracklets

        for t in trackletsData:
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)
            horizontal_deviation = (frame_center[0]) - ((x2 - x1) / 2 + x1)
            vertical_deviation = (frame_center[1]) - ((y2 - y1) / 2 + y1)
            if str(t.id) == id:
                count = count + 1 if old_horizontal_deviation == horizontal_deviation and old_vertical_deviation == vertical_deviation else 0
                if count < 5:
                    if abs(horizontal_deviation) > 20:
                        pan = pan_pid.get_pid(horizontal_deviation, 1)
                        focuser.set(Focuser.OPT_MOTOR_X,focuser.get(Focuser.OPT_MOTOR_X) + int(pan))
                    if abs(vertical_deviation) > 15:
                        tilt = tilt_pid.get_pid(vertical_deviation, 1)
                        focuser.set(Focuser.OPT_MOTOR_Y,focuser.get(Focuser.OPT_MOTOR_Y) + int(tilt))
                old_horizontal_deviation = horizontal_deviation
                old_vertical_deviation = vertical_deviation
                print("偏离中心:{0}, 图像中心坐标:{1}, 检测框中心坐标:{2}".format((horizontal_deviation, vertical_deviation), frame_center, (((x2 - x1) / 2 + x1), ((y2 - y1) / 2 + y1))))
            elif key == ord('q'):
                break

            cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        cv2.imshow("tracker", frame)
        key = cv2.waitKey(1) # 键盘控制选择跟踪目标
        if key != -1:
            id = chr(key)
        if key == ord('q'):
            break