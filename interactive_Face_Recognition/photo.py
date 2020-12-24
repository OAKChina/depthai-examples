import depthai
import cv2
import numpy as np
import os

def mkdir(name):
    path = './images/' + name
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")


def photo(name):
    pipeline = depthai.Pipeline()
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(300,300)
    cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setCamId(0)
    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("cam_out")
    cam.preview.link(cam_xout.input)
    device = depthai.Device()
    device.startPipeline(pipeline)
    cam_out = device.getOutputQueue("cam_out", 1, True)
    mkdir(name)
    count = 0
    while(True):
        frame = np.array(cam_out.get().getData()).reshape((3, 300, 300)).transpose(1, 2, 0).astype(np.uint8)
        cv2.imshow("capture", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('s'):
            cv2.imwrite(f"./images/{name}/{name}_{count}.jpg", frame)
            count += 1
    cv2.destroyAllWindows()