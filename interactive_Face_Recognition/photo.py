import depthai
import cv2
import numpy as np

pipeline = depthai.Pipeline()
device = depthai.Device()
cam = pipeline.createColorCamera()
cam.setPreviewSize(300,300)
cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setCamId(0)
cam_xout = pipeline.createXLinkOut()
cam_xout.setStreamName("cam_out")
cam.preview.link(cam_xout.input)
device.startPipeline(pipeline)
cam_out = device.getOutputQueue("cam_out", 1, True)
while(1):
    frame = np.array(cam_out.get().getData()).reshape((3, 300, 300)).transpose(1, 2, 0).astype(np.uint8)
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("camera.jpg", frame)
        break
cv2.destroyAllWindows()