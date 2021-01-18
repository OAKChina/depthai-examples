import argparse
from datetime import datetime, timedelta
from pathlib import Path
import cv2
import numpy as np
import depthai
import time
# import torch
# import torch.nn.functional as m_func
from scipy.special import log_softmax

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
    装饰器函数timer
    :param function:想要计时的函数
    :return:
    """

    def wrapper(*args, **kwargs):
        time_start = time.time()
        res = function(*args, **kwargs)
        cost_time = time.time() - time_start
        print("【 %s 】运行时间：【 %s 】秒" % (function.__name__, cost_time))
        return res

    return wrapper

def wait_for_results(queue):
    start = datetime.now()
    while not queue.has():
        if datetime.now() - start > timedelta(seconds=1):
            return False
    return True


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


def to_nn_result(nn_data):
    return np.array(nn_data.getFirstLayerFp16())


def to_tensor_result(packet):
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }


def to_bbox_result(nn_data):
    try:
        arr = to_nn_result(nn_data)
        arr = arr[:np.where(arr == -1)[0][0]]
        arr = arr.reshape((arr.size // 7, 7))
        return arr
    except:
        return []

#@timer
def run_nn(x_in, x_out, in_dict):
    nn_data = depthai.NNData()
    for key in in_dict:
        nn_data.setLayer(key, in_dict[key])
    x_in.send(nn_data)
    has_results = wait_for_results(x_out)
    if not has_results:
        raise RuntimeError("No data from nn!")
    return x_out.get()


def frame_norm(frame, *xy_vals):
    height, width = frame.shape[:2]
    result = []
    for i, val in enumerate(xy_vals):
        if i % 2 == 0:
            result.append(max(0, min(width, int(val * width))))
        else:
            result.append(max(0, min(height, int(val * height))))
    return result

class Main:
    def __init__(self,file=None,camera=False):
        print("Loading pipeline...")
        self.file = file
        self.camera = camera
        self.create_pipeline()
        self.start_pipeline()
    
    def create_pipeline(self):
        print("Creating pipeline...")
        self.pipeline = depthai.Pipeline()
        if self.camera:
            print("Creating Color Camera...")
            cam = self.pipeline.createColorCamera()
            cam.setPreviewSize(300,300)
            cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setInterleaved(False)
            cam.setCamId(0)
            cam_xout = self.pipeline.createXLinkOut()
            cam_xout.setStreamName("cam_out")
            cam.preview.link(cam_xout.input)
        
        self.models("models/face-detection-retail-0004.blob","face")
        self.models("models/sbd_mask4.blob","mask")

    def models(self,model_path,name):
        print(f"开始创建{model_path}神经网络")
        model_in = self.pipeline.createXLinkIn()
        model_in.setStreamName(f"{name}_in")
        model_nn = self.pipeline.createNeuralNetwork()
        model_nn.setBlobPath(str(Path(model_path).resolve().absolute()))
        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{name}_nn")
        model_in.out.link(model_nn.input)
        model_nn.out.link(model_nn_xout.input)

    def start_pipeline(self):
        self.device = depthai.Device(self.pipeline,usb2Mode=True)
        print("Starting pipeline...")
        self.device.startPipeline()
        self.face_in = self.device.getInputQueue("face_in")
        self.face_nn = self.device.getOutputQueue("face_nn")
        self.mask_in = self.device.getInputQueue("mask_in")
        self.mask_nn = self.device.getOutputQueue("mask_nn")
        if self.camera:
            self.cam_out = self.device.getOutputQueue("cam_out", 1, True)
    

    def full_frame_cords(self, cords):
        original_cords = self.face_coords[0]
        return [
            original_cords[0 if i % 2 == 0 else 1] + val
            for i, val in enumerate(cords)
        ]

    def full_frame_bbox(self, bbox):
        relative_cords = self.full_frame_cords(bbox)
        height, width = self.frame.shape[:2]
        y_min = max(0, relative_cords[1])
        y_max = min(height, relative_cords[3])
        x_min = max(0, relative_cords[0])
        x_max = min(width, relative_cords[2])
        result_frame = self.frame[y_min:y_max, x_min:x_max]
        return result_frame, relative_cords


    def draw_bbox(self, bbox, color):
        cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    def run_face(self):
        nn_data = run_nn(self.face_in, self.face_nn, {"data": to_planar(self.frame, (300, 300))})
        results = to_bbox_result(nn_data)
        self.face_coords = [
            frame_norm(self.frame, *obj[3:7])
            for obj in results
            if obj[2] > 0.4
        ]
        if len(self.face_coords) == 0:
            return False
        if len(self.face_coords) > 0:
            for face_coord in self.face_coords:
                face_coord[0] -= 15
                face_coord[1] -= 15
                face_coord[2] += 15
                face_coord[3] += 15
            self.face_frame = [self.frame[
                face_coord[1]:face_coord[3],
                face_coord[0]:face_coord[2]
            ] for face_coord in self.face_coords]
        return True
    
    def run_mask(self,frame,count):
        try:
            nn_data = run_nn(self.mask_in,self.mask_nn,{"data":to_planar(frame,(224,224))})
            out = to_tensor_result(nn_data).get('349')
            # match = m_func.log_softmax(torch.from_numpy(out), dim=0).data.numpy()
            match = log_softmax(np.array(out))
            # print(match)
            index = np.argmax(match)
            # print(index)
            ftype = 0 if index > 0.5 else 1
            # print(ftype)
            color = (0,0,255) if ftype else (0,255,0)
            self.draw_bbox(self.face_coords[count],color)
            cv2.putText(self.debug_frame,'{:.2f}'.format(match[0]),(self.face_coords[count][0],self.face_coords[count][1]-10),cv2.FONT_HERSHEY_COMPLEX,1,color)
            cnt_mask,cnt_nomask = 0,0
            if ftype == 0: cnt_mask += 1
            else: cnt_nomask += 1
            proportion = cnt_mask / len(self.face_frame) * 100
            # print(round(proportion,2))
            cv2.putText(self.debug_frame,"masks:" + str(round(proportion,2)) + "%",(10,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(255,0,0))
        except:
            pass


    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()

        face_success = self.run_face()
        if face_success:
            for i in range(len(self.face_frame)):
                self.run_mask(self.face_frame[i],i)
        if debug:
            aspect_ratio = self.frame.shape[1] / self.frame.shape[0]
            cv2.imshow("Camera_view", cv2.resize(self.debug_frame, ( int(900),  int(900 / aspect_ratio))))
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
            self.frame = np.array(self.cam_out.get().getData()).reshape((3, 300, 300)).transpose(1, 2, 0).astype(np.uint8)
            try:
                self.parse()
            except StopIteration:
                break
    
    def run(self):
        if self.file is not None:
            self.run_video()
        else:
            self.run_camera()
        del self.device

if __name__ == '__main__':
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()