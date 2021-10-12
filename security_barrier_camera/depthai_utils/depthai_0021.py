from pathlib import Path

from imutils.video import FPS

from .utils import *


class DepthAI:
    def __init__(
        self,
        file=None,
        camera=False,
    ):
        print("Loading pipeline...")
        self.file = file
        self.camera = camera
        self.fps_cam = FPS()
        self.fps_nn = FPS()
        self.create_pipeline()
        self.start_pipeline()
        self.fontScale = 1 if self.camera else 2
        self.lineType = 0 if self.camera else 3

    def create_pipeline(self):
        print("Creating pipeline...")
        self.pipeline = depthai.Pipeline()

        if self.camera:
            # ColorCamera
            print("Creating Color Camera...")
            self.cam = self.pipeline.createColorCamera()
            self.cam.setPreviewSize(self._cam_size[1], self._cam_size[0])
            self.cam.setResolution(
                depthai.ColorCameraProperties.SensorResolution.THE_1080_P
            )
            self.cam.setInterleaved(False)
            self.cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
            self.cam.setColorOrder(depthai.ColorCameraProperties.ColorOrder.BGR)

            self.cam_xout = self.pipeline.createXLinkOut()
            self.cam_xout.setStreamName("preview")
            self.cam.preview.link(self.cam_xout.input)

        self.create_nns()

        print("Pipeline created.")

    def create_nns(self):
        pass

    def create_nn(self, model_path: str, model_name: str, first: bool = False):
        """
        :param model_path: model path
        :param model_name: model abbreviation
        :param first: Is it the first model
        :return:
        """
        # NeuralNetwork
        print(f"Creating {Path(model_path).stem} Neural Network...")
        model_nn = self.pipeline.createNeuralNetwork()
        model_nn.setBlobPath(str(Path(model_path).resolve().absolute()))
        model_nn.input.setBlocking(False)
        if first and self.camera:
            print("linked cam.preview to model_nn.input")
            self.cam.preview.link(model_nn.input)
        else:
            model_in = self.pipeline.createXLinkIn()
            model_in.setStreamName(f"{model_name}_in")
            model_in.out.link(model_nn.input)

        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)

    def create_mobilenet_nn(
        self,
        model_path: str,
        model_name: str,
        conf: float = 0.5,
        first: bool = False,
    ):
        """

        :param model_path: 模型名称
        :param model_name: 模型简称
        :param conf: 置信度阈值
        :param first: 是否为首个模型
        :return:
        """
        # NeuralNetwork
        print(f"Creating {Path(model_path).stem} Neural Network...")
        model_nn = self.pipeline.createMobileNetDetectionNetwork()
        model_nn.setBlobPath(str(Path(model_path).resolve().absolute()))
        model_nn.setConfidenceThreshold(conf)
        model_nn.input.setBlocking(False)

        if first and self.camera:
            self.cam.preview.link(model_nn.input)
            model_nn.passthrough.link(self.cam_xout.input)
        else:
            model_in = self.pipeline.createXLinkIn()
            model_in.setStreamName(f"{model_name}_in")
            model_in.out.link(model_nn.input)

        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)

    def start_pipeline(self):
        self.device = depthai.Device(self.pipeline)
        print("Starting pipeline...")

        self.start_nns()

        if self.camera:
            self.preview = self.device.getOutputQueue(
                name="preview", maxSize=4, blocking=False
            )

    def start_nns(self):
        pass

    def put_text(self, text, dot, color=(0, 0, 255), font_scale=None, line_type=None):
        font_scale = font_scale if font_scale else self.fontScale
        line_type = line_type if line_type else self.lineType
        dot = tuple(dot[:2])
        cv2.putText(
            img=self.debug_frame,
            text=text,
            org=dot,
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=font_scale,
            color=color,
            lineType=line_type,
        )
    def draw_bbox(self, bbox, color):
        cv2.rectangle(
            img=self.debug_frame,
            pt1=(bbox[0], bbox[1]),
            pt2=(bbox[2], bbox[3]),
            color=color,
            thickness=2,
        )
    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()

        self.parse_fun()

        if debug:
            cv2.imshow(
                "Camera_view",
                self.debug_frame,
            )
            self.fps_cam.update()
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                self.fps_cam.stop()
                self.fps_nn.stop()
                print(
                    f"FPS_CAMERA: {self.fps_cam.fps():.2f} , FPS_NN: {self.fps_nn.fps():.2f}"
                )
                raise StopIteration()

    def parse_fun(self):
        pass

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
            in_rgb = self.preview.tryGet()
            if in_rgb is not None:
                self.frame = in_rgb.getCvFrame()
                try:
                    self.parse()
                except StopIteration:
                    break

    @property
    def cam_size(self):
        return self._cam_size

    @cam_size.setter
    def cam_size(self, v):
        self._cam_size = v

    def run(self):
        self.fps_cam.start()
        self.fps_nn.start()
        if self.file is not None:
            self.run_video()
        else:
            self.run_camera()
        del self.device
