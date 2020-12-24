from pathlib import Path

from tools import *


class DepthAI:
    def __init__(self, file=None, camera=False):
        print("Loading pipeline...")
        self.file = file
        self.camera = camera
        self.cam_size()
        self.create_pipeline()
        self.start_pipeline()
        self.fontScale = 0.5 if self.camera else 3
        self.lineType = 1 if self.camera else 5

    def create_pipeline(self):
        print("Creating pipeline...")
        self.pipeline = depthai.Pipeline()

        if self.camera:
            # ColorCamera
            print("Creating Color Camera...")
            cam = self.pipeline.createColorCamera()
            cam.setPreviewSize(self.first_size[1], self.first_size[0])

            cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setInterleaved(False)
            cam.setCamId(0)
            cam_xout = self.pipeline.createXLinkOut()
            cam_xout.setStreamName("cam_out")
            cam.preview.link(cam_xout.input)

        self.create_nns()

        print("Pipeline created.")

    def create_nns(self):
        pass

    def create_nn(self, model_path, model_name):
        """

        :param model_path: 模型名称
        :param model_name: 模型简称
        :return:
        """
        # NeuralNetwork
        print(f"Creating {model_path} Neural Network...")
        model_in = self.pipeline.createXLinkIn()
        model_in.setStreamName(f"{model_name}_in")
        model_nn = self.pipeline.createNeuralNetwork()
        model_nn.setBlobPath(str(Path(f"{model_path}").resolve().absolute()))
        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_in.out.link(model_nn.input)
        model_nn.out.link(model_nn_xout.input)

    def start_pipeline(self):
        try:
            self.device = depthai.Device()
            print("Starting pipeline...")
            self.device.startPipeline(self.pipeline)
        except TypeError:
            self.device = depthai.Device(self.pipeline, usb2Mode=True)
            print("Starting pipeline...")
            self.device.startPipeline()
        except RuntimeError:
            return

        self.start_nns()

        if self.camera:
            self.cam_out = self.device.getOutputQueue("cam_out", 1, True)

    def start_nns(self):
        pass

    # def full_frame_cords(self, cords):
    #     original_cords = self.face_coords[0]
    #     return [
    #         original_cords[0 if i % 2 == 0 else 1] + val for i, val in enumerate(cords)
    #     ]
    #
    # def full_frame_bbox(self, bbox):
    #     relative_cords = self.full_frame_cords(bbox)
    #     height, width = self.frame.shape[:2]
    #     y_min = max(0, relative_cords[1])
    #     y_max = min(height, relative_cords[3])
    #     x_min = max(0, relative_cords[0])
    #     x_max = min(width, relative_cords[2])
    #     result_frame = self.frame[y_min:y_max, x_min:x_max]
    #     return result_frame, relative_cords

    def draw_bbox(self, bbox, color):
        cv2.rectangle(
            self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2
        )

    def draw_dot(self, dot, color):
        cv2.circle(
            self.debug_frame,
            (
                dot[0],
                dot[1],
            ),
            1,
            color,
            -1,
        )

    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()

        # try:
        #     self.parse_fun()
        # except Exception as e:
        #     print(f'Error:{e}')
        self.parse_fun()

        if debug:
            aspect_ratio = self.frame.shape[1] / self.frame.shape[0]
            cv2.imshow(
                "Camera_view",
                cv2.resize(self.debug_frame, (int(900), int(900 / aspect_ratio))),
            )
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
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
            self.frame = (
                np.array(self.cam_out.get().getData())
                    .reshape((3, self.first_size[0], self.first_size[1]))
                    .transpose(1, 2, 0)
                    .astype(np.uint8)
            )
            try:
                self.parse()
            except StopIteration:
                break

    def cam_size(self):
        self.first_size = (0, 0)

    def run(self):
        if self.file is not None:
            self.run_video()
        else:
            self.run_camera()
        del self.device
