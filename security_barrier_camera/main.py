from pathlib import Path

from tools import *


class Main:
    def __init__(self, file=None, camera=False):

        print("Loading pipeline...")
        self.file = file
        self.camera = camera
        self.create_pipeline()
        self.start_pipeline()
        self.fontScale = 0.5 if self.camera else 3
        self.lineType = 1 if self.camera else 3

    def create_pipeline(self):
        print("Creating pipeline...")
        self.pipeline = depthai.Pipeline()
        if self.camera:
            # ColorCamera
            print("Creating Color Camera...")
            cam = self.pipeline.createColorCamera()
            cam.setPreviewSize(300, 300)
            cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_4_K)
            cam.setInterleaved(False)
            cam.setCamId(0)
            cam_xout = self.pipeline.createXLinkOut()
            cam_xout.setStreamName("cam_out")
            cam.preview.link(cam_xout.input)

        self.create_nn(
            "models/vehicle-license-plate-detection-barrier-0106.blob", "vlpd"
        )
        self.create_nn("models/vehicle-attributes-recognition-barrier-0039.blob", "var")
        self.create_nn("models/license-plate-recognition-barrier-0007.blob", "lpr")

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
        self.device = depthai.Device(self.pipeline, usb2Mode=True)
        print("Starting pipeline...")
        self.device.startPipeline()

        if self.camera:
            self.cam_out = self.device.getOutputQueue("cam_out", 1, True)

        self.vlpd_in = self.device.getInputQueue("vlpd_in")
        self.vlpd_nn = self.device.getOutputQueue("vlpd_nn")
        self.var_in = self.device.getInputQueue("var_in")
        self.var_nn = self.device.getOutputQueue("var_nn")
        self.lpr_in = self.device.getInputQueue("lpr_in")
        self.lpr_nn = self.device.getOutputQueue("lpr_nn")

    def full_frame_cords(self, cords):
        original_cords = self.vehicle_coords[0]
        return [
            original_cords[0 if i % 2 == 0 else 1] + val for i, val in enumerate(cords)
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
        cv2.rectangle(
            self.debug_frame,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color,
            self.lineType + 1,
        )

    def run_vlpd(self):
        nn_data = run_nn(
            self.vlpd_in, self.vlpd_nn, {"data": to_planar(self.frame, (300, 300))}
        )
        results = to_bbox_result(nn_data)

        self.vehicle_coords = [
            frame_norm(self.frame, *obj[3:7])
            for obj in results
            if obj[2] > 0.6 and obj[1] == 1
        ]

        self.lpr_coords = [
            frame_norm(self.frame, *obj[3:7])
            for obj in results
            if obj[2] > 0.6 and obj[1] == 2
        ]

        flag_vehicle = False if len(self.vehicle_coords) == 0 else True
        flag_card = False if len(self.lpr_coords) == 0 else True
        if flag_vehicle:
            self.vehicle_()
        if flag_card:
            self.lpr_()
            # pass
        return flag_vehicle, flag_card

    def vehicle_(self):
        self.vehicle_frame = [
            self.frame[
                self.vehicle_coords[i][1] : self.vehicle_coords[i][3],
                self.vehicle_coords[i][0] : self.vehicle_coords[i][2],
            ]
            for i in range(len(self.vehicle_coords))
        ]

        if debug:
            for bbox in self.vehicle_coords:
                self.draw_bbox(bbox, (10, 245, 10))

    def lpr_(self):
        self.lpr_frame = [
            self.frame[
                int(
                    (self.lpr_coords[i][3] + self.lpr_coords[i][1]) / 2
                    - (self.lpr_coords[i][3] - self.lpr_coords[i][1]) / 2 * 1.5
                ) : int(
                    (self.lpr_coords[i][3] + self.lpr_coords[i][1]) / 2
                    + (self.lpr_coords[i][3] - self.lpr_coords[i][1]) / 2 * 1.5
                ),
                int(
                    (self.lpr_coords[i][2] + self.lpr_coords[i][0]) / 2
                    - (self.lpr_coords[i][2] - self.lpr_coords[i][0]) / 2 * 1.5
                ) : int(
                    (self.lpr_coords[i][2] + self.lpr_coords[i][0]) / 2
                    + (self.lpr_coords[i][2] - self.lpr_coords[i][0]) / 2 * 1.5
                ),
            ]
            for i in range(len(self.lpr_coords))
        ]

        if debug:
            for bbox in self.lpr_coords:

                self.draw_bbox(bbox, (10, 245, 10))

    def run_var(self):

        colors = ["white", "gray", "yellow", "red", "green", "blue", "black"]
        types = ["car", "bus", "truck", "van"]
        for i in range(len(self.vehicle_frame)):
            nn_data = run_nn(
                self.var_in,
                self.var_nn,
                {"data": to_planar(self.vehicle_frame[i], (72, 72))},
            )
            results = to_tensor_result(nn_data)
            for key in results.keys():
                results[key] = results[key][: len(results[key]) // 2]

            color_ = colors[results["color"].argmax()]
            type_ = types[results["type"].argmax()]

            cv2.putText(
                self.debug_frame,
                color_,
                (self.vehicle_coords[i][0], self.vehicle_coords[i][1]),
                cv2.FONT_HERSHEY_COMPLEX,
                self.fontScale,
                (0, 0, 255),
                self.lineType,
            )
            cv2.putText(
                self.debug_frame,
                type_,
                (
                    self.vehicle_coords[i][0],
                    self.vehicle_coords[i][1] + self.lineType * 15,
                ),
                cv2.FONT_HERSHEY_COMPLEX,
                self.fontScale,
                (0, 0, 255),
                self.lineType,
            )

    def run_lpr(self):
        items = [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
            "<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
            "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
            "<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
            "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
            "<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
            "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
            "<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
            "<Zhejiang>", "<police>",
            "A", "B", "C", "D", "E", "F", "G",
            "H", "I", "J", "K", "L", "M", "N",
            "O", "P", "Q", "R", "S", "T", "U",
            "V", "W", "X", "Y", "Z",
        ]

        for i in range(len(self.lpr_frame)):

            nn_data = run_nn(
                self.lpr_in,
                self.lpr_nn,
                {"data": to_planar(self.lpr_frame[i], (94, 24))},
            )

            lpr_str = ""
            results = to_nn_result(nn_data)

            for j in results:
                if j == -1:
                    break
                lpr_str += items[int(j)]

            cv2.imshow(
                f"CAR_CARD{i}",
                cv2.resize(self.lpr_frame[i], (188, 48)),
            )
            cv2.putText(
                self.debug_frame,
                lpr_str,
                (self.lpr_coords[i][0] - self.lineType * 20, self.lpr_coords[i][1] - 5),
                cv2.FONT_HERSHEY_COMPLEX,
                self.fontScale,
                (0, 0, 255),
                self.lineType,
            )

    def parse(self):

        if debug:
            self.debug_frame = self.frame.copy()
        try:
            vehicle_success = self.run_vlpd()
            if vehicle_success[0]:
                self.run_var()
            if vehicle_success[1]:
                self.run_lpr()
        except:
            pass

        if debug:
            aspect_ratio = self.frame.shape[1] / self.frame.shape[0]
            cv2.imshow(
                "Camera_view",
                cv2.resize(self.debug_frame, (int(900), int(900 / aspect_ratio))),
            )

            if cv2.waitKey(1) == ord("q"):
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
            self.frame = (
                np.array(self.cam_out.get().getData())
                .reshape((3, 300, 300))
                .transpose(1, 2, 0)
                .astype(np.uint8)
            )

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


if __name__ == "__main__":
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()
