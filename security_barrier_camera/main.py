# coding=utf-8
from collections import deque

import blobconverter
from depthai_sdk import toPlanar, frameNorm, toTensorResult

from depthai_utils import *

parentDir = Path(__file__).parent
shaves = 6 if args.camera else 8
blobconverter.set_defaults(output_dir=parentDir / Path("models"))


class Vehicle(DepthAI):
    def __init__(self, file=None, camera=False):
        self.lpr_coords = deque()
        self.vehicle_coords = deque()
        self.cam_size = (300, 300)

        super().__init__(file, camera)

    def create_nns(self):

        self.create_mobilenet_nn(
            blobconverter.from_zoo(
                "vehicle-license-plate-detection-barrier-0106", shaves=shaves
            ),
            "vlpd",
            first=True if self.camera else False,
        )
        self.create_nn(
            blobconverter.from_zoo(
                "vehicle-attributes-recognition-barrier-0039", shaves=shaves
            ),
            "var",
        )
        self.create_nn(
            blobconverter.from_zoo("license-plate-recognition-barrier-0007", shaves=shaves),
            "lpr",
        )

    def start_nns(self):
        if not self.camera:
            self.vlpd_in = self.device.getInputQueue("vlpd_in")
        self.vlpd_nn = self.device.getOutputQueue("vlpd_nn")
        self.var_in = self.device.getInputQueue("var_in")
        self.var_nn = self.device.getOutputQueue("var_nn")
        self.lpr_in = self.device.getInputQueue("lpr_in")
        self.lpr_nn = self.device.getOutputQueue("lpr_nn")

    def run_vlpd(self):
        if not self.camera:
            nn_data = run_nn(
                self.vlpd_in, self.vlpd_nn, {"data": toPlanar(self.frame, (300, 300))}
            )
        else:
            nn_data = self.vlpd_nn.tryGet()

        if nn_data is None:
            return False, False
        self.fps_handler.tick("vehicle-license-plate-detection")
        bboxes = nn_data.detections
        self.vehicle_coords.clear()
        self.lpr_coords.clear()
        for bbox in bboxes:
            if bbox.label == 1:
                self.vehicle_coords.append(
                    frameNorm(
                        self.debug_frame, [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
                    )
                )
            elif bbox.label == 2:
                self.lpr_coords.append(
                    frameNorm(
                        self.debug_frame, [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
                    )
                )

        flag_vehicle = False if len(self.vehicle_coords) == 0 else True
        flag_card = False if len(self.lpr_coords) == 0 else True
        if flag_vehicle:
            self.vehicle_()
        if flag_card:
            self.lpr_()
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
                self.lpr_coords[i][1] : self.lpr_coords[i][3],
                self.lpr_coords[i][0] : self.lpr_coords[i][2],
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
                {"data": toPlanar(self.vehicle_frame[i], (72, 72))},
            )
            if nn_data is None:
                break
            results = toTensorResult(nn_data)
            # for key in results.keys():
            #     results[key] = results[key][: len(results[key]) // 2]

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
                    self.vehicle_coords[i][1] + 15,
                ),
                cv2.FONT_HERSHEY_COMPLEX,
                self.fontScale,
                (0, 0, 255),
                self.lineType,
            )

    def run_lpr(self):
        items = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "<Anhui>",
            "<Beijing>",
            "<Chongqing>",
            "<Fujian>",
            "<Gansu>",
            "<Guangdong>",
            "<Guangxi>",
            "<Guizhou>",
            "<Hainan>",
            "<Hebei>",
            "<Heilongjiang>",
            "<Henan>",
            "<HongKong>",
            "<Hubei>",
            "<Hunan>",
            "<InnerMongolia>",
            "<Jiangsu>",
            "<Jiangxi>",
            "<Jilin>",
            "<Liaoning>",
            "<Macau>",
            "<Ningxia>",
            "<Qinghai>",
            "<Shaanxi>",
            "<Shandong>",
            "<Shanghai>",
            "<Shanxi>",
            "<Sichuan>",
            "<Tianjin>",
            "<Tibet>",
            "<Xinjiang>",
            "<Yunnan>",
            "<Zhejiang>",
            "<police>",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
        ]

        for i in range(len(self.lpr_frame)):

            nn_data = run_nn(
                self.lpr_in,
                self.lpr_nn,
                {"data": toPlanar(self.lpr_frame[i], (94, 24))},
            )

            if nn_data is None:
                break

            lpr_str = ""
            results = toTensorResult(nn_data)["d_predictions.0"].squeeze()

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

    def parse_fun(self):

        vehicle_success = self.run_vlpd()
        if vehicle_success[0]:
            self.run_var()
        if vehicle_success[1]:
            self.run_lpr()


if __name__ == "__main__":
    if args.video:
        Vehicle(file=args.video).run()
    else:
        Vehicle(camera=args.camera).run()
