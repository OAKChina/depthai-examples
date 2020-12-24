from pathlib import Path

from tools import *


class Vehicle(DepthAI):
    def __init__(self, file=None, camera=False):
        super().__init__(file, camera)

    def create_nns(self):

        self.create_nn(
            "models/vehicle-license-plate-detection-barrier-0106.blob", "vlpd"
        )
        self.create_nn("models/vehicle-attributes-recognition-barrier-0039.blob", "var")
        self.create_nn("models/license-plate-recognition-barrier-0007.blob", "lpr")

    def start_nns(self):
        self.vlpd_in = self.device.getInputQueue("vlpd_in")
        self.vlpd_nn = self.device.getOutputQueue("vlpd_nn")
        self.var_in = self.device.getInputQueue("var_in")
        self.var_nn = self.device.getOutputQueue("var_nn")
        self.lpr_in = self.device.getInputQueue("lpr_in")
        self.lpr_nn = self.device.getOutputQueue("lpr_nn")


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
            self.vehicle_coords[i][1]: self.vehicle_coords[i][3],
            self.vehicle_coords[i][0]: self.vehicle_coords[i][2],
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
            ): int(
                (self.lpr_coords[i][3] + self.lpr_coords[i][1]) / 2
                + (self.lpr_coords[i][3] - self.lpr_coords[i][1]) / 2 * 1.5
            ),
            int(
                (self.lpr_coords[i][2] + self.lpr_coords[i][0]) / 2
                - (self.lpr_coords[i][2] - self.lpr_coords[i][0]) / 2 * 1.5
            ): int(
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

    def parse_fun(self):

        vehicle_success = self.run_vlpd()
        if vehicle_success[0]:
            self.run_var()
        if vehicle_success[1]:
            self.run_lpr()


    def cam_size(self):
        self.first_size = (300, 300)


if __name__ == "__main__":
    if args.video:
        Vehicle(file=args.video).run()
    else:
        Vehicle(camera=args.camera).run()
