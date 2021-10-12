# coding=utf-8
from queue import Queue

import blobconverter

from depthai_utils import *

parentDir = Path(__file__).parent
shaves = 6 if args.camera else 8
blobconverter.set_defaults(output_dir=parentDir / Path("models"))

class Main(DepthAI):
    def __init__(self, file=None, camera=False):
        self.cam_size = (720, 1280)
        super(Main, self).__init__(file, camera)
        self.age_frame = Queue()
        self.emo_frame = Queue()

    def create_nns(self):
        self.create_nn(blobconverter.from_zoo("face-detection-retail-0004", shaves=shaves),
                       "face")
        self.create_nn(blobconverter.from_zoo("age-gender-recognition-retail-0013", shaves=shaves),
                       "ag")
        self.create_nn(blobconverter.from_zoo("emotions-recognition-retail-0003", shaves=shaves),
                       "emo")

    def start_nns(self):
        self.face_in = self.device.getInputQueue("face_in")
        self.face_nn = self.device.getOutputQueue("face_nn")
        self.ag_in = self.device.getInputQueue("ag_in")
        self.ag_nn = self.device.getOutputQueue("ag_nn")
        self.emo_in = self.device.getInputQueue("emo_in")
        self.emo_nn = self.device.getOutputQueue("emo_nn")

    def run_face(self):
        data, scale, top, left = to_planar(self.frame, (300, 300))
        nn_data = run_nn(
            self.face_in,
            self.face_nn,
            {"data": data},
        )

        if nn_data is None:
            return False

        results = to_bbox_result(nn_data)

        self.face_coords = [
            frame_norm((300, 300), *obj[3:7]) for obj in results if obj[2] > 0.8
        ]

        self.face_num = len(self.face_coords)
        if self.face_num == 0:
            return False

        self.face_coords = restore_point(self.face_coords, scale, top, left).astype(int)

        self.face_coords = scale_bboxes(self.face_coords, scale=True, scale_size=1.5)

        for i in range(self.face_num):
            face = self.frame[
                self.face_coords[i][1] : self.face_coords[i][3],
                self.face_coords[i][0] : self.face_coords[i][2],
            ]
            self.age_frame.put(face)
            self.emo_frame.put(face)
        if debug:
            for bbox in self.face_coords:
                self.draw_bbox(bbox, (10, 245, 10))
        return True

    def run_ag(self):
        for i in range(self.face_num):
            age_frame = self.age_frame.get()
            nn_data = run_nn(
                self.ag_in,
                self.ag_nn,
                {"data": to_planar(age_frame, (62, 62))[0]},
            )
            if nn_data is None:
                return
            results = to_tensor_result(nn_data)
            age = results["age_conv3"][0] * 100
            gender = "Male" if results["prob"].argmax() else "Female"

            # print(results)

            self.put_text(
                f"Age: {age:0.0f}",
                (self.face_coords[i][0], self.face_coords[i][1] + 20),
                (244, 0, 255),
            )
            self.put_text(
                f"G: {gender}",
                (self.face_coords[i][0], self.face_coords[i][1] + 50),
                (0, 244, 255),
            )

    def run_emo(self):
        for i in range(self.face_num):
            emo_frame = self.emo_frame.get()
            emo = ["neutral", "happy", "sad", "surprise", "anger"]
            nn_data = run_nn(
                self.emo_in,
                self.emo_nn,
                {"prob": to_planar(emo_frame, (64, 64))[0]},
            )
            if nn_data is None:
                return
            out = to_nn_result(nn_data)
            # print(out)
            emo_r = emo[out.argmax()]

            self.put_text(
                emo_r,
                (self.face_coords[i][0], self.face_coords[i][1] + 80),
                (0, 0, 255),
            )

    def parse_fun(self):
        face_success = self.run_face()
        if face_success:
            self.run_ag()
            self.run_emo()


if __name__ == "__main__":
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()