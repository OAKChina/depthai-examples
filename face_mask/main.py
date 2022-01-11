# coding=utf-8
from queue import Queue

import blobconverter

from depthai_utils import *

parentDir = Path(__file__).parent
shaves = 6 if args.camera else 8
blobconverter.set_defaults(output_dir=parentDir / Path("models"))


class Main(DepthAI):
    def __init__(self, file=None, camera=False):
        self.face_coords = Queue()
        self.face_frames = Queue()
        self.cam_size = (300, 300)
        super(Main, self).__init__(file, camera)

    def create_nns(self):
        self.create_mobilenet_nn(
            blobconverter.from_zoo("face-detection-retail-0004", shaves=shaves),
            "face",
            first=True,
        )
        # https://github.com/sbdcv/sbd_mask/tree/master/model
        # https://github.com/sbdcv/sbd_mask/blob/8e25fbd550339857f6466016d3ed0866e759ab47/deploy.py#L11-L12
        self.create_nn(
            blobconverter.from_onnx(
                (Path(__file__).parent / Path("models/sbd_mask.onnx")).as_posix(),
                optimizer_params=[
                    "--scale_values=[255,255,255]",
                    "--reverse_input_channels",
                ],
                shaves=shaves,
            ),
            "mask",
        )

    def start_nns(self):
        if not self.camera:
            self.face_in = self.device.getInputQueue("face_in", 4, False)
        self.face_nn = self.device.getOutputQueue("face_nn", 4, False)
        self.mask_in = self.device.getInputQueue("mask_in", 4, False)
        self.mask_nn = self.device.getOutputQueue("mask_nn", 4, False)

    def run_face(self):
        # img, scale, top, left = resize_padding(self.frame, 300, 300)

        if not self.camera:
            nn_data = run_nn(
                self.face_in,
                self.face_nn,
                {"data": to_planar(self.frame, (300, 300))},
            )
        else:
            nn_data = self.face_nn.tryGet()
        if nn_data is None:
            return False

        bboxes = nn_data.detections
        self.number_of_people = len(bboxes)
        for bbox in bboxes:
            face_coord = frame_norm(
                (300, 300), *[bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
            )
            # face_coord = restore_point(face_coord, scale, top, left).astype(int)
            face_coord = scale_bbox(face_coord)
            self.face_frames.put(
                self.frame[face_coord[1] : face_coord[3], face_coord[0] : face_coord[2]]
            )
            self.face_coords.put(face_coord)
            self.draw_bbox(face_coord, (10, 245, 10))

        return True

    def run_mask(self):
        masked = 0
        while self.face_frames.qsize():
            face_frame = self.face_frames.get()
            face_coord = self.face_coords.get()
            nn_data = run_nn(
                self.mask_in,
                self.mask_nn,
                {"data": to_planar(face_frame, (224, 224))},
            )
            if nn_data is None:
                return
            self.fps_nn.update()
            out = softmax(to_nn_result(nn_data))
            mask = np.argmax(out) > 0.5

            color = (0, 255, 0) if mask else (0, 0, 255)
            self.draw_bbox(face_coord, color)
            self.put_text(f"{out[1]:.2%}", face_coord, color)

            if mask:
                masked += 1

            self.put_text(
                f"masks: {masked / self.number_of_people:.2%} ",
                (10, 30),
                (255, 0, 0),
                0.75,
            )

    def parse_fun(self):
        face_success = self.run_face()
        if face_success:
            self.run_mask()


if __name__ == "__main__":
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()
