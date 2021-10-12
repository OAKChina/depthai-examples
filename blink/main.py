# coding=utf-8
from queue import Queue

import blobconverter

from depthai_utils import *

blobconverter.set_defaults(output_dir=Path(__file__).parent / Path("models"))


class Main(DepthAI):
    def __init__(self, file=None, camera=False):
        self.cam_size = (300, 300)
        super().__init__(file, camera)
        self.face_coords = Queue()
        self.face_frame = Queue()
        self.left_eye_blink = []
        self.number_of_blinks_of_left_eye = 0
        self.right_eye_blink = []
        self.number_of_blinks_of_right_eye = 0

    def create_nns(self):
        self.create_mobilenet_nn(
            blobconverter.from_zoo("face-detection-retail-0004", shaves=6),
            "face",
            conf=0.8,
            first=True,
        )
        # https://github.com/PINTO0309/PINTO_model_zoo/tree/main/032_FaceMesh
        self.create_nn(
            # blobconverter.from_onnx(
            #     "models/face_mesh_192x192.onnx",
            #     optimizer_params=[
            #         "--mean_values=[127.5,127.5,127.5]",
            #         "--scale_values=[127.5,127.5,127.5]",
            #     ],
            #     shaves=6,
            # ),
            blobconverter.from_openvino(
                "models/face_mesh_192x192.xml",
                "models/face_mesh_192x192.bin",
                shaves=6,
            ),
            "mesh",
        )
        self.create_nn(blobconverter.from_zoo("open-closed-eye-0001", shaves=6), "eye")

    def start_nns(self):
        if not self.camera:
            self.face_in = self.device.getInputQueue("face_in")
        self.face_nn = self.device.getOutputQueue("face_nn", maxSize=4, blocking=False)
        self.mesh_in = self.device.getInputQueue("mesh_in", maxSize=4, blocking=False)
        self.mesh_nn = self.device.getOutputQueue("mesh_nn", maxSize=4, blocking=False)
        self.eye_in = self.device.getInputQueue("eye_in", maxSize=4, blocking=False)
        self.eye_nn = self.device.getOutputQueue("eye_nn", maxSize=4, blocking=False)

    def run_face(self):
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
        self.fps_nn.update()

        bboxes = nn_data.detections
        for bbox in bboxes:
            face_coord = frame_norm(
                (300, 300), *[bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax]
            )
            self.face_frame.put(
                self.frame[face_coord[1] : face_coord[3], face_coord[0] : face_coord[2]]
            )

            self.face_coords.put(face_coord)
            self.draw_bbox(face_coord, (10, 245, 10))
        return True

    def run_mesh(self):
        for i in range(self.face_frame.qsize()):
            frame = self.face_frame.get()
            nn_data = run_nn(
                self.mesh_in,
                self.mesh_nn,
                {"data": to_planar(frame, (192, 192))},
            )

            if nn_data is None:
                return

            results = to_tensor_result(nn_data)
            detections = results.get("conv2d_20").reshape(-1, 3)
            detections[..., 0] /= 192
            detections[..., 1] /= 192
            detections = [frame_norm(frame, *obj[:2]) for obj in detections]

            left_eye = np.array(
                (
                    detections[71],
                    detections[107],
                    detections[116],
                    detections[197],
                )
            )
            right_eye = np.array(
                (
                    detections[301],
                    detections[336],
                    detections[345],
                    detections[197],
                )
            )
            (
                self.left_eye_blink,
                self.number_of_blinks_of_left_eye,
            ) = self.frame_eyes(frame, left_eye, "l")
            (
                self.right_eye_blink,
                self.number_of_blinks_of_right_eye,
            ) = self.frame_eyes(frame, right_eye, "r")

            #
            # # count=0
            # for dot in detections:
            #     dot += self.face_coords[i][:2]
            #     self.draw_dot(dot, (255, 0, 255))
            #     # self.put_text(f"{count}",dot,font_scale=0.5,line_type=1)
            #     # count += 1

    def frame_eyes(self, face_frame, eye_array, mark: str):
        eye_blink = self.right_eye_blink if mark == "r" else self.left_eye_blink
        number_of_blinks_of_eye = (
            self.number_of_blinks_of_right_eye
            if mark == "r"
            else self.number_of_blinks_of_left_eye
        )
        n = "right" if mark == "r" else "left"

        rect = cv2.minAreaRect(eye_array)
        box = cv2.boxPoints(rect)
        eye = np.append(np.min(box, axis=0), np.max(box, axis=0)).astype(int)
        eye = np.where(eye > 0, eye, 0)
        self.eye_img = face_frame[eye[1] : eye[3], eye[0] : eye[2]]
        cv2.imshow(n, self.eye_img)
        closed = self.run_eye()
        if closed is not None:
            eye_blink.append(closed)
            if len(eye_blink) > 20:
                eye_blink.pop(0)
            if eye_blink[-5:-1].count(closed) == 0 and len(eye_blink) > 5:
                self.put_text("Blink", (20, 30))
                number_of_blinks_of_eye += 1
        return eye_blink, number_of_blinks_of_eye

    def run_eye(self):

        nn_data = run_nn(
            self.eye_in,
            self.eye_nn,
            {"input.1": to_planar(self.eye_img, (32, 32))},
        )

        if nn_data is None:
            return
        out = to_nn_result(nn_data)
        if np.max(out) < 0.5:
            return
        res = np.argmax(out)
        return res

    def parse_fun(self):
        try:
            if self.run_face():
                self.run_mesh()
        except Exception as e:
            print(e)
            pass
        finally:
            self.put_text(
                f"NumberOfBlinksOfRightEye:{self.number_of_blinks_of_right_eye}",
                (20, 50),
                font_scale=0.5,
                line_type=1,
            )
            self.put_text(
                f"NumberOfBlinksOfLeftEye:{self.number_of_blinks_of_left_eye}",
                (20, 80),
                font_scale=0.5,
                line_type=1,
            )


if __name__ == "__main__":
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()
