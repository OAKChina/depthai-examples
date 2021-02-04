# coding=utf-8

from depthai_utils import *

from Xlib.display import Display
from pynput.mouse import Controller

mouse = Controller()
screen = Display().screen()
screen_size = (screen.width_in_pixels, screen.height_in_pixels)
print(screen_size)


def distance(pt1, pt2):
    """
    两点间距离
    """
    assert len(pt1) == len(pt2), f"两点维度要一致，pt1:{len(pt1)}维, pt2:{len(pt2)}维"
    return np.sqrt(np.float_power(np.array(pt1) - pt2, 2).sum())


def point_mapping(dot, center, original_side_length, target_side_length):
    """

    :param dot: 点座标
    :param center: 帧中心点座标
    :param original_side_length: 源边长
    :param target_side_length: 目标边长
    :return:
    """
    if isinstance(original_side_length, (int, float)):
        original_side_length = np.array(
            (original_side_length, original_side_length)
        )
    if isinstance(target_side_length, (int, float)):
        target_side_length = np.array((target_side_length, target_side_length))

    return center + (np.array(dot) - center) * (
        np.array(target_side_length) / original_side_length
    )


class Main(DepthAI):
    def __init__(self, file=None, camera=False):
        self.cam_size = (128, 128)
        super().__init__(file, camera)

    def create_nns(self):
        self.create_nn("models/palm_detection.blob", "palm", first=True)

    def start_nns(self):
        if not self.camera:
            self.palm_in = self.device.getInputQueue(
                "palm_in", maxSize=4, blocking=False
            )
        self.palm_nn = self.device.getOutputQueue(
            "palm_nn", maxSize=4, blocking=False
        )

    def run_palm(self):
        """
        Each palm detection is a tensor consisting of 19 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 7 key_points
            - confidence score
        :return:
        """
        shape = (128, 128)
        num_keypoints = 7
        min_score_thresh = 0.7
        anchors = np.load("anchors_palm.npy")

        if not self.camera:
            nn_data = run_nn(
                self.palm_in,
                self.palm_nn,
                {"input": to_planar(self.frame, shape)},
            )
        else:
            nn_data = self.palm_nn.tryGet()

        if nn_data is None:
            return

        # Run the neural network
        results = to_tensor_result(nn_data)

        raw_box_tensor = results.get("regressors").reshape(
            -1, 896, 18
        )  # regress
        raw_score_tensor = results.get("classificators").reshape(
            -1, 896, 1
        )  # classification

        detections = raw_to_detections(
            raw_box_tensor, raw_score_tensor, anchors, shape, num_keypoints
        )
        # print(detections.shape)

        self.palm_coords = [
            frame_norm(self.frame, *obj[:4])
            for det in detections
            for obj in det
            if obj[-1] > min_score_thresh
        ]

        self.palm_confs = [
            obj[-1]
            for det in detections
            for obj in det
            if obj[-1] > min_score_thresh
        ]

        if len(self.palm_coords) == 0:
            return False

        self.palm_coords = non_max_suppression(
            boxes=np.concatenate(self.palm_coords).reshape(-1, 4),
            probs=self.palm_confs,
            overlapThresh=0.1,
        )

        for bbox in self.palm_coords:
            self.draw_bbox(bbox, (10, 245, 10))
            dot_x = (bbox[2] + bbox[0]) / 2
            dot_y = (bbox[3] + bbox[1]) / 2
            self.move_mouse((dot_x, dot_y))

    dots = []

    # @timer
    def move_mouse(self, dot):
        if len(self.dots) > 0 and distance(dot, self.dots[-1]) < 5:
            dot = self.dots[-1]
        self.dots.append(dot)
        if len(self.dots) >= 10:
            dot = np.mean(self.dots, axis=0)
            dot_s = point_mapping(dot, (64, 64), 108, 128)
            dot_l = point_mapping(
                dot_s, (0, 0), self.frame.shape[:2], screen_size
            )
            mouse.position = (dot_l[0], dot_l[1])
            self.dots.pop(0)

    # @timer
    def parse_fun(self):
        self.run_palm()

if __name__ == "__main__":
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()
