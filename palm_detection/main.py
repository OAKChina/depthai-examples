# coding=utf-8

from depthai_utils import *


class Main(DepthAI):
    def __init__(self, file=None, camera=False):
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
        min_score_thresh = 0.6
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

    def parse_fun(self):
        self.run_palm()

    def cam_size(self):
        self.first_size = (128, 128)


if __name__ == "__main__":
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()
