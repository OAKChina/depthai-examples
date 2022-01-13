# coding=utf-8
import blobconverter
from depthai_sdk import toPlanar, toTensorResult

from depthai_utils import *

parentDir = Path(__file__).parent
shaves = 6 if args.camera else 8
blobconverter.set_defaults(output_dir=parentDir / Path("models"))


class Main(DepthAI):
    def __init__(self, file=None, camera=False):
        self.cam_size = (255, 255)
        super().__init__(file, camera)

    def create_nns(self):
        # FireDetector: https://github.com/StephanXu/FireDetector/tree/python/example
        self.create_nn(
            blobconverter.from_tf(
                (Path(__file__).parent/Path("models/fire_detection_mobilenet_v2_100_224.pb")).as_posix(),
                optimizer_params=[
                    "--input_shape=[1,224,224,3]",
                    "--scale_values=[255,255,255]",
                ],
                shaves=shaves,
            ),
            "fire",
        )

    def start_nns(self):
        self.fire_in = self.device.getInputQueue("fire_in", 4, False)
        self.fire_nn = self.device.getOutputQueue("fire_nn", 4, False)

    def run_fire(self):
        labels = ["fire", "normal", "smoke"]
        w, h = self.frame.shape[:2]
        nn_data = run_nn(
            self.fire_in,
            self.fire_nn,
            {"Placeholder": toPlanar(self.frame, (224, 224))},
        )
        if nn_data is None:
            return
        self.fps_handler.tick("NN")
        results = toTensorResult(nn_data).get("final_result").squeeze()
        i = int(np.argmax(results))
        label = labels[i]
        if label == "normal":
            return
        else:
            if results[i] > 0.5:
                self.put_text(
                    f"{label}:{results[i]:.2f}",
                    (10, 25),
                    color=(0, 0, 255),
                    font_scale=1,
                )

    def parse_fun(self):
        self.run_fire()


if __name__ == "__main__":
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()
