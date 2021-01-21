# coding=utf-8
from depthai_utils import *


class Main(DepthAI):
    def __init__(self, file=None, camera=False):
        super().__init__(file, camera)

    def create_nns(self):
        self.create_nn("models/fire_detection.blob", "fire")

    def start_nns(self):

        self.fire_in = self.device.getInputQueue("fire_in")
        self.fire_nn = self.device.getOutputQueue("fire_nn")

    def run_fire(self):
        labels = ["fire", "normal", "smoke"]
        w,h = self.frame.shape[:2]
        nn_data = run_nn(
            self.fire_in,
            self.fire_nn,
            {"Placeholder": to_planar(self.frame, (224, 224))},
        )
        results = to_tensor_result(nn_data).get("final_result")
        i = int(np.argmax(results))
        label = labels[i]
        # print(i)
        # print(self.frame.shape)
        dot = int(w//3), int(h//3)
        if label == 'normal':
            self.put_text(f"normal:{results[i]:.2f}", dot, color=(0, 255, 0), font_scale=1)
        else:
            self.put_text(f"{label}:{results[i]:.2f}", dot, color=(0, 0, 255),font_scale=1)
            if label == 'fire' and results[i] > 0.8:
                cv2.imwrite('fire_demo1.png', self.debug_frame)
            if label == 'smoke' and results[i] >= 0.98:
                cv2.imwrite('fire_demo2.png', self.debug_frame)



        # print(results)
        # print("=" * 20)
        # print("fire", to_tensor_result(nn_data))

    def parse_fun(self):
        self.run_fire()

    def cam_size(self):
        self.first_size = (224, 224)


if __name__ == "__main__":
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()
