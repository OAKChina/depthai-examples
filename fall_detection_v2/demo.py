import argparse, time
from MovenetRenderer import MovenetRenderer
import numpy as np
import cv2
import depthai as dai


parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="thunder",
    help="Model to use : 'thunder' or 'lightning' or path of a blob file (default=%(default)s)",
)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="rgb",
    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default=%(default)s)",
)
parser.add_argument(
    "-c", "--crop", action="store_true", help="Center crop frames to a square shape"
)
parser.add_argument(
    "-s",
    "--score_threshold",
    default=0.2,
    type=float,
    help="Confidence score to determine whether a keypoint prediction is reliable (default=%(default)f)",
)
parser.add_argument(
    "--internal_fps",
    type=int,
    help="Fps of internal color camera. Too high value lower NN fps (default: depends on the model",
)
parser.add_argument(
    "--internal_frame_height",
    type=int,
    default=640,
    help="Internal color camera frame height in pixels (default=%(default)i)",
)
parser.add_argument("-o", "--output", help="Path to output video file")


args = parser.parse_args()

from MovenetDepthai import MovenetDepthai

pose = MovenetDepthai(
    input_src=args.input,
    model=args.model,
    score_thresh=args.score_threshold,
    crop=args.crop,
    internal_fps=args.internal_fps,
    internal_frame_height=args.internal_frame_height,
)

renderer = MovenetRenderer(pose, output=args.output)

while True:
    # Run movenet on next frame
    frame, body = pose.next_frame()

    if frame is None:
        break
    if sum(body.scores > body.score_thresh) > 8:
        keypoints = np.clip(body.keypoints, [0, 0], [frame.shape[1], frame.shape[0]])
        x, y, w, h = cv2.boundingRect(keypoints)

        I = np.zeros_like(frame, dtype=np.uint8)
        I = renderer.draw(I, body)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        I = np.clip(I, 0, 1) * 255
        I = pose.crop_and_resize(I, pose.crop_region)

        # I = I[y : y + h, x : x + w]
        I = cv2.resize(I, (128, 128))

        frame_ac = dai.ImgFrame()
        frame_ac.setTimestamp(time.monotonic())
        frame_ac.setWidth(128)
        frame_ac.setHeight(128)
        frame_ac.setData(I)
        pose.q_ac_in.send(frame_ac)
        crown_proportion = w / h
        # Get result from device
        predect = pose.q_ac_out.get()
        predect = np.array(predect.getLayerFp16("output")).reshape(-1, 2)
        action_id = int(np.argmax(predect))
        possible_rate = 0.6 * predect[:, action_id] + 0.4 * (crown_proportion - 1)

        if possible_rate > 0.55:
            pose_action = "fall"
            print(predect)
            if possible_rate > 0.7:
                cv2.putText(
                    frame,
                    pose_action,
                    (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    2,
                )

            if possible_rate > 1:
                possible_rate = 1
            action_fall = possible_rate
            action_normal = 1 - possible_rate
        else:
            pose_action = "normal"

            if possible_rate >= 0.5:
                action_fall = 1 - possible_rate
                action_normal = possible_rate
            else:
                cv2.putText(
                    frame,
                    pose_action,
                    (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    2,
                )
                action_fall = possible_rate
                action_normal = 1 - possible_rate

        # print(pose_action)
        cv2.imshow("", I)

    # Draw 2d skeleton
    renderer.draw(frame, body)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord("q"):
        break
renderer.exit()
pose.exit()
