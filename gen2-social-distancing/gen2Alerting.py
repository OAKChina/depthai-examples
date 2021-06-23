import math
from datetime import datetime, timedelta

import cv2
import numpy as np

class Close:
    confidence_threshold = 0.5

    def __init__(self):
        self.set_defaults()

    def set_defaults(self):
        self.last_reported = None
        self.statuses = []

    def alerting(self, results):
        if len(results) > 0:
            has_danger = any(map(lambda item: item['dangerous'], results))
            if has_danger:
                self.last_reported = datetime.now()
            if self.last_reported is not None:
                self.statuses = self.statuses[-50:] + [has_danger]

        if self.last_reported is not None and datetime.now() - self.last_reported > timedelta(seconds=5):
            self.set_defaults()

        if len(self.statuses) > 10 and sum(self.statuses) / len(self.statuses) > self.confidence_threshold:
            return True
        else:
            return False

    def parse_frame(self, frame, results):
        should_alert = self.alerting(results)
        if should_alert:
            img_h = frame.shape[0]
            img_w = frame.shape[1]
            cv2.putText(frame, "Too close", (int(img_w / 3), int(img_h / 2)), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 1)
        return results, should_alert

class Bird(Close):
    max_z = 6
    min_z = 0
    max_x = 1.3
    min_x = -0.5

    def __init__(self, frame_size):
        super().__init__()
        self.height_size = frame_size[1]
        self.distance_bird_frame = self.make_bird_frame()

    def make_bird_frame(self):
        fov = 68.7938
        min_distance = 0.827
        frame = np.zeros((self.height_size, 100, 3), np.uint8)
        min_y = int((1 - (min_distance - self.min_z) / (self.max_z - self.min_z)) * frame.shape[0])
        cv2.rectangle(frame, (0, min_y), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1)

        alpha = (180 - fov) / 2
        center = int(frame.shape[1] / 2)
        max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
        fov_cnt = np.array([
            (0, frame.shape[0]),
            (frame.shape[1], frame.shape[0]),
            (frame.shape[1], max_p),
            (center, frame.shape[0]),
            (0, max_p),
            (0, frame.shape[0]),
        ])
        cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        return frame

    def calc_x(self, val):
        norm = min(self.max_x, max(val, self.min_x))
        center = (norm - self.min_x) / (self.max_x - self.min_x) * self.distance_bird_frame.shape[1]
        bottom_x = max(center - 2, 0)
        top_x = min(center + 2, self.distance_bird_frame.shape[1])
        return int(bottom_x), int(top_x)

    def calc_z(self, val):
        norm = min(self.max_z, max(val, self.min_z))
        center = (1 - (norm - self.min_z) / (self.max_z - self.min_z)) * self.distance_bird_frame.shape[0]
        bottom_z = max(center - 2, 0)
        top_z = min(center + 2, self.distance_bird_frame.shape[0])
        return int(bottom_z), int(top_z)

    def parse_frame(self, frame, results, boxse):
        distance_results, should_alert = super().parse_frame(frame, results)

        bird_frame = self.distance_bird_frame.copy()
        too_close_ids = []
        for result in distance_results:
            if result['dangerous']:
                left1, right1 = self.calc_x(result['detection1']['depth_x'])
                top1, bottom1 = self.calc_z(result['detection1']['depth_z'])
                cv2.rectangle(bird_frame, (left1, top1), (right1, bottom1), (0, 0, 255), 2)
                too_close_ids.append(result['detection1']['id'])
                left2, right2 = self.calc_x(result['detection2']['depth_x'])
                top2, bottom2 = self.calc_z(result['detection2']['depth_z'])
                cv2.rectangle(bird_frame, (left2, top2), (right2, bottom2), (0, 0, 255), 2)
                too_close_ids.append(result['detection2']['id'])
                print((left1, right1, top1, bottom1),(left2, top2, right2, bottom2))

        for result in boxse:
            if result['id'] not in too_close_ids:
                left, right = self.calc_x(result['depth_x'])
                top, bottom = self.calc_z(result['depth_z'])
                cv2.rectangle(bird_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                print((left, top, right, bottom))
        
        return bird_frame
