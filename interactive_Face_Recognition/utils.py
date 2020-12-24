import cv2
import numpy as np
from numpy import clip

class Utils:
    def __init__(self):
        self.REFERENCE_LANDMARKS = [
                (30.2946 / 96, 51.6963 / 112), # left eye
                (65.5318 / 96, 51.5014 / 112), # right eye
                (48.0252 / 96, 71.7366 / 112), # nose tip
                (33.5493 / 96, 92.3655 / 112), # left lip corner
                (62.7299 / 96, 92.2041 / 112)] # right lip corner


    def _align_rois(self,face_images, face_landmarks):
        image = face_images.copy()

        scale = np.array((image.shape[1], image.shape[0]))
        desired_landmarks = np.array(self.REFERENCE_LANDMARKS, dtype=np.float64) * scale
        landmarks = face_landmarks * scale

        transform = self.get_transform(desired_landmarks, landmarks)
        img = cv2.warpAffine(image, transform, tuple(scale),flags=cv2.WARP_INVERSE_MAP)
        return img

    def get_transform(self,src, dst):
        src_col_mean, src_col_std = self.normalize(src, axis=(0))
        dst_col_mean, dst_col_std = self.normalize(dst, axis=(0))

        u, _, vt = np.linalg.svd(np.matmul(src.T, dst))
        r = np.matmul(u, vt).T

        transform = np.empty((2, 3))
        transform[:, 0:2] = r * (dst_col_std / src_col_std)
        transform[:, 2] = dst_col_mean.T - \
            np.matmul(transform[:, 0:2], src_col_mean.T)
        return transform

    def normalize(self,array, axis):
        mean = array.mean(axis=axis)
        array -= mean
        std = array.std()
        array /= std
        return mean, std


    def resize_input(self,frame, target_shape):
        h, w, c = target_shape

        input = frame
        if not np.array_equal(target_shape[-2:], frame.shape[-2:]):
            input = input.transpose((1, 2, 0)) # to HWC
            input = cv2.resize(input, (w, h))
            input = input.transpose((2, 0, 1)) # to CHW

        return input.reshape((h, w, c))

    def preprocess(self,frame,landmarks,frame_shape):
        image = self._align_rois(frame,landmarks)
        image = self.resize_input(image,frame_shape)
        return [image]