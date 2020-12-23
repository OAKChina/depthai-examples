"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import cv2
import numpy as np
from numpy import clip

REFERENCE_LANDMARKS = [
        (30.2946 / 96, 51.6963 / 112), # left eye
        (65.5318 / 96, 51.5014 / 112), # right eye
        (48.0252 / 96, 71.7366 / 112), # nose tip
        (33.5493 / 96, 92.3655 / 112), # left lip corner
        (62.7299 / 96, 92.2041 / 112)] # right lip corner


def _align_rois(face_images, face_landmarks):
    image = face_images.copy()

    scale = np.array((image.shape[1], image.shape[0]))
    desired_landmarks = np.array(REFERENCE_LANDMARKS, dtype=np.float64) * scale
    landmarks = face_landmarks * scale

    transform = get_transform(desired_landmarks, landmarks)
    img = cv2.warpAffine(image, transform, tuple(scale),flags=cv2.WARP_INVERSE_MAP)
    return img

def preprocess(frame,landmarks,frame_shape):
    image = _align_rois(frame,landmarks)
    image = resize_input(image,frame_shape)
    return [image]

def get_transform(src, dst):
    src_col_mean, src_col_std = normalize(src, axis=(0))
    dst_col_mean, dst_col_std = normalize(dst, axis=(0))

    u, _, vt = np.linalg.svd(np.matmul(src.T, dst))
    r = np.matmul(u, vt).T

    transform = np.empty((2, 3))
    transform[:, 0:2] = r * (dst_col_std / src_col_std)
    transform[:, 2] = dst_col_mean.T - \
        np.matmul(transform[:, 0:2], src_col_mean.T)
    return transform

def normalize(array, axis):
    mean = array.mean(axis=axis)
    array -= mean
    std = array.std()
    array /= std
    return mean, std


def resize_input(frame, target_shape):
    # assert len(frame.shape) == len(target_shape), \
    #     "Expected a frame with %s dimensions, but got %s" % \
    #     (len(target_shape), len(frame.shape))

    # assert frame.shape[0] == 1, "Only batch size 1 is supported"
    h, w, c = target_shape

    input = frame
    if not np.array_equal(target_shape[-2:], frame.shape[-2:]):
        input = input.transpose((1, 2, 0)) # to HWC
        input = cv2.resize(input, (w, h))
        input = input.transpose((2, 0, 1)) # to CHW

    return input.reshape((h, w, c))