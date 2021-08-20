#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = round(box[0])
        y0 = round(box[1])
        x1 = round(box[2])
        y1 = round(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        if class_names is not None:
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 - int(1.5*txt_size[1])),
                (x0 + txt_size[0] + 1, y0 + 1),
                # (x0 + y0 + 1),
                # (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            # cv2.putText(img, text, (x0, y0 - txt_size[1]), font, 0.4, txt_color, thickness=1)
            cv2.putText(img, text, (x0, y0 ), font, 0.4, txt_color, thickness=1)

    return img


_COLORS = np.array(
    [
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],

    ]
).astype(np.float32).reshape(-1, 3)
