#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT DeepLab v3 MobileNetEdgeTPUV2.

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import time
import os

import cv2
import numpy as np
import onnxruntime


WINDOW_NAME = "ONNX DeepLab v3 MobileNetEdgeTPUdV2 example."


def normalize(im):
    im = np.asarray(im, dtype="float32")
    im = im.transpose(2, 0, 1)
    input_data = np.expand_dims(im, axis=0)
    input_data = input_data.astype(np.float) / 128 - 0.5
    return input_data


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1
    )


def create_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    ind = np.arange(256, dtype=np.uint8)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap


def label_to_color_image(colormap, label):
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of trt model.", required=True)
    parser.add_argument(
        "--input_shape",
        type=str,
        default="512,512",
        help="Specify an input shape for inference.",
    )
    parser.add_argument("--input", help="File path of input image file.", required=True, type=str)
    parser.add_argument("--output", help="File path of output image.", required=True, type=str)
    args = parser.parse_args()

    # Initialize colormap
    colormap = create_label_colormap()

    # Load model.
    input_shape = tuple(map(int, args.input_shape.split(",")))
    session = onnxruntime.InferenceSession(args.model)

    frame = cv2.imread(args.input)
    h, w, _ = frame.shape
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_im = cv2.resize(im, input_shape)
    normalized_im = normalize(resized_im)

    # inference.
    start = time.perf_counter()
    ort_inputs = {session.get_inputs()[0].name: normalized_im.astype("float32")}
    outputs = session.run(None, ort_inputs)
    inference_time = (time.perf_counter() - start) * 1000

    seg_map = np.array(outputs[0])
    seg_map = seg_map.reshape(input_shape[0], input_shape[1]).astype(np.uint8)
    seg_image = label_to_color_image(colormap, seg_map)
    seg_image = cv2.resize(seg_image, (w, h))
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) // 2 + seg_image // 2
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    # Output image file.
    cv2.imwrite(args.output, im)


if __name__ == "__main__":
    main()
