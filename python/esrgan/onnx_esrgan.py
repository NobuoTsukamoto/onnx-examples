#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT Super resolution.

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import time

import cv2
import numpy as np
import onnxruntime


def normalize(img):
    img = np.asarray(img, dtype="float32")
    # img = img / 255.0
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of onnx model.", required=True)
    parser.add_argument(
        "--input_shape",
        type=str,
        default="50,50",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--image", help="File path of input image.", type=str, required=True
    )
    parser.add_argument(
        "--output", help="File path of output image.", type=str, required=True
    )
    args = parser.parse_args()

    # Read image file.
    im = cv2.imread(args.image)
    w, h, c = im.shape
    print("Input Image (height, width, channel): ", h, w, c)

    # Load model.
    input_shape = tuple(map(int, args.input_shape.split(",")))
    session = onnxruntime.InferenceSession(args.model)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    normalized_im = normalize(im)
    print(normalized_im.shape)

    # inference.
    start = time.perf_counter()
    ort_inputs = {session.get_inputs()[0].name: normalized_im[None, :, :, :]}
    outputs = session.run(None, ort_inputs)
    inference_time = (time.perf_counter() - start) * 1000

    output_image = np.array(outputs[0])
    output_image = output_image.clip(0.0, 255.0)
    output_image = output_image.reshape(input_shape[0] * 4, input_shape[1] * 4, 3)
    output_image = np.asarray(output_image, dtype="uint8")
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Display fps
    fps_text = "Inference: {0:.2f}ms".format(inference_time)
    print(fps_text)

    # Output image file
    cv2.imwrite(args.output, output_image)


if __name__ == "__main__":
    main()
