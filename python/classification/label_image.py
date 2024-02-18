#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    ONNX Runtime image classification

    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import time

import cv2
import numpy as np
import onnxruntime


def normalize(img):
    img = img / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


def load_labels(filename):
    with open(filename, "r") as f:
        return [line.rstrip() for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="File path of image.", required=True)
    parser.add_argument("--model", help="File path of ONNX model.", required=True)
    parser.add_argument(
        "--label",
        help="Specify an input shape for inference.",
    )
    parser.add_argument("--count", help="Repeat count.", default=1, type=int)
    parser.add_argument(
        "--display_every",
        type=int,
        default=100,
        help="Number of iterations executed between two consecutive display of metrics",
    )
    parser.add_argument(
        "--num_warmup_iterations",
        type=int,
        default=50,
        help="Number of initial iterations skipped from timing",
    )
    args = parser.parse_args()

    # Load model.
    session = onnxruntime.InferenceSession(args.model)
    input_shape = session.get_inputs()[0].shape

    # load label file
    labels = load_labels(args.label) if args.label else None

    # Load image
    img = cv2.imread(args.image, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[3], input_shape[2]))
    img = normalize(img)

    elapsed_list = []

    # inference.
    ort_inputs = {session.get_inputs()[0].name: img}

    for i in range(args.count):
        start = time.perf_counter()
        preds = session.run(None, ort_inputs)[0]

        inference_time = time.perf_counter() - start
        elapsed_list.append(inference_time)

        if i == 0:
            print("First Inference : {0:.2f} ms".format(inference_time * 1000))

        if (i + 1) % args.display_every == 0:
            print(
                "  step %03d/%03d, iter_time(ms)=%.0f"
                % (i + 1, args.count, elapsed_list[-1] * 1000)
            )

    preds = np.squeeze(preds)

    print("\nInference result:")
    top_k = preds.argsort()[-5:][::-1]
    for i in top_k:
        print("  class={} ; probability={:08.6f}".format(labels[i], preds[i]))

    if args.count > 1:
        print("\nBenchmark result:")
        results = {}
        iter_times = np.array(elapsed_list)
        results["total_time"] = np.sum(iter_times)
        iter_times = iter_times[args.num_warmup_iterations :]
        results["images_per_sec"] = np.mean(1 / iter_times)
        results["99th_percentile"] = (
            np.percentile(iter_times, q=99, method="lower") * 1000
        )
        results["latency_mean"] = np.mean(iter_times) * 1000
        results["latency_median"] = np.median(iter_times) * 1000
        results["latency_min"] = np.min(iter_times) * 1000

        print("  images/sec: %d" % results["images_per_sec"])
        print("  99th_percentile(ms): %.2f" % results["99th_percentile"])
        print("  total_time(s): %.1f" % results["total_time"])
        print("  latency_mean(ms): %.2f" % results["latency_mean"])
        print("  latency_median(ms): %.2f" % results["latency_median"])
        print("  latency_min(ms): %.2f" % results["latency_min"])


if __name__ == "__main__":
    main()
