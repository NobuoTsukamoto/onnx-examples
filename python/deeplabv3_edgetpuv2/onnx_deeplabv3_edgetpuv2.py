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
    parser.add_argument(
        "--videopath", help="File path of input video file.", default=None, type=str
    )
    parser.add_argument("--output", help="File path of output image.", type=str)
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 50)

    # Initialize colormap
    colormap = create_label_colormap()

    # Video capture.
    if args.videopath == "":
        print("open camera.")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    else:
        print("open video file", args.videopath)
        cap = cv2.VideoCapture(args.videopath)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Input Video (height, width, fps): ", h, w, fps)

    # Load model.
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    input_shape = tuple(map(int, args.input_shape.split(",")))
    session = onnxruntime.InferenceSession(args.model)

    # Output Video file
    # Define the codec and create VideoWriter object
    video_writer = None
    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    elapsed_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("VideoCapture read return false.")
            break

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

        # Calc fps.
        elapsed_list.append(inference_time)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)

        # Display fps
        fps_text = "Inference: {0:.2f}ms".format(inference_time)
        display_text = model_name + " " + fps_text + avg_text
        draw_caption(im, (10, 30), display_text)

        # Output video file
        if video_writer is not None:
            video_writer.write(im)

        # Display
        cv2.imshow(WINDOW_NAME, im)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break

    # When everything done, release the window
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
