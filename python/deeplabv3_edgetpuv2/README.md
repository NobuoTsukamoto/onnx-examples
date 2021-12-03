# ONNX DeepLab v3+ EdgeTPUV2 and AutoSeg EdgeTPU

## Description
This sample contains code that convert TensorFlow Hub DeepLab v3+ EdgeTPUV2 and AutoSeg EdgeTPU model to ONNX model and performs inference.
1. Convert TensorFlow Lite model from TensorFlow Hub model.
2. Convert TensorFlow Lite model to ONNX Model.
4. Inference.

## Reference
- [autoseg-edgetpu](https://tfhub.dev/google/collections/autoseg-edgetpu/1)
- [deeplab-edgetpu](https://tfhub.dev/google/collections/deeplab-edgetpu/1)
- [ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)
- [tf2onnx](https://github.com/onnx/tensorflow-onnx)


## Convert ONNX Model on your Google Colab
[Convert DeepLab v3+ EdgeTPUv2 TF-Hub model to ONNX model Notebook](https://github.com/NobuoTsukamoto/tensorrt-examples/blob/main/python/deeplabv3_edgetpuv2/convert_deeplabv3_edgetpuv2_tfhub2onnx.ipynb) contains all the steps to convert from TensorFlow Hub model to the ONNX model.  
Run it on Google Colab and download the converted ONNX model. Of course, you can also run it on your own host PC.

## Run ONNX

### Install dependency
Install onnxruntime.  
```
pip3 install onnxruntime
pip3 install opencv-python
```

### Clone this repository.
Clone repository.
```
cd ~
git clone https://github.com/NobuoTsukamoto/onnx-examples.git
python/deeplabv3_edgetpuv2
```

### Finally you can run the demo.
```
python3 onnx_deeplabv3_edgetpuv2_image.py 
usage: onnx_deeplabv3_edgetpuv2_image.py [-h] --model MODEL
                                         [--input_shape INPUT_SHAPE] --input
                                         INPUT --output OUTPUT
onnx_deeplabv3_edgetpuv2_image.py: error: the following arguments are required: --model, --input, --output
```