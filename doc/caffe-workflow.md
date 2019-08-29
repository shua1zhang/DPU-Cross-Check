# Caffe Workflow 

## 1. Prerequisite 
### Host environment 

Please setup the environment according to Chapter 1 of [UG1327](https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf).

### Board environment

For avaliable Xilinx evalution boards, please make sure board image and DNNDK are correctly installed and configured.

For custom FPGA platform, please make sure DPU and DNDNK are correctly implemented based on DPU TRD. 

Related files can be downloaded in [Xilinx AI Developer Hub](https://www.xilinx.com/products/design-tools/ai-inference/ai-developer-hub.html#edge).

### Tool

This tutorial requires DECENT_Q full version and assumes that it is renamed to `decent_q_full` and placed under `/usr/local/bin/decent_q_full` in the system. Please contact shuaizh@xilinx.com for tools. 

### Model

Resnet50 from [Xilinx Model Zoo](https://github.com/Xilinx/AI-Model-Zoo) is used in this tutorial. The float model is already placed in `GPU-DPU-cross-check/caffe_resnet50/float_model/` and complete ResNet50 package can be downloaded in [download link](https://www.xilinx.com/bin/public/openDownload?filename=resnet50_20190528.zip).

## 2. Generate Quantized Inference Model

### Add ImageData layer into prototxt

In newly download Resnet float.prototxt, it contains only simple input definiation shown as below: 
```
input: "data"
input_shape {
  dim: 1
  dim: 3 
  dim: 224
  dim: 224
}
```

In order to use image files in calibration and test, ImageData layer (both TRAIN phase and TEST phase) needs to be added to prototxt. Please modify path for "**source**" and "**root_folder**" according to environment. 
```
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 224
    mean_value: 104
    mean_value: 107
    mean_value: 123
  }
  image_data_param {
    source: "/PATH_TO/GPU-DPU-cross-check/images/image500/caffe_calib.txt"
    root_folder: "/PATH_TO/GPU-DPU-cross-check/images/image500/"
    batch_size: 20 #Modify according to GPU memory
    shuffle: true
  }
}
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 224
    mean_value: 104
    mean_value: 107
    mean_value: 123
  }
  image_data_param {
    source: "/PATH_TO/GPU-DPU-cross-check/images/image224/caffe_dump.txt"
    root_folder: "/PATH_TO/GPU-DPU-cross-check/images/image224/"
    batch_size: 1
    shuffle: false 
  }
}
```

### Quantize float model

Run script `0_quantize.sh` under `/caffe_resnet50` to quantize ResNet50.
```
decent_q_full quantize -model float_model/float.prototxt \
                       -weights float_model/float.caffemodel \
                       -output_dir quantize_model \
                       2>&1 | tee ./log/quantize.log 
```


Following files will be generated under `/caffe_resnet50/quantize_model/`: 
+ deploy.prototxt
+ deploy.caffemodel
+ quantize_train_test.prototxt
+ quantize_train_test.caffemodel

The deploy.prototxt and deploy.caffemodel will be used to generate DPU elf file while the quantize_train_test.prototxt and quantize_train_test.caffemodel will be used to generate reference INT8 inference result. 

### Generate reference INT8 inference result
Run script `1_dump.sh` under `/caffe_resnet50` to generate reference data.
```
DECENT_DEBUG=5 decent_q_full test -model quantize_model/quantize_train_test.prototxt \
                                  -weights quantize_model/quantize_train_test.caffemodel \
                                  -test_iter 1 \
                                  2>&1 | tee ./log/dump.log
```


### Generate DPU elf file 
Run script "2_compile.sh" to generate DPU elf file. Please modify `dcf` parameter according to your board. 
```
dnnc-3.1 --parser=caffe \
         --dcf=../dcf/ZCU102.dcf \
         --prototxt=quantize_model/deploy.prototxt \
         --caffemodel=quantize_model/deploy.caffemodel \
         --cpu_arch=arm64 \
         --output_dir=compile_model \
         --net_name=caffe_resnet50 \
         --mode=debug \
         --save_kernel
```

### Generate DPU inference result

Transfer folder `/board_caffe` onto board system and generate executable `caffe-resnet50` by command: 
```
make
``` 

Enable DPU debug mode with DNNDK dexplorer (detailed information is in Chapter 11 of [UG1327](https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf)):
```
dexplorer -m debug
```

Run DPU inference with reference input data `data.txt` by below command and inference result of layers will be save in folder `dump_xxx`.
```
caffe-resnet50 data.txt
```
