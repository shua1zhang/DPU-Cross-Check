# Caffe Workflow 

## 1. Prerequisite 
### Host environment 

Please setup the environment according to Chapter 1 of [UG1327](https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf).

### Board environment

For avaliable Xilinx evalution boards, please make sure board image and DNNDK are correctly installed and configured according to Chapter 1 of [UG1327](https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf).

For custom FPGA platform, please make sure DPU and DNDNK are correctly implemented based on [DPU TRD](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=zcu102-dpu-trd-2019-1-190809.zip). 

Related files can be downloaded in [Xilinx AI Developer Hub](https://www.xilinx.com/products/design-tools/ai-inference/ai-developer-hub.html#edge).

### Tool

This tutorial requires DECENT_Q full version and assumes that it is renamed to `decent_q_full` and placed under `/usr/local/bin/decent_q_full` in the system. Please contact shuaizh@xilinx.com for tools. 

### Model

Resnet50 from [Xilinx Model Zoo](https://github.com/Xilinx/AI-Model-Zoo) is used in this tutorial. The float model is already placed in `GPU-DPU-cross-check/caffe_resnet50/float_model/` and complete ResNet50 package can be downloaded in [here](https://www.xilinx.com/bin/public/openDownload?filename=cf_resnet50_imagenet_224_224_7.7G.zip).

## 2. Generate Quantized Inference Model and Reference Result

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

![Caffe quantize result](https://github.com/shua1zhang/GPU-DPU-cross-check/blob/master/doc/pic/caffe_quantize.PNG)

The deploy.prototxt and deploy.caffemodel will be used to generate DPU elf file while the quantize_train_test.prototxt and quantize_train_test.caffemodel will be used to generate reference INT8 inference result. 

### Generate reference INT8 inference result
Run script `1_dump.sh` under `/caffe_resnet50` to generate reference data.
```
DECENT_DEBUG=5 decent_q_full test -model quantize_model/quantize_train_test.prototxt \
                                  -weights quantize_model/quantize_train_test.caffemodel \
                                  -test_iter 1 \
                                  2>&1 | tee ./log/dump.log
```
After running the script, folder "dump_gpu" will be generated and partial files are shown as below.

![GPU Reference Result](https://github.com/shua1zhang/GPU-DPU-cross-check/blob/master/doc/pic/GPU_dump.PNG)

## 3. Generate DPU Inference Result 
### Generate DPU elf file 
Run script `2_compile.sh` to generate DPU elf file. Please modify `dcf` parameter according to your board. 
```
dnnc-3.1 --parser=caffe \
         --dcf=../dcf/ZCU102.dcf \
         --prototxt=quantize_model/deploy.prototxt \
         --caffemodel=quantize_model/deploy.caffemodel \
         --cpu_arch=arm64 \
         --output_dir=compile_model \
         --net_name=caffe_resnet50 \
         --mode=debug \
         --save_kernel \
         --dump all
```

![Caffe Complie Result](https://github.com/shua1zhang/GPU-DPU-cross-check/blob/master/doc/pic/caffe_compile.PNG)

Files dumped for analysis purpose are stored in folder `dump` whose contents are as follow: 

![DNNC Dump](https://github.com/shua1zhang/GPU-DPU-cross-check/blob/master/doc/pic/dnnc_dump.PNG)

With internal function of DNNC, the relationship between DPU super layers and actual network layers could be generated as [Caffe ResNet50 Super Layer](https://github.com/shua1zhang/GPU-DPU-cross-check/blob/master/caffe_resnet50/kernel_graph.jpg).

### Generate DPU inference result

Transfer folder `/board_caffe` onto board system and generate executable `caffe-resnet50` by command: 
```
make
``` 

Enable DPU debug mode with DNNDK dexplorer (detailed information is in Chapter 11 of [UG1327](https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf)):
```
dexplorer -m debug
```

Run DPU inference with reference input data `data.txt` by below command and inference result of layers will be save in folder `dump_xxx`, the partial files are shown below. 
```
caffe-resnet50 data.txt
```
![DPU Inference Result](https://github.com/shua1zhang/GPU-DPU-cross-check/blob/master/doc/pic/DPU_dump.PNG)

## 4. Cross Check Inference Result
### Understand layer correspondence between refenecne result and DPU inference result
When DNNC generates elf file, it will conduct several optimization strategies on certain layer conbinations and form super layers so as to get better performance. In order to cross check inference result correctness, [super layer graph](https://github.com/shua1zhang/GPU-DPU-cross-check/blob/master/caffe_resnet50/kernel_graph.jpg) will be used to find correct files to cross check. 

![Beginning of Caffe ResNet50 Super Layer](https://github.com/shua1zhang/GPU-DPU-cross-check/blob/master/doc/pic/Super_layer.PNG).

Take first several layers of caffe resnet shown above as example. The names of DPU super layers are shown on the top of every blocks (e.g, `data, conv1, res2a_branch2a` and `res2a_branch1`) while the names of network layers are shown in every blobs (e.g, `data, conv1, conv1_relu` and `pool1`). 

Super Layer Name| Referenece file            | DPU file
----------------|:---------------------------|:------------
conv1 (input)   | data.bin                   | caffe_resnet50_0_conv1_in0.bin   
conv1 (output)  | pool1.bin                  | caffe_resnet50_0_conv1_out0.bin 
res2a_branch2a (input)  | pool1.bin                   | caffe_resnet50_0_res2a_branch2a_in0.bin
res2a_branch2a (output) | res2a_branch2a_relu.bin     | caffe_resnet50_0_res2a_branch2a_out0.bin
res2a_branch1 (input)   | pool1.bin                   | caffe_resnet50_0_res2a_branch1_in0.bin
res2a_branch1 (input)   | res2a_branch2c.bin          | caffe_resnet50_0_res2a_branch1_in1.bin
res2a_branch1 (output)  | res2a_relu.bin              | caffe_resnet50_0_res2a_branch1_out0.bin

### Cross check refenecne result and DPU inference result

The cross check mechinism is to first make sure input(s) to one super layer is identicial to reference and then the output(s) is identical too, which can be done with command `diff`, `vimdiff`, `cmp` and etc. If two files are identical, command `diff` and `cmp` will return nothing in command line. 

For example as the initial step, the inputs of super layer 'conv1' (i.e., `data.bin` from reference result and `caffe_resnet50_0_conv1_in0.bin` from DPU result) need to be compared and should be exactly same. Then the outputs of super layer 'conv1' (i.e., `pool1.bin` from referenece result and `caffe_resnet50_0_conv1_out0.bin` from DPU result) should be compared and so on. 

For super layers that have multiple input or output (e.g., `res2a_branch1`), input correctness should be checked first and then check output. 

![Caffe Check Result](https://github.com/shua1zhang/GPU-DPU-cross-check/blob/master/doc/pic/caffe_check.PNG)

### Files to submit if cross check fails

If certain super layer proves to be wrong on DPU, please prepare following files as one package for further analysis by factory and send to shuaizh@xilinx.com with detailed descrption.  

- .ir file genereated by DNNC
- super layer input file (.bin) generated by DPU 
- DNNC tag info generated by `dnnc-xxx --tag`

For example, if super layer `res2a_branch1` is not running correctly, following files shall be collected: 

- res2a_branch1.ir
- caffe_resnet50_0_res2a_branch1_in0.bin and caffe_resnet50_0_res2a_branch1_in1.bin
- DNNC tag info 
