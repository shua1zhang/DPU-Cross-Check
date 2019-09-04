# TensorFlow Workflow

## 1. Prerequisite 
### Host environment 

Please setup the environment according to Chapter 1 of [UG1327](https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf).

### Board environment

For avaliable Xilinx evalution boards, please make sure board image and DNNDK are correctly installed and configured according to Chapter 1 of [UG1327](https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf).

For custom FPGA platform, please make sure DPU and DNDNK are correctly implemented based on [DPU TRD](https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=zcu102-dpu-trd-2019-1-190809.zip). 

Related files can be downloaded in [Xilinx AI Developer Hub](https://www.xilinx.com/products/design-tools/ai-inference/ai-developer-hub.html#edge).

### Tool

This tutorial requires TensorFlow decent_q released in [DNNDK 3.1](https://www.xilinx.com/member/forms/download/dnndk-eula-xef.html?filename=xilinx_dnndk_v3.1_190809.tar.gz) on Xilinx AI Developer Hub.

### Model

Resnet50 from [Xilinx Model Zoo](https://github.com/Xilinx/AI-Model-Zoo) is used in this tutorial. The pb file is already placed in `GPU-DPU-cross-check/tf_resnet50/quantized_model/` and complete ResNet50 package can be downloaded in [here](https://www.xilinx.com/bin/public/openDownload?filename=tf_resnet50_imagenet_224_224_6.97G.zip).

## 2. Generate Reference Result

Under folder `tf_resnet50`, run `0_dump.sh` to generate reference data.

```
decent_q dump --input_frozen_graph quantize_model/quantize_eval_model.pb \
              --input_fn resnet_v1_50_input_fn.dump_input \
              --output_dir=dump_gpu \
```

After running the script, folder "dump_gpu" will be generated and partial files are shown as below. 

~~~snapshot for reference files~~~

## 3. Generate DPU Inference Result

Run script `1_compile.sh` to genereate DPU elf file. Please modify `dcf` parameter according to your board.

```
dnnc-3.1 --parser=tensorflow \
         --dcf=../dcf/ZCU102.dcf \
         --frozen_pb=quantize_model/deploy_model.pb \
         --cpu_arch=arm64 \
         --output_dir=compile_model \
         --net_name=tf_resnet50 \
         --mode=debug \
         --save_kernel \
         --dump all
```

~~~snapshot for DNNC log~~~~

Files dumped for analysis purpose are stored in folder `dump` whose contents are as follow:

~~~snapshot for DNNC dump files~~~~
