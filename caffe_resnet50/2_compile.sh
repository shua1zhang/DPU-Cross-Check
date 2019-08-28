dnnc-3.1 --parser=caffe \
         --dcf=../dcf/ZCU102.dcf \
         --prototxt=quantize_model/deploy.prototxt \
         --caffemodel=quantize_model/deploy.caffemodel \
         --cpu_arch=arm64 \
         --output_dir=compile_model \
         --net_name=caffe_resnet50 \
         --mode=debug \
         --save_kernel
