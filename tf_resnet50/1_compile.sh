dnnc-3.1 --parser=tensorflow \
         --dcf=../dcf/ZCU102.dcf \
         --frozen_pb=quantize_model/deploy_model.pb \
         --cpu_arch=arm64 \
         --output_dir=compile_model \
         --net_name=tf_resnet50 \
         --mode=debug \
         --save_kernel
