decent_q_full quantize -model float_model/float.prototxt \
                       -weights float_model/float.caffemodel \
                       -output_dir quantize_model \
                       2>&1 | tee ./log/quantize.log 

