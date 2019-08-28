DECENT_DEBUG=5 decent_q_full test -model quantize_model/quantize_train_test.prototxt \
                                  -weights quantize_model/quantize_train_test.caffemodel \
                                  -test_iter 1 \
                                  2>&1 | tee ./log/dump.log



rm update* 
