decent_q dump --input_frozen_graph quantize_model/quantize_eval_model.pb \
              --input_fn resnet_v1_50_input_fn.dump_input \
              --output_dir=dump_gpu \

