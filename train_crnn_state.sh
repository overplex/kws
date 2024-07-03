#!/bin/bash

KWS_PATH=$PWD
DATA_PATH=$KWS_PATH/data
MODELS_PATH=$KWS_PATH/models
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"
TF_ENABLE_ONEDNN_OPTS=1

export CUDA_VISIBLE_DEVICES=''
#PATH="/usr/local/cuda/bin:$PATH"

rm -rf $MODELS_PATH

$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/crnn_state/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 20000,20000,20000,20000 \
--learning_rate 0.001,0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 40 \
--dct_num_features 20 \
--resample 0.15 \
--alsologtostderr \
--wanted_words sheila,marvin \
--train 1 \
--lr_schedule 'exp' \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 10 \
--frequency_masks_number 2 \
--frequency_mask_max_size 5 \
crnn \
--cnn_filters '16,16' \
--cnn_kernel_size '(3,3),(5,3)' \
--cnn_act "'relu','relu'" \
--cnn_dilation_rate '(1,1),(1,1)' \
--cnn_strides '(1,1),(1,1)' \
--gru_units 256 \
--return_sequences 0 \
--dropout1 0.1 \
--units1 '128,256' \
--act1 "'linear','relu'" \
--stateful 1