#! /bin/bash

python tools/create_pre_train_data.py \
       --train_corpus pretrain_corpus.txt \
       --output_file training_data_inject.json \
       --tem_pool_path tem_inject_data/ \
       --metrics_file metrics_inject_data.json \
       --neg_entity 1 \
       --inject 1 \
