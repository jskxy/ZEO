#! /bin/bash

<<<<<<< HEAD
python pretrain.py \
=======
python pretrain_gpt2.py \
>>>>>>> ae0c66fcdb5d0fc0a7cc4b6007e0871da46ca9a6
       --model-parallel-size 1 \
       --num-layers 23 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 2 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --lr-decay-iters 3000 \
       --save ./350m_sum_with_con \
       --load ./350m_with_con \
       --data-path megatron/dataset/text_document \
       --vocab-file megatron/tokenizer/bpe_4w_pcl/vocab \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --log-interval 100 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --attention-dropout 0.1 \
       --hidden-dropout 0.1 \
       --reset-attention-mask \
       --finetune \
       --pregenerated_data  \
       --training_data_file training_data_sum.json \
       --metrics_file metrics_sum_ft_data.json \
       --inject 0
