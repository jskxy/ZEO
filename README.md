# Enhancing Rare Entity Understanding for Pre-trained Language Models via Generative and Unconventional Contrastive Joint Learning

 The application of Pre-trained Language Models (PLMs) in specific domains has recently become increasingly signifi- cant. These models, however, cannot capture the semantics of rare entities of the military equipment domain limited by the heavy-tailed data distribution. This paper proposes to inject the semantics of rare domain entities for PLMs via generative and unconventional contrastive joint learning. Different from general contrastive learning, the positive and negative samples of the triplet training instances do not participate in training but serve as guides for PLMs to adjust model parameters related to the anchor. Following the method, we trained a military equipment-oriented PLM. Due to the absence of testing data, we construct four domain knowledge-required Natural Language Processing (NLP) tasks to evaluate our model. Experimental results illustrated that our model outperforms the baselines by a large margin in all metrics across the tasks, which proves the effectiveness of domain entity injection.

The directory tree of ZEO:
```
ZEO
├── Domain test data
│   │   ├── data_for_mul
│   │   ├── data_for_ner
│   │   ├── data_for_qa
│   │   ├── data_for_rel
│   │   ├── data_for_sum
├── Trainning
│   ├── megatron
│   ├── pretrain.py
├── examples
├── test
├── tools
```

## Requirement
Software:
```
Python3
torch 
apex
megatron
```

## Preprocess Data
* Preproce data by create_pre_train_data.py
* Generate knowledge vector by generate_knowledge_vector.py

Create Training Instance:
```
python tools/create_pre_train_data.py \
       --train_corpus pretrain_corpus.txt \
       --output_file training_data_inject.json \
       --tem_pool_path tem_inject_data/ \
       --metrics_file metrics_inject_data.json \
       --neg_entity 1 \
       --inject 1 \
```

## Knowledge Injection

* Ner Joint Training:
```
#! /bin/bash
#! /bin/bash

python pretrain.py \
       --model-parallel-size 1 \
       --num-layers 23 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 1 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --lr-decay-iters 20000 \
       --save  \
       --load  \
       --vocab-file /megatron/tokenizer/bpe_4w_pcl/vocab \
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
       --training_data_file training_data_inject.json \
       --metrics_file metrics_inject_data.json \
       --max_ngram_in_sequence 10 \
       --neg_entity 1 \
       --inject 2 \
```

## Fine-Tune

* Fine-tune a model by:
```
#! /bin/bash
#! /bin/bash
python pretrain.py \
       --model-parallel-size 1 \
       --num-layers 23 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 1 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --lr-decay-iters 3000 \
       --save  \
       --load  \
       --vocab-file /megatron/tokenizer/bpe_4w_pcl/vocab \
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
       --training_data_file training_data_ner.json \
       --metrics_file metrics_ner_ft_data.json \
       --inject 0
```

## Test

* Testing code is in ./test


## Reference
