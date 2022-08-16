# A Chinese equipment-oriented pretrained language model

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
Statistical results on downstream tasks

| Dataset | FT     | Validation | Test   | Length    | Classes     |
| :-----  | :----: | :----:     | :----: | :----:    | :----:      |
| ET      | 1024   | 128        | 128    | 74        | 8           |
| RC      | 1280   | 160        | 160    | 137       | 10          |
| QA      | 960    | 120        | 120    | 876       | \           |
| Sum     | 1208   | 151        | 151    | 958       | \           |

Preprocess Data:
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
       --batch-size 2 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --lr-decay-iters 20000 \
       --save ./350m_1neg/ \
       --load ./pangu-alpha_350m_fp16/ \
       --data-path /megatron/dataset/text_document \
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
       --batch-size 2 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --lr-decay-iters 3000 \
       --save ./350m_ner_inject_model \
       --load ./350m_1neg \
       --data-path /megatron/dataset/text_document \
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

## Result

* Experiment results on entity typing.

| Models | Precision   | Recall  | F1  |
| :----: | :----: | :----:| :----:|
| Pangu-α<sub>350M</sub> | 88.29 | 86.33 | 86.37 |
| Pangu-α<sub>2.6B</sub> | 91.86 | 88.28 | 88.01 |
| ZEO    | 92.09 | 91.40 | 91.24 |

* Experiment results on relation classification.

| Models | Precision   | Recall  | F1  |
| :----: | :----: | :----:| :----:|
| Pangu-α<sub>350M</sub> | 65.19 | 68.13 | 65.36 |
| Pangu-α<sub>2.6B</sub> | 52.45 | 45.31 | 42.96 |
| ZEO    | 83.54 | 78.44 | 77.35 |

* Experiment results on question answering.

| Models | Pangu-α+ $\mathcal{L}$ <sub>gen</sub>   | Pangu-α+ $\mathcal{L}$ <sub>con</sub> | ZEO  |
| :----: | :----:   | :----:| :----: |
| Acc    | 43.64    | 23.65 | 58.62  |    

* Experiment results on summarization.

| Models | RG-1   | RG-2  | RG-L  |
| :----: | :----: | :----:| :----:|
| Pangu-α+ $\mathcal{L}$ <sub>gen</sub> | 37.17  | 23.56 | 33.35 |
| Pangu-α+ $\mathcal{L}$ <sub>con</sub> | 30.86  | 17.36 | 28.28 |
| ZEO    | 42.51  | 29.22 | 38.35 |

* Experiment results on how different pre-training tasks influence ZEO’s performance on downstream tasks. We also show the impacts of multi-learning strategy and denote our model trained in multi-learning setting with ZEO-multi.

| Model     | P(ET)  | R(ET)  | F1(ET) | Acc    | P(RC)  | R(RC)  | F1(RC) | RG-1  | RG-2  | RG-L  |
| :-----    | :----: | :----: | :----- | :---:  | :----: | :----: | :----: | :----:| :----:| :----:|
| Pangu-α+ $\mathcal{L}$ <sub>gen</sub>   | 89.19  | 87.50  | 86.99  | 46.31  | 79.21  | 74.69  | 72.00  | 40.75 | 27.37 | 36.93 |
| Pangu-α+ $\mathcal{L}$ <sub>gen</sub>    | 63.00  | 64.45  | 58.70  |  \     | 12.20  | 20.94  | 13.91  | 10.12 | 3.86  | 9.67  |
| ZEO       | 92.09  | 91.40  | 91.24  | 58.62  | 83.54  | 78.44  | 77.35  | 42.51 | 29.21 | 38.35 |
| ZEO-multi | 94.94  | 94.53  | 94.59  | 57.64  | 80.69  | 79.69  | 79.23  | 43.38 | 30.42 | 39.59 |  

## Reference
