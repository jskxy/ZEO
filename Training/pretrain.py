"""Pretrain ZEO"""
import json

import torch
import numpy as np
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt2_dataset import build_train_valid_test_datasets
from megatron.model import GPT2Model
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import reduce_losses
from pathlib import Path

def model_provider():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_tokentypes=0, parallel_output=True)

    return model

def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()
    key1 = ['tokens']
    key2 = ['contras_embedding']
    key3 = ['entity_mapping']
    datatype1 = torch.int64
    datatype2 = torch.float32
    if args.inject == 1:
        data, contras_embedding,entity_mapping = data_iterator
        contras_embedding = mpu.broadcast_data(key2, contras_embedding, datatype2)
        # print('contras_embedding shape is', contras_embedding.shape, '==\n')
        entity_mapping = mpu.broadcast_data(key3, entity_mapping, datatype2)
        # print('entity_mapping shape is', entity_mapping.shape, '==\n')
    else:
        data = data_iterator
        contras_embedding = None
        entity_mapping = None

    data = mpu.broadcast_data(key1, data, datatype1)
    tokens_ = data.long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids, contras_embedding, entity_mapping


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()
    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids, contras_embedding, entity_mapping = get_batch(
        data_iterator)
    timers('batch generator').stop()
    # Forward model.
    losses = model(tokens, position_ids, attention_mask, contras_embedding, entity_mapping, labels=labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    reduced_loss = reduce_losses([loss])
    return loss, {'lm loss': reduced_loss[0]}


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets ')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating datasets ...")

    return train_ds, valid_ds, test_ds
def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--inject", type=int,
                       default=2, help='whether inject knowledge.')
    group.add_argument("--pregenerated_data", type=Path, default='',
                       help='Sampling temperature.')
    group.add_argument("--max_ngram_in_sequence", type=int, default=10,
                       help='aligned entities or relations of each sentence.')
    group.add_argument("--neg_entity", type=int, default=1,
                       help='number for contrastive learning.')
    group.add_argument('--embedding_path',type=Path, default='',
                       help='Sampling temperature.')
    group.add_argument('--training_data_file', type=str,default='',
                       help='the name of file.')
    group.add_argument("--metrics_file",type=str, default='', help='the name of metric file')
    return parser

if __name__ == "__main__":

    pretrain(model_provider, forward_step, extra_args_provider=add_text_generate_args,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
