# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT-2 model."""

import torch
import torch.nn as nn
from megatron import get_args
from megatron import mpu
from megatron.module import MegatronModule
from megatron.mpu import LayerNorm
from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal
from pytorch_metric_learning.losses import NTXentLoss



def gpt2_attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores


class GPT2Model(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self, num_tokentypes=0, parallel_output=True):
        super(GPT2Model, self).__init__()
        args = get_args()
        self.inject = args.inject

        self.parallel_output = parallel_output
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=gpt2_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers))
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon).cuda()


    # motif
    def forward(self, input_ids, position_ids, attention_mask=None, contras_embedding=None,entity_mapping=None,labels=None, encode=False,
                tokentype_ids=None, layer_past=None, get_key_value=False,
                forward_method_parallel_output=None ):

        #<<<<<<<<<<<<<<<<<<<<<<<<<<<   debug mp <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # import numpy as np
        # from megatron import print_rank_0
        # from megatron import get_tokenizer
        # from megatron.text_generation_utils import pad_batch, get_batch
        # print_rank_0(input_ids.shape)
        # tokenizer = get_tokenizer()
        # input_ids = [39,2401,17,4448,615,5188,2783,2762,10,2371,5720,11050,3373,24420,1613]
        # input_len = len(input_ids)
        # input_ids.extend([6] * (1024 - len(input_ids)))
        # input_ids = torch.Tensor(np.atleast_2d(input_ids)).long().cuda()
        # context_tokens_tensor = torch.cuda.LongTensor(input_ids)
        # tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)
        # print_rank_0('='*150)
        # print_rank_0('='*150)
        # print_rank_0('input id is: ')
        # for ids in input_ids.cpu().numpy().tolist():
        #     print_rank_0(ids)
        #     # print_rank_0(tokenizer.convert_ids_to_tokens(ids))
        # print_rank_0('*'*50)
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # Language model.
        # motif

        lm_output = self.language_model(input_ids,
                                        position_ids,
                                        attention_mask,
                                        tokentype_ids=tokentype_ids,
                                        layer_past=layer_past,
                                        get_key_value=get_key_value)

        #<<<<<<<<<<<<<<<<<<<<<<<<<<<   debug mp <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # with torch.no_grad():
        #     logits = lm_output.cpu().numpy()
        # index = np.argmax(logits, axis=2)
        # ids_ = index.tolist()
        # for ids in ids_:
        #     print(f"rank is {torch.distributed.get_rank()}")
        #     print(f"next id of input is {ids[input_len]}")
        #     print(f'out put id is: {ids}')
        #     # print_rank_0(tokenizer.convert_ids_to_tokens(ids))
        # raise ValueError("stop")
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        if get_key_value:
            lm_output, presents = lm_output
        if encode is True:
            lm_output = self.input_layernorm(lm_output)
            return lm_output
        lm_output = torch.add(lm_output,0)
        # print('lm_output shape is',lm_output.shape)
        # Output.
        parallel_output = self.parallel_output
        if forward_method_parallel_output is not None:
            parallel_output = forward_method_parallel_output
        output = parallel_lm_logits(
            lm_output,
            self.language_model.embedding.word_embeddings.weight,
            parallel_output)
        # print('output.shape=', output.shape)

        if get_key_value:
            output = [output, presents]

        if labels is None:
            return output
        else:
            # print('self.fp16_lm_cross_entropy =', self.fp16_lm_cross_entropy)
            if self.fp16_lm_cross_entropy:
                assert output.dtype == torch.half
                loss1 = mpu.vocab_parallel_cross_entropy(output.float(), labels)
                if self.inject == 0:
                    return loss1
                else:
                    print('== injecting ==')
                    entity = torch.matmul(entity_mapping, lm_output)
                    loss2 = 0
                    contras_embedding[:, :, 0] = entity
                    a = contras_embedding.shape[0]
                    b = contras_embedding.shape[1]
                    c = contras_embedding.shape[2]
                    # print('contras_embedding.shape[-2] = ', c)
                    labels = torch.arange(c)
                    new_labels = torch.empty(a, b, c)
                    new_labels[:, :] = labels
                    new_labels[:, :, 1] = new_labels[:, :, 0]
                    loss2 = NTXentLoss(contras_embedding, new_labels)
                    return loss1 + loss2
            else:
                if self.inject == 0:
                    # only lm
                    loss1 = mpu.vocab_parallel_cross_entropy(output.float(), labels)
                    return loss1
                elif self.inject == 1:
                    print('== only cl ==')
                    entity_mapping = entity_mapping[:,:,1:]
                    entity = torch.matmul(entity_mapping, lm_output)
                    pos = contras_embedding[:, :, 1]
                    neg = contras_embedding[:, :, 2]
                    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
                    loss2 = triplet_loss(entity, pos, neg)
                    return loss2
                else:
                    print('== injecting ==')
                    loss1 = mpu.vocab_parallel_cross_entropy(output.float(), labels)
                    entity_mapping = entity_mapping[:, :, 1:]
                    entity = torch.matmul(entity_mapping, lm_output)
                    pos = contras_embedding[:, :, 1]
                    neg = contras_embedding[:, :, 2]
                    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
                    loss2 = triplet_loss(entity, pos, neg)
                    return loss1 + loss2



    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)
