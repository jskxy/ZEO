# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for OpenAI GPT."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import json
import logging
import os
import regex as re
from io import open
import sentencepiece as spm
import jieba

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func


class JIEBATokenizer(object):

    def __init__(self, vocab_file, max_len=None):
        self.max_len = max_len if max_len is not None else int(1e12)
        # self.encoder = json.load(open(vocab_file))
        model_file = vocab_file + ".model"
        vocab_file = vocab_file + ".vocab"
        f = open(vocab_file,'r', encoding='utf-8')
        lines = f.readlines()
        self.encoder = {}
        for line in enumerate(lines):
            #print (line[0] = index)
            key = line[1].split('\t')[0]
            self.encoder[key] = line[0]


        self.decoder = {v:k for k,v in self.encoder.items()}

        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.translator = str.maketrans(" \n", "\u2582\u2583")

        self.eod_id = self.encoder['<eot>']
        self.eot_id = self.encoder['<eot>']
        self.pad_id = self.encoder['<pad>']
        self.blank_id = self.encoder['▁']
        self.unk = 0


    @property
    def vocab_size(self):
        return len(self.encoder)

    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    @property
    def eod(self):
        return self.eot_id

    def tokenize(self, text,out_type):
        """ Tokenize a string. """
        seg_list = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        new_seg = " ".join(seg_list)
        # print("\n new_seg is", self.sp.encode(new_seg,out_type), '\n')
        print("new_seg is",new_seg)
        new_dict = {'主战坦克': len(self.encoder)}
        self.encoder.update(new_dict)
        return self.sp.encode(new_seg,out_type)

    def convert_tokens_to_ids(self, tokens):
        return self.sp.PieceToId(tokens)

    def convert_ids_to_tokens(self, ids):
        return self.decode(ids)

    def detokenize(self, token_ids):
        return self.decode(token_ids)

    def encode(self, text,out_type):
        res = self.tokenize(text,out_type)
        return res

    def decode(self, tokens):
        text = self.sp.decode(tokens)
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text

if __name__ == '__main__':
    ids = [26,   80,  373, 5440, 2527, 1663,   11, 1268, 4896, 7902]
    vocab_file = '/home/ferry/knowledge_injection/panguAlpha_pytorch/megatron/tokenizer/bpe_4w_pcl/vocab'
    tokenizer = JIEBATokenizer(vocab_file)
    # print('the encoder contains',tokenizer.decoder.values())
    jieba.add_word('主战坦克')
    test_id = tokenizer.encode("M1A1主战坦克是M1主战坦克的第一种大规模改良型",out_type=str)
    print(test_id)

    # print([tokenizer.convert_tokens_to_ids(i) for i in test_id])
    test_id = tokenizer.encode("M1A1主战坦克是M1主战坦克的第一种大规模改良型", out_type=int)
    print(test_id)
    # print(tokenizer.convert_ids_to_tokens(test_id))
    pass
