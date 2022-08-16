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

"""Sample Generate GPT2"""

import os
import sys
import numpy as np
import torch
from rouge import Rouge
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))


from megatron.text_generation_utils import pad_batch, get_batch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPT2Model
from megatron.training import get_model
from megatron.text_generation_utils import generate_and_write_samples_unconditional
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron.text_generation_utils import generate_samples_interactive

# from megatron.model.transformer import LayerNorm


def model_provider():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_tokentypes=0, parallel_output=False)

    return model


def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=5,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--sample-input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--sample-output-file", type=str, default=None,
                       help='Output file got from --sample-input-file')
    group.add_argument("--num-samples", type=int, default=0,
                       help='Number of samples to generate unconditionally, '
                       'defaults to 0 and interactive conditional sampling')
    group.add_argument("--genfile", type=str,
                       help='Output file when generating unconditionally')
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                       'instead of using previously computed keys/values.')
    group.add_argument("--inject", type=bool, default=False,
                       help='inject knowledge during training.')
    group.add_argument("--test_data", type=str, default='',
                       help='the path of test data.')

    return parser


def generate(model, context_tokens, args, tokenizer, max_num=100):

    valid_length = len(context_tokens)
    context_tokens_, context_lengths = pad_batch([context_tokens],
                                                 tokenizer.pad_id, args)
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens_)
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)
    type_ids = None
    bs,_  = tokens.shape
    cnt = 0
    # print('the input tokens =', tokens)
    while valid_length < args.seq_length:
        with torch.no_grad():
            logits = model(tokens,
                           position_ids,
                           attention_mask,
                           tokentype_ids=type_ids,
                           forward_method_parallel_output=False)
        logits = logits[:,:,:tokenizer.vocab_size].cpu()
        logits = logits.numpy()
        logits = logits.reshape(bs, args.seq_length, -1)
        probs = logits[0, valid_length-1, :]
        p_args = probs.argsort()[::-1][:args.top_k]

        p = probs[p_args]
        p = p / sum(p)
        # choose the not unk word from top_k
        for i in range(1000):
            target_index = np.random.choice(len(p), p=p)
            if p_args[target_index] != tokenizer.unk:
                break

        special_tokens = [tokenizer.eod, tokenizer.pad_id]
        # if valid_length == args.seq_length - 1 or cnt >= max_num:
        if p_args[target_index] in special_tokens or valid_length == args.seq_length-1 or cnt>=max_num:
            outputs = tokens.cpu().numpy()
            break
        tokens[0][valid_length] = p_args[target_index]
        valid_length += 1
        cnt += 1
    # print('outputs id ==', outputs)
    length = np.sum(outputs != tokenizer.pad_id)
    outputs = outputs[0][:length]

    return outputs
def clean_sent(sent):
    output = sent.strip()

    output = re.sub(r'[\[“ ”,，。？！\]]', '', output)
    output = output.replace(" ", '')
    # output = output.split(',')
    # output = list(set(output))
    # output = ''.join(output)
    output = ' '.join(list(output))

    return output

def main():
    """Main program."""
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    # Set up model and load checkpoint.
    model = get_model(model_provider)
    model.eval()

    args = get_args()
    test_data = args.test_data
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    import pandas as pd
    # a = [['ab'], ['a1 b']]
    # b = [['ab'], ['a2 b']]
    # all_score = []
    # evaluator = Rouge()
    # for i in range(len(a)):
    #     rouge_score = evaluator.get_scores(hyps=a[i], refs=b[i])
    #     all_score.append(rouge_score)
    # for item in all_score:
    #     print(item[0]['rouge-1']['r'])


    frame = pd.read_csv(test_data)
    total_num = 0
    all_score = []
    for index, sample in frame.iterrows():
        raw_text = sample['content']
        tokenizer = get_tokenizer()
        context_tokens = tokenizer.tokenize(raw_text, out_type=int)
        output_ids = generate(model, context_tokens, args, tokenizer)
        output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
        output = output_samples[len(raw_text):]

        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('Input is:', raw_text, '\n')
        print('output is:', output, '\n')
        output = clean_sent(output)
        if output == '':
            output = '<eod>'
        print('reference is:', sample['ans'], '\n')
        ans = re.sub(r'[\[“ ” ,。？！!\]]', '', sample['ans'])
        ans = ' '.join(list(ans))
        evaluator = Rouge()
        rouge_score = evaluator.get_scores(hyps=output, refs=ans)
        all_score.append(rouge_score)

        total_num += 1
    rouge1_r = sum([score[0]['rouge-1']['r'] for score in all_score]) / total_num
    rouge2_r = sum([score[0]['rouge-2']['r'] for score in all_score]) / total_num
    rougeL_r = sum([score[0]['rouge-l']['r'] for score in all_score]) / total_num
    rouge1_p = sum([score[0]['rouge-1']['p'] for score in all_score]) / total_num
    rouge2_p = sum([score[0]['rouge-2']['p'] for score in all_score]) / total_num
    rougeL_p = sum([score[0]['rouge-l']['p'] for score in all_score]) / total_num
    rouge1_f = sum([score[0]['rouge-1']['f'] for score in all_score]) / total_num
    rouge2_f = sum([score[0]['rouge-2']['f'] for score in all_score]) / total_num
    rougeL_f = sum([score[0]['rouge-l']['f'] for score in all_score]) / total_num

    print('rouge1_r = ', rouge1_r, '\t', 'rouge2_r = ', rouge2_r, '\t', 'rougeL_r = ', rougeL_r)
    print('rouge1_p = ', rouge1_p, '\t', 'rouge2_p = ', rouge2_p, '\t', 'rougeL_p = ', rougeL_p)
    print('rouge1_f = ', rouge1_f, '\t', 'rouge2_f = ', rouge2_f, '\t', 'rougeL_f = ', rougeL_f)



if __name__ == "__main__":

    main()
