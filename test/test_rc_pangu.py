import os
import sys
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import precision_score, recall_score,f1_score
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
    group.add_argument("--test_data", type=str, default='',
                       help='the path of test data.')
    group.add_argument("--inject", type=int, default=0,
                       help='the path of test data.')
    return parser


def generate(model, context_tokens, args, tokenizer, max_num=50):

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
        for i in range(1000):
            target_index = np.random.choice(len(p), p=p)
            if p_args[target_index] != tokenizer.unk:
                break

        if p_args[target_index] == tokenizer.eod or p_args[target_index] == tokenizer.pad_id \
                or valid_length == args.seq_length-1 or cnt>=max_num:
            outputs = tokens.cpu().numpy()
            break


        tokens[0][valid_length] = p_args[target_index]
        valid_length += 1
        cnt += 1
    # print('outputs id ==', outputs)
    length = np.sum(outputs != tokenizer.pad_id)
    outputs = outputs[0][:length]

    return outputs


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
    mapping = {'搭载': 0, '协作': 1, '研发': 2, '隶属': 3, '优于': 4,
               '对抗': 5, '前型': 6, '侦搜系统': 7, '装备': 8, '其他': 9}

    label_type = len(mapping)
    frame = pd.read_csv(test_data)
    total = len(frame)
    num = int(total / label_type)
    print('total', total)
    y_predict = np.zeros((total, label_type), dtype=int)
    label = np.eye(label_type, dtype=int)
    y_label = label.repeat(num, axis=0)
    print(y_label[31], y_label[32])
    for idx, sample in frame.iterrows():
        raw_text = sample['corpus']
        tokenizer = get_tokenizer()
        context_tokens = tokenizer.tokenize(raw_text,out_type=int)
        output_ids = generate(model, context_tokens, args, tokenizer)
        output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
        output = output_samples[len(raw_text):]
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('Input is:', raw_text, '\n')

        output = output.strip()
        output = output.strip('<eod>')
        output = output.strip('是')
        output = output.replace('[', '')
        output = output.replace(']', '')
        output = output.replace('，', '')
        output = output.replace('。', '')
        print('Output is:', output, '\n')
        print('Reference is:', sample['ans'], '\n')
        if output in mapping:
            y_index = mapping[output]
            y_predict[idx, y_index] = 1
    print('y_predict', y_predict)
    print('the macro precision = ', precision_score(y_label, y_predict, average='macro'))
    print('the macro recall = ', recall_score(y_label, y_predict, average='macro'))
    print('the macro f1 = ', f1_score(y_label, y_predict, average='macro'))

if __name__ == "__main__":

    main()
