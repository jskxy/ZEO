import json
import os
import sys
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import precision_score, recall_score,f1_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))


from megatron.text_generation_utils import pad_batch
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
from megatron.finetune_ner_pangu import GPT2ModelForNer

# from megatron.model.transformer import LayerNorm


def model_provider():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2ModelForNer(num_tokentypes=0, parallel_output=False)

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
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                            'instead of using previously computed keys/values.')
    group.add_argument("--test_data", type=str, default='',
                       help='the path of test data.')
    return parser

def get_batch(context_tokens):
    """Generate batch from context tokens."""
    args = get_args()
    tokenizer = get_tokenizer()

    # Move to GPU.
    tokens = context_tokens.view(args.batch_size, -1).contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, attention_mask, loss_mask, position_ids

def generate(model, context_tokens):
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    tokens, attention_mask, loss_mask, position_ids = get_batch(context_tokens_tensor)
    type_ids = None
    bs,_  = tokens.shape
    # print('the input tokens =', tokens)
    with torch.no_grad():
        logits = model(tokens,
                       position_ids,
                       loss_mask,
                       labels=True,
                       tokentype_ids=type_ids,
                       forward_method_parallel_output=False)
    tags = model.crf.decode(logits, loss_mask)
    print('tag shape', tags.shape)
    tags = tags.squeeze(0).cpu().numpy().tolist()
    predicts = tags[0]
    return predicts


def main():
    """Main program."""
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    # Set up model and load checkpoint.
    model = get_model(model_provider)
    model.eval()
    args = get_args()
    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    path = args.test_data

    total = 0
    precision = 0
    recall = 0
    f1_score = 0
    with open(path, 'r') as f:
        for i, line in enumerate(tqdm(f, total=num_samples, desc="testing examples")):
            line = line.strip()
            example = json.loads(line)
            labels = example['labels']
            context_tokens = example['tokens']
            mask = [1 for i in context_tokens if i != 6]
            num = mask.sum()
            labels = labels[:num]
            y_predict = generate(model, context_tokens)
            print('length of predict', len(y_predict))
            y_predict =y_predict[:num]
            print('y_predict',y_predict)
            precision += precision_score(labels, y_predict, average='micro')
            print(precision)
            recall += recall_score(labels, y_predict, average='micro')
            f1_score += f1_score(labels, y_predict, average='micro')
        # print('the micro precision = ', precision_score(y_label, y_predict, average='micro'))

        # print('the micro recall = ', recall_score(y_label, y_predict, average='micro'))
        # print('the micro f1 = ', f1_score(y_label, y_predict, average='micro'))
    print('the macro precision = ', precision/total)
    print('the macro recall = ', recall/total)
    print('the macro f1 = ',f1_score/total )




if __name__ == "__main__":

    main()
