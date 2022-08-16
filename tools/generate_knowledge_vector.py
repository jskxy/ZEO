"""Sample Generate GPT2"""

import os
import sys
import numpy as np
import torch
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
import re



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

    return parser


def generate(model, context_tokens, args, tokenizer):
    valid_length = len(context_tokens)
    context_tokens_, context_lengths = pad_batch([context_tokens],
                                                 tokenizer.pad_id, args)
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens_)
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)
    type_ids = None
    with torch.no_grad():
        logits = model(tokens,
                       position_ids,
                       attention_mask,
                       tokentype_ids=type_ids,
                       encode=True,
                       forward_method_parallel_output=False)
    logits = logits.cpu()
    logits = logits.numpy()
    knowledge_vec = logits[0, valid_length - 1, :]
    return knowledge_vec


def generate_embeddings(path):
    """Main program."""
    intro_list = np.load(path, allow_pickle = True)

    print(np.array(intro_list).shape)
    knowledge_list = []
    tokenizer = get_tokenizer()
    for intro in intro_list:
        intro = intro.strip()
        intro = intro.replace(' ','')
        intro = re.sub(r'[\(（].*?[\)）]','',intro)
        print(intro)
        context_tokens = tokenizer.tokenize(intro,out_type = int)
        output_ids = generate(model, context_tokens, args, tokenizer)
        knowledge_list.append(output_ids)
    return knowledge_list, len(knowledge_list)


if __name__ == "__main__":
    initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    # Set up model and load checkpoint.
    model = get_model(model_provider)
    model.eval()

    args = get_args()
    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    path = './training_data/'
    entity_embeddings, entity_num = generate_embeddings(path+'entity_intro.npy')
    np.save('/root/nlp_project/knowledge_injection/panguAlpha_pytorch/GNN/data/qiangdi/entity_embeddings.npy', entity_embeddings)
