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

"""Pretrain utilities."""

from datetime import datetime
import math
import sys
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import json
import logging
from tqdm import tqdm
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from apex.optimizers import FusedAdam as Adam
from megatron import get_tokenizer
from megatron import get_args
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import mpu
from megatron import print_rank_0
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.fp16 import FP16_Module
from megatron.fp16 import FP16_Optimizer
from megatron.initialize import initialize_megatron
from megatron.learning_rates import AnnealingLR
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import get_params_for_weight_decay_optimization
from megatron.model.realm_model import ICTBertModel
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import make_data_loader
from megatron.utils import report_memory
from pathlib import Path
from argparse import ArgumentParser
from collections import namedtuple
def pretrain(model_provider,forward_step_func, extra_args_provider=None, args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the modle using the forward_step_func.

    Arguments:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    args = get_args()
    num_workers = args.num_workers
    # parser = ArgumentParser()
    # parser.add_argument('--pregenerated_data', type=Path, required=True)
    # parser.add_argument('--output_dir', type=Path, required=True)
    tokenizer = get_tokenizer()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model and optimizer').stop()

    # Data stuff.
    timers('train/valid/test data iterators').start()
    # print('data type is', type(epoch_dataset), '==', epoch_dataset)
    knowledge_list = np.load(
        '/root/nlp_project/knowledge_injection/panguAlpha_pytorch/GNN/data/qiangdi/entity_embeddings_350m.npy',
        allow_pickle=True)
    # print('knowledge_shape=',knowledge_list.shape)
    knowledge_list = [x for x in knowledge_list]
    # print('knowledge_list=',knowledge_list[0],'\n')
    knowledge_embeding = torch.FloatTensor(knowledge_list)
    knowledge_embed = torch.nn.Embedding.from_pretrained(knowledge_embeding)
    # print('knowledge_list[0]:',knowledge_list[0][:3],'\n')
    # print('555:', knowledge_embed(torch.LongTensor([0])),'\n')

    epoch_dataset = PregeneratedDataset(knowledge_embed,args,
                                        tokenizer=tokenizer,
                                        fp16=args.fp16)


    def collate_fn(batch_dataset):
        batch_dataset = np.array(batch_dataset,dtype=object)
        # print('===', batch_dataset.shape, '====\n')
        # print(batch_dataset[0:-1,0].shape,'\t element value is', batch_dataset[0:1,0],'\n\n')
        tokens = np.asarray(batch_dataset[:, 0]).astype('int64')
        ngram_ids = np.asarray(batch_dataset[:, 1]).astype('int64')
        entity_mapping = np.asarray(batch_dataset[:, 2]).astype('float32')
        neg_ids = np.asarray(batch_dataset[:, 3]).astype('int64')
        # tokens,ngram_ids,entity_mapping,neg_ids = batch_dataset[:,0:-1]
        batch_size = batch_dataset.shape[0]


        neg_entity = 10
        max_length = 1024
        max_ngram = 10
        hidden_size = 2560
        contras_embedding = torch.Tensor(batch_size, max_ngram, neg_entity+2, hidden_size).zero_()
        # print('knowledge_embed', knowledge_embed.weight.size())
        # zipped = zip(ngram_positions,ngram_lengths,ngram_ids)
        # batch_knowledge_vec = np.zeros(shape=(batch_size, max_length - 1, args.hidden_size), dtype=np.float32)
        # for i in range(batch_size):
        #     for j, (pos_index, length, id) in enumerate(zipped[i]):
        #         entity_mapping[i, j, pos_index:pos_index+length] = 1.0 / length
        #         contras_embedding[i, j, 1] = knowledge_embed(torch.LongTensor(id))
        #         for k, neg in enumerate(neg_index[i][j]):
        #             contras_embedding[i, j, k+2] = knowledge_embed(torch.LongTensor(neg))
        for i in range(batch_size):
            for j, id in enumerate(ngram_ids[i]):
                # print('id=',id,'\t',type(id),'\n')
                if id != -1:
                    # print('666:', contras_embedding[i, j, 1].shape, '777',torch.squeeze(knowledge_embed(torch.LongTensor(id)),dim=0))
                    contras_embedding[i, j, 1] = torch.squeeze(knowledge_embed(torch.LongTensor([id])),dim=0)
                    # contras_embedding[i, j, 1] = torch.squeeze(knowledge_embed(torch.LongTensor(id)), dim=0)
                    for k, neg in enumerate(neg_ids[i][j]):
                        # print('neg_ids=', neg, '\t',type(id),'\n')
                        contras_embedding[i, j, k + 2] = torch.squeeze(knowledge_embed(torch.LongTensor([neg])),dim=0)
                else:
                    break
        # for i in range(batch_size):
        #     for pos_index, length, id in enumerate(zipped):
        #         batch_knowledge_vec[i, pos_index:pos_index+length] = knowledge_embed(torch.LongTensor(id))
        tokens = torch.IntTensor(tokens)
        entity_mapping = torch.FloatTensor(entity_mapping)
        return tokens, contras_embedding, entity_mapping

    # train_data_iterator = build_train_valid_test_data_iterators(epoch_dataset)
    train_dataloader = make_data_loader(epoch_dataset)
    timers('train/valid/test data iterators').stop()

    # total_iteration = len(epoch_dataset) // args.batch_size
    # if len(epoch_dataset) % args.batch_size != 0:
    #     total_iteration += 1

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['model and optimizer', 'train/valid/test data iterators'])
    print_rank_0('training ...')

    # if args.local_rank == -1:
    #     train_sampler = RandomSampler(epoch_dataset)
    # else:
    #     train_sampler = DistributedSampler(epoch_dataset)
    # train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=num_workers)

    iteration = 0
    # if args.do_train and args.train_iters > 0:

    iteration, _ = train(forward_step_func, model, optimizer, lr_scheduler, train_dataloader)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler)

    '''
    if args.do_valid:
        prefix = 'the end of training for val data'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, False)



    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   0, True)
    '''




def get_model(model_provider_func):
    """Build the model."""
    args = get_args()

    # Build model on cpu.
    model = model_provider_func()

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training."""
    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        model = torchDDP(model, device_ids=[i], output_device=i,
                         process_group=mpu.get_data_parallel_group())
        return model
    if args.DDP_impl == 'local':
        model = LocalDDP(model)
        return model

    raise NotImplementedError('Unknown DDP implementation specified: {}. '
                              'Exiting.'.format(args.DDP_impl))


def get_optimizer(model):
    """Set up the optimizer."""
    args = get_args()

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (torchDDP, LocalDDP, FP16_Module)):
        model = model.module
    param_groups = get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    # Use Adam.
    optimizer = Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=warmup_iter,
        total_iters=num_iters,
        decay_style=args.lr_decay_style,
        last_iter=init_step,
        min_lr=args.min_lr,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


def setup_model_and_optimizer(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func)
    optimizer = get_optimizer(model)
    lr_scheduler = get_learning_rate_scheduler(optimizer)

    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler)
    else:
        args.iteration = 0

    # get model without FP16 and/or TorchDDP wrappers
    unwrapped_model = model
    while hasattr(unwrapped_model, 'module'):
        unwrapped_model = unwrapped_model.module

    if args.iteration == 0 and hasattr(unwrapped_model, 'init_state_dict_from_bert'):
        print("Initializing ICT from pretrained BERT model", flush=True)
        unwrapped_model.init_state_dict_from_bert()

    return model, optimizer, lr_scheduler


def backward_step(optimizer, model, loss):
    """Backward step."""
    args = get_args()
    timers = get_timers()

    # Backward pass.
    timers('backward-backward').start()
    #optimizer.zero_grad(set_grads_to_None=True)
    optimizer.zero_grad()
    if args.fp16:
        optimizer.backward(loss, update_master_grads=False)
    else:
        loss.backward()
    timers('backward-backward').stop()

    # All-reduce if needed.
    if args.DDP_impl == 'local':
        timers('backward-allreduce').start()
        model.allreduce_params(reduce_after=False,
                               fp32_allreduce=args.fp32_allreduce)
        timers('backward-allreduce').stop()

    # Update master gradients.
    timers('backward-master-grad').start()
    if args.fp16:
        optimizer.update_master_grads()
    timers('backward-master-grad').stop()

    # Clipping gradients helps prevent the exploding gradient.
    timers('backward-clip-grad').start()
    if args.clip_grad > 0:
        if not args.fp16:
            mpu.clip_grad_norm(model.parameters(), args.clip_grad)
        else:
            optimizer.clip_master_grads(args.clip_grad)
    timers('backward-clip-grad').stop()


def train_step(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Forward model for one step.
    timers('forward').start()
    loss, loss_reduced = forward_step_func(data_iterator, model)
    timers('forward').stop()

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    backward_step(optimizer, model, loss)
    timers('backward').stop()

    # Update parameters.
    timers('optimizer').start()
    optimizer.step()
    timers('optimizer').stop()

    # Update learning rate.
    skipped_iter = 0
    if not (args.fp16 and optimizer.overflow):
        lr_scheduler.step()
    else:
        skipped_iter = 1

    return loss_reduced, skipped_iter


def training_log(loss_dict, total_loss_dict, learning_rate, iteration,
                 loss_scale, report_memory_flag):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()

    # Update losses.
    for key in loss_dict:
        total_loss_dict[key] = total_loss_dict.get(key, 0.) + loss_dict[key]

    # Logging.
    timers_to_log = []

    def add_to_logging(name):
        if name in timers.timers:
            timers_to_log.append(name)
    add_to_logging('forward')
    add_to_logging('backward')
    add_to_logging('backward-backward')
    add_to_logging('backward-allreduce')
    add_to_logging('backward-master-grad')
    add_to_logging('backward-clip-grad')
    add_to_logging('optimizer')
    add_to_logging('batch generator')

    # Tensorboard values.
    if writer and torch.distributed.get_rank() == 0:
        writer.add_scalar('learning_rate', learning_rate, iteration)
        for key in loss_dict:
            writer.add_scalar(key, loss_dict[key], iteration)
        if args.fp16:
            writer.add_scalar('loss_scale', loss_scale, iteration)
        normalizer = iteration % args.log_interval
        if normalizer == 0:
            normalizer = args.log_interval
        timers.write(timers_to_log, writer, iteration,
                     normalizer=normalizer)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval time').elapsed()
        if writer and torch.distributed.get_rank() == 0:
            writer.add_scalar('iteration_time',
                              elapsed_time / args.log_interval, iteration)
        log_string = ' iteration {:8d}/{:8d} |'.format(iteration,
                                                       args.train_iters)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time * 1000.0 / args.log_interval)
        log_string += ' learning rate: {:.3E} |'.format(learning_rate)
        for key in total_loss_dict:
            avg = total_loss_dict[key].item() / args.log_interval
            log_string += ' {}: {:.6E} |'.format(key, avg)
            total_loss_dict[key] = 0.0
        if args.fp16:
            log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        print_rank_0(log_string)
        if report_memory_flag:
            report_memory('after {} iterations'.format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def train(forward_step_func, model, optimizer, lr_scheduler,
          train_dataloader):
    """Train the model function."""
    args = get_args()
    timers = get_timers()

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    # iteration = args.iteration
    iteration = 0
    skipped_iters = 0


    timers('interval time').start()
    report_memory_flag = True
    # while iteration <= total_iteration:
    args.train_iters = len(train_dataloader)
    for step, batch in enumerate(train_dataloader):
        loss_dict, skipped_iter = train_step(forward_step_func,
                                             batch,
                                             model,
                                             optimizer,
                                             lr_scheduler)
        skipped_iters += skipped_iter
        iteration += 1

        # Logging.
        loss_scale = None
        if args.fp16:
            loss_scale = optimizer.loss_scale
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          report_memory_flag)

        # Autoresume
        if args.adlr_autoresume and \
                (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              lr_scheduler)

        # Checkpointing
        if args.save and args.save_interval and \
                iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)

        # Evaluation
        # if args.eval_interval and iteration % args.eval_interval == 0 and \
        #    args.do_valid:
        #     prefix = 'iteration {}'.format(iteration)
        #     evaluate_and_print_results(prefix, forward_step_func,
        #                                valid_data_iterator, model,
        #                                iteration, False)

        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print_rank_0('rank: {} | time: {} | exiting the program at '
                         'iteration {}'.format(rank, time_str, iteration))
            sys.exit()

    return iteration, skipped_iters


def evaluate(forward_step_func, data_iterator, model, verbose=False):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss_dict = {}

    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration,
                                                            args.eval_iters))
            # Forward evaluation.
            _, loss_dict = forward_step_func(data_iterator, model)
            # Reduce across processes.
            for key in loss_dict:
                total_loss_dict[key] = total_loss_dict.get(key, 0.) + \
                    loss_dict[key]
    # Move model back to the train mode.
    model.train()

    for key in total_loss_dict:
        total_loss_dict[key] /= args.eval_iters

    return total_loss_dict


def evaluate_and_print_results(prefix, forward_step_func,
                               data_iterator, model,
                               iteration, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    writer = get_tensorboard_writer()

    total_loss_dict = evaluate(forward_step_func, data_iterator, model, verbose)
    string = ' validation loss at {} | '.format(prefix)
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer and torch.distributed.get_rank() == 0:
            writer.add_scalar('{} value'.format(key),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} ppl'.format(key), ppl, iteration)

    length = len(string) + 1
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)
InputFeatures = namedtuple("InputFeatures","input_ids ngram_ids ngram_positions ngram_starts ngram_lengths")
def convert_example_to_features(example, tokenizer, max_seq_length=1024, max_ngram_in_sequence=10):
    input_ids = example["tokens"]
    # add ngram level information
    ngram_ids = example["ngram_ids"]
    ngram_positions = example["ngram_positions"]
    ngram_lengths = example["ngram_lengths"]


    assert len(input_ids) <= max_seq_length

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    # add ngram pads
    ngram_id_array = np.zeros(max_ngram_in_sequence, dtype=np.int)
    ngram_id_array[:len(ngram_ids)] = ngram_ids

    # record the masked positions

    # The matrix here take too much space either in disk or in memory, so the usage have to be lazily convert the
    # the start position and length to the matrix at training time.

    ngram_positions_matrix = np.zeros(shape=(max_seq_length-1, max_ngram_in_sequence), dtype=np.bool)
    for i in range(len(ngram_ids)):
        ngram_positions_matrix[ngram_positions[i]:ngram_positions[i]+ngram_lengths[i], i] = 1

    ngram_start_array = np.zeros(max_ngram_in_sequence, dtype=np.int32)
    ngram_start_array[:len(ngram_ids)] = ngram_positions

    ngram_length_array = np.zeros(max_ngram_in_sequence, dtype=np.int32)
    ngram_length_array[:len(ngram_ids)] = ngram_lengths

    # ngram_mask_array = np.zeros(max_ngram_in_sequence, dtype=np.bool)
    # ngram_mask_array[:len(ngram_ids)] = 1

    # ngram_segment_array = np.zeros(max_ngram_in_sequence, dtype=np.bool)
    # ngram_segment_array[:len(ngram_ids)] = ngram_segment_ids
    features = InputFeatures(input_ids=input_array,
                             ngram_ids=ngram_id_array,
                             ngram_positions=ngram_positions_matrix,
                             ngram_starts=ngram_start_array,
                             ngram_lengths=ngram_length_array)
    return features
class PregeneratedDataset(Dataset):
    def __init__(self, knowledge_embedding,args, tokenizer, fp16=False):
        # self.vocab = tokenizer.vocab_file
        self.tokenizer = tokenizer
        # self.epoch = epoch
        # self.data_epoch = epoch % num_data_epochs
        args = get_args()
        training_path = args.pregenerated_data
        file_name = args.training_data_file
        data_file = training_path / file_name
        metrics_file = args.metrics_file
        metrics_path = training_path / metrics_file
        assert data_file.is_file() and metrics_path.is_file()
        metrics = json.loads(metrics_path.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        max_ngram_in_sequence = metrics['max_ngram_in_sequence']
        self.temp_dir = None
        self.working_dir = None
        self.fp16 = fp16

        self.temp_dir = None
        self.working_dir = None
        self.fp16 = fp16
        self.temp_dir = "/tmp"
        self.embed = knowledge_embedding
        self.inject = args.inject
        hidden_size = args.hidden_size
        neg_entity = args.neg_entity
        input_ids = torch.empty(size=(num_samples, seq_len), dtype=torch.int64)
        if self.inject == 1 or self.inject == 2:
            entity_mapping = torch.empty(size=(num_samples, max_ngram_in_sequence, seq_len), dtype=torch.float32)
            contras_embedding = torch.empty(num_samples, max_ngram_in_sequence, neg_entity + 2, hidden_size)
            ngram_ids = torch.empty(size=(num_samples, max_ngram_in_sequence), dtype=torch.int64)
            neg_ids = torch.empty(size=(num_samples, max_ngram_in_sequence, neg_entity), dtype=torch.int64)
            with data_file.open() as f:
                # print('the len of data_file is =', len(f))
                for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                    # line = line.strip()
                    example = json.loads(line)
                    # print('line.keys is -----', example.keys())
                    input_ids[i] = torch.tensor(example['tokens'])
                    ngram_ids[i] = torch.tensor(example['ngram_ids'])
                    entity_mapping[i] = torch.tensor(example['entity_mapping'])
                    neg_ids[i] = torch.tensor(example['neg_ids'])

                    for j, id in enumerate(ngram_ids[i]):
                        if id != -1:
                            contras_embedding[i, j, 1] = torch.squeeze(self.embed(torch.LongTensor([id])), dim=0)
                            # print('neg_idx=', neg_ids[i][j], 'len=', len(neg_ids[i][j]),'\n')
                            for k, neg in enumerate(neg_ids[i][j]):

                                contras_embedding[i, j, k + 2] = torch.squeeze(self.embed(torch.LongTensor([neg])), dim=0)
                        else:
                            break
                    # print('====',knowledge_embedding[i].shape)
            self.entity_mapping = entity_mapping
            self.contras_embedding = contras_embedding
            # print('i=',i)
            assert i == num_samples - 1  # Assert that the sample count metric was true
        if self.inject == 0:
            with data_file.open() as f:
                for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                    example = json.loads(line)
                    input_ids[i] = torch.tensor(example['tokens'])
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids


    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        # position = torch.tensor(self.ngram_positions[item].astype(np.double))
        # if self.fp16:
        #     position = position.half()
        # else:
        #     position = position.float()
        if self.inject == 1:
            return self.input_ids[item],self.contras_embedding[item],self.entity_mapping[item]
        else:
            return self.input_ids[item]

def build_train_valid_test_data_iterators(epoch_data):
    args = get_args()
    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')
    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        # Rank, size, and global batch size.
        data_parallel_size = mpu.get_data_parallel_world_size()
        global_batch_size = args.batch_size * data_parallel_size

        # Number of train/valid/test samples.
        # train_iters = args.train_iters
        # eval_iters = (train_iters // args.eval_interval + 1) * args.eval_iters
        # test_iters = args.eval_iters
        # train_val_test_num_samples = [train_iters * global_batch_size,
        #                               eval_iters * global_batch_size,
        #                               test_iters * global_batch_size]
        # print_rank_0(' > datasets target sizes (minimum size):')
        # print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        # print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        # print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))



        # Build dataloders.
        train_dataloader = make_data_loader(epoch_data)
        # valid_dataloader = make_data_loader(valid_ds)
        # test_dataloader = make_data_loader(test_ds)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None
        # do_valid = valid_dataloader is not None and args.eval_iters > 0
        # do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor([int(do_train)])
    else:
        flags = torch.cuda.LongTensor([0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    args.do_train = flags[0].item()
    # args.do_valid = flags[1].item()
    # args.do_test = flags[2].item()

    # Shift the start iterations.
    # total_iteration = len(epoch_data) // args.batch_size
    # if len(epoch_data) % args.batch_size != 0:
    #     total_iteration += 1
    # if train_dataloader is not None:
    #     train_dataloader.batch_sampler.start_iter = total_iteration % \
    #         len(train_dataloader)
    #     print_rank_0('setting training data start iteration to {}'.
    #                  format(train_dataloader.batch_sampler.start_iter))

    # if valid_dataloader is not None:
    #     start_iter_val = (args.iteration // args.eval_interval) * \
    #         args.eval_iters
    #     valid_dataloader.batch_sampler.start_iter = start_iter_val % \
    #         len(valid_dataloader)
    #     print_rank_0('setting validation data start iteration to {}'.
    #                  format(valid_dataloader.batch_sampler.start_iter))

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None
    # if valid_dataloader is not None:
    #     valid_data_iterator = iter(valid_dataloader)
    # else:
    #     valid_data_iterator = None
    #
    # if test_dataloader is not None:
    #     test_data_iterator = iter(test_dataloader)
    # else:
    #     test_data_iterator = None

    return train_data_iterator
