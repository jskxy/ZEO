"""Create data for pretrain."""
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from megatron.text_generation_utils import pad_batch, get_batch
from megatron.tokenizer.tokenization_jieba import JIEBATokenizer
from megatron import get_args
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve
import random
import numpy as np
import json
from multiprocessing import Pool
import collections


class DocumentDatabase():
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        # Uses the current iteration counter to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def get_knowledge_embedding(args, ngram_positions, ngram_lengths, ngram_ids):
    assert len(ngram_positions) == len(ngram_lengths) == len(ngram_ids)
    ngram_positions_matrix = np.zeros(shape=(args.seq_length-1, args.max_ngram_in_sequence), dtype=np.float32)
    knowledge_matrix = np.zeros(shape=(args.seq_length-1, args.hidden_size), dtype=np.float32)
    for i in range(len(ngram_ids)):

        ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1

    return knowledge_matrix
def create_instances_from_doc(args, doc_database, doc_idx, align_dic,tokenizer,name2type, type2entity):
    # triplet_file = triplet_path + '/entity_rel_list.npy'
    max_entity_in_seq = args.max_ngram_in_sequence
    neg_entity = args.neg_entity
    seq_length = args.seq_length
    inject = args.inject
    # remove the symbol of blank to achieve the alignment between sentences and entities
    blank = '▁'
    # args = get_args()
    doc = doc_database[doc_idx]
    doc = doc.replace(" ",'')
    doc = tokenizer.encode(doc, out_type=str)
    # doc.extend(['<eot>'])
    sentence = []
    # split doc into sentence that consist of 1024 units
    if len(doc) >= seq_length:
        sent_num = len(doc) // seq_length
        j = 0
        for i in range(sent_num):
            sentence.append(doc[j:j+seq_length])
            j += seq_length
        index = len(doc) % seq_length
        if index != 0:
            last_sent = doc[sent_num * seq_length:]
            last_sent.extend(['<pad>'] * (seq_length - len(last_sent)))
            sentence.append(last_sent)
    else:
        doc.extend(['<pad>'] * (seq_length - len(doc)))
        sentence.append(doc)
    all_instances = []



    # dic_num = len(align_dic)
    if inject:
        for sub_sent in sentence:

            tokens = [tokenizer.convert_tokens_to_ids(i) for i in sub_sent]
            strip_sent = [i.strip(blank) for i in sub_sent]
            # print(strip_sent)
            #  ['和', '奥斯曼', '的']
            #  Filter the ngram segment from 2 to 20 to check whether there is a element of triplets
            ngram_matches = [-1 for i in range(max_entity_in_seq)]
            # neg_ids = np.full((max_entity_in_seq, neg_entity),-1,dtype=int)
            neg_ids = [-1 for i in range(max_entity_in_seq)]
            entity_mapping = np.zeros((max_entity_in_seq, seq_length), dtype=float)
            entity_dis = [0 for i in range(seq_length)]
            tem_instances = {}
            count_ent = 0
            # p at least = 1
            for p in range(1, 15):
                # start_pos represent the start index of a element
                # have to + 1 because the right line won't be thought
                for start_pos in range(0, len(strip_sent) - p + 1):
                    character_segment = strip_sent[start_pos:start_pos + p]
                    character_segment = "".join(character_segment)
                    element_index = [k for k, v in align_dic.items() if character_segment in v]

                    if element_index and count_ent < max_entity_in_seq:
                        # print(element_index)
                        print("==", character_segment, "==")
                        element_index = element_index[0]
                        key_ent = align_dic[element_index][0]
                        # print('match string and index is ', character_segment, ';',align_dic[str(element_index)][0])
                        # print('pos and length is', start_pos, p,'=======')
                        label = name2type[key_ent]
                        neg_count = 0
                        for i in type2entity.keys():
                            if i != label:
                                idx = random.randint(0, len(type2entity[i])-1)
                                neg_id = type2entity[i][idx]
                                # neg_count += 1
                                # if neg_count >= neg_entity:
                                #     break
                        left_index = start_pos
                        len_ent = p
                        tem_instances[(left_index, len_ent)] = {'id': int(element_index), 'neg_id': neg_id}
                        count_ent += 1

            cn = 0
            print('num of tem_instances', len(tem_instances))
            for pos_index, sub_dict in tem_instances.items():
                ngram_matches[cn] = sub_dict['id']
                neg_ids[cn] = sub_dict['neg_id']
                left_index, len_ent = pos_index
                # print(left_index, len_ent)
                #for example [0,2]
                if len_ent >= 3:
                    entity_dis[left_index] = 1
                    for i in range(1, len_ent-1):
                        entity_dis[left_index+i] = 2
                    entity_dis[left_index+len_ent-1] = 3
                elif len_ent == 2:
                    entity_dis[left_index] = 1
                    entity_dis[left_index+1] = 3
                else:
                    entity_dis[left_index] = 1
                for i in range(len_ent):
                    entity_mapping[cn][left_index+i] = 1
                cn += 1
            entity_mapping = entity_mapping.tolist()
            # print(type(sub_dict['tokens']), type(neg_ids),type(ngram_matches),type(entity_mapping),type(entity_dis))
            instance = {
                    "tokens": tokens,
                    "ngram_ids": ngram_matches,
                    "neg_ids": neg_ids,
                    'entity_mapping': entity_mapping,
                    'symbol': strip_sent,
                    'entity_dis': entity_dis}
            all_instances.append(instance)
            # print(instance)

    else:
        for sub_sent in sentence:
            tokens = [tokenizer.convert_tokens_to_ids(i) for i in sub_sent]
            instance = {
                "tokens": tokens}
            all_instances.append(instance)

    return all_instances


def extract(path, idx, args):
    tokenizer = JIEBATokenizer(args.vocab_file)
    # train_corpus = args.train_corpus / 'pretrain_corpus.txt'
    num_instances = 0
    with DocumentDatabase() as docs:
        with open(path,'r') as f:
            # print('read')
            corpus = f.read()
            sent_list = corpus.split('\n\n')
            print('====',len(sent_list),'====')
            for sent in sent_list:
                docs.add_document(sent)
            print('the length of docs is ',len(docs))
        if len(docs) < 1:
            exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                 "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                 "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                 "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                 "sections or paragraphs.")

        args.tem_out_dir.mkdir(exist_ok=True)
        suffix = "instance%d.json"%idx
        epoch_filename = args.tem_out_dir/suffix
        # epoch_filename = args.output_dir / "training_data.json"
        with open(args.align_file,'r') as f:
            align_dict = json.load(f)

        with open('/root/nlp_project/knowledge_injection/panguAlpha_pytorch/merge_data/ent2type.json','r') as f:
            name2type = json.load(f)
        with open('/root/nlp_project/knowledge_injection/panguAlpha_pytorch/merge_data/type2idx.json','r') as f:
            type2entity = json.load(f)
        # align_list = np.load(align_list, allow_pickle=True)
        # align_dict = {index: list for index, list in enumerate(align_list)}
        # vector_file = args.vector_path / "knowledge_embedding.json"
        # embedding_dic = json.loads(vector_file.read_text())
        # for index, list in enumerate(triplet_list):
        #     print('index=',index,'list =',list)
        with epoch_filename.open('w') as epoch_file:
            for doc_idx in trange(len(docs), desc="Document"):
                doc_instances = create_instances_from_doc(args, docs, doc_idx, align_dict, tokenizer, name2type, type2entity)
                doc_instances = [json.dumps(instance, ensure_ascii=False) for instance in doc_instances]
                # json.dump(doc_instances, epoch_file, ensure_ascii=False)
                # num_instances = len(doc_instances)
                for instance in doc_instances:
                    epoch_file.write(instance + '\n')
                    # json.dump(instance,epoch_file)
                    num_instances += 1
        return num_instances

def run_proc(idx, thread, file_lists, args):
    for i in range(len(file_lists)):
        if i % thread == idx:
            print('It is: == ',file_lists[i])
            extract(file_lists[i], i, args)

if __name__ == '__main__':
    # train_corpus = '/root/nlp_project/knowledge_injection/DataProcesor/TheFirstDataProcess/training_data/merge_data/pretrain_corpus-1.txt'
    parser = ArgumentParser()
    parser.add_argument('--pretrain_corpus', type=Path,
                        default="/root/nlp_project/knowledge_injection/panguAlpha_pytorch/merge_data/pretrain_corpus-1.txt")
    parser.add_argument("--tem_in_dir", type=Path,
                        default="/root/nlp_project/knowledge_injection/panguAlpha_pytorch/tem_inject_data")
    parser.add_argument("--tem_out_dir", type=Path,
                        default="/root/nlp_project/knowledge_injection/panguAlpha_pytorch/tem_inject_data_out")
    parser.add_argument("--output_file", type=Path,
                        default="/root/nlp_project/knowledge_injection/panguAlpha_pytorch/pretrain_data/training_data-1.json")

    # parser.add_argument("--epochs_to_generate", type=int, default=3,
    #                     help="Number of epochs of data to pregenerate")
    parser.add_argument("--align_file", type=str,
                        default="/root/nlp_project/knowledge_injection/panguAlpha_pytorch/merge_data/entity2aligned.json")
    parser.add_argument("--max_ngram_in_sequence", type=int, default=70)
    parser.add_argument("--neg_entity", type=int, default=1)
    parser.add_argument("--vocab_file", type=str,
                        default='/root/nlp_project/knowledge_injection/panguAlpha_pytorch/megatron/tokenizer/bpe_4w_pcl/vocab')
    parser.add_argument("--vector_path", type=Path,
                        default='/root/nlp_project/knowledge_injection/panguAlpha_pytorch/')
    parser.add_argument("--hidden_size", type=int, default=2560)
    # parser.add_argument("--max_ngram", type=int, default=10)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--inject", type=int, default=1)
    parser.add_argument("--metrics_file", type=str,
                        default='/root/nlp_project/knowledge_injection/panguAlpha_pytorch/pretrain_data/training_data_metrics.json')
    parser.add_argument("--num_thread", type=int, default=10)
    args = parser.parse_args()
    train_corpus = args.pretrain_corpus
    with open(train_corpus,'r') as f:
        corpus = f.read()
        sent_list = corpus.split('\n\n')

    num = len(sent_list)
    print('=================================', num)
    thread = args.num_thread

    gap = num // thread
    split_sent = []
    for i in range(thread):
        if i != thread-1:
            split_sent.append(sent_list[i*gap:(i+1)*gap])
        else:
            split_sent.append(sent_list[i*gap:])
    tem_in_path = args.tem_in_dir
    for i in range(len(split_sent)):
        suffix = 'data%d.txt'%i
        with open(tem_in_path/suffix, 'w') as fin:
            for line in split_sent[i]:
                fin.write(line)
                fin.write('\n\n')

    # process corpus with multi-pool
    p = Pool(thread)
    file_lists = []
    num_instances = 0
    result = []
    for filename in os.listdir(tem_in_path):
        file_lists.append(os.path.join(tem_in_path,filename))
    print('file_list=',file_lists)
    start = time.time()
    for i in range(thread):
        result.append(p.apply_async(run_proc, args=(i,thread,file_lists, args)))
    p.close()
    p.join()
    end = time.time()
    all_time = int(end - start)
    print('it cost ', all_time)

    for i in result:
        if i.get():
            num_instances += i.get()

    # merge multiple processed json file
    num_ins = 0
    tem_out_path = args.tem_out_dir
    output_path = args.output_file
    # all_data = []
    with open(output_path, 'w') as fin:
        for filename in os.listdir(tem_out_path):
            file_path = os.path.join(tem_out_path, filename)
            print(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                # print(file_path)
                for i in f:
                    sent = json.loads(i)
                    sent = json.dumps(sent, ensure_ascii=False)
                    fin.write(sent + '\n')
        # json.dump(all_data, fin, ensure_ascii=False)

    metric_path = args.metrics_file
    with open(output_path, 'r') as f:
        line = f.readlines()
        # example = json.loads(line)
        num_ins = len(line)
    metrics = {
        "num_training_examples": num_ins,
        "max_seq_len": 1024,
        "max_ngram_in_sequence": 70,
        "time": all_time
    }
    with open(metric_path,'w') as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False)




