#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess Semeval2010 Task8 data, implement DataSet & DataProvider

@author: Ma
@data: 2020/12/21

"""
import os
import re
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer as Tokenizer
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

# Load & Save portable serialized representations (pickle) of Python objects.
def load_pickle(filename):
    path = os.path.join("./data/",\
                                filename)
    with open(path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    path = os.path.join("./data/",\
                                filename)
    with open(path, 'wb') as output:
        pickle.dump(data, output)

def process_text(text, mode='train'):
    sents, relations, comments, blanks = [], [], [], []
    for i in range(int(len(text)/4)):
        sent = text[4*i]
        relation = text[4*i + 1]
        comment = text[4*i + 2]
        blank = text[4*i + 3]
        
        # check entries
        if mode == 'train':
            assert int(re.match("^\d+", sent)[0]) == (i + 1)
        else:
            assert (int(re.match("^\d+", sent)[0]) - 8000) == (i + 1)
        assert re.match("^Comment", comment)
        assert len(blank) == 1
        
        sent = re.findall("\"(.+)\"", sent)[0]
        sent = re.sub('<e1>', '<E1>', sent)
        sent = re.sub('</e1>', '</E1>', sent)
        sent = re.sub('<e2>', '<E2>', sent)
        sent = re.sub('</e2>', '</E2>', sent)
        sents.append(sent); relations.append(relation), comments.append(comment); blanks.append(blank)
    return sents, relations, comments, blanks

def preprocess_semeval2010_8(args):
    '''
    Data preprocessing for SemEval2010 task 8 dataset
    '''
    data_path = args.train_data #'./data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    logger.info("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()
    
    sents, relations, comments, blanks = process_text(text, 'train')
    df_train = pd.DataFrame(data={'sents': sents, 'relations': relations})
    
    data_path = args.test_data #'./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    logger.info("Reading test file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()
    
    sents, relations, comments, blanks = process_text(text, 'test')
    df_test = pd.DataFrame(data={'sents': sents, 'relations': relations})
    
    rm = Relations_Mapper(df_train['relations'])
    save_as_pickle('relations.pkl', rm)
    df_test['relations_id'] = df_test.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    df_train['relations_id'] = df_train.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    save_as_pickle('df_train.pkl', df_train)
    save_as_pickle('df_test.pkl', df_test)
    logger.info("Finished and saved!")
    
    return df_train, df_test, rm

class Relations_Mapper(object):
    def __init__(self, relations):
        self.rel2idx = {}
        self.idx2rel = {}
        
        logger.info("Mapping relations to IDs...")
        self.n_classes = 0
        for relation in tqdm(relations):
            if relation not in self.rel2idx.keys():
                self.rel2idx[relation] = self.n_classes
                self.n_classes += 1
        
        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key

class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __init__(self, seq_pad_value, label_pad_value=-1, label2_pad_value=-1):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=self.seq_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])
        
        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_value)
        y_lengths = torch.LongTensor([len(x) for x in labels])
        
        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(labels2, batch_first=True, padding_value=self.label2_pad_value)
        y2_lengths = torch.LongTensor([len(x) for x in labels2])
        
        return seqs_padded, labels_padded, labels2_padded, \
                x_lengths, y_lengths, y2_lengths

def get_e1e2_start(x, e1_id, e2_id):
    try:
        e1_e2_start = ([i for i, e in enumerate(x) if e == e1_id][0],\
                        [i for i, e in enumerate(x) if e == e2_id][0])
    except Exception as e:
        e1_e2_start = None
        print(e)
    return e1_e2_start

class semeval_dataset(Dataset):
    def __init__(self, df, tokenizer, e1_id, e2_id):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.df = df
        logger.info("Tokenizing data...")
        self.df['input'] = self.df.progress_apply(lambda x: tokenizer.encode(x['sents']),\
                                                             axis=1)
        
        self.df['e1_e2_start'] = self.df.progress_apply(lambda x: get_e1e2_start(x['input'],\
                                                       e1_id=self.e1_id, e2_id=self.e2_id), axis=1)
        print("\nInvalid rows/total: %d/%d" % (df['e1_e2_start'].isnull().sum(), len(df)))
        self.df.dropna(axis=0, inplace=True)
    
    def __len__(self,):
        return len(self.df)
        
    def __getitem__(self, idx):
        return torch.LongTensor(self.df.iloc[idx]['input']),\
                torch.LongTensor(self.df.iloc[idx]['e1_e2_start']),\
                torch.LongTensor([self.df.iloc[idx]['relations_id']])

def load_dataloaders(args, rank = None):
    # Add '<E1>', '</E1>', '<E2>', '</E2>', '<BLANK>' to Token List, then save it
    tokenizer = Tokenizer.from_pretrained(args.model, do_lower_case=False)  # args.model: Name (bert-base-uncased, bert-large-uncased) or Path-to-bert-dir
    tokenizer.add_tokens(['<E1>', '</E1>', '<E2>', '</E2>', '<BLANK>'])
    save_as_pickle("BERT_tokenizer.pkl", tokenizer)
    logger.info("Saved BERT tokenizer at ./data/BERT_tokenizer.pkl")
    
    e1_id = tokenizer.convert_tokens_to_ids('<E1>')
    e2_id = tokenizer.convert_tokens_to_ids('<E2>')
    assert e1_id != e2_id != 1
    
    # Preprocress Semeval2010 Task8 data
    relations_path = './data/relations.pkl'
    train_path = './data/df_train.pkl'
    test_path = './data/df_test.pkl'
    if os.path.isfile(relations_path) and os.path.isfile(train_path) and os.path.isfile(test_path):
        rm = load_pickle('relations.pkl')
        df_train = load_pickle('df_train.pkl')
        df_test = load_pickle('df_test.pkl')
        logger.info("Loaded preproccessed data.")
    else:
        df_train, df_test, rm = preprocess_semeval2010_8(args)
    
    # Tokenize data
    train_set = semeval_dataset(df_train, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
    test_set = semeval_dataset(df_test, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id)
    train_length = len(train_set); test_length = len(test_set)

    # DataLoader
    collate_fn = Pad_Sequence(seq_pad_value=tokenizer.pad_token_id,\
                        label_pad_value=tokenizer.pad_token_id,\
                        label2_pad_value=-1)
    
    world_size = args.n_gpu * args.n_node
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, \
                                num_workers=0, collate_fn=collate_fn, sampler=sampler, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, \
                                num_workers=0, collate_fn=collate_fn, pin_memory=False)
        
    return train_loader, test_loader, train_length, test_length