#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT RE Model Infer on Test
See: Soares, Livio Baldini, et al. "Matching the Blanks: Distributional Similarity for Relation Learning." 
     Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019. 

@author: Ma
@data: 2020/12/21

"""


import pickle
import os
import re
import pandas as pd
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from argparse import ArgumentParser
from transformers import BertTokenizer as Tokenizer

import logging

tqdm.pandas(desc="prog-bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

class infer_from_trained(object):
    def __init__(self, args=None):
        self.args = args
        self.cuda = torch.cuda.is_available()

        self.entities_of_interest = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", \
                                     "WORK_OF_ART", "LAW", "LANGUAGE", 'PER']
        
        logger.info("Loading tokenizer and model...")
        from utils import load_state
        
        # preparing the distributed env
        # using nccl for distributed training
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=1,
            rank=0)
        from model import BertmodelRE as Model
        model = Model.from_pretrained(args.model)
        model.cuda()

        self.tokenizer = load_pickle("BERT_tokenizer.pkl")
        self.tokenizer = Tokenizer.from_pretrained(args.model, do_lower_case=False)  # args.model: Name (bert-base-uncased, bert-large-uncased) or Path-to-bert-dir
        self.tokenizer.add_tokens(['<E1>', '</E1>', '<E2>', '</E2>', '<BLANK>'])

        model.resize_token_embeddings(len(self.tokenizer))

        self.net = DDP(model,
                    device_ids=[0],
                    output_device=0,
                    find_unused_parameters=True)

        start_epoch, best_pred, amp_checkpoint = load_state(self.net, None, None, load_best=False)
        logger.info("Done!")
        
        self.e1_id = self.tokenizer.convert_tokens_to_ids('<E1>')
        self.e2_id = self.tokenizer.convert_tokens_to_ids('<E2>')
        self.pad_id = self.tokenizer.pad_token_id
        self.rm = load_pickle("relations.pkl")
    
    def get_e1e2_start(self, x):
        e1_e2_start = ([i for i, e in enumerate(x) if e == self.e1_id][0],\
                        [i for i, e in enumerate(x) if e == self.e2_id][0])
        return e1_e2_start
    
    def infer_one_sentence(self, sentence):
        self.net.eval()
        tokenized = self.tokenizer.encode(sentence);
        print(tokenized)
        e1_e2_start = self.get_e1e2_start(tokenized); 
        print("e1_id=", self.e1_id, "\te2_id=", self.e2_id)
        print("e1_e2_start=", e1_e2_start)
        tokenized = torch.LongTensor(tokenized).unsqueeze(0)
        e1_e2_start = torch.LongTensor(e1_e2_start).unsqueeze(0)
        attention_mask = (tokenized != self.pad_id).float()
        token_type_ids = torch.zeros((tokenized.shape[0], tokenized.shape[1])).long()
        
        if self.cuda:
            tokenized = tokenized.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
        
        with torch.no_grad():
            classification_logits = self.net(tokenized, token_type_ids=token_type_ids, attention_mask=attention_mask, e1_e2_start=e1_e2_start)
            predicted = torch.softmax(classification_logits, dim=1).max(1)[1].item()
        print("Sentence: ", sentence)
        print("Predicted: ", self.rm.idx2rel[predicted].strip(), '\n')
        return self.rm.idx2rel[predicted].strip()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='./model/bert-base-uncased', \
                        help="Name or path to BERT pretrained model folder")
    parser.add_argument("--train_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT', \
                        help="training data .txt file path")
    parser.add_argument("--test_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT', \
                        help="test data .txt file path")
    parser.add_argument("--log_dir", type=str, default='./log', \
                        help="Tensorboard Log dir")
    parser.add_argument("--continueTraining", type=bool, default=False, \
                        help="Whether to continue Training")
    parser.add_argument("--num_classes", type=int, default=19, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.00007, help="learning rate")
    parser.add_argument("--seed", type=int, default=42, help='Seed for random init')

    # Distributed Training args
    parser.add_argument("--n_gpu", type=int, default=8, help='Number of GPUs in one node')
    parser.add_argument("--n_node", type=int, default=1, help='Number of nodes in total')
    parser.add_argument("--node_rank", type=int, default=0, help='Node rank for this machine. 0 for master, and 1,2... for slaves')
    parser.add_argument("--MASTER_ADDR", type=str, default='10.104.91.31', help='Master Address')
    parser.add_argument("--MASTER_PORT", type=str, default='29501', help='Master port')

    args = parser.parse_args()
    print(args)

    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT

    sentences_no = []
    sentences = []
    ref = []
    # Read test file
    with open('./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT') as f:
        while True:
            line = f.readline()
            if not line:
                break
            pos = line.index("\t")
            sentences_no.append(line[:pos].strip())
            sent = line[pos+1:].strip().strip("\"")
            sent = re.sub('<e1>', '<E1>', sent)
            sent = re.sub('</e1>', '</E1>', sent)
            sent = re.sub('<e2>', '<E2>', sent)
            sent = re.sub('</e2>', '</E2>', sent)
            sentences.append(sent)
            line = f.readline()
            if not line:
                break
            ref.append(line.strip())
            line = f.readline()
            line = f.readline()
            if not line:
                break

    inferer = infer_from_trained(args)
    infer = []
    for test in sentences:
        print(test)
        infer.append(inferer.infer_one_sentence(test))
    
    with open('Ref.txt', 'w') as f:
        for i in range(len(sentences_no)):
            f.write(f"{sentences_no[i]}\t{ref[i]}\n")
    
    with open('Predict.txt', 'w') as f:
        for i in range(len(sentences_no)):
            f.write(f"{sentences_no[i]}\t{infer[i]}\n")
    
    with open('Full_text.txt', 'w') as f:
        for i in range(len(sentences_no)):
            f.write(f"{sentences_no[i]}\t{sentences[i]}\n")
            f.write(f"Reference: {ref[i]}\n")
            f.write(f"Predicted: {infer[i]}\n\n")
