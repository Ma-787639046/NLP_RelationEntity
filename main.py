#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT RE Model
Fine-tune the BERT model on SemEval
See: Soares, Livio Baldini, et al. "Matching the Blanks: Distributional Similarity for Relation Learning." 
     Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019. 

@author: Ma
@data: 2020/12/22

"""
import os
from train import train
import torch.multiprocessing as mp
import logging
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

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

     mp.spawn(train, args=(args, ), nprocs=args.n_gpu)