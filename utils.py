#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT RE Model Load, Save, Evaluate Functions
See: Soares, Livio Baldini, et al. "Matching the Blanks: Distributional Similarity for Relation Learning." 
     Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019. 

@author: Ma
@data: 2020/12/22

"""

import os
import torch
import torch.nn as nn
from preprocessing import save_as_pickle, load_pickle
from seqeval.metrics import precision_score, recall_score, f1_score
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def load_state(net, optimizer, scheduler, load_best=False):
    """ Loads saved model and optimizer states if exists """
    base_path = "./data/"
    amp_checkpoint = None
    checkpoint_path = os.path.join(base_path,"task_test_checkpoint.pth.tar")
    best_path = os.path.join(base_path,"task_test_model_best.pth.tar")
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        amp_checkpoint = checkpoint['amp']
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_pred, amp_checkpoint

def load_results():
    """ Loads saved results if exists """
    losses_path = "./data/task_test_losses_per_epoch.pkl"
    accuracy_path = "./data/task_train_accuracy_per_epoch.pkl"
    f1_path = "./data/task_test_f1_per_epoch.pkl"
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path) and os.path.isfile(f1_path):
        losses_per_epoch = load_pickle("task_test_losses_per_epoch.pkl")
        accuracy_per_epoch = load_pickle("task_train_accuracy_per_epoch.pkl")
        f1_per_epoch = load_pickle("task_test_f1_per_epoch.pkl")
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch, f1_per_epoch = [], [], []
    return losses_per_epoch, accuracy_per_epoch, f1_per_epoch


def evaluate_(output, labels, ignore_idx):
    ### ignore index 0 (padding) when calculating accuracy
    idxs = (labels != ignore_idx).squeeze()
    o_labels = torch.softmax(output, dim=1).max(1)[1]
    l = labels.squeeze()[idxs]; o = o_labels[idxs]

    if len(idxs) > 1:
        acc = (l == o).sum().item()/len(idxs)
    else:
        acc = (l == o).sum().item()
    l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
    o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()

    return acc, (o, l)

def evaluate_results(net, test_loader, pad_id, cuda):
    logger.info("Evaluating test samples...")
    acc = 0; out_labels = []; true_labels = []
    net.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            x, e1_e2_start, labels, _,_,_ = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                
            classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, e1_e2_start=e1_e2_start)
            
            accuracy, (o, l) = evaluate_(classification_logits, labels, ignore_idx=-1)
            out_labels.append([str(i) for i in o]); true_labels.append([str(i) for i in l])
            acc += accuracy
    
    accuracy = acc/(i + 1)
    results = {
        "accuracy": accuracy,
        "precision": precision_score(true_labels, out_labels),
        "recall": recall_score(true_labels, out_labels),
        "f1": f1_score(true_labels, out_labels, average='macro')
    }
    # logger.info("***** Eval results *****")
    # for key in sorted(results.keys()):
    #     logger.info("  %s = %s", key, str(results[key]))
    
    return results