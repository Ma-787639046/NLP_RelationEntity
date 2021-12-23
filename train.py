#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT RE Model Train Functions
See: Soares, Livio Baldini, et al. "Matching the Blanks: Distributional Similarity for Relation Learning." 
     Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019. 

@author: Ma
@data: 2020/12/22

"""

import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from preprocessing import load_dataloaders, save_as_pickle, load_pickle
from utils import load_state, load_results, evaluate_, evaluate_results
from model import BertmodelRE
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import time
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def train(rank, *tuples):
    args = tuples[0]    # mp.spawm() pass a tuple (args, ) to train() function, so args = tuples[0]
    # add world size for distributed training
    args.world_size = args.n_gpu * args.n_node
    global_rank = args.node_rank * args.n_gpu + rank

    if args.fp16:    
        from apex import amp
    else:
        amp = None
    
    cuda = torch.cuda.is_available()
    
    # preparing the distributed env
    # using nccl for distributed training
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=global_rank)

    set_random_seed(args.seed)

    # DataLoaders
    train_loader, test_loader, train_len, test_len = load_dataloaders(args, rank=global_rank)
    if rank == 0:
        logger.info(f"Loaded {train_len} Training samples.")

    # Create Model
    model = BertmodelRE.from_pretrained(args.model)
    
    if rank == 0:
        logger.info("finish initing paramters")

    # Revise BERT token lists
    tokenizer = load_pickle("BERT_tokenizer.pkl")
    model.resize_token_embeddings(len(tokenizer))
    e1_id = tokenizer.convert_tokens_to_ids('<E1>')
    e2_id = tokenizer.convert_tokens_to_ids('<E2>')
    assert e1_id != e2_id != 1
    
    if cuda:    # Move model to GPU:rank
        model.cuda(rank)
    
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True)

    dist.barrier()  # synchronizes all processes
    if rank == 0:
        logger.info("Finish initing all processors.")


    logger.info("Frezzing most of Bert layers, Activate Last Bert encoder layer and FC layer.")
    if "base" in args.model:
        unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", \
                            "classification_layer", "lm_linear", "cls"]
    elif "large" in args.model:
        unfrozen_layers = ["classifier", "pooler", "encoder.layer.23", \
                            "classification_layer", "lm_linear", "cls"]
    else:
        logger.info("Unsuported bert folder name, terminated!")
        return
        
    for name, param in model.named_parameters():
        if not any([layer in name for layer in unfrozen_layers]):
            print("[FROZE]: %s" % name)
            param.requires_grad = False
        else:
            print("[FREE]: %s" % name)
            param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam([{"params":model.parameters(), "lr": args.lr}])
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
                                                                      24,26,30], gamma=0.8)
    
    if args.continueTraining:
        start_epoch, best_pred, amp_checkpoint = load_state(model, optimizer, scheduler, load_best=False)  
    else:
        start_epoch, best_pred, amp_checkpoint = 0, 0, None
    
    if (args.fp16) and (amp is not None):
        logger.info("Fp16 Activate!")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
        if amp_checkpoint is not None:
            amp.load_state_dict(amp_checkpoint)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
                                                                          24,26,30], gamma=0.8)
    
    if args.continueTraining:
        losses_per_epoch, accuracy_per_epoch, test_f1_per_epoch = load_results()
    else:
        losses_per_epoch, accuracy_per_epoch, test_f1_per_epoch = [], [], []
    
    if rank == 0:   # Tensorboard log dir
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        writer = SummaryWriter(log_dir=args.log_dir)
    
    logger.info("Starting training!")
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    update_interval = len(train_loader)//10

    step_cnt = 0
    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        model.train(); total_loss = 0.0; losses_per_batch = []; total_acc = 0.0; accuracy_per_batch = []
        for i, data in enumerate(train_loader, 0):
            x, e1_e2_start, labels, _,_,_ = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            if cuda:
                x = x.cuda(rank)
                labels = labels.cuda(rank)
                attention_mask = attention_mask.cuda(rank)
                token_type_ids = token_type_ids.cuda(rank)
                
            classification_logits = model(x, token_type_ids=token_type_ids, attention_mask=attention_mask, e1_e2_start=e1_e2_start)
            
            
            loss = criterion(classification_logits, labels.squeeze(1))
            loss = loss/args.gradient_acc_steps
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
            else:
                clip_grad_norm_(model.parameters(), args.max_norm)
            
            if (i % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            total_acc += evaluate_(classification_logits, labels, ignore_idx=-1)[0]
            step_cnt += 1
            
            if (i % update_interval) == (update_interval - 1) and global_rank == 0:
                losses_per_batch.append(args.gradient_acc_steps * total_loss / update_interval)
                accuracy_per_batch.append(total_acc / update_interval)
                writer.add_scalar("Train Loss", losses_per_batch[-1], global_step=step_cnt)
                writer.add_scalar("Train Accuracy", accuracy_per_batch[-1], global_step=step_cnt)
                print(f'[Epoch: {epoch + 1: 2d}, {(i + 1)*args.batch_size*args.world_size: 5d}/ {train_len: d} items] total loss {losses_per_batch[-1]: .3f}, accuracy per batch: {accuracy_per_batch[-1]: .3f}')
                total_loss = 0.0; total_acc = 0.0
        
        scheduler.step()

        if global_rank == 0:    # Evaluate after one epoch
            results = evaluate_results(model, test_loader, pad_id, cuda)
            losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
            accuracy_per_epoch.append(sum(accuracy_per_batch)/len(accuracy_per_batch))
            test_f1_per_epoch.append(results['f1'])
            writer.add_scalar("Test F1", test_f1_per_epoch[-1], global_step=step_cnt)
            print("Epoch finished, took %.2f seconds." % (time.time() - start_time))
            print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
            print("Train accuracy at Epoch %d: %.7f" % (epoch + 1, accuracy_per_epoch[-1]))
            # print("Test f1 at Epoch %d: %.7f" % (epoch + 1, test_f1_per_epoch[-1]))
            
            # Save best model
            if accuracy_per_epoch[-1] > best_pred:
                best_pred = accuracy_per_epoch[-1]
                torch.save({
                        'epoch': epoch + 1,\
                        'state_dict': model.state_dict(),\
                        'best_acc': accuracy_per_epoch[-1],\
                        'optimizer' : optimizer.state_dict(),\
                        'scheduler' : scheduler.state_dict(),\
                        'amp': amp.state_dict() if amp is not None else amp
                    }, os.path.join("./data/" , "task_test_model_best.pth.tar"))
            
            # Save loss, acc, f1 score per epoch
            if (epoch % 1) == 0:
                save_as_pickle("task_test_losses_per_epoch.pkl", losses_per_epoch)
                save_as_pickle("task_train_accuracy_per_epoch.pkl", accuracy_per_epoch)
                save_as_pickle("task_test_f1_per_epoch.pkl", test_f1_per_epoch)
                torch.save({
                        'epoch': epoch + 1,\
                        'state_dict': model.state_dict(),\
                        'best_acc': accuracy_per_epoch[-1],\
                        'optimizer' : optimizer.state_dict(),\
                        'scheduler' : scheduler.state_dict(),\
                        'amp': amp.state_dict() if amp is not None else amp
                    }, os.path.join("./data/" , "task_test_checkpoint.pth.tar"))
    
    df = pd.DataFrame({'Train Loss': losses_per_epoch, 'Train Accuracy':accuracy_per_epoch})
    print(df)

    dist.destroy_process_group()