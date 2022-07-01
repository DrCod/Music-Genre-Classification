# ====================================================
# train loop
# ====================================================
import os
import gc
import re
import cv2
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
import argparse

import pandas as pd
import numpy as np

import sklearn
from tqdm.auto import tqdm
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
import torch_optimizer as optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


import timm
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

from data.dataloaders import AudioDataset
from src.model import *
from functions.functions import *
from src.train import *



@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')

def init_logger(log_file='./outputs/train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


SEED_VAL  = 42
# Set the seed value all over the place to make this reproducible.
def seed_all(SEED):
  random.seed(SEED_VAL)
  np.random.seed(SEED_VAL)
  torch.manual_seed(SEED_VAL)
  # torch.cuda.manual_seed_all(SEED_VAL) # uncomment if training on cuda devices
  os.environ['PYTHONHASHSEED'] = str(SEED_VAL)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def free_memory(sleep_time=0.1):
  """ Black magic function to free torch memory and some jupyter whims """
  gc.collect()
  # torch.cuda.synchronize() uncomment if training on cuda devices
  gc.collect()
  # torch.cuda.empty_cache() uncomment if training on cuda devices
  time.sleep(sleep_time)



def train_loop(folds, fold, args):

    LOGGER.info(f"========== fold: {fold} training ==========")


    if  args.kfolds is not None:

      if 'fold' not in folds.columns.tolist():

        folds = create_folds(folds)
        trn_idx = folds[folds['fold'] != fold].index
        train_folds = folds.loc[trn_idx].reset_index(drop=True)

        val_idx = folds[folds['fold'] == fold].index
        valid_folds = folds.loc[val_idx].reset_index(drop=True)

      else:

        trn_idx = folds[folds['fold'] != fold].index
        train_folds = folds.loc[trn_idx].reset_index(drop=True)

        val_idx = folds[folds['fold'] == fold].index
        valid_folds = folds.loc[val_idx].reset_index(drop=True)

    train_dataset = AudioDataset(df = train_folds, task =  "train",  size = args.size)
    valid_dataset = AudioDataset(df = valid_folds, task =  "train",  size = args.size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, 
                              shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    if args.family in ["Densenet201", "Densenet161"]:
        model = AudioModel(arch_name = args.model_name,pretrained=args.pretrained, Family =args.family)
    elif "efficient" in args.family:
        model = model_with_attention(CFG=args)
    else:
        RuntimeError("Sorry, invalid model family!")

    model.to(device)

    if args.optimizer_name == "shampoo":
      optimizer = optim.Shampoo(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name == "madgrad":
      optimizer = optim.MADGRAD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name == "adam":
      optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name == "adamw":
      optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
      RuntimeError("optimizer name is invalid!")

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, verbose=True, eps=args.eps)

    es = EarlyStopping(
        patience=args.patience, verbose=True, path=f'./outputs/{args.model_name}_fold{fold}_best.pth'
    ) 

  
    criterion = nn.CrossEntropyLoss() 


    best_score = np.inf
    best_loss = np.inf
    
    for epoch in range(args.epochs):

        start_time = time.time()
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device, args)
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device, args)
        valid_labels = valid_folds[args.target_col].values

        es(avg_val_loss, model)

        scheduler.step(avg_val_loss)
        score = get_score(valid_labels, preds)
        elapsed = time.time() - start_time
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Logloss: {score}')
        if score < best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 'preds': preds}, f'./outputs/{args.model_name}_fold{fold}_best.pth')

        elif es.early_stop:
            print("early stopping...")
            break
    
    check_point = torch.load(f'./outputs/{args.model_name}_fold{fold}_best.pth')
    valid_folds.loc[:, preds_cols] = check_point['preds']
    valid_folds['preds'] = check_point['preds'].argmax(1)

    return valid_folds



def main(args):

    seed_all(SEED_VAL) # set random seed

    folds = pd.read_csv(args.train_csv)

    # format the spec path name

    # check if new train csv is uploaded

    if "spec_name" in folds.columns:

      # change spectrogram path to path pointing to the spectrogram directory
      folds["spec_name"] = folds["spec_name"].apply(lambda x : x.replace("/content/Imgs/", args.root_dir))

    else:
      # create new column to store spec paths if new train csv is uploaded
      folds["spec_name"] = folds["filename"].apply(lambda x : f"{args.root_dir}{x}")


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_result(result_df):
        preds = result_df[preds_cols].values
        labels = result_df[args.target_col].values
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.5f}')
    
    oof_df = pd.DataFrame()
    for fold in range(args.n_fold):
        free_memory(sleep_time=0.1)
        if fold in args.use_folds:
            _oof_df = train_loop(folds, fold, args)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== fold: {fold} result ==========")
            get_result(_oof_df)
            
    LOGGER.info(f"========== CV Score ==========")
    get_result(oof_df)
    oof_df.to_csv(f'./outputs/{args.model_name}_{args.optimizer_name}_oof_df.csv', index=False)



# ====================================================
# main
# ====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Train Runner")

    parser.add_argument("--train_csv", type = str,    default = './data/raw/train.csv', help = "path/to/train csv")
    parser.add_argument("--root_dir" , type = str,    default = './data/spectrograms/', help = "path/to/spectrograms")
    parser.add_argument("--debug", action = "store_true", default=False, help = "run in debug mode")
    parser.add_argument("--print_freq", type =int, default=100, help="print frequency")
    parser.add_argument("--num_workers", type =int, default=2, help="number of workers")
    parser.add_argument("--model_name", type =str, default="densenet201", help="model name")
    parser.add_argument("--family", type =str, default="Densenet201", choices =["Densenet201", "Densenet161", "tf_efficientnet_b4_ns"], help="model family")
    parser.add_argument("--pretrained", action="store_true", default=False, help="use pretrained model flag")
    parser.add_argument("--size", type = tuple, default=(500, 230), help="image size")
    parser.add_argument("--optimizer_name", type =str, default="adamw", help="optimizer name")
    parser.add_argument("--epochs", type =int, default=50, help="number of training epochs")
    parser.add_argument("--batch_size", type = int, default=8, help = "batch size")
    parser.add_argument("--factor", type =float, default=0.2, help="factor param")
    parser.add_argument("--drop_rate", type =float, default=0.4, help="dropout rate")
    parser.add_argument("--patience", type =int, default=5, help="patience")
    parser.add_argument("--eps", type =float, default=1e-6, help="epsilon")
    parser.add_argument("--lr", type =float, default=1e-4, help="learning rate")
    parser.add_argument("--min_lr", type =float, default=1e-6, help="minimum learning rate")
    parser.add_argument("--weight_decay", type =float, default=1e-6, help="weight_decay")
    parser.add_argument("--gradient_accumulation_steps", type =int, default=1, help="grad accumulation steps")
    parser.add_argument("--max_grad_norm", type =int, default=1e3, help="max grad norm")
    parser.add_argument("--eta_min", type =float, default=1e-5, help="minimum eta")
    parser.add_argument("--target_size", type =int, default=10, help="number of music genres")
    parser.add_argument("--target_col", type =str, default="label", help="label column")
    parser.add_argument("--n_fold", type =int, default=20, help="number of folds")
    parser.add_argument("--kfolds", action = "store_true", default=False, help="use kfolds")
    parser.add_argument("--use_folds", nargs = "+", type = int, default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], help="list all folds to cross validate model on")
    parser.add_argument("--seed", type =int, default=42, help="set random seed value")

    args = parser.parse_args()

    main(args)
