#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script computes classification logits on given datasets. 

Usage: 

To compute logits on source (train/valid/gen) datasets:
    python compute_logits.py --ckpt MODEL_PATH --run_dir OUTPUT_PATH --batch_size 32 --source

To compute logits on all test datasets:
    python compute_logits.py --ckpt MODEL_PATH --run_dir OUTPUT_PATH --batch_size 32 --test --all

To compute logits on particular test datasets:
    python compute_logits.py --ckpt MODEL_PATH --run_dir OUTPUT_PATH --batch_size 32 --test --svhn --places

    
Author: Illia Tesliuk
Date: 2024-07-29
"""

import os
import argparse
import torch
import numpy as np
from models import WideResNet
from energy_ood.utils.dataset import get_test_dataset, get_cifar_train_valid, get_cifar_generated, get_cifar_train_transform, get_cifar_test_transform
from energy_ood.utils.energy import write_neg_scores, compute_logits

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED=42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "../data"
TEST_DS = {
    "cifar_test": "cifarpy",
    "svhn": "svhn",
    "isun": "iSUN",
    "lsun_c": "LSUN",
    "lsun_r": "LSUN_resize",
    "dtd": "dtd",
    "places": "places365"
}
MODEL_NAME = "wrn"


def load_wrn(wrn_path):
    wrn = WideResNet(depth=40, num_classes=10, widen_factor=2, dropRate=0.3)

    assert os.path.isfile(wrn_path)
    wrn.load_state_dict(torch.load(wrn_path))
    
    print(f"[WRN] Loaded from \'{wrn_path}\'")
    return wrn


def compute_and_save(model, dataloader, temp, include_labels, logits_dir, save_logits, save_scores, fname):
    logits, neg_scores = compute_logits(model, dataloader, include_labels, DEVICE, temp)

    if save_logits:
        full_path = os.path.join(logits_dir, fname)
        write_neg_scores(logits, full_path)
        print(f"Saved logits to \'{full_path}\'")

    if save_scores:
        neg_scores_dir = os.path.join(logits_dir, "neg_scores")
        os.makedirs(neg_scores_dir, exist_ok=True)
        full_path = os.path.join(neg_scores_dir, f"neg_score_{fname}")
        write_neg_scores(neg_scores, full_path)
        print(f"Saved negative energy scores to \'{full_path}\'")


def compute_test(model, batch_size, temp, run_dir, save_logits, save_scores, sets):
    logits_dir = os.path.join(run_dir, "logits_test")
    os.makedirs(logits_dir, exist_ok=True)

    for i, set_name in enumerate(sets):
        print(f">>>>>>>> {i+1}. {set_name} : {sets[set_name]}")
        if sets[set_name]:
            test_loader, _ = get_test_dataset(
                os.path.join(DATA_DIR, TEST_DS[set_name]),
                batch_size,
                shuffle=False,
                name=set_name
            )            
            compute_and_save(
                model, test_loader, temp,
                include_labels=True,
                logits_dir=logits_dir,
                save_logits=save_logits,
                save_scores=save_scores,
                fname=f"{set_name}_{MODEL_NAME}.csv"
            )


def compute_train_valid_gen(model, batch_size, temp, logits_dir, save_logits, save_scores, trn, aug_mode):
    # get training and validation dataloaders
    train_loader, valid_loader = get_cifar_train_valid(
        os.path.join(DATA_DIR, "cifarpy"), 
        trn, trn, 
        batch_size, 
        SEED
    )
    # get a synthetic dataloader
    gen_loader = get_cifar_generated(os.path.join(DATA_DIR, "cifar10_generated"), trn, batch_size)
        
    loaders = [
        (train_loader, "cifar_train"),
        (valid_loader, "cifar_valid"),
        (gen_loader, "cifar_gen")
    ]
    for loader, set_name in loaders:
        compute_and_save(
            model, loader, temp,
            include_labels=True,
            logits_dir=logits_dir,
            save_logits=save_logits,
            save_scores=save_scores,
            fname=f"{set_name}{aug_mode}{MODEL_NAME}.csv"
    )


def create_dataset_dict(args):
    sets = dict()
    if args.all:
        for key in TEST_DS:
            sets[key] = True 
        return sets
    
    sets["cifar_test"] = args.cifar
    sets["svhn"] = args.svhn
    sets["isun"] = args.isun
    sets["lsun_c"] = args.lsunc
    sets["lsun_r"] = args.lsunr
    sets["dtd"] = args.dtd
    sets["places"] = args.places
    return sets


def main(args):
    # create and load WideResnet classifier
    model = load_wrn(args.ckpt)
    model.to(DEVICE)
    model.eval()

    # compute logits on test datasets    
    if args.test:
        compute_test(
            model, 
            args.batch_size, 
            temp=1.0, 
            run_dir=args.run_dir, 
            save_logits=True, 
            save_scores=True, 
            sets=create_dataset_dict(args)
        )

    # compute logits on 'source' dataset for threshold selection   
    if args.source:
        # create a directory to store logits
        logits_dir = os.path.join(args.run_dir, "logits_threshold")
        os.makedirs(logits_dir, exist_ok=True)

        trns = [
            (get_cifar_train_transform(), "_aug_"),
            (get_cifar_test_transform(), "_")
        ]
        for trn, aug_mode in trns:
            compute_train_valid_gen(
                model,
                args.batch_size,
                temp=1.0,
                logits_dir=logits_dir,
                save_logits=True,
                save_scores=True,
                trn=trn,
                aug_mode=aug_mode
            )



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str)
    p.add_argument("--run_dir", type=str)
    p.add_argument("--test", action="store_true")
    p.add_argument("--source", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--cifar", action="store_true")
    p.add_argument("--svhn", action="store_true")
    p.add_argument("--dtd", action="store_true")
    p.add_argument("--isun", action="store_true")
    p.add_argument("--lsunc", action="store_true")
    p.add_argument("--lsunr", action="store_true")
    p.add_argument("--places", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    main(args)
    
            
    
