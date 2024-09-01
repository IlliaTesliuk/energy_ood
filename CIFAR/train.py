#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script computes classifier's fine-tuning on a given Out-of-distribution (OOD) dataset. 

Usage: 

    python train.py cifar10 --model wrn --model_path MODEL_PATH --score energy --m_in M_IN --m_out M_OUT --ood_dataset tinyimages300k --seed 42
    
Author: Illia Tesliuk
Date: 2024-07-29
"""

import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.wrn import WideResNet

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.randomimages_300k_loader import RandomImages
    from utils.validation_dataset import validation_split
    from utils.dataset import get_cifar_train_valid, get_cifar_train_transform, get_cifar_test_transform, CIFAR_MEAN, CIFAR_STD
    from utils.energy import compute_neg_ood_score, compute_logits

# directory with training and validation CIFAR datasets
DATA_DIR = '../../data'
    

########################## SCRIPT UTILS ###############################

def find_checkpoint(args):
    model_found = False
    model_prefix = args.dataset + '_' + args.model + '_pretrained_epoch_'

    if args.load != '':
        for i in range(1000 - 1, -1, -1):
            model_path = os.path.join(args.load, model_prefix + str(i) + '.pt')
            
            if os.path.isfile(model_path):       
                print('Checkpoint found! Epoch:', i)
                model_found = True
                break

        if not model_found:
            assert False, "could not find model to restore"            
    return model_path


def parse_args():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                        help='Choose between CIFAR-10, CIFAR-100.')
    parser.add_argument('--model', '-m', type=str, default='allconv',
                        choices=['allconv', 'wrn', 'densenet'], help='Choose architecture.')
    parser.add_argument('--calibration', '-c', action='store_true',
                        help='Train a model to be used for calibration. This holds out some data for validation.')
    parser.add_argument('--model_path', type=str, default=None, 
                        help="path to the pretrained checkpoint")
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--ood_batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--test_bs', type=int, default=200)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    # WRN Architecture
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, default='./snapshots/pretrained', help='Checkpoint path to resume / test.')
    parser.add_argument('--mean_energy', '-m', action='store_true', help='Compute mean energies only flag')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
    # EG specific
    parser.add_argument('--m_in', type=float, default=-25., help='margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=-7., help='margin for out-distribution; below this value will be penalized')
    parser.add_argument('--score', type=str, default='OE', help='OE|energy')
    parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
    parser.add_argument('--ood_dataset', type=str, choices=['tin597','tinyimages300k'],
                        help='An out-of-distribution dataset for energy-finetuning')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.score == 'OE':
        save_info = 'oe_tune'
    elif args.score == 'energy':
        save_info = 'energy_ft'

    args.save = args.save+save_info
    if os.path.isdir(args.save) == False:
        os.mkdir(args.save)
    state = {k: v for k, v in args._get_kwargs()}
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ############################### IN-DISTRIBUTION DATASET ################################
    train_transform = get_cifar_train_transform()
    test_transform = get_cifar_test_transform()
    
    if args.dataset == 'cifar10':
        test_data = dset.CIFAR10(
            os.path.join(DATA_DIR,"cifarpy"), train=False, transform=test_transform)
        num_classes = 10
    else:
        test_data = dset.CIFAR100(
            os.path.join(DATA_DIR,"cifarpy"), train=False, transform=test_transform)
        num_classes = 100


    ############################# OUT-OF-DISTRIBUTION DATASET ##############################
    ood_transform = trn.Compose([
        trn.ToTensor(), 
        trn.ToPILImage(), 
        trn.RandomCrop(32, padding=4),
        trn.RandomHorizontalFlip(), 
        trn.ToTensor(), 
        trn.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    if args.ood_dataset == 'tinyimages300k':
        ood_data = RandomImages(
            bin_file=os.path.join(DATA_DIR,"300K_random_images.bin"), size=300000, transform=ood_transform) 
    elif args.ood_dataset == 'tin597':
        ood_data = dset.ImageFolder(
            root=os.path.join(DATA_DIR,"tin597/"), transform=ood_transform)
    

    ##################################### DATA LOADERS ####################################
    
    train_loader_in, _ = get_cifar_train_valid(
        os.path.join(DATA_DIR,"cifarpy"), train_transform, test_transform, args.batch_size, args.seed)
    
    train_loader_out = torch.utils.data.DataLoader(
        ood_data, batch_size=args.ood_batch_size, shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False)

    print("Training datasets:")
    print(f"ID ({args.dataset})\n \
          {len(train_loader_in.dataset)} images\n \
          {len(train_loader_in)} batches x ({train_loader_in.batch_size}) images")
    print(f"OOD ({args.ood_dataset})\n \
          {len(train_loader_in.dataset)} images\n \
          {len(train_loader_in)} batches x ({train_loader_out.batch_size}) images")
    
    print("Testing datasets:")
    print(f"ID ({args.dataset})\n \
          {len(test_loader.dataset)} images\n \
          {len(test_loader)} batches x ({test_loader.batch_size}) images")


    ###################################### MODEL ###########################################
    
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

    model_path = find_checkpoint(args) if args.model_path is None else args.model_path
    net.load_state_dict(torch.load(model_path))
    
    if args.ngpu > 0:
        net.to(device)
        torch.cuda.manual_seed(1)
    cudnn.benchmark = True  # fire on all cylinders

    
    # compute mean negative energy scores for ID and OOD datasets
    if args.mean_energy:
        _, in_scores = compute_logits(net, train_loader_in, True, device, T=1.0)
        _, out_scores = compute_logits(net, train_loader_out, True, device, T=1.0)

        print("Mean negative energy scores -E(x):")
        print(f"ID ({args.dataset}): {in_scores.mean():.2f}")
        print(f"OOD ({args.ood_dataset}): {out_scores.mean():.2f}")
        return


    ################################ SETUP TRAINING FOLDER ################################
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)
    print(f"Training results are saved to: {args.save}")

    run_prefix = f"{args.dataset}_{args.model}_s{str(args.seed)}_{save_info}"
    # create a file with training results
    with open(os.path.join(args.save, run_prefix + '_training_results.csv'), 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')


    ############################## TRAINING PARAMETERS ####################################

    optimizer = torch.optim.SGD(
        net.parameters(), state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader_in),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate
        )
    )

    ood_coef = 0.2 # 0.1

    num_batches = min(len(train_loader_in), len(train_loader_out))

    
    ################################## TRAINING ##########################################
    
    print('Training...\n')
    for epoch in range(0, args.epochs):
        state['epoch'] = epoch
        print(f">>>>>>>>>>> Epoch: {epoch} <<<<<<<<<<<<<<")
        begin_epoch = time.time()

        # 1. Training stage
        net.train()  

        loss_avg = 0.0
       
        # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
        train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))

        for in_set, out_set in tqdm(zip(train_loader_in, train_loader_out), total=num_batches):
            # concatenate ID and OOD batches
            data = torch.cat((in_set[0], out_set[0]), 0)
            # ID classification target
            target = in_set[1]

            # move data to device
            data = data.to(device)
            target = target.to(device)

            # forward pass
            x = net(data)

            # backward pass
            scheduler.step()
            optimizer.zero_grad()

            # compute ID cross-entropy loss
            loss = F.cross_entropy(x[:len(in_set[0])], target)
            
            # compute additional energy-based term
            if args.score == 'energy':
                # penalize ID samples with -E(x) < m_in 
                Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=1)
                loss_in = torch.pow(F.relu(Ec_in - args.m_in), 2).mean()
                # penalize OOD samples with -E(X) > m_out
                Ec_out = -torch.logsumexp(x[len(in_set[0]):], dim=1)
                loss_out = torch.pow(F.relu(args.m_out - Ec_out), 2).mean()
                # add weighted energy-based term to ID loss
                loss += ood_coef * (loss_in + loss_out)            
            elif args.score == 'OE': 
                # outlier exposure
                loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

            loss.backward()
            optimizer.step()

            # exponential moving average
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        # save the training loss
        state['train_loss'] = loss_avg


        # 2. evaluation stage
        net.eval()
        loss_avg = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                # move data to device
                data = data.to(device) 
                target = target.to(device)

                # forward pass
                output = net(data)

                # compute ID cross-entropy loss
                loss = F.cross_entropy(output, target)

                # compute classification accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        # save the testing loss and accuracy
        state['test_loss'] = loss_avg / len(test_loader)
        state['test_accuracy'] = correct / len(test_loader.dataset)


        # save model
        torch.save(net.state_dict(), os.path.join(args.save, f"{run_prefix}_epoch_{str(epoch)}.pt"))
        
        # delete the previous model
        prev_path = os.path.join(args.save, f"{run_prefix}_epoch_{str(epoch - 1)}.pt")
        if os.path.exists(prev_path): os.remove(prev_path)

        # save results
        with open(os.path.join(args.save, run_prefix + '_training_results.csv'), 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                time.time() - begin_epoch,
                state['train_loss'],
                state['test_loss'],
                100 - 100. * state['test_accuracy'],
            ))


if __name__ == "__main__":
    main()