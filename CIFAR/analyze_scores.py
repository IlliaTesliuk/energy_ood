#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script computes negative energy scores, thresholds, OOD detection errors and produces the corresponding plots.

Usage: 

    python analyze_scores.py --run_dir INPUT_DIR
    
Author: Illia Tesliuk
Date: 2024-07-29
"""

import os
import argparse
import torch
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
from itertools import product

from energy_ood.utils.energy import *


SEED=42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Mame of the tested classiferr
MODEL = "wrn"
# Tested temperatures of negative energy function
TEMPERATURES = [1] #[0.1, 1, 2, 5, 10, 100, 200, 500, 1000]

# CIFAR10-test and OOD test dataset
TEST_SETS = ['cifar_test', 'dtd', 'isun', 'lsun_c', 'lsun_r', 'places', 'svhn']
# All possible pairs of <CIFAR10-test, OOD dataset>  
TEST_COMBS = list(product(TEST_SETS[:1], TEST_SETS[1:]))

# Tested types of source datasets for threshold selection
MODES = {"train": "Train", "valid": "Validation", "gen": "Generated"}
# Tested types of augmentations applied to the source datasets
AUG_MODES = {"no_aug": "Original", "aug": "Augmented"}
# All possible pairs of <source dataset type, augmentation type> for threshold selection  
THR_COMBS = list(product(['no_aug','aug'],['train','valid','gen']))

# Regular Matplotlib colors
COLORS = ['red', 'green', 'black', 'red', 'green', 'black']
# Scientific style colors
with plt.style.context('science'):
    SCOLORS = {
        "blue": sns.color_palette()[0],
        "green": sns.color_palette()[1],
        "yellow": sns.color_palette()[2],
        "red": sns.color_palette()[3],
        "purple": sns.color_palette()[4],
        "black": sns.color_palette()[5],
        "gray": sns.color_palette()[6],
    }

DS2COL = {
    'cifar_test':'CIFAR10',
    'dtd':'DTD',
    'isun':'ISUN',
    'lsun_c':'LSUNC',
    'lsun_r':'LSUNR',
    'places':'PLACES',
    'svhn':'SVHN'
}


# Draws and saves a plot with negative energy score distribution and a threshold
def plot_thr(neg_scores, thr, name, legend, fullpath):
    with plt.style.context('science'):
        plt.rcParams.update({"font.size":16}) 

        fig = plt.figure(figsize=(10,6))
        sns.kdeplot(neg_scores,color=SCOLORS["blue"])
        ax =  plt.gca()
        l1 = ax.lines[0]
        x1 = l1.get_xydata()[:,0]
        y1 = l1.get_xydata()[:,1]

        plt.vlines(x=thr, ymin=0, ymax=y1.max()+0.01,colors=[SCOLORS["black"]],linewidth=2)

        x11_mask = x1 <= thr
        x11, y11 = x1[x11_mask], y1[x11_mask]
        x12_mask = x1 >= thr
        x12, y12 = x1[x12_mask], y1[x12_mask]
    
        ax.fill_between(x12,y12, color=SCOLORS["blue"], alpha=0.6)
        ax.fill_between(x11,y11, color=SCOLORS["blue"], alpha=0.25)

        plt.legend([legend,"$5^{th}$ quantile"],frameon=True,loc='upper right')
        plt.title(name)
        plt.xlabel("Negative energy score")
        plt.savefig(fullpath)
        plt.clf()
        plt.close()


# Draws and saves a plot with negative energy score distribution and 6 tested thresholds
def plot_all_thr(all_neg_scores, all_thrs, name, full_path):
    with plt.style.context('science'):
        SCOLORS = {
            "blue": sns.color_palette()[0],
            "green": sns.color_palette()[1],
            "yellow": sns.color_palette()[2],
            "red": sns.color_palette()[3],
            "purple": sns.color_palette()[4],
            "black": sns.color_palette()[5],
            "gray": sns.color_palette()[6],
        }
        plt.rcParams.update({"font.size":16}) 

        fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(10*3,6*2),sharex=True,sharey=True)
        for i, a in enumerate(['no_aug','aug']):
            cur_thrs = all_thrs[a]
            for j, m in enumerate(['train','valid','gen']): 
                thr = cur_thrs[m]
                neg_scores = all_neg_scores[a][m]

                sns.kdeplot(neg_scores,color=SCOLORS["blue"],ax=axes[i,j])
                l1 = axes[i,j].lines[0]
                x1 = l1.get_xydata()[:,0]
                y1 = l1.get_xydata()[:,1]

                axes[i,j].vlines(
                    x=thr, ymin=0, ymax=y1.max()+0.01,
                    colors=[SCOLORS["black"]], linewidth=2
                )

                x11_mask = x1 <= thr
                x11, y11 = x1[x11_mask], y1[x11_mask]
                x12_mask = x1 >= thr
                x12, y12 = x1[x12_mask], y1[x12_mask]
    
                axes[i,j].fill_between(x12,y12, color=SCOLORS["blue"], alpha=0.6)
                axes[i,j].fill_between(x11,y11, color=SCOLORS["blue"], alpha=0.25)

                legend = f"CIFAR-10:\n{MODES[m]}, {AUG_MODES[a]}"
                axes[i,j].legend([legend,"$5^{th}$ quantile"],frameon=True,loc='upper right')
                axes[i,j].set_xlabel("Negative energy score")

        plt.suptitle(name)
        plt.savefig(full_path)
        plt.clf()
        plt.close()
        

###########################################################################

# Reads logits of every test dataset
def load_test_logits(logits_dir, model):
    logits = dict()
    for test_ds in TEST_SETS:
        fname = f"{test_ds}_{model}.csv"
        ns = load_logits(os.path.join(logits_dir, fname))
        logits[test_ds] = ns
    return logits

# Reads logits of every source dataset
def load_source_scores(logits_dir, model):
    modes = ['train','valid','gen']
    aug = ['_','_aug_']

    logits = {"aug": dict(), "no_aug": dict()}
    for m, a in list(product(modes, aug)):
        fname = f"cifar_{m}{a}{model}.csv"
        ns = load_logits(os.path.join(logits_dir,fname))

        key = "aug" if a == '_aug_' else 'no_aug'
        logits[key][m] = ns
    return logits
   

############################################################################################
#                                Negative Energy Scores
############################################################################################

# Limits number of tested images within a <ID, OOD> dataset pair based on a given mode 
def filter_scores(id_scores, ood_scores, ood_name, comp_mode):
    
    if ood_scores.shape[0] < id_scores.shape[0] and comp_mode == "smallest":
        all_ids = np.arange(0,id_scores.shape[0])
        np.random.shuffle(all_ids)
        id_scores = id_scores[all_ids[:ood_scores.shape[0]]]
    
    if id_scores.shape[0] < ood_scores.shape[0] and comp_mode in ["smallest", "10000"]:
        all_ids = np.arange(0,ood_scores.shape[0])
        np.random.shuffle(all_ids)
        ood_scores = ood_scores[all_ids[:id_scores.shape[0]]]
    
    return id_scores, ood_scores


# Plots ID and OOD negative energy score distributions
def plot_test(id_scores, ood_scores, legend, fig_title, full_path):
    with plt.style.context('science'):
        plt.rcParams.update({"font.size":16}) 
        # 1. simple plot
        fig = plt.figure(figsize=(10,6))
        sns.kdeplot(id_scores,color=SCOLORS['blue'])
        sns.kdeplot(ood_scores,color=SCOLORS['gray'])
        if True:
            ax =  plt.gca()
            l1, l2 = ax.lines[0], ax.lines[1]
            x1 = l1.get_xydata()[:,0]
            y1 = l1.get_xydata()[:,1]
            x2 = l2.get_xydata()[:,0]
            y2 = l2.get_xydata()[:,1]
            ax.fill_between(x1,y1, color=SCOLORS['blue'], alpha=0.25)
            ax.fill_between(x2,y2, color=SCOLORS['gray'], alpha=0.25)
            
        plt.legend(legend, frameon=True)
        plt.title(fig_title) 
        plt.xlabel("Negative energy score")
        plt.savefig(full_path)
        print(f"Saved to {full_path}")
        plt.clf()
        plt.close()


# Plots ID and OOD negative energy score distributions with the ID/OOD threshold
def plot_test_thr(id_scores, ood_scores, legend, fig_title, full_path, thresholds):
    with plt.style.context('science'):
        plt.rcParams.update({"font.size":16}) 
        fig = plt.figure(figsize=(10,6))
        
        sns.kdeplot(id_scores,color=SCOLORS['blue'])
        sns.kdeplot(ood_scores,color=SCOLORS['gray'])
        
        ax = plt.gca() # get axis handle
        ymax = max(
            ax.lines[0].get_ydata().max(), 
            ax.lines[1].get_ydata().max()
        )
        for k, thr_comb in enumerate(THR_COMBS):
            aug_mode, set_mode = thr_comb
            linestyle = 'solid' if aug_mode == 'no_aug' else 'dashed'

            plt.vlines(
                x=thresholds[aug_mode][set_mode],
                ymin=0, ymax=ymax, colors=[COLORS[k]],
                linestyles=linestyle,linewidth=2
            )
        if True:
            ax =  plt.gca()
            l1, l2 = ax.lines[0], ax.lines[1]
            x1 = l1.get_xydata()[:,0]
            y1 = l1.get_xydata()[:,1]
            x2 = l2.get_xydata()[:,0]
            y2 = l2.get_xydata()[:,1]
            ax.fill_between(x1,y1, color=SCOLORS['blue'], alpha=0.25)
            ax.fill_between(x2,y2, color=SCOLORS['gray'], alpha=0.25)

        plt.title(fig_title) 
        plt.xlabel("Negative energy score")
        plt.legend(legend, frameon=True)
        plt.savefig(full_path)
        print(f"Saved to {full_path}")
        plt.clf()
        plt.close()

            
######################################################################################################
#                                   Errors
######################################################################################################


# Computes Type 1 error: FN / (TP + FN)
def error_type1(preds, labels, id=1, ood=0):
    id_mask = labels == id  
    preds_for_id = preds[id_mask] 
    # correct predictions
    err_preds = (preds_for_id == ood).sum()
    total_preds = preds_for_id.shape[0]
    return err_preds / total_preds


# Computes Type 2 error: FP / (TN + FP)
def error_type2(preds, labels, id=1, ood=0):
    ood_mask = labels == ood 
    preds_for_ood = preds[ood_mask]
    # correct predictions
    err_preds = (preds_for_ood == id).sum()
    total_preds = preds_for_ood.shape[0]
    return err_preds /total_preds


# Threshold-based OOD detector
def predict(neg_ood_scores, threshold):
    # compare negative energy with threshold: -E(OOD) | -E(ID)
    # 0=OOD, 1=ID
    ood_preds = neg_ood_scores > threshold
    return ood_preds.to(int)


# Performs OOD predictions for a pair of ID, OOD datasets
def predict_for_pair(id_scores, ood_scores, threshold, equal_size=False, T=1.0):
    id_preds = predict(id_scores, threshold)
    ood_preds = predict(ood_scores, threshold)

    if equal_size:
        if id_preds.shape[0] > ood_preds.shape[0]:
            size = ood_preds.shape[0]
            indices = torch.randint(0,
                             id_preds.shape[0],
                             (size,),
                             generator=torch.manual_seed(SEED)
                             )
            id_preds = id_preds[indices]
        elif id_preds.shape[0] < ood_preds.shape[0]:
            size = id_preds.shape[0]
            indices = torch.randint(0,
                             ood_preds.shape[0],
                             (size,),
                             generator=torch.manual_seed(SEED)
                             )
            ood_preds = ood_preds[indices]
        assert id_preds.shape == ood_preds.shape

    labels = torch.concat([
        torch.ones_like(id_preds),
        torch.zeros_like(ood_preds)
    ])
    preds = torch.concat([
        id_preds,
        ood_preds
    ])
    err1, err2 = error_type1(preds,labels), error_type2(preds, labels)
    return err1.item(), err2.item()


# Computes negative energy scores, OOD detection thresholds and errors and produces their plots.
def produce_neg_ood_scores_model(run_dir, comp_mode='default'):
    modes = ['train','valid','gen']
    aug = ['no_aug','aug']
    
    # load logits of source ID datasets: training, validation, synthetic
    source_logits = load_source_scores(os.path.join(run_dir, 'logits_threshold'), MODEL)
    # load logits of testing ID and OOD datasets
    test_logits = load_test_logits(os.path.join(run_dir, 'logits_test'), MODEL)
    
    # iterate over energy temperatures
    for temperature in TEMPERATURES:
        print(f">>> {temperature}")
        temp_name = f"t_{str(temperature).replace('.','')}"
        # create a directory for the given temperature
        temp_dir = os.path.join(run_dir, temp_name)
        os.makedirs(temp_dir, exist_ok=True)

        # create a directory for negative energy scores of test ID and OOD datasets        
        neg_scores_test_dir = os.path.join(temp_dir, "neg_scores_test")
        os.makedirs(neg_scores_test_dir, exist_ok=True)

        # create a directory for negative energy scores of source ID datasets
        neg_scores_source_dir = os.path.join(temp_dir, "neg_scores_source")
        os.makedirs(neg_scores_source_dir, exist_ok=True)

        # create a plot directory for test ID and OOD datasets
        plots_test_dir = os.path.join(temp_dir, "plots_test")
        os.makedirs(plots_test_dir, exist_ok=True)

        # create a plot directory for source ID datasets
        plots_source_dir = os.path.join(temp_dir, "plots_source")
        os.makedirs(plots_source_dir, exist_ok=True)


        # STEP 1. Source Distributions & Thresholds
        print(">>>>>> STEP 1. Source Disitributions: Thresholds")

        neg_scores_source = {"aug": dict(), "no_aug": dict()}
        thresholds = {"aug": dict(), "no_aug": dict()}

        # iterate over dataset-mode pairs ([train, valid, synth] x [aug, no aug])
        for m, a in list(product(modes, aug)):
            # compute negative energy scores
            neg_scores = compute_neg_ood_score(source_logits[a][m], temperature)
            neg_scores_source[a][m] = neg_scores

            a_str =  '_aug_' if a == "aug" else '_'
            fname = f"neg_score_cifar_{m}{a_str}{MODEL}" 
            
            write_neg_scores(neg_scores, os.path.join(neg_scores_source_dir, fname+".csv"))
            
            # define threshold as the 5th negative energy quantile
            thresholds[a][m] = torch.quantile(neg_scores, 0.05).item()
            
            # draw and save the threshold plot
            plot_thr(
                neg_scores, 
                thresholds[a][m], 
                "Negative Energy Score Distribution (T={temperature})", 
                f"CIFAR-10:\n{MODES[m]}, {AUG_MODES[a]}", 
                os.path.join(plots_source_dir, fname+".png")
            )

        # draw and save all 6 thresholds on a single plot
        plot_all_thr(
            neg_scores_source, 
            thresholds, 
            f"Negative Energy Score Distribution (T={temperature})", 
            os.path.join(plots_source_dir, "cifar_distr.png")
        )
        # save thresholds
        with open(os.path.join(neg_scores_source_dir,"thresholds.json"),'w') as fout:
            json.dump(thresholds, fout)
        

        # STEP 2: Test distributions
        print(">>>>>> STEP 2. Test Disitributions")
        neg_scores_test = dict()
        # iterate over test datasets
        for test_ds in test_logits:
            # compute negative energy scores
            neg_scores = compute_neg_ood_score(test_logits[test_ds],temperature)
            neg_scores_test[test_ds] = neg_scores
        
            fname = f"neg_score_{test_ds}_{MODEL}"
            write_neg_scores(neg_scores, os.path.join(neg_scores_test_dir, fname+".csv"))
            

        model_errors = []
        # iterate over combinations of CIFAR10-test and OOD datasets
        for it, names in enumerate(TEST_COMBS):
            id_name, ood_name = names
            print(f">>>>>>>> {id_name} - {ood_name}")

            # load the corresponding energy values
            id_scores = neg_scores_test[id_name]
            ood_scores = neg_scores_test[ood_name]
            # filter energies depending on the mode
            id_scores, ood_scores = filter_scores(
                id_scores, ood_scores, ood_name, comp_mode
            )

            # 1. plot ID and OOD distributions       
            print("- Plot")     
            legend = [DS2COL[id_name], DS2COL[ood_name]]
            plot_test(
                id_scores,ood_scores,legend,
                f"Negative Energy Score Distribution (T={temperature})",
                os.path.join(plots_test_dir,f"{id_name}_{ood_name}.png")
            )

            # 2. plot ID and OOD distributions with the thresholds
            print("- Thresholded plot")
            legend += [
                '$\\tau$: Train, Original','$\\tau$: Valid, Original','$\\tau$: Gen, Original',
                '$\\tau$: Train, Augmented','$\\tau$: Valid, Augmented','$\\tau$: Gen, Augmented',
            ]
            plot_test_thr(
                id_scores,ood_scores,legend,
                f"Negative Energy Score Distribution (T={temperature})",
                os.path.join(plots_test_dir,f"{id_name}_{ood_name}_threshold.png"),
                thresholds
                )
            
            # 3. calculate errors for each of the thresholds
            print("- Errors")
            pair_errors = []
            for a in ['no_aug','aug']:
                for m in ['train','valid','gen']:
                    e1, e2 = predict_for_pair(
                        id_scores, 
                        ood_scores, 
                        thresholds[a][m], 
                        equal_size=False, 
                        T=temperature
                    )

                    pair_errors.append(e1)
                    pair_errors.append(e2)
            model_errors.append(pair_errors)
        
        # save errors in a table
        errors_ft = pd.DataFrame(
            data=np.array(model_errors),
            index=[f"CIFAR10-{ood_set_name.upper()}" for ood_set_name in TEST_SETS[1:]],
            columns=pd.MultiIndex.from_product([
                ['Original','Augmented'],
                ['Train','Valid','Generated'],
                ['1-TPR','FPR']
            ])
        )
        errors_ft.to_csv(os.path.join(neg_scores_test_dir, "errors.csv"))
            

def main(args):
    produce_neg_ood_scores_model(args.run_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str)
    args = parser.parse_args()
    main(args)
