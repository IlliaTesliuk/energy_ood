import torch 
import numpy as np
import pandas as pd
from tdqm import tqdm

# reads logits from a CSV file 
def load_logits(full_path):
  logits_df = pd.read_csv(full_path,header=None)
  return torch.tensor(logits_df.values)

# computes energy scores
def compute_ood_score(logits, T):
    return -(T * torch.logsumexp(logits / T, dim=1))

# computes negative energy scores
def compute_neg_ood_score(logits, T):
    scores = compute_ood_score(logits, T)
    neg_scores = scores * (-1)
    return neg_scores 

# write negative energy scores to a CSV file
def write_neg_scores(neg_scores, full_path):
    pd.DataFrame(neg_scores).to_csv(full_path,index=False,header=False)

# reads negative energy scores from a CSV file
def load_neg_scores(full_path):
    neg_scores_np = pd.read_csv(full_path, header=None).to_numpy()
    neg_scores = torch.from_numpy(neg_scores_np).squeeze()
    return neg_scores

# produces logits for given classifier and dataset
def compute_logits(model, dataloader, include_labels, device, temp=1.0):
    total_logits, total_neg_scores = [], []
    for images in tqdm(dataloader):
        if include_labels:
            images = images[0]
        
        images = images.to(device)
        logits = model(images).detach().cpu()
            
        total_logits.append(logits)
        total_neg_scores.append(compute_neg_ood_score(logits, temp).numpy())

    total_logits = np.concatenate(total_logits, axis=0)
    total_neg_scores = np.concatenate(total_neg_scores, axis=0)
    return total_logits, total_neg_scores


