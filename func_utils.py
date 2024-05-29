from typing import Optional, Sequence
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# def stats(label, results_estimated):
#     # label = np.concatenate(label, 0)
#     # results_estimated = np.concatenate(results_estimated, 0)
#     label_estimated = np.argmax(results_estimated, 1)
#     f1 = f1_score(label, label_estimated, average='weighted')
#     acc = np.sum(label == label_estimated) / label.size
#     return acc, f1


class CEDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.data)
    

class MultiTaskCEDataset(Dataset):
    def __init__(self, data, labels, in_states, out_states):
        self.data = data
        self.labels = labels
        self.in_states = in_states
        self.out_states = out_states

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        in_state = self.in_states[index]
        out_state = self.out_states[index]
        return data, label, in_state, out_state

    def __len__(self):
        return len(self.data)


def create_src_causal_mask(sz):
    src_mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(torch.bool)
    return src_mask