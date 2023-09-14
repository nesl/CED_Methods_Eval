from typing import Optional, Sequence
import numpy as np
import random
from tqdm import tqdm

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from sklearn.metrics import f1_score, precision_score, recall_score


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl

def Focal_loss(logits, labels, alpha=None, gamma=2):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bc_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class Loss(torch.nn.Module):
    def __init__(
        self,
        loss_type: str = "cross_entropy",
        beta: float = 0.999,
        fl_gamma=2,
        samples_per_class=None,
        class_balanced=False,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf

        Args:
            loss_type: string. One of "focal_loss", "cross_entropy",
                "binary_cross_entropy", "softmax_binary_cross_entropy".
            beta: float. Hyperparameter for Class balanced loss.
            fl_gamma: float. Hyperparameter for Focal loss.
            samples_per_class: A python list of size [num_classes].
                Required if class_balance is True.
            class_balanced: bool. Whether to use class balanced loss.
        Returns:
            Loss instance
        """
        super(Loss, self).__init__()

        if class_balanced is True and samples_per_class is None:
            raise ValueError("samples_per_class cannot be None when class_balanced is True")

        self.loss_type = loss_type
        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced

    def forward(
        self,
        logits: torch.tensor,
        labels: torch.tensor,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.

        Args:
            logits: A float tensor of size [batch, num_classes].
            labels: An int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """
        if logits.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = logits.shape[1]
            logits = logits.permute(0, *range(2, logits.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            labels = labels.view(-1)
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        labels_one_hot = F.one_hot(labels, num_classes).float()

        if self.class_balanced:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * num_classes
            weights = torch.tensor(weights, device=logits.device).float()

            if self.loss_type != "cross_entropy":
                weights = weights.unsqueeze(0)
                weights = weights.repeat(batch_size, 1) * labels_one_hot
                weights = weights.sum(1)
                weights = weights.unsqueeze(1)
                weights = weights.repeat(1, num_classes)

        else:
            weights = None

        if self.loss_type == "focal_loss":
            cb_loss = Focal_loss(logits, labels_one_hot, alpha=weights, gamma=self.fl_gamma)
        elif self.loss_type == "cross_entropy":
            cb_loss = F.cross_entropy(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "binary_cross_entropy":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax_binary_cross_entropy":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss


def stats(label, results_estimated):
    # label = np.concatenate(label, 0)
    # results_estimated = np.concatenate(results_estimated, 0)
    label_estimated = np.argmax(results_estimated, 1)
    f1 = f1_score(label, label_estimated, average='weighted')
    acc = np.sum(label == label_estimated) / label.size
    return acc, f1


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


def create_src_causal_mask(sz):
    src_mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(torch.bool)
    return src_mask


def train(model, data_loader, n_epochs, lr, criterion, src_mask=None, device='cpu'):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.to(device)

    summary = {'loss': [[] for _ in range(n_epochs)], 'acc': [[] for _ in range(n_epochs)]}
    for e in tqdm(range(n_epochs)):
        for i, (data, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            data = data.to(device)
            labels = labels.to(device)
    #         labels = labels.squeeze(-1).to(device)

            # Run the Net
            if src_mask is not None: # For transformer
                src_mask = src_mask.to(device)
                x = model(data, src_mask)
            else:
                x = model(data)

            x = x.transpose(-1,1) # sequence cross entropy loss accepts input of dimension (N, C, L)

            # Optimize net
            criterion = criterion.to(device)
            loss = criterion(x, labels)
            loss.backward()
            optimizer.step()
            summary['loss'][e].append(loss.item())

                # Calculat accuracy
            _, pred = x.data.topk(1, dim=1)
    #         print(pred.squeeze(1).shape,labels.shape)
            pred = pred.squeeze(1)
            acc = torch.sum(pred == labels)/(x.shape[0] * x.shape[-1])
            summary['acc'][e].append(acc.item())
        print(pred)
        print('Loss: {}, Accuracy: {}'.format(np.mean(summary['loss'][e]), np.mean(summary['acc'][e])))


    return summary


def test(model, data_loader, criterion, src_mask=None):
    model.eval()
    model.to('cpu')
    summary = {'loss': [] , 'acc': []}

    all_pred = []
    all_labels = []

    for i, (data, labels) in enumerate(tqdm(data_loader)):
    #   data = data
    #   labels = labels.to(device)
        labels = labels.squeeze(-1)

        # Run the Net
        with torch.no_grad():
            if src_mask is not None: # For transformer
                src_mask = src_mask.to('cpu')
                x = model(data, src_mask)
            else:
                x = model(data)
            x = x.transpose(-1,1)

        # Optimize net
        criterion = criterion.to('cpu')
        loss = criterion(x, labels)
        summary['loss'].append(loss.item())

            # Calculat accuracy

        _, pred = x.data.topk(1, dim=1)
    #     print(pred.shape,labels.reshape(-1).shape)
        pred = pred.squeeze(1)
    #     if max(pred[0]) > 0:

        for j in range(len(pred)):
            print("pred:",pred[j])
            print("label:", labels[j])
        acc = torch.sum(pred == labels)/(x.shape[0] * x.shape[-1])
        summary['acc'].append(acc.item())
        all_pred.append(pred.reshape(-1))
        all_labels.append(labels.reshape(-1))

    all_pred = np.concatenate(all_pred)
    all_labels = np.concatenate(all_labels)

    f1_all = f1_score(all_labels, all_pred, average='macro')
    f1_pos = f1_score(all_labels, all_pred, labels=[1,2,3], average='macro')

    precision = precision_score(all_labels, all_pred, average=None)
    recall = recall_score(all_labels, all_pred, average=None)
    precision_avg = precision_score(all_labels, all_pred, average='macro')
    recall_avg = recall_score(all_labels, all_pred, average='macro')

    print('Loss: {}, Accuracy: {}, F1_all: {}, F1_positive: {}, Precision: {}, Avg_P: {}, Recall: {}, Avg_R: {}'.format(np.mean(summary['loss']),
                                                                                                                np.mean(summary['acc']),
                                                                                                                f1_all,
                                                                                                                f1_pos,
                                                                                                                precision,
                                                                                                                precision_avg,
                                                                                                                recall,
                                                                                                                recall_avg
                                                                                                                ))