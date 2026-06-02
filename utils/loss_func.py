import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations
import pdb


class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function combining survival prediction and stage classification
    """
    def __init__(self, survival_alpha=0.0, multitask_weight=0.3, eps=1e-7, reduction='sum'):
        super().__init__()
        self.survival_loss = NLLSurvLoss(alpha=survival_alpha, eps=eps, reduction=reduction)
        self.stage_loss = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore invalid stage labels
        self.multitask_weight = multitask_weight

    def __call__(self, survival_logits, stage_logits, y, t, c, stage_labels, logits_text, logit_te):
        """
        Parameters
        ----------
        survival_logits: (n_batches, n_survival_classes)
            The neural network output for survival prediction
        stage_logits: (n_batches, n_stage_classes)
            The neural network output for stage prediction
        y: (n_batches,)
            The true survival time bin label
        t: (n_batches,)
            The true survival time
        c: (n_batches,)
            The true censorship indicator
        stage_labels: (n_batches,)
            The true stage labels
        """
        # Survival loss
        surv_loss = self.survival_loss(survival_logits, y, t, c)
        
        # Stage classification loss (only for valid stage labels)
        valid_stage_mask = stage_labels != -1
        # if valid_stage_mask.sum() > 0:
        stage_loss = self.stage_loss(stage_logits[valid_stage_mask], stage_labels[valid_stage_mask])
        # else:
        #     stage_loss = torch.tensor(0.0, device=survival_logits.device)
        
        # Combined loss
        # total_loss = (1 - self.multitask_weight) * surv_loss + self.multitask_weight * stage_loss
        total_loss = surv_loss + self.multitask_weight * stage_loss
        
        if logits_text is not None and logit_te is not None:
            loss_text = self.survival_loss(logits_text, y, t, c)
            loss_te = self.survival_loss(logit_te, y, t, c)
            
            # kl_loss
            nde = logits_text
            p_te = torch.nn.functional.softmax(logit_te, -1).clone().detach()
            p_nde = torch.nn.functional.softmax(nde, -1)
            kl_loss = - p_te*p_nde.log()
            kl_loss = kl_loss.sum(1).mean()
            
        else:   
            kl_loss = 0
            loss_text = 0
            loss_te = 0
            
        other_loss = loss_text + loss_te + kl_loss # 改
        # other_loss = loss_text + loss_te
        total_loss += other_loss
        
        # return total_loss-surv_loss, surv_loss, stage_loss
        return total_loss, surv_loss, stage_loss
    


# TODO: document better and clean up
def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='sum'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        The weight on uncensored loss 
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # print("h shape", h.shape)

    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)
    # print("hazards shape", hazards.shape)

    S = torch.cumprod(1 - hazards, dim=1)
    # print("S.shape", S.shape, S)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # hazards[y] = hazards(1)
    # S[1] = S(1)
    # TODO: document and check

    # print("S_padded.shape", S_padded.shape, S_padded)


    # TODO: document/better naming
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    # print('s_prev.s_prev', s_prev.shape, s_prev)
    # print('h_this.shape', h_this.shape, h_this)
    # print('s_this.shape', s_this.shape, s_this)

    # c = 1 means censored. Weight 0 in this case 
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    

    # print('uncensored_loss.shape', uncensored_loss.shape)
    # print('censored_loss.shape', censored_loss.shape)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss