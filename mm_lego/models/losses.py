import torch
from torch import nn

def nll_loss(hazards, S, Y, c, weights=None, alpha=0.4, eps=1e-7):
    """
    hazards: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    Y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    if weights is not None:
        # Normalize weights and ensure it's a 2D tensor with the same number of rows as the input
        weights = weights / torch.sum(weights)
        weights = weights.view(1, -1).expand_as(hazards)
        # Use gather to select the weights corresponding to the target classes
        gathered_weights = torch.gather(weights, 1, Y)
        neg_l *= gathered_weights

    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


def l1_loss(model: nn.Module, l1: float):
    """
    L1 regularisation loss
    Args:
        model:
        l1:

    Returns:

    """
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    reg_loss = float(l1) * l1_norm
    return reg_loss