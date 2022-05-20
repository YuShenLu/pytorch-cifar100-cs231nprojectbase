import numpy as np
import torch
import torch.nn.functional as F


def compute_el2n_score(outputs, Y):
    one_hot = torch.zeros_like(outputs)
    rows = torch.arange(len(Y))
    one_hot[rows, Y] = 1
    errors = F.softmax(outputs, dim = -1) - one_hot
    errors = errors.detach().cpu().numpy()
    scores = np.linalg.norm(errors, ord = 2, axis = -1)
    return scores