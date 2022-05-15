import numpy as np
import torch
import torch.nn.functional as F


def compute_el2n_score(outputs, Y):
    Y = torch.unsqueeze(Y, dim=-1)
    errors = F.softmax(outputs, dim = -1) - Y
    errors = errors.detach().numpy()
    scores = np.linalg.norm(errors, ord = 2, axis = -1)
    return scores