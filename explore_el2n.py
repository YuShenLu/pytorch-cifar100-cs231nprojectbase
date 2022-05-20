import numpy as np
from conf import settings

from el2n_data_loader import get_training_dataloader_el2n

KEEP_PERCENT = 0.75

scores = np.load('el2n/resnet18/Thursday_19_May_2022_13h_52m_46s/ckpt.npy')
num_keep = int(KEEP_PERCENT * len(scores))

highest_scoring_indices = np.argsort(scores)[::-1][:num_keep]

mask = np.zeros(len(scores), dtype=bool)
mask[highest_scoring_indices] = True