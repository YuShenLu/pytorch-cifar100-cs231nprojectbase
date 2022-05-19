import numpy as np
from conf import settings

from el2n_data_loader import get_training_dataloader_el2n

KEEP_PERCENT = 0.75

scores = np.load('el2n/resnet18/Thursday_19_May_2022_13h_52m_46s/ckpt.npy')
num_keep = int(KEEP_PERCENT * len(scores))

lowest_scoring = np.sort(scores)[:num_keep]
lowest_scoring_indices = np.argsort(scores)[:num_keep]

mask = np.zeros(len(scores), dtype=bool)
mask[lowest_scoring_indices] = True

cifar100_training_loader = get_training_dataloader_el2n(
    settings.CIFAR100_TRAIN_MEAN,
    settings.CIFAR100_TRAIN_STD,
    num_workers=4,
    batch_size=128,
    mask = mask,
    shuffle=False
)