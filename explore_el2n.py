import numpy as np

KEEP_PERCENT = 0.75

scores = np.load('el2n/resnet18/Sunday_15_May_2022_16h_46m_23s/ckpt.npy')
num_keep = int(KEEP_PERCENT * len(scores))

lowest_scoring = np.sort(scores)[:num_keep]
lowest_scoring_indices = np.argsort(scores)[:num_keep]

print(lowest_scoring)
print(lowest_scoring_indices)