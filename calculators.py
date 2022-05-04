import numpy as np
import math
import collections

class RelativeProbabilityCalculator:
    def __init__(self, device, sampling_min=0, history_length=1024, beta=3):
        self.device = device
        self.sampling_min = sampling_min
        self.beta = beta
        self.history_length = history_length
        self.recent_losses = collections.deque(maxlen=self.history_length)

    def append(self, loss):
        self.recent_losses.append(loss)

    def append_losses(self, losses):
        for loss in losses:
            self.recent_losses.append(loss)

    def calculate_probability(self, loss):
        count = 0
        for i in self.recent_losses:
            if i < loss:
                count += 1
        percentile = count * 100. / len(self.recent_losses)
        return max(self.sampling_min, math.pow(percentile / 100., self.beta))

    def get_probability(self, examples):
        losses = [em for em in examples]
        self.append_losses(losses)
        probs = [self.calculate_probability(loss) for loss in losses]
        return probs


