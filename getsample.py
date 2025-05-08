
import numpy as np
import random


def get_all_samples(conjunction):

    pos = []
    neg = []
    for index in range(conjunction.shape[0]):
        for col in range(conjunction.shape[1]):
            if conjunction[index, col] == 1:
                pos.append([index, col, 1])
            else:
                neg.append([index, col, 0])
    pos_len = len(pos)
    new_neg = random.sample(neg, pos_len)
    samples = pos + new_neg
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    return samples



