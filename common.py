import random
import numpy as np

def split_data(x, y, batch_size, shuffle=False):
    z = list(zip(x, y))
    if shuffle:
        random.shuffle(z)
    batches = []
    for i in range(0, len(z), batch_size):
        chunk = z[i:i+batch_size]
        x, y = zip(*chunk)
        batches.append((np.array(x), np.array(y)))
    return batches

def transform_to_one_hot(labels, num_outputs):
    z = np.zeros((labels.size, num_outputs))
    z[np.arange(labels.size), labels] = 1
    return z
