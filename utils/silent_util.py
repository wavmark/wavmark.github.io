import numpy as np


def is_silent(data, silence_threshold=0.01):
    rms = np.sqrt(np.mean(data ** 2))
    return rms < silence_threshold


def has_silent_part(trunck):
    num_part = 3
    part_length = int(len(trunck) / num_part)
    for i in range(num_part):
        start = part_length * i
        end = start + part_length
        mini_trunck = trunck[start:end]
        if is_silent(mini_trunck):
            return True
    return False
