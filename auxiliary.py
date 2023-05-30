import numpy as np

def fancy_zip(l1, l2):
    i = 0
    for e1, e2 in zip(l1, l2):
        yield e1, e2
        i += 1
    if i < len(l2):
        for k in range(i, len(l2)):
            yield np.array([]), l2[k]

def innerProduct_sums(m1, m2):
    for v in reversed(m2):
        m1 = m1 @ v
    return m1