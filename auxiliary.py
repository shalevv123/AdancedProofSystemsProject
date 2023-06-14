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

def GF_get_bits(z):
    bytes_z = reversed(z.tobytes())
    bits = []
    for byte in bytes_z:
        bits.extend(list(format(byte, 'b').zfill(8)))
    return tuple(int(bit) for bit in reversed(bits))

def int_get_bits(i):
    bits = []
    while i:
        bits.append(i%2)
        i//=2
    return tuple(int(bit) for bit in reversed(bits))

def get_at(seq, i):
    if i<len(seq): return seq[i]
    return 0


def get_delimiters(z, n=3):
    '''
        y[k]=prod_{i=1}^{m/2} I0(z[i], k[i])
    '''
    dels = []
    for i in range(n):
        pass

