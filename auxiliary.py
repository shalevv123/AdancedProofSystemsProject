import numpy as np
import functools
import polynomial

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


def get_delimiters(z, m=None, n=3):
    '''
        y[k]=prod_{i=1}^{m/2} I0(z[i], k[i])
    '''
    bits_z = GF_get_bits(z)
    m=len(bits_z) if not m else m
    dels = []
    for j in range(n):
        y=[]
        for k in range(m//n):
            bits_k = int_get_bits(k)
            Is = (polynomial.I0(get_at(bits_z, i), get_at(bits_k, i)) for i in range(j*(m//n), (j+1)*(m//n)))
            y.append(functools.reduce(lambda a,b: a*b, Is))
        dels.append(y)
    return dels

