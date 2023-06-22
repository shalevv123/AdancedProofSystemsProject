import numpy as np
import functools
import polynomial
import LDE
import galois

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

def theoretic_range(n):
    return range(1, n+1)

def GF_get_bits(z):
    bytes_z = reversed(z.tobytes())
    bits = []
    for byte in bytes_z:
        bits.extend(list(format(byte, 'b').zfill(8)))
    return tuple(int(bit) for bit in reversed(bits))

def int_get_bits(i):
    return tuple(int(c) for c in format(i, 'b')[::-1])

def get_at(seq, i):
    if i<len(seq): return seq[i]
    return 0


def get_delimiters(z:tuple, n=3):
    '''
        y[k]=prod_{i=1}^{m/2} I0(z[i], k[i])
        z ∈ F^m 
    '''
    m=len(z)
    dels = []
    for j in range(n):
        y=[]
        for k in theoretic_range(m//n):
            bits_k = int_get_bits(k)
            Is = (LDE.I0(get_at(z, i), get_at(bits_k, i-j*(m//n))) for i in range(j*(m//n), (j+1)*(m//n)))
            y.append(functools.reduce(lambda a,b: a*b, Is))
        dels.append(y)
    return dels

