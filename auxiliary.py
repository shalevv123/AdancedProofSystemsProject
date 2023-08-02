import numpy as np
import functools
import LDE


def fancy_zip(l1, l2):
    i = 0
    for e1, e2 in zip(l1, l2):
        yield e1, e2
        i += 1
    if i < len(l2):
        for k in range(i, len(l2)):
            yield np.array([]), l2[k]

def prod(seq):
    return functools.reduce(lambda a,b: a*b, seq)

def innerProduct_sums(m1, m2):
    for v in reversed(m2):
        m1 = v @ m1
    return m1

def theoretic_range(n):
    return range(1, int(n+1))

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


def get_delimiters(z: tuple, n=2):
    #print('z: ' + str(z))
    m = len(z)
    #print('m: ' + str(m))
    M = 2**m
    #print('M: ' + str(M))
    dels = []
    for j in range(n):
        y = []
        for k in range(int(np.round(np.power(M, (1. / n))))):
            #print('int(np.round(np.power(M, (1. / n)))): ' + str(int(np.round(np.power(M, (1. / n))))))
            #print('k: ' + str(k))
            bits_k = int_get_bits(k)
            #print('bits_k: ' + str(bits_k))
            Is = tuple(LDE.I0(get_at(z, i), get_at(bits_k, i - j * (m // n))) for i in range(j * (m // n), (j + 1) * (m // n)))
            #print('Is: ' + str(Is))
            non_empty_arrays = [arr for arr in Is if np.any(arr)]  # Filter out empty arrays
            if non_empty_arrays:
                y.append(prod(non_empty_arrays))
        dels.append(y)
    #print('dels: ' + str(dels))
    return tuple(reversed(dels))

