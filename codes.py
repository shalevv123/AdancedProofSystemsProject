import numpy
from numpy import random
from numpy import moveaxis


class basecode:
    @staticmethod
    def code(x):
        return x  # TODO: change to real code

    @staticmethod
    def test(x):
        return True  # TODO: change to real test


'''
TODO:
    - local testability of Tensor code
    - lincheck (move to Tensor code)
    - code
    - local decode of Tensor code(i -> i,j,k) DONE
    - 
'''


def oneDimIndexToNDimIndex(index: int, nDims, dimSizes):
    if index == 0:
        return [0] * nDims
    n = index
    res = []
    for i in range(nDims):
        res.append(n % dimSizes[i])
        n //= dimSizes[i]
    return tuple(res)


class TensorCode:
    def __init__(self, C_0, dim=3):
        self.dim = dim
        self.C = lambda x: x
        for i in range(dim):
            self.C = lambda x: C_0(self.C(x))
        pass

    def code(self, x):
        # return self.C(x) if x.dim() == 3 else None
        # TODO: return some np-array
        raise NotImplemented

    def local_test(self, C_x):
        # get a row from the n dimensional matrix C_x, then choose a random column along that row, then chose a random
        # depth along that column, and test them all to be codewords
        dims = C_x.shape
        C_x_copy = C_x.copy()
        indices = [random.randint(0, dims[i]) for i in range(len(dims))]
        for i, dim in enumerate(dims):
            array = []
            for j in range(dim):
                array.append(C_x_copy[tuple([j]) + tuple(indices[1:])])
            if not basecode.test(numpy.array(array)):
                return False
            C_x_copy = moveaxis(C_x_copy, 0, -1)
            indices = indices[1:] + [indices[0]]
            indices[-1] = random.randint(0, dims[i])
        return True

    def local_decode(self, C_x, idx):
        return C_x[oneDimIndexToNDimIndex(idx, self.dim, C_x.shape)]
