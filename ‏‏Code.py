
class Code:
    pass
'''
TODO:
    - local testability of Tensor code
    - lincheck (move to Tensor code)
    - code
    - local decode of Tensor code(i -> i,j,k)
    - 
'''

class TensorCode:
    def __init__(self, C_0):
        # self.C = lambda x: C_0(C_0(C_0(x)))
        pass

    def code(self, x):
        # return self.C(x) if x.dim() == 3 else None
        # TODO: return some np-array
        raise NotImplemented

    # def decode(self, C_x): # dont need it
    #     # TODO: some calculation
    #     x = 0 # Just for now
    #     return x

    def local_test(self, C_x):
        # size = C_x.size()
        # x = random.choice(range(size(dim=0)))
        # y = random.choice(range(size(dim=1)))
        # z = random.choice(range(size(dim=2)))
        # def check(x): return self.decode(x) is not None
        # return check(x) and check(y) and check(z)
        pass

    def local_decode(self, C_x, idx):
        # x, y, z = idx
        # # TODO: some check
        # return C_x[x, y, z]
        pass
    # add lin-check here