import random

#my files
from GKR import GKR
from GKRProver import GKRProver
from GKRVerifier import GKRVerifier


class TensorCode():
    def __init__(self, C_0):
        self.C = lambda x: C_0(C_0(C_0(x)))

    def code(self, x):
        return self.C(x) if x.dim()==3 else None

    def decode(self, C_x):
        # TODO: some calculation
        x = 0 # Just for now
        return x

    def local_test(self, C_x):
        size = C_x.size()
        x = random.choice(range(size(dim=0)))
        y = random.choice(range(size(dim=1)))
        z = random.choice(range(size(dim=2)))
        check = lambda x: self.decode(x) is not None
        return check(x) and check(y) and check(z)

    def local_decode(self, C_x, idx):
        x, y, z = idx
        # TODO: some check
        return C_x[x, y, z]



class verifier():
    def __init__(self, x, L, C : TensorCode):
        self.x = x
        self.L = L
        self.C = C

    def talk_phase1(self, C_w):
        if self.C.local_test(C_w) is False:
            exit('This is a cheaty prover')

    def talk_phase2(self, gkr: GKR):
        gkr_verifier = GKRVerifier(gkr, g, asserted_sum)



class prover():
    def __init__(self, x, w, L, C):
        self.x = x
        self.w = w
        self.L = L
        self.C = C

    def talk_phase1(self):
        return self.C(self.w)

    def talk_phase2(self, gkr: GKR):
        gkr_proover = GKRProver(gkr)



def runProtocol(x, w, L, p: int):
    """
        :param x:
        :param w:
        :param L:
        :param p: field size
    """
    # need to add assert on the dimensions

    C_0 = lambda x: x  # Just for now
    C = TensorCode(C_0)

    v = verifier(x, L, C)
    p = prover(x, w, L, C)

    # talk_phase1:
    # The prover:   send to the verifier C(w)
    # The verifier: checks it with LDT
    C_w = p.talk_phase1()
    v.talk_phase1(C_w)

    # talk_phase2:
    # The prover:   GKR
    # The verifier: GKR
    gkr = GKR(...)
    p.talk_phase2(gkr)
    v.talk_phase2(gkr)






