import random
import torch

#my files
from GKR import GKR
from GKRProver import GKRProver
from GKRVerifier import GKRVerifier
from typing import List
from multilinear_extension import extend_sparse


class TensorCode():
    def __init__(self, C_0):
        self.C = lambda x: C_0(C_0(C_0(x)))

    def code(self, x):
        return self.C(x) if x.dim()==3 else None

    # def decode(self, C_x): # dont need it
    #     # TODO: some calculation
    #     x = 0 # Just for now
    #     return x

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
    def __init__(self, x, L, C : TensorCode, p):
        self.x = x
        self.L = L
        self.C = C
        self.p = p

    def talk_phase1(self, C_w):
        if self.C.local_test(C_w) is False:
            exit('This is a cheaty prover')
        self.C_w = C_w

    def talk_phase2(self, gkr: GKR, g : List[int]):
        gkr_verifier = GKRVerifier(gkr, g, s)
        l = 5 # what is the range of the indexes?
        idx = random.randint(0, l)

        data = {random.randint(0, (1 << L) - 1): random.randint(0, p - 1) for _ in range(256)}
        poly = extend_sparse(data, L, self.p)

    def talk_phase3(self, raw, v, k):
        col_check_idx = random.randint(0, k)

        # Test #1
        counter = 0
        for i in range(k):
            counter += v[i] * self.C_w[i, col_check_idx]
        if counter != raw[col_check_idx]:
            exit('This is a cheaty prover')

        # Test #2
        overall_sum = 0
        for i in range(k):
            overall_sum += raw[i]
        if overall_sum != _: # _ is the value we want to check if it is equal to
            exit('This is a cheaty prover')



class prover():
    def __init__(self, x, w, L, C, p):
        self.x = x
        self.w = w
        self.L = L
        self.C = C
        self.p = p

    def talk_phase1(self):
        return self.C(self.w)

    def talk_phase2(self, gkr: GKR, g : List[int]):
        gkr_proover = GKRProver(gkr)
        A_hg, G, s = gkr_proover.initializeAndGetSum(g)
        return A_hg, G, s

    def talk_phase3(self, u, k):
        matrix = self.C(self.w)
        raw = torch.zeroes(matrix.size(dim=1))
        for i in range(k):
            raw += u[i] * matrix[i]
        return raw



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

    v = verifier(x, L, C, p)
    p = prover(x, w, L, C, p)

    # talk_phase1:
    # The prover:   send to the verifier C(w)
    # The verifier: checks it with LDT
    C_w = p.talk_phase1()
    v.talk_phase1(C_w)

    # talk_phase2:
    # The prover:   GKR
    # The verifier: GKR
    gkr = GKR(...); g = []
    p.talk_phase2(gkr, g)
    v.talk_phase2(gkr, g)

    # Example of use from testGKR_Prover.py:
    # gkr = randomGKR(L, p)
    # g = [random.randint(0, p - 1) for _ in range(L)]
    # pv = GKRProver(gkr)
    # A_hg, G, s = pv.initializeAndGetSum(g)
    # v = GKRVerifier(gkr, g, s)
    # assert v.state == GKRVerifierState.PHASE_ONE_LISTENING, "Verifier sanity check failed"
    # pv.proveToVerifier(A_hg, G, s, v)
    # self.assertEqual(v.state, GKRVerifierState.ACCEPT)
    # print(f"\b\b\b\b\b\b\bPASS")

    raw = p.talk_phase3(u, k)  # coeff_matrix = u * v, how do we determine k?
    v.talk_phase3(raw, v, k)
