import random
import galois
import polynomial
import numpy as np

# my files
# from GKR import GKR
# from GKRProver import GKRProver
# from GKRVerifier import GKRVerifier
# from typing import List
# from multilinear_extension import extend_sparse
from LDE import lde, I
from CNF3 import eval, witness, clouse


class TensorCode:
    def __init__(self, C_0):
        self.C = lambda x: C_0(C_0(C_0(x)))

    def code(self, x):
        return self.C(x) if x.dim() == 3 else None

    # def decode(self, C_x): # dont need it
    #     # TODO: some calculation
    #     x = 0 # Just for now
    #     return x

    def local_test(self, C_x):
        size = C_x.size()
        x = random.choice(range(size(dim=0)))
        y = random.choice(range(size(dim=1)))
        z = random.choice(range(size(dim=2)))
        def check(x): return self.decode(x) is not None
        return check(x) and check(y) and check(z)

    def local_decode(self, C_x, idx):
        x, y, z = idx
        # TODO: some check
        return C_x[x, y, z]


class ProofException(Exception):
    pass


# Phase 1 structures:


class CodeWordVerifier:
    def __init__(self, code):
        self.code = code

    def receive(self, codeword):
        if self.code.local_test(codeword) is False:
            raise ProofException('This is a cheaty prover')


class CodeWordProver:
    def __init__(self, witness, code):
        self.codeword = code(witness)

    def receive(self):
        return self.codeword


class CodeWordProof:
    def __init__(self, witness, code):
        self.verifier = CodeWordVerifier(code)
        self.prover = CodeWordProver(witness, code)
        self.codeword = None

    def prove(self):
        self.codeword = self.prover.receive()
        self.verifier.receive(self.codeword)


# phase 2 structures:


class AllZeroVerifier:
    def __init__(self):
        self.done = None
        self.rs = []
        self.alpha = 0

    @staticmethod
    def randomFieldElement(gf, m):
        return gf(np.random.randint(0, gf.order, m, dtype=int))

    def receive(self, polynomial, gf, n):
        # check the poly is zero, return random field member
        # if this is the last phase return the indexes for phase 3
        if len(self.rs) == 0:
            # sample randome element from the field
            self.m = int(np.ceil(3*np.log2(n)+3))
            self.z = self.randomFieldElement(gf, self.m)
            return self.z
        else:
            if not polynomial(0)+polynomial(1) == self.alpha:
                self.done = False     
            self.m -= 1
            r_i = self.randomFieldElement(gf, 1)
            self.rs.append(r_i)
            if self.m == 1:
                self.done = True
            self.alpha = polynomial(r_i)
            return r_i

class AllZeroProver:
    def __init__(self, formula, w, n):
        def phi(*args):
            return clouse(formula, *args)
        phi_hat = lde(f=phi, H=(0, 1), m=int(3*np.ceil(np.log2(n))+3))
        w_hat = lde(f=witness(w), H=(0, 1), m=int(np.ceil(np.log2(n))))
        # calculate p

        def P(*args):
            b1, b2, b3 = args[-3:]
            phi_hat_res = phi_hat(*args)
            i_s = args[:-3]
            size = len(i_s)//3
            i1, i2, i3 = i_s[:size], i_s[size:2*size], i_s[2*size:]
            w1, w2, w3 = (w_hat(i1)-b1), (w_hat(i2)-b2), (w_hat(i3)-b3)
            return phi_hat_res*w1*w2*w3
        self.n = n
        self.P = P
        self.round = 0
        self.Q = None
        self.m = int(3*np.ceil(np.log2(self.n))+3)
    
    @staticmethod
    def intToBits(i):
        return [int(digit) for digit in bin(i)[2:]] # [2:] to chop off the "0b" part 
    
    def sumGF2(self, x, m):
        return sum(self.Q(x, *self.intToBits(i)) for i in range(2**m))

    def receive(self, z):
        # do the all zero calculation and return the polynomial.
        if self.Q is None:
            self.Q = lambda *args: self.P(*args)*I(args, z, H=(0,1), m=self.m)
        else:
            self.Q = lambda *args: self.Q(z, *args)
        self.m-=1
        return lambda x: self.sumGF2(x, self.m) # Q tilda
        


class AllZeroProof:
    def __init__(self, formula, witness, n, gf):
        self.V = AllZeroVerifier()
        self.P = AllZeroProver(formula, witness, n)
        self.n = n
        self.gf = gf

    def prove(self):
        # implement the all zero proof
        Q = None
        r_i =  self.V.receive(Q, self.gf, self.n)
        while self.V.done is None:
            # prover
            Q = self.P.receive(r_i)
            # verifier
            r_i = self.V.receive(Q, self.gf, self.n)
        return self.V.done, self.V.z, self.V.rs, Q

# Phase 3 structures:
'''
    now check if indeed Q_z(rs) equals Q(rs) sent by the all-zero prover in the last phase
    * Q_z(rs)=P(rs)*I(rs,z)
    * P(i1, i2, i3, b1, b2, b3)=phi(i1, i2, i3, b1, b2, b3)*(w(i1)-b1)*(w(i1)-b2)*(w(i3)-b3)
'''
#TODO: how to check if a string is a legitimate codeword?

class LDEVerifier:
    def __init__(self, formula, index_value_list, code, code_word):
        # TODO: Assaf
        pass

    def receive(self):  # TODO: complete inputs
        # TODO: Assaf
        pass


class LDEProver:
    def __init__(self):  # TODO: complete inputs
        pass

    def receive(self):  # TODO: complete inputs
        # TODO: Assaf
        pass


class LDEProof:
    def __init__(self):  # TODO: complete inputs
        pass

    def prove(self):  # TODO: complete inputs
        # TODO: Assaf
        pass


class MainProof:
    def __init__(self, formula, witness, code):
        # create all the verifiers provers and proofs
        pass

    def prove(self):
        # save relevant info between the steps and run all the proofs.
        pass


'''
the protocol:
prover:                              verifier:
                -> C(w) ->
        ------------------------------
          run _all zero check_ from hw2
          on the polynomial P (from class)
        ------------------------------
          linearity check on C(w)
        ------------------------------
                                            ACK/REJ
'''
