import random

# my files
from GKR import GKR
from GKRProver import GKRProver
from GKRVerifier import GKRVerifier
from typing import List
from multilinear_extension import extend_sparse

class TensorCode:
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
        pass

    def receive(self, polynomial):
        # check the poly is zero, return random field member
        # if this is the last phase return the indexes for phase 3
        pass


class AllZeroProver:
    def __init__(self, formula, witness):
        # calculate LDE of p
        pass

    def receive(self, constant):
        # do the all zero calculation and return the polynomial.
        pass


class AllZeroProof:
    def __init__(self, formula, witness):
        # create verifier and prover
        pass

    def prove(self):
        # implement the all zero proof
        pass

# Phase 3 structures:


class LDEVerifier:
    def __init__(self, formula, index_value_list, code, code_word):
        # TODO: Assaf
        pass

    def receive(self):  # TODO: complete inputs
        # TODO: Assaf
        pass


class LDEProver:
    def __init__(self): # TODO: complete inputs
        pass

    def receive(self): # TODO: complete inputs
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


