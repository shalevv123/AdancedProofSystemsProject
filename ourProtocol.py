import random
import galois
import polynomial
import numpy as np
from enum import Enum
from auxiliary import *
from TensorCode import TensorCode, LinCode
from LDE import lde, I
from CNF3 import eval, witness, clause
from parameters import *

Status = Enum('Status', ['IN_PROCCESS', 'ACK', 'REJ'])

class ProofException(Exception):
    pass


# Phase 1 structures:


class CodeWordVerifier:
    def __init__(self, code):
        self.code = code

    def next_msg(self, codeword):
        if self.code.local_test(codeword) is False:
            raise ProofException('This is a cheaty prover')


class CodeWordProver:
    def __init__(self, witness, code):
        self.codeword = code(witness)

    def next_msg(self):
        return self.codeword


class CodeWordProof:
    def __init__(self, witness, code):
        self.verifier = CodeWordVerifier(code)
        self.prover = CodeWordProver(witness, code)
        self.codeword = None

    def prove(self):
        self.codeword = self.prover.next_msg()
        self.verifier.next_msg(self.codeword)


# phase 2 structures:


class AllZeroVerifier:
    def __init__(self):
        self.status = None
        self.z = None
        self.rs = []
        self.alpha = 0

    @staticmethod
    def randomFieldElementVector(gf, m=1):
        return gf(np.random.randint(0, gf.order, m, dtype=int))

    def receive(self, polynomial, gf, n):#TODO: move n to c'tor
        # check the poly is zero, return random field member
        # if this is the last phase return the indexes for phase 3
        if self.z is None:
            # sample randome elements vector from the field
            self.m = int(3*np.ceil(np.log2(n))+3) #TODO: move to c'tor
            self.z = self.randomFieldElementVector(gf, self.m)
            return self.z
        else:
            if not polynomial(0)+polynomial(1) == self.alpha:
                self.status = False
                return 0    
            # self.m -= 1
            r_i = self.randomFieldElementVector(gf, 1)
            self.rs.append(r_i)
            if self.m - len(self.rs) == 1:
                self.status = True
            self.alpha = polynomial(r_i)
            return r_i

class AllZeroProver:
    def __init__(self, formula, w, n):
        def phi(*args):
            return clause(formula, *args)
        phi_hat = lde(f=phi, H=(0, 1), m=int(3*np.ceil(np.log2(n))+3))#TODO: check H with GF
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
        return sum(self.Q(x, *int_get_bits(i)) for i in range(2**m))

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
        while self.V.status is None:
            # prover
            Q = self.P.receive(r_i)
            # verifier
            r_i = self.V.receive(Q, self.gf, self.n)
        return self.V.status, self.V.z, self.V.rs, Q

# Phase 3 structures:
'''
    now check if indeed Q_z(rs) equals Q(rs) sent by the all-zero prover in the last phase
    * Q_z(rs)=P(rs)*I(rs,z)
    * P(i1, i2, i3, b1, b2, b3)=phi(i1, i2, i3, b1, b2, b3)*(w(i1)-b1)*(w(i2)-b2)*(w(i3)-b3)
    * i1,i2,i3 \in F^m
'''

NUM_ROWS_TO_CHECK = 3
class LinVerifier:
    '''
        IP for verifing that indeed Q_z(rs) equals Q(rs) sent by the all-zero prover in the last phase
    '''
    def __init__(self, n, formula, code, codeword: LinCode, z, r_s):
        pass
class LinVerifier_aux:
    '''
        IP for verifing that w_hat(index_value)=value
        w (formula) is encoded using code to obtain codeword
    '''
    def __init__(self, n, formula, index_value, codeword:LinCode, value):
        self.code_word = codeword.trunc(n)
        self.delimiter_vecs = get_delimiters(index_value) # 2-D
        self.formula = formula
        self.index_value = index_value
        self.code_word = codeword
        self.value = value
        self.status = Status.IN_PROCCESS

    def receive(self, partial_sums):
        if self.delimiter_vecs[-1] @ partial_sums != self.value:
            self.status = Status.REJ
            return
        for _ in range(NUM_ROWS_TO_CHECK):
            index = tuple(np.random.randint(dim) for dim in self.code_word.shape[:-1])
            last_dim_sum = self.delimiter_vecs[0] @ self.code_word[:,index]
            if last_dim_sum != partial_sums[(index, )]:
                self.status = Status.REJ
                return
        self.status = Status.ACK
        #TODO: what do we return?

        
        


class LinProver:
    def __init__(self, code_word:LinCode, index_value):
        self.code_word = code_word.codeword
        self.delimiter_vec = get_delimiters(index_value) # 2-D

    def receive(self):
        return self.delimiter_vec[0] @ self.code_word


class LinProof:
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

def main():
    def C_0(x):
        return x
    formula = ''
    witness = ''
    code = TensorCode(C_0, dim=2)


'''
the protocol:
prover:                              verifier:
                -> C(w) ->
        ------------------------------
          run _all zero check_ from hw2
          on the polynomial P (from class)
        ------------------------------
          lin-check on C(w)
        ------------------------------
                                            ACK/REJ
'''

# aim for soudness error 2**-32