import random
import galois
import polynomial
import itertools
import CNF3
import numpy as np
from enum import Enum
from auxiliary import get_delimiters, int_get_bits, prod
from TensorCode import TensorCode, LinCode
from LDE import lde, I
from CNF3 import eval, witness, clause
from parameters import phi, z, tensor_code, NUM_ROWS_TO_CHECK
from field import F, baseSubset

Status = Enum('Status', ['IN_PROCCESS', 'ACC', 'REJ'])

class ProofException(Exception):
    pass


# Phase 1 structures:


class CodeWordVerifier:
    def __init__(self, code):
        self.code = code
        self.status = Status.IN_PROCCESS

    def receive(self, codeword):
        if self.code.test(codeword) is False:
            # raise ProofException('This is a cheaty prover')
            self.status = Status.REJ
            return
        self.status = Status.ACC
        return codeword

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
        self.codeword = self.verifier.receive(self.codeword)
        if self.verifier.status != Status.ACC:
            raise ProofException('1')
        return self.codeword


# phase 2 structures:


class AllZeroVerifier:
    def __init__(self, gf, n):
        self.status = Status.IN_PROCCESS
        self.z = None
        self.n = n
        self.gf = gf
        self.rs = []
        self.alpha = 0

    @staticmethod
    def randomFieldElementVector(gf, m=1):
        return gf(np.random.randint(0, gf.order, m, dtype=int))

    def receive(self, polynomial):
        # check the poly is zero, return random field member
        # if this is the last phase return the indexes for phase 3
        if self.z is None:
            # sample randome elements vector from the field
            self.m = int(3*np.ceil(np.log2(self.n))+3) #TODO: move to c'tor
            self.z = self.randomFieldElementVector(self.gf, self.m)
            return self.z
        else:
            if not (polynomial(0)+polynomial(1) == self.alpha):
                self.status = Status.REJ
                return    
            # self.m -= 1
            r_i = self.randomFieldElementVector(self.gf, 1)
            self.rs.append(r_i)
            if self.m - len(self.rs) == 1:
                self.status = Status.ACC
            self.alpha = polynomial(r_i)
            return r_i

class AllZeroProver:
    def __init__(self, formula, w, n, gf):
        self.gf = gf
        def phi(*args):
            return clause(formula, *args)
        phi_hat = lde(f=phi, H=baseSubset, m=int(3*np.ceil(np.log2(n))+3))#TODO: check H with GF
        w_hat = lde(f=witness(w), H=baseSubset, m=int(np.ceil(np.log2(n))))
        # calculate p

        def P(*args):
            i1,i2,i3,b1,b2,b3 = CNF3.convert_args(args, to_bits=True)
            phi_hat_res = phi_hat(args)
            w1, w2, w3 = (w_hat(i1)-gf(b1)), (w_hat(i2)-gf(b2)), (w_hat(i3)-gf(b3))
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
        return sum((self.Q(x, *i) for i in itertools.product(baseSubset, repeat=m)), start=self.gf(0))

    def receive(self, z):
        # do the all zero calculation and return the polynomial.
        if self.Q is None:
            self.Q = lambda *args: self.P(*args)*I(args, z, H=baseSubset, m=self.m)
        else:
            self.Q = lambda *args: self.Q(z, *args)
        self.m-=1
        return lambda x: self.sumGF2(x, self.m) # Q tilda

class AllZeroProof:
    def __init__(self, formula, witness, n, gf):
        self.n = n
        self.gf = gf
        self.V = AllZeroVerifier(self.gf, self.n)
        self.P = AllZeroProver(formula, witness, n, self.gf)
        self.Q = None


    def prove(self):
        # implement the all zero proof
        Q = None
        r_i =  self.V.receive(Q)
        round_count = 0
        while self.V.status is Status.IN_PROCCESS:
            # prover
            Q = self.P.receive(r_i)
            if self.Q is None: self.Q = Q
            # verifier
            r_i = self.V.receive(Q)
            round_count += 1
        if self.V.status is Status.REJ:
            raise ProofException(f'2, {round_count}')
        return self.V.z, self.V.rs, Q

# Phase 3 structures:
'''
    now check if indeed Q_z(rs) equals Q(rs) sent by the all-zero prover in the last phase
    * Q_z(rs)=P(rs)*I(rs,z)
    * P(i1, i2, i3, b1, b2, b3)=phi(i1, i2, i3, b1, b2, b3)*(w(i1)-b1)*(w(i2)-b2)*(w(i3)-b3)
    * i1,i2,i3 \in F^m
'''


class LinVerifier_aux:
    '''
        IP for verifing that w_hat(index_value)=value
        w (formula) is encoded using code to obtain codeword
    '''
    def __init__(self, n, index_value, codeword:LinCode):
        self.code_word = codeword.trunc(n)
        self.delimiter_vecs = get_delimiters(index_value) # 2-D
        self.index_value = index_value
        self.code_word = codeword
        self.status = Status.IN_PROCCESS

    def receive(self, partial_sums):
        for _ in range(NUM_ROWS_TO_CHECK):
            index = tuple(np.random.randint(dim) for dim in self.code_word.shape[:-1])
            last_dim_sum = self.delimiter_vecs[0] @ self.code_word[:,index]
            if last_dim_sum != partial_sums[(index, )]:
                self.status = Status.REJ
                return
        value= self.delimiter_vecs[-1] @ partial_sums
        self.status = Status.ACC
        return value
  
class LinVerifier:
    '''
        IP for verifing that indeed Q_z(rs) equals Q(rs) sent by the all-zero prover in the last phase
    '''
    def __init__(self, n, formula, codeword: LinCode, z, r_s, Q):
        self.n = n
        self.r_s = r_s
        self.m = int(3*np.ceil(np.log2(self.n))+3)
        self.formula = formula
        self.phi_hat = phi_hat = lde(lambda *args: clause(self.formula, *args), H=baseSubset, m=self.m)
        self.b_s = tuple(a for a in r_s[-3:])
        size = len(r_s[:-3])//3
        self.i_s = tuple(r_s[i*size : (i+1)*size] for i in range(3))
        self.w_is = [None]*3
        self.z = z
        self.codeword = codeword
        self.counter = 0
        self.Q = Q
        self.status = Status.IN_PROCCESS

    def receive(self, partial_sums):
        if self.status == Status.REJ: return

        aux_verifier = LinVerifier_aux(self.n, self.i_s[self.counter], self.codeword)
        self.w_is.append(aux_verifier.receive(partial_sums))
        if aux_verifier.status != Status.ACC:
            self.status = Status.REJ
            return
        self.counter += 1
        if self.counter >=3:
            Qz = self.phi_hat(self.r_s)*(prod(wi-bi for wi,bi in zip(self.w_is, self.b_s)))*I(self.r_s, self.z)
            if Qz != self.Q(self.r_s):
                self.status = Status.REJ
                return
            self.status = Status.ACC

class LinProver:
    def __init__(self, code_word:LinCode, r_s):
        self.code_word = code_word.codeword
        size = len(r_s[:-3])//3
        self.i_s = tuple(r_s[i*size : (i+1)*size] for i in range(3))
        self.delimiters = tuple(get_delimiters(index_value) for index_value in self.i_s)
        self.counter = 0

    def receive(self):
        delimiter_vec = self.delimiters[self.counter]
        self.counter += 1
        return delimiter_vec[0] @ self.code_word

class LinProof:
    def __init__(self, n, formula, codeword, z, r_s, Q):
        self.V = LinVerifier(n, formula, codeword, z, r_s, Q)
        self.P = LinProver(codeword, r_s)

    def prove(self):
        while self.V.status is Status.IN_PROCCESS:
            partial_sums = self.P.receive()
            self.V.receive(partial_sums)
        if self.V.status is Status.REJ:
            raise ProofException('3')

def count_vars(formula):
    vars = set()
    for f_clause in formula:
        for var in f_clause:
            vars.add(abs(var))
    return len(vars)

class MainProof:
    def __init__(self, formula, witness, code=tensor_code, gf=F):
        # create all the verifiers provers and proofs
        self.formula = formula
        self.witness = witness
        self.n = count_vars(formula)
        self.code = code
        self.gf = gf
        # phase 1
        self.codeword_proof = CodeWordProof(self.witness, self.code)
        # phase 2
        self.all_zero_proof = AllZeroProof(self.formula, self.witness, self.n, self.gf)
        # phase 3
        self.lin_proof = None # not all data is present at this point, this will be initiialized in the proof
        pass

    def prove(self):
        # save relevant info between the steps and run all the proofs.
        # phase 1
        codeword = self.codeword_proof.prove()
        # phase 2
        z, r_s, Q = self.all_zero_proof.prove()
        # phase 3
        self.lin_proof = LinProof(self.n, self.formula, codeword, z, r_s, Q)
        self.lin_proof.prove()

def main():
    proof = MainProof(phi, z)
    try:
        proof.prove()
        print('proof succsessfull')
    except ProofException as e:
        print('proof failed at phase' , e)


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
                                            ACC/REJ
'''

# aim for soudness error 2**-32

if __name__=='__main__':
    main()