from TensorCode import C0, TensorCode
import galois

DEBUG = True

def count_vars(formula):
    vars = set()
    for f_clause in formula:
        for var in f_clause:
            vars.add(abs(var))
    return len(vars)

field_size = 17

F = galois.GF(2**field_size)
baseSubset = F((0,1))

basecode = C0

tensor_code = TensorCode(C0)

NUM_ROWS_TO_CHECK = 3

phi = [(1,2,3), (-2, 4,3), (-1, 2, -3)] # w.l.o.g the variables are numberd from 1 to n
z = [1, 0, 1, 1] # the assignment is a list (or a tuple) of length n
n = count_vars(phi)