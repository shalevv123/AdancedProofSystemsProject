from TensorCode import C0, TensorCode

field_size = 17
basecode = C0

tensor_code = TensorCode(C0)

NUM_ROWS_TO_CHECK = 3

phi = [(1,2,3), (-2, 0,3), (-1, 2, -3)] # w.l.o.g the variables are numberd from 0 to n-1
z = [1, 1, 0, 1] # the assignment is a list (or a tuple) of length n