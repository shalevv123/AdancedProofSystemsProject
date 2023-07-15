


def bits_to_int_clause(func):
    def inner(*args):
        b1,b2,b3 = args[-3:]
        phi = args[0]
        i_s = args[1:-3]
        size = len(i_s)//3
        i1, i2, i3 = convert(i_s[:size]), convert(i_s[size:2*size]), convert(i_s[2*size:])
        return func(phi, i1, i2, i3, b1, b2, b3)
    return inner

def bits_to_int_witness(func):
    def inner(*args):
        x = convert(args)
        return func(x)
    return inner

def eval(phi, z):
    '''
    phi is a 3-CNF formula represented as a list of tuples.
        each tuple represent a clause, and each element in the tuple can be a <number> or -<number> (negative number, representing "not") 
    '''
    for clause in phi:
        if not any(z[x] if x>=0 else not z[-x] for x in clause):
            return False
    return True

@bits_to_int_clause
def clause(phi, i1, i2, i3, b1, b2, b3):
    '''
        tells whether or not exists in phi a clause that satisfies the assingment i1:b1, i2:b2, i3:b3
    '''
    phi = [tuple(sorted(x)) for x in phi]
    tmp = tuple(sorted((-i1*pow(-1, b1), -i2*pow(-1, b2), -i3*pow(-1, b3))))
    return  tmp in phi

def witness(w):
    return bits_to_int_witness(lambda x: w[x])


if __name__ == "__main__":
    phi = [(1,2,3), (-2, 4,3), (-1, 2, -3)]
    z={1:True, 2:False, 3:True, 4:True}
    print(f'{eval(phi, z)=}')
    print(f'{clause(phi,0,1,1,0,1,1,0,1,0)=}')
    print(f'{witness(z)(1,1)=}')
    print(f'{witness(z)(1,0)=}')
    print(f'{witness(z)(1,0,0)=}')