
def eval(phi, z):
    '''
    phi is a 3-CNF formula represented as a list of tuples.
        each tuple represent a clouse, and each element in the tuple can be a <number> or -<number> (negative number, representing "not") 
    '''
    for clouse in phi:
        if not any(z[x] if x>=0 else not z[-x] for x in clouse):
            return False
    return True

def clouse(phi):
    '''
        returns a function that tells whether or not exists in phi a clouse that satisfies the assingment i1:b1, i2:b2, i3:b3
    '''
    phi = [tuple(sorted(x)) for x in phi]
    return lambda i1, i2, i3, b1, b2, b3: tuple(sorted((-i1*pow(-1, b1), -i2*pow(-1, b2), -i3*pow(-1, b3)))) in phi

def witness(w):
    return lambda x: w[x]


if __name__ == "__main__":
    phi = [(1,2,3), (-2, 4,3), (-1, 2, -3)]
    z={1:True, 2:False, 3:True, 4:True}
    print(f'{eval(phi, z)=}')
    print(f'{clouse(phi)(1,2,3,0,1,0)=}')