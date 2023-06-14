import numpy as np
import functools
import itertools

def I0(xi,yi):
    return 1-xi-yi+2*xi*yi

def I(x, y, H, m):
    I_s = (I0(xi, yi) for xi,yi in zip(x,y))
    return functools.reduce(lambda a,b: a*b, I_s)
'''    numerator, denumenator = 1, 1
    for i in range(0, m):
        for h in H:
            if h == y[i]: continue #TODO: change to the form without devision
            numerator *= (x[i]-h)
            denumenator *= (y[i]-h)
    return numerator / denumenator'''

def lde(f, H, m):
    '''
        f:H^m->F
        H is a subset of F
        m is the number of variables of f
    '''
    return lambda x: sum(f(h)*I(x,h, H, m) for h in itertools.product(H, repeat=m)) 

if __name__ == "__main__":
    H=(0,1)
    d={0:5, 1:9}
    f=lambda x: d[x[1]]*d[x[0]]
    m = 2
    r=lde(f,H,m)
    for i in itertools.product(range(10), repeat=m):
        print(f'{i=}\t{r(i)=}') 