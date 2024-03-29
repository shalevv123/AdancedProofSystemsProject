import functools
import itertools
import concurrent.futures
from parameters import F


def I0(xi, yi):
    if isinstance(xi, F) or isinstance(yi, F):
        return F(1) - F(xi) - F(yi) + 2 * F(xi) * F(yi)
    return 1 - xi - yi + 2 * xi * yi


def I(x, y, H, m):
    I_s = (I0(xi, yi) for xi, yi in itertools.zip_longest(x, y, fillvalue=0))
    return functools.reduce(lambda a, b: a * b, I_s)


def calculate_I(args):
    h, x, f, H, m = args
    return f(*h) * I(x, h, H, m)


def lde(f, H, m):
    def r(x):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            args = [(h, x, f, H, m) for h in itertools.product(H, repeat=m)]
            results = executor.map(calculate_I, args)
        return functools.reduce(lambda a, b: a + b, results)

    return r


if __name__ == "__main__":
    H = (0, 1)
    d = {0: 5, 1: 9}

    def f(x):
        return d[x[1]] * d[x[0]]

    m = 2
    r = lde(f, H, m)
    for i in itertools.product(range(10), repeat=m):
        print(f'{i=}\t{r(i)=}')
