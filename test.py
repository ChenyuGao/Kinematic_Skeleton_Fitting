import numpy as np
from scipy.optimize import root
from scipy.misc import derivative
import sympy
from sympy import diff, symbols

# a, b, x0, x1 = symbols("a b x0 x1")
# a = x0 * sympy.cos(x1) - 4
# b = x1 * x0 - x1 - 5
# funcs = sympy.Matrix([a, b])
# args = sympy.Matrix([x0, x1])
# res = funcs.jacobian(args)
# print(funcs, args, res)
# x = [1, 1]
# f = [a.evalf(subs={x0: x[0], x1: x[1]}), b.evalf(subs={x0: x[0], x1: x[1]})]
# j = np.array(res.evalf(subs={x0: x[0], x1: x[1]}))
# print(type(f), type(j))


def f(y):
    w,p1,p2,p3,p4,p5,p6,p7 = y[:8]
    def t(p,w,a):
        b = -0.998515304
        e1 = 0.92894164
        e2 = 1.1376649
        return(w-a*p**e1 - b*(1-p)**e2)
    t1 = t(p1,w,1.0)
    t2 = t(p2,w,4.0)
    t3 = t(p3,w,16.0)
    t4 = t(p4,w,64.0)
    t5 = t(p5,w,256.0)
    t6 = t(p6,w,512.0)
    t7 = t(p7,w,1024.0)
    t8 = p1 + p2 + p3 + p4 + p5 + p6 + p7 - 1.0
    return (t1,t2,t3,t4,t5,t6,t7,t8)


# guess = 0.01
# x0 = np.array([-1000.0,guess,guess,guess,guess,guess,guess,guess])
# sol = root(f, x0, method='lm')
# print('w=-1000: ', sol.x, sol.success, sol.nfev, sol.fun, np.sum(f(sol.x)))


def func3(a):
    a += 1
    return a


def func2(x):
    a = x[0] * np.cos(x[1]) - 4
    a = func3(a)
    b = x[1] * x[0] - x[1] - 5
    if x[0] > 5.:
        c = (x[0] - 5.) ** 2
    else:
        c = 0.
    # df = np.array([[np.cos(x[1]), -x[0] * np.sin(x[1])], [x[1], x[0] - 1]])
    return a, b, c


# sol = root(func2, [1, 1], method='lm')
# print(sol.x, sol.success, sol.nfev, sol.fun)

fd = open('./out/logs.txt', 'a+')
fd.write('11\n')
