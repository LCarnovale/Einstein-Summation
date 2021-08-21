import pytest
import numpy as np
import sympy
from es_tensors import AbstractESTensor, ESTensor, TensorSum

def initialize():
    a, k, r, rho, p, phi = sympy.symbols(r"a k r \rho p \phi")
    _T = np.array([
        [1, 0, 0],
        [0, -a**2 * (1/(1 - k*r**2)),  0],
        [0, 0, -a**2 * r**2]
    ])
    _T_inv = np.array([
        [1, 0, 0],
        [0, (-1 + k*r**2)/(a**2), 0],
        [0, 0, -1/(r**2 * a**2)]
    ])
    g = ESTensor("g", _T)
    g_inv = ESTensor("g^-1", _T_inv)
    T = ESTensor("T", np.array([
        [rho, 0, 0],
        [0, -p, 0],
        [0, 0, -p]
    ]))
    p = g("a", "c") * T("c", "b") * T("b", "a")
    return g, g_inv, T, p

def test_tensors():
    g, g_inv, T, p = initialize()
    print(p.expand())
    print(p.expand().evaluate())
    return g, g_inv, T, p

if __name__ == "__main__":
    # g, g_inv, T, p = test_tensors()
    # t, x, y, z = sympy.symbols("t x y z")
    # metric = np.diag([-1, 1, 1, 1])
    # mT = ESTensor("g", metric)
    # X1 = ESTensor("X", np.array([t, x, y, z]), metric)
    # print(X1("mu"))
    # print(X1("_mu"))
    # print(X1("mu").evaluate(mu=1))
    # print((X1("mu") * X1("_mu")).expand().evaluate())
    # print((X1("mu") * X1("nu") * mT("mu", "nu")).expand().evaluate())
    # print((mT("b", "_nu") * mT("a", "nu")).expand().evaluate(a=0,b=0))

    # D = DifferentialOperator

    # x, y, z, t = sympy.symbols("x y z t")
    # Dmu = ESTensor("D", [D(t), D(x), D(y), D(z)])

    # bigX = ESTensor("X", 
    #     [[2*x, x*y, x*z, x*t],
    #     [x**2, 4*z, 4*x, 4*y],
    #     [2*x, x*y, x*z, x*t],
    #     [x**2, 4*z, 4*x, 4*y],
    #     ])
    # from es_tensors import RankOneTensor, TensorProduct
    # zt = RankOneTensor(0)
    # tensor = ESTensor("3dTensor", np.ones((2,2,2)))
    # prod = TensorProduct(zt, tensor(0, "a", "b"))
    # print(prod.expand().evaluate(a=0, b=0))

    import numpy as np
    # import tensorflow as tf
    import sympy
    # from sympy import Matrix, Inverse, Derivative
    # from diffop import D, evaluateExpr

    from es_tensors import ESTensor, AbstractESTensor
    t = sympy.Symbol("t")
    christoff_data = np.zeros((2,2,2),dtype=object)
    christoff_data[1,0,1] = 1
    christoff_data[1,1,0] = 1
    christoff_data[0,1,1] = sympy.exp(2*t)

    metric = np.array([
        [1, 0],
        [0, -sympy.exp(2*t)]
    ])
    dt, dr = sympy.symbols("Dt Dr")
    dx = np.array([dt, 0])

    def sum_over(tensor, index, start, stop):
        idx = {index:start}
        a = tensor.evaluate(**idx)
        for i in range(start+1, stop):
            a = a + tensor.evaluate(**{index:i})
        return a
    CS = ESTensor("CHR-SYM", christoff_data, metric=metric)
    # DX = ESTensor("D", dx, metric=metric)
    DX = AbstractESTensor("D")
    prod = DX("mu") * CS("mu", "a", "b")
    print(prod.expand().evaluate(a=1, b=1))
    # R^mu_{a nu b}
    Rieman = DX("nu") * CS("mu", "a", "b") + (DX("b")*-1) * CS("mu", "a","nu") + \
        CS("d", "a", "b")*CS("mu", "d", "nu") + \
        (CS("d", "nu", "a")*-1)*CS("mu", "d", "b")
    print(Rieman.expand())
    tt_ = Rieman.evaluate(mu=1,a=1,nu=0,b=0).evaluate().expand()
    print(tt_)
    g = ESTensor("g", metric)
    # R_{y a nu b}
    Rieman_cont = g("y", "mu") * Rieman
    argh = Rieman_cont.evaluate(y=1, a=0, nu=1, b=0).expand()
    print(argh.evaluate().evaluate())
    # argh
    # print(argh.evaluate())
    # sum_over(argh.expand().evaluate(y=1, a=0, nu=1, b=0), 'mu', 0,2)
    delta = ESTensor("delta", np.eye(2),metric=metric)
    Ricci = Rieman.evaluate(nu='mu')
    # print(Ricci.expand())
    print(Ricci.expand().evaluate(mu=0).evaluate(a=0, b=0))
    # _temp = Ricci.expand()
    # _temp = ESTensor("eye", np.eye(2))("mu","nu") * _temp
    # sum_over(Ricci.evaluate(a='b',b=1), 'nu', 0, 2)
    # _temp.children[-1].children[1].children = list([_temp.children[-1].children[1].children[i] for i in [1,0,2]])
    # print(_temp.expand().evaluate(a=0,b=0))
    # get_sum = lambda a, b: _temp.evaluate(a=a,b=b, mu=0,nu=0) + _temp.evaluate(a=a,b=b,mu=1,nu=1)
    # get_sum(0,0)
    print(Ricci.expand())
    inv = ESTensor("g-inv", np.array([
        [1, 0],
        [0, -sympy.exp(-2*t)]
    ]))
    R_s_s = R_s = (inv("a","b")*Ricci).expand().evaluate()

    # sum_over(R_s.expand(), 'nu',0,2).expand().evaluate()
    print("\n>> ".join([str(x) for x in R_s_s.evaluate(nu=1).expand().children]))
    print(R_s.expand().evaluate(a=0, b=0).expand().evaluate().expand())

