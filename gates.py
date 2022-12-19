"""
Write gates here: 
    - AND 
    - NOT
    - OR
    - MAJORITY
    - Half adder
    - Full adder
"""


alpha = 10
Kd = 1
n = 2
delta = 1


def AND(in1, in2, params=(alpha, Kd, n)):
    alpha, Kd, n = params

    frac1 = (in1 / Kd) ** n
    frac2 = (in2 / Kd) ** n
    dy_dt = alpha * (frac1 * frac2) / (1 + frac1 + frac2 + frac1 * frac2)

    return dy_dt


def NOT(x, params=(alpha, Kd, n)):
    alpha, Kd, n = params

    frac = (x / Kd) ** n
    dy_dt = alpha * 1 / (1 + frac)

    return dy_dt


# NOR
def NOR(in1, in2, params=(alpha, Kd, n)):
    alpha, Kd, n = params

    frac1 = (in1 / Kd) ** n
    frac2 = (in2 / Kd) ** n
    dy_dt = alpha * 1 / (1 + frac1 + frac2 + frac1 * frac2)

    return dy_dt


def OR(in1, in2, params=(alpha, Kd, n)):
    alpha, Kd, n = params

    frac1 = (in1 / Kd) ** n
    frac2 = (in2 / Kd) ** n
    dy_dt = alpha * (frac1 + frac2 + frac1 * frac2) / (1 + frac1 + frac2 + frac1 * frac2)

    return dy_dt


def XOR(a, b, params=(alpha, Kd, n)):
    alpha, Kd, n = params

    a_and_b = AND(a, b, params)
    a_or_b = OR(a, b, params)
    not_a_and_b = NOT(a_and_b, params)
    final = AND(not_a_and_b, a_or_b, params)

    return final


def MAJ(a, b, c, params=(alpha, Kd, n)):
    alpha, Kd, n = params

    d1 = AND(a, b, params)
    d2 = AND(a, c, params)
    d3 = AND(b, c, params)

    d_or1 = OR(d1, d2, params)
    d_or_final = OR(d_or1, d3, params)
    return d_or_final


# degradation
def degrade(x, delta=delta):
    dx_dt = - x * delta

    return dx_dt


if __name__ == "__main__":
    print("Run main.py")
