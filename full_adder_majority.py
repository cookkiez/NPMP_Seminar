from gates import *
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def full_adder(T, state, *params):
    alpha, Kd, n, delta = params
    par = alpha, Kd, n
    a, b, c, carry, s = state

    abc_maj = MAJ(a, b, c, par)

    abcnot_maj = MAJ(a, b, NOT(c, par), par)


    carry_out = abc_maj + degrade(carry, delta)
    s_out = MAJ(NOT(abc_maj, par), c, abcnot_maj, par) + degrade(s, delta)

    return a, b, c, carry_out, s_out


if __name__ == "__main__":
    params = alpha, Kd, n, delta
    t_end = 100
    # set simulation parameters
    N = t_end * 10  # number of samples
    T = np.linspace(0, t_end, N)

    # Inputs for full adder
    inputs = [
        (0, 0, 0),
        (0, 0, 10),
        (0, 10, 0),
        (0, 10, 10),
        (10, 0, 0),
        (10, 0, 10),
        (10, 10, 0),
        (10, 10, 10)
    ]

    # For full adder
    f, axs = plt.subplots(4, 2, sharey=True)
    for ax, ins in zip(axs.flat, inputs):
        # set initial conditions
        Y0 = np.zeros(5)

        # Set inputs for current iteration
        for ix, i in enumerate(ins):
            Y0[ix] = i

        # solving the initial value problems with scipy
        sol = solve_ivp(full_adder, [0, t_end], Y0, args=params, dense_output=True)
        z = sol.sol(T)

        # For full adder
        ax.plot(T, z.T[:, 3:])
        ax.legend(['Carry', 'Sum'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentrations')
        ax.set_title(f"A={ins[0]}, B={ins[1]}, Cin={ins[2]}")

    f.set_size_inches(10, 10)
    f.tight_layout()
    plt.tight_layout()
    plt.show()
