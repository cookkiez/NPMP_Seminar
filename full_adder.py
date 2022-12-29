from gates import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def full_adder(T, state, *params):
    alpha, Kd, n, delta = params
    par = alpha, Kd, n
    a, b, c, carry, s = state

    a_xor_b = XOR(a, b, par)  # a XOR b
    s_out = XOR(a_xor_b, c, par) + degrade(s, delta)  # (a XOR b) XOR c

    c_and_a_xor_b = AND(c, a_xor_b, par)  # c AND (a XOR b)
    a_and_b = AND(a, b, par)  # a AND b
    carry_out = OR(c_and_a_xor_b, a_and_b, par) + degrade(carry, delta)  # (a AND b) OR (c AND (a XOR b))

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
