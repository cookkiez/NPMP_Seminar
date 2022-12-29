from gates import *
import numpy as np  
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def majority(T, state, *params):
    a, b, c, y = state
    alpha, Kd, n, delta = params
    par = alpha, Kd, n

    dY = MAJ(a, b, c, par) + degrade(y, delta)

    return [a, b, c, dY]


if __name__ == "__main__":
    params = alpha, Kd, n, delta
    t_end = 100
    # set simulation parameters
    N = t_end * 10  # number of samples
    T = np.linspace(0, t_end, N)

    # Inputs for majority
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

    # For majority
    f, axs = plt.subplots(4, 2, sharey=True)
    for ax, ins in zip(axs.flat, inputs):
        # set initial conditions
        Y0 = np.zeros(4)

        # Set inputs for current iteration
        for ix, i in enumerate(ins):
            Y0[ix] = i

        # solving the initial value problems with scipy
        sol = solve_ivp(majority, [0, t_end], Y0, args=params, dense_output=True)
        z = sol.sol(T)

        # For majority gate
        ax.plot(T, z.T[:, 3:])
        ax.legend(['Output'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentrations')
        ax.set_title(f"A={ins[0]}, B={ins[1]}, C={ins[2]}")

    f.set_size_inches(10, 10)
    f.tight_layout()
    plt.tight_layout()
    plt.show()
