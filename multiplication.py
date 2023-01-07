from carry_save_adder import carry_save_adder
from gates import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def do_plots(ix, ax_ix, num_plots, legend_string, title_string):
    while ix < num_plots:
        bit_i = num_plots - ix - 1
        ax = axs.flat[ax_ix]
        ax.plot(T, z.T[:, ix:ix + 1])
        ax.legend([legend_string(bit_i)])
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentrations')
        ax.set_title(title_string(bit_i))
        ix += 1
        ax_ix += 1

# return list of all numbers that needs to be added to calculate multiplication
def multiply(x, y):
    max_len = len(x) + len(y)
    seznam = []
    odmik = 0
    while len(y) > 0:
        element = y.pop()
        if element == 10:
            vrstica = [0] * (max_len - odmik - len(x)) + x + [0] * odmik
        else:
            vrstica = [0] * max_len
        seznam.append(vrstica)
        odmik += 1
    return seznam


if __name__ == "__main__":
    number1 = [10, 0, 10]
    number2 = [0, 10, 0]

    n_bits = len(number1) + len(number2)
    params = alpha, Kd, n, delta, n_bits
    t_end = 100
    # set simulation parameters
    N = t_end * 10  # number of samples
    T = np.linspace(0, t_end, N)

    seznam = multiply(number1, number2)

    parameters = seznam[0] + seznam[1] + seznam[2] + [0] * n_bits + [0] * n_bits + [0]
    # Inputs for Carry save adder
    inputs = [
        tuple(parameters)
    ]

    f, axs = plt.subplots(n_bits + 1, 2, sharey=True)
    for ins in inputs:
        # set initial conditions
        Y0 = np.zeros(len(inputs[0]))

        # Set inputs for current iteration
        for ix, i in enumerate(ins):
            Y0[ix] = i

        # solving the initial value problems with scipy
        sol = solve_ivp(carry_save_adder, [0, t_end],
                        Y0, args=params, dense_output=True)
        z = sol.sol(T)

        sum_ix = 3 * n_bits
        ax_ix = 0
        do_plots(sum_ix, ax_ix, sum_ix + n_bits,
                 lambda i: f"Sum bit {i} concentration",
                 lambda i: f"Sum bit index {i}")

        carry_ix = 4 * n_bits
        num_plots = len(inputs[0])
        ax_ix = n_bits
        do_plots(carry_ix, ax_ix, num_plots,
                 lambda i: f"Carry bit {i} concentration",
                 lambda i: f"Carry bit index: {i}")

    f.set_size_inches(10, 10)
    f.tight_layout()
    plt.tight_layout()
    plt.show()