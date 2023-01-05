from gates import *
from full_adder import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


n_bits = 6


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def carry_save_adder(T, state, *params):
    alpha, Kd, n, delta = params
    x, y, z, s, c, prev_carry = chunks(state, n_bits)
    carry_out = np.zeros(len(c) + len(prev_carry))
    sum_out = np.zeros(len(s))
    carry_out[0] = prev_carry
    for i in range(n_bits):
        state_i = [int(x[i]), int(y[i]), int(z[i]), 0, 0]
        sum_degrade = degrade(s[i], delta)
        x[i], y[i], z[i], carry_i, sum_out[i] = full_adder(
            T, state_i, alpha, Kd, n, delta)
        sum_out[i] += sum_degrade
        carry_out[i] = carry_i + degrade(c[i], delta)

    to_return = np.concatenate((x, y, z, sum_out, carry_out))
    return to_return


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

def multiply(x, y):
    # sestavi vse vrstice, ki jih bo potrebno sesteti
    max_len = len(x) + len(y) - 1
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
    params = alpha, Kd, n, delta
    t_end = 100
    # set simulation parameters
    N = t_end * 10  # number of samples
    T = np.linspace(0, t_end, N)

    # pridobim seznam stevil, ki jih moram sesteti
    seznam = multiply([10,0,10], [0, 10, 0])

    parameters = [0] + seznam[0] + [0] + seznam[1] + [0] + seznam[2] + [0] * n_bits + [0] * n_bits + [0]
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





#print(multiply([10,0,10], [0, 10, 0]))


