from carry_save_adder import carry_save_adder
from gates import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import itertools
from carry_save_adder import *
import plotly.express as px
import seaborn as sns


def binatodeci(binary):
    bin = [1 if b > 5 else 0 for b in binary]
    return sum(val*(2**idx) for idx, val in enumerate(reversed(bin)))


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


def append_to_list(app_list, start_ix, end_ix, z, temp_list):
    while start_ix < end_ix:
        temp_list.append(z.T[:, start_ix:start_ix + 1][999][0])
        start_ix += 1
    app_list.append(binatodeci(temp_list))
    return binatodeci(temp_list)


if __name__ == "__main__":
    n_bits = 6
    params = alpha, Kd, n, delta, n_bits
    t_end = 100
    # set simulation parameters
    N = t_end * 10  # number of samples
    T = np.linspace(0, t_end, N)
    combs = list(itertools.product([0, 10], repeat=int(n_bits / 2), ))
    combs_list = [list(c) for c in combs]
    inputs = [ [cc, c] for cc in combs_list for c in combs_list ]

    sum_results = []
    carry_results = []

    results = []

    for curr_in in inputs:
        # set initial conditions
        ins = multiply(curr_in[0].copy(), curr_in[1].copy())

        params = alpha, Kd, n, delta, n_bits
        parameters = ins[0] + ins[1] + ins[2] + [0] * n_bits + [0] * n_bits + [0]
        Y0 = np.zeros(len(parameters))

        # Set inputs for current iteration
        for ix, i in enumerate(parameters):
            Y0[ix] = i

        # solving the initial value problems with scipy
        sol = solve_ivp(carry_save_adder, [0, t_end],
                        Y0, args=params, dense_output=True)
        z = sol.sol(T)

        sum_ix = 3 * n_bits
        num_res = sum_ix + n_bits
        carry_ix = 4 * n_bits

        results.append(
            append_to_list(sum_results, sum_ix, num_res, z, []) + 
            append_to_list(carry_results, carry_ix, len(parameters), z, [])
        )

    to_plot = list(chunks(results, 8))

    sns.heatmap(to_plot, annot=True).set(title='Results of multiplication')
    plt.savefig("multiplication_heatmap.png")
    plt.show()
