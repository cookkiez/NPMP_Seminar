from gates import *
from full_adder import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


n_bits = 3


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



def carry_save_adder(T, state, *params):
    alpha, Kd, n, delta = params
    par = alpha, Kd, n, delta
    x, y, z, c, s, prev_carry = chunks(state, n_bits)
    carry_out = np.zeros(len(c))
    sum_out = np.zeros(len(s))
    carry_out[0] = prev_carry
    for i in range(n_bits):
        state_i = [int(x[i]), int(y[i]), int(z[i]), 0, 0]
        sum_degrade = degrade(s[i], delta)
        x[i], y[i], z[i], carry_i, sum_out[i] = full_adder(T, state_i, alpha, Kd, n, delta) 
        sum_out[i] += sum_degrade
        if i < n_bits - 1: 
            carry_out[i + 1] = carry_i + degrade(c[i + 1], delta)
        else:
            prev_carry = carry_i + degrade(prev_carry, delta)
        
    to_return = np.concatenate((x, y, z, carry_out, sum_out, prev_carry))
    return to_return


if __name__ == "__main__":
    params = alpha, Kd, n, delta
    t_end = 100
    # set simulation parameters
    N = t_end * 10  # number of samples
    T = np.linspace(0, t_end, N)

    # Inputs for full adder
    inputs = [
        (            
            10, 10, 0, # A
            0, 0, 10,  # B
            10, 10, 0,   # C
            0, 0, 0,   # Carry out
            0, 0, 0,   # Sum out
            0,         # Previous carry value
        )
    ] 

    f, axs = plt.subplots(n_bits, 2, sharey=True)
    for ins in inputs:
        # set initial conditions
        Y0 = np.zeros(len(inputs[0]))

        # Set inputs for current iteration
        for ix, i in enumerate(ins):
            Y0[ix] = i

        # solving the initial value problems with scipy
        sol = solve_ivp(carry_save_adder, [0, t_end], Y0, args=params, dense_output=True)
        z = sol.sol(T)

        sum_ix = 4 * n_bits
        carry_ix = 3 * n_bits
        num_plots = sum_ix + n_bits
        while sum_ix < num_plots:
            bit_i = num_plots - sum_ix - 1
            ax = axs.flat[bit_i]
            ax.plot(T, z.T[:, sum_ix:sum_ix + 1])
            ax.legend([f"Sum bit {bit_i} concentration"])
            ax.set_xlabel('Time') 
            ax.set_ylabel('Concentrations')
            ax.set_title(f"Sum bit index: {bit_i}")

            bit_i = num_plots - carry_ix - 1
            ax = axs.flat[bit_i]
            bit_i -= n_bits
            ax.plot(T, z.T[:, carry_ix:carry_ix + 1])
            ax.legend([f"Carry bit {bit_i} concentration"])
            ax.set_xlabel('Time') 
            ax.set_ylabel('Concentrations')
            ax.set_title(f"Carry bit index: {bit_i}")
            sum_ix += 1
            carry_ix += 1

    f.set_size_inches(10, 10)
    f.tight_layout()
    plt.tight_layout()
    plt.show()
