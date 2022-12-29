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
    x, y, z, carry_out, sum_out, prev_carry = chunks(state, n_bits)
    carry_out[0] = prev_carry
    for i in range(n_bits):
        state_i = [int(x[i]), int(y[i]), int(z[i]), 0, 0]
        x[i], y[i], z[i], carry_i, sum_out[i] = full_adder(T, state_i, alpha, Kd, n, delta) 
        if i < n_bits - 1:
            carry_out[i + 1] = carry_i
        else:
            prev_carry = carry_i
        
    to_return = np.concatenate((x, y, z, carry_out, sum_out, [prev_carry]))
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
            0, 0, 0,   # C
            0, 0, 0,    # Sum out
            0, 0, 0,   # Carry out
            0,         # Previous carry value
        )
    ] 

    # For full adder
    f, axs = plt.subplots(4, 2, sharey=True)
    for ax, ins in zip(axs.flat, inputs):
        # set initial conditions
        Y0 = np.zeros(len(inputs[0]))

        # Set inputs for current iteration
        for ix, i in enumerate(ins):
            Y0[ix] = i

        # solving the initial value problems with scipy
        sol = solve_ivp(carry_save_adder, [0, t_end], Y0, args=params, dense_output=True)
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
