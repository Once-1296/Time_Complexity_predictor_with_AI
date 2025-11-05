import math, time
DERIVATIVE_ITERATIONS = 3
def helper(lis_t, lis_n):
    length_1 = len(lis_t)
    derivative_list, new_lis_t, new_lis_n = [], [], []
    for i in range(length_1):
        for j in range(i + 1, length_1):
            dy, dx = lis_n[j] - lis_n[i], lis_t[j] - lis_t[i]
            if math.isclose(dx, 0, abs_tol=1e-5):
                dx = dx + 1e-5 if dx > 0 else dx - 1e-5
            derivative = dy / dx
            derivative_list.append(derivative)
            if j == i + 1:
                new_lis_n.append(dy)
                new_lis_t.append(dx)
    dy, dx = new_lis_n[-1] - new_lis_n[0], new_lis_t[-1] - new_lis_t[0]
    if math.isclose(dx, 0, abs_tol=1e-5):
        dx = dx + 1e-5 if dx > 0 else dx - 1e-5
    new_lis_n.append(dy)
    new_lis_t.append(dx)
    return derivative_list, new_lis_t, new_lis_n


def power_simulation(power,n):
    if power == 0:
        return
    for _ in range(n):
        power_simulation(power - 1,n)

def non_linear_simulation(W,B,n):
    start_time = time.time()
    n_complexity = int(math.ceil(n))
    for _ in range(W):
        for __ in range(n_complexity):
            ___ = 0
    for _ in range(B):
        __ = 0
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time