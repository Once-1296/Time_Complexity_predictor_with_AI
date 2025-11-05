def fibonacci_recursive(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_iterative(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def quadratic_function(n):
    total = 0
    for i in range(n):
        for j in range(n):
            total += i + j
    return total

def linear_function(n):
    total = 0
    for i in range(n):
        total += i
    return total