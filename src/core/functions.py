import numpy as np



def ackley(position:np.ndarray):
    """ Ackley function.

    Args:
        position (np.ndarray): Position array.

    Returns:
        float: The Ackley function value at the given position.
    """
    x = position
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    n = x.shape[1]
    sum1 = np.zeros(x.shape[0])
    sum2 = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        for j in range(n):
            sum1[i] += x[i, j] ** 2
            sum2[i] += np.cos(c * x[i, j])

    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)

    return term1 + term2 + a + np.exp(1.0)


def rastrigin(position:np.ndarray):
    """ Rastrigin function.

    Args:
        position (np.ndarray): Position array.
        
    Returns:
        float: The Rastrigin function value at the given position.
    """
    x = position
    n = x.shape[1]
    result = 10.0 * n * np.ones(x.shape[0])

    for i in range(x.shape[0]):
        for j in range(n):
            result[i] += x[i, j] ** 2 - 10.0 * np.cos(2.0 * np.pi * x[i, j])

    return result


def rosenbrock(position:np.ndarray):
    """ Rosenbrock function.
    
    Args:
        position (np.ndarray): Position array.
        
    Returns:
        float: The Rosenbrock function value at the given position.
    """
    x = position
    n = x.shape[1]
    result = np.zeros(x.shape[0])

    for i in range(x.shape[0]):
        for j in range(n - 1):
            result[i] += 100.0 * (x[i, j + 1] - x[i, j] ** 2) ** 2 + (x[i, j] - 1.0) ** 2

    return result


def objective(position:np.ndarray, function:str):
    """Objective function.

    Args:
        position (np.ndarray): Position array.
        function (str): The objective function to optimize. Supported functions: "ackley", "rastrigin", "rosenbrock".

    Raises:
        ValueError: If the function is not supported.

    Returns:
        float: The objective function value at the given position.
    """
    if function not in ["ackley", "rastrigin", "rosenbrock"]:
            raise ValueError(f"Unsupported function: {function}")

    # Ensure position is not unidimensional
    if position.ndim == 1:
        position = position.reshape(1, -1)
    
    return eval(function)(position)

    