import numpy as np 

def ackley(position:np.ndarray):
    """ Ackley function.

    Args:
        position (np.ndarray): Position array.

    Returns:
        float: The Ackley function value at the given position.
    """
    dim = position.shape[1]
    a, b, c = 20, 0.2, 2 * np.pi
    sum1 = np.sum(position ** 2, axis=1)
    sum2 = np.sum(np.cos(c * position), axis=1)
    term1 = -a * np.exp(-b * np.sqrt(sum1 / dim))
    term2 = -np.exp(sum2 / dim)
    return term1 + term2 + a + np.exp(1)

def rastrigin(position:np.ndarray):
    """ Rastrigin function.

    Args:
        position (np.ndarray): Position array.
        
    Returns:
        float: The Rastrigin function value at the given position.
    """
    position = np.clip(position, -5.12, 5.12)
    dim = position.shape[1]
    return 10 * dim + np.sum(position ** 2 - 10 * np.cos(2 * np.pi * position), axis=1)

def rosenbrock(position:np.ndarray):
    """ Rosenbrock function.
    
    Args:
        position (np.ndarray): Position array.
        
    Returns:
        float: The Rosenbrock function value at the given position.
    """
    dim = position.shape[1]
    result = np.zeros(position.shape[0])
    for i in range(dim - 1):
        x = position[:, i]
        y = position[:, i + 1]
        result += 100 * (y - x ** 2) ** 2 + (1 - x) ** 2
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

    