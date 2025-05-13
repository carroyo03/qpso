import os
import time
import numpy as np
import pandas as pd
import logging

import sys
from tqdm import tqdm
from itertools import product
from contextlib import contextmanager

@contextmanager
def suppress_output():
    """Delete  output to stdout and stderr."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)



available_backends = []
try:
    import pyswarms as ps
    from pyswarms.utils.functions import single_obj as fx
    PYSWARMS_AVAILABLE = True
    available_backends.append('pyswarms')
except ImportError as e:
    PYSWARMS_AVAILABLE = False
    print(f"pyswarms is not available. Skipping pyswarms backend. Error{e}")
try:
    import implementations.threading.pso.pso as threading_pso
    THREADING_AVAILABLE = False
    available_backends.append('threading')
except ImportError as e:
    THREADING_AVAILABLE = False
    print(f"threading is not available. Skipping threading backend. Error{e}")
try:
    import implementations.openmp.pso.pso as openmp_pso
    OPENMP_AVAILABLE = True
    available_backends.append('openmp')
except ImportError as e:
    OPENMP_AVAILABLE = False
    print(f"openmp is not available. Skipping multiprocessing backend. Error{e}")
try:
    import implementations.async_prog.pso.pso as async_pso
    ASYNC_AVAILABLE = True
    available_backends.append('async')
except ImportError:
    ASYNC_AVAILABLE = False
    print("async is not available. Skipping async backend.")

CPU_COUNT = os.cpu_count() -1 # Reserve one CPU for the main process

def get_backend_name(backend_func):
    """
    This function retrieves the name of the backend function being useed for the PSO algorithm.

    Parameters:
    backend_func: Callable
        The backend function used for the PSO algorithm.
    Returns:
        str: The name of the backend function.
    """
    if backend_func.__name__ == 'pyswarms_wrapper':
        return 'pyswarms'
    elif hasattr(backend_func, '__module__'):
        module_name = backend_func.__module__
        if 'threading' in module_name:
            return 'threading'
        elif 'openmp' in module_name:
            return 'openmp'
        elif 'async' in module_name:
            return 'async'
    return backend_func.__name__

def pyswarms_wrapper(function, n_particles, bounds, iters, w, c1, c2, dim):
    """
    This function serves as a wrapper for using the PySwarms package to perform
    Particle Swarm Optimization (PSO). It allows optimization on predefined objective
    functions and provides flexibility in setting PSO configuration parameters,
    including the number of particles, bounds, inertia, cognitive, and social
    coefficients.

    Parameters:
    function: str
        The name of the objective function to optimize. Supported options are:
        'rastrigin', 'ackley', and 'rosenbrock'.
    n_particles: int
        The number of particles in the swarm.
    bounds: tuple[tuple[float, float], tuple[float, float]]
        The lower and upper bounds for each dimension in a tuple format.
    iters: int
        The maximum number of iterations for the optimization process.
    w: float
        Inertia weight controlling the influence of the previous velocity.
    c1: float
        Cognitive parameter emphasizing the particle's own best-known position.
    c2: float
        Social parameter emphasizing the swarm's collective best-known position.
    dim: int
        Dimensionality of the search space.

    Returns:
    tuple[float, numpy.ndarray, list[float]]
        A tuple containing:
        - cost: The best solution's cost achieved during optimization.
        - pos: The position of the best solution found.
        - cost_history: A list documenting the cost at each iteration.

    Raises:
    ValueError
        If the specified function name is not recognized.
    """
    if function == 'rastrigin':
        func = fx.rastrigin
    elif function == 'ackley':
        func = fx.ackley
    elif function == 'rosenbrock':
        func = fx.rosenbrock
    else:
        raise ValueError(f"Unknown function: {function}")
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dim, options={'c1': c1, 'c2': c2, 'w': w}, bounds=bounds)
    with suppress_output():
        cost, pos = optimizer.optimize(func, iters=iters)
    cost_history = optimizer.cost_history
    return cost, pos, cost_history

def run_optimization(backend_func, n_particles, iters, w, c1, c2, dim, function, repetition=0):
    """
    Runs an optimization process using a specified backend function for a given
    objective function and parameter configuration. It returns the PSO
    results, including key information about the process, performance metrics,
    and optimization parameters.

    Args:
        backend_func (Callable): The optimization function backend that handles
            the core optimization logic.
        n_particles (int): The number of particles used in the optimization.
        iters (int): The number of iterations of the optimization process.
        w (float): The inertia weight parameter in the optimization algorithm.
        c1 (float): The cognitive parameter used by the optimization algorithm.
        c2 (float): The social parameter used by the optimization algorithm.
        dim (int): The number of dimensions of the solution space.
        function (str): The name of the objective function to optimize.
        repetition (int, optional): An identifier indicating the repetition
            iteration for benchmarking purposes. Default is 0.

    Returns:
        Dict: A dictionary containing optimization results, parameters, and
        metadata. Includes:
            - 'method': The name of the backend optimization library or module.
            - 'function': The name of the objective function optimized.
            - 'n_particles': The number of particles used.
            - 'iters': The number of iterations performed.
            - 'w': The inertia weight used.
            - 'c1': The cognitive parameter used.
            - 'c2': The social parameter used.
            - 'dim': The dimension of the problem solved.
            - 'cost': The achieved cost (objective function value).
            - 'position': The optimized solution position.
            - 'execution_time': The total execution time of the optimization.
            - 'cost_history': The history of the cost function over iterations.
            - 'repetition': The repetition identifier provided during execution.
    """
    bounds = (np.array([-5.12] * dim), np.array([5.12] * dim)) \
        if function == 'rastrigin' else (np.array([-32] * dim), np.array([32] * dim))
    start_time = time.time()
    backend_name = get_backend_name(backend_func).lower()
    if 'thread' in backend_name:
        cost, pos, cost_history = backend_func(function=function, bounds=bounds, n_particles=n_particles, num_iterations=iters, w=w, c1=c1, c2=c2, dim=dim, max_workers = min(CPU_COUNT, n_particles))
    elif 'openmp' in backend_name:
        cost, pos, cost_history = backend_func(function=function, bounds=bounds, n_particles=n_particles, iters=iters, w=w, c1=c1, c2=c2, dim=dim, vel_min=-.1,vel_max=.1)
    elif 'async' in backend_name:
        cost, pos, cost_history = backend_func(function=function, bounds=bounds, n_particles=n_particles, iters=iters, w=w, c1=c1, c2=c2, dim=dim, vel_min=-.1, vel_max=.1)
    else:
        cost, pos, cost_history = backend_func(function=function, n_particles=n_particles, bounds=bounds, iters=iters, w=w, c1=c1, c2=c2, dim=dim)
    execution_time = time.time() - start_time
    return {
        'method': get_backend_name(backend_func),
        'function': function,
        'n_particles': n_particles,
        'iters': iters,
        'w': w,
        'c1': c1,
        'c2': c2,
        'dim': dim,
        'cost': cost,
        'position':pos.tolist() if not isinstance(pos,list) else pos,
        'execution_time': execution_time,
        'cost_history': cost_history,
        'repetition': repetition
    }

if __name__ == '__main__':
    logging.getLogger('pyswarms').setLevel(logging.CRITICAL)

    """
    n_particles_list = np.linspace(300,400,3,dtype=int)
    iters_list = np.linspace(100,200,3,dtype=int)
    w_list = np.linspace(.5,0.9,3)
    c1_list = np.linspace(.5,1.5,3)
    c2_list = np.linspace(.5,1.5,3)
    dim_list = np.linspace(100,300,3,dtype=int)
    """
    # Simplified parameter ranges for testing
    n_particles_list = np.linspace(5,10, 2, dtype=int)
    iters_list = np.linspace(10, 30, 2, dtype=int)
    w_list = [0.8,1.0]
    c1_list = [1,0,1.5]
    c2_list = [1.0,1.5]
    dim_list = np.linspace(2, 5, 2, dtype=int)
    function_list = ['rastrigin', 'ackley','rosenbrock']
    num_repetitions = 10

    all_combinations = list(product(n_particles_list, iters_list, w_list, c1_list, c2_list, dim_list, function_list))
    results = []
    backends = {
        'pyswarms': pyswarms_wrapper if PYSWARMS_AVAILABLE else None,
        'threading': threading_pso.my_threading_pso if THREADING_AVAILABLE else None,
        'openmp': openmp_pso.my_openmp_pso if OPENMP_AVAILABLE else None,
        'async': async_pso.my_async_pso if ASYNC_AVAILABLE else None
    }

    print(f"Available backends: {available_backends}")
    print(f"Running on {CPU_COUNT} CPU cores")
    print("=="*40)

    for backend_name, backend_func in backends.items():
        if backend_func is None:
            continue
        print(f"Running {backend_name} backend")
        total_tasks = len(all_combinations) * num_repetitions
        with tqdm(
            total = total_tasks,
            desc=f'Processing {backend_name}',
            ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ) as pbar:
            for params in all_combinations:
                for rep in range(num_repetitions):
                    result = run_optimization(backend_func,*params,repetition=rep)
                    results.append(result)
                    pbar.update(1)
        print(f"Finished {backend_name} backend")
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('analysis/pso_results.csv', index=False)
        print("Results saved to analysis/pso_results.csv")
    else:
        print('No results obtained.')