import time
import pandas as pd #type: ignore
from pso.pso import ParticleSwarmOptimization #type: ignore
import pyswarms as ps #type: ignore
from pyswarms.utils.functions import single_obj as fx #type: ignore
from functools import partial
from tqdm import tqdm  #type: ignore

def my_pso_optimize(function: str, bounds, n_particles, iters, w: float, c1: float, c2: float, dim=2) -> tuple:
    """My PSO implementation.
    Args:
        function (str): The objective function to optimize. Supported functions: "ackley", "rastrigin", "rosenbrock".
        bounds (tuple): Tuple containing the minimum and maximum bounds for the position.
        n_particles (int): Number of particles in the swarm.
        iters (int): Number of iterations for the optimization.
        w (float): Inertia weight.
        c1 (float): Cognitive coefficient.
        c2 (float): Social coefficient.
        dim (int): Number of dimensions for the optimization problem.
    Returns:
        tuple: Best cost, best position, and cost history.
    """
    optimizer = ParticleSwarmOptimization(
        number_of_particles=n_particles, dim=dim, bounds=bounds,
        vel_min=-0.1, vel_max=0.1, w=w, c1=c1, c2=c2, function=function
    )
    # Initialize the cost, position and cost history obtained from the optimizer
    cost, pos, cost_history = optimizer.optimize(num_iterations=iters)
    return cost, pos, cost_history

def pyswarms_pso_optimize(function: str, bounds, n_particles, iters, options: dict) -> tuple:
    """PySwarms PSO implementation.
    Args:
        function (str): The objective function to optimize. Supported functions: "ackley", "rastrigin", "rosenbrock".
        bounds (tuple): Tuple containing the minimum and maximum bounds for the position.
        n_particles (int): Number of particles in the swarm.
        iters (int): Number of iterations for the optimization.
        options (dict): Options for PySwarms PSO.
    Returns:
        tuple: Best cost, best position, and cost history.
    """
    # Only supporting 3 functions: ackley, rastrigin and rosenbrock
    if function == "ackley":
        func = fx.ackley
    elif function == "rastrigin":
        func = fx.rastrigin
    elif function == "rosenbrock":
        func = fx.rosenbrock
    else:
        return None, None, None

    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=len(bounds[0]),
        options=options,
        bounds=bounds,
        velocity_clamp=(-0.1, 0.1)
    )
    cost, pos = optimizer.optimize(func, iters=iters)
    cost_history = optimizer.cost_history
    return cost, pos, cost_history

def run_optimization(params_and_function: tuple, rep_num=0):
    """Run the optimization for a given set of parameters and function.
    Args:
        params_and_function (tuple): Tuple containing parameters and function name.
        rep_num (int): Repetition number for the optimization.
    Returns:
        list: List containing the results of the optimization.
    """
    # Disable PySwarms logging completely
    import logging
    logging.getLogger("pyswarms").setLevel(logging.CRITICAL)
    
    # Also disable PySwarms standard output during optimization
    import sys
    import os
    from contextlib import contextmanager
    
    @contextmanager
    def suppress_stdout_stderr():
        # Redirect stdout and stderr to /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        
        try:
            yield
        finally:
            # Restore stdout and stderr
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
    
    params, function = params_and_function
    n_particles, iters, w, c1, c2, dim = params
    
    # Set bounds based on the function
    if function == "rastrigin":
        bounds = ([-5.12] * dim, [5.12] * dim)
    else:
        bounds = ([-5] * dim, [5] * dim)
    
    # My PSO (without internal progress bar)
    time_start1 = time.time()
    my_result = my_pso_optimize(
        function=function, bounds=bounds, n_particles=n_particles, 
        iters=iters, dim=dim, w=w, c1=c1, c2=c2
    )
    my_pso_time = time.time() - time_start1
    
    # PySwarms
    time_start2 = time.time()
    with suppress_stdout_stderr():
        pyswarms_result = pyswarms_pso_optimize(
            function, bounds=bounds, n_particles=n_particles, 
            iters=iters, options={'w': w, 'c1': c1, 'c2': c2}
        )
    pyswarms_pso_time = time.time() - time_start2
    
    if pyswarms_result[0] is None:
        # Error handling for PySwarms
        print(f"Error: PySwarms optimization failed for function {function} with params {params}")
        return None
    ""
    # Create result dictionaries without printing anything
    my_result_dict = {
        'method': 'my_pso',
        'function': function,
        'n_particles': n_particles,
        'iters': iters,
        'w': w,
        'c1': c1,
        'c2': c2,
        'dim': dim,
        'cost': my_result[0],
        'position': my_result[1].tolist(),
        'execution_time': my_pso_time,
        'cost_history': my_result[2],
        'repetition': rep_num
    }
    
    pyswarms_result_dict = {
        'method': 'pyswarms_pso',
        'function': function,
        'n_particles': n_particles,
        'iters': iters,
        'w': w,
        'c1': c1,
        'c2': c2,
        'dim': dim,
        'cost': pyswarms_result[0],
        'position': pyswarms_result[1].tolist(),
        'execution_time': pyswarms_pso_time,
        'cost_history': pyswarms_result[2],
        'repetition': rep_num
    }
    
    # Remove all print statements
    
    return [my_result_dict, pyswarms_result_dict]