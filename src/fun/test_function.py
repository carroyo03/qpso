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
    params, function = params_and_function
    n_particles, iters, w, c1, c2, dim = params
    
    # Set bounds based on the function
    if function == "rastrigin":
        bounds = ([-5.12] * dim, [5.12] * dim)
    else:
        bounds = ([-5] * dim, [5] * dim)
    desc = f"{function[:4]} p{n_particles}i{iters}w{w}"
    with tqdm(total=2, desc=desc, leave=False) as pbar: 
        # My PSO
        time_start1 = time.time()
        my_result = my_pso_optimize(
            function=function, bounds=bounds, n_particles=n_particles, 
            iters=iters, dim=dim, w=w, c1=c1, c2=c2
        )
        my_pso_time = time.time() - time_start1
        pbar.update(1)
    
        # PySwarms
        time_start2 = time.time()
        pyswarms_result = pyswarms_pso_optimize(
            function, bounds=bounds, n_particles=n_particles, 
            iters=iters, options={'w': w, 'c1': c1, 'c2': c2}
        )
        pyswarms_pso_time = time.time() - time_start2
        pbar.update(1)
    
    if pyswarms_result[0] is None:
        # Error handling for PySwarms
        print(f"Error: PySwarms optimization failed for function {function} with params {params}")
        return None
    
    # Create result dictionaries
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
    
    # Print comparison between My PSO' and PySwarms' results
    print("-" * 50)
    print(f"Function: {function}")
    print(f"Parameters: n_particles={n_particles}, iters={iters}, w={w}, c1={c1}, c2={c2}, dim={dim}")
    print("-" * 50)
    print()
    print("* My PSO:")
    for key in my_result_dict.keys():
        if key not in ['function', 'position', 'cost_history', 'repetition']:
            print(f"\t-> {key}: {my_result_dict[key]}")
    print()
    print("* PySwarms PSO:")
    for key in pyswarms_result_dict.keys():
        if key not in ['function', 'position', 'cost_history', 'repetition']:
            print(f"\t-> {key}: {pyswarms_result_dict[key]}")
    print()
    
    # Compare results
    if pyswarms_result[0] < my_result[0]:
        print(f"PySwarms PSO is {my_result[0] - pyswarms_result[0]} units cheaper than My PSO.")
    elif pyswarms_result[0] > my_result[0]:
        print(f"My PSO is {pyswarms_result[0] - my_result[0]} units cheaper than PySwarms PSO.")
    else:
        print("Both PSOs have the same cost.")

    if pyswarms_pso_time < my_pso_time:
        print(f"PySwarms PSO is faster than My PSO by {my_pso_time - pyswarms_pso_time} seconds.")
    elif pyswarms_pso_time > my_pso_time:
        print(f"My PSO is faster than PySwarms PSO by {pyswarms_pso_time - my_pso_time} seconds.")
    else:
        print("Both PSOs have the same execution time.")
    print("-" * 50)
    print()
    
    return [my_result_dict, pyswarms_result_dict]

