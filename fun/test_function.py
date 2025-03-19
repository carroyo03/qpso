import time
import numpy as np
import pandas as pd

from pso.pso import ParticleSwarmOptimization
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx


def my_pso_optimize(bounds, n_particles, iters, dim=2):
    """
    Optimize a function using the Particle Swarm Optimization algorithm.
    Args:
        bounds:
        n_particles:
        iters:
        dim:

    Returns:
        np.float64, np.ndarray: The cost and the position of the best particle.
    """
    optimizer = ParticleSwarmOptimization(number_of_particles=n_particles, dim=dim, bounds=bounds, vel_min=-1, vel_max=1, w=.9,
                                    c1=.5, c2=.3)
    cost, pos = optimizer.optimize(num_iterations=iters)
    return cost, pos, optimizer.cost_history

def pyswarms_pso_optimize(func, bounds, n_particles, iters):
    """
    Optimize a function using the Particle Swarm Optimization algorithm from the pyswarms library.
    Args:
        func: The objective function to be optimized.
        bounds: The bounds of the search space.
        n_particles: The number of particles in the swarm.
        iters: The number of iterations.

    Returns:
        np.float64, np.ndarray: The cost and the position of the best particle.
    """
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=len(bounds), options={'c1': 0.5, 'c2': 0.3, 'w': 0.9})
    cost, pos = optimizer.optimize(func, iters=iters)
    return cost, pos, optimizer.cost_history

if __name__ == '__main__':
    bounds = (-5,5)
    n_particles = 10
    iters = 100

    time_start1 = time.time()
    my_results = my_pso_optimize(bounds, n_particles, iters, dim=len(bounds))
    my_pso_time = time.time() - time_start1

    time_start2 = time.time()
    pyswarms_results = pyswarms_pso_optimize(func = fx.ackley, bounds=bounds, n_particles=n_particles, iters=iters)
    pyswarms_pso_time = time.time() - time_start2

    print(f"My PSO: {my_results}, execution_time: {my_pso_time}")
    print(f"Pyswarm PSO: {pyswarms_results}, execution_time: {pyswarms_pso_time}")

    # Save results to results.csv

    results = {
        'method': ['my_pso', 'pyswarms_pso'],
        'cost': [my_results[0], pyswarms_results[0]],
        'position': [my_results[1], pyswarms_results[1]],
        'execution_time': [my_pso_time, pyswarms_pso_time],
        'cost_history': [my_results[2], pyswarms_results[2]]
    }

    df = pd.DataFrame(results)
    df.to_csv('../analysis/results.csv', index=False)