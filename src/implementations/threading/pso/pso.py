from typing import override
import numpy as np
import concurrent.futures
from core.pso import PSO as BasePSO
from implementations.threading.pso.particle import Particle

class ThreadingPSO(BasePSO):

    def __init__(self, number_of_particles: int, dim: int, bounds: tuple, vel_min: float, vel_max: float, w: float, c1: float, c2: float, max_workers:int=None):
        """
        Initialize a new instance of ThreadingPSO.

        Parameters:
        - number_of_particles (int): The number of particles in the swarm.
        - dim (int): The dimensionality of the problem.
        - bounds (tuple): A tuple containing the lower and upper bounds for each dimension.
        - vel_min (float): The minimum velocity for particles.
        - vel_max (float): The maximum velocity for particles.
        - w (float): The inertia weight for the PSO algorithm.
        - c1 (float): The cognitive component for the PSO algorithm.
        - c2 (float): The social component for the PSO algorithm.
        - max_workers (int, optional): The maximum number of threads to use for parallel processing. Defaults to the number of CPU cores.

        Returns:
        - None
        """
        super().__init__(number_of_particles, dim, bounds, vel_min, vel_max, w, c1, c2)

        self.particles = [Particle(dim, self.pos_min, self.pos_max, vel_min, vel_max) for _ in range(number_of_particles)]

        self.max_workers = max_workers if max_workers else concurrent.futures.cpu_count()

    def initialize(self, objective_function):

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(particle.evaluate,objective_function) for particle in self.particles]
            costs = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        for i, particle in enumerate(self.particles):
            if particle.pbest_cost < self.gbest_cost:
                self.gbest = particle.position.copy()
                self.gbest_cost = particle.pbest_cost
    
    def move_particles(self, w: float, objective_function):
        
        for particle in self.particles:
            particle.update_velocity(w, self.c1, self.c2, self.gbest)
            particle.update_position()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(particle.evaluate, objective_function) for particle in self.particles]
            _ = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        for particle in self.particles:
            if particle.pbest_cost < self.gbest_cost:
                self.gbest = particle.position.copy()
                self.gbest_cost = particle.pbest_cost



def my_threading_pso(function: str, bounds:tuple, n_particles: int, iters: int, w: float, c1: float, c2: float, dim:int=2, vel_min: float=-0.1, vel_max: float=0.1, max_workers:int = None) -> tuple:
    """My Threading PSO implementation.
    Args:
        function (str): The objective function to optimize. Supported functions: "ackley", "rastrigin", "rosenbrock".
        bounds (tuple): Tuple containing the minimum and maximum bounds for the position.
        n_particles (int): Number of particles in the swarm.
        iters (int): Number of iterations for the optimization.
        w (float): Inertia weight.
        c1 (float): Cognitive coefficient.
        c2 (float): Social coefficient.
        dim (int): Number of dimensions for the optimization problem.
        vel_min (float): The minimum velocity for particle movement.
        vel_max (float): The maximum velocity for particle movement.
        max_workers (int): The maximum number of threads to use for parallel processing. Defaults to the number of CPU cores.
    Returns:
        tuple: Best cost, best position, and cost history.
    """
    # Validar la función objetivo
    if function not in ["ackley", "rastrigin", "rosenbrock"]:
        return None, None, None
       
    # Inicializar el optimizador
    optimizer = ThreadingPSO(n_particles, dim, bounds, vel_min, vel_max, w, c1, c2, max_workers)
    
    
    # Ejecutar la optimización asíncrona
    cost, pos, cost_history = optimizer.optimize(iters)
    
    return cost, pos, cost_history
    

def pyswarms_pso(function: str, bounds, n_particles, iters, options: dict) -> tuple:
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
    
    # Unpack parameters and function
    n_particles, iters, w, c1, c2, dim, function = params_and_function
    
    # Set bounds based on the function
    if function == "rastrigin":
        bounds = ([-5.12] * dim, [5.12] * dim)
    else:
        bounds = ([-5] * dim, [5] * dim)
    
    # My Async PSO (without internal progress bar)
    time_start1 = time.time()
    my_result = my_threading_pso(
        function=function, bounds=bounds, n_particles=n_particles, 
        iters=iters, dim=dim, w=w, c1=c1, c2=c2
    )
    my_threading_pso_time = time.time() - time_start1
    
    # PySwarms
    time_start2 = time.time()
    with suppress_stdout_stderr():
        pyswarms_result = pyswarms_pso(
            function, bounds=bounds, n_particles=n_particles, 
            iters=iters, options={'w': w, 'c1': c1, 'c2': c2}
        )
    pyswarms_pso_time = time.time() - time_start2
    
    if pyswarms_result[0] is None:
        params = {'Number of particles' : n_particles, 'Iterations' : iters, 'w': w, 'c1': c1, 'c2': c2, 'Dimensions': dim, 'Function': function}
        # Error handling for PySwarms
        print(f"Error: PySwarms optimization failed for function {function} with params {params}")
        return None
    ""
    # Create result dictionaries without printing anything
    my_result_dict = {
        'method': 'my_threading_pso',
        'function': function,
        'n_particles': n_particles,
        'iters': iters,
        'w': w,
        'c1': c1,
        'c2': c2,
        'dim': dim,
        'cost': my_result[0],
        'position': my_result[1].tolist(),
        'execution_time': my_threading_pso_time,
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
