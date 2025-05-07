from typing import override
import numpy as np
import asyncio
from core.pso import PSO as BasePSO
from implementations.async_prog.pso.particle import Particle
from core.functions import objective

class AsyncPSO(BasePSO):

    def __init__(self, number_of_particles: int, dim: int, bounds: tuple, vel_min: float, vel_max: float, w: float, c1: float, c2: float):
        """
        Initialize the Asynchronous Particle Swarm Optimization (AsyncPSO) algorithm.
    
        Parameters:
        - number_of_particles (int): The number of particles in the swarm.
        - dim (int): The number of dimensions for the optimization problem.
        - bounds (tuple): A tuple containing the minimum and maximum bounds for the position.
        - vel_min (float): The minimum velocity for particle movement.
        - vel_max (float): The maximum velocity for particle movement.
        - w (float): The inertia weight for the velocity update.
        - c1 (float): The cognitive coefficient for the velocity update.
        - c2 (float): The social coefficient for the velocity update.
    
        Returns:
        - None
        """
        super().__init__(number_of_particles, dim, bounds, vel_min, vel_max, w, c1, c2)
    
        self.particles = [Particle(dim,self.pos_min, self.pos_max,vel_min, vel_max) for _ in range(number_of_particles)]
    
    async def initialize_async(self, objective_function):
        """
        Initialize particles asynchronously using the provided objective function.
        
        Args:
            objective_function: The objective function to be evaluated.
        """

        tasks = [particle.evaluate_async(objective_function) for particle in self.particles]
        costs = await asyncio.gather(*tasks)

        best_idx = np.argmin(costs)
        self.gbest = self.particles[best_idx].position.copy()
        self.gbest_cost = costs[best_idx]
    
    async def move_particles_async(self, w: float, objective_function):
        """
        Update the position of each particle in the swarm asynchronously using the objective function.
        
        Parameters:
        - w (float): The inertia weight for the velocity update.
        - objective_function: The objective function to be evaluated.
        
        This method iterates over each particle in the swarm, updates its velocity and position,
        evaluates the objective function asynchronously, and updates the global best position and cost
        if a better solution is found.
        """

        for particle in self.particles:
            particle.update_velocity(w, self.c1,self.c2,self.gbest)
            particle.update_position()
        
        tasks = [particle.evaluate_async(objective_function) for particle in self.particles]
        costs = await asyncio.gather(*tasks)

        for i, cost in enumerate(costs):
            if cost < self.gbest_cost:
                self.gbest = self.particles[i].position.copy()
                self.gbest_cost = cost
    
    async def optimize_async(self, objective_function, max_iterations: int):

        """
        Asynchronously optimizes the swarm using the provided objective function for a specified number of iterations.
        
        Parameters:
        - objective_function: The objective function to be evaluated.
        - max_iterations (int): The maximum number of iterations for the optimization.
        
        Returns:
        - tuple: A tuple containing the best cost, best position, and cost history.
        """
        
        # Inicializa el enjambre de forma asíncrona
        await self.initialize_async(objective_function)
        
        # Inicializa el programa de peso de inercia
        # Disminuye linealmente el peso de inercia desde w_init hasta w_min
        w_min = 0.4
        w_values = np.linspace(self.w_init, w_min, max_iterations)
        
        # Inicializa el historial de costos
        self.cost_history = []
        
        for it in range(max_iterations):
            # Actualiza el peso de inercia y mueve las partículas de forma asíncrona
            await self.move_particles_async(w_values[it], objective_function)
            self.cost_history.append(self.gbest_cost)
        
        # Evaluación final
        return self.gbest_cost, self.gbest, self.cost_history


def my_async_pso(function: str, bounds:tuple, n_particles: int, iters: int, w: float, c1: float, c2: float, dim:int=2, vel_min: float=-0.1, vel_max: float=0.1) -> tuple:
    """
    My Async Particle Swarm Optimization (AsyncPSO) implementation.
    This function uses the asyncio module to run the optimization in an asynchronous manner.
    The objective function is converted to an asynchronous function and then used in the optimization process.

    Args:
        function (str): The objective function to optimize. Supported functions: "ackley", "rastrigin", "rosenbrock".
        bounds (tuple): Tuple containing the minimum and maximum bounds for the position.
        n_particles (int): Number of particles in the swarm.
        iters (int): Number of iterations for the optimization.
        w (float): The inertia weight for the velocity update.
        c1 (float): The cognitive coefficient for the velocity update.
        c2 (float): The social coefficient for the velocity update.
        dim (int): Number of dimensions for the optimization problem.
        vel_min (float): The minimum velocity for particle movement.
        vel_max (float): The maximum velocity for particle movement.

    Returns:
        Tuple: A tuple containing the best cost, best position, and cost history.
    """
    
    def async_objective(position):
        return objective(position.reshape(1, -1), function=function)[0]
    
    optimizer = AsyncPSO(n_particles, dim, bounds, vel_min, vel_max, w, c1, c2)
    
    # Ejecutar la optimización asíncrona
    cost, pos, cost_history = asyncio.run(optimizer.optimize_async(async_objective, iters))
    
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
    my_result = my_async_pso(
        function=function, bounds=bounds, n_particles=n_particles, 
        iters=iters, dim=dim, w=w, c1=c1, c2=c2
    )
    my_async_pso_time = time.time() - time_start1
    
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
        'method': 'my_async_pso',
        'function': function,
        'n_particles': n_particles,
        'iters': iters,
        'w': w,
        'c1': c1,
        'c2': c2,
        'dim': dim,
        'cost': my_result[0],
        'position': my_result[1].tolist(),
        'execution_time': my_async_pso_time,
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