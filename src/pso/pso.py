import numpy as np
from fun.functions import objective #type: ignore
import time
import pyswarms as ps #type: ignore
from pyswarms.utils.functions import single_obj as fx #type: ignore
from pso.particle import Particle  # type: ignore


class ParticleSwarmOptimization:
    def __init__(self, number_of_particles: int, dim: int,
    bounds: tuple, vel_min: float, vel_max: float,
    w: float, c1: float, c2: float, function: str):
        """Particle Swarm Optimization (PSO) algorithm implementation.
        Args:
            number_of_particles (int): Number of particles in the swarm.
            dim (int): Number of dimensions for the optimization problem.
            bounds (tuple): Tuple containing the minimum and maximum bounds for the position.
            vel_min (float): Minimum velocity for particles.
            vel_max (float): Maximum velocity for particles.
            w (float): Inertia weight.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
            function (str): The objective function to optimize. Supported functions: "ackley", "rastrigin", "rosenbrock".
        """
        pos_min, pos_max = bounds
        pos_min = np.array(pos_min)
        pos_max = np.array(pos_max)

        if np.any(pos_min >= pos_max):
            raise ValueError("The minimum position cannot be greater or equal than the maximum")

        self.number_of_particles = number_of_particles
        self.dim = dim
        self.pos_min, self.pos_max = pos_min, pos_max
        self.vel_min, self.vel_max = vel_min, vel_max
        self.w_init = w  # Initial inertia weight
        self.c1, self.c2 = c1, c2
        self.function = function

        # Initialize particles using the Particle class
        self.particles = [
            Particle(dim, pos_min, pos_max, vel_min, vel_max) 
            for _ in range(number_of_particles)
        ]
        
        # Evaluate all particles initially
        for particle in self.particles:
            particle.evaluate(lambda x: objective(x.reshape(1, -1))[0])
        
        # Find global best
        gbest_idx = np.argmin([p.pbest_cost for p in self.particles])
        self.gbest = self.particles[gbest_idx].pbest_position.copy()
        self.gbest_cost = self.particles[gbest_idx].pbest_cost
        
        # Initialize cost history
        self.cost_history:list[float] = []

    def move_particles(self, w: float):
        """Update the particles' positions and velocities.
        Args:
            w (float): Inertia weight.
        """
        for particle in self.particles:
            # Update velocity and position
            particle.update_velocity(w, self.c1, self.c2, self.gbest)
            particle.update_position()
            
            # Evaluate new position
            cost = particle.evaluate(lambda x: objective(x.reshape(1, -1))[0])
            
            # Update global best if needed
            if cost < self.gbest_cost:
                self.gbest = particle.position.copy()
                self.gbest_cost = cost

    def optimize(self, num_iterations: int) -> tuple:
        """Optimize the objective function using PSO.
        Args:
            num_iterations (int): Number of iterations for the optimization.
        Returns:
            tuple: Best cost, best position, and cost history.
        """
        # Initialize the inertia weight schedule
        # Linearly decrease the inertia weight from w_init to w_min
        w_min = 0.4
        w_values = np.linspace(self.w_init, w_min, num_iterations)
        
        # Initialize cost history
        self.cost_history = []
        
        for it in range(num_iterations):
            # Update the inertia weight and move particles
            self.move_particles(w_values[it])
            self.cost_history.append(self.gbest_cost)
        
        # Final evaluation
        return self.gbest_cost, self.gbest, self.cost_history

def my_pso(function: str, bounds:tuple, n_particles: int, iters: int, w: float, c1: float, c2: float, dim:int=2, vel_min: float=-0.1, vel_max: float=0.1) -> tuple:
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
        vel_min=vel_min, vel_max=vel_max, w=w, c1=c1, c2=c2, function=function
    )
    # Initialize the cost, position and cost history obtained from the optimizer
    cost, pos, cost_history = optimizer.optimize(num_iterations=iters)
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
    
    params, function = params_and_function
    n_particles, iters, w, c1, c2, dim = params
    
    # Set bounds based on the function
    if function == "rastrigin":
        bounds = ([-5.12] * dim, [5.12] * dim)
    else:
        bounds = ([-5] * dim, [5] * dim)
    
    # My PSO (without internal progress bar)
    time_start1 = time.time()
    my_result = my_pso(
        function=function, bounds=bounds, n_particles=n_particles, 
        iters=iters, dim=dim, w=w, c1=c1, c2=c2
    )
    my_pso_time = time.time() - time_start1
    
    # PySwarms
    time_start2 = time.time()
    with suppress_stdout_stderr():
        pyswarms_result = pyswarms_pso(
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