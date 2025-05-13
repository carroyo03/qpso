from typing import Optional

import numpy as np
import concurrent.futures
from core.pso import PSO as BasePSO
from core.functions import objective
from implementations.threading.pso.particle import Particle
import os

class ThreadingPSO(BasePSO):

    def __init__(self, number_of_particles: int, dim:int, bounds:tuple, vel_min:float, vel_max:float, w:float, c1:float, c2:float, max_workers:Optional[int]=None):
        """
        Initializes the ThreadingPSO instance.

        This constructor sets up the attributes and parameters needed to initialize
        a collection of particles for PSO Algorithm implemented with threading. The particles are
        created within the given dimensionality and bounded space while adhering
        to the specified velocity constraints. A thread pool executor is also
        initialized for efficient task distribution across available workers.

        Parameters:
            number_of_particles: int
                The number of particles in the swarm.
            dim: int
                The number of dimensions for each particle in the search space.
            bounds: tuple
                A tuple defining the lower and upper bounds for particle positions in
                the search space.
            vel_min: float
                The minimum permissible velocity for the particles.
            vel_max: float
                The maximum permissible velocity for the particles.
            w: float
                The inertia weight applied to particle velocities during updates.
            c1: float
                The cognitive coefficient influencing self-learning of particles.
            c2: float
                The social coefficient affecting the attraction to the global best.
            max_workers: Optional[int]
                An optional parameter specifying the maximum number of threads for
                parallel execution. Defaults to the number of available CPU cores
                minus one if not provided.

        Attributes:
            particles: list
                A list of Particle objects representing the particles in the swarm.
            max_workers: int
                The actual number of threads in use by the multithreaded executor.
            executor: concurrent.futures.ThreadPoolExecutor
                The thread pool executor used for distributing tasks among multiple threads.
        """
        super().__init__(number_of_particles, dim, bounds, vel_min, vel_max, w, c1, c2)
        self.pos_min, self.pos_max = bounds

        self.particles = [Particle(dim, self.pos_min, self.pos_max, vel_min, vel_max) for _ in range(number_of_particles)]

        self.max_workers = max_workers if max_workers else os.cpu_count() -1


    def initialize(self, objective_function):
        """
        Initializes the global best position and its corresponding cost by evaluating all particles
        against the provided objective function.

        This method evaluates the given objective function for each particle in the swarm
        using a parallel execution context. It identifies the particle with the best (minimum)
        objective function cost and sets the global best position and corresponding cost accordingly.

        Parameters:
        objective_function : Callable
            The objective function to be evaluated for each particle.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            costs = list(executor.map(lambda p: p.evaluate(objective_function), self.particles))
        best_idx = np.argmin(costs)
        self.gbest = self.particles[best_idx].pbest_position.copy()
        self.gbest_cost = costs[best_idx]
    
    def move_particles(self, w: float, objective_function):
        """
        Moves particles in the swarm by updating their velocities and positions and evaluates their costs
        using the provided objective function. Updates the global best position and cost if a better solution
        is found among the particles.

        Parameters:
            w (float): Inertia weight controlling the impact of the previous velocity on the current velocity.
            objective_function: A callable function that computes the cost for a given particle position.
        """
        _ = [particle.update_velocity_and_position(w, self.c1, self.c2, self.gbest) for particle in self.particles]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            costs = list(executor.map(lambda p: p.evaluate(objective_function), self.particles))
        best_idx = np.argmin(costs)
        if costs[best_idx] < self.gbest_cost:
            self.gbest = self.particles[best_idx].pbest_position.copy()
            self.gbest_cost = costs[best_idx]

    def optimize(self, objective_function, num_iterations):
        """
        Optimize the given objective function using a particle swarm optimization technique.
        This function iteratively adjusts the particles' movements based on the inertia weight
        to find the minimum cost value.

        Parameters:
            objective_function: Callable
                The objective function to minimize.
            num_iterations: int
                The number of iterations to perform the optimization.

        Returns:
            tuple:
                - gbest_cost: float
                    The best cost achieved by the swarm.
                - gbest: ndarray
                    The optimized parameters at the best cost.
                - cost_history: list
                    History of the cost during the optimization iterations.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            self.executor = executor
            # Initialize the particles and find the global best position
            self.initialize(objective_function)
            w_min = .4
            w_values = np.linspace(self.w_init,w_min,num_iterations)
            cost_history = [self.gbest_cost]
            for w in w_values:
                self.move_particles(w, objective_function)
                cost_history.append(self.gbest_cost)

        return self.gbest_cost, self.gbest, cost_history


def my_threading_pso(function: str, bounds: tuple, n_particles: int, num_iterations: int, w: float, c1: float,
                     c2: float, dim: int = 2, vel_min: float = -0.1, vel_max: float = 0.1,
                     max_workers: int = None) -> tuple:
    """
    Performs Particle Swarm Optimization (PSO) using multithreaded processing.

    This function allows optimizing a given mathematical function using the PSO algorithm.
    It distributes the optimization tasks across multiple threads to enhance performance
    and achieves the optimal solution based on the given bounds, particle count, and
    other hyperparameters defining the swarm's behavior.

    Parameters:
        function: str
            The name of the function to be optimized.
        bounds: tuple
            A tuple of tuples, where each inner tuple defines the lower and upper
            bounds for the dimensions of the solution space.
        n_particles: int
            The number of particles in the swarm.
        num_iterations: int
            The number of iterations for the algorithm to execute.
        w: float
            The inertia weight for velocity calculation.
        c1: float
            The cognitive weight indicating personal learning influence.
        c2: float
            The social weight indicating global learning influence.
        dim: int, optional
            The number of dimensions in the solution space. Defaults to 2.
        vel_min: float, optional
            The minimum velocity that a particle can have. Defaults to -0.1.
        vel_max: float, optional
            The maximum velocity that a particle can have. Defaults to 0.1.
        max_workers: int, optional
            The maximum number of threads to use for processing. Defaults to None,
            which uses the available system threads.

    Returns:
        tuple
            A tuple containing the best solution found and the corresponding value
            of the objective function.
    """
    def obj_function(position):
        return objective(function=function, position=position)[0]
    optimizer = ThreadingPSO(n_particles, dim, bounds, vel_min, vel_max, w, c1, c2)
    return optimizer.optimize(obj_function, num_iterations)