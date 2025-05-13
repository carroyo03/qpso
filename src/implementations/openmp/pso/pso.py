from ast import Import
import numpy as np
from core.functions import objective
import time
import pyswarms as ps #type: ignore
from pyswarms.utils.functions import single_obj as fx #type: ignore
from implementations.openmp.pso.particle import Particle

from core.pso import PSO as BasePSO


class OpenMP_PSO(BasePSO):
    def __init__(self, number_of_particles: int, dim: int,
    bounds: tuple, vel_min: float, vel_max: float,
    w: float, c1: float, c2: float, function: str):
        """Particle Swarm Optimization (PSO) algorithm implementation for OpenMP.
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
        super().__init__(number_of_particles, dim, bounds, vel_min, vel_max, w, c1, c2)
        self.function = function

        
        self.particles = [
            Particle(dim, self.pos_min, self.pos_max, vel_min, vel_max)
            for _ in range(number_of_particles)
        ]
        
        # Inicializa el enjambre con la función objetivo específica
        self.initialize(
            lambda x: objective(
                position=x.reshape(1, -1),
                function=self.function
            )[0]
        )
        

    def move_particles(self, w: float):
        """Update the particles' positions and velocities.
        Args:
            w (float): Inertia weight.
        """
        for particle in self.particles:
            # Update velocity and position
            particle.update_velocity_and_position(
                w=w, c1=self.c1, c2=self.c2, gbest_position=self.gbest
            )
            
            # Evaluate new position
            cost = particle.evaluate(
                lambda x: objective(
                    position=x.reshape(1, -1), 
                    function=self.function
                    )[0],
                function_name=self.function
            )
            
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

def my_openmp_pso(function: str, bounds:tuple, n_particles: int, iters: int, w: float, c1: float, c2: float, dim:int=2, vel_min: float=-0.1, vel_max: float=0.1) -> tuple:
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
    optimizer = OpenMP_PSO(
        number_of_particles=n_particles, dim=dim, bounds=bounds,
        vel_min=vel_min, vel_max=vel_max, w=w, c1=c1, c2=c2, function=function
    )
    # Initialize the cost, position and cost history obtained from the optimizer
    cost, pos, cost_history = optimizer.optimize(num_iterations=iters)
    return cost, pos, cost_history

