import numpy as np
from .particle import Particle


# The `PSO` class implements a particle swarm optimization algorithm with specified parameters,
# initializing particles and updating their positions to optimize an objective function.
class PSO:

    
    def __init__(self, number_of_particles: int, dim: int,
                 bounds: tuple, vel_min: float, vel_max: float,
                 w: float, c1: float, c2: float):
        """
        The function initializes a particle swarm optimization algorithm with specified parameters and
        initializes particles and global best values.
        
        Args:
          number_of_particles (int): `number_of_particles` is the number of particles in the particle
        swarm optimization algorithm. It represents the population size, i.e., the number of potential
        solutions that will be updated and optimized in each iteration of the algorithm.
          dim (int): The `dim` parameter represents the dimensionality of the problem space. It
        indicates the number of dimensions in which the particles will be moving and searching for the
        optimal solution.
          bounds (tuple): The `bounds` parameter in the `__init__` method represents the boundaries
        within which the particles will be initialized. It is a tuple containing the minimum and maximum
        values for each dimension of the search space. These boundaries define the valid range for the
        position of the particles in each dimension.
          vel_min (float): The `vel_min` parameter represents the minimum velocity that a particle can
        have in the particle swarm optimization algorithm. It is used to control the movement of
        particles within the search space.
          vel_max (float): `vel_max` represents the maximum velocity that a particle can have in the
        particle swarm optimization algorithm. It is used to update the position of each particle based
        on its current velocity during the optimization process.
          w (float): The parameter `w` in the initialization function represents the inertia weight in a
        particle swarm optimization algorithm. It controls the impact of the previous velocity on the
        current velocity of particles during optimization. A higher inertia weight allows particles to
        maintain their current direction, while a lower weight enables particles to explore new areas
        more
          c1 (float): The parameter `c1` in the code snippet you provided represents the cognitive
        component of the particle swarm optimization algorithm. It is a constant that determines the
        weight of the particle's personal best position when updating its velocity. In other words, `c1`
        influences how much a particle is influenced by its
          c2 (float): The parameter `c2` in the initialization function represents the cognitive
        component of the particle swarm optimization algorithm. It is a constant that determines the
        weight of the particle's personal best position when updating its velocity. This component
        influences how much a particle is attracted to its own best-known position.
        """

        pos_min, pos_max = bounds
        pos_min = np.array(pos_min)
        pos_max = np.array(pos_max)

        if np.any(pos_min >= pos_max):
            raise ValueError("The minimum position cannot be greater than or equal to the maximum position")

        self.number_of_particles = number_of_particles
        self.dim = dim
        self.pos_min, self.pos_max = pos_min, pos_max
        self.vel_min, self.vel_max = vel_min, vel_max
        self.w_init = w  # Initial inertia weight
        self.c1, self.c2 = c1, c2
        
        # Initialize particles using the Particle class
        self.particles = [
            Particle(dim, pos_min, pos_max, vel_min, vel_max) 
            for _ in range(number_of_particles)
        ]
        
        # Initialize global best
        self.gbest = None
        self.gbest_cost = float('inf')
        
        # Initialize cost history
        self.cost_history = []
    
    def initialize(self, objective_function):
        """
        The function initializes particles and updates the global best position based on the evaluation
        of an objective function.
        
        Args:
          objective_function: The `objective_function` parameter in the given code snippet is a function
        that represents the objective function being optimized by a particle swarm optimization
        algorithm. This function takes a particle's position as input and returns the cost or fitness
        value associated with that position in the optimization problem.
        """
      
        # Evaluate all particles initially
        for particle in self.particles:
            cost = particle.evaluate(objective_function)
            
            # Update global best if necessary
            if cost < self.gbest_cost:
                self.gbest = particle.position.copy()
                self.gbest_cost = cost
    
    def move_particles(self, w: float, objective_function):
        """
        The function `move_particles` updates the velocity and position of particles in a swarm
        optimization algorithm and evaluates the new positions to update the global best if necessary.
        
        Args:
          w (float): In the context of the code snippet you provided, the parameter `w` represents the
        inertia weight in the particle swarm optimization algorithm. This weight is used to control the
        impact of the previous velocity on the current velocity of particles during the optimization
        process. It helps balance exploration and exploitation in the search space.
          objective_function: The `objective_function` parameter in the `move_particles` method is a
        function that represents the objective function that you are trying to optimize using the
        particle swarm optimization algorithm. This function takes the current position of a particle as
        input and returns the cost or fitness value associated with that position.
        """



        for particle in self.particles:
            # update velocity and position
            particle.update_velocity_and_position(w, self.c1, self.c2, self.gbest)

            # Evaluate the new position and update personal best
            cost = particle.evaluate(objective_function)

            # Update global best if necessary
            if cost < self.gbest_cost:
                self.gbest = particle.position.copy()
                self.gbest_cost = cost
    
    def optimize(self, objective_function, num_iterations: int) -> tuple:
        """
        The `optimize` function initializes a swarm, updates the inertia weight, moves particles, and
        tracks the cost history for a specified number of iterations in a particle swarm optimization
        algorithm.
        
        Args:
          objective_function: The `objective_function` parameter in the `optimize` method is a function
        that represents the objective function that the particle swarm optimization algorithm will try
        to minimize or maximize. This function should take the current position of a particle as input
        and return the cost or fitness value associated with that position.
          num_iterations (int): The `num_iterations` parameter in the `optimize` method represents the
        number of iterations or steps that the optimization algorithm will perform to search for the
        optimal solution. It determines how many times the algorithm will update the particle positions
        and evaluate the objective function during the optimization process. Increasing the number of
        iterations can
        
        Returns:
          The `optimize` method returns a tuple containing the following elements:
        1. The best cost found (`self.gbest_cost`)
        2. The best particle found (`self.gbest`)
        3. The history of costs during the optimization process (`self.cost_history`)
        """
        
        # Initialize the swarm
        self.initialize(objective_function)
        
        # Initialize inertia weight schedule
        # Linearly decrease inertia weight from w_init to w_min
        w_min = 0.4
        w_values = np.linspace(self.w_init, w_min, num_iterations)
        
        # Initialize cost history
        self.cost_history = []
        
        for it in range(num_iterations):
            # Update inertia weight and move particles
            self.move_particles(w_values[it], objective_function)
            self.cost_history.append(self.gbest_cost)
        
        # Final evaluation
        return self.gbest_cost, self.gbest, self.cost_history