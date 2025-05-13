import numpy as np
from core.particle import Particle as BaseParticle

class Particle(BaseParticle):

    def __init__(self, dim, pos_min, pos_max, vel_min, vel_max):
        """
        Represents a particle for my asynchronous implementation of the Particle Swarm
        Optimization (PSO) algorithm. The particle has a position and velocity in a
        multi-dimensional space, personal best position and cost, and boundaries for
        position and velocity.

        Attributes:
            position (numpy.ndarray): Current position of the particle.
            velocity (numpy.ndarray): Current velocity of the particle.
            pbest_position (numpy.ndarray): Best position encountered by the particle
                based on its personal cost evaluation.
            pbest_cost (float): Best personal cost encountered by the particle.
            pos_min (float): Minimum bound for the particle's position.
            pos_max (float): Maximum bound for the particle's position.
            vel_min (float): Minimum bound for the particle's velocity.
            vel_max (float): Maximum bound for the particle's velocity.

        Methods:
            __init__: Initializes the particle with random position and velocity,
                along with its associated attributes.
        """
        # Initialize Particle variables
        self.position = np.random.uniform(low=pos_min, high=pos_max, size=dim)
        self.velocity = np.random.uniform(
            low=-0.1 * (pos_max - pos_min),
            high=0.1 * (pos_max - pos_min),
            size=dim
        )

        # Best personal position and cost
        self.pbest_position = self.position.copy()
        self.pbest_cost = float('inf')

        # Particle bounds
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.vel_min = vel_min
        self.vel_max = vel_max


    async def evaluate_async(self, objective_function):
        """
        Asynchronously evaluates the objective function for the current position of the particle,
        updates the personal best if a new optimal value is found, and returns the computed cost.

        Parameters:
        objective_function : Callable
            A function that takes a position as input and returns the calculated cost for
            that position. It defines the objective the particle is evaluating against.

        Returns:
        float
            The evaluated cost for the current position of the particle.
        """
        # Asyncronously evaluate the objective function
        cost = objective_function(self.position)

        # Update personal best if needed
        if cost < self.pbest_cost:
            self.pbest_position = self.position.copy()
            self.pbest_cost = cost

        return cost
