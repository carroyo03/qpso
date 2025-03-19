import numpy as np

class Particle():
    """
    Represents a particle in the swarm.

    Args:
        position (np.ndarray): The position of the particle in the search space.
        velocity (np.ndarray): The velocity of the particle in the search space.

    """

    def __init__(self, position, velocity):
        self.position = np.array(position)
        self.velocity = np.array(velocity)