
from core.particle import Particle as BaseParticle

class Particle(BaseParticle):
    """
    A Particle class representing an individual in a particle swarm optimization algorithm.

    Args:
        BaseParticle: The base class for particles.
    """
    # Due to the fact that the BaseParticle and Particle will be sharing the same memory space, 
    # the base implementation will be used.