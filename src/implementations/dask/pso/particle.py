from typing import override
import numpy as np
from core.particle import Particle as BaseParticle

class DaskParticle(BaseParticle):

    def __init__(self, dim: int, pos_min: np.ndarray, pos_max: np.ndarray, vel_min: float, vel_max: float):
        super().__init__(dim, pos_min, pos_max, vel_min, vel_max)
    
    def evaluate(self, objective_function):
        cost = objective_function(self.position)

        if cost < self.pbest_cost:
            self.pbest_position = self.position.copy()
            self.pbest_cost = cost
        return cost