from typing import override
import numpy as np
from numba import njit, float64
from core.particle import Particle as BaseParticle

@njit(fastmath=True)
def update_velocity_numba(velocity, position, pbest_position, gbest_position, w, c1, c2, vel_min, vel_max):

    r1 = np.random.random()
    r2 = np.random.random()
    cognitive = c1 * r1 * (pbest_position - position)
    social = c2 * r2 * (gbest_position - position)
    new_velocity = w * velocity + cognitive + social

    return np.clip(new_velocity, vel_min, vel_max)

@njit(fastmath=True)
def update_position_numba(position, velocity, pos_min, pos_max):
    new_position = position + velocity
    return np.clip(new_position, pos_min, pos_max)

@njit(fastmath=True)
def evaluate_position_numba(position, objective_function):
    return objective_function(position)

class NumbaParticle(BaseParticle):
    def __init__(self, dim: int, pos_min: np.ndarray, pos_max: np.ndarray, vel_min: float, vel_max: float):
        super().__init__(dim, pos_min, pos_max, vel_min, vel_max)
    def update_velocity(self, pbest_position, gbest_position, w, c1, c2):
        self.velocity = update_velocity_numba(
            self.velocity, self.position, self.pbest_position, gbest_position, 
            w, c1, c2, self.vel_min, self.vel_max
        )
    
    def update_position(self):
        self.position = update_position_numba(
            self.position, self.velocity, self.pos_min, self.pos_max
        )
    
    def update_position(self):
        self.position = update_position_numba(
            self.position, self.velocity, self.pos_min, self.pos_max
        )
    def evaluate(self, objective_function):
        try:
            cost = evaluate_position_numba(self.position, objective_function)
        except Exception as e:
            print(f"Function not compatible with Numba: {e}")
            cost = objective_function(self.position)
        
        if cost < self.pbest_cost:
            self.pbest_position = self.position.copy()
            self.pbest_cost = cost
        
        return cost
