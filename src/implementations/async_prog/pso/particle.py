import numpy as np
from core.particle import Particle as BaseParticle

class Particle(BaseParticle):

    async def evaluate_async(self, objective_function):

        cost = await objective_function(self.position)

        if cost < self.pbest_cost:
            self.pbest_position = self.position.copy()
            self.pbest_cost = cost
        return cost