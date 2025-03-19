#import numpy as np
from pso import ParticleSwarmOptimization

if __name__ == "__main__":
    pso = ParticleSwarmOptimization(number_of_particles=20, dim=2, pos_min=-5, pos_max=5, vel_min=-1, vel_max=1, w=0.7,
                                    c1=2, c2=2)
    result = pso.optimize(100)
    print(f"Final gBest: {result}, Objective value: {pso.objective_function(result)}")
