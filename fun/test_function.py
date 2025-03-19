from pso.pso import ParticleSwarmOptimization
import pyswarms as ps

def optimize(func, bounds, n_particles, iters):
    #options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ParticleSwarmOptimization(number_of_particles=20, dim=2, pos_min=-5, pos_max=5, vel_min=-1, vel_max=1, w=.9,
                                    c1=.5, c2=.3)
    cost, pos = optimizer.optimize()

def pyswarm_pso_optimize(func, bounds, n_particles, iters):
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=len(bounds), options={'c1': 0.5, 'c2': 0.3, 'w': 0.9})
    cost, pos = optimizer.optimize(func, iters=iters, bounds=bounds)
    return cost, pos

if __name__ == '__main__':
    pso = ParticleSwarmOptimization()