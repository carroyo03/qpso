import numpy as np
from pso.particle import Particle

class ParticleSwarmOptimization():
    """
    Represents a particle swarm optimization.

    Args:
        number_of_particles (int): The number of particles in the swarm.
        dim (int): The dimension of the search space.
        pos_min (int): The minimum value of the position.
        pos_max (int): The maximum value of the position.
        vel_min (int): The minimum value of the velocity.
        vel_max (int): The maximum value of the velocity.
        w (float): The inertia weight.
        c1 (float): The cognitive weight.
        c2 (float): The social weight.
    """
    def __init__(self, number_of_particles: int, dim: int,
                 bounds: tuple, vel_min: int, vel_max: int,
                 w: float, c1: float, c2: float):
        pos_min,pos_max = bounds
        if pos_min >= pos_max:
            raise ValueError("The minimum position cannot be greater or equal than the maximum")
        self.dim, self.pos_min, self.pos_max = dim, pos_min, pos_max
        self.number_of_particles = number_of_particles
        positions = np.random.uniform(low=pos_min, high=pos_max, size=(number_of_particles, dim))
        velocities = np.random.uniform(low=vel_min, high=vel_max, size=(number_of_particles, dim))
        self.particles = {f"Particle {i}": Particle(position, velocity) for position, velocity, i in
                          zip(positions, velocities, range(number_of_particles))}
        self.pbest = {f"Particle {i}": self.particles[f"Particle {i}"].position.copy() for i in
                      range(number_of_particles)}
        self.gbest = self.get_gbest()
        self.w, self.c1, self.c2 = w, c1, c2

        self.vel_min, self.vel_max = vel_min, vel_max

    def get_gbest(self):
        """
        Gets the gBest particle.
        returns:
            gbest (np.ndarray): The gBest particle position.
        """
        gbest_candidates = {f"Particle {i}": self.objective_function(self.pbest[f"Particle {i}"]) for i in
                            range(self.number_of_particles)}
        name_gbest, _ = min(gbest_candidates.items(), key=lambda item: item[1])
        return self.pbest[name_gbest]

    def update_pbest(self):
        """
        Updates the pBest particles.
        """
        for name, particle in self.particles.items():
            pbest_temp = self.pbest[name]
            if self.objective_function(particle.position) < self.objective_function(pbest_temp):
                self.pbest[name] = particle.position.copy()

    def objective_function(self, position):
        """
        The objective function to be optimized. In this case, the Ackley function.
        Args:
            position (np.ndarray): The position of the particle.
        Returns:
            float: The value of the objective function.
        """



        return -20 * np.exp(-0.2 * np.sqrt((1/self.dim * np.sum(position**2))))-np.exp((1/self.dim * np.sum(np.cos(2 *
  np.pi * position)))) + np.e + 20


    def move_particles(self):
        for particle_name in self.particles.keys():
            pos, vel = self.particles[particle_name].position, self.particles[particle_name].velocity
            r1, r2 = (np.random.uniform(low=0, high=1) for _ in range(2))
            pbest = self.pbest[particle_name]
            gbest = self.gbest
            w, c1, c2 = self.w, self.c1, self.c2
            vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
            vel = np.clip(vel, self.vel_min, self.vel_max)
            pos += vel
            pos = np.clip(pos, self.pos_min, self.pos_max)
            self.particles[particle_name].position, self.particles[particle_name].velocity = pos, vel
        self.update_pbest()
        self.gbest = self.get_gbest()

    def optimize(self, num_iterations: int):
        # Ensure gbest is properly initialized
        if self.gbest is None:
            raise ValueError("self.gbest is not initialized")

        previous_gbest = self.gbest.copy()  # Ensure this is a deep copy if needed
        iterations_with_same_gbest = 0

        for it in range(num_iterations):
            print("-" * 50)
            print(f"Iteration {it + 1}:")
            self.move_particles()
            print(f"-> gBest value: {self.gbest}")
            print(f"-> Objective function value: {self.objective_function(self.gbest)}")
            print()

            # Ensure proper comparison of gbest values
            if np.array_equal(self.gbest, previous_gbest):
                iterations_with_same_gbest += 1
                if iterations_with_same_gbest > 15:
                    print(f"Stopping optimization: gbest has not changed for {iterations_with_same_gbest} iterations.")
                    break
            else:
                iterations_with_same_gbest = 0

            # Update previous gbest for the next iteration
            previous_gbest = self.gbest.copy()

        return self.objective_function(self.gbest),self.gbest
