import numpy as np

# The Particle class implements a particle for Particle Swarm Optimization (PSO) with methods for
# updating velocity, position, and evaluating the particle's position.
class Particle:

    def __init__(self, dim: int, pos_min: np.ndarray, pos_max: np.ndarray,
                vel_min: float, vel_max: float):
        """
        The function initializes a particle with random position and velocity within specified limits
        for a particle swarm optimization algorithm.
        
        Args:
          dim (int): The `dim` parameter in the `__init__` method represents the dimensionality of the
        problem space. It indicates the number of dimensions in which the particles in the optimization
        algorithm will be moving and searching for the optimal solution.
          pos_min (np.ndarray): `pos_min` is a numpy array representing the minimum position values for
        each dimension of the particle in a particle swarm optimization algorithm. It defines the lower
        bounds within which the particle's position can vary during optimization.
          pos_max (np.ndarray): The `pos_max` parameter represents the maximum position values allowed
        for each dimension in the search space. It is used to define the upper boundary for the randomly
        initialized position of a particle in a particle swarm optimization algorithm. The particle's
        position is initialized within the range defined by `pos_min` and `
          vel_min (float): The `vel_min` parameter in the `__init__` method represents the minimum
        velocity value that can be assigned to the particle during initialization. This value is used to
        set the lower bound for the random initialization of the velocity components of the particle
        within a certain range. It helps control the speed at
          vel_max (float): The `vel_max` parameter in the `__init__` method appears to represent the
        maximum velocity value allowed for the particle in a particle swarm optimization (PSO)
        algorithm. This value is used to initialize the velocity of the particle within a certain range
        during the initialization phase.
        """
       
        
        # Inicializa la posición aleatoriamente dentro de los límites
        self.position = np.random.uniform(low=pos_min, high=pos_max, size=dim)
        
        # Inicializa la velocidad aleatoriamente (típicamente menor que el rango de posición)
        self.velocity = np.random.uniform(
            low=-0.1 * (pos_max - pos_min),
            high=0.1 * (pos_max - pos_min),
            size=dim
        )
        
        # Inicializa el mejor personal
        self.pbest_position = self.position.copy()
        self.pbest_cost = float('inf')  # Se actualizará en la primera evaluación
        
        # Almacena los límites para recorte
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.vel_min = vel_min
        self.vel_max = vel_max
    
    def update_velocity(self, w: float, c1: float, c2: float, gbest_position: np.ndarray):
        """
        The function `update_velocity` calculates the new velocity of a particle in a particle swarm
        optimization algorithm based on inertia, cognitive, and social components.
        
        Args:
          w (float): The parameter `w` in the `update_velocity` function represents the inertia weight
        in a particle swarm optimization algorithm. It controls the impact of the previous velocity on
        the current velocity update. A higher inertia weight allows the particle to maintain its current
        direction, while a lower inertia weight enables the particle to explore
          c1 (float): In the provided code snippet, the parameter `c1` represents the cognitive
        parameter used in the Particle Swarm Optimization (PSO) algorithm. This parameter controls the
        particle's attraction towards its personal best position. It is multiplied by a random value
        `r1` and the difference between the particle's personal
          c2 (float): The parameter `c2` in the `update_velocity` function represents the cognitive
        component, which determines the attraction towards the global best position in the particle
        swarm optimization algorithm. It is a constant that scales the influence of the global best
        position on updating the velocity of a particle.
          gbest_position (np.ndarray): The `gbest_position` parameter in the `update_velocity` function
        represents the best global position found by any particle in the swarm. It is used to calculate
        the social component of the velocity update, which attracts the particle towards this global
        best position. This helps the particle explore the search space efficiently by
        """
       

        # Componentes aleatorios para comportamiento estocástico
        r1 = np.random.random()
        r2 = np.random.random()
        
        # Componente cognitivo (atracción hacia el mejor personal)
        cognitive = c1 * r1 * (self.pbest_position - self.position)
        
        # Componente social (atracción hacia el mejor global)
        social = c2 * r2 * (gbest_position - self.position)
        
        # Actualiza la velocidad con componentes de inercia, cognitivo y social
        self.velocity = w * self.velocity + cognitive + social
        
        # Recorta la velocidad a los límites
        self.velocity = np.clip(self.velocity, self.vel_min, self.vel_max)
    
    def update_position(self):
        """
        The `update_position` function updates the position of an object and ensures it stays within
        specified limits.
        """

        # Actualiza la posición
        self.position += self.velocity
        
        # Recorta la posición a los límites
        self.position = np.clip(self.position, self.pos_min, self.pos_max)
    
    def evaluate(self, objective_function):
        """
        This function evaluates the current position, updates the personal best if the current position
        is better, and returns the cost.
        
        Args:
          objective_function: The `objective_function` parameter in the `evaluate` method is a function
        that takes a position as input and returns a cost value. This function is used to evaluate the
        current position of a particle in a particle swarm optimization algorithm. The cost value
        represents how well the particle's position performs with respect to
        
        Returns:
          The `evaluate` method returns the cost of the current position after evaluating it with the
        objective function.
        """
       
        
        # Evalúa la posición actual
        cost = objective_function(self.position)
        
        # Actualiza el mejor personal si la posición actual es mejor
        if cost < self.pbest_cost:
            self.pbest_position = self.position.copy()
            self.pbest_cost = cost
            
        return cost
    
    def __del__():
      """
      The `__del__` method is a special method that is automatically called when an object is deleted
      from memory. In this case, it is used to release any resources or memory allocated to the
      particle object.
      """
      pass