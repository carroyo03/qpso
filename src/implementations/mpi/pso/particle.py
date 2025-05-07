from core.particle import Particle as BaseParticle
import numpy as np
from mpi4py import MPI # type: ignore

class MPIParticle(BaseParticle):
    
    def __init__(self, dim: int, pos_min: np.ndarray, pos_max: np.ndarray, vel_min: float, vel_max: float, comm=None, rank=None):

        super().__init__(dim, pos_min, pos_max, vel_min, vel_max)
        self.comm = comm or MPI.COMM_WORLD
        self.rank = rank or self.comm.Get_rank()
    
