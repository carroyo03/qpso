import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
from src.core.particle import Particle

DIM = 3
POS_MIN = np.array([-5, -5, -5])
POS_MAX = np.array([5, 5, 5])
VEL_MIN = -1.0
VEL_MAX = 1.0

# A simple objective function for testing (Sphere function)
mock_objective_func = lambda x: np.sum(x**2)

@pytest.fixture
def particle_instance():
    """Returns a Particle instance for testing."""
    return Particle(DIM, POS_MIN, POS_MAX, VEL_MIN, VEL_MAX)

# Initialization Tests
def test_particle_initialization_position_bounds(particle_instance):
    p = particle_instance
    assert np.all(p.position >= POS_MIN) and np.all(p.position <= POS_MAX)

def test_particle_initialization_velocity_bounds(particle_instance):
    p = particle_instance
    # Default velocity initialization is [-0.1*(pos_max-pos_min), 0.1*(pos_max-pos_min)]
    # The Particle class itself does not clip initial velocity to VEL_MIN, VEL_MAX passed to constructor.
    # It stores vel_min and vel_max for later use in update_velocity_and_position.
    expected_vel_min_init = -0.1 * (POS_MAX - POS_MIN)
    expected_vel_max_init = 0.1 * (POS_MAX - POS_MIN)
    assert np.all(p.velocity >= expected_vel_min_init) and np.all(p.velocity <= expected_vel_max_init)
    # Also check that vel_min and vel_max (from constructor) are stored correctly
    assert p.vel_min == VEL_MIN
    assert p.vel_max == VEL_MAX


def test_particle_initialization_pbest_position(particle_instance):
    p = particle_instance
    assert_array_equal(p.pbest_position, p.position)

def test_particle_initialization_pbest_cost(particle_instance):
    p = particle_instance
    assert p.pbest_cost == float('inf')

# Evaluate method tests
def test_evaluate_updates_pbest_first_time(particle_instance):
    p = particle_instance
    cost = p.evaluate(mock_objective_func)
    
    assert_almost_equal(cost, mock_objective_func(p.position))
    assert_array_equal(p.pbest_position, p.position)
    assert_almost_equal(p.pbest_cost, cost)

def test_evaluate_updates_pbest_better_position(particle_instance):
    p = particle_instance
    # First evaluation
    p.evaluate(mock_objective_func)
    
    # Force a better position (closer to origin for sphere function)
    original_cost = p.pbest_cost
    p.position = p.pbest_position / 2.0 
    new_cost = p.evaluate(mock_objective_func)
    
    assert new_cost < original_cost
    assert_array_equal(p.pbest_position, p.position)
    assert_almost_equal(p.pbest_cost, new_cost)

def test_evaluate_does_not_update_pbest_worse_position(particle_instance):
    p = particle_instance
    # First evaluation
    p.evaluate(mock_objective_func)
    
    original_pbest_pos = p.pbest_position.copy()
    original_pbest_cost = p.pbest_cost
    
    # Ensure initial position is not [0,0,0] for sphere func to guarantee 'worse' means higher cost
    if np.all(p.position == 0): 
        p.position = np.array([1.0, 1.0, 1.0])
        p.evaluate(mock_objective_func) 
        original_pbest_pos = p.pbest_position.copy()
        original_pbest_cost = p.pbest_cost

    p.position = original_pbest_pos * 10.0 # Make position worse
    new_cost = p.evaluate(mock_objective_func)

    assert new_cost > original_pbest_cost
    assert_array_equal(p.pbest_position, original_pbest_pos)
    assert_almost_equal(p.pbest_cost, original_pbest_cost)

# update_velocity_and_position method tests
W = 0.5
C1 = 1.5
C2 = 1.5
GBEST_POSITION = np.array([0.1, 0.2, 0.3]) # Example global best

def test_update_velocity_and_position_updates_velocity_position(particle_instance):
    p = particle_instance
    initial_pos = p.position.copy()
    initial_vel = p.velocity.copy()
    
    p.update_velocity_and_position(W, C1, C2, GBEST_POSITION)
    
    assert not np.array_equal(p.position, initial_pos)
    assert not np.array_equal(p.velocity, initial_vel)

def test_update_velocity_clipping(particle_instance):
    p = particle_instance
    
    # Scenario 1: Velocity goes above VEL_MAX
    p.velocity = np.array([VEL_MAX * 10, VEL_MAX * 10, VEL_MAX*10]) 
    p.update_velocity_and_position(1.0, 0, 0, p.position) 
    assert np.all(p.velocity <= VEL_MAX)
    assert_array_almost_equal(p.velocity, np.array([VEL_MAX, VEL_MAX, VEL_MAX]))

    # Scenario 2: Velocity goes below VEL_MIN
    p.velocity = np.array([VEL_MIN * 10, VEL_MIN * 10, VEL_MIN * 10]) 
    p.update_velocity_and_position(1.0, 0, 0, p.position) 
    assert np.all(p.velocity >= VEL_MIN)
    assert_array_almost_equal(p.velocity, np.array([VEL_MIN, VEL_MIN, VEL_MIN]))

def test_update_position_clipping(particle_instance):
    p = particle_instance
    
    p.position = POS_MAX.copy() 
    p.velocity = np.array([VEL_MAX, VEL_MAX, VEL_MAX]) 
    p.update_velocity_and_position(1.0, 0, 0, p.position)
    assert np.all(p.position <= POS_MAX)
    if VEL_MAX > 0:
         assert_array_almost_equal(p.position, POS_MAX)

    p.position = POS_MIN.copy()
    p.velocity = np.array([VEL_MIN, VEL_MIN, VEL_MIN]) 
    p.update_velocity_and_position(1.0, 0, 0, p.position)
    assert np.all(p.position >= POS_MIN)
    if VEL_MIN < 0:
        assert_array_almost_equal(p.position, POS_MIN)

# Example of how to test with a fixed random seed if needed for stochastic parts,
# though here we forced states to test clipping deterministically.
# def test_update_with_fixed_seed():
#     np.random.seed(0)
#     p1 = Particle(DIM, POS_MIN, POS_MAX, VEL_MIN, VEL_MAX)
#     p1.update_velocity_and_position(W, C1, C2, GBEST_POSITION)
#     pos1, vel1 = p1.position.copy(), p1.velocity.copy()
#
#     np.random.seed(0)
#     p2 = Particle(DIM, POS_MIN, POS_MAX, VEL_MIN, VEL_MAX) # Need to re-initialize for same state if init is random
#     p2.update_velocity_and_position(W, C1, C2, GBEST_POSITION)
#     pos2, vel2 = p2.position.copy(), p2.velocity.copy()
#
#     assert_array_equal(pos1, pos2)
#     assert_array_equal(vel1, vel2)

# The Particle class stores pos_min, pos_max, vel_min, vel_max.
# The 'dim' parameter in __init__ is used to set the size of self.position and self.velocity.
def test_particle_attributes_and_dimensions(particle_instance):
    p = particle_instance
    assert len(p.position) == DIM
    assert len(p.velocity) == DIM
    assert len(p.pbest_position) == DIM
    
    assert_array_equal(p.pos_min, POS_MIN)
    assert_array_equal(p.pos_max, POS_MAX)
    assert p.vel_min == VEL_MIN
    assert p.vel_max == VEL_MAX

# Test that compute_velocity and compute_position are actually used
# This is more of an integration test within the particle,
# but it's important. The current tests for update_velocity_and_position
# inherently test this.
# For example, test_update_velocity_clipping relies on compute_velocity's clipping.
# test_update_position_clipping relies on compute_position's clipping.
