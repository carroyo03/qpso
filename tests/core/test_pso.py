import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal, assert_array_equal
from src.core.pso import PSO
from src.core.particle import Particle # For type checking if needed, and understanding

# Standard PSO parameters for testing
N_PARTICLES = 10
DIM_PSO = 2
POS_MIN_PSO = np.array([-5, -5])
POS_MAX_PSO = np.array([5, 5])
BOUNDS_PSO = (POS_MIN_PSO, POS_MAX_PSO)
VEL_MIN_PSO = -1.0
VEL_MAX_PSO = 1.0
W_PSO = 0.5
C1_PSO = 1.5
C2_PSO = 1.5

# A simple objective function for testing (Sphere function: sum(x_i^2))
mock_objective_func_pso = lambda x: np.sum(x**2)

@pytest.fixture
def pso_instance():
    """Returns a PSO instance for testing."""
    return PSO(N_PARTICLES, DIM_PSO, BOUNDS_PSO, VEL_MIN_PSO, VEL_MAX_PSO, W_PSO, C1_PSO, C2_PSO)

# Initialization Tests
def test_pso_initialization_particle_creation(pso_instance):
    pso = pso_instance
    assert len(pso.particles) == N_PARTICLES
    assert all(isinstance(p, Particle) for p in pso.particles)
    for p in pso.particles:
        assert p.position.shape == (DIM_PSO,)
        assert np.all(p.position >= POS_MIN_PSO) and np.all(p.position <= POS_MAX_PSO)

def test_pso_initialization_gbest_cost(pso_instance):
    pso = pso_instance
    assert pso.gbest_cost == float('inf')
    assert pso.gbest is None # gbest position should be None initially

def test_pso_initialization_invalid_bounds_value_error():
    invalid_bounds_min_greater_max = (np.array([5, 5]), np.array([-5, -5]))
    with pytest.raises(ValueError, match="The minimum position cannot be greater than or equal to the maximum position"):
        PSO(N_PARTICLES, DIM_PSO, invalid_bounds_min_greater_max, VEL_MIN_PSO, VEL_MAX_PSO, W_PSO, C1_PSO, C2_PSO)

    invalid_bounds_min_equal_max = (np.array([1, 1]), np.array([1, 1]))
    with pytest.raises(ValueError, match="The minimum position cannot be greater than or equal to the maximum position"):
        PSO(N_PARTICLES, DIM_PSO, invalid_bounds_min_equal_max, VEL_MIN_PSO, VEL_MAX_PSO, W_PSO, C1_PSO, C2_PSO)

def test_pso_initialization_parameter_storage(pso_instance):
    pso = pso_instance
    assert pso.number_of_particles == N_PARTICLES
    assert pso.dim == DIM_PSO
    assert_array_equal(pso.pos_min, POS_MIN_PSO)
    assert_array_equal(pso.pos_max, POS_MAX_PSO)
    assert pso.vel_min == VEL_MIN_PSO
    assert pso.vel_max == VEL_MAX_PSO
    assert pso.w_init == W_PSO # w_init stores the initial inertia weight
    assert pso.c1 == C1_PSO
    assert pso.c2 == C2_PSO


# initialize method tests
def test_pso_initialize_method_updates_gbest(pso_instance):
    pso = pso_instance
    # Ensure particles have non-inf pbest_cost after their own evaluation
    # The Particle.evaluate method is called within pso.initialize
    
    # Set a seed for reproducibility of particle positions
    np.random.seed(42)
    pso_seeded = PSO(N_PARTICLES, DIM_PSO, BOUNDS_PSO, VEL_MIN_PSO, VEL_MAX_PSO, W_PSO, C1_PSO, C2_PSO)

    pso_seeded.initialize(mock_objective_func_pso)
    
    assert pso_seeded.gbest is not None
    assert pso_seeded.gbest_cost != float('inf')
    
    # Verify gbest_cost is indeed the minimum of particle best costs
    min_particle_cost = float('inf')
    best_particle_pos = None
    for p_idx, p in enumerate(pso_seeded.particles):
        # Particle.evaluate is called in pso.initialize, which updates p.pbest_cost
        # and p.pbest_position.
        # pso.initialize then updates gbest based on these initial evaluations.
        if p.pbest_cost < min_particle_cost:
            min_particle_cost = p.pbest_cost
            best_particle_pos = p.pbest_position
            
    assert_almost_equal(pso_seeded.gbest_cost, min_particle_cost)
    assert_array_almost_equal(pso_seeded.gbest, best_particle_pos)

# optimize method tests
NUM_ITERATIONS_TEST = 2

def test_pso_optimize_method_returns_correct_types(pso_instance):
    pso = pso_instance
    cost, best_pos, history = pso.optimize(mock_objective_func_pso, NUM_ITERATIONS_TEST)
    
    assert isinstance(cost, float)
    assert isinstance(best_pos, np.ndarray)
    assert isinstance(history, list)
    assert len(history) == NUM_ITERATIONS_TEST
    assert best_pos.shape == (DIM_PSO,)

def test_pso_optimize_method_cost_improvement_or_equal(pso_instance):
    # With a simple function like sphere, cost should generally improve or stay same.
    pso = pso_instance
    pso.initialize(mock_objective_func_pso)
    initial_gbest_cost = pso.gbest_cost
    
    final_cost, _, _ = pso.optimize(mock_objective_func_pso, NUM_ITERATIONS_TEST)
    
    assert final_cost <= initial_gbest_cost
    if initial_gbest_cost != 0: # If not already at global minimum
         # With sphere and non-zero initial, expect improvement usually
         # but for very few iterations, it's not guaranteed if particles move away initially.
         # The test should be robust to stochastic nature.
         # A more robust check: final_cost should be <= any cost in history.
         if history := pso.cost_history: # If history is not empty
            assert final_cost <= history[0]


def test_pso_optimize_cost_history_values(pso_instance):
    pso = pso_instance
    _, _, history = pso.optimize(mock_objective_func_pso, NUM_ITERATIONS_TEST)
    
    assert len(history) == NUM_ITERATIONS_TEST
    if NUM_ITERATIONS_TEST > 1:
        # Costs in history should be non-increasing for gbest
        for i in range(len(history) - 1):
            assert history[i+1] <= history[i]
    
    # The final gbest_cost of the pso object should match the last item in history
    assert_almost_equal(pso.gbest_cost, history[-1])

def test_pso_w_values_schedule_in_optimize(pso_instance):
    # This test is a bit white-box, checking if w is updated.
    # We can mock move_particles or check its effect if w changes.
    # A simpler check: w_values are generated.
    pso = pso_instance
    
    # To check if w_values are used, we'd need to inspect Particle.update_velocity_and_position
    # or mock it. This might be too detailed for a basic unit test of PSO itself.
    # We trust that pso.move_particles passes the w_values[it] correctly.
    # The generation of w_values itself:
    w_min = 0.4 # As hardcoded in pso.py
    expected_w_values = np.linspace(W_PSO, w_min, NUM_ITERATIONS_TEST)
    
    # To actually test this, we'd need to capture the 'w' passed to move_particles
    # One way: monkeypatch move_particles
    actual_w_values_passed = []
    original_move_particles = pso.move_particles
    def mocked_move_particles(w, objective_function):
        actual_w_values_passed.append(w)
        return original_move_particles(w, objective_function)
    
    pso.move_particles = mocked_move_particles
    pso.optimize(mock_objective_func_pso, NUM_ITERATIONS_TEST)
    
    assert_array_almost_equal(np.array(actual_w_values_passed), expected_w_values)

# Test with a function where the global minimum is known (e.g., [0,0] for sphere)
def test_pso_optimize_sphere_convergence_direction():
    np.random.seed(0) # For reproducibility
    # Use more particles/iterations to increase chance of getting close
    pso = PSO(number_of_particles=20, dim=2, bounds=(np.array([-1,-1]), np.array([1,1])),
              vel_min=-0.1, vel_max=0.1, w=0.7, c1=1.5, c2=1.5)
    
    initial_cost = pso.gbest_cost # Should be inf
    pso.initialize(mock_objective_func_pso)
    cost_after_init = pso.gbest_cost
    assert cost_after_init < initial_cost # Should have found some gbest

    # Run optimization for a few iterations
    num_iter = 10
    final_cost, final_pos, _ = pso.optimize(mock_objective_func_pso, num_iter)

    assert final_cost <= cost_after_init
    # For sphere function, global minimum cost is 0 at [0,0]
    # We expect final_cost to be closer to 0 than cost_after_init
    # And final_pos to be closer to [0,0]
    
    # Example: check if magnitude of final_pos is smaller than initial gbest position
    if pso.gbest is not None and not np.all(pso.particles[0].pbest_position == 0): # if not already at optimum
        initial_gbest_magnitude = np.linalg.norm(pso.particles[0].pbest_position) # using one particle's pbest after init as a proxy
        final_pos_magnitude = np.linalg.norm(final_pos)
        # This is not a strict guarantee for PSO but a general expectation for simple functions
        # A better check is that final_cost < cost_after_init if not already 0.
        if cost_after_init > 1e-8 : # if not already effectively zero
             assert final_cost < cost_after_init
    
    assert final_cost >= 0 # Cost should be non-negative for sphere
    assert_array_almost_equal(final_pos, np.array([0.0, 0.0]), decimal=1) # Check if it's close to [0,0]
                                                                      # Decimal might need adjustment
                                                                      # For robust check, might need more iterations or specific seed
                                                                      # Or check against the best particle's cost in the swarm
    
    # Ensure the gbest in the object is updated
    assert_array_almost_equal(pso.gbest, final_pos)
    assert_almost_equal(pso.gbest_cost, final_cost)

# Test that particles actually move and evaluate
def test_pso_particles_behavior_during_optimize():
    pso = PSO(N_PARTICLES, DIM_PSO, BOUNDS_PSO, VEL_MIN_PSO, VEL_MAX_PSO, W_PSO, C1_PSO, C2_PSO)
    
    # Store initial particle positions and pbest_costs
    initial_positions = [p.position.copy() for p in pso.particles]
    initial_pbest_costs = [p.pbest_cost for p in pso.particles] # all should be inf
    
    pso.optimize(mock_objective_func_pso, 1) # Run for 1 iteration
    
    # Check if positions have changed (most likely, unless all random numbers are zero, etc.)
    positions_changed = any(not np.array_equal(initial_positions[i], p.position) for i, p in enumerate(pso.particles))
    assert positions_changed
    
    # Check if pbest_costs have been updated from inf
    pbests_updated = all(p.pbest_cost != float('inf') for p in pso.particles)
    assert pbests_updated
    
    # Check if gbest_cost has been updated from inf
    assert pso.gbest_cost != float('inf')
