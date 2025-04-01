from fun.test_function import my_pso_optimize, pyswarms_pso_optimize, run_optimization
import pandas as pd #type: ignore
import itertools
import multiprocessing as mp
from functools import partial
from tqdm import tqdm  #type: ignore

if __name__ == '__main__':  
    n_particles_list = [10, 20, 50]
    iters_list = [100, 200, 500, 1000]
    w_list = [0.5, 0.7, 0.9]
    c1_list = [1.0, 1.5, 2.0]
    c2_list = [0.5, 1.0, 1.5]
    dim_list = [2, 5, 10]
    
    # Generate all combinations of parameters
    param_combinations = list(itertools.product(n_particles_list, iters_list, w_list, c1_list, c2_list, dim_list))
    functions = ['ackley', 'rosenbrock', 'rastrigin']
    
    # Create all combinations of parameters and functions
    all_combinations = [(params, func) for params in param_combinations for func in functions]
    num_repetitions = 5
    results = []
    
    # Set the processing pool
    num_cores = mp.cpu_count() - 1  # Leave one core free for system tasks
    print(f"Running on {num_cores} cores")
    
    # Calculate total iterations for the progress bar
    total_iterations = len(all_combinations) * num_repetitions
    
    # Create a master progress bar for all repetitions
    with tqdm(total=total_iterations, desc="Total Progress") as pbar:
        # Execute the optimizations in parallel
        for rep in range(num_repetitions):
            # Create a partial function to pass the repetition number
            run_opt_with_rep = partial(run_optimization, rep_num=rep)
            
            # Execute the optimizations in parallel
            with mp.Pool(processes=num_cores) as pool:
                # Use imap instead of map to process results as they come
                for result in pool.imap_unordered(run_opt_with_rep, all_combinations):
                    if result is not None:
                        results.extend(result)
                    pbar.update(1)  # Update progress bar for each completed task
    
    # Save final results in a CSV file
    df = pd.DataFrame(results)
    df.to_csv('../analysis/results.csv', index=False)
    print("Results saved to ../analysis/results.csv")