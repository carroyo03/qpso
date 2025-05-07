try:
    from pso.pso import run_optimization
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    sys.path.insert(0, project_root)
    from pso.pso import run_optimization

import pandas as pd #type: ignore
import itertools
import multiprocessing as mp
from functools import partial
from tqdm import tqdm  #type: ignore
import os  # Add import for directory creation

if __name__ == '__main__':  
    n_particles_list = [10, 20, 50]
    iters_list = [100, 200, 500, 1000]
    w_list = [0.5, 0.7, 0.9]
    c1_list = [1.0, 1.5, 2.0]
    c2_list = [0.5, 1.0, 1.5]
    dim_list = [2, 5, 10]
    
    # Generate all combinations of parameters
    functions = ['ackley', 'rosenbrock', 'rastrigin']
    all_combinations = list(itertools.product(n_particles_list, iters_list, w_list, c1_list, c2_list, dim_list, functions))

    
    # Create all combinations of parameters and functions
    num_repetitions:int = 5
    results:list = []
    
    # Set the processing pool
    num_cores = mp.cpu_count() - 1  # Leave one core free for system tasks
    print(f"Running on {num_cores} cores")
    
    # Create a single, simple progress bar
    total_tasks = len(all_combinations) * num_repetitions
    with tqdm(
        total=total_tasks, 
        desc="Optimization",
        bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt}',
        dynamic_ncols=True,
        miniters=max(1, total_tasks // 100)  # Update the bar less frequently
    ) as progress_bar:
        # Execute the optimizations in parallel
        for rep in range(num_repetitions):
            run_opt_with_rep = partial(run_optimization, rep_num=rep)
            
            # Properly initialize and clean up the pool
            pool = mp.Pool(processes=num_cores)
            try:
                # Process results and update progress bar
                for result in pool.imap_unordered(run_opt_with_rep, all_combinations):
                    if result is not None:
                        results.extend(result)
                    progress_bar.update(1)
            finally:
                # Ensure proper cleanup
                pool.close()
                pool.join()
    
    # Create the analysis directory if it doesn't exist
    analysis_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Save final results in a CSV file
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(analysis_dir, 'results.csv'), index=False)
    print(f"Results saved to {os.path.join(analysis_dir, 'results.csv')}")