import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from ..algorithms.irwa import irwa_solver
from ..algorithms.adal import adal_solver
from ..functions import penalized_quadratic_objective, quadratic_objective
from .data_gen import generate_optimization_data


def run_experiment(n, m1, m2, num_trials=5, device='cuda'):
    """
    Run multiple trials for fixed problem dimensions and collect performance metrics
    
    Returns:
    dict: Dictionary containing lists of metrics for each solver
    """
    metrics = {
        'IRWA_CPU': defaultdict(list),
        'ADAL_CPU': defaultdict(list),
        'IRWA_GPU': defaultdict(list),
        'ADAL_GPU': defaultdict(list)
    }
    
    for trial in range(1, num_trials+1):
        # Generate problem data with different random seeds
        data = generate_optimization_data(n=n, m1=m1, m2=m2, numpy_output=True, torch_output=True, seed=trial)
        
        # CPU (NumPy) data
        H_np, g_np = data['numpy']['H'], data['numpy']['g']
        A_eq_np, b_eq_np = data['numpy']['A_eq'], data['numpy']['b_eq']
        A_ineq_np, b_ineq_np = data['numpy']['A_ineq'], data['numpy']['b_ineq']
        
        # GPU (PyTorch) data
        H_torch = data['torch']['H'].to(device)
        g_torch = data['torch']['g'].to(device)
        A_eq_torch = data['torch']['A_eq'].to(device)
        b_eq_torch = data['torch']['b_eq'].to(device)
        A_ineq_torch = data['torch']['A_ineq'].to(device)
        b_ineq_torch = data['torch']['b_ineq'].to(device)
        
        # List of solvers to benchmark
        solvers = [
            # CPU solvers
            {'name': 'IRWA_CPU', 'args': (H_np, g_np, A_eq_np, b_eq_np, A_ineq_np, b_ineq_np), 'func': irwa_solver},
            {'name': 'ADAL_CPU', 'args': (H_np, g_np, A_eq_np, b_eq_np, A_ineq_np, b_ineq_np), 'func': adal_solver},
            # GPU solvers  
            {'name': 'IRWA_GPU', 'args': (H_torch, g_torch, A_eq_torch, b_eq_torch, A_ineq_torch, b_ineq_torch), 'func': irwa_solver},
            {'name': 'ADAL_GPU', 'args': (H_torch, g_torch, A_eq_torch, b_eq_torch, A_ineq_torch, b_ineq_torch), 'func': adal_solver}
        ]

        # Print table header
        print(f"\n===== Iteration {trial} Metrics =====")
        print(f"{'Solver':<15} {'Time(s)':<15} {'CG Steps':<15} {'CG Time(s)':<15}")

        # Benchmark each solver
        verbose = False
        for solver in solvers:
            # Time the solver execution
            start = time.time()
            _, _, n_cg, cg_time = solver['func'](*solver['args'], verbose=verbose)
            total_time = time.time() - start
            
            # Store metrics
            metrics[solver['name']]['time'].append(total_time)
            metrics[solver['name']]['cg_steps'].append(n_cg) 
            metrics[solver['name']]['cg_time'].append(cg_time)
            
            # Print results
            print(f"{solver['name']:<15} {total_time:<15.4f} {n_cg:<15} {cg_time:<15.4f}")

        # Add spacing after each benchmark iteration    
        print()

    return metrics


def analyze_metrics(metrics):
    """Calculate statistics from collected metrics"""
    results = {}
    for solver in metrics:
        times = metrics[solver]['time']
        cg_steps = metrics[solver]['cg_steps']
        cg_times = metrics[solver]['cg_time']  # Add this line
        
        results[solver] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'avg_cg_steps': np.mean(cg_steps),
            'min_cg_steps': min(cg_steps),
            'max_cg_steps': max(cg_steps),
            'std_cg_steps': np.std(cg_steps),
            'avg_cg_time': np.mean(cg_times),
            'std_cg_time': np.std(cg_times)
        }
        
    return results


def plot_results(results, problem_size):
    """Visualize performance: CG Time as part of Total Time, and CG Step counts"""
    solvers = list(results.keys())

    # Extract data
    avg_total_times = [results[s]['avg_time'] for s in solvers]
    avg_cg_times = [results[s]['avg_cg_time'] for s in solvers]
    non_cg_times = [total - cg for total, cg in zip(avg_total_times, avg_cg_times)]

    avg_cg_steps = [results[s]['avg_cg_steps'] for s in solvers]
    min_cg_steps = [results[s]['min_cg_steps'] for s in solvers]
    max_cg_steps = [results[s]['max_cg_steps'] for s in solvers]

    # Set up the figure
    plt.figure(figsize=(12, 5))
    # --- Subplot 1: Total Time Breakdown ---
    plt.subplot(1, 2, 1)
    plt.bar(solvers, avg_cg_times, label='CG Time', color='skyblue')
    plt.bar(solvers, non_cg_times, bottom=avg_cg_times, label='Other Time', color='orange')
    plt.ylabel('Average Total Time (s)')
    plt.title(f'Total Time Breakdown\nProblem Size: {problem_size}')
    plt.legend()

    # --- Subplot 2: CG Steps ---
    plt.subplot(1, 2, 2)
    for i, solver in enumerate(solvers):
        plt.errorbar(i, avg_cg_steps[i],
                     yerr=[[avg_cg_steps[i] - min_cg_steps[i]], [max_cg_steps[i] - avg_cg_steps[i]]],
                     fmt='o', capsize=5, label=solver)
    plt.xticks(range(len(solvers)), solvers)
    plt.ylabel('CG Steps')
    plt.title(f'CG Steps Comparison\nProblem Size: {problem_size}')
    plt.legend()
    
    plt.tight_layout()
    # plt.savefig(f'results_{problem_size}.png')
    plt.show()
    

# Example usage:
if __name__ == "__main__":
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(device)
    print(f"Using device: {device}")
    
    # Fixed problem dimensions with multiple trials
    scale = 5
    n = 1000 * scale
    m1 = 300 * scale
    m2 = 300 * scale
    num_trials = 5
    
    print(f"\nRunning {num_trials} trials for problem size: {n}x{m1}x{m2}")
    
    # Run experiments
    metrics = run_experiment(n, m1, m2, num_trials, device)
    
    # Analyze results
    results = analyze_metrics(metrics)
    
    # Print summary
    print("\nPerformance Summary:")
    print(f"{'Solver':<10} {'Avg Time':<10} {'Time Std':<10} {'Avg CG':<10} {'Min CG':<10} {'Max CG':<10} {'CG Std':<10}")
    for solver in results:
        r = results[solver]
        print(f"{solver:<10} {r['avg_time']:<10.4f} {r['std_time']:<10.4f} "
              f"{r['avg_cg_steps']:<10.1f} {r['min_cg_steps']:<10} {r['max_cg_steps']:<10} {r['std_cg_steps']:<10.1f}")
    
    # Visualize results
    plot_results(results, f"{n}x{m1}x{m2}")