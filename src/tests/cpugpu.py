import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from ..numpy_ver.irwa import irwa_solver as irwa_cpu_solver
from ..numpy_ver.adal import adal_solver as adal_cpu_solver
from ..torch_ver.irwa import irwa_solver as irwa_gpu_solver
from ..torch_ver.adal import adal_solver as adal_gpu_solver
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
    
    for trial in range(num_trials):
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
        
        # Run CPU solvers
        start = time.time()
        _, k_irwa, n_cg_irwa, _ = irwa_cpu_solver(H_np, g_np, A_eq_np, b_eq_np, A_ineq_np, b_ineq_np)
        metrics['IRWA_CPU']['time'].append(time.time() - start)
        metrics['IRWA_CPU']['cg_steps'].append(n_cg_irwa)
        
        start = time.time()
        _, k_adal, n_cg_adal, _ = adal_cpu_solver(H_np, g_np, A_eq_np, b_eq_np, A_ineq_np, b_ineq_np)
        metrics['ADAL_CPU']['time'].append(time.time() - start)
        metrics['ADAL_CPU']['cg_steps'].append(n_cg_adal)
        
        # Run GPU solvers
        start = time.time()
        _, k_irwa, n_cg_irwa, _ = irwa_gpu_solver(H_torch, g_torch, A_eq_torch, b_eq_torch, A_ineq_torch, b_ineq_torch)
        metrics['IRWA_GPU']['time'].append(time.time() - start)
        metrics['IRWA_GPU']['cg_steps'].append(n_cg_irwa)
        
        start = time.time()
        _, k_adal, n_cg_adal, _ = adal_gpu_solver(H_torch, g_torch, A_eq_torch, b_eq_torch, A_ineq_torch, b_ineq_torch)
        metrics['ADAL_GPU']['time'].append(time.time() - start)
        metrics['ADAL_GPU']['cg_steps'].append(n_cg_adal)
    
    return metrics


def analyze_metrics(metrics):
    """Calculate statistics from collected metrics"""
    results = {}
    for solver in metrics:
        times = metrics[solver]['time']
        cg_steps = metrics[solver]['cg_steps']
        
        results[solver] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'avg_cg_steps': np.mean(cg_steps),
            'min_cg_steps': min(cg_steps),
            'max_cg_steps': max(cg_steps),
            'std_cg_steps': np.std(cg_steps)
        }
    return results


def plot_results(results, problem_size):
    """Visualize the performance comparison"""
    solvers = list(results.keys())
    
    # Time comparison
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    avg_times = [results[s]['avg_time'] for s in solvers]
    std_times = [results[s]['std_time'] for s in solvers]
    plt.bar(solvers, avg_times, yerr=std_times, capsize=5)
    plt.ylabel('Average Time (s)')
    plt.title(f'Time Comparison\nProblem Size: {problem_size}')
    
    # CG Steps comparison
    plt.subplot(1, 2, 2)
    avg_cg = [results[s]['avg_cg_steps'] for s in solvers]
    min_cg = [results[s]['min_cg_steps'] for s in solvers]
    max_cg = [results[s]['max_cg_steps'] for s in solvers]
    
    for i, solver in enumerate(solvers):
        plt.errorbar(i, avg_cg[i], 
                    yerr=[[avg_cg[i] - min_cg[i]], [max_cg[i] - avg_cg[i]]],
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
    num_trials = 10
    
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