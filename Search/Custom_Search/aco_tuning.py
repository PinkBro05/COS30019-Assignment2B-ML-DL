import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import components from aco_search.py
from aco_routing.aco import ACO
from aco_routing.network import Network

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "data_reader"))

from parser import parse_graph_file

class ACOTuner:
    def __init__(self, graph_file, num_trials=5, output_dir="tuning_results"):
        """
        Initialize the ACO parameter tuner.
        
        Args:
            graph_file (str): Path to the graph file
            num_trials (int): Number of trials to run for each parameter set
            output_dir (str): Directory to save the tuning results
        """
        self.graph_file = graph_file
        self.num_trials = num_trials
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Parse graph and set up network
        self.nodes, self.edges, self.origin, self.destinations = parse_graph_file(graph_file)
        
        # Create the graph
        self.G = Network()
        self.G.graph = {node: [] for node in self.nodes}
        self.G.pos = self.nodes
        
        # Add edges
        for (start, end), weight in self.edges.items():
            self.G.add_edge(start, end, cost=float(weight))
        
        # Get baseline adaptive parameters
        self.baseline_params = 51+1, 100, 51, 0.5, 1, 2
        
        # Unpack baseline parameters
        self.baseline_ant_max_steps, self.baseline_iterations, self.baseline_num_ants, \
        self.baseline_evaporation_rate, self.baseline_alpha, self.baseline_beta = self.baseline_params
        
        print(f"Loaded graph with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        print(f"Origin: {self.origin}, Destinations: {self.destinations}")
        print(f"Baseline parameters: {self.baseline_params}")
    
    def run_aco_with_params(self, params):
        """
        Run ACO algorithm with the given parameters.
        
        Args:
            params (dict): Parameters for the ACO algorithm
            
        Returns:
            tuple: (path_cost, path_found, execution_time)
        """
        ant_max_steps = params['ant_max_steps']
        iterations = params['iterations']
        num_ants = params['num_ants']
        evaporation_rate = params['evaporation_rate']
        alpha = params['alpha']
        beta = params['beta']
        
        aco = ACO(self.G, 
            ant_max_steps=ant_max_steps,
            num_iterations=iterations, 
            evaporation_rate=evaporation_rate, 
            alpha=alpha, 
            beta=beta, 
            mode=2, # 0: any destination, 1: all destinations, 2: TSP mode
            log_step=None, # Setting log, Int or None
            visualize=None,  # Enable visualization
            visualization_step=10  # Update visualization every 10 iterations
        )
        
        start_time = time.time()
        path, cost = aco.find_shortest_path(
            source=self.origin,
            destination=self.destinations,
            num_ants=num_ants
        )
        execution_time = time.time() - start_time
        
        path_found = path is not None and len(path) > 0
        
        return cost if path_found else float('inf'), path_found, execution_time
    
    def evaluate_params(self, params):
        """
        Evaluate a set of parameters by running multiple trials.
        
        Args:
            params (dict): Parameters for the ACO algorithm
            
        Returns:
            dict: Evaluation results
        """
        costs = []
        times = []
        success_count = 0
        
        for _ in range(self.num_trials):
            cost, found, exec_time = self.run_aco_with_params(params)
            if found:
                costs.append(cost)
                success_count += 1
            times.append(exec_time)
        
        success_rate = success_count / self.num_trials
        avg_cost = np.mean(costs) if costs else float('inf')
        avg_time = np.mean(times)
        
        return {
            'params': params,
            'avg_cost': avg_cost,
            'success_rate': success_rate,
            'avg_time': avg_time,
            'min_cost': min(costs) if costs else float('inf'),
            'costs': costs,
            'times': times
        }
    
    def run_grid_search(self):
        """
        Perform grid search to find optimal parameters.
        
        Returns:
            list: Evaluation results for all parameter combinations
        """
        print("Starting grid search...")
        
        # Define parameter grid relative to baseline
        param_grid = {
            'ant_max_steps': [int(self.baseline_ant_max_steps * m) for m in [0.7, 1.0, 1.3]],
            'iterations': [int(self.baseline_iterations * m) for m in [0.7, 1.0, 1.3]],
            'num_ants': [int(self.baseline_num_ants * m) for m in [0.7, 1.0, 1.3]],
            'evaporation_rate': [self.baseline_evaporation_rate * m for m in [0.5, 1.0, 1.5]],
            'alpha': [self.baseline_alpha * m for m in [0.5, 1.0, 1.5]],
            'beta': [self.baseline_beta * m for m in [0.5, 1.0, 1.5]]
        }
        
        # Ensure minimum values
        param_grid['ant_max_steps'] = [max(10, v) for v in param_grid['ant_max_steps']]
        param_grid['iterations'] = [max(5, v) for v in param_grid['iterations']]
        param_grid['num_ants'] = [max(5, v) for v in param_grid['num_ants']]
        param_grid['evaporation_rate'] = [max(0.01, min(0.9, v)) for v in param_grid['evaporation_rate']]
        param_grid['alpha'] = [max(0.01, min(1.0, v)) for v in param_grid['alpha']]
        param_grid['beta'] = [max(0.01, min(2.0, v)) for v in param_grid['beta']]
        
        # Generate all combinations of parameters
        param_keys = param_grid.keys()
        param_values = param_grid.values()
        param_combinations = [dict(zip(param_keys, combination)) for combination in product(*param_values)]
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        results = []
        for i, params in enumerate(param_combinations):
            print(f"Evaluating parameter set {i+1}/{len(param_combinations)}: {params}")
            result = self.evaluate_params(params)
            results.append(result)
            print(f"Result: Avg Cost = {result['avg_cost']}, Success Rate = {result['success_rate']}, Avg Time = {result['avg_time']}")
        
        # Sort results by average cost and success rate
        results.sort(key=lambda x: (1 - x['success_rate'], x['avg_cost']))
        
        # Save results
        self.save_results(results, "grid_search_results.txt")
        
        return results
    
    def run_random_search(self, num_samples=20):
        """
        Perform random search to find optimal parameters.
        
        Args:
            num_samples (int): Number of random parameter combinations to try
            
        Returns:
            list: Evaluation results for all parameter combinations
        """
        print("Starting random search...")
        
        # Define parameter ranges relative to baseline
        param_ranges = {
            'ant_max_steps': (int(self.baseline_ant_max_steps * 0.5), int(self.baseline_ant_max_steps * 2.0)),
            'iterations': (int(self.baseline_iterations * 0.5), int(self.baseline_iterations * 2.0)),
            'num_ants': (int(self.baseline_num_ants * 0.5), int(self.baseline_num_ants * 2.0)),
            'evaporation_rate': (max(0.01, self.baseline_evaporation_rate * 0.3), min(0.9, self.baseline_evaporation_rate * 3.0)),
            'alpha': (max(0.01, self.baseline_alpha * 0.3), min(1.0, self.baseline_alpha * 3.0)),
            'beta': (max(0.01, self.baseline_beta * 0.3), min(2.0, self.baseline_beta * 3.0))
        }
        
        # Generate random parameter combinations
        param_combinations = []
        for _ in range(num_samples):
            params = {
                'ant_max_steps': max(10, random.randint(*param_ranges['ant_max_steps'])),
                'iterations': max(5, random.randint(*param_ranges['iterations'])),
                'num_ants': max(5, random.randint(*param_ranges['num_ants'])),
                'evaporation_rate': round(random.uniform(*param_ranges['evaporation_rate']), 2),
                'alpha': round(random.uniform(*param_ranges['alpha']), 2),
                'beta': round(random.uniform(*param_ranges['beta']), 2)
            }
            param_combinations.append(params)
        
        # Add baseline parameters to the list
        baseline_params = {
            'ant_max_steps': self.baseline_ant_max_steps,
            'iterations': self.baseline_iterations,
            'num_ants': self.baseline_num_ants,
            'evaporation_rate': self.baseline_evaporation_rate,
            'alpha': self.baseline_alpha,
            'beta': self.baseline_beta
        }
        param_combinations.append(baseline_params)
        
        print(f"Testing {len(param_combinations)} random parameter combinations...")
        
        results = []
        for i, params in enumerate(param_combinations):
            print(f"Evaluating parameter set {i+1}/{len(param_combinations)}: {params}")
            result = self.evaluate_params(params)
            results.append(result)
            print(f"Result: Avg Cost = {result['avg_cost']}, Success Rate = {result['success_rate']}, Avg Time = {result['avg_time']}")
        
        # Sort results by average cost and success rate
        results.sort(key=lambda x: (1 - x['success_rate'], x['avg_cost']))
        
        # Save results
        self.save_results(results, "random_search_results.txt")
        
        return results
    
    def run_bayesian_optimization(self, n_iterations=25):
        """
        Perform Bayesian optimization to find optimal parameters.
        
        Args:
            n_iterations (int): Number of iterations for Bayesian optimization
                
        Returns:
            list: Evaluation results for all parameter combinations
        """
        print("Starting Bayesian optimization...")
        
        # Define the search space
        space = [
            Integer(int(self.baseline_ant_max_steps * 0.5), int(self.baseline_ant_max_steps * 2.0), name='ant_max_steps'),
            Integer(int(self.baseline_iterations * 0.5), int(self.baseline_iterations * 2.0), name='iterations'),
            Integer(int(self.baseline_num_ants * 0.5), int(self.baseline_num_ants * 2.0), name='num_ants'),
            Real(max(0.01, self.baseline_evaporation_rate * 0.3), min(0.9, self.baseline_evaporation_rate * 3.0), 
                name='evaporation_rate'),
            Real(max(0.01, self.baseline_alpha * 0.3), min(1.0, self.baseline_alpha * 3.0), name='alpha'),
            Real(max(0.01, self.baseline_beta * 0.3), min(2.0, self.baseline_beta * 3.0), name='beta')
        ]
        # Store all evaluated parameters and results for later retrieval
        evaluated_params = []
        evaluated_results = []
        
        # Define the objective function
        @use_named_args(space)
        def objective(**params):
            # Ensure minimum values
            params['ant_max_steps'] = max(10, params['ant_max_steps'])
            params['iterations'] = max(5, params['iterations'])
            params['num_ants'] = max(5, params['num_ants'])
            
            # Round floating point parameters for better readability
            params['evaporation_rate'] = round(params['evaporation_rate'], 2)
            params['alpha'] = round(params['alpha'], 2)
            params['beta'] = round(params['beta'], 2)
            
            print(f"Evaluating parameters: {params}")
            result = self.evaluate_params(params)
            
            # Store parameters and result
            evaluated_params.append(params.copy())
            evaluated_results.append(result)
            
            # Our objective is to minimize a combination of cost and failure rate
            if result['success_rate'] == 0:
                return 1e6
            
            objective_value = result['avg_cost'] + (1 - result['success_rate']) * 1000
            print(f"Result: Avg Cost = {result['avg_cost']}, Success Rate = {result['success_rate']}, " 
                f"Objective Value = {objective_value}")
                
            return objective_value
        
        # Run Bayesian optimization
        res_gp = gp_minimize(
            objective,
            space,
            n_calls=n_iterations,
            random_state=42,
            verbose=True,
            n_random_starts=min(5, n_iterations)
        )
        
        # Get the best parameters
        best_params = {
            'ant_max_steps': max(10, res_gp.x[0]),
            'iterations': max(5, res_gp.x[1]),
            'num_ants': max(5, res_gp.x[2]),
            'evaporation_rate': round(res_gp.x[3], 2),
            'alpha': round(res_gp.x[4], 2),
            'beta': round(res_gp.x[5], 2)
        }
        
        # Also evaluate the baseline parameters for comparison
        baseline_params = {
            'ant_max_steps': self.baseline_ant_max_steps,
            'iterations': self.baseline_iterations,
            'num_ants': self.baseline_num_ants,
            'evaporation_rate': self.baseline_evaporation_rate,
            'alpha': self.baseline_alpha,
            'beta': self.baseline_beta
        }
        
        print(f"Evaluating baseline parameters: {baseline_params}")
        baseline_result = self.evaluate_params(baseline_params)
        
        # Collect all evaluated parameter sets from optimization history
        all_params = []
        for i, x in enumerate(res_gp.x_iters):
            params = {
                'ant_max_steps': max(10, x[0]),
                'iterations': max(5, x[1]),
                'num_ants': max(5, x[2]),
                'evaporation_rate': round(x[3], 2),
                'alpha': round(x[4], 2),
                'beta': round(x[5], 2)
            }
            all_params.append(params)
        
        # Evaluate all parameter sets including baseline
        results = []
        for params in all_params:
            result = self.evaluate_params(params)
            results.append(result)
        
        # Add baseline result
        results.append(baseline_result)
        
        # Sort results by success rate and average cost
        results.sort(key=lambda x: (1 - x['success_rate'], x['avg_cost']))
        
        # Save results
        self.save_results(results, "bayesian_opt_results.txt")
        
        return results
    
    def save_results(self, results, filename):
        """
        Save tuning results to a file.
        
        Args:
            results (list): List of evaluation results
            filename (str): Name of the file to save results to
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("ACO Parameter Tuning Results\n")
            f.write(f"Graph: {self.graph_file}\n")
            f.write(f"Origin: {self.origin}, Destinations: {self.destinations}\n")
            f.write(f"Baseline parameters: {self.baseline_params}\n\n")
            
            f.write("Top 10 Parameter Sets:\n")
            for i, result in enumerate(results[:10]):
                params = result['params']
                f.write(f"Rank {i+1}:\n")
                f.write(f"  Parameters: {params}\n")
                f.write(f"  Avg Cost: {result['avg_cost']}\n")
                f.write(f"  Success Rate: {result['success_rate']}\n")
                f.write(f"  Avg Time: {result['avg_time']:.4f} seconds\n")
                f.write(f"  Min Cost: {result['min_cost']}\n\n")
        
        print(f"Results saved to {filepath}")
    
    def visualize_results(self, results, result_type="grid"):
        """
        Visualize tuning results.
        
        Args:
            results (list): List of evaluation results
            result_type (str): Type of search result ('grid' or 'random')
        """
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f"ACO Parameter Tuning Results ({result_type.capitalize()} Search)", fontsize=16)
        
        # Plot success rate vs. average cost
        ax1 = fig.add_subplot(2, 2, 1)
        success_rates = [r['success_rate'] for r in results]
        avg_costs = [r['avg_cost'] if r['avg_cost'] != float('inf') else np.nan for r in results]
        
        # Filter out inf values
        valid_indices = [i for i, cost in enumerate(avg_costs) if not np.isnan(cost)]
        filtered_success_rates = [success_rates[i] for i in valid_indices]
        filtered_costs = [avg_costs[i] for i in valid_indices]
        
        ax1.scatter(filtered_success_rates, filtered_costs, alpha=0.7)
        ax1.set_xlabel('Success Rate')
        ax1.set_ylabel('Average Cost')
        ax1.set_title('Success Rate vs. Average Cost')
        ax1.grid(True)
        
        # Plot average execution time
        ax2 = fig.add_subplot(2, 2, 2)
        avg_times = [r['avg_time'] for r in results]
        indices = list(range(len(results)))
        ax2.bar(indices[:15], avg_times[:15])
        ax2.set_xlabel('Parameter Set Index')
        ax2.set_ylabel('Average Execution Time (s)')
        ax2.set_title('Average Execution Time (Top 15 Parameter Sets)')
        ax2.grid(True)
        
        # Plot parameter values for top N results
        top_n = min(10, len(results))
        param_names = ['ant_max_steps', 'iterations', 'num_ants', 
                        'evaporation_rate', 'alpha', 'beta']
        
        ax3 = fig.add_subplot(2, 2, 3)
        top_params = [results[i]['params'] for i in range(top_n)]
        
        # Plot relative parameter values compared to baseline
        relative_values = np.zeros((len(param_names), top_n))
        baseline_values = [self.baseline_ant_max_steps, self.baseline_iterations, 
                           self.baseline_num_ants, self.baseline_evaporation_rate, 
                           self.baseline_alpha, self.baseline_beta]
        
        for i, param in enumerate(param_names):
            for j, p in enumerate(top_params):
                relative_values[i, j] = p[param] / baseline_values[i] if baseline_values[i] != 0 else 0
        
        im = ax3.imshow(relative_values, cmap='viridis')
        ax3.set_yticks(range(len(param_names)))
        ax3.set_yticklabels(param_names)
        ax3.set_xticks(range(top_n))
        ax3.set_xticklabels([f"Rank {i+1}" for i in range(top_n)], rotation=45)
        ax3.set_title('Parameter Values Relative to Baseline')
        plt.colorbar(im, ax=ax3)
        
        # Plot top 5 cost distributions
        ax4 = fig.add_subplot(2, 2, 4)
        for i in range(min(5, top_n)):
            result = results[i]
            costs = [c for c in result['costs'] if c != float('inf')]
            if costs:
                ax4.boxplot(costs, positions=[i+1])
        
        ax4.set_xlabel('Rank')
        ax4.set_ylabel('Cost')
        ax4.set_title('Cost Distribution for Top 5 Parameter Sets')
        ax4.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = os.path.join(self.output_dir, f"{result_type}_search_results.png")
        plt.savefig(plot_path)
        print(f"Visualization saved to {plot_path}")
        plt.close()

def main():
    graph_file = f"Data/TSP_Test_case_4.txt"
    # Create ACO tuner
    tuner = ACOTuner(graph_file, num_trials=5)
    
    # Run Bayesian optimization
    # results = tuner.run_bayesian_optimization(n_iterations=15)
    # tuner.visualize_results(results, "bayesian")
    
    # Run grid search
    # results = tuner.run_grid_search()
    # tuner.visualize_results(results, "grid")
    
    # Run random search
    results = tuner.run_random_search(num_samples=20)
    tuner.visualize_results(results, "random")
    
    # Print best parameters
    best_params = results[0]
    print("\nBest parameters:")
    print(f"Parameters: {best_params['params']}")
    print(f"Average Cost: {best_params['avg_cost']}")
    print(f"Success Rate: {best_params['success_rate']}")
    print(f"Average Time: {best_params['avg_time']:.4f} seconds")
    
    # Generate Python code for the best parameters
    params = best_params['params']
    print("\nTo use these parameters in aco_search.py, replace the adaptive parameters with:")
    print("# Override adaptive parameters with tuned values")
    print(f"ant_max_steps = {params['ant_max_steps']}")
    print(f"iterations = {params['iterations']}")
    print(f"num_ants = {params['num_ants']}")
    print(f"evaporation_rate = {params['evaporation_rate']}")
    print(f"alpha = {params['alpha']}")
    print(f"beta = {params['beta']}")
        

if __name__ == "__main__":
    main()