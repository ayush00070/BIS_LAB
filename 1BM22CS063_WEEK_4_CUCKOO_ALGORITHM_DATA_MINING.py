import numpy as np

# Objective function to optimize (replace this with your data mining objective)
def objective_function(x):
    """
    Example function to optimize. Replace with a function relevant to the data mining problem,
    such as a machine learning model accuracy or feature selection metric.
    """
    return -np.sum(x ** 2)  # Maximize the negative squared sum (simple optimization example)

# Generate initial nests randomly within bounds
def initialize_nests(num_nests, dim, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, size=(num_nests, dim))

# Levy flight for generating new solutions
def levy_flight(Lambda, dim):
    sigma_u = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
               (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    sigma_v = 1.0
    u = np.random.normal(0, sigma_u, size=dim)
    v = np.random.normal(0, sigma_v, size=dim)
    step = u / np.abs(v) ** (1 / Lambda)
    return step

# Cuckoo Search Algorithm
def cuckoo_search():
    # User inputs
    num_nests = int(input("Enter the number of nests (population size): "))
    dim = int(input("Enter the problem dimension: "))
    lower_bound = float(input("Enter the lower bound for search space: "))
    upper_bound = float(input("Enter the upper bound for search space: "))
    max_iterations = int(input("Enter the number of iterations: "))
    pa = float(input("Enter the fraction of nests to abandon (0 to 1): "))
    Lambda = float(input("Enter the step-size parameter for Levy flights (1.5 is common): "))

    # Initialize nests and fitness
    nests = initialize_nests(num_nests, dim, lower_bound, upper_bound)
    fitness = np.array([objective_function(nest) for nest in nests])
    best_nest = nests[np.argmax(fitness)]
    best_fitness = np.max(fitness)

    # Main loop
    for iteration in range(max_iterations):
        # Generate new solutions via Levy flight
        for i in range(num_nests):
            step = levy_flight(Lambda, dim)
            new_nest = nests[i] + step * (nests[i] - best_nest)
            new_nest = np.clip(new_nest, lower_bound, upper_bound)  # Keep within bounds
            new_fitness = objective_function(new_nest)

            # Update nest if new solution is better
            if new_fitness > fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness

        # Sort nests by fitness and abandon a fraction of the worst nests
        num_abandon = int(pa * num_nests)
        if num_abandon > 0:
            worst_indices = np.argsort(fitness)[:num_abandon]
            for idx in worst_indices:
                nests[idx] = np.random.uniform(lower_bound, upper_bound, size=dim)
                fitness[idx] = objective_function(nests[idx])

        # Update global best
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            best_nest = nests[current_best_idx]
            best_fitness = fitness[current_best_idx]

        print(f"Iteration {iteration + 1}/{max_iterations}: Best Fitness = {best_fitness:.4f}")

    print("\nBest solution found:")
    print(f"Position: {best_nest}")
    print(f"Fitness: {best_fitness:.4f}")

# Run the Cuckoo Search Algorithm
cuckoo_search()
