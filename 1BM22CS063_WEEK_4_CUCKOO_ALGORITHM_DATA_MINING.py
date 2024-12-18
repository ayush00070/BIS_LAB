import numpy as np
import random

# Objective function: Sum of Squared Deviations (minimize SSD)
def objective_function(data, outliers):
    cleaned_data = [x for i, x in enumerate(data) if i not in outliers]
    mean = np.mean(cleaned_data)
    ssd = sum((x - mean) ** 2 for x in cleaned_data)
    return ssd

# Lévy Flight step
def levy_flight(Lambda):
    u = np.random.normal(0, 1)
    v = np.random.normal(0, 1)
    step = u / abs(v) ** (1 / Lambda)
    return step

# Generate initial population (nests)
def initialize_population(num_nests, data):
    population = []
    for _ in range(num_nests):
        outliers = random.sample(range(len(data)), random.randint(0, len(data) // 3))
        population.append(set(outliers))
    return population

# Replace a fraction of the worst nests with new random positions
def abandon_worst_nests(population, fitnesses, abandon_fraction, data):
    num_to_replace = int(len(population) * abandon_fraction)
    sorted_indices = np.argsort(fitnesses)
    worst_indices = sorted_indices[-num_to_replace:]
    for i in worst_indices:
        new_outliers = random.sample(range(len(data)), random.randint(0, len(data) // 3))
        population[i] = set(new_outliers)

# Cuckoo Search Algorithm
def cuckoo_search(data, num_nests=20, num_iterations=100, discovery_prob=0.25, Lambda=1.5):
    # Step 1: Initialize population
    population = initialize_population(num_nests, data)
    best_solution = None
    best_fitness = float('inf')

    for iteration in range(num_iterations):
        # Step 2: Evaluate fitness of nests
        fitnesses = [objective_function(data, outliers) for outliers in population]

        # Update best solution
        for i, fitness in enumerate(fitnesses):
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = population[i]

        # Step 3: Generate new solutions via Lévy flights
        new_population = []
        for outliers in population:
            new_outliers = outliers.copy()
            for i in range(len(data)):
                if random.random() < discovery_prob:
                    new_outliers.add(i) if i not in new_outliers else new_outliers.remove(i)
                # Apply Lévy flight to introduce randomness
                if random.random() < 0.5:
                    jump = int(levy_flight(Lambda))
                    idx = (i + jump) % len(data)
                    if idx in new_outliers:
                        new_outliers.remove(idx)
                    else:
                        new_outliers.add(idx)
            new_population.append(new_outliers)

        # Step 4: Abandon worst nests
        abandon_worst_nests(new_population, fitnesses, discovery_prob, data)

        # Update the population
        population = new_population

        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

    # Return the best solution and its fitness
    return best_solution, best_fitness

# Example dataset (with some obvious outliers)
data = [10, 12, 11, 13, 50, 10, 12, 11, 400, 15, 14, 11]

# Run Cuckoo Search to manage outliers
best_outliers, best_fitness = cuckoo_search(data, num_nests=10, num_iterations=50)
print("\nBest Outliers (Indices):", best_outliers)
print("Best Fitness (SSD):", best_fitness)

# Cleaned data after removing outliers
cleaned_data = [x for i, x in enumerate(data) if i not in best_outliers]
print("Cleaned Data:", cleaned_data)
