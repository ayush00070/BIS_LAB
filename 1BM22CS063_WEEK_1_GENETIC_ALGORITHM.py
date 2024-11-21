import numpy as np
import random

# Define the mathematical function to optimize
def objective_function(x):
    return x * np.sin(x)

# Initialize the population with random values within the bounds
def initialize_population(pop_size, x_min, x_max):
    return np.random.uniform(x_min, x_max, pop_size)

# Evaluate the fitness of the population
def evaluate_fitness(pop):
    return np.array([objective_function(x) for x in pop])

def select_individual(pop, fitness):
    # Shift fitness to be non-negative by adding the absolute minimum value if necessary
    min_fitness = np.min(fitness)
    if min_fitness < 0:
        fitness = fitness - min_fitness
    
    # Calculate probabilities
    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness
    return pop[np.random.choice(len(pop), p=probabilities)]
101

# Perform crossover between two individuals
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        alpha = random.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child
    return parent1

# Apply mutation to an individual
def mutate(individual, mutation_rate, x_min, x_max):
    if random.random() < mutation_rate:
        mutation_value = np.random.uniform(-0.5, 0.5)
        individual += mutation_value
        individual = np.clip(individual, x_min, x_max)  # Ensure within bounds
    return individual

# Main Genetic Algorithm loop
def genetic_algorithm():
    # Get user input
    pop_size = int(input("Enter the population size: "))
    gens = int(input("Enter the number of generations: "))
    mutation_rate = float(input("Enter the mutation rate (0 to 1): "))
    crossover_rate = float(input("Enter the crossover rate (0 to 1): "))
    x_min = float(input("Enter the minimum bound for x: "))
    x_max = float(input("Enter the maximum bound for x: "))
    
    population = initialize_population(pop_size, x_min, x_max)
    best_solution = None
    best_fitness = float('-inf')
    
    for generation in range(gens):
        fitness = evaluate_fitness(population)
        
        # Track the best solution
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx]
        
        # Generate a new population
        new_population = []
        for _ in range(pop_size):
            parent1 = select_individual(population, fitness)
            parent2 = select_individual(population, fitness)
            offspring = crossover(parent1, parent2, crossover_rate)
            offspring = mutate(offspring, mutation_rate, x_min, x_max)
            new_population.append(offspring)
        
        population = np.array(new_population)
        print(f"Generation {generation+1}: Best fitness = {best_fitness:.4f}")
    
    print(f"\nBest solution found: x = {best_solution}, f(x) = {best_fitness:.4f}")

# Run the Genetic Algorithm
genetic_algorithm()
