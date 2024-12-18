import random
import numpy as np

# Distance matrix representing distances between cities
distance_matrix = [
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
]

# Parameters
population_size = 10
mutation_rate = 0.1
crossover_rate = 0.8
num_generations = 100

# Number of cities
num_cities = len(distance_matrix)

# Function to calculate the total distance of a route
def calculate_fitness(route):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    # Add distance to return to the starting city
    total_distance += distance_matrix[route[-1]][route[0]]
    return 1 / total_distance  # Fitness is the inverse of the distance

# Generate a random initial population
def generate_population(size, num_cities):
    population = []
    for _ in range(size):
        individual = list(range(num_cities))
        random.shuffle(individual)
        population.append(individual)
    return population

# Selection (Tournament Selection)
def select_parents(population, fitnesses):
    parents = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), 3)
        winner = max(tournament, key=lambda x: x[1])
        parents.append(winner[0])
    return parents

# Crossover (Order Crossover)
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child = [-1] * len(parent1)
        child[start:end] = parent1[start:end]
        pointer = 0
        for gene in parent2:
            if gene not in child:
                while child[pointer] != -1:
                    pointer += 1
                child[pointer] = gene
        return child
    return parent1

# Mutation (Swap Mutation)
def mutate(individual):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]

# Main Genetic Algorithm
def genetic_algorithm():
    # Step 1: Initialize Population
    population = generate_population(population_size, num_cities)
    best_solution = None
    best_fitness = -float('inf')

    for generation in range(num_generations):
        # Step 2: Evaluate Fitness
        fitnesses = [calculate_fitness(individual) for individual in population]

        # Track the best solution
        max_fitness = max(fitnesses)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_solution = population[fitnesses.index(max_fitness)]

        # Step 3: Selection
        parents = select_parents(population, fitnesses)

        # Step 4: Crossover and Mutation
        next_generation = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % len(parents)]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            mutate(child1)
            mutate(child2)
            next_generation.extend([child1, child2])

        # Update population
        population = next_generation[:population_size]

        # Print the best fitness of each generation
        print(f"Generation {generation + 1}: Best Fitness = {1 / best_fitness}")

    # Return the best solution and its distance
    best_distance = 1 / best_fitness
    return best_solution, best_distance

# Run the Genetic Algorithm
best_route, best_distance = genetic_algorithm()
print("\nBest Route:", best_route)
print("Best Distance:", best_distance)
