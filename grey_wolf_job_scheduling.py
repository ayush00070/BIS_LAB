import numpy as np
import random

# Objective Function: Calculate Makespan (total processing time across machines)
def calculate_makespan(schedule, processing_times):
    machine_times = [0] * len(schedule)
    for machine, jobs in enumerate(schedule):
        machine_times[machine] = sum(processing_times[job] for job in jobs)
    return max(machine_times)

# Generate a random initial schedule
def initialize_population(num_wolves, num_jobs, num_machines):
    population = []
    for _ in range(num_wolves):
        wolf = [[] for _ in range(num_machines)]
        for job in range(num_jobs):
            wolf[random.randint(0, num_machines - 1)].append(job)
        population.append(wolf)
    return population

# Update positions based on alpha, beta, and delta wolves
def update_positions(population, alpha, beta, delta, a, num_machines, num_jobs):
    new_population = []
    for wolf in population:
        new_wolf = [[] for _ in range(num_machines)]
        for job in range(num_jobs):
            # Calculate the position updates relative to alpha, beta, and delta
            A1, C1 = 2 * a * random.random() - a, 2 * random.random()
            D_alpha = abs(C1 * alpha[job % num_machines] - wolf[job % num_machines])
            X1 = alpha[job % num_machines] - A1 * D_alpha

            A2, C2 = 2 * a * random.random() - a, 2 * random.random()
            D_beta = abs(C2 * beta[job % num_machines] - wolf[job % num_machines])
            X2 = beta[job % num_machines] - A2 * D_beta

            A3, C3 = 2 * a * random.random() - a, 2 * random.random()
            D_delta = abs(C3 * delta[job % num_machines] - wolf[job % num_machines])
            X3 = delta[job % num_machines] - A3 * D_delta

            # Calculate the new position
            new_position = (X1 + X2 + X3) / 3
            selected_machine = int(new_position) % num_machines
            new_wolf[selected_machine].append(job)

        new_population.append(new_wolf)
    return new_population

# Grey Wolf Optimizer
def grey_wolf_optimizer(processing_times, num_machines=3, num_wolves=10, num_iterations=100):
    num_jobs = len(processing_times)

    # Step 1: Initialize population
    population = initialize_population(num_wolves, num_jobs, num_machines)
    alpha, beta, delta = None, None, None
    alpha_fitness, beta_fitness, delta_fitness = float('inf'), float('inf'), float('inf')

    for iteration in range(num_iterations):
        # Step 2: Evaluate fitness of wolves
        fitnesses = [calculate_makespan(wolf, processing_times) for wolf in population]

        # Update alpha, beta, and delta wolves
        for i, fitness in enumerate(fitnesses):
            if fitness < alpha_fitness:
                alpha, beta, delta = population[i], alpha, beta
                alpha_fitness, beta_fitness, delta_fitness = fitness, alpha_fitness, beta_fitness
            elif fitness < beta_fitness:
                beta, delta = population[i], beta
                beta_fitness, delta_fitness = fitness, beta_fitness
            elif fitness < delta_fitness:
                delta = population[i]
                delta_fitness = fitness

        # Step 3: Update positions
        a = 2 - 2 * iteration / num_iterations  # Decreasing coefficient
        population = update_positions(population, alpha, beta, delta, a, num_machines, num_jobs)

        print(f"Iteration {iteration + 1}: Alpha Fitness (Best Makespan) = {alpha_fitness}")

    # Step 4: Return the best solution
    return alpha, alpha_fitness

# Example: Job processing times
processing_times = [5, 10, 15, 20, 25, 30, 35, 40]

# Run GWO for job scheduling
best_schedule, best_makespan = grey_wolf_optimizer(processing_times, num_machines=3, num_wolves=5, num_iterations=50)

# Print results
print("\nBest Schedule:", best_schedule)
print("Best Makespan (Total Processing Time):", best_makespan)
