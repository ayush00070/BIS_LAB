import numpy as np
import random

# Objective Function: Calculate Makespan
def calculate_makespan(schedule, processing_times):
    machine_times = [0] * len(schedule)
    for machine, jobs in enumerate(schedule):
        machine_times[machine] = sum(processing_times[job] for job in jobs)
    return max(machine_times)

# Generate a random initial schedule
def initialize_population(num_wolves, num_jobs, num_machines):
    population = []
    for _ in range(num_wolves):
        wolf = [random.randint(0, num_machines - 1) for _ in range(num_jobs)]
        population.append(wolf)
    return population

# Decode a wolf into a schedule
def decode_wolf(wolf, num_machines):
    schedule = [[] for _ in range(num_machines)]
    for job, machine in enumerate(wolf):
        schedule[machine].append(job)
    return schedule

# Update wolf positions
def update_positions(population, alpha, beta, delta, a, num_machines):
    new_population = []
    for wolf in population:
        new_wolf = []
        for job in range(len(wolf)):
            A1, A2, A3 = 2 * a * random.random() - a, 2 * a * random.random() - a, 2 * a * random.random() - a
            C1, C2, C3 = 2 * random.random(), 2 * random.random(), 2 * random.random()

            # Calculate distances
            D_alpha = abs(C1 * alpha[job] - wolf[job])
            D_beta = abs(C2 * beta[job] - wolf[job])
            D_delta = abs(C3 * delta[job] - wolf[job])

            # Update positions
            X1 = alpha[job] - A1 * D_alpha
            X2 = beta[job] - A2 * D_beta
            X3 = delta[job] - A3 * D_delta

            new_position = (X1 + X2 + X3) / 3
            new_position = int(round(new_position)) % num_machines  # Ensure valid machine index
            new_wolf.append(new_position)

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
        # Step 2: Evaluate fitness
        fitnesses = [calculate_makespan(decode_wolf(wolf, num_machines), processing_times) for wolf in population]

        # Update alpha, beta, delta
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
        a = 2 - 2 * iteration / num_iterations
        population = update_positions(population, alpha, beta, delta, a, num_machines)

        print(f"Iteration {iteration + 1}: Alpha Fitness (Best Makespan) = {alpha_fitness}")

    # Decode the best solution
    best_schedule = decode_wolf(alpha, num_machines)
    return best_schedule, alpha_fitness

# Example Processing Times
processing_times = [8, 4, 6, 7, 9, 3, 5, 2]

# Run Grey Wolf Optimizer
best_schedule, best_makespan = grey_wolf_optimizer(processing_times, num_machines=3, num_wolves=5, num_iterations=50)

# Display Results
print("\nBest Schedule:", best_schedule)
print("Best Makespan:", best_makespan)
