import numpy as np

# Problem Definition: Minimize velocity variations across a ring (1D circular grid)
def calculate_fitness_1d(ring):
    """
    Calculate fitness of the ring.
    Fitness is defined as the inverse of the total velocity variation.
    """
    total_variation = sum(abs(ring[i] - ring[(i + 1) % len(ring)]) for i in range(len(ring)))
    return 1 / (1 + total_variation)  # Higher fitness for smoother flows

# Initialize the ring population
def initialize_ring(num_cells, min_velocity, max_velocity):
    """
    Initialize a 1D ring with random velocities.
    """
    return np.random.uniform(min_velocity, max_velocity, size=num_cells)

# Update cell states based on neighbors
def update_ring(ring, influence_factor=0.1):
    """
    Update each cell based on its neighbors using weighted averaging.
    """
    new_ring = np.copy(ring)
    num_cells = len(ring)
    for i in range(num_cells):
        # Neighbors are circular (left and right)
        left_neighbor = ring[(i - 1) % num_cells]
        right_neighbor = ring[(i + 1) % num_cells]
        # Update based on neighbors
        new_ring[i] += influence_factor * ((left_neighbor + right_neighbor) / 2 - ring[i])
    return new_ring

# Parallel Cellular Algorithm for 1D ring
def parallel_cellular_algorithm_1d(num_cells=20, min_velocity=0.0, max_velocity=1.0, num_iterations=50):
    """
    Implement Parallel Cellular Algorithm for a 1D ring optimization.
    """
    # Step 1: Initialize the ring
    ring = initialize_ring(num_cells, min_velocity, max_velocity)
    best_fitness = -np.inf
    best_ring = None

    # Step 2: Iterate and optimize
    for iteration in range(num_iterations):
        # Evaluate fitness
        fitness = calculate_fitness_1d(ring)

        # Update the best solution
        if fitness > best_fitness:
            best_fitness = fitness
            best_ring = np.copy(ring)

        # Update ring states
        ring = update_ring(ring)

        # Print intermediate results
        if iteration % 10 == 0 or iteration == num_iterations - 1:
            print(f"Iteration {iteration + 1}: Fitness = {fitness:.5f}")

    # Step 3: Return the best solution
    print("\nOptimization Complete!")
    return best_ring, best_fitness

# Run the Parallel Cellular Algorithm for 1D ring
optimized_ring, best_fitness = parallel_cellular_algorithm_1d(num_cells=30, num_iterations=50)

# Display Final Results
print("\nBest Fitness:", best_fitness)
print("Optimized Velocities:", np.round(optimized_ring, 2))
