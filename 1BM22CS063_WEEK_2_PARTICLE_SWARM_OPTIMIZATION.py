import numpy as np

# Define the mathematical function to optimize
def objective_function(x):
    # Example function: Sphere function. Replace with any user-defined function.
    return sum([xi**2 for xi in x])

# Initialize particles with random positions and velocities
def initialize_particles(num_particles, dimensions, x_min, x_max, v_min, v_max):
    positions = np.random.uniform(x_min, x_max, (num_particles, dimensions))
    velocities = np.random.uniform(v_min, v_max, (num_particles, dimensions))
    return positions, velocities

# PSO Algorithm
def particle_swarm_optimization():
    # User inputs
    dimensions = int(input("Enter the number of dimensions: "))
    num_particles = int(input("Enter the number of particles: "))
    iterations = int(input("Enter the number of iterations: "))
    w = float(input("Enter the inertia weight (e.g., 0.5 to 1.0): "))
    c1 = float(input("Enter the cognitive coefficient (e.g., 1.5 to 2.0): "))
    c2 = float(input("Enter the social coefficient (e.g., 1.5 to 2.0): "))
    x_min = float(input("Enter the minimum bound for x: "))
    x_max = float(input("Enter the maximum bound for x: "))
    v_min = float(input("Enter the minimum velocity: "))
    v_max = float(input("Enter the maximum velocity: "))
    
    # Initialize particles
    positions, velocities = initialize_particles(num_particles, dimensions, x_min, x_max, v_min, v_max)
    personal_best_positions = np.copy(positions)
    personal_best_scores = np.array([objective_function(pos) for pos in positions])
    
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)
    
    # Iterative optimization
    for iteration in range(iterations):
        for i in range(num_particles):
            # Update velocities
            r1, r2 = np.random.random(dimensions), np.random.random(dimensions)
            cognitive_velocity = c1 * r1 * (personal_best_positions[i] - positions[i])
            social_velocity = c2 * r2 * (global_best_position - positions[i])
            velocities[i] = w * velocities[i] + cognitive_velocity + social_velocity
            
            # Clamp velocity to max/min limits
            velocities[i] = np.clip(velocities[i], v_min, v_max)
            
            # Update positions
            positions[i] = positions[i] + velocities[i]
            
            # Clamp position to bounds
            positions[i] = np.clip(positions[i], x_min, x_max)
            
            # Evaluate fitness
            fitness = objective_function(positions[i])
            
            # Update personal best
            if fitness < personal_best_scores[i]:
                personal_best_positions[i] = positions[i]
                personal_best_scores[i] = fitness
                
        # Update global best
        best_particle_index = np.argmin(personal_best_scores)
        if personal_best_scores[best_particle_index] < global_best_score:
            global_best_score = personal_best_scores[best_particle_index]
            global_best_position = personal_best_positions[best_particle_index]
        
        # Output progress
        print(f"Iteration {iteration+1}/{iterations}: Best Score = {global_best_score:.4f}")
    
    print(f"\nBest solution found: Position = {global_best_position}, Score = {global_best_score:.4f}")

# Run the PSO Algorithm
particle_swarm_optimization()
