import numpy as np
import random

# Calculate the Euclidean distance between two cities
def calculate_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# Generate the distance matrix
def create_distance_matrix(cities):
    num_cities = len(cities)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = calculate_distance(cities[i], cities[j])
    return dist_matrix

# ACO Algorithm for TSP
def ant_colony_optimization():
    # User inputs
    num_cities = int(input("Enter the number of cities: "))
    print("Enter the coordinates of the cities (x, y) one by one:")
    cities = [tuple(map(float, input(f"City {i+1}: ").split())) for i in range(num_cities)]
    
    num_ants = int(input("Enter the number of ants: "))
    iterations = int(input("Enter the number of iterations: "))
    alpha = float(input("Enter the importance of pheromone (alpha): "))
    beta = float(input("Enter the importance of heuristic information (beta): "))
    rho = float(input("Enter the pheromone evaporation rate (rho): "))
    initial_pheromone = float(input("Enter the initial pheromone value: "))
    
    # Create distance matrix
    dist_matrix = create_distance_matrix(cities)
    num_cities = len(cities)
    
    # Initialize pheromone matrix
    pheromones = np.full((num_cities, num_cities), initial_pheromone)
    
    # Initialize best solution
    best_distance = float('inf')
    best_solution = None
    
    # ACO Iterations
    for iteration in range(iterations):
        all_solutions = []
        all_distances = []
        
        # Each ant constructs a solution
        for ant in range(num_ants):
            visited = [False] * num_cities
            current_city = random.randint(0, num_cities - 1)
            tour = [current_city]
            visited[current_city] = True
            
            while len(tour) < num_cities:
                probabilities = []
                for next_city in range(num_cities):
                    if not visited[next_city]:
                        pheromone = pheromones[current_city][next_city] ** alpha
                        heuristic = (1.0 / dist_matrix[current_city][next_city]) ** beta
                        probabilities.append(pheromone * heuristic)
                    else:
                        probabilities.append(0)
                
                probabilities = np.array(probabilities)
                probabilities /= np.sum(probabilities)
                next_city = np.random.choice(range(num_cities), p=probabilities)
                tour.append(next_city)
                visited[next_city] = True
                current_city = next_city
            
            # Complete the tour
            tour.append(tour[0])
            distance = sum(dist_matrix[tour[i]][tour[i+1]] for i in range(len(tour) - 1))
            all_solutions.append(tour)
            all_distances.append(distance)
        
        # Update best solution
        min_distance = min(all_distances)
        if min_distance < best_distance:
            best_distance = min_distance
            best_solution = all_solutions[all_distances.index(min_distance)]
        
        # Update pheromones
        pheromones *= (1 - rho)  # Evaporation
        for tour, distance in zip(all_solutions, all_distances):
            pheromone_deposit = 1.0 / distance
            for i in range(len(tour) - 1):
                pheromones[tour[i]][tour[i+1]] += pheromone_deposit
        
        print(f"Iteration {iteration+1}/{iterations}: Best Distance = {best_distance:.4f}")
    
    # Output the best solution
    print("\nBest solution found:")
    print(" -> ".join(map(str, best_solution)))
    print(f"Total Distance = {best_distance:.4f}")

# Run the ACO Algorithm
ant_colony_optimization()
