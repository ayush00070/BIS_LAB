import random
import numpy as np

# Define the problem: DNA Sequence Alignment using an optimization function
def calculate_alignment_score(seq1, seq2, gap_penalty=-2, mismatch_penalty=-1, match_reward=2):
    """
    Calculate the alignment score between two DNA sequences.
    The score is calculated based on matches, mismatches, and gaps.
    """
    score = 0
    for s1, s2 in zip(seq1, seq2):
        if s1 == '-' or s2 == '-':
            score += gap_penalty  # Penalize gaps
        elif s1 == s2:
            score += match_reward  # Reward for matches
        else:
            score += mismatch_penalty  # Penalize mismatches
    return score

# Initialize Population: Generate a population of random DNA sequences
def initialize_population(pop_size, seq_len):
    """
    Initialize a population of random DNA sequences.
    Each sequence consists of 'A', 'T', 'C', 'G' and may include gaps ('-').
    """
    population = []
    for _ in range(pop_size):
        seq = ''.join(random.choice('ATCG-') for _ in range(seq_len))
        population.append(seq)
    return population

# Evaluate Fitness: Evaluate the fitness of each sequence based on the alignment score
def evaluate_population(population, target_sequence):
    """
    Evaluate the fitness of each sequence by aligning it with a target DNA sequence.
    """
    fitness_scores = []
    for individual in population:
        score = calculate_alignment_score(individual, target_sequence)
        fitness_scores.append(score)
    return fitness_scores

# Selection: Select individuals based on their fitness for reproduction
def select_parents(population, fitness_scores):
    """
    Select individuals based on their fitness using a roulette-wheel selection mechanism.
    """
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    parents = random.choices(population, weights=probabilities, k=2)
    return parents

# Crossover: Perform crossover between selected sequences
def crossover(parent1, parent2, crossover_rate=0.7):
    """
    Perform a single-point crossover between two parents to produce offspring.
    """
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        return offspring1, offspring2
    else:
        return parent1, parent2

# Mutation: Apply mutation to introduce variability
def mutate(sequence, mutation_rate=0.1):
    """
    Apply mutation by randomly changing nucleotides or inserting gaps.
    """
    sequence = list(sequence)
    for i in range(len(sequence)):
        if random.random() < mutation_rate:
            sequence[i] = random.choice('ATCG-')
    return ''.join(sequence)

# Gene Expression: Translate genetic sequence into a functional solution
def gene_expression(sequence, target_sequence):
    """
    Translate the genetic sequence (alignment) into a fitness score.
    The gene expression is the alignment score between the sequence and the target.
    """
    return calculate_alignment_score(sequence, target_sequence)

# Gene Expression Algorithm (GEA)
def gene_expression_algorithm(target_sequence, pop_size=100, seq_len=20, num_generations=50, mutation_rate=0.1, crossover_rate=0.7):
    """
    Implement Gene Expression Algorithm for DNA sequence alignment.
    """
    # Step 1: Initialize the population
    population = initialize_population(pop_size, seq_len)
    
    # Step 2: Run the genetic algorithm
    best_score = -float('inf')
    best_sequence = None

    for generation in range(num_generations):
        # Step 3: Evaluate the population
        fitness_scores = evaluate_population(population, target_sequence)

        # Track the best solution
        max_score = max(fitness_scores)
        if max_score > best_score:
            best_score = max_score
            best_sequence = population[fitness_scores.index(max_score)]
        
        # Step 4: Selection and Crossover
        next_generation = []
        for _ in range(pop_size // 2):  # Generate next generation
            parent1, parent2 = select_parents(population, fitness_scores)
            offspring1, offspring2 = crossover(parent1, parent2, crossover_rate)
            next_generation.append(mutate(offspring1, mutation_rate))
            next_generation.append(mutate(offspring2, mutation_rate))

        # Step 5: Update the population for the next generation
        population = next_generation

        # Print progress every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation + 1}: Best Score = {best_score}")

    # Step 6: Output the best solution
    print(f"\nOptimization Complete! Best Alignment Score = {best_score}")
    print(f"Best Alignment: {best_sequence}")
    return best_sequence, best_score

# Run the Gene Expression Algorithm for DNA sequence alignment
target_sequence = "ATGCCGTAAGCCTGAACTG"
optimized_sequence, best_alignment_score = gene_expression_algorithm(target_sequence, num_generations=100)

# Final Result
print("\nOptimized DNA Sequence Alignment:")
print(optimized_sequence)
print(f"Best Alignment Score: {best_alignment_score}")
