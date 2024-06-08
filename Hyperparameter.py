import numpy as np
from sklearn.cluster import KMeans

# Random key encoding
def random_key_encoding(population_size, hyperparameters):
    population = []
    for _ in range(population_size):
        individual = []
        for param, space in hyperparameters.items():
            if isinstance(space, np.ndarray):
                # For integer and continuous spaces
                individual.append(np.random.choice(space))
            else:
                # For categorical spaces
                individual.append(np.random.choice(range(len(space))))
        population.append(individual)
    return np.array(population)


def improved_de_mutation(population, F=0.8):
    # Apply k-means clustering to divide the population
    kmeans = KMeans(n_clusters=max(2, int(len(population)/5))).fit(population)  # Ensure at least 2 clusters
    labels = kmeans.labels_

    # Find the cluster with the minimum mean objective function value
    unique_labels = np.unique(labels)
    cluster_scores = [np.mean([objective_function(population[i]) for i in range(len(population)) if labels[i] == label]) for label in unique_labels]
    win_region = unique_labels[np.argmin(cluster_scores)]
    win_indices = np.where(labels == win_region)[0]

    # Perform the mutation based on the winning region
    mutated_population = []
    for i in range(len(population)):
        if len(win_indices) >= 2:
            r1, r2 = population[np.random.choice(win_indices, 2, replace=False)]
        elif len(win_indices) == 1:
            # Only one individual in the winning cluster, use it as r1 and select another from the entire population
            r1 = population[win_indices[0]]
            r2 = population[np.random.choice(np.delete(np.arange(len(population)), win_indices[0]))]
        else:
            # No individuals in the winning cluster, fall back to random selection from the entire population
            r1, r2 = population[np.random.choice(len(population), 2, replace=False)]

        win = population[np.random.choice(win_indices)] if len(win_indices) > 0 else population[np.random.randint(len(population))]
        mutant = win + F * (r1 - r2)
        mutated_population.append(mutant)

    return np.array(mutated_population)

def objective_function(individual):
    return np.random.rand()

def main_optimization_loop(population_size, generations, hyperparameters, F=0.8):
    # Initialize population using random key encoding
    population = random_key_encoding(population_size, hyperparameters)

    # Main loop for generations
    for generation in range(generations):
        # Perform mutation using the improved DE mutation strategy
        mutated_population = improved_de_mutation(population, F=F)

        # Evaluate the original and mutated populations
        fitness_original = np.array([objective_function(individual) for individual in population])
        fitness_mutated = np.array([objective_function(individual) for individual in mutated_population])

        # Selection: Create a new population
        new_population = []
        for i in range(population_size):
            if fitness_mutated[i] < fitness_original[i]:  # Assuming minimization
                new_population.append(mutated_population[i])
            else:
                new_population.append(population[i])

        population = np.array(new_population)

        # Here, you could add logging of best performance, etc.
        print(f"Generation {generation}: Best Fitness = {np.min(fitness_original)}")

    # Return the final population and its fitness
    return population, fitness_original


