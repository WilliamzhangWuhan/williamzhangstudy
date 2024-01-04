import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math

def generate_cities(num_cities):
    x_range=(0, 10)
    y_range=(0, 10)
    x_coordinates = np.random.uniform(low=x_range[0], high=x_range[1], size=num_cities)
    y_coordinates = np.random.uniform(low=y_range[0], high=y_range[1], size=num_cities)
    return np.column_stack((x_coordinates, y_coordinates))  

def calculate_distance(city1, city2):
    return np.linalg.norm(city1 - city2)

def initialize_pheromones(num_cities):
    return np.ones((num_cities, num_cities))

def update_pheromones(pheromones, delta_pheromones):
    decay_factor = 0.5
    pheromones = pheromones * (1 - decay_factor) + delta_pheromones
    return pheromones

def select_next_city(current_city, pheromones, distances, visited, alpha=1, beta=2):
    num_cities = len(pheromones)
    remaining_cities = [city for city in range(num_cities) if not visited[city]]

    probabilities = []
    for city in remaining_cities:
        pheromone = pheromones[current_city, city]
        distance = distances[current_city, city]
        probability = (pheromone ** alpha) * ((1 / distance) ** beta)
        probabilities.append(probability)

    probabilities = probabilities / np.sum(probabilities)
    next_city = np.random.choice(remaining_cities, p=probabilities)
    return next_city

def ant_tour(pheromones, distances, alpha=1, beta=2):
    num_cities = len(pheromones)
    start_city = np.random.randint(num_cities)
    current_city = start_city
    tour = [current_city]
    visited = [False] * num_cities
    visited[current_city] = True

    while len(tour) < num_cities:
        next_city = select_next_city(current_city, pheromones, distances, visited, alpha, beta)
        tour.append(next_city)
        visited[next_city] = True
        current_city = next_city

    return tour

def calculate_delta_pheromones(tours, distances):
    num_cities = len(distances)
    delta_pheromones = np.zeros((num_cities, num_cities))

    for tour in tours:
        for i in range(num_cities - 1):
            city1, city2 = tour[i], tour[i + 1]
            delta_pheromones[city1, city2] += 1 / distances[city1, city2]
            delta_pheromones[city2, city1] += 1 / distances[city1, city2]

    return delta_pheromones

def plot_ants(cities, ant_tours, iteration):
    plt.figure(figsize=(8, 8))
    plt.scatter(cities[:, 0], cities[:, 1], color='red', s=100, marker='o', zorder=2)
    
    for i, city in enumerate(cities):
        plt.text(city[0], city[1], f"{i}", fontsize=10, ha='center', va='center', color='white')

    for tour in ant_tours:
        tour = np.array(tour)
        tour_cities = cities[tour]
        tour_line = mlines.Line2D(tour_cities[:, 0], tour_cities[:, 1], linewidth=2, linestyle='dashed', color='red')
        plt.gca().add_line(tour_line)

    plt.title(f"Iteration {iteration}")
    plt.show()

def ant_colony_visualization(num_cities, num_ants, num_iterations):
    cities = generate_cities(num_cities)
    distances = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distances[i, j] = calculate_distance(cities[i], cities[j])
            distances[j, i] = distances[i, j]

    pheromones = initialize_pheromones(num_cities)
    best_tour = None
    best_distance = np.inf

    for iteration in range(num_iterations):
        ant_tours = [ant_tour(pheromones, distances) for _ in range(num_ants)]

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Best Distance: {best_distance}")
            plot_ants(cities, ant_tours, iteration)

        if iteration == num_iterations - 1:
            print("Best Tour:", ant_tours[0])
            plot_ants(cities, [ant_tours[0]], iteration)

        delta_pheromones = calculate_delta_pheromones(ant_tours, distances)
        pheromones = update_pheromones(pheromones, delta_pheromones)

        for tour in ant_tours:
            tour_distance = sum(distances[tour[i], tour[i + 1]] for i in range(num_cities - 1))
            if tour_distance < best_distance:
                best_distance = tour_distance
                best_tour = tour

    return best_tour, best_distance

# 示例运行
num_cities = 15
num_ants = 3
num_iterations = 100
best_tour, best_distance = ant_colony_visualization(num_cities, num_ants, num_iterations)

print("Best Tour:", best_tour)
print("Best Distance:", best_distance)
