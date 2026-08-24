import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Distance Matrix Between Bus Stops

distance = np.array([
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 17],
    [15, 35, 0, 30, 28],
    [20, 25, 30, 0, 22],
    [25, 17, 28, 22, 0]
])

num_stops = len(distance)

# Route Cost Function
def route_cost(route):
    cost = 0

    for i in range(len(route)-1):
        cost += distance[route[i]][route[i+1]]

    cost += distance[route[-1]][route[0]]

    return cost

# Initial Route
current_route = list(range(num_stops))
random.shuffle(current_route)

current_cost = route_cost(current_route)

best_route = current_route.copy()
best_cost = current_cost

# Simulated Annealing Parameters
temperature = 1000
cooling_rate = 0.995
iterations = 1000

cost_history = []

# Simulated Annealing
for _ in range(iterations):

    new_route = current_route.copy()

    i, j = random.sample(range(num_stops), 2)

    new_route[i], new_route[j] = new_route[j], new_route[i]

    new_cost = route_cost(new_route)

    delta = new_cost - current_cost

    if delta < 0:
        current_route = new_route
        current_cost = new_cost

    else:
        probability = math.exp(-delta / temperature)

        if random.random() < probability:
            current_route = new_route
            current_cost = new_cost

    if current_cost < best_cost:
        best_route = current_route.copy()
        best_cost = current_cost

    cost_history.append(best_cost)

    temperature *= cooling_rate

# Results
print("Optimal Bus Route:")

for stop in best_route:
    print(f"Stop {stop}", end=" -> ")

print(f"Stop {best_route[0]}")

print("\nMinimum Distance:", best_cost)

# Plot Optimization Progress
plt.figure(figsize=(8,5))
plt.plot(cost_history)
plt.title("Simulated Annealing Optimization")
plt.xlabel("Iteration")
plt.ylabel("Route Cost")
plt.grid(True)
plt.show()