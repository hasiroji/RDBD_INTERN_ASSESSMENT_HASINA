import random
import math
import matplotlib.pyplot as plt

# Courses
courses = [
    "ICT-3101",
    "ICT-3103",
    "ICT-3105",
    "ICT-3107",
    "ICT-3109"
]

# Available Time Slots
time_slots = [
    "Sun 9:00",
    "Sun 11:00",
    "Mon 9:00",
    "Mon 11:00",
    "Tue 9:00"
]

# Initial Random Routine
routine = {}

for course in courses:
    routine[course] = random.choice(time_slots)

# Fitness Function
# Lower conflict = Better Routine

def calculate_conflicts(schedule):

    slots = list(schedule.values())

    conflicts = len(slots) - len(set(slots))

    return conflicts

# Simulated Annealing Parameters
temperature = 100
cooling_rate = 0.95

current_solution = routine.copy()
current_cost = calculate_conflicts(current_solution)

best_solution = current_solution.copy()
best_cost = current_cost

history = []

while temperature > 1:

    new_solution = current_solution.copy()

    course = random.choice(courses)

    new_solution[course] = random.choice(time_slots)

    new_cost = calculate_conflicts(new_solution)

    delta = new_cost - current_cost

    if delta < 0:
        current_solution = new_solution
        current_cost = new_cost

    else:
        probability = math.exp(-delta / temperature)

        if random.random() < probability:
            current_solution = new_solution
            current_cost = new_cost

    if current_cost < best_cost:
        best_solution = current_solution.copy()
        best_cost = current_cost

    history.append(best_cost)

    temperature *= cooling_rate

# Display Result
print("Optimized University Routine\n")

for course, slot in best_solution.items():
    print(f"{course:10} --> {slot}")

print("\nTotal Conflicts:", best_cost)

# Plot
plt.figure(figsize=(8,5))
plt.plot(history)
plt.title("Simulated Annealing Routine Optimization")
plt.xlabel("Iteration")
plt.ylabel("Conflicts")
plt.grid(True)
plt.show()