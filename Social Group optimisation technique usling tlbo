import numpy as np
import matplotlib.pyplot as plt

# Function definitions
def sphere_function(x):
    return np.sum(x**2)

def schwefel_12_function(x):
    return np.sum([np.sum(x[:i+1])**2 for i in range(len(x))])

def schwefel_222_function(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def rastrigin_function(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def sgo(N, D, g, LL, UL, c, objective_function):
    population = LL + (UL - LL) * np.random.rand(N, D)
    fitness = np.array([objective_function(x) for x in population])
    best_fitness_history = []

    for _ in range(g):
        gbest_idx = np.argmin(fitness)
        gbestg = population[gbest_idx].copy()
        for i in range(N):
            r1, r2 = np.random.rand(), np.random.rand()
            Xr = population[np.random.randint(N)]
            for j in range(D):
                Xnew = population[i, j] + r1 * (population[i, j] - Xr[j]) + r2 * (gbestg[j] - population[i, j])
                Xnew = np.clip(Xnew, LL[j], UL[j])
                population[i, j] = Xnew
            
            new_fitness = objective_function(population[i])
            if new_fitness < fitness[i]:
                fitness[i] = new_fitness

        for i in range(N):
            r1, r2 = np.random.rand(), np.random.rand()
            Xr_idx = np.random.randint(N)
            Xr = population[Xr_idx]

            if fitness[i] < fitness[Xr_idx]:
                for j in range(D):
                    Xnew = population[i, j] + r1 * (population[i, j] - Xr[j]) + r2 * (gbestg[j] - population[i, j])
                    Xnew = np.clip(Xnew, LL[j], UL[j])
                    population[i, j] = Xnew
            
                    new_fitness = objective_function(population[i])
                    if new_fitness < fitness[i]:
                        fitness[i] = new_fitness

            else:
                for j in range(D):
                    Xnew = population[i, j] + r1 * (Xr[j] - population[i, j]) + r2 * (gbestg[j] - population[i, j])
                    Xnew = np.clip(Xnew, LL[j], UL[j])
                    population[i, j] = Xnew

                    new_fitness = objective_function(population[i])
                    if new_fitness < fitness[i]:
                        fitness[i] = new_fitness

        best_fitness_history.append(fitness[gbest_idx])

    return best_fitness_history

def sgo_elite(N, D, g, LL, UL, c, objective_function):
    population = LL + (UL - LL) * np.random.rand(N, D)
    fitness = np.array([objective_function(x) for x in population])
    best_fitness_history = []

    gbest_idx = np.argmin(fitness)
    gbestg = population[gbest_idx].copy()

    for _ in range(g):
        # Improving phase
        for i in range(N):
            r1, r2 = np.random.rand(), np.random.rand()
            Xr_idx = np.random.randint(N)
            Xr = population[Xr_idx]

            if fitness[i] < fitness[Xr_idx]:
                for j in range(D):
                    Xnew = population[i, j] + r1 * (population[i, j] - Xr[j])
                    Xnew = np.clip(Xnew, LL[j], UL[j])
                    population[i, j] = Xnew

                new_fitness = objective_function(population[i])
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

        # Acquisition phase
            if fitness[i] < fitness[Xr_idx]:
                for j in range(D):
                    Xnew = population[i, j] + r1 * (population[i, j] - Xr[j]) + r2 * (gbestg[j] - population[i, j])
                    Xnew = np.clip(Xnew, LL[j], UL[j])
                    population[i, j] = Xnew
            
                    new_fitness = objective_function(population[i])
                    if new_fitness < fitness[i]:
                        fitness[i] = new_fitness

            else:
                for j in range(D):
                    Xnew = population[i, j] + r1 * (Xr[j] - population[i, j]) + r2 * (gbestg[j] - population[i, j])
                    Xnew = np.clip(Xnew, LL[j], UL[j])
                    population[i, j] = Xnew

                    new_fitness = objective_function(population[i])
                    if new_fitness < fitness[i]:
                        fitness[i] = new_fitness

        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < fitness[gbest_idx]:
            gbest_idx = current_best_idx
            gbestg = population[gbest_idx].copy()
        
        # Find the index of the worst fitness in the population
        worst_idx = np.argmax(fitness)
        # Replace the worst fitness with the current global best fitness
        population[worst_idx] = gbestg
        # Update the fitness of the replaced individual
        fitness[worst_idx] = fitness[gbest_idx]

        best_fitness_history.append(fitness[gbest_idx])

    return best_fitness_history

# Additional function definitions
def step_function(x):
    return np.sum(np.floor(x))

def zakharov_function(x):
    return np.sum(x*2) + (np.sum(0.5 * np.arange(1, len(x) + 1) * x))2 + (np.sum(0.5 * np.arange(1, len(x) + 1) * x))*4

def powell_function(x):
    x1 = x[0]
    x2 = x[1]
    return (x1 + 10 * x2)*2 + 5 * (x1 - x[2])2 + (x1 - 2 * x[3])4 + 10 * (x2 - x[4])*4

def booth_function(x):
    return (x[0] + 2 * x[1] - 7)*2 + (2 * x[0] + x[1] - 5)*2

def griewank_function(x):
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

def elliptic_function(x):
    return np.sum((10*6) * (np.arange(len(x)) / (len(x) - 1)) * x*2)

# Additional parameters
N, g = 50, 100
functions = [sphere_function, schwefel_12_function, schwefel_222_function, rastrigin_function,
             step_function, zakharov_function, powell_function, booth_function,
             griewank_function, elliptic_function]
function_names = ['Sphere', 'Schwefel 12', 'Schwefel 222', 'Rastrigin',
                  'Step', 'Zakharov', 'Powell', 'Booth', 'Griewank', 'Elliptic']
D_values = [2, 10, 10, 10, 2, 10, 5, 2, 10, 10]

# Lower and upper bounds for each function
function_bounds = {
    'Sphere': (-5.12, 5.12),
    'Schwefel 12': (-500, 500),
    'Schwefel 222': (-10, 10),
    'Rastrigin': (-5.12, 5.12),
    'Step': (-100, 100),
    'Zakharov': (-5, 10),
    'Powell': (-5, 5),
    'Booth': (-10, 10),
    'Griewank': (-600, 600),
    'Elliptic': (-100, 100)
}

# Plotting
plt.figure(figsize=(20, 10))

for i, (func, name, D) in enumerate(zip(functions, function_names, D_values)):
    LL, UL = function_bounds[name]
    history_normal = sgo(N, D, g, LL * np.ones(D), UL * np.ones(D), 0.5, func)
    history_elite = sgo_elite(N, D, g, LL * np.ones(D), UL * np.ones(D), 0.5, func)
    
    plt.subplot(2, 5, i+1)
    plt.plot(history_normal, label='SGO')
    plt.plot(history_elite, label='SGO with Elite')
    plt.title(name)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.legend()

plt.tight_layout()
plt.show()
