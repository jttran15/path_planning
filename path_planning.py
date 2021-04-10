import matplotlib.pyplot as plt
import numpy as np


N_MOVES = 150
CHROMOSOME_SIZE = N_MOVES*2
CROSS_RATE = 0.8
MUTATE_RATE = 0.0001
POP_SIZE = 100
N_GENERATIONS = 200
GOAL_POINT = [90, 90]
START_POINT = [5, 5]
START_POINT2 = [10, 75]
START_POINT3 = [95, 15]


class GA(object):
    def __init__(self, CHROMOSOME_size, cross_rate, mutation_rate, pop_size, ):
        self.CHROMOSOME_size = CHROMOSOME_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        self.pop = np.random.randint(-1, 2 , size=(pop_size, CHROMOSOME_size))

    def decode(self, CHROMOSOME, n_moves, start_point):                 # convert to readable string
        pop = self.pop
        pop[:, 0], pop[:, n_moves] = start_point[0], start_point[1]
        lines_x = np.cumsum(pop[:, :n_moves], axis=1)
        lines_y = np.cumsum(pop[:, n_moves:], axis=1)
        return lines_x, lines_y

    def get_fitness(self, lines_x, lines_y, goal_point):
        dist2goal = np.sqrt((goal_point[0] - lines_x[:, -1]) ** 2 + (goal_point[1] - lines_y[:, -1]) ** 2)
        fitness = np.power(1 / (dist2goal + 1), 2)
        for i in lines_x:
            for j in i:
                if(j>20 and j<40):
                    fitness[i] = 0
        return fitness

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness/fitness.sum())
        return self.pop[idx]

    def crossover(self, parent1, parent2):
        child1 = parent1
        child2 = parent2
        if np.random.rand() < self.cross_rate:
            cross_points = np.random.randint(0, 2, self.CHROMOSOME_size).astype(np.bool)   # choose crossover points
            child1[cross_points], child2[cross_points] = child2[cross_points], child1[cross_points]                             # mating and produce one child
        return child1, child2

    def mutate(self, child):
        for point in range(self.CHROMOSOME_size):
            if np.random.rand() < self.mutate_rate:
                child[point] = np.random.randint(-1,2)
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop = np.random.permutation(pop)
        i = 0
        while i< len(pop)//2 +1:
            parent1 = pop[i]
            parent2 = pop[-i]
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            pop[i] = child1
            pop[-i] = child2
            i+=1
        self.pop = pop


class Line(object):
    def __init__(self, n_moves, goal_point, start_point,start_point2, start_point3):
        self.n_moves = n_moves
        self.goal_point = goal_point
        self.start_point = start_point
        self.start_point2 = start_point2
        self.start_point3 = start_point3
        plt.ion()

    def plotting(self, lines_x, lines_y,lines_x2,lines_y2,lines_x3,lines_y3):
        plt.cla()
        plt.grid(True)
        obj = plt.Rectangle((20,20),20,10, fc='red',ec="black") # (starting position, width, height)
        obj2 = plt.Rectangle((80,20),10,45, fc='red',ec="black")
        obj3 = plt.Rectangle((10,80),55,10, fc='red',ec="black")
        plt.gca().add_patch(obj)
        plt.gca().add_patch(obj2)
        plt.gca().add_patch(obj3)
        plt.scatter(*self.goal_point, s=200, c='r')
        plt.scatter(*self.start_point, s=100, c='b')
        plt.scatter(*self.start_point2, s=100, c='b')
        plt.scatter(*self.start_point3, s=100, c='b')
        plt.plot(lines_x.T, lines_y.T, c='k')
        plt.plot(lines_x2.T, lines_y2.T, c='k')
        plt.plot(lines_x3.T, lines_y3.T, c='k')
        plt.xlim((-5, 120))
        plt.ylim((-5, 120))
        plt.pause(0.01)


ga = GA(CHROMOSOME_size=CHROMOSOME_SIZE,
        cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
ga2 = GA(CHROMOSOME_size=CHROMOSOME_SIZE,
        cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
ga3 = GA(CHROMOSOME_size=CHROMOSOME_SIZE,
        cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)

env = Line(N_MOVES, GOAL_POINT, START_POINT, START_POINT2,START_POINT3)

for generation in range(N_GENERATIONS):
    lx, ly = ga.decode(ga.pop, N_MOVES, START_POINT)
    lx2, ly2 = ga2.decode(ga2.pop, N_MOVES, START_POINT2)
    lx3, ly3 = ga3.decode(ga3.pop, N_MOVES, START_POINT3)
    fitness = ga.get_fitness(lx, ly, GOAL_POINT)
    fitness2 = ga2.get_fitness(lx2, ly2, GOAL_POINT)
    fitness3 = ga3.get_fitness(lx3, ly3, GOAL_POINT)
    ga.evolve(fitness)
    ga2.evolve(fitness2)
    ga3.evolve(fitness3)
    #print('Gen:', generation, '| best fit:', fitness2.max())
    env.plotting(lx, ly, lx2, ly2, lx3, ly3)

plt.ioff()
plt.show()