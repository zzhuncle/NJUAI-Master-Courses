import random
import numpy as np
import networkx as nx
from functools import partial

best_list, mean_list = [], []

# individual
class Individual:
	def __init__(self, genotype = None):
		if genotype is None:
			genotype = []
		self.genotype = np.array(genotype)
		self.fitness = 0

	def initialize(length):
		individual = Individual()
		individual.genotype = np.random.choice((0, 1), p = (0.5, 0.5), size = length)
		return individual

# selection
def select_best_solution(candidates):
	best_ind = np.argmax([ind.fitness for ind in candidates])
	return candidates[best_ind]

def best_selection(population, offspring):
    selection_pool = np.concatenate((population, offspring), axis = None).tolist()
    selection_pool.sort(key = lambda x : x.fitness, reverse = True)
    return selection_pool[ : len(population)]

def tournament_selection(population, offspring):
	selection_pool = np.concatenate((population, offspring), axis = None)
	tournament_size = 4
	assert len(selection_pool) % tournament_size == 0, "Population size should be a multiple of 2"

	selection = []
	number_of_rounds = tournament_size // 2
	for _ in range(number_of_rounds):
		number_of_tournaments = len(selection_pool) // tournament_size
		order = np.random.permutation(len(selection_pool))
		for j in range(number_of_tournaments):
			indices = order[tournament_size * j : tournament_size * (j + 1)]
			best = select_best_solution(selection_pool[indices])
			selection.append(best)
	assert(len(selection) == len(population))
	return selection

# mutation
def onebit_mutation(ind_list):
	for ind in ind_list:
		pos =  np.random.randint(len(ind.genotype))
		ind.genotype[pos] = 1 - ind.genotype[pos]
	return ind_list

def bitwise_mutation(ind_list):
	l = len(ind_list[0].genotype)
	p = 1 / l
	for ind in ind_list:
		m = np.random.choice((0,1), p=(1-p, p), size=l)
		flip = 1 - ind.genotype
		ind.genotype = np.where(m, flip, ind.genotype)
	return ind_list

# crossover
def uniform_crossover(individual_a: Individual, individual_b: Individual, p = 0.5):
	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)

	m = np.random.choice((0,1), p=(p, 1-p), size=l)
	offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
	offspring_b.genotype = np.where(1 - m, individual_a.genotype, individual_b.genotype)

	return [offspring_a, offspring_b]

def one_point_crossover(individual_a: Individual, individual_b: Individual):
	l = len(individual_a.genotype)
	offspring_a = Individual(l)
	offspring_b = Individual(l)

	m = np.arange(l) < np.random.randint(l+1)
	offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
	offspring_b.genotype = np.where(~m, individual_a.genotype, individual_b.genotype)

	return [offspring_a, offspring_b]

def two_point_crossover(individual_a: Individual, individual_b: Individual):
	offspring_a = Individual()
	offspring_b = Individual()

	l = len(individual_a.genotype)
	m = (np.arange(l) < np.random.randint(l+1)) ^ (np.arange(l) < np.random.randint(l+1))
	offspring_a.genotype = np.where(m, individual_b.genotype, individual_a.genotype)
	offspring_b.genotype = np.where(~m, individual_b.genotype, individual_a.genotype)

	return [offspring_a, offspring_b]

# 演化算法
class GeneticAlgorithm(object):
	def __init__(self, args, graph, population_size):
		self.args = args
		self.population_size = population_size
		self.graph = graph
		self.n_nodes = len(graph.nodes)
		self.n_edges = len(graph.edges)
		self.population = []
		self.number_of_generations = 0

		self.variation_operator = uniform_crossover
		self.mutation_operator = onebit_mutation
		self.selection_operator = best_selection
		self.parents_operator = self.fitness_parents
		self.save_label = args.graph_type if args.graph_type != 'gset' else f'{args.graph_type}{args.gset_id}'
		self.verbose = True

	# parents_selection
	def uniform_parents(self):
		parents_pool = []
		N = len(self.population)
		for _ in range(N):
			parents_pool.append(self.population[np.random.randint(N)])
		return parents_pool

	def fitness_parents(self):
		wheel = np.array([_.fitness for _ in self.population])
		wheel /= wheel.sum()
		N = len(self.population)
		parents_pool = []
		for _ in range(N):
			rand_val, p = random.random(), 0
			for i in range(N):
				p += wheel[i]
				if p >= rand_val:
					parents_pool.append(self.population[i])
					break
		return parents_pool

	def get_fitness(self, individual: Individual):
		g1 = np.where(individual.genotype == 0)[0]
		g2 = np.where(individual.genotype == 1)[0]
		individual.fitness = nx.cut_size(self.graph, g1, g2) / self.n_edges

	def initialize_population(self): # 初始化种群
		self.population = [Individual.initialize(self.n_nodes) for _ in range(self.population_size)]
		for individual in self.population:
			self.get_fitness(individual)

	def make_offspring(self): # 产生后代
		offspring = []
		order = np.random.permutation(self.population_size)
		parents_pool = self.parents_operator()
		for i in range(len(order) // 2):
			offspring += self.mutation_operator(self.variation_operator(
				parents_pool[order[2 * i]], parents_pool[order[2 * i + 1]]))
		for individual in offspring:
			self.get_fitness(individual)
		return offspring

	def make_selection(self, offspring): # 选择后代
		return self.selection_operator(self.population, offspring)

	def statistics(self):
		fitness_list = [ind.fitness for ind in self.population]
		best, mean = max(fitness_list), np.mean(fitness_list)
		print(f"Generation {self.number_of_generations}: Best: {best}, Mean: {mean}")
		best_list.append(best)
		mean_list.append(mean)
		np.save(f'{self.save_label}_best.npy', best_list)
		np.save(f'{self.save_label}_mean.npy', mean_list)

	def get_best_fitness(self): # 最优解
		return max(ind.fitness for ind in self.population)

	def run(self):
		self.initialize_population()
		while (self.number_of_generations < self.args.T):
			self.number_of_generations += 1
			offspring = self.make_offspring()
			self.population = self.make_selection(offspring)
			if (self.verbose):
				self.statistics()
		return self.get_best_fitness(), self.number_of_generations
