'''
author: Xu Junzhe
date: 2023/12/1
unicoding: utf-8
'''
import numpy as np
import pandas as pd
import copy
from stA_star import *
from map import *
import time
import matplotlib.pyplot as plt
import random

def load_data(file_path,sheet_name):
    data =pd.read_excel(file_path,sheet_name=sheet_name,header=None)
    return data[1:]

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, sku_info):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.sku_info = sku_info
        self.population = self.initialize_population()

    def initialize_population(self):
        # 初始化种群
        population = []
        for _ in range(self.population_size):
            chromosome = self.generate_chromosome_with_check(self.sku_info)
            population.append(chromosome)
        return population

    def generate_chromosome_with_check(self,sku_info):
        # 生成随机基因
        global demand
        tmp = copy.deepcopy(demand)
        chromosome = []
        for i in range(len(sku_info)):
            goal_idx = np.where(tmp[:,sku_info[i][0]]>0)[0]
            if goal_idx.shape[0] > 0:
                gene = random.choice(goal_idx)
                chromosome.append(gene)
                tmp[gene][sku_info[i][0]] -= 1
            else:
                gene = -1
                chromosome.append(gene)
        return chromosome
        
    def evaluate_fitness(self, chromosome):
       # 评价适应度
        entrances = {0:(0,0),1:(0,3),2:(0,6),3:(9,1),4:(9,4),5:(9,7)}
        goal = {0:(3,2),1:(3,5),2:(6,2),3:(6,5)}
        start =[]
        end = []
        for i in range(len(self.sku_info)):
            start.append(entrances[self.sku_info[i][2]] + (self.sku_info[i][1],))
        for i in range(len(chromosome)):
            if chromosome[i] != -1:
                end.append(goal[chromosome[i]])
            else:
                end.append((-1,-1,-1))
        ST_Table = np.expand_dims(raw_map.map, axis=0)
        fitness = 0
        for i in range(len(start)):
            road, work_time=StAstar(raw_map.map,start[i],end[i],ST_Table)
            #print(f'商品{j}最短路径：',road,'用时：',work_time+1)
            ST_Table = update_StTable(raw_map.map,ST_Table,road[:-1])
            fitness += (work_time+1)
        return -fitness

    def roulette_wheel_selection(self):
        # 轮盘赌选择父代
        fitness_values = [self.evaluate_fitness(individual) for individual in self.population]

        # 按照适应度从大到小排序
        fitness_values = np.array(fitness_values)
        idx = np.argsort(fitness_values)[::-1]
        fitness_values = [fitness_values[i] for i in idx]
        self.population = [self.population[i] for i in idx]

        '''# 总适应度
        total_fitness = sum(fitness_values)

        # 计算每个个体的选择概率
        selection_probabilities = [fitness / (-total_fitness) for fitness in fitness_values]
        
        # 计算每个个体的选择概率
        selection_probabilities = [fitness / total_fitness for fitness in adjusted_fitness]'''

        # 将适应度转换为正数（假设更接近0的适应度更好）
        min_fitness = min(fitness_values)
        adjusted_fitness = [fitness + abs(min_fitness) + 1e-4 for fitness in fitness_values]  # 转换为正数，越接近0的适应度变得越大
        # 计算调整后的总适应度
        total_fitness = sum(adjusted_fitness)
        
        # 计算每个个体的选择概率
        selection_probabilities = [fitness / total_fitness for fitness in adjusted_fitness]

        # 计算个体累积选则概率
        cumulative_probabilities = np.cumsum(selection_probabilities)

        # 选择父代
        population_indices = list(range(len(self.population)))
        selected_indices = []
        for _ in range(2):
            random_number = random.random()
            for (index, probability) in enumerate(cumulative_probabilities):
                if random_number <= probability:
                    selected_indices.append(index)
                    break
        parents = [self.population[index] for index in selected_indices]
    
        '''# 选择父代
        population_indices = list(range(len(self.population)))
        selected_indices = random.choices(population_indices, weights=selection_probabilities, k=2)
        parents = [self.population[index] for index in selected_indices]'''

        return parents

    def crossover(self, parent1, parent2):
        # 两点交叉
        global demand
        tmp = copy.deepcopy(demand)
        check = np.zeros(demand.shape)
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)
        # 生成随机交叉点
        crossover_point1 = random.randint(0, len(parent1) - 1)
        crossover_point2 = random.randint(0, len(parent1) - 1)
        # 交换基因片段
        if crossover_point1 > crossover_point2:
            crossover_point1, crossover_point2 = crossover_point2, crossover_point1
        for i in range(crossover_point1, crossover_point2 + 1):
            offspring1[i] = parent2[i]
            offspring2[i] = parent1[i]
        # 检查是否满足需求
        for i in range(len(offspring1)):
            if offspring1[i] != -1:
                check[int(offspring1[i])][sku_info[i][0]] += 1
        wrong_demand = tmp - check
        for i in range(max(self.sku_info[:,0])+1):
            na_demand_idx = np.where(wrong_demand[:,i] < 0)[0]
            po_demand_idx = np.where(wrong_demand[:,i] > 0)[0]
            sku_idx = np.where(self.sku_info[:,0] == i)[0]
            while len(na_demand_idx) > 0:
                for j in sku_idx:
                    if offspring1[j] in na_demand_idx:
                        fix_pos = np.random.choice(po_demand_idx)
                        wrong_demand[offspring1[j]][i] += 1
                        wrong_demand[fix_pos][i] -= 1
                        offspring1[j] = fix_pos
                        na_demand_idx = np.where(wrong_demand[:,i] < 0)[0]
                        po_demand_idx = np.where(wrong_demand[:,i] > 0)[0]
                        break
        tmp = copy.deepcopy(demand)
        check = np.zeros(demand.shape)
        for i in range(len(offspring2)):
            if offspring2[i] != -1:
                check[int(offspring2[i])][sku_info[i][0]] += 1
        wrong_demand = tmp - check
        for i in range(max(self.sku_info[:,0])+1):
            na_demand_idx = np.where(wrong_demand[:,i] < 0)[0]
            po_demand_idx = np.where(wrong_demand[:,i] > 0)[0]
            sku_idx = np.where(self.sku_info[:,0] == i)[0]
            while len(na_demand_idx) > 0:
                for j in sku_idx:
                    if offspring2[j] in na_demand_idx:
                        fix_pos = np.random.choice(po_demand_idx)
                        wrong_demand[offspring2[j]][i] += 1
                        wrong_demand[fix_pos][i] -= 1
                        offspring2[j] = fix_pos
                        na_demand_idx = np.where(wrong_demand[:,i] < 0)[0]
                        po_demand_idx = np.where(wrong_demand[:,i] > 0)[0]
                        break
        return offspring1, offspring2
    def mutate(self, chromosome):
        # Mutate a chromosome by randomly changing its genes
        global demand
        tmp = copy.deepcopy(demand)
        # 计算需求
        check = np.zeros(demand.shape)
        for i in range(len(chromosome)):
            if chromosome[i] != -1:
                check[int(chromosome[i])][sku_info[i][0]] += 1
        wrong_demand = tmp - check
        if len(np.where(wrong_demand!=0)[0]) == 0:
            for i in range(max(self.sku_info[:,0])+1):
                sku_idx = np.where(self.sku_info[:,0] == i)[0]
                for j in sku_idx:
                    if random.random() <= self.mutation_rate:
                        #随机调换在chromosome中的位置
                        random_pos = random.choice(sku_idx)
                        chromosome[j], chromosome[random_pos] = chromosome[random_pos], chromosome[j]
        if len(np.where(wrong_demand!=0)[0]) > 0:
             for i in range(max(self.sku_info[:,0])+1):
                sku_idx = np.where(self.sku_info[:,0] == i)[0]
                for j in sku_idx:
                    if len(np.where(wrong_demand[:,i] > 0)[0]) > 0:
                        if random.random() <= self.mutation_rate:
                            #随机变异成另一个有需求的位置
                            random_pos = random.choice(np.where(wrong_demand[:,i]>0)[0])
                            wrong_demand[int(chromosome[j])][i] += 1
                            wrong_demand[random_pos][i] -= 1
                            chromosome[j] = random_pos
                    if len(np.where(wrong_demand[:,i] < 0)[0]) > 0:
                        if random.random() <= self.mutation_rate:
                            #随机变异成另一个有需求的位置
                            random_pos = random.choice(sku_idx)
                            chromosome[j], chromosome[random_pos] = chromosome[random_pos], chromosome[j]
        return chromosome
    
    # 精英保留策略
    def select_best_individuals(self, parents, offspring):
        # 选择父代最优个体
        parents_fitness = [self.evaluate_fitness(individual) for individual in parents]
        parents_fitness = np.array(parents_fitness)
        idx = np.argsort(parents_fitness)[::-1]
        parents_fitness = [parents_fitness[i] for i in idx]
        parents = [parents[i] for i in idx]
        # 父代最优1/5个体加入下一代
        num = int(self.population_size/5)
        offspring.extend(parents[:num])
        fitness = []
        for i in range(len(offspring)):
            fitness.append(self.evaluate_fitness(offspring[i]))
        fitness = np.array(fitness)
        idx = np.argsort(fitness)[::-1]
        fitness = [fitness[i] for i in idx]
        next_generation=[offspring[i] for i in idx[:self.population_size]]
        print(fitness)
        return next_generation
        

    def run(self, generations):
    # 运行遗传算法指定的代数
        best_fitness_list = []
        best_fitness = -100000
        for gen in range(generations):
            # 初始化下一代的后代列表
            next_generation = []

            # 继续选择和交叉，直到达到种群大小
            while len(next_generation) < self.population_size:
                # 从当前种群中选择父代, 轮盘赌
                parents = self.roulette_wheel_selection()
                # 对每对父代进行交叉
                for i in range(0, len(parents), 2):
                    parent1 = parents[i]
                    parent2 = parents[min(i + 1, len(parents) - 1)]  # 确保不越界
                    if random.random() <= self.crossover_rate:
                        offspring1, offspring2 = self.crossover(parent1, parent2)
                        next_generation.extend([offspring1, offspring2])
                    else:
                        next_generation.extend([parent1, parent2])

                    # 如果已经生成足够数量的后代，就跳出循环
                    if len(next_generation) >= self.population_size:
                        break

            # 变异
            for i in range(len(next_generation)):
                next_generation[i] = self.mutate(next_generation[i])

            # 评估后代的适应度，并选择下一代
            self.population = self.select_best_individuals(parents,next_generation)

            # 输出当前全局最优解
            best_chromosome = self.population[0]
            best_fitness_now = self.evaluate_fitness(best_chromosome)
            if best_fitness_now > best_fitness:
                best_fitness = best_fitness_now
            print(f'第{gen+1}代的最优解：', best_chromosome, '适应度：', -best_fitness_now)
            best_fitness_list.append(-best_fitness)
            '''if self.evaluate_fitness(self.population[0]) > best_fitness:
                best_chromosome = self.population[0]
                best_fitness = self.evaluate_fitness(best_chromosome)
            print(f'第{gen+1}代的最优解：', best_chromosome, '适应度：', -best_fitness)'''

        return best_chromosome, best_fitness, best_fitness_list
    
    # 绘制算法迭代过程
    def plot(self, best_fitness_list):
        plt.plot(best_fitness_list)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()

# 测试迭代
if __name__ == '__main__':
    file_path = "CAsimulator\instance\s_1.xlsx"
    sku_data = load_data(file_path,sheet_name="sku_info")
    sku_info = np.array(sku_data.iloc[:,1:])
    demand = np.array(load_data(file_path,sheet_name="demand"))
    raw_map = Map()  
    population_size = 30
    mutation_rate = 0.03
    crossover_rate = 0.9
    generations = 200
    start_time = time.time()
    genetic_algorithm = GeneticAlgorithm(population_size, mutation_rate, crossover_rate, sku_info)
    best_chromosome, best_fitness, best_fitness_list = genetic_algorithm.run(generations)
    end_time = time.time()
    print('运行时间：',end_time-start_time)
    
    # 绘制算法迭代过程
    genetic_algorithm.plot(best_fitness_list)

    