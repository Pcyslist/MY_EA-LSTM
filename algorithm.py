import math
import random
from itertools import combinations
# 初始化种群，每个时间步的权重用长度为code_length的二进制串去表示，每个individual有time_stemps个时间步二进制串形式的权重组成。
def initialize_population(pop_size, time_steps, code_length):
    pop = []
    for _ in range(pop_size):
        individual = []
        for _ in range(time_steps * code_length):
            individual.append(1 if random.random() > 0.5 else 0)
        pop.append(individual)
    return pop
# 将一个个体中的一个特征的bin_code的权重转换为10进制
def decode(code):
    code_length = len(code)
    code = [str(i) for i in code]
    # transform binary system to decimal system
    element = round(int(''.join(code), 2) / (math.pow(2, code_length) - 1), 2)
    return element
# 将一个个体的二进制形式的特征权重转换为十进制,然后将其softmax化
def get_weight(indiv_code, time_steps, code_length):
    weight = []
    for i in range(time_steps):
        weight.append(decode(indiv_code[i * code_length: (i + 1) * code_length]))
    return weight
# 将一个种群的所有个体都转换为10进制权重
def pop_to_weights(pop, time_steps, code_length):
    weights = []
    for indiv in pop:
        weights.append(get_weight(indiv, time_steps, code_length))
    return weights
# 将二进制代码形式的pop和10进制的weights进行合并，产生初始pop和初始weights
def pop_weights_init(pop_size, time_steps, code_length):
    first_pop=initialize_population(pop_size,time_steps,code_length)
    weights=pop_to_weights(first_pop,time_steps,code_length)
    return first_pop,weights
# 获得交叉或变异的随机的几个断点，indiv_length就是一个个体的二进制串长度 time_steps*encode_length
def get_segment_ids(indiv_length):
    index = []
    while True:
        for i in range(indiv_length):
            if random.random() > 0.5:
                index.append(i)
        if len(index) > 0:
            break
    return index
# 变异
def mutation(indiv):
    indiv_length = len(indiv)
    # 随机确定变异的基因点
    index = get_segment_ids(indiv_length)
    for i in index:
        if indiv[i] == 0:
            indiv[i] = 1
        else:
            indiv[i] = 0
    return indiv
# 交叉并变异产生新的子代个体
def crossover(indiv1, indiv2):
    indiv_length = len(indiv1) # len(indiv1) == len(indiv2)
    # 获得父亲的基因断点位置
    a_index = get_segment_ids(indiv_length)
    # 母亲的基因断点位置就是不是父亲的基因断点位置的那些位置
    b_index = []
    for i in range(indiv_length):
        if i not in a_index:
            b_index.append(i)
    # 新的子代个体，初始化为0000...的长度为indiv_length的二进制序列串
    new = list()
    for i in range(indiv_length):
        new.append(0)
    # 从父亲那里得到遗传片段
    for i in a_index:
        new[i] = indiv1[i]
    # 从母亲那里得到遗传片段
    for i in b_index:
        new[i] = indiv2[i]
    # 只有很少的几率（此处为1-0.8的概率）得到的子代个体会发生变异
    if random.random() > 0.8:
        new = mutation(new)
    return new
# 对种群进行分组，先将种群中每个个体进行随机打乱，然后分组。
def group_population(pop, n_group):
    assert len(pop) % n_group == 0, "pop_size must be a multiple of n_group."
    # 每组的个体数
    per_group = len(pop) // n_group
    group_index = list(range(0, len(pop)))
    random.shuffle(group_index)
    group_pop = []
    for i in range(n_group):
        temp_index = group_index[i * per_group: (i + 1) * per_group]
        temp_pop = []
        for j in temp_index:
            temp_pop.append(pop[j])
        group_pop.append(temp_pop)
    return group_pop
# 将二进制串形式的个体转换为字符串的key，用于放入集合中，确保每个个体的独特性，字符串是hashable的
def individual_to_key(indiv):
    temp = [str(i) for i in indiv]
    key = ''.join(temp)
    return key
# 从当代种群中挑选n_select个个体，挑选几个个体就将种群分成几个组，从每个组中选择其中适应度最高（此处即rmse最小）的一个个体
def select(pop, n_select, key_to_rmse):
    # n_select==分组的个数n_group
    group_pop = group_population(pop, n_select)
    fitness_selected = []
    pop_selected = []
    for sub_group in group_pop:
        fitness = []
        for indiv in sub_group:
            key = individual_to_key(indiv)
            fitness.append(key_to_rmse[key])
        min_fitness = min(fitness)
        pop_selected.append(sub_group[fitness.index(min_fitness)])
        fitness_selected.append(min_fitness)
    return pop_selected, fitness_selected
# 重建种群
def reconstruct_population(pop_selected, pop_size):
    new_pop = list()
    # 保留挑选出来的个体
    new_pop.extend(pop_selected)
    pop_map = set()
    for i in range(len(new_pop)):
        pop_map.add(individual_to_key(new_pop[i]))
    # 从挑选出的个体中两两组成一对夫妇
    index = list(combinations(range(len(pop_selected)), 2))
    # 只要种群大小小于预定的种群大小就不停地让组成的夫妻生孩子
    while len(new_pop) < pop_size:
        for combi in index:
            new_indiv = crossover(pop_selected[combi[0]], pop_selected[combi[1]])
            # 保证产生的个体的独特性唯一性
            if not individual_to_key(new_indiv) in pop_map:
                new_pop.append(new_indiv)
                pop_map.add(individual_to_key(new_indiv))
            if len(new_pop) == pop_size:
                break
    return new_pop
