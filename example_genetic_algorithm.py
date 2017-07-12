
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# In[2]:

get_ipython().run_cell_magic(u'time', u'', u'test_set = [[np.random.binomial(1, 0.003) for i in range(100000)] for j in range(100)]\n#print(np.mean([check_recursive(ts, [1,3,8,30]) for ts in test_set]))')


# In[3]:

def evolve(generations, start_population):
    populations = []
    current_population = start_population
    for i in range(generations):
        populations.append(current_population)
        print("Generation {0} starts".format(i))
        test_set = get_test_set()
        performances = get_performances(current_population, test_set)    
        current_population = reproduct(current_population, performances)
    test_set = get_test_set()
    get_performances(current_population, test_set)
    populations.append(current_population)
    return populations

def get_performances(gen_arr, test_set):
    performances = []
    
    print("Getting performances.")
    for i in range(len(gen_arr)):
        performances.append(np.mean([check_recursive(ts, gen_arr[i])-len(ts) for ts in test_set]))
    
    top_args = np.argsort(performances)[:5]
    print("Top scores:")
    for i in range(len(top_args)):
        print(gen_arr[top_args[i]])
        print(performances[top_args[i]])
    return performances

def reproduct(gen_arr, perf_arr, mutate_rate = 0.05):
    new_genes = []
    
    top_args = np.argsort(perf_arr)[:5]
    for i in range(len(top_args)):
        new_genes.append(gen_arr[top_args[i]])
    
    probs = [1/perf for perf in perf_arr]
    
    print("Reproducting.")
    
    for i in top_args:
        first = i
        second = np.argmax(np.random.multinomial(1, probs/sum(probs)))
        new_gene = combine_arrays(gen_arr[first], gen_arr[second])
        if np.random.uniform() < mutate_rate:
            #print("gene: {0} mutated!".format(new_gene))
            new_gene = mutate(new_gene)
            #print("Mutation: {0}".format(new_gene))
        new_genes.append(new_gene)
    
    for i in range(len(gen_arr) - 10):
        first = np.argmax(np.random.multinomial(1, probs/sum(probs)))
        second = np.argmax(np.random.multinomial(1, probs/sum(probs)))
        new_gene = combine_arrays(gen_arr[first], gen_arr[second])
        if np.random.uniform() < mutate_rate:
            #print("gene: {0} mutated!".format(new_gene))
            new_gene = mutate(new_gene)
            #print("Mutation: {0}".format(new_gene))
        new_genes.append(new_gene)
    return new_genes

def mutate(gene):
    new_gene = [1]
    
    for i in range(1,len(gene)):
        proc = np.random.uniform()*4
        new_val = max(1, gene[i] * proc)
        new_gene.append(new_val)
    for i in range(1, len(new_gene)-1)[::-1]:
        while new_gene[i] > new_gene[i+1]:
            new_gene[i] = max(1,new_gene[i]/2)
        if new_gene[i]*2 > new_gene[i+1]:
            new_gene[i] = max(1,new_gene[i]/2)
    return new_gene

def combine_arrays(gene1, gene2):
    new_gene = []
    for i in range( len(gene1)):
        proc = np.random.uniform()
        new_val = gene1[i] * proc + gene2[i] * (1-proc)
        new_gene.append(new_val)
    for i in range(1, len(new_gene)-1)[::-1]:
        while new_gene[i] > new_gene[i+1]:
            new_gene[i] = max(1,new_gene[i]/2)
        if new_gene[i]*2 > new_gene[i+1]:
            new_gene[i] = max(1,new_gene[i]/2)
    return new_gene


def check_recursive(tab, k_arr):
    cost = 0
    last_remain = True
    if k_arr[-1]>1:
        for i in np.arange(0, len(tab), k_arr[-1]):
            if not ((i+k_arr[-1]) >= len(tab) and last_remain):
                cost+=len(tab[int(i): int(i+k_arr[-1])]) + 4
            if check(tab[int(i): int(i+k_arr[-1])]):
                last_remain = False
                cost+=check_recursive(tab[int(i): int(i+k_arr[-1])], k_arr[:-1])  
        return cost
    else:
        return 5*len(tab)

def check(tab):
    if max(tab) > 0:
        return True
    return False


def get_test_set(error_rate = 0.003, samples = 10000, sets = 100):
    test_set = [[np.random.binomial(1, error_rate) for i in range(samples)] for j in range(sets)]
    return test_set


# In[4]:

population = []
for i in range(1,5):
    for j in range(1,5):
        for k in range(1,5):
            for l in range(1,5):
                    population.append([1, 2*i, j*5, k * 10, l*20])


# In[ ]:

pops = evolve(40, population)

