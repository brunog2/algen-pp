"""
Operadores e funções do algoritmo genético.
"""

import random
import numpy as np
import config


def create_random_individual(param_ranges):
    """
    Cria um indivíduo aleatório com parâmetros dentro dos intervalos.
    
    Args:
        param_ranges: Dicionário com intervalos dos parâmetros
    
    Returns:
        Dicionário com parâmetros do indivíduo
    """
    ind = {}
    for k, v in param_ranges.items():
        mn, mx, t = v
        if t == 'int':
            ind[k] = int(random.randint(mn, mx))
        else:
            ind[k] = float(random.uniform(mn, mx))
    return ind


def normalize_weights(ind):
    """
    Normaliza weight_size + weight_shape = 1.0
    
    Args:
        ind: Indivíduo (modificado in-place)
    """
    s = ind['weight_size'] + ind['weight_shape']
    if s == 0:
        ind['weight_size'] = 0.5
        ind['weight_shape'] = 0.5
    else:
        ind['weight_size'] /= s
        ind['weight_shape'] /= s


def fix_size_constraints(ind):
    """
    Garante que size_min <= size_max
    
    Args:
        ind: Indivíduo (modificado in-place)
    
    Returns:
        Indivíduo corrigido
    """
    if ind['size_min'] > ind['size_max']:
        ind['size_min'], ind['size_max'] = ind['size_max'], ind['size_min']
    return ind


def select_survivors(population, fitnesses):
    """
    Seleciona os melhores indivíduos (metade da população).
    
    Args:
        population: Lista de indivíduos
        fitnesses: Lista de fitnesses
    
    Returns:
        tupla: (survivors, (best_ind, best_fit))
    """
    paired = list(zip(population, fitnesses))
    paired.sort(key=lambda x: x[1])  # Menor é melhor
    survivors = [p for p, f in paired[:len(paired) // 2]]
    best = paired[0]
    return survivors, best


def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Seleção por torneio - seleciona melhor entre k indivíduos aleatórios.
    
    Args:
        population: Lista de indivíduos
        fitnesses: Lista de fitnesses
        tournament_size: Tamanho do torneio
    
    Returns:
        Indivíduo selecionado
    """
    tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]  # Menor é melhor
    return population[winner_idx]


def crossover(parent_a, parent_b, param_ranges, alpha=0.5):
    """
    Crossover BLX-alpha (Blend Crossover).
    Em vez de apenas média, explora região ao redor dos pais.
    
    Args:
        parent_a: Primeiro pai
        parent_b: Segundo pai
        param_ranges: Intervalos dos parâmetros
        alpha: Controla o tamanho da região explorada (0.5 = ±50% do intervalo entre pais)
    
    Returns:
        Filho gerado
    """
    child = {}
    for k, (mn, mx, t) in param_ranges.items():
        va = parent_a[k]
        vb = parent_b[k]
        
        # Calcular intervalo entre pais
        min_val = min(va, vb)
        max_val = max(va, vb)
        interval = max_val - min_val
        
        # Expandir intervalo por alpha
        expanded_min = min_val - alpha * interval
        expanded_max = max_val + alpha * interval
        
        # Garantir que está dentro dos limites originais
        expanded_min = max(mn, expanded_min)
        expanded_max = min(mx, expanded_max)
        
        # Escolher valor aleatório no intervalo expandido
        if t == 'int':
            val = random.randint(int(expanded_min), int(expanded_max))
            val = max(mn, min(mx, val))
        else:
            val = random.uniform(expanded_min, expanded_max)
            val = max(mn, min(mx, val))
        
        child[k] = val
    
    child = fix_size_constraints(child)
    return child


def mutate(ind, param_ranges, mutation_rate=None):
    """
    Mutação melhorada: altera genes com maior amplitude e mais frequência.
    
    Args:
        ind: Indivíduo a mutar
        param_ranges: Intervalos dos parâmetros
        mutation_rate: Taxa de mutação (usa config.MUTATION_RATE se None)
    
    Returns:
        Indivíduo mutado
    """
    if mutation_rate is None:
        mutation_rate = config.MUTATION_RATE
    
    if random.random() > mutation_rate:
        return ind
    
    out = ind.copy()
    for k, (mn, mx, t) in param_ranges.items():
        if random.random() < 0.7:  # Mutar ~70% dos genes quando mutação ocorre
            # Mutação mais agressiva: ±30%
            factor = random.uniform(0.70, 1.30)
            
            # 30% chance de mutação uniforme (resetar para valor aleatório)
            if random.random() < 0.3:
                if t == 'int':
                    out[k] = random.randint(mn, mx)
                else:
                    out[k] = random.uniform(mn, mx)
            else:
                # Mutação por fator
                if t == 'int':
                    nv = int(round(out[k] * factor))
                    nv = max(mn, min(mx, nv))
                    out[k] = nv
                else:
                    nv = out[k] * factor
                    nv = max(mn, min(mx, nv))
                    out[k] = nv
    
    out = fix_size_constraints(out)
    return out

