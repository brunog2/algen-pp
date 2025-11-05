"""
Configurações do algoritmo genético e parâmetros.
"""

import os

# Diretórios
IMAGES_DIR = "../images"
OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parâmetros do Algoritmo Genético
POP_SIZE = 20
NUM_GENERATIONS = 100
MUTATION_RATE = 0.50  # 50% chance de mutação
ELITISM = 2
DIVERSITY_REINJECTION_RATE = 0.20  # 20% chance de criar indivíduo aleatório
DIVERSITY_STAGNATION_THRESHOLD = 5  # Reintroduzir diversidade após N gerações sem melhoria

# Seed para reprodutibilidade
RANDOM_SEED = 42

# Intervalos dos genes (parâmetros a serem otimizados)
PARAM_RANGES = {
    'gaussian_sigma': (0.5, 2.5, 'float'),
    'median_ksize': (1, 5, 'int'),
    'erosion': (0, 5, 'int'),
    'dilation': (0, 5, 'int'),
    'size_min': (20, 200, 'int'),
    'size_max': (80, 800, 'int'),
    'weight_size': (0.0, 1.0, 'float'),
    'weight_shape': (0.0, 1.0, 'float'),
    'closing_kernel': (1, 11, 'int'),
    'merge_threshold': (0.0, 0.3, 'float'),
    'min_area': (5, 200, 'int'),
    'intensity_weight': (0.0, 1.0, 'float'),  # Peso para marcadores baseados em intensidade
    'refinement_iterations': (0, 2, 'int'),  # Número de iterações de refinamento
}

# Pesos da fitness combinada
FITNESS_WEIGHT_ALMOD = 0.85
FITNESS_WEIGHT_QUALITY = 0.15
FITNESS_WEIGHT_CELLS = 0.10

# Threshold de seleção ALC
ALC_SELECTION_THRESHOLD = 0.3

