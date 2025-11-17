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
MUTATION_RATE = 0.70  # 70% chance de mutação (aumentado de 50%)
ELITISM = 2
DIVERSITY_REINJECTION_RATE = 0.30  # 30% chance de criar indivíduo aleatório (aumentado de 20%)
DIVERSITY_STAGNATION_THRESHOLD = 3  # Reintroduzir diversidade após N gerações sem melhoria (reduzido de 5 para 3)

# Seed para reprodutibilidade
RANDOM_SEED = 42

# Intervalos dos genes (parâmetros a serem otimizados)
PARAM_RANGES = {
    'gaussian_sigma': (0.5, 2.5, 'float'),
    'median_ksize': (1, 5, 'int'),
    'erosion': (0, 5, 'int'),
    'dilation': (0, 5, 'int'),
    'size_min': (20, 200, 'int'),
    'size_max': (80, 1200, 'int'),  # Aumentado para 1200 para detectar células grandes
    'weight_size': (0.0, 1.0, 'float'),
    'weight_shape': (0.0, 1.0, 'float'),
    'closing_kernel': (1, 11, 'int'),
    'merge_threshold': (0.0, 0.3, 'float'),
    'min_area': (5, 200, 'int'),
    'intensity_weight': (0.0, 1.0, 'float'),  # Peso para marcadores baseados em intensidade
    'refinement_iterations': (0, 2, 'int'),  # Número de iterações de refinamento
    'use_morphological_gradient': (0, 1, 'int'),  # 0=False, 1=True - usar gradiente morfológico
    'use_edge_detection': (0, 1, 'int'),  # 0=False, 1=True - usar detecção de bordas Canny
}

# Pesos da fitness combinada
FITNESS_WEIGHT_ALMOD = 0.85
FITNESS_WEIGHT_QUALITY = 0.15
FITNESS_WEIGHT_CELLS = 0.10

# Threshold de seleção ALC
ALC_SELECTION_THRESHOLD = 0.3

