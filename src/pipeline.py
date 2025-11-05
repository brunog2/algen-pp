"""
Pipeline completo de segmentação.
"""

import numpy as np
import preprocessing
import segmentation
import postprocessing
from metrics import compute_fitness


def segment_image(img, individual):
    """
    Aplica o pipeline completo de segmentação em uma imagem.
    
    Args:
        img: Imagem de entrada
        individual: Dicionário com parâmetros do indivíduo
    
    Returns:
        tupla: (segmentação binária, fitness)
    """
    # Pré-processamento
    pre = preprocessing.preprocess_image(
        img,
        gaussian_sigma=individual['gaussian_sigma'],
        median_ksize=int(individual['median_ksize']),
        erosion_size=int(individual['erosion']),
        dilation_size=int(individual['dilation'])
    )
    
    # Watershed híbrido com intensidade
    intensity_weight = individual.get('intensity_weight', 0.3)
    labels = segmentation.watershed_segmentation(pre, intensity_weight=float(intensity_weight))
    
    # Ajustar constraints de tamanho
    size_min = int(individual['size_min'])
    size_max = int(individual['size_max'])
    if size_min > size_max:
        size_min, size_max = size_max, size_min
    
    # Seleção por tamanho/forma
    selected = segmentation.select_regions_by_size_shape(
        labels,
        size_min=size_min,
        size_max=size_max,
        weight_size=individual['weight_size'],
        weight_shape=individual['weight_shape']
    )
    
    # Refinamento iterativo
    refinement_iter = int(individual.get('refinement_iterations', 0))
    refined = postprocessing.post_processing_learned(
        selected,
        img,
        closing_kernel=int(individual['closing_kernel']),
        merge_threshold=float(individual['merge_threshold']),
        min_area=int(individual['min_area']),
        refinement_iterations=refinement_iter
    )
    
    # Calcular fitness
    seg_binary = (refined > 0).astype(np.uint8)
    fitness = compute_fitness(img, seg_binary)
    
    return seg_binary, fitness


def evaluate_individual(individual, images, names):
    """
    Avalia um indivíduo sobre todas as imagens do dataset.
    
    Args:
        individual: Dicionário com parâmetros do indivíduo
        images: Lista de imagens
        names: Lista de nomes das imagens
    
    Returns:
        Fitness médio (menor é melhor)
    """
    total_fitness = 0.0
    
    for img, name in zip(images, names):
        seg_binary, fitness = segment_image(img, individual)
        total_fitness += fitness
    
    mean_fitness = total_fitness / len(images)
    return mean_fitness

