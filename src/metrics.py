"""
Métricas de avaliação de segmentação.
"""

import numpy as np
import math
from skimage import measure
import config


def compute_ellipse_fit(pixels_coords):
    """
    Calcula score de forma baseado em ellipse fit (eq 3.2 do artigo).
    Retorna: area_objeto / area_elipse (quanto mais próximo de 1.0, melhor)
    
    Args:
        pixels_coords: Coordenadas dos pixels (Nx2 array)
    
    Returns:
        Score de forma (0.0 a 1.0)
    """
    if len(pixels_coords) < 5:
        return 0.0
    
    # Coordenadas
    xs = pixels_coords[:, 1].astype(np.float64)  # colunas
    ys = pixels_coords[:, 0].astype(np.float64)  # linhas
    
    # Momentos
    m00 = len(xs)
    xs_sum = xs.sum()
    ys_sum = ys.sum()
    xxs = (xs**2).sum()
    yys = (ys**2).sum()
    xys = (xs*ys).sum()
    
    # Momentos centralizados
    m02 = yys - (ys_sum**2 / m00)
    m11 = xys - (xs_sum * ys_sum / m00)
    m20 = xxs - (xs_sum**2 / m00)
    
    # Matriz de covariância
    M = np.array([[m02, m11],
                  [m11, m20]])
    
    # Autovalores
    try:
        lambdas = np.linalg.eigvals(M)
        lambdas = np.sort(np.real(lambdas))[::-1]
    except Exception:
        return 0.0
    
    if lambdas[0] <= 0 or lambdas[1] <= 0:
        return 0.0
    
    # Semi-eixos da elipse
    a = 2.0 * math.sqrt(lambdas[0] / m00)
    b = 2.0 * math.sqrt(lambdas[1] / m00)
    area_ellipse = math.pi * a * b
    
    if area_ellipse <= 0:
        return 0.0
    
    score = m00 / area_ellipse
    return max(0.0, min(1.0, score))


def almod_metric(orig, seg_bin):
    """
    Métrica Almod: soma das diferenças absolutas pixel a pixel normalizada.
    Menor é melhor.
    
    Normaliza pela área segmentada para não penalizar segmentações com mais células.
    
    Args:
        orig: Imagem original
        seg_bin: Segmentação binária (0 ou 1)
    
    Returns:
        Score Almod normalizado
    """
    seg_img = (seg_bin > 0).astype(np.uint8) * 255
    diff = np.abs(orig.astype(np.int32) - seg_img.astype(np.int32))
    total_diff = diff.sum()
    
    # Normalizar pela área segmentada para evitar penalizar mais células
    area_segmented = np.sum(seg_bin > 0)
    if area_segmented > 0:
        # Média da diferença por pixel segmentado
        normalized_diff = total_diff / area_segmented
        # Multiplicar pela raiz quadrada da área para balancear
        # (favorece mais células mas controla qualidade)
        return normalized_diff * np.sqrt(area_segmented)
    else:
        # Penalidade máxima se não há segmentação
        return 1e9


def compute_segmentation_quality(seg_bin):
    """
    Calcula qualidade média da segmentação baseada em forma das células.
    Retorna média dos scores de forma (ellipse fit) de todas as regiões segmentadas.
    
    Args:
        seg_bin: Segmentação binária (0 ou 1)
    
    Returns:
        Penalidade de qualidade (menor é melhor)
    """
    if np.sum(seg_bin > 0) == 0:
        return 1.0  # Penalidade máxima se não há segmentação
    
    labels = measure.label(seg_bin, connectivity=2)
    props = measure.regionprops(labels)
    
    if len(props) == 0:
        return 1.0
    
    shape_scores = []
    for prop in props:
        if prop.area >= 5:  # Mínimo de pixels para calcular forma
            coords = prop.coords
            score = compute_ellipse_fit(coords)
            shape_scores.append(score)
    
    if len(shape_scores) == 0:
        return 1.0
    
    # Retorna inverso da média (para manter menor=melhor como Almod)
    # Quanto melhor a forma, menor o valor
    mean_shape_score = np.mean(shape_scores)
    quality_penalty = 1.0 - mean_shape_score
    return quality_penalty * 1000000  # Escalar para magnitude similar ao Almod


def compute_completeness_penalty(orig_img, seg_binary):
    """
    Calcula penalidade por segmentação incompleta de células.
    Detecta áreas claras (células) na imagem original que não foram segmentadas
    e penaliza baseado na área não coberta.
    
    Args:
        orig_img: Imagem original
        seg_binary: Segmentação binária (0 ou 1)
    
    Returns:
        Penalidade por incompletude (menor é melhor)
    """
    # Normalizar imagem original para 0-1
    orig_norm = orig_img.astype(np.float32) / 255.0
    
    # Encontrar áreas claras (células) na imagem original
    # Usar threshold adaptativo baseado na mediana + desvio padrão
    median_intensity = np.median(orig_norm)
    std_intensity = np.std(orig_norm)
    # Threshold para detectar células (median + 1.5*std)
    cell_threshold = median_intensity + 1.5 * std_intensity
    cell_threshold = max(0.3, min(0.7, cell_threshold))  # Entre 0.3 e 0.7
    
    # Máscara de células (áreas claras na imagem original)
    cell_mask = (orig_norm > cell_threshold).astype(np.uint8)
    
    # Área total de células na imagem original
    total_cell_area = np.sum(cell_mask > 0)
    
    if total_cell_area == 0:
        return 0.0  # Não há células para detectar
    
    # Área de células que foi segmentada
    segmented_cell_area = np.sum((cell_mask > 0) & (seg_binary > 0))
    
    # Área de células que NÃO foi segmentada (incompleta)
    unsegmented_cell_area = total_cell_area - segmented_cell_area
    
    # Penalidade proporcional à área não segmentada
    completeness_ratio = segmented_cell_area / total_cell_area  # 0 a 1, quanto maior melhor
    incompleteness_penalty = (1.0 - completeness_ratio) * 500000  # Penalidade máxima de 500k se nada segmentado
    
    return incompleteness_penalty


def compute_fitness(orig_img, seg_binary):
    """
    Calcula fitness combinada: Almod + qualidade de forma + recompensa por número de células + penalidade por incompletude.
    
    Args:
        orig_img: Imagem original
        seg_binary: Segmentação binária (0 ou 1)
    
    Returns:
        Fitness combinada (menor é melhor)
    """
    almod_score = almod_metric(orig_img, seg_binary)
    quality_score = compute_segmentation_quality(seg_binary)
    
    # CORREÇÃO CRÍTICA: Penalidade por segmentação incompleta (células parcialmente segmentadas)
    completeness_penalty = compute_completeness_penalty(orig_img, seg_binary)
    
    # Recompensa por número de células detectadas (favorece mais cobertura)
    labels_final = measure.label(seg_binary, connectivity=2)
    num_cells = labels_final.max()  # Número de células detectadas
    
    # Penalizar se não há células, mas recompensar proporcionalmente
    if num_cells == 0:
        cell_penalty = 500000  # Penalidade alta se não detecta células
    else:
        # Pequena penalização que diminui com mais células (até ~50 células)
        cell_penalty = max(0, 100000 - (num_cells * 1000))
    
    # Combinar com pesos
    # CORREÇÃO: Adicionar peso para penalidade de incompletude
    fitness = (config.FITNESS_WEIGHT_ALMOD * almod_score + 
               config.FITNESS_WEIGHT_QUALITY * quality_score + 
               config.FITNESS_WEIGHT_CELLS * cell_penalty +
               config.FITNESS_WEIGHT_COMPLETENESS * completeness_penalty)
    
    return fitness

