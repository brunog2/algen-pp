"""
Funções de segmentação: Watershed e seleção por tamanho/forma.
"""

import numpy as np
from skimage import measure, filters, segmentation, feature, exposure
from scipy import ndimage as ndi
import config
from metrics import compute_ellipse_fit


def watershed_segmentation(img_pre, intensity_weight=0.3):
    """
    Watershed híbrido que combina marcadores baseados em:
    1. Distance transform (método original)
    2. Intensidade local (detecta células por intensidade)
    
    Args:
        img_pre: Imagem pré-processada
        intensity_weight: Peso para marcadores baseados em intensidade (0-1)
    
    Returns:
        Label map com regiões segmentadas
    """
    # Normalizar para uint8
    if img_pre.dtype != np.uint8:
        img = exposure.rescale_intensity(img_pre, out_range=np.uint8).astype(np.uint8)
    else:
        img = img_pre.copy()
    
    # Binarização Otsu
    val = filters.threshold_otsu(img)
    bw = img > val
    
    # Distance transform (método original)
    dist = ndi.distance_transform_edt(bw)
    
    # Marcadores baseados em distance transform
    coords_dist = feature.peak_local_max(dist, footprint=np.ones((3, 3)), labels=bw)
    local_maxi_dist = np.zeros_like(dist, dtype=bool)
    if len(coords_dist) > 0:
        local_maxi_dist[tuple(coords_dist.T)] = True
    
    # Marcadores baseados em intensidade local (detecta células brilhantes)
    if intensity_weight > 0:
        # Usar imagem original normalizada
        img_norm = exposure.rescale_intensity(img, out_range=(0, 1)).astype(np.float32)
        
        # Encontrar máximos locais de intensidade (células são mais brilhantes)
        coords_intensity = feature.peak_local_max(
            img_norm, 
            footprint=np.ones((5, 5)), 
            threshold_abs=0.6,  # Threshold de intensidade
            min_distance=5  # Distância mínima entre marcadores
        )
        local_maxi_intensity = np.zeros_like(img, dtype=bool)
        if len(coords_intensity) > 0:
            local_maxi_intensity[tuple(coords_intensity.T)] = True
        
        # Combinar marcadores: distance transform + intensidade
        local_maxi = local_maxi_dist.copy()
        # Adicionar marcadores de intensidade dentro da máscara binária
        local_maxi[bw & local_maxi_intensity] = True
    else:
        local_maxi = local_maxi_dist
    
    markers = ndi.label(local_maxi)[0]
    
    # Watershed
    labels = segmentation.watershed(-dist, markers, mask=bw)
    
    return labels


def select_regions_by_size_shape(labels, size_min, size_max, weight_size, weight_shape):
    """
    Seleção de regiões baseada em métricas ALC:
    - Score de tamanho (eq 3.1)
    - Score de forma (ellipse fit, eq 3.2)
    - Combinação ponderada
    
    Args:
        labels: Label map das regiões
        size_min: Tamanho mínimo de células
        size_max: Tamanho máximo de células
        weight_size: Peso do score de tamanho
        weight_shape: Peso do score de forma
    
    Returns:
        Máscara binária com regiões selecionadas
    """
    # Garantir size_min <= size_max
    if size_min > size_max:
        size_min, size_max = size_max, size_min
    
    # Intervalo estendido (conforme artigo: 2/3 e 4/3)
    min_ext = (2/3) * size_min
    max_ext = (4/3) * size_max
    
    props = measure.regionprops(labels)
    selected_mask = np.zeros(labels.shape, dtype=np.uint8)
    
    for prop in props:
        area = prop.area
        
        # Filtrar por intervalo estendido
        if not (min_ext <= area <= max_ext):
            continue
        
        # Score de tamanho (eq 3.1)
        if size_min <= area <= size_max:
            score_size = 1.0
        elif area >= size_max:
            score_size = size_max / area
        else:  # area < size_min
            score_size = area / size_min
        
        # Score de forma (ellipse fit, eq 3.2)
        coords = prop.coords
        score_shape = compute_ellipse_fit(coords)
        
        # Combinação ponderada
        score = weight_size * score_size + weight_shape * score_shape
        
        # Seleção (threshold configurável)
        if score >= config.ALC_SELECTION_THRESHOLD:
            selected_mask[prop.coords[:, 0], prop.coords[:, 1]] = 255
    
    return selected_mask

