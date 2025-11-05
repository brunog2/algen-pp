"""
Funções de pós-processamento aprendido.
"""

import cv2
import numpy as np
from skimage import measure


def merge_adjacent_regions(bin_mask, orig_img, merge_threshold):
    """
    Fusão de regiões adjacentes baseada em similaridade de intensidade média.
    
    Args:
        bin_mask: Máscara binária
        orig_img: Imagem original
        merge_threshold: Threshold de similaridade para fusão
    
    Returns:
        Máscara binária com regiões fundidas
    """
    labels = measure.label(bin_mask, connectivity=2)
    props = measure.regionprops(labels, intensity_image=orig_img)
    n = labels.max()
    
    if n <= 1:
        return bin_mask
    
    # Intensidade média de cada região
    mean_intensities = np.zeros(n + 1)
    for p in props:
        mean_intensities[p.label] = p.mean_intensity if p.mean_intensity is not None else 0.0
    
    # Construir grafo de adjacência
    adjacency = {i: set() for i in range(1, n + 1)}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    for lab in range(1, n + 1):
        region_mask = (labels == lab).astype(np.uint8)
        dil = cv2.dilate(region_mask, kernel)
        overlap = np.unique(labels[(dil == 1) & (labels != lab)])
        for o in overlap:
            if o > 0:
                adjacency[lab].add(int(o))
                adjacency[o].add(int(lab))
    
    # Union-Find para fusão
    parent = list(range(n + 1))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    
    # Fusão baseada em intensidade
    for a, neighs in adjacency.items():
        for b in neighs:
            if a < b:
                mi = mean_intensities[a]
                mj = mean_intensities[b]
                denom = max(1.0, max(abs(mi), abs(mj)))
                if abs(mi - mj) / denom <= merge_threshold:
                    union(a, b)
    
    # Reconstruir labels
    new_labels = np.zeros_like(labels)
    mapping = {}
    cur = 1
    for lab in range(1, n + 1):
        root = find(lab)
        if root not in mapping:
            mapping[root] = cur
            cur += 1
        new_labels[labels == lab] = mapping[root]
    
    merged_mask = (new_labels > 0).astype(np.uint8) * 255
    return merged_mask


def post_processing_learned(seg_bin, orig_img, closing_kernel, merge_threshold, min_area, refinement_iterations=0):
    """
    Pós-processamento aprendido com refinamento adaptativo iterativo:
    1. Fechamento morfológico
    2. Remoção de regiões pequenas
    3. Fusão de regiões adjacentes
    4. Refinamento iterativo (aplica etapas 1-3 múltiplas vezes se necessário)
    
    Args:
        seg_bin: Segmentação binária
        orig_img: Imagem original
        closing_kernel: Tamanho do kernel de fechamento
        merge_threshold: Threshold de fusão
        min_area: Área mínima para manter região
        refinement_iterations: Número de iterações de refinamento
    
    Returns:
        Segmentação refinada
    """
    out = seg_bin.copy()
    
    # Refinamento iterativo
    for iteration in range(max(1, refinement_iterations + 1)):
        # Fechamento morfológico
        k = max(1, int(closing_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
        
        # Remoção de regiões pequenas
        labels = measure.label(out, connectivity=2)
        props = measure.regionprops(labels)
        filtered = np.zeros_like(out)
        for p in props:
            if p.area >= min_area:
                filtered[labels == p.label] = 255
        out = filtered
        
        # Fusão de regiões adjacentes (apenas na última iteração ou se threshold > 0)
        if merge_threshold > 0 and (iteration == refinement_iterations):
            out = merge_adjacent_regions(out, orig_img, merge_threshold)
    
    return out

