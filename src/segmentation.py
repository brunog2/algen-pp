"""
Funções de segmentação: Watershed e seleção por tamanho/forma.
"""

import numpy as np
import cv2
from skimage import measure, filters, segmentation, feature, exposure
from scipy import ndimage as ndi
import config
from metrics import compute_ellipse_fit


def watershed_segmentation(img_pre, intensity_weight=0.3, use_edge_detection=False):
    """
    Watershed híbrido que combina marcadores baseados em:
    1. Distance transform (método original)
    2. Intensidade local (detecta células por intensidade)
    3. Detecção de bordas (opcional, para melhor identificação de células)
    
    Args:
        img_pre: Imagem pré-processada
        intensity_weight: Peso para marcadores baseados em intensidade (0-1)
        use_edge_detection: Se True, usa detecção de bordas (Canny) para melhorar marcadores
    
    Returns:
        Label map com regiões segmentadas
    """
    # Normalizar para uint8
    if img_pre.dtype != np.uint8:
        img = exposure.rescale_intensity(img_pre, out_range=np.uint8).astype(np.uint8)
    else:
        img = img_pre.copy()
    
    # MELHORIA: Detecção de bordas MELHORADA para melhor identificação de células
    # Sempre detecta bordas, mas usa de forma adaptativa baseado no parâmetro
    edges = None
    edge_mask = None
    
    # Detecção de bordas (sempre aplicada, mas intensidade varia)
    # Canny edge detection para identificar bordas de células
    v = np.median(img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges_canny = cv2.Canny(img, lower, upper)
    
    if use_edge_detection:
        # Modo agressivo: usar bordas diretamente na binarização
        # Dilatar bordas para conectar bordas próximas
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges_canny, kernel, iterations=1)
        # Usar bordas como informação adicional na binarização
        img_with_edges = np.maximum(img, edges_dilated.astype(np.uint8) * 255)
        edges = edges_canny
        edge_mask = edges_dilated
    else:
        # Modo conservador: usar bordas apenas para melhorar marcadores
        img_with_edges = img
        edges = edges_canny
        # Criar máscara de bordas para usar nos marcadores
        kernel = np.ones((5, 5), np.uint8)
        edge_mask = cv2.dilate(edges_canny, kernel, iterations=1)
    
    # Binarização Otsu
    val = filters.threshold_otsu(img_with_edges)
    bw = img_with_edges > val
    
    # Distance transform (método original)
    dist = ndi.distance_transform_edt(bw)
    
    # Marcadores baseados em distance transform
    # MELHORIA: Usar footprint maior para detectar células grandes
    # Footprint de 5x5 detecta melhor células grandes no centro
    # CORREÇÃO: Threshold balanceado - não muito alto (perde células) nem muito baixo (detecta ruído)
    max_dist = np.max(dist)
    # CORREÇÃO: Threshold mais permissivo para detectar mais células (20% ao invés de 25%)
    dist_threshold = max(3.0, max_dist * 0.20)  # Threshold mínimo de 3 pixels ou 20% do máximo
    
    coords_dist = feature.peak_local_max(
        dist, 
        footprint=np.ones((5, 5)), 
        labels=bw, 
        min_distance=3,  # Reduzido de 4 para 3 para detectar células próximas
        threshold_abs=dist_threshold  # Threshold mais permissivo
    )
    local_maxi_dist = np.zeros_like(dist, dtype=bool)
    if len(coords_dist) > 0:
        local_maxi_dist[tuple(coords_dist.T)] = True
    
    # MELHORIA ADICIONAL: Se há poucos marcadores, usar threshold mais baixo
    # CORREÇÃO: Tornar mais permissivo para detectar células escuras ou grandes
    if np.sum(local_maxi_dist) < 5 and max_dist > 5:  # Mais permissivo: < 5 ao invés de < 2, > 5 ao invés de > 8
        # Tentar com threshold mais baixo e footprint maior para células grandes/escuras
        coords_dist_large = feature.peak_local_max(
            dist, 
            footprint=np.ones((7, 7)), 
            labels=bw, 
            min_distance=4,  # Reduzido de 6 para 4
            threshold_abs=max_dist * 0.15  # Threshold mais baixo (15% do máximo) para detectar mais células
        )
        if len(coords_dist_large) > 0:
            local_maxi_dist[tuple(coords_dist_large.T)] = True
    
    # Marcadores baseados em intensidade local (detecta células brilhantes)
    if intensity_weight > 0:
        # Usar imagem original normalizada
        img_norm = exposure.rescale_intensity(img, out_range=(0, 1)).astype(np.float32)
        
        # Encontrar máximos locais de intensidade (células são mais brilhantes)
        # CORREÇÃO: Threshold mais baixo para detectar células escuras também
        # Usar percentil mais baixo para incluir células menos brilhantes
        pixels_positive = img_norm[img_norm > 0]
        if len(pixels_positive) > 0:
            # Reduzir de 70º para 50º percentil para incluir células mais escuras
            intensity_threshold = np.percentile(pixels_positive, 50)  # 50º percentil
            # Ampliar range: entre 0.3 e 0.7 (antes 0.5-0.8) para detectar células escuras
            intensity_threshold = max(0.3, min(0.7, intensity_threshold))
        else:
            # Fallback: usar threshold mais baixo se não há pixels positivos
            intensity_threshold = 0.4  # Reduzido de 0.6
        
        coords_intensity = feature.peak_local_max(
            img_norm, 
            footprint=np.ones((7, 7)),  # Footprint maior para células grandes
            threshold_abs=intensity_threshold,  # Threshold adaptativo (mais baixo)
            min_distance=5  # Reduzido de 7 para 5 para detectar células próximas
        )
        local_maxi_intensity = np.zeros_like(img, dtype=bool)
        if len(coords_intensity) > 0:
            local_maxi_intensity[tuple(coords_intensity.T)] = True
        
        # Combinar marcadores: distance transform + intensidade
        local_maxi = local_maxi_dist.copy()
        # CORREÇÃO: Adicionar marcadores de intensidade mesmo que não estejam na máscara binária
        # Isso ajuda a detectar células escuras que podem ter sido perdidas na binarização
        local_maxi[local_maxi_intensity] = True
        # Também adicionar marcadores dentro da máscara binária
        local_maxi[bw & local_maxi_intensity] = True
        
        # MELHORIA: Usar bordas para melhorar marcadores (sempre que possível)
        # CORREÇÃO: Balancear entre detectar células escuras e evitar falsos positivos
        if edge_mask is not None and np.sum(edge_mask > 0) > 0 and max_dist > 3:  # Reduzido de 5 para 3 para detectar mais células
            # Encontrar máximos locais próximos às bordas (células têm bordas definidas)
            # Usar edge_mask que já foi dilatação das bordas
            # Procurar centros de células dentro das regiões delimitadas por bordas
            if use_edge_detection:
                # Modo agressivo: mais marcadores baseados em bordas
                # CORREÇÃO: Threshold mais permissivo para detectar células escuras
                edge_dilated = cv2.dilate(edges.astype(np.uint8), np.ones((9, 9), np.uint8))
                edge_maxima = feature.peak_local_max(
                    dist,
                    footprint=np.ones((7, 7)),
                    labels=(edge_dilated > 0).astype(bool),
                    min_distance=4,  # Reduzido de 6 para 4 para detectar células próximas
                    threshold_abs=max_dist * 0.2  # Threshold mais permissivo (20% ao invés de 30%)
                )
            else:
                # Modo conservador: marcadores mais seletivos baseados em bordas
                # CORREÇÃO: Threshold um pouco mais permissivo
                edge_maxima = feature.peak_local_max(
                    dist,
                    footprint=np.ones((5, 5)),
                    labels=(edge_mask > 0).astype(bool),
                    min_distance=5,  # Reduzido de 8 para 5
                    threshold_abs=max_dist * 0.25  # Threshold mais permissivo (25% ao invés de 40%)
                )
            
            if len(edge_maxima) > 0:
                local_maxi[tuple(edge_maxima.T)] = True
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
    # MELHORIA: Tornar mais flexível para células grandes (até 2x o máximo)
    min_ext = max(5, (2/3) * size_min)  # Mínimo absoluto de 5 pixels
    max_ext = max(size_max, (4/3) * size_max)  # Pelo menos o size_max
    # Permitir células grandes até 2x o máximo (para não perder células grandes importantes)
    max_ext_large = max(size_max * 2.0, max_ext * 1.5)
    
    props = measure.regionprops(labels)
    selected_mask = np.zeros(labels.shape, dtype=np.uint8)
    
    # CORREÇÃO: NÃO filtrar por posição nas bordas - permitir células cortadas
    height, width = labels.shape
    
    for prop in props:
        area = prop.area
        
        # Filtrar por intervalo estendido (mais flexível para células grandes)
        if area < min_ext:
            continue
        if area > max_ext_large:
            continue  # Células muito grandes são descartadas
        
        bbox = prop.bbox  # (min_row, min_col, max_row, max_col)
        bbox_height = bbox[2] - bbox[0]
        bbox_width = bbox[3] - bbox[1]
        
        # FILTRO CRÍTICO: Rejeitar linhas artificiais/alongadas ANTES de qualquer processamento
        # Células têm aspect ratio mais equilibrado, linhas são muito alongadas
        aspect_ratio = max(bbox_height, bbox_width) / max(1.0, min(bbox_height, bbox_width))
        
        # Se a região é muito alongada (aspect ratio > 4:1), provavelmente é uma linha
        # OU se é fina E alongada (dimensão menor < 8 pixels E aspect ratio > 3:1)
        is_elongated_line = aspect_ratio > 4.0
        is_thin_elongated = (min(bbox_height, bbox_width) < 8) and (aspect_ratio > 3.0)
        
        if is_elongated_line or is_thin_elongated:
            continue  # Descartar linhas artificiais imediatamente
        
        # CORREÇÃO: NÃO rejeitar células que tocam bordas - permitir células cortadas pela metade
        # Apenas rejeitar linhas artificiais muito óbvias (independente de tocar borda ou não)
        # Células cortadas nas bordas são válidas e devem ser detectadas
        
        # Score de tamanho (eq 3.1) - MELHORADO para células grandes
        if size_min <= area <= size_max:
            score_size = 1.0
        elif area > size_max:
            # Penalização mais suave para células grandes
            # Linear de 1.0 até 0.5 (quando área = 2×size_max)
            if area <= max_ext:
                # Dentro do intervalo estendido normal
                score_size = size_max / area
            else:
                # Fora do intervalo estendido mas ainda dentro do limite grande
                # Penalização mais suave: mantém score mínimo de 0.3 para células grandes válidas
                ratio = area / size_max
                if ratio <= 2.0:
                    score_size = max(0.3, size_max / area)  # Mínimo de 0.3
                else:
                    score_size = 0.1  # Muito grande, mas ainda considerar
        else:  # area < size_min
            score_size = area / size_min
        
        # Score de forma (ellipse fit, eq 3.2)
        coords = prop.coords
        score_shape = compute_ellipse_fit(coords)
        
        # Combinação ponderada
        # MELHORIA: Normalizar pesos se necessário para evitar scores artificiais
        total_weight = weight_size + weight_shape
        if total_weight > 0:
            normalized_weight_size = weight_size / total_weight
            normalized_weight_shape = weight_shape / total_weight
        else:
            normalized_weight_size = 0.5
            normalized_weight_shape = 0.5
        
        score = normalized_weight_size * score_size + normalized_weight_shape * score_shape
        
        # Seleção (threshold configurável)
        # MELHORIA: Threshold adaptativo - mais baixo para células grandes válidas
        # CORREÇÃO: Tornar threshold mais permissivo para detectar mais células
        threshold = config.ALC_SELECTION_THRESHOLD
        # CORREÇÃO: Reduzir threshold base para ser mais permissivo
        threshold = max(0.2, threshold - 0.05)  # Reduzir base de 0.3 para 0.25
        
        if area > size_max and area <= size_max * 2.0:
            # Células grandes: usar threshold ainda mais baixo (0.15 ao invés de 0.2)
            threshold = max(0.15, threshold - 0.05)
        elif area >= size_min:
            # Células dentro do tamanho ideal: manter threshold reduzido
            threshold = max(0.2, threshold)
        
        # CORREÇÃO ADICIONAL: Se a célula tem forma muito ruim (score_shape muito baixo),
        # mesmo que o score combinado seja alto, pode ser um falso positivo
        # CORREÇÃO: Tornar mais permissivo - só rejeitar se forma MUITO ruim E pequena
        # Células grandes podem ter forma menos ideal mas ainda serem válidas
        if score_shape < 0.2 and (area < size_min * 0.8):  # Apenas rejeitar se muito pequena E forma muito ruim
            # Forma muito ruim e muito pequena = provável falso positivo
            continue
        
        # FILTRO FINAL: Verificação adicional de aspect ratio usando propriedades regionais
        # Usar major_axis_length e minor_axis_length da regionprops para detectar linhas
        try:
            major_axis = prop.major_axis_length
            minor_axis = prop.minor_axis_length
            if minor_axis > 0:
                axis_ratio = major_axis / minor_axis
                # Se o eixo maior é muito maior que o menor (> 5:1), é provavelmente uma linha
                if axis_ratio > 5.0:
                    continue
        except:
            pass  # Se não conseguir calcular, usa critério de bbox que já aplicamos
        
        if score >= threshold:
            selected_mask[prop.coords[:, 0], prop.coords[:, 1]] = 255
    
    return selected_mask

