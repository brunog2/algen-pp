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
    # CORREÇÃO: Usar threshold mais alto para evitar falsos positivos de ruído
    max_dist = np.max(dist)
    # CORREÇÃO: Threshold ainda mais alto para evitar detecção de linhas/ruído
    dist_threshold = max(4.0, max_dist * 0.25)  # Threshold mínimo de 4 pixels ou 25% do máximo
    
    coords_dist = feature.peak_local_max(
        dist, 
        footprint=np.ones((5, 5)), 
        labels=bw, 
        min_distance=4,  # Aumentado de 3 para 4
        threshold_abs=dist_threshold  # Threshold para evitar ruído
    )
    local_maxi_dist = np.zeros_like(dist, dtype=bool)
    if len(coords_dist) > 0:
        local_maxi_dist[tuple(coords_dist.T)] = True
    
    # MELHORIA ADICIONAL: Se há poucos marcadores, usar threshold mais baixo
    # Isso ajuda a detectar células grandes que podem ter menos contrastes locais
    # MAS apenas se não detectou quase nenhum marcador (para evitar falsos positivos)
    # CORREÇÃO: Aumentar threshold mínimo do fallback
    if np.sum(local_maxi_dist) < 2 and max_dist > 8:  # Mais restritivo: < 2 ao invés de < 3, > 8 ao invés de > 5
        # Tentar com threshold mais baixo e footprint maior para células grandes
        coords_dist_large = feature.peak_local_max(
            dist, 
            footprint=np.ones((7, 7)), 
            labels=bw, 
            min_distance=6,  # Aumentado de 5 para 6
            threshold_abs=max_dist * 0.18  # Threshold mais baixo (18% do máximo) apenas se necessário
        )
        if len(coords_dist_large) > 0:
            local_maxi_dist[tuple(coords_dist_large.T)] = True
    
    # Marcadores baseados em intensidade local (detecta células brilhantes)
    if intensity_weight > 0:
        # Usar imagem original normalizada
        img_norm = exposure.rescale_intensity(img, out_range=(0, 1)).astype(np.float32)
        
        # Encontrar máximos locais de intensidade (células são mais brilhantes)
        # MELHORIA: Threshold adaptativo para detectar células grandes com intensidade variável
        # CORREÇÃO: Verificar se há pixels > 0 antes de calcular percentil
        pixels_positive = img_norm[img_norm > 0]
        if len(pixels_positive) > 0:
            intensity_threshold = np.percentile(pixels_positive, 70)  # 70º percentil
            intensity_threshold = max(0.5, min(0.8, intensity_threshold))  # Entre 0.5 e 0.8
        else:
            # Fallback: usar threshold padrão se não há pixels positivos
            intensity_threshold = 0.6
        
        coords_intensity = feature.peak_local_max(
            img_norm, 
            footprint=np.ones((7, 7)),  # Footprint maior para células grandes
            threshold_abs=intensity_threshold,  # Threshold adaptativo
            min_distance=7  # Distância mínima maior
        )
        local_maxi_intensity = np.zeros_like(img, dtype=bool)
        if len(coords_intensity) > 0:
            local_maxi_intensity[tuple(coords_intensity.T)] = True
        
        # Combinar marcadores: distance transform + intensidade
        local_maxi = local_maxi_dist.copy()
        # Adicionar marcadores de intensidade dentro da máscara binária
        local_maxi[bw & local_maxi_intensity] = True
        
        # MELHORIA: Usar bordas para melhorar marcadores (sempre que possível)
        # CORREÇÃO: Ser mais seletivo com marcadores baseados em bordas para evitar falsos positivos
        # CORREÇÃO ADICIONAL: Threshold ainda mais alto para evitar detecção de linhas
        if edge_mask is not None and np.sum(edge_mask > 0) > 0 and max_dist > 5:  # Aumentado de 3 para 5
            # Encontrar máximos locais próximos às bordas (células têm bordas definidas)
            # Usar edge_mask que já foi dilatação das bordas
            # Procurar centros de células dentro das regiões delimitadas por bordas
            if use_edge_detection:
                # Modo agressivo: mais marcadores baseados em bordas
                # MAS usar threshold mais alto para evitar falsos positivos
                edge_dilated = cv2.dilate(edges.astype(np.uint8), np.ones((9, 9), np.uint8))
                edge_maxima = feature.peak_local_max(
                    dist,
                    footprint=np.ones((7, 7)),
                    labels=(edge_dilated > 0).astype(bool),
                    min_distance=6,  # Aumentado de 5 para 6
                    threshold_abs=max_dist * 0.3  # Threshold ainda mais alto (30% ao invés de 25%)
                )
            else:
                # Modo conservador: marcadores mais seletivos baseados em bordas
                edge_maxima = feature.peak_local_max(
                    dist,
                    footprint=np.ones((5, 5)),
                    labels=(edge_mask > 0).astype(bool),
                    min_distance=8,  # Aumentado de 7 para 8
                    threshold_abs=max_dist * 0.4  # Threshold ainda mais alto (40% ao invés de 35%)
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
    
    # CORREÇÃO: Definir margem de borda (regiões próximas às bordas são suspeitas)
    border_margin = 5  # Pixels das bordas para considerar como "borda"
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
        
        # CORREÇÃO INTELIGENTE: Filtrar apenas linhas artificiais nas bordas, não células válidas
        # Uma célula válida que toca borda tem área e forma adequadas
        # Uma linha artificial é muito fina e alongada ao longo da borda
        # Verificar se toca bordas
        touches_top = bbox[0] < border_margin
        touches_bottom = bbox[2] > (height - border_margin)
        touches_left = bbox[1] < border_margin
        touches_right = bbox[3] > (width - border_margin)
        
        touches_border = touches_top or touches_bottom or touches_left or touches_right
        
        if touches_border:
            # FILTRO INTELIGENTE: Descarta apenas se for claramente uma linha artificial
            # Linhas artificiais são muito finas (altura ou largura < 10 pixels) E alongadas
            # OU são muito pequenas em área mas alongadas ao longo da borda
            is_thin_line = (min(bbox_height, bbox_width) < 10) and (aspect_ratio > 3.0)
            is_small_elongated = (area < 50) and (aspect_ratio > 2.5)
            
            # Se toca borda E é uma linha artificial, descarta
            # Mas se é uma célula válida (tem área adequada e forma razoável), mantém
            if is_thin_line or is_small_elongated:
                continue  # Descartar apenas linhas artificiais
            # Caso contrário, mantém a célula mesmo que toque a borda (é uma célula válida cortada)
        
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
        # CORREÇÃO: Tornar seleção mais seletiva para evitar falsos positivos
        threshold = config.ALC_SELECTION_THRESHOLD
        if area > size_max and area <= size_max * 2.0:
            # Células grandes: usar threshold um pouco mais baixo (0.25 ao invés de 0.3)
            threshold = max(0.2, threshold - 0.05)
        
        # CORREÇÃO ADICIONAL: Se a célula tem forma muito ruim (score_shape muito baixo),
        # mesmo que o score combinado seja alto, pode ser um falso positivo
        # Rejeitar células com score de forma muito baixo (< 0.3) e área não ideal
        if score_shape < 0.3 and (area < size_min or area > size_max * 1.5):
            # Forma muito ruim e tamanho fora do ideal = provável falso positivo
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

