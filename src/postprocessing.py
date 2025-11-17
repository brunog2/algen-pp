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
    
    # Intensidade média e área de cada região
    mean_intensities = np.zeros(n + 1)
    region_areas = np.zeros(n + 1)
    for p in props:
        mean_intensities[p.label] = p.mean_intensity if p.mean_intensity is not None else 0.0
        region_areas[p.label] = p.area
    
    # Construir grafo de adjacência
    # CORREÇÃO: Usar dilatação maior para detectar regiões próximas que podem ser partes da mesma célula
    adjacency = {i: set() for i in range(1, n + 1)}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Aumentado de 3x3 para 5x5
    
    for lab in range(1, n + 1):
        region_mask = (labels == lab).astype(np.uint8)
        # Dilatar mais para detectar regiões próximas
        dil = cv2.dilate(region_mask, kernel, iterations=2)  # 2 iterações para detectar regiões próximas
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
    
    # Fusão baseada em intensidade, proximidade E tamanho
    # CORREÇÃO CRÍTICA: Tornar fusão mais agressiva para unir partes da mesma célula
    for a, neighs in adjacency.items():
        for b in neighs:
            if a < b:
                mi = mean_intensities[a]
                mj = mean_intensities[b]
                denom = max(1.0, max(abs(mi), abs(mj)))
                
                # CORREÇÃO: Se intensidades são muito similares, fundir mesmo com threshold mais baixo
                # Isso ajuda a unir partes da mesma célula que foram segmentadas separadamente
                intensity_diff = abs(mi - mj) / denom if denom > 0 else 1.0
                
                # CORREÇÃO ADICIONAL: Considerar tamanho das regiões
                # Regiões pequenas adjacentes com intensidade similar devem ser fundidas
                area_a = region_areas[a] if a < len(region_areas) else 0
                area_b = region_areas[b] if b < len(region_areas) else 0
                min_area = min(area_a, area_b)
                max_area = max(area_a, area_b)
                
                # Se pelo menos uma região é pequena (< 200 pixels) E intensidade similar, fundir
                # Isso ajuda a unir partes pequenas da mesma célula
                is_small_region = min_area < 200 or max_area < 300
                similar_intensity = intensity_diff <= merge_threshold * 2.0  # Threshold muito mais permissivo
                
                # CORREÇÃO: Aumentar threshold efetivo ainda mais para regiões pequenas
                effective_threshold = merge_threshold * 2.0 if is_small_region else merge_threshold * 1.5
                
                # Fundir se:
                # 1. Intensidade similar E threshold permite, OU
                # 2. Regiões pequenas E intensidade similar (mesmo com threshold mais alto)
                if intensity_diff <= effective_threshold or (is_small_region and similar_intensity):
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
    3. MELHORIA: Refinamento usando bordas para melhorar contornos
    4. Fusão de regiões adjacentes
    5. Refinamento iterativo (aplica etapas 1-4 múltiplas vezes se necessário)
    
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
    
    # MELHORIA: Detectar bordas da imagem original para refinar segmentação
    v = np.median(orig_img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges_orig = cv2.Canny(orig_img, lower, upper)
    
    # CORREÇÃO: NÃO remover células nas bordas - permitir células cortadas pela metade
    # Apenas remover linhas artificiais muito óbvias (independente de posição)
    # Células cortadas nas bordas são válidas e devem ser mantidas
    height, width = out.shape
    
    # Identificar e remover apenas linhas artificiais muito óbvias
    labels_temp = measure.label(out, connectivity=2)
    props_temp = measure.regionprops(labels_temp)
    cleaned_borders = np.zeros_like(out)
    
    for p in props_temp:
        bbox = p.bbox
        bbox_height = bbox[2] - bbox[0]
        bbox_width = bbox[3] - bbox[1]
        aspect_ratio = max(bbox_height, bbox_width) / max(1.0, min(bbox_height, bbox_width))
        
        # FILTRO: Rejeitar apenas linhas artificiais MUITO óbvias
        # Linhas muito finas (< 5 pixels) E muito alongadas (> 6:1)
        # OU muito pequenas (< 20 pixels) E muito alongadas (> 5:1)
        is_very_thin_line = (min(bbox_height, bbox_width) < 5) and (aspect_ratio > 6.0)
        is_very_small_elongated = (p.area < 20) and (aspect_ratio > 5.0)
        
        # Se é uma linha artificial MUITO óbvia, descarta
        # Caso contrário, mantém (incluindo células nas bordas)
        if is_very_thin_line or is_very_small_elongated:
            continue  # Descarta apenas linhas artificiais MUITO óbvias
        
        # Manter região válida (incluindo células nas bordas)
        cleaned_borders[labels_temp == p.label] = 255
    
    out = cleaned_borders
    
    # Refinamento iterativo
    for iteration in range(max(1, refinement_iterations + 1)):
        # Fechamento morfológico mais agressivo para garantir conectividade
        # CORREÇÃO: Usar kernel maior para conectar partes da mesma célula
        k = max(1, int(closing_kernel))
        # CORREÇÃO: Aumentar kernel em 50% para conectar melhor partes da célula
        k_large = max(k, int(k * 1.5))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_large, k_large))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
        
        # MELHORIA: Fechamento adicional para garantir segmentações contínuas
        # CORREÇÃO: Aplicar fechamento adicional em todas as iterações para conectar partes
        # Fechamento adicional com kernel menor para suavizar contornos e conectar partes próximas
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, k), max(1, k)))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel_smooth)
        
        # Remoção de regiões pequenas
        labels = measure.label(out, connectivity=2)
        props = measure.regionprops(labels)
        filtered = np.zeros_like(out)
        
        # CORREÇÃO: Filtrar também por posição (não tocar bordas)
        for p in props:
            # Verificar tamanho
            if p.area < min_area:
                continue
            
            bbox = p.bbox
            bbox_height = bbox[2] - bbox[0]
            bbox_width = bbox[3] - bbox[1]
            aspect_ratio = max(bbox_height, bbox_width) / max(1.0, min(bbox_height, bbox_width))
            
            # FILTRO: Rejeitar apenas linhas artificiais MUITO óbvias
            # Linhas muito finas (< 5 pixels) E muito alongadas (> 6:1)
            # OU muito pequenas (< 20 pixels) E muito alongadas (> 5:1)
            is_very_thin_line = (min(bbox_height, bbox_width) < 5) and (aspect_ratio > 6.0)
            is_very_small_elongated = (p.area < 20) and (aspect_ratio > 5.0)
            
            if is_very_thin_line or is_very_small_elongated:
                continue  # Descartar apenas linhas MUITO óbvias
            
            # Verificação adicional usando eixos principais (mais preciso)
            try:
                major_axis = p.major_axis_length
                minor_axis = p.minor_axis_length
                if minor_axis > 0:
                    axis_ratio = major_axis / minor_axis
                    # Só rejeitar se for MUITO alongado (> 7:1)
                    if axis_ratio > 7.0:
                        continue
            except:
                pass
            
            # CORREÇÃO: NÃO filtrar por posição nas bordas - permitir células cortadas
            # Manter todas as células válidas (incluindo nas bordas)
            
            filtered[labels == p.label] = 255
        out = filtered
        
        # MELHORIA: Usar bordas para refinar contornos das células detectadas
        # Se uma borda detectada está próxima de um contorno segmentado, ajustar
        if iteration == refinement_iterations:
            # CORREÇÃO: Filtrar apenas bordas muito próximas das bordas da imagem (1 pixel)
            # Isso evita usar bordas artificiais das bordas, mas permite células cortadas
            edge_margin = 1  # Apenas 1 pixel das bordas para evitar bordas artificiais
            edge_center_mask = np.zeros_like(edges_orig, dtype=np.uint8)
            edge_center_mask[edge_margin:height-edge_margin, edge_margin:width-edge_margin] = 255
            edges_orig_filtered = cv2.bitwise_and(edges_orig, edge_center_mask)
            
            # Encontrar contornos da segmentação
            contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Criar máscara de bordas próximas aos contornos
                # Dilatar segmentação ligeiramente e usar bordas para refinar
                out_dilated = cv2.dilate(out, np.ones((3, 3), np.uint8), iterations=1)
                
                # Se há bordas detectadas dentro da região dilatada (e não nas bordas da imagem), incluir
                # Isso ajuda a capturar bordas que podem ter sido perdidas
                edges_in_region = cv2.bitwise_and(edges_orig_filtered, out_dilated)
                
                # Aplicar operação morfológica para conectar bordas próximas à segmentação
                if np.sum(edges_in_region) > 0:
                    # Fechamento morfológico para conectar bordas próximas
                    edges_connected = cv2.morphologyEx(edges_in_region, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                    # Adicionar bordas refinadas à segmentação
                    out = cv2.bitwise_or(out, edges_connected)
                    
                    # Preencher buracos e garantir conectividade
                    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
                    
                    # CORREÇÃO: NÃO remover células nas bordas - apenas linhas artificiais muito óbvias
                    labels_border_check = measure.label(out, connectivity=2)
                    props_border_check = measure.regionprops(labels_border_check)
                    cleaned_after_edges = np.zeros_like(out)
                    
                    for p_bc in props_border_check:
                        bbox_bc = p_bc.bbox
                        bbox_h = bbox_bc[2] - bbox_bc[0]
                        bbox_w = bbox_bc[3] - bbox_bc[1]
                        aspect_ratio = max(bbox_h, bbox_w) / max(1.0, min(bbox_h, bbox_w))
                        
                        # Rejeitar apenas linhas MUITO óbvias (independente de posição)
                        is_very_thin = (min(bbox_h, bbox_w) < 5) and (aspect_ratio > 6.0)
                        is_very_small = (p_bc.area < 20) and (aspect_ratio > 5.0)
                        
                        if is_very_thin or is_very_small:
                            continue  # Descarta apenas linhas MUITO óbvias
                        
                        # Manter todas as células válidas (incluindo nas bordas)
                        cleaned_after_edges[labels_border_check == p_bc.label] = 255
                    
                    out = cleaned_after_edges
        
        # Fusão de regiões adjacentes (apenas na última iteração ou se threshold > 0)
        # CORREÇÃO: Aplicar fusão em todas as iterações, não apenas na última
        # Isso ajuda a unir partes da mesma célula que foram separadas
        if merge_threshold > 0:
            # Se threshold é muito baixo, aplicar fusão mais agressiva
            # Aumentar threshold efetivo se necessário para unir partes da célula
            effective_merge_threshold = max(merge_threshold, 0.15)  # Mínimo de 0.15 para fusão
            out = merge_adjacent_regions(out, orig_img, effective_merge_threshold)
            
            # CORREÇÃO ADICIONAL: Após fusão, aplicar fechamento para conectar partes próximas
            # Isso ajuda a unir partes da mesma célula que ainda estão separadas
            if iteration == refinement_iterations:
                # Fechamento adicional após fusão para conectar partes da mesma célula
                kernel_post_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k//2), max(3, k//2)))
                out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel_post_merge)
    
    # CORREÇÃO FINAL: Preencher buracos e garantir conectividade contínua
    # Para cada região detectada, preencher buracos internos
    labels_regions = measure.label(out, connectivity=2)
    filled = np.zeros_like(out)
    
    for region_id in range(1, labels_regions.max() + 1):
        region_mask = (labels_regions == region_id).astype(np.uint8) * 255
        
        if np.sum(region_mask) == 0:
            continue
        
        # Preencher buracos dentro desta região
        # Usar findContours com RETR_CCOMP para encontrar buracos
        contours, hierarchy = cv2.findContours(region_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0 and hierarchy is not None:
            # Preencher todos os contornos (externo + buracos)
            filled_region = np.zeros_like(region_mask)
            cv2.drawContours(filled_region, contours, -1, 255, -1)
            
            # Verificar se ainda não toca bordas após preencher
            labels_check = measure.label(filled_region, connectivity=2)
            props_check = measure.regionprops(labels_check)
            
            for p in props_check:
                if p.area >= min_area:
                    bbox = p.bbox
                    bbox_h = bbox[2] - bbox[0]
                    bbox_w = bbox[3] - bbox[1]
                    
                    # CORREÇÃO: NÃO filtrar por posição nas bordas - permitir células cortadas
                    # Apenas rejeitar linhas artificiais muito óbvias
                    aspect_ratio = max(bbox_h, bbox_w) / max(1.0, min(bbox_h, bbox_w))
                    
                    # Rejeitar apenas linhas MUITO óbvias
                    is_very_thin = (min(bbox_h, bbox_w) < 5) and (aspect_ratio > 6.0)
                    is_very_small = (p.area < 20) and (aspect_ratio > 5.0)
                    
                    if is_very_thin or is_very_small:
                        continue  # Descarta apenas linhas MUITO óbvias
                    
                    # Mantém célula válida (incluindo células cortadas nas bordas)
                    filled[labels_check == p.label] = 255
        else:
            # Se não há hierarquia, usar região como está
            filled[region_mask > 0] = 255
    
    # Aplicar fechamento final para garantir contornos contínuos e suaves
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_final)
    
    # CORREÇÃO FINAL: Remover apenas linhas artificiais nas bordas, manter células válidas
    # Limpeza final: remover regiões pequenas e linhas artificiais
    labels_final = measure.label(filled, connectivity=2)
    props_final = measure.regionprops(labels_final)
    cleaned = np.zeros_like(filled)
    
    for p in props_final:
        if p.area < min_area:
            continue
        
        bbox = p.bbox
        bbox_h = bbox[2] - bbox[0]
        bbox_w = bbox[3] - bbox[1]
        aspect_ratio = max(bbox_h, bbox_w) / max(1.0, min(bbox_h, bbox_w))
        
        # CORREÇÃO: NÃO filtrar por posição nas bordas - permitir células cortadas
        # Apenas rejeitar linhas artificiais muito óbvias
        is_very_thin = (min(bbox_h, bbox_w) < 5) and (aspect_ratio > 6.0)
        is_very_small = (p.area < 20) and (aspect_ratio > 5.0)
        
        if is_very_thin or is_very_small:
            continue  # Descarta apenas linhas MUITO óbvias
        
        # Mantém célula válida (incluindo células cortadas nas bordas)
        cleaned[labels_final == p.label] = 255
    
    return cleaned

