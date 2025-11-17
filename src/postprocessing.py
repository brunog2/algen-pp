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
    
    # CORREÇÃO INTELIGENTE: Remover apenas linhas artificiais nas bordas, não células válidas
    # Isso evita falsos positivos nas bordas mas mantém células cortadas válidas
    border_margin = 5
    height, width = out.shape
    
    # Identificar e remover apenas linhas artificiais nas bordas
    labels_temp = measure.label(out, connectivity=2)
    props_temp = measure.regionprops(labels_temp)
    cleaned_borders = np.zeros_like(out)
    
    for p in props_temp:
        bbox = p.bbox
        bbox_height = bbox[2] - bbox[0]
        bbox_width = bbox[3] - bbox[1]
        
        touches_top = bbox[0] < border_margin
        touches_bottom = bbox[2] > (height - border_margin)
        touches_left = bbox[1] < border_margin
        touches_right = bbox[3] > (width - border_margin)
        touches_border = touches_top or touches_bottom or touches_left or touches_right
        
        if touches_border:
            # FILTRO INTELIGENTE: Descarta apenas se for claramente uma linha artificial
            is_thin_line = (min(bbox_height, bbox_width) < 10) and (max(bbox_height, bbox_width) > min(bbox_height, bbox_width) * 3)
            is_small_elongated = (p.area < 50) and (max(bbox_height, bbox_width) > min(bbox_height, bbox_width) * 2)
            
            # Se é uma linha artificial, descarta
            if is_thin_line or is_small_elongated:
                continue  # Descarta linhas artificiais
            # Caso contrário, mantém a célula (é válida mesmo que toque a borda)
        
        # Manter região válida
        cleaned_borders[labels_temp == p.label] = 255
    
    out = cleaned_borders
    
    # Refinamento iterativo
    for iteration in range(max(1, refinement_iterations + 1)):
        # Fechamento morfológico mais agressivo para garantir conectividade
        k = max(1, int(closing_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
        
        # MELHORIA: Fechamento adicional para garantir segmentações contínuas
        # Se houver muitos buracos ou descontinuidades, aplicar fechamento extra
        if iteration == refinement_iterations:
            # Fechamento adicional com kernel menor para suavizar contornos
            kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, k//2), max(1, k//2)))
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
            
            # FILTRO CRÍTICO: Rejeitar linhas artificiais/alongadas ANTES de qualquer verificação
            # Células têm aspect ratio mais equilibrado, linhas são muito alongadas
            is_elongated_line = aspect_ratio > 4.0
            is_thin_elongated = (min(bbox_height, bbox_width) < 8) and (aspect_ratio > 3.0)
            
            if is_elongated_line or is_thin_elongated:
                continue  # Descartar linhas artificiais imediatamente
            
            # Verificação adicional usando eixos principais (mais preciso)
            try:
                major_axis = p.major_axis_length
                minor_axis = p.minor_axis_length
                if minor_axis > 0:
                    axis_ratio = major_axis / minor_axis
                    if axis_ratio > 5.0:  # Eixo maior > 5x o menor = linha
                        continue
            except:
                pass
            
            # CORREÇÃO INTELIGENTE: Verificar se toca bordas - descartar apenas linhas artificiais
            touches_top = bbox[0] < border_margin
            touches_bottom = bbox[2] > (height - border_margin)
            touches_left = bbox[1] < border_margin
            touches_right = bbox[3] > (width - border_margin)
            touches_border = touches_top or touches_bottom or touches_left or touches_right
            
            if touches_border:
                # FILTRO INTELIGENTE: Descarta apenas linhas artificiais, mantém células válidas
                is_thin_line = (min(bbox_height, bbox_width) < 10) and (aspect_ratio > 3.0)
                is_small_elongated = (p.area < 50) and (aspect_ratio > 2.5)
                
                if is_thin_line or is_small_elongated:
                    continue  # Descarta apenas linhas artificiais
                # Caso contrário, mantém a célula válida
            
            filtered[labels == p.label] = 255
        out = filtered
        
        # MELHORIA: Usar bordas para refinar contornos das células detectadas
        # Se uma borda detectada está próxima de um contorno segmentado, ajustar
        if iteration == refinement_iterations:
            # CORREÇÃO: Remover bordas detectadas nas bordas da imagem antes de usar
            # Criar máscara para excluir bordas próximas às bordas da imagem
            edge_center_mask = np.zeros_like(edges_orig, dtype=np.uint8)
            edge_center_mask[border_margin:height-border_margin, border_margin:width-border_margin] = 255
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
                    
                    # CORREÇÃO INTELIGENTE: Remover apenas linhas artificiais que apareceram nas bordas
                    labels_border_check = measure.label(out, connectivity=2)
                    props_border_check = measure.regionprops(labels_border_check)
                    cleaned_after_edges = np.zeros_like(out)
                    
                    for p_bc in props_border_check:
                        bbox_bc = p_bc.bbox
                        bbox_h = bbox_bc[2] - bbox_bc[0]
                        bbox_w = bbox_bc[3] - bbox_bc[1]
                        
                        touches_bc = (bbox_bc[0] < border_margin or 
                                     bbox_bc[2] > (height - border_margin) or
                                     bbox_bc[1] < border_margin or 
                                     bbox_bc[3] > (width - border_margin))
                        
                        if touches_bc:
                            is_thin = (min(bbox_h, bbox_w) < 10) and (max(bbox_h, bbox_w) > min(bbox_h, bbox_w) * 3)
                            is_small = (p_bc.area < 50) and (max(bbox_h, bbox_w) > min(bbox_h, bbox_w) * 2)
                            if is_thin or is_small:
                                continue
                        
                        cleaned_after_edges[labels_border_check == p_bc.label] = 255
                    
                    out = cleaned_after_edges
        
        # Fusão de regiões adjacentes (apenas na última iteração ou se threshold > 0)
        if merge_threshold > 0 and (iteration == refinement_iterations):
            out = merge_adjacent_regions(out, orig_img, merge_threshold)
    
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
                    
                    touches_top = bbox[0] < border_margin
                    touches_bottom = bbox[2] > (height - border_margin)
                    touches_left = bbox[1] < border_margin
                    touches_right = bbox[3] > (width - border_margin)
                    touches_border = touches_top or touches_bottom or touches_left or touches_right
                    
                    # FILTRO INTELIGENTE: Mantém células válidas mesmo que toquem bordas
                    if touches_border:
                        is_thin = (min(bbox_h, bbox_w) < 10) and (max(bbox_h, bbox_w) > min(bbox_h, bbox_w) * 3)
                        is_small = (p.area < 50) and (max(bbox_h, bbox_w) > min(bbox_h, bbox_w) * 2)
                        if is_thin or is_small:
                            continue  # Descarta apenas linhas artificiais
                    
                    # Mantém célula válida (com ou sem tocar bordas)
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
        
        touches_top = bbox[0] < border_margin
        touches_bottom = bbox[2] > (height - border_margin)
        touches_left = bbox[1] < border_margin
        touches_right = bbox[3] > (width - border_margin)
        touches_border = touches_top or touches_bottom or touches_left or touches_right
        
        # FILTRO INTELIGENTE: Mantém células válidas mesmo que toquem bordas
        if touches_border:
            is_thin = (min(bbox_h, bbox_w) < 10) and (max(bbox_h, bbox_w) > min(bbox_h, bbox_w) * 3)
            is_small = (p.area < 50) and (max(bbox_h, bbox_w) > min(bbox_h, bbox_w) * 2)
            if is_thin or is_small:
                continue  # Descarta apenas linhas artificiais
        
        # Mantém célula válida
        cleaned[labels_final == p.label] = 255
    
    return cleaned

