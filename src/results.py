"""
Funções para salvar resultados e visualizações.
"""

import os
import cv2
import numpy as np
from skimage import exposure
import pipeline
import image_utils


def save_individual_results(individual, images, names, output_dir, generation=None, fitness=None):
    """
    Aplica um indivíduo a todas as imagens e salva os resultados.
    Se generation for fornecido, salva em subpasta da geração.
    
    Args:
        individual: Dicionário com parâmetros do indivíduo
        images: Lista de imagens
        names: Lista de nomes das imagens
        output_dir: Diretório de saída
        generation: Número da geração (opcional)
        fitness: Fitness do indivíduo (opcional)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    prefix = ""
    if generation is not None:
        prefix = f"gen{generation:02d}_"
        if fitness is not None:
            prefix += f"fit{int(fitness)}_"
    
    for idx, (img, name) in enumerate(zip(images, names)):
        # Aplicar pipeline
        seg_binary, _ = pipeline.segment_image(img, individual)
        seg_binary_uint8 = (seg_binary > 0).astype(np.uint8) * 255
        
        # Salvar segmentação binária
        base_name = os.path.splitext(name)[0]
        seg_filename = os.path.join(output_dir, f"{prefix}{base_name}_segmented.png")
        cv2.imwrite(seg_filename, seg_binary_uint8)
        
        # Criar imagem comparativa (original + contornos)
        img_normalized = image_utils.normalize_image_for_display(img)
        img_colored = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(seg_binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Desenhar contornos em verde
        cv2.drawContours(img_colored, contours, -1, (0, 255, 0), 2)
        
        # Salvar imagem comparativa
        comp_filename = os.path.join(output_dir, f"{prefix}{base_name}_comparison.png")
        cv2.imwrite(comp_filename, img_colored)
        
        # Criar imagem lado a lado (original | segmentada)
        img_side_by_side = np.hstack([img_normalized, seg_binary_uint8])
        side_filename = os.path.join(output_dir, f"{prefix}{base_name}_side_by_side.png")
        cv2.imwrite(side_filename, img_side_by_side)

