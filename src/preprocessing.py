"""
Funções de pré-processamento de imagens.
"""

import cv2
import numpy as np


def preprocess_image(img, gaussian_sigma, median_ksize, erosion_size, dilation_size, use_morphological_gradient=True):
    """
    Pré-processamento da imagem conforme artigo Daguano (2020):
    1. Gaussian blur (suavização)
    2. Median filter (redução de ruído)
    3. Gradiente morfológico (dilatação - erosão) OU operações separadas
    
    O artigo menciona que o resultado final deve ser o gradiente morfológico
    obtido da diferença entre dilatação e erosão, o que realça bordas e diferencia
    objetos do background.
    
    Args:
        img: Imagem de entrada (numpy array)
        gaussian_sigma: Parâmetro sigma do filtro Gaussian
        median_ksize: Tamanho do kernel do filtro mediano
        erosion_size: Tamanho do kernel de erosão
        dilation_size: Tamanho do kernel de dilatação
        use_morphological_gradient: Se True, usa gradiente morfológico (artigo original)
                                   Se False, usa dilatação após erosão (versão alternativa)
    
    Returns:
        Imagem pré-processada
    """
    # Gaussian blur
    k = max(3, int(2*round(gaussian_sigma*2)+1))
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(img, (k, k), gaussian_sigma)
    
    # Median filter
    mks = median_ksize if median_ksize % 2 == 1 else median_ksize + 1
    if mks < 1:
        mks = 1
    medianed = cv2.medianBlur(blurred, mks)
    
    if use_morphological_gradient:
        # GRADIENTE MORFOLÓGICO (conforme artigo): dilatação - erosão
        # Isso realça as bordas e diferencia objetos do background
        if erosion_size > 0 and dilation_size > 0:
            # Usar o maior tamanho de kernel para consistência
            kernel_size = max(erosion_size, dilation_size)
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated = cv2.dilate(medianed, ker)
            eroded = cv2.erode(medianed, ker)
            # Gradiente morfológico = diferença entre dilatação e erosão
            morphological_gradient = cv2.subtract(dilated, eroded)
            return morphological_gradient
        elif dilation_size > 0:
            ker_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
            dilated = cv2.dilate(medianed, ker_d)
            eroded = medianed
            return cv2.subtract(dilated, eroded)
        elif erosion_size > 0:
            ker_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
            dilated = medianed
            eroded = cv2.erode(medianed, ker_e)
            return cv2.subtract(dilated, eroded)
        else:
            return medianed
    else:
        # VERSÃO ALTERNATIVA: erosão seguida de dilatação (abertura/fechamento)
        # Esta versão pode ser útil para comparação
        if erosion_size > 0:
            ker_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
            eroded = cv2.erode(medianed, ker_e)
        else:
            eroded = medianed
        
        if dilation_size > 0:
            ker_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
            dilated = cv2.dilate(eroded, ker_d)
        else:
            dilated = eroded
        
        return dilated

