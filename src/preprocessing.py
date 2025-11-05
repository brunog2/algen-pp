"""
Funções de pré-processamento de imagens.
"""

import cv2
import numpy as np


def preprocess_image(img, gaussian_sigma, median_ksize, erosion_size, dilation_size):
    """
    Pré-processamento da imagem:
    1. Gaussian blur
    2. Median filter
    3. Erosão
    4. Dilatação
    
    Args:
        img: Imagem de entrada (numpy array)
        gaussian_sigma: Parâmetro sigma do filtro Gaussian
        median_ksize: Tamanho do kernel do filtro mediano
        erosion_size: Tamanho do kernel de erosão
        dilation_size: Tamanho do kernel de dilatação
    
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
    
    # Erosão
    if erosion_size > 0:
        ker_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        eroded = cv2.erode(medianed, ker_e)
    else:
        eroded = medianed
    
    # Dilatação
    if dilation_size > 0:
        ker_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        dilated = cv2.dilate(eroded, ker_d)
    else:
        dilated = eroded
    
    return dilated

