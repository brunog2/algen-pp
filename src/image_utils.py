"""
Utilitários para carregamento e manipulação de imagens.
"""

import os
import cv2
import numpy as np
from glob import glob
from skimage import exposure


def load_images_from_folder(folder, ext="tif"):
    """
    Carrega imagens da pasta especificada.
    
    Args:
        folder: Caminho da pasta
        ext: Extensão dos arquivos (padrão: "tif")
    
    Returns:
        tupla: (lista de imagens, lista de nomes)
    """
    files = sorted(glob(os.path.join(folder, f"*.{ext}")))
    imgs = []
    names = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(img)
        names.append(os.path.basename(f))
    return imgs, names


def normalize_image_for_display(img):
    """
    Normaliza imagem para uint8 para visualização.
    
    Args:
        img: Imagem numpy array
    
    Returns:
        Imagem normalizada em uint8
    """
    return exposure.rescale_intensity(img, out_range=np.uint8).astype(np.uint8)

