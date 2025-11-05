# -*- coding: utf-8 -*-
"""
ALGEN-PP — Algoritmo Genético com Pós-processamento Aprendido
Baseado em Daguano (2020) — Extensão com refinamento morfológico aprendido.

Requisitos:
    pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from glob import glob


# ============================================================
# 1. PARÂMETROS GERAIS DO ALGORITMO GENÉTICO
# ============================================================

POP_SIZE = 20
NUM_GENERATIONS = 15
MUT_RATE = 0.1
ELITISM = 2

# Configuração do dataset
IMAGES_DIR = "./images_tif"
MAX_IMAGES = None  # None = todas as imagens, ou número para limitar (ex: 5 para testes rápidos)

# Intervalos dos genes (min, max)
GENE_BOUNDS = {
    "gaussian_sigma": (0.5, 2.5),
    "erosion": (1, 5),
    "dilation": (1, 5),
    "size_min": (50, 200),
    "size_max": (200, 500),
    "weight_size": (0.0, 1.0),
    "weight_shape": (0.0, 1.0),
    "closing_kernel": (1, 10),
    "merge_threshold": (0.05, 0.3),
    "min_area": (20, 200)
}


# ============================================================
# 2. FUNÇÕES AUXILIARES — GERAÇÃO E AVALIAÇÃO
# ============================================================

def gerar_individuo():
    """Cria um novo indivíduo com genes aleatórios dentro dos intervalos."""
    return {g: random.uniform(v[0], v[1]) for g, v in GENE_BOUNDS.items()}


def mutar(individuo):
    """Aplica mutação em alguns genes."""
    novo = individuo.copy()
    for g, (min_v, max_v) in GENE_BOUNDS.items():
        if random.random() < MUT_RATE:
            delta = (max_v - min_v) * 0.1
            novo[g] = np.clip(novo[g] + random.uniform(-delta, delta), min_v, max_v)
    return novo


def crossover(pai1, pai2):
    """Realiza cruzamento simples entre dois pais (média aritmética)."""
    filho = {}
    for g in GENE_BOUNDS.keys():
        filho[g] = (pai1[g] + pai2[g]) / 2.0
    return filho


# ============================================================
# 3. PIPELINE DE SEGMENTAÇÃO (Watershed + ALC simplificado)
# ============================================================

def watershed_ALC(image, params):
    """Etapa de segmentação simulando a abordagem Watershed + ALC."""
    # 1. Suavização
    blur = cv2.GaussianBlur(image, (5, 5), params["gaussian_sigma"])

    # 2. Operações morfológicas
    erosion = cv2.erode(blur, np.ones((int(params["erosion"]),) * 2, np.uint8))
    dilation = cv2.dilate(erosion, np.ones((int(params["dilation"]),) * 2, np.uint8))

    # 3. Limiarização adaptativa (simulação do watershed)
    _, thresh = cv2.threshold(dilation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Componentes conectados (representa regiões da ALC)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    segmented = np.zeros_like(image)

    # 5. Seleção de regiões por tamanho
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        if params["size_min"] <= area <= params["size_max"]:
            segmented[output == i] = 255

    return segmented


# ============================================================
# 4. PÓS-PROCESSAMENTO APRENDIDO
# ============================================================

def merge_adjacent_regions(image, threshold):
    """Função simplificada de fusão de regiões por intensidade média."""
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    diff = cv2.absdiff(image, blurred)
    _, merged = cv2.threshold(diff, int(threshold * 255), 255, cv2.THRESH_BINARY_INV)
    return merged


def pos_processamento_aprendido(seg, params):
    """Aplica o pós-processamento aprendido."""
    kernel_size = int(params["closing_kernel"])
    merge_thr = params["merge_threshold"]
    min_area = int(params["min_area"])

    # Fechamento morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel)

    # Remoção de pequenas regiões
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    filtered = np.zeros_like(seg)
    for i in range(1, nb_components):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[output == i] = 255

    # Fusão opcional
    refined = merge_adjacent_regions(filtered, merge_thr)
    return refined


# ============================================================
# 5. FUNÇÃO DE FITNESS (VALIDAÇÃO)
# ============================================================

def calcular_fitness(original, segmentada, params):
    """Avalia a qualidade da segmentação usando métrica Almod (como no artigo)."""
    # Métrica Almod: soma das diferenças absolutas pixel a pixel
    # Menor é melhor (quanto mais similar à original, melhor)
    seg_bin = (segmentada > 0).astype(np.uint8) * 255
    diff = np.abs(original.astype(np.int32) - seg_bin.astype(np.int32))
    almod = diff.sum()
    return almod


# ============================================================
# 6. LOOP PRINCIPAL DO ALGORITMO GENÉTICO
# ============================================================

def load_images_from_folder(folder, ext="tif", max_images=None):
    """Carrega imagens do dataset."""
    files = sorted(glob(os.path.join(folder, f"*.{ext}")))
    if max_images:
        files = files[:max_images]
    
    imgs = []
    names = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # Garantir grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(img)
        names.append(os.path.basename(f))
    return imgs, names


def algen_pp(images, names):
    """Executa o Algen-PP completo em múltiplas imagens (dataset completo)."""
    populacao = [gerar_individuo() for _ in range(POP_SIZE)]
    melhor_global = None
    melhor_fitness = float("inf")

    for geracao in range(NUM_GENERATIONS):
        avaliados = []

        print(f"\nGeração {geracao+1}/{NUM_GENERATIONS} - Avaliando {len(populacao)} indivíduos...")
        for idx, ind in enumerate(populacao):
            # Avaliar sobre todas as imagens
            total_fitness = 0.0
            for img, name in zip(images, names):
                seg = watershed_ALC(img, ind)
                seg_ref = pos_processamento_aprendido(seg, ind)
                fit = calcular_fitness(img, seg_ref, ind)
                total_fitness += fit
            
            # Fitness médio sobre todas as imagens
            mean_fitness = total_fitness / len(images)
            avaliados.append((ind, mean_fitness))
            if (idx + 1) % 5 == 0:
                print(f"  Indivíduos avaliados: {idx+1}/{len(populacao)}")

        # Ordena por fitness (menor é melhor)
        avaliados.sort(key=lambda x: x[1])
        populacao = [ind for ind, _ in avaliados]

        # Atualiza melhor global
        if avaliados[0][1] < melhor_fitness:
            melhor_global = avaliados[0][0].copy()
            melhor_fitness = avaliados[0][1]
            print(f"  ✓ Novo melhor global: {melhor_fitness:.2f}")
        else:
            print(f"  Melhor desta geração: {avaliados[0][1]:.2f} (global: {melhor_fitness:.2f})")

        # Seleção e reprodução
        nova_pop = populacao[:ELITISM].copy()
        while len(nova_pop) < POP_SIZE:
            pai1, pai2 = random.sample(populacao[:10], 2)
            filho = crossover(pai1, pai2)
            filho = mutar(filho)
            nova_pop.append(filho)

        populacao = nova_pop

    return melhor_global, melhor_fitness


# ============================================================
# 7. EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ALGEN-PP: Algoritmo Genético para Segmentação de Células")
    print("=" * 60)
    
    # Carregar imagens do dataset
    print(f"\nCarregando imagens de: {IMAGES_DIR}")
    images, names = load_images_from_folder(IMAGES_DIR, ext="tif", max_images=MAX_IMAGES)
    
    if len(images) == 0:
        print("ERRO: Nenhuma imagem encontrada!")
        print("Verifique se a pasta 'images_tif' existe e contém arquivos .tif")
        exit(1)
    
    print(f"✓ {len(images)} imagens carregadas")
    if MAX_IMAGES:
        print(f"  (Limitado a {MAX_IMAGES} imagens para teste rápido)")
    else:
        print(f"  (Todas as imagens do dataset serão processadas)")
    
    # Mostrar informações das imagens
    if len(images) > 0:
        print(f"\nInformações do dataset:")
        print(f"  - Dimensões: {images[0].shape}")
        print(f"  - Tipo: {images[0].dtype}")
        print(f"  - Primeira imagem: {names[0]}")
        if len(images) > 1:
            print(f"  - Última imagem: {names[-1]}")
    
    print("\n" + "=" * 60)
    print("Iniciando Algoritmo Genético...")
    print("=" * 60)
    
    # Executar GA
    best_params, best_fitness = algen_pp(images, names)
    
    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    print(f"\nMelhor fitness médio: {best_fitness:.2f}")
    print("\nMelhor conjunto de parâmetros encontrado:")
    for k, v in best_params.items():
        print(f"  {k:18s}: {v:.3f}")
    
    # Aplicar melhor segmentação em algumas imagens de exemplo
    print("\n" + "=" * 60)
    print("Aplicando melhor segmentação em imagens de exemplo...")
    print("=" * 60)
    
    os.makedirs("outputs/algen_2_pp_results", exist_ok=True)
    
    # Processar algumas imagens de exemplo (primeira, do meio, última)
    example_indices = [0]
    if len(images) > 1:
        example_indices.append(len(images) // 2)
    if len(images) > 2:
        example_indices.append(len(images) - 1)
    
    fig, axes = plt.subplots(len(example_indices), 2, figsize=(12, 4*len(example_indices)))
    if len(example_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_idx in enumerate(example_indices):
        img = images[img_idx]
        name = names[img_idx]
        
        # Aplicar pipeline completo
        seg = watershed_ALC(img, best_params)
        seg_ref = pos_processamento_aprendido(seg, best_params)
        
        # Salvar imagem segmentada
        output_path = f"outputs/algen_2_pp_results/{name.replace('.tif', '_segmented.png')}"
        cv2.imwrite(output_path, seg_ref)
        
        # Visualização
        axes[idx, 0].imshow(img, cmap="gray")
        axes[idx, 0].set_title(f"Original: {name}")
        axes[idx, 0].axis("off")
        
        axes[idx, 1].imshow(seg_ref, cmap="gray")
        axes[idx, 1].set_title(f"Segmentada (fitness: {calcular_fitness(img, seg_ref, best_params):.0f})")
        axes[idx, 1].axis("off")
        
        print(f"  ✓ Processada: {name}")
    
    plt.tight_layout()
    output_file = "outputs/algen_2_pp_results/comparison.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"\n✓ Comparação salva em: {output_file}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Processo concluído!")
    print("=" * 60)