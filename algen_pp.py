"""
Algen-PP: implementação adaptada em Python
- pipeline: pré-processamento -> watershed (marker-based) -> seleção por tamanho/forma -> pós-processamento aprendido -> avaliação (Almod)
- algoritmo genético conforme pseudocódigo do artigo (Algen), incluindo genes de pós-processamento.
- Referências e inspiração: Daguano (2020) - "Algoritmo Genético para Segmentação..." (trechos consultados). 
"""

import os
import cv2
import numpy as np
from skimage import measure, morphology, filters, segmentation, feature, exposure
from scipy import ndimage as ndi
from glob import glob
from tqdm import tqdm
import random
import math
import pickle

# ---------- CONFIG ----------
IMAGES_DIR = "./images_tif"        # pasta onde estão as 69 .tif
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GA hyperparams (padrões baseados no artigo, ajustados para melhor diversidade)
POP_SIZE = 20
NUM_GENERATIONS = 30
MUTATION_CHANCE = 0.20          # 20% (aumentado de 10% para evitar estagnação)
MUTATION_FACTOR_RANGE = (0.70, 1.30)  # ±30% (aumentado de ±15% para maior exploração)
ELITISM = 2
DIVERSITY_REINJECTION_RATE = 0.15  # 15% chance de reintroduzir indivíduo aleatório a cada geração
DIVERSITY_REINJECTION_STAGNATION = 5  # Reintroduzir diversidade após N gerações sem melhoria

# Seeds / determinismo (opcional)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------- UTIL: leitura imagens ----------
def load_images_from_folder(folder, ext="tif"):
    files = sorted(glob(os.path.join(folder, f"*.{ext}")))
    imgs = []
    names = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # garantir grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(img)
        names.append(os.path.basename(f))
    return imgs, names

# ---------- MÉTRICAS (Almod, ellipse fit) ----------
def almod_metric(orig, seg_bin):
    """
    Almod = sum |I(i,j) - 255 * Seg(i,j)|
    where seg_bin is binary (0/1)
    menor é melhor
    """
    seg_img = (seg_bin > 0).astype(np.uint8) * 255
    diff = np.abs(orig.astype(np.int32) - seg_img.astype(np.int32))
    return diff.sum()

def compute_ellipse_fit(pixels_coords):
    """
    Implementa computeEllipseFit conforme pseudocódigo do artigo (momentos e axes)
    pixels_coords: Nx2 array of (row, col)
    retorna score_forma = area_objeto / area_elipse
    """
    if len(pixels_coords) < 5:
        return 0.0
    # coordenadas x,y (coluna, linha)
    xs = pixels_coords[:,1].astype(np.float64)
    ys = pixels_coords[:,0].astype(np.float64)
    m00 = len(xs)
    xs_sum = xs.sum()
    ys_sum = ys.sum()
    xxs = (xs**2).sum()
    yys = (ys**2).sum()
    xys = (xs*ys).sum()
    xsxs = xs_sum**2
    xsys = xs_sum * ys_sum
    ysys = ys_sum**2
    m02 = yys - (ysys/m00)
    m11 = xys - (xsys/m00)
    m20 = xxs - (xsxs/m00)
    M = np.array([[m02, m11],
                  [m11, m20]])
    # eigenvalues
    try:
        lambdas = np.linalg.eigvals(M)
    except Exception:
        return 0.0
    # ordem desc
    lambdas = np.sort(np.real(lambdas))[::-1]
    if lambdas[0] <= 0 or lambdas[1] <= 0:
        return 0.0
    a = 2.0 * math.sqrt(lambdas[0] / m00)
    b = 2.0 * math.sqrt(lambdas[1] / m00)
    area_ellipse = math.pi * a * b
    if area_ellipse <= 0:
        return 0.0
    score = m00 / area_ellipse
    # clip entre 0 e 1 (em teoria pode ultrapassar)
    return max(0.0, min(1.0, score))

# ---------- PRÉ-PROCESSAMENTO ----------
def preprocess_image(img, gaussian_sigma, median_ksize, erosion_size, dilation_size):
    """
    img: numpy grayscale
    gaussian_sigma: float
    median_ksize: int (odd)
    erosion_size/dilation_size: int (kernel radius)
    """
    # Gaussian blur
    k = max(3, int(2*round(gaussian_sigma*2)+1))
    if k % 2 == 0: k += 1
    blurred = cv2.GaussianBlur(img, (k,k), gaussian_sigma)
    # median
    mks = median_ksize if median_ksize % 2 == 1 else median_ksize+1
    if mks < 1: mks = 1
    medianed = cv2.medianBlur(blurred, mks)
    # morphological operations
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

# ---------- SEGMENTAÇÃO: Watershed + construção ALC-like ----------
def watershed_segmentation(img_pre):
    """
    Marker-based watershed using distance transform and watershed on inverted image gradient.
    Retorna label map (labels).
    """
    # Otsu for rough foreground
    img = img_pre
    if img.dtype != np.uint8:
        img = exposure.rescale_intensity(img, out_range=np.uint8).astype(np.uint8)

    val = filters.threshold_otsu(img)
    bw = img > val

    # distance transform
    dist = ndi.distance_transform_edt(bw)
    
    # === Correção para versões recentes do scikit-image ===
    coords = feature.peak_local_max(dist, footprint=np.ones((3,3)), labels=bw)
    local_maxi = np.zeros_like(dist, dtype=bool)
    local_maxi[tuple(coords.T)] = True
    markers = ndi.label(local_maxi)[0]

    # Watershed segmentation
    # Use negative distance transform (watershed fills from minima, we want maxima from distance transform)
    labels = segmentation.watershed(-dist, markers, mask=bw)

    # labels 0..N
    return labels

# ---------- SELEÇÃO POR TAMANHO/FORMA (ALC metrics approximated) ----------
def select_regions_by_size_shape(labels, size_min, size_max, size_ext_factor=(2/3, 4/3), weight_size=0.5, weight_shape=0.5):
    """
    Percorre regions do label map e calcula score_tamanho (eq 3.1) e score_forma (eq 3.2),
    produz máscara binária com regiões selecionadas (score ponderado).
    size_ext_factor define intervalo estendido [2/3*lim_inf, 4/3*lim_sup] conforme artigo.
    """
    min_ext = size_ext_factor[0] * size_min
    max_ext = size_ext_factor[1] * size_max

    props = measure.regionprops(labels)
    selected_mask = np.zeros(labels.shape, dtype=np.uint8)
    for prop in props:
        area = prop.area
        if not (min_ext <= area <= max_ext):
            continue  # descarta fora do intervalo estendido
        # score tamanho
        if (size_min <= area <= size_max):
            score_size = 1.0
        elif area >= size_max:
            score_size = size_max / area
        else:  # area < size_min
            score_size = area / size_min
        # score forma (ellipse fit)
        coords = prop.coords
        score_shape = compute_ellipse_fit(coords)
        score = weight_size * score_size + weight_shape * score_shape
        # threshold simples: considerar region se score > 0.5 (padrão) -- pode ser sintonizado
        if score >= 0.5:
            selected_mask[prop.coords[:,0], prop.coords[:,1]] = 255
    return selected_mask

# ---------- PÓS-PROCESSAMENTO APRENDIDO ----------
def merge_adjacent_regions(bin_mask, orig_img, merge_threshold):
    """
    Aproximação simples: rótulos conexos, para cada par adjacente comparar médias de intensidade
    Se diferença relativa < merge_threshold, funde (une rótulos).
    merge_threshold é fração (e.g., 0.1 = 10%).
    """
    labels = measure.label(bin_mask, connectivity=2)
    props = measure.regionprops(labels, intensity_image=orig_img)
    n = labels.max()
    if n <= 1:
        return bin_mask
    mean_intensities = np.zeros(n+1)
    for p in props:
        mean_intensities[p.label] = p.mean_intensity if p.mean_intensity is not None else 0.0
    # construir adjacência rápida via borda dilatada
    adjacency = {i:set() for i in range(1, n+1)}
    # dilate each region slightly and check overlap
    se = morphology.disk(1)
    for lab in range(1, n+1):
        region_mask = (labels == lab).astype(np.uint8)
        dil = cv2.dilate(region_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        overlap = np.unique(labels[(dil==1) & (labels!=lab)])
        for o in overlap:
            if o > 0:
                adjacency[lab].add(int(o))
                adjacency[o].add(int(lab))
    # agora decidir merges
    parent = list(range(n+1))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, neighs in adjacency.items():
        for b in neighs:
            if a < b:
                mi = mean_intensities[a]
                mj = mean_intensities[b]
                # evitar divisão por zero
                denom = max(1.0, max(abs(mi), abs(mj)))
                if abs(mi - mj) / denom <= merge_threshold:
                    union(a,b)
    # construir novo rótulo
    new_labels = np.zeros_like(labels)
    mapping = {}
    cur = 1
    for lab in range(1, n+1):
        root = find(lab)
        if root not in mapping:
            mapping[root] = cur
            cur += 1
        new_labels[labels==lab] = mapping[root]
    merged_mask = (new_labels > 0).astype(np.uint8) * 255
    return merged_mask

def post_processing_learned(seg_bin, orig_img, closing_kernel, merge_threshold, min_area):
    """
    Aplica: closing morfológico, remoção de regiões pequenas (min_area), fusão por similaridade.
    """
    out = seg_bin.copy()
    # fechamento
    k = max(1, int(closing_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    # remover regiões pequenas
    labels = measure.label(out, connectivity=2)
    props = measure.regionprops(labels)
    filtered = np.zeros_like(out)
    for p in props:
        if p.area >= min_area:
            filtered[labels==p.label] = 255
    out = filtered
    # fusão
    if merge_threshold is not None and merge_threshold > 0:
        out = merge_adjacent_regions(out, orig_img, merge_threshold)
    return out

# ---------- Cromossomo / genes ----------
def random_individual(param_ranges):
    """
    param_ranges: dict param -> (min, max, type) where type in {'int','float'}
    returns dict genes
    Genes:
    gaussian_sigma, median_ksize, erosion, dilation, size_min, size_max, weight_size, weight_shape,
    closing_kernel, merge_threshold, min_area
    """
    ind = {}
    for k, v in param_ranges.items():
        mn, mx, t = v
        if t == 'int':
            ind[k] = int(random.randint(mn, mx))
        else:
            ind[k] = float(random.uniform(mn, mx))
    return ind

# ---------- GA: avaliação de uma indivíduo sobre todas as imagens ----------
def evaluate_individual_on_dataset(ind, images, names, show_progress=False):
    """
    Executa pipeline para cada imagem: preprocess -> watershed -> select -> postprocessing -> Almod
    Retorna soma/mean das fitness (Almod) sobre dataset (menor melhor).
    Também retorna uma pasta com imagens segmentadas do indivíduo (opcional).
    """
    total = 0.0
    per_image = []
    iterator = zip(images, names)
    if show_progress:
        iterator = tqdm(iterator, total=len(images), desc="Avaliando imagens", leave=False)
    for img, name in iterator:
        pre = preprocess_image(img,
                               gaussian_sigma=ind['gaussian_sigma'],
                               median_ksize=int(ind['median_ksize']),
                               erosion_size=int(ind['erosion']),
                               dilation_size=int(ind['dilation']))
        labels = watershed_segmentation(pre)
        selected = select_regions_by_size_shape(labels,
                                               size_min=int(ind['size_min']),
                                               size_max=int(ind['size_max']),
                                               weight_size=ind['weight_size'],
                                               weight_shape=ind['weight_shape'])
        refined = post_processing_learned(selected, img,
                                          closing_kernel=int(ind['closing_kernel']),
                                          merge_threshold=float(ind['merge_threshold']),
                                          min_area=int(ind['min_area']))
        fit = almod_metric(img, (refined>0).astype(np.uint8))
        total += fit
        per_image.append((name, fit, refined))
    mean_fit = total / len(images)
    return mean_fit, per_image

# ---------- GA operators: selection, crossover, mutation ----------
def select_population(population, fitnesses):
    """
    Ordena por fitness (menor melhor), mantém metade melhor (descarta pior metade).
    Retorna survivors list.
    """
    paired = list(zip(population, fitnesses))
    paired.sort(key=lambda x: x[1])
    survivors = [p for p,f in paired[:len(paired)//2]]
    return survivors, paired[0]  # retorna melhor também

def crossover_avg(parent_a, parent_b, param_ranges):
    """
    Faz média simples entre genes (algoritmo do artigo).
    Para inteiros, arredonda.
    """
    child = {}
    for k,(mn,mx,t) in param_ranges.items():
        va = parent_a[k]
        vb = parent_b[k]
        val = (va + vb) / 2.0
        if t == 'int':
            val = int(round(val))
            val = max(mn, min(mx, val))
        else:
            val = max(mn, min(mx, val))
        child[k] = val
    return child

def mutate(ind, param_ranges, chance=MUTATION_CHANCE):
    """
    Mutação com chance ajustável e amplitude maior.
    Quando mutação é ativada, muta ~50% dos genes.
    """
    if random.random() > chance:
        return ind
    out = ind.copy()
    for k,(mn,mx,t) in param_ranges.items():
        if random.random() < 0.5:  # mutate ~50% genes when mutation triggered
            factor = random.uniform(MUTATION_FACTOR_RANGE[0], MUTATION_FACTOR_RANGE[1])
            if t == 'int':
                nv = int(round(out[k] * factor))
                nv = max(mn, min(mx, nv))
                out[k] = nv
            else:
                nv = out[k] * factor
                nv = max(mn, min(mx, nv))
                out[k] = nv
    return out

# ---------- Política de inicialização param ranges ----------
PARAM_RANGES = {
    'gaussian_sigma': (0.5, 2.5, 'float'),
    'median_ksize': (1, 5, 'int'),        # odd recommended
    'erosion': (0, 5, 'int'),
    'dilation': (0, 5, 'int'),
    'size_min': (20, 200, 'int'),
    'size_max': (80, 800, 'int'),
    'weight_size': (0.0, 1.0, 'float'),
    'weight_shape': (0.0, 1.0, 'float'),
    'closing_kernel': (1, 11, 'int'),
    'merge_threshold': (0.0, 0.3, 'float'),
    'min_area': (5, 200, 'int'),
}

# fix constraint: weight_size + weight_shape -> normalize later when used
def normalize_weights(ind):
    s = ind['weight_size'] + ind['weight_shape']
    if s == 0:
        ind['weight_size'] = 0.5
        ind['weight_shape'] = 0.5
    else:
        ind['weight_size'] /= s
        ind['weight_shape'] /= s

# ---------- SCRIPT PRINCIPAL: GA ----------
def run_algen_pp(images, names, save_best=True):
    # inicializa população
    population = [random_individual(PARAM_RANGES) for _ in range(POP_SIZE)]
    for ind in population:
        normalize_weights(ind)
    history = []
    best_global = None
    best_global_fit = float('inf')
    generations_without_improvement = 0
    for gen in tqdm(range(NUM_GENERATIONS), desc="Gerações", unit="geração"):
        print(f"\n=== Geração {gen+1}/{NUM_GENERATIONS} ===")
        fits = []
        # avaliar população
        for i, ind in enumerate(population):
            mean_fit, _ = evaluate_individual_on_dataset(ind, images, names, show_progress=False)
            fits.append(mean_fit)
            print(f"Ind {i:02d} fit={mean_fit:.2f}")
        # selecionar
        survivors, best_pair = select_population(population, fits)
        best_this, best_fit = best_pair
        print(f"Melhor desta geração fit={best_fit:.2f}")
        if best_fit < best_global_fit:
            best_global_fit = best_fit
            best_global = best_this.copy()
            generations_without_improvement = 0
            print(">> Novo best global encontrado")
            # salvar melhor temporariamente
            if save_best:
                with open(os.path.join(OUTPUT_DIR, "best_individual.pkl"), "wb") as f:
                    pickle.dump(best_global, f)
        else:
            generations_without_improvement += 1
        # criar novos indivíduos por crossover (média)
        new_pop = survivors.copy()
        # parear survivors: 1-ultimo, 2-penultimo...
        rem = len(population) - len(new_pop)
        surv_count = len(survivors)
        # gerar rem novos filhos por cruzamento
        while len(new_pop) < len(population):
            # Reintrodução de diversidade: ocasionalmente criar indivíduo aleatório
            if random.random() < DIVERSITY_REINJECTION_RATE:
                new_ind = random_individual(PARAM_RANGES)
                normalize_weights(new_ind)
                new_pop.append(new_ind)
            else:
                # pareamento determinístico: i with (surv_count-1-i)
                i = random.randrange(surv_count)
                j = surv_count - 1 - i
                parent_a = survivors[i]
                parent_b = survivors[j]
                child = crossover_avg(parent_a, parent_b, PARAM_RANGES)
                # mutação
                child = mutate(child, PARAM_RANGES)
                normalize_weights(child)
                new_pop.append(child)
        
        # Reintrodução forçada de diversidade após estagnação
        if generations_without_improvement >= DIVERSITY_REINJECTION_STAGNATION:
            # Substituir alguns dos piores indivíduos por aleatórios
            # Ordenar por fitness (usando os fits da geração anterior como referência)
            # Como não temos os novos fits, vamos substituir alguns aleatórios da parte inferior
            num_to_replace = max(1, len(new_pop) // 5)  # Substituir ~20% dos piores
            for _ in range(num_to_replace):
                # Substituir um dos últimos indivíduos (que são os novos filhos, geralmente piores)
                replace_idx = random.randint(len(survivors), len(new_pop) - 1)
                new_pop[replace_idx] = random_individual(PARAM_RANGES)
                normalize_weights(new_pop[replace_idx])
            generations_without_improvement = 0  # Reset counter
            print(f"  [Diversidade] Reintroduzidos {num_to_replace} indivíduos aleatórios após estagnação")
        
        # Elitismo: garantir que o melhor global esteja sempre na população
        # (substituir um dos novos indivíduos, não um survivor)
        if best_global is not None and len(new_pop) > len(survivors):
            # Substituir um dos novos indivíduos (não survivors) pelo melhor global
            # Isso garante que o melhor nunca seja perdido
            replace_idx = random.randint(len(survivors), len(new_pop) - 1)
            new_pop[replace_idx] = best_global.copy()
        
        population = new_pop
        history.append((gen, best_fit, best_this))
    print("\n=== FIM do GA ===")
    print("Melhor global fit:", best_global_fit)
    print("Melhor global genes:", best_global)
    # salvar best e history
    with open(os.path.join(OUTPUT_DIR, "algen_pp_history.pkl"), "wb") as f:
        pickle.dump({'best': best_global, 'fit': best_global_fit, 'history': history}, f)
    return best_global, best_global_fit

# ---------- Helper: aplicar melhor individuo e salvar segmentações ----------
def apply_and_save_best(best_ind, images, names):
    out_dir = os.path.join(OUTPUT_DIR, "best_segments")
    os.makedirs(out_dir, exist_ok=True)
    for img, name in zip(images, names):
        pre = preprocess_image(img,
                               gaussian_sigma=best_ind['gaussian_sigma'],
                               median_ksize=int(best_ind['median_ksize']),
                               erosion_size=int(best_ind['erosion']),
                               dilation_size=int(best_ind['dilation']))
        labels = watershed_segmentation(pre)
        selected = select_regions_by_size_shape(labels,
                                               size_min=int(best_ind['size_min']),
                                               size_max=int(best_ind['size_max']),
                                               weight_size=best_ind['weight_size'],
                                               weight_shape=best_ind['weight_shape'])
        refined = post_processing_learned(selected, img,
                                          closing_kernel=int(best_ind['closing_kernel']),
                                          merge_threshold=float(best_ind['merge_threshold']),
                                          min_area=int(best_ind['min_area']))
        # salvar overlay (original + contornos)
        contours, _ = cv2.findContours((refined>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = cv2.cvtColor(exposure.rescale_intensity(img, out_range=np.uint8).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, contours, -1, (0,255,0), 1)
        cv2.imwrite(os.path.join(out_dir, name.replace(".tif", "_seg.png")), vis)
    print("Segmentações salvas em:", out_dir)

# ---------- MAIN ----------
if __name__ == "__main__":
    imgs, names = load_images_from_folder(IMAGES_DIR, ext="tif")
    if len(imgs) == 0:
        print("Nenhuma imagem encontrada em:", IMAGES_DIR)
        raise SystemExit(1)
    print(f"{len(imgs)} imagens carregadas. Iniciando Algen-PP ...")
    best, fit = run_algen_pp(imgs, names)
    print("Aplicando melhor indivíduo ao dataset e salvando resultados...")
    apply_and_save_best(best, imgs, names)
    print("Pronto.")
