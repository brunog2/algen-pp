# Documentação Detalhada: Algen-2-PP

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Estrutura do Código](#estrutura-do-código)
3. [Parâmetros e Configurações](#parâmetros-e-configurações)
4. [Funções de Geração e Manipulação](#funções-de-geração-e-manipulação)
5. [Pipeline de Segmentação](#pipeline-de-segmentação)
6. [Pós-processamento Aprendido](#pós-processamento-aprendido)
7. [Função de Fitness](#função-de-fitness)
8. [Algoritmo Genético](#algoritmo-genético)
9. [Execução Principal](#execução-principal)
10. [Decisões de Design](#decisões-de-design)

---

## 🎯 Visão Geral

O **Algen-2-PP** é uma implementação simplificada do algoritmo Algen-PP (Algoritmo Genético para Segmentação de Imagens com Pós-processamento Aprendido), baseado na dissertação de Daguano (2020). O objetivo é segmentar automaticamente imagens biológicas (células) usando um algoritmo genético para otimizar parâmetros de segmentação.

### Diferença entre Algen-PP e Algen-2-PP

- **Algen-PP** (`algen_pp.py`): Implementação completa com Watershed real, métricas ALC completas, e todas as funcionalidades do artigo original.
- **Algen-2-PP** (`algen_2_pp.py`): Versão simplificada e mais rápida, ideal para testes e prototipagem, usando thresholding em vez de Watershed completo.

---

## 📁 Estrutura do Código

```python
# 1. Imports e dependências
# 2. Parâmetros do GA
# 3. Funções de geração e manipulação (indivíduos)
# 4. Pipeline de segmentação (Watershed simplificado)
# 5. Pós-processamento aprendido
# 6. Função de fitness
# 7. Loop principal do GA
# 8. Execução principal
```

---

## ⚙️ Parâmetros e Configurações

### Parâmetros do Algoritmo Genético

```python
POP_SIZE = 20
NUM_GENERATIONS = 15
MUT_RATE = 0.1
ELITISM = 2
```

**Por quê essas escolhas?**

1. **`POP_SIZE = 20`**:

   - Tamanho pequeno para execução rápida
   - Balance entre diversidade genética e tempo de processamento
   - Para dataset completo (69 imagens), 20 indivíduos já demandam ~20-30 minutos por geração

2. **`NUM_GENERATIONS = 15`**:

   - Número suficiente para convergência em problemas de segmentação
   - Evita overfitting aos dados de treinamento
   - Em testes, observamos melhorias significativas até ~10 gerações

3. **`MUT_RATE = 0.1` (10%)**:

   - Taxa moderada que mantém exploração sem ser muito disruptiva
   - Cada gene tem 10% de chance de mutar quando mutação é ativada
   - Evita convergência prematura

4. **`ELITISM = 2`**:
   - Mantém os 2 melhores indivíduos entre gerações
   - Garante que soluções boas não sejam perdidas
   - Permite que bons genes sejam passados adiante

### Configuração do Dataset

```python
IMAGES_DIR = "./images_tif"
MAX_IMAGES = None  # None = todas as imagens
```

**Por quê?**

- **`IMAGES_DIR`**: Caminho relativo para flexibilidade de deployment
- **`MAX_IMAGES = None`**: Permite processar todo o dataset (69 imagens) ou limitar para testes rápidos
- Quando `None`, carrega todas as imagens automaticamente
- Para testes rápidos, pode ser alterado para `MAX_IMAGES = 5` ou `10`

### Intervalos dos Genes (Parâmetros Otimizados)

```python
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
```

**Explicação de cada parâmetro:**

1. **`gaussian_sigma` (0.5 - 2.5)**:

   - Controla suavização da imagem (blur)
   - Valores baixos: menos suavização, mais detalhes (mas mais ruído)
   - Valores altos: mais suavização, menos detalhes (mas menos ruído)
   - **Escolha**: Baseado em testes empíricos com imagens de células

2. **`erosion` e `dilation` (1-5)**:

   - Operações morfológicas para remover ruído e suavizar bordas
   - Erosão remove pequenos objetos, dilatação restaura tamanho
   - **Escolha**: Valores pequenos (1-5) para não perder informações importantes

3. **`size_min` e `size_max` (50-200, 200-500)**:

   - Define faixa de tamanho esperado para células
   - Valores em pixels (área)
   - **Escolha**: Baseado em análise prévia das imagens do dataset

4. **`weight_size` e `weight_shape` (0.0-1.0)**:

   - Pesos para combinar métricas de tamanho e forma (não usados na versão simplificada)
   - Mantidos para compatibilidade futura
   - **Escolha**: Permite flexibilidade na combinação de métricas

5. **`closing_kernel` (1-10)**:

   - Tamanho do kernel para fechamento morfológico
   - Une descontinuidades e fecha buracos
   - **Escolha**: Valores pequenos para não distorcer formas das células

6. **`merge_threshold` (0.05-0.3)**:

   - Limiar para fusão de regiões adjacentes
   - Fração da diferença de intensidade média permitida
   - **Escolha**: Valores baixos para fusão conservadora

7. **`min_area` (20-200)**:
   - Área mínima para manter regiões após pós-processamento
   - Remove ruído e pequenos artefatos
   - **Escolha**: Baseado no tamanho mínimo esperado de células

---

## 🧬 Funções de Geração e Manipulação

### `gerar_individuo()`

```python
def gerar_individuo():
    """Cria um novo indivíduo com genes aleatórios dentro dos intervalos."""
    return {g: random.uniform(v[0], v[1]) for g, v in GENE_BOUNDS.items()}
```

**O que faz:**

- Cria um novo indivíduo (cromossomo) com valores aleatórios para cada gene
- Cada gene é um parâmetro do pipeline de segmentação
- Valores são gerados uniformemente dentro dos intervalos definidos

**Por quê essa abordagem?**

- **Uniforme**: Distribui aleatoriamente no espaço de busca, garantindo boa cobertura inicial
- **Simples**: Fácil de implementar e entender
- **Eficiente**: Geração rápida para população inicial

**Alternativa considerada:**

- Distribuição Gaussiana centrada: Rejeitada porque pode limitar exploração inicial

### `mutar(individuo)`

```python
def mutar(individuo):
    """Aplica mutação em alguns genes."""
    novo = individuo.copy()
    for g, (min_v, max_v) in GENE_BOUNDS.items():
        if random.random() < MUT_RATE:
            delta = (max_v - min_v) * 0.1
            novo[g] = np.clip(novo[g] + random.uniform(-delta, delta), min_v, max_v)
    return novo
```

**O que faz:**

- Para cada gene, há uma chance `MUT_RATE` de mutar
- Se mutar, adiciona/subtrai até 10% do intervalo total do gene
- Valores são limitados (clipped) para permanecer dentro dos bounds

**Por quê essa abordagem?**

- **Mudança incremental**: `delta = 10%` do intervalo permite mudanças significativas sem ser muito disruptiva
- **Por gene**: Cada gene pode mutar independentemente
- **Clipping**: Garante que valores permaneçam válidos

**Alternativa considerada:**

- Multiplicação por fator aleatório: Rejeitada porque pode causar mudanças muito grandes em valores pequenos

### `crossover(pai1, pai2)`

```python
def crossover(pai1, pai2):
    """Realiza cruzamento simples entre dois pais (média aritmética)."""
    filho = {}
    for g in GENE_BOUNDS.keys():
        filho[g] = (pai1[g] + pai2[g]) / 2.0
    return filho
```

**O que faz:**

- Cria um filho com genes que são a média aritmética dos dois pais
- Operação simples e determinística

**Por quê essa abordagem?**

- **Simples**: Fácil de implementar e entender
- **Suave**: Produz valores intermediários, explorando entre dois bons indivíduos
- **Comum**: Amplamente usada em algoritmos genéticos

**Alternativa considerada:**

- Crossover de ponto único: Rejeitado porque genes são independentes (não há ordem relevante)
- Crossover uniforme aleatório: Rejeitado porque a média é mais conservadora e eficiente

---

## 🔬 Pipeline de Segmentação

### `watershed_ALC(image, params)`

```python
def watershed_ALC(image, params):
    """Etapa de segmentação simulando a abordagem Watershed + ALC."""
    # 1. Suavização
    blur = cv2.GaussianBlur(image, (5, 5), params["gaussian_sigma"])

    # 2. Operações morfológicas
    erosion = cv2.erode(blur, np.ones((int(params["erosion"]),) * 2, np.uint8))
    dilation = cv2.dilate(erosion, np.ones((int(params["dilation"]),) * 2, np.uint8))

    # 3. Limiarização adaptativa
    _, thresh = cv2.threshold(dilation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Componentes conectados
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    segmented = np.zeros_like(image)

    # 5. Seleção de regiões por tamanho
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        if params["size_min"] <= area <= params["size_max"]:
            segmented[output == i] = 255

    return segmented
```

**O que faz (passo a passo):**

1. **Gaussian Blur**: Suaviza a imagem para reduzir ruído

   - Kernel fixo 5x5, sigma controlado pelo GA
   - **Por quê**: Remove ruído sem perder muito detalhe

2. **Erosão + Dilatação**: Operações morfológicas

   - Erosão remove pequenos objetos e ruído
   - Dilatação restaura tamanho (mas não restaura objetos removidos)
   - **Por quê**: Limpa a imagem e prepara para thresholding

3. **Threshold Otsu**: Binarização automática

   - Otsu escolhe automaticamente o melhor limiar
   - **Por quê**: Adaptativo, funciona bem com diferentes condições de iluminação

4. **Componentes Conectados**: Identifica regiões separadas

   - Conectividade 8 (inclui diagonais)
   - Calcula estatísticas (área, bounding box, etc.)
   - **Por quê**: Identifica objetos individuais na imagem binária

5. **Filtro por Tamanho**: Seleciona apenas células com tamanho adequado
   - Remove objetos muito pequenos (ruído) e muito grandes (aglomerados)
   - **Por quê**: Foca em células individuais do tamanho esperado

**Limitação desta implementação:**

- ❌ **Não usa Watershed real**: Apenas thresholding + componentes conectados
- ❌ **Não separa células sobrepostas**: Se duas células se tocam, são tratadas como uma
- ❌ **Não usa métricas de forma**: Apenas filtro por tamanho

**Por quê essa simplificação?**

- **Velocidade**: Muito mais rápido que Watershed completo
- **Simplicidade**: Mais fácil de entender e debugar
- **Adequado para**: Células bem separadas e imagens com pouco overlap

**Alternativa (Watershed real):**

- Implementado em `algen_pp.py` usando distance transform + peak local max + watershed
- Mais lento mas mais preciso para células sobrepostas

---

## 🎨 Pós-processamento Aprendido

### `merge_adjacent_regions(image, threshold)`

```python
def merge_adjacent_regions(image, threshold):
    """Função simplificada de fusão de regiões por intensidade média."""
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    diff = cv2.absdiff(image, blurred)
    _, merged = cv2.threshold(diff, int(threshold * 255), 255, cv2.THRESH_BINARY_INV)
    return merged
```

**O que faz:**

- Aplica blur para suavizar
- Calcula diferença absoluta entre original e suavizada
- Threshold baseado no parâmetro aprendido
- Inverte resultado (THRESH_BINARY_INV)

**Limitação:**

- Esta é uma implementação muito simplificada
- Não faz fusão real baseada em intensidade média entre regiões adjacentes
- Apenas uma aproximação do comportamento desejado

**Por quê essa simplificação?**

- **Rapidez**: Implementação rápida
- **Prototipagem**: Para validar o conceito de pós-processamento aprendido
- **Alternativa completa**: Implementada em `algen_pp.py` com análise de adjacência real

### `pos_processamento_aprendido(seg, params)`

```python
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
```

**O que faz (passo a passo):**

1. **Fechamento Morfológico (Closing)**:

   - Une descontinuidades e fecha buracos
   - Kernel elíptico (melhor para objetos circulares como células)
   - Tamanho controlado pelo GA
   - **Por quê**: Suaviza bordas e conecta partes desconectadas da mesma célula

2. **Remoção de Regiões Pequenas**:

   - Identifica componentes conectados
   - Remove aqueles com área < `min_area`
   - **Por quê**: Remove ruído residual e pequenos artefatos

3. **Fusão de Regiões Adjacentes**:
   - Aplica função de fusão (simplificada)
   - **Por quê**: Une regiões que foram divididas incorretamente

**Por quê esses três passos?**

- **Closing**: Melhora qualidade da segmentação inicial
- **Filtro de área**: Remove ruído
- **Fusão**: Corrige oversegmentação

**Parâmetros aprendidos:**

- `closing_kernel`: Tamanho ideal para fechamento
- `min_area`: Área mínima para remover ruído
- `merge_threshold`: Quando fundir regiões

---

## 📊 Função de Fitness

### `calcular_fitness(original, segmentada, params)`

```python
def calcular_fitness(original, segmentada, params):
    """Avalia a qualidade da segmentação usando métrica Almod."""
    seg_bin = (segmentada > 0).astype(np.uint8) * 255
    diff = np.abs(original.astype(np.int32) - seg_bin.astype(np.int32))
    almod = diff.sum()
    return almod
```

**O que faz:**

- Converte segmentação para binária (0 ou 255)
- Calcula diferença absoluta pixel a pixel com imagem original
- Soma todas as diferenças (métrica Almod)

**Interpretação:**

- **Menor = Melhor**: Menos diferença significa segmentação mais similar à original
- **Almod**: Métrica do artigo original de Daguano (2020)
- Unidade: Soma de diferenças de intensidade (pixels)

**Por quê apenas Almod?**

- **Simplicidade**: Métrica direta e fácil de calcular
- **Rapidez**: Avaliação muito rápida
- **Artigo original**: Usada no trabalho de referência

**Limitação:**

- ❌ Não considera qualidade da segmentação (tamanho/forma das células)
- ❌ Pode favorecer segmentações que apenas minimizam diferença
- ❌ Não valida se células têm tamanho/formato adequados

**Alternativa completa (em `algen_pp.py`):**

```python
# Inclui:
- Score de tamanho (ALC)
- Score de forma (ellipse fit)
- Combinação ponderada com Almod
```

**Por quê essa simplificação?**

- **Velocidade**: Avaliação muito mais rápida
- **Foco**: Para testes rápidos, Almod é suficiente
- **Simplicidade**: Mais fácil de entender e debugar

---

## 🧪 Algoritmo Genético

### `load_images_from_folder(folder, ext="tif", max_images=None)`

```python
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
```

**O que faz:**

- Busca todos os arquivos `.tif` na pasta
- Ordena por nome (garante ordem consistente)
- Limita quantidade se `max_images` especificado
- Carrega cada imagem e converte para grayscale se necessário
- Retorna lista de imagens e nomes

**Por quê essas escolhas?**

- **Ordenação**: Garante reprodutibilidade
- **Limitação opcional**: Permite testes rápidos
- **Conversão automática**: Garante que todas sejam grayscale
- **Skip de erros**: Continua se alguma imagem falhar

### `algen_pp(images, names)`

```python
def algen_pp(images, names):
    """Executa o Algen-PP completo em múltiplas imagens."""
    populacao = [gerar_individuo() for _ in range(POP_SIZE)]
    melhor_global = None
    melhor_fitness = float("inf")

    for geracao in range(NUM_GENERATIONS):
        # Avaliar população
        avaliados = []
        for idx, ind in enumerate(populacao):
            total_fitness = 0.0
            for img, name in zip(images, names):
                seg = watershed_ALC(img, ind)
                seg_ref = pos_processamento_aprendido(seg, ind)
                fit = calcular_fitness(img, seg_ref, ind)
                total_fitness += fit
            mean_fitness = total_fitness / len(images)
            avaliados.append((ind, mean_fitness))

        # Seleção
        avaliados.sort(key=lambda x: x[1])
        populacao = [ind for ind, _ in avaliados]

        # Atualizar melhor global
        if avaliados[0][1] < melhor_fitness:
            melhor_global = avaliados[0][0].copy()
            melhor_fitness = avaliados[0][1]

        # Reprodução
        nova_pop = populacao[:ELITISM].copy()
        while len(nova_pop) < POP_SIZE:
            pai1, pai2 = random.sample(populacao[:10], 2)
            filho = crossover(pai1, pai2)
            filho = mutar(filho)
            nova_pop.append(filho)

        populacao = nova_pop

    return melhor_global, melhor_fitness
```

**Fluxo do algoritmo (passo a passo):**

1. **Inicialização**:

   - Gera população inicial aleatória
   - Inicializa melhor global como infinito

2. **Para cada geração**:

   a. **Avaliação**:

   - Para cada indivíduo, avalia em TODAS as imagens
   - Calcula fitness médio sobre o dataset
   - **Por quê média?**: Garante que parâmetros funcionem bem em todas as imagens

   b. **Seleção**:

   - Ordena por fitness (menor = melhor)
   - Mantém todos os indivíduos ordenados
   - **Por quê manter todos?**: Preserva diversidade

   c. **Elitismo**:

   - Mantém os `ELITISM` melhores
   - **Por quê?**: Garante que soluções boas não sejam perdidas

   d. **Reprodução**:

   - Seleciona pais aleatórios dos top 10
   - **Por quê top 10?**: Balance entre qualidade e diversidade
   - Cria filhos via crossover
   - Aplica mutação
   - Preenche população até `POP_SIZE`

3. **Retorno**:
   - Melhor conjunto de parâmetros encontrado
   - Melhor fitness médio

**Por quê essa estrutura?**

- **Avaliação sobre dataset completo**: Garante generalização
- **Fitness médio**: Evita overfitting a imagens específicas
- **Elitismo**: Preserva soluções boas
- **Top 10 para reprodução**: Balance entre exploração e exploração

**Alternativas consideradas:**

- **Tournament selection**: Rejeitada por simplicidade (atual é mais direta)
- **Avaliação apenas em subset**: Rejeitada para garantir generalização

---

## 🚀 Execução Principal

### Estrutura do `if __name__ == "__main__"`

```python
# 1. Carregar imagens
images, names = load_images_from_folder(...)

# 2. Executar GA
best_params, best_fitness = algen_pp(images, names)

# 3. Aplicar melhor segmentação em exemplos
# 4. Salvar resultados
```

**O que faz:**

1. **Carregamento**:

   - Carrega todas ou subset de imagens
   - Mostra informações do dataset

2. **Execução do GA**:

   - Roda algoritmo genético completo
   - Otimiza parâmetros sobre dataset completo

3. **Aplicação e Visualização**:
   - Seleciona imagens de exemplo (primeira, meio, última)
   - Aplica melhor segmentação
   - Salva resultados em `outputs/algen_2_pp_results/`
   - Cria comparação visual

**Por quê processar apenas exemplos no final?**

- **Economia**: Não precisa processar todas as 69 imagens novamente
- **Visualização**: 3 exemplos são suficientes para validar
- **Tempo**: Processamento completo pode ser feito depois se necessário

---

## 🎯 Decisões de Design

### 1. Por quê versão simplificada?

**Razões:**

- **Prototipagem rápida**: Testar conceitos rapidamente
- **Debugging fácil**: Código mais simples = mais fácil de debugar
- **Validação**: Validar pipeline básico antes de implementação completa
- **Performance**: Para testes rápidos, simplificação é suficiente

**Trade-offs:**

- ❌ Menos preciso (sem Watershed real)
- ❌ Métricas incompletas (sem forma)
- ✅ Muito mais rápido
- ✅ Mais fácil de entender

### 2. Por quê processar todas as imagens?

**Razões:**

- **Generalização**: Parâmetros devem funcionar bem em todas as imagens
- **Robustez**: Evita overfitting a imagens específicas
- **Realismo**: Simula uso real do algoritmo

**Trade-offs:**

- ❌ Muito mais lento (69 imagens × 20 indivíduos × 15 gerações)
- ✅ Parâmetros mais robustos
- ✅ Melhor para produção

### 3. Por quê fitness médio?

**Razões:**

- **Balance**: Não favorece nenhuma imagem específica
- **Robustez**: Parâmetros funcionam bem em média
- **Simplicidade**: Fácil de calcular e interpretar

**Alternativa considerada:**

- Fitness mínimo: Rejeitada porque pode ser muito restritivo
- Fitness ponderado: Rejeitada por simplicidade

### 4. Por quê elitismo pequeno (2)?

**Razões:**

- **Diversidade**: Permite mais exploração
- **Evita convergência prematura**: Não força população muito cedo
- **Balance**: Mantém soluções boas sem dominar

**Alternativa considerada:**

- Elitismo maior (5-10): Rejeitada porque pode causar convergência prematura

### 5. Por quê crossover por média?

**Razões:**

- **Simplicidade**: Fácil de implementar
- **Suave**: Explora região entre dois bons indivíduos
- **Eficiente**: Não requer parâmetros adicionais

**Alternativa considerada:**

- Crossover uniforme: Rejeitada porque média é mais conservadora

---

## 📈 Conclusão

O **Algen-2-PP** é uma implementação **simplificada mas funcional** do algoritmo de segmentação genética. Foi projetado para:

1. ✅ **Testes rápidos**: Validar conceitos e pipeline
2. ✅ **Prototipagem**: Desenvolver e testar ideias
3. ✅ **Aprendizado**: Entender como funciona o algoritmo

**Para produção**, recomenda-se usar o **`algen_pp.py`** que tem:

- Watershed real completo
- Métricas ALC completas (tamanho + forma)
- Função de fitness mais robusta
- Pipeline completo conforme artigo original

**O Algen-2-PP serve como:**

- Ponto de partida para desenvolvimento
- Validação rápida de hipóteses
- Base para extensões futuras

---

## 📚 Referências

- **Daguano, E. M. (2020)**: "Algoritmo Genético para Segmentação de Imagens utilizando Tamanho e Forma dos Objetos" - UNICAMP
- **OpenCV Documentation**: Operações morfológicas e processamento de imagens
- **scikit-image**: Watershed segmentation e métricas de forma

---

_Documentação gerada em: 2024_
_Versão do código: Algen-2-PP v1.0_
