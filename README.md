# Algen-PP: Algoritmo Genético para Segmentação de Imagens

Algoritmo genético para segmentação automática de imagens biológicas (células) usando otimização de parâmetros de processamento de imagem.

## Descrição

Este projeto implementa um algoritmo genético melhorado baseado no trabalho de Daguano (2020) para segmentação de imagens de células. O algoritmo otimiza automaticamente parâmetros de um pipeline de segmentação que inclui:

- Pré-processamento (Gaussian blur, filtros morfológicos)
- Segmentação Watershed híbrida (distance transform + intensidade)
- Seleção por tamanho e forma (métricas ALC)
- Pós-processamento aprendido (refinamento iterativo)
- Avaliação combinada (Almod + qualidade de forma)

## Características Principais

### Melhorias em relação ao algoritmo original:

1. **Watershed Híbrido**: Combina marcadores baseados em distance transform e intensidade local
2. **Fitness Combinada**: Almod (85%) + Qualidade de Forma (15%) + Recompensa por células
3. **Seleção por Torneio**: Maior diversidade genética
4. **Crossover BLX-alpha**: Melhor exploração do espaço de busca
5. **Refinamento Adaptativo**: Pós-processamento iterativo (0-2 iterações)
6. **Mecanismos Anti-Estagnação**: Reinjeção de diversidade e mutação aumentada (50%)

## Requisitos

```bash
pip install opencv-python numpy scikit-image scipy matplotlib tqdm
```

## Instalação

1. Clone o repositório
2. Crie um ambiente virtual (recomendado):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate  # Windows
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
   (ou instale manualmente os pacotes listados acima)

## Uso

### Execução básica:

```bash
cd src
python3 main.py
```

### Configuração:

1. Coloque suas imagens `.tif` na pasta `images/` (na raiz do projeto)
2. Ajuste os parâmetros do GA no arquivo `src/config.py` se necessário:
   ```python
   POP_SIZE = 20                    # Tamanho da população
   NUM_GENERATIONS = 100            # Número de gerações
   MUTATION_RATE = 0.50             # Taxa de mutação (50%)
   ELITISM = 2                      # Número de melhores preservados
   ```
3. Execute o algoritmo

## Estrutura do Projeto

```
.
├── src/                           # Código fonte
│   ├── main.py                    # Script principal
│   ├── config.py                  # Configurações e parâmetros
│   ├── image_utils.py             # Utilitários de imagem
│   ├── preprocessing.py           # Pré-processamento
│   ├── segmentation.py            # Watershed e seleção ALC
│   ├── postprocessing.py          # Pós-processamento aprendido
│   ├── metrics.py                 # Métricas de avaliação
│   ├── genetic_algorithm.py       # Operadores do GA
│   ├── pipeline.py                # Pipeline completo
│   ├── ga_runner.py               # Executor do GA
│   ├── logger_utils.py            # Utilitários de logging
│   └── results.py                 # Salvamento de resultados
├── images/                        # Imagens de entrada (.tif)
├── outputs/                       # Resultados gerados
│   ├── generation_results/        # Imagens de cada geração
│   ├── algen_basic_results/       # Resultado final
│   └── logs/                      # Logs e histórico JSON
├── .gitignore                     # Arquivos ignorados pelo Git
└── README.md                      # Este arquivo
```

## Parâmetros Otimizados

O algoritmo genético otimiza os seguintes parâmetros:

### Pré-processamento:

- `gaussian_sigma`: Parâmetro sigma do filtro Gaussian (0.5 - 2.5)
- `median_ksize`: Tamanho do kernel do filtro mediano (1 - 5)
- `erosion`: Tamanho do kernel de erosão (0 - 5)
- `dilation`: Tamanho do kernel de dilatação (0 - 5)

### Watershed:

- `intensity_weight`: Peso para marcadores baseados em intensidade (0.0 - 1.0)

### Seleção ALC:

- `size_min`: Tamanho mínimo de células (20 - 200)
- `size_max`: Tamanho máximo de células (80 - 800)
- `weight_size`: Peso do score de tamanho (0.0 - 1.0)
- `weight_shape`: Peso do score de forma (0.0 - 1.0)

### Pós-processamento:

- `closing_kernel`: Tamanho do kernel de fechamento (1 - 11)
- `merge_threshold`: Threshold de fusão de regiões (0.0 - 0.3)
- `min_area`: Área mínima para manter região (5 - 200)
- `refinement_iterations`: Número de iterações de refinamento (0 - 2)

## Resultados

O algoritmo gera automaticamente:

### 1. Logs detalhados:

- Arquivo de log completo em `outputs/logs/algen_evolution_YYYYMMDD_HHMMSS.log`
- Histórico JSON estruturado em `outputs/logs/algen_history_YYYYMMDD_HHMMSS.json`
- Informações de cada geração: fitness, estatísticas, parâmetros

### 2. Imagens por geração:

- Cada geração tem sua própria pasta em `outputs/generation_results/YYYYMMDD_HHMMSS/generation_XX/`
- Para cada imagem: segmentação binária, comparação com contornos, lado a lado
- Permite visualizar a evolução da segmentação

### 3. Resultado final:

- Melhor segmentação encontrada em `outputs/algen_basic_results/final/`
- Parâmetros otimizados salvos no log

## Diferenciais do Algoritmo

### Problemas resolvidos:

1. **Convergência prematura**:

   - Mutação aumentada para 50%
   - Amplitude de mutação ±30%
   - Reinjeção automática de diversidade após 5 gerações sem melhoria

2. **Baixa cobertura de células**:

   - Métrica Almod normalizada (não penaliza mais células)
   - Threshold de seleção reduzido de 0.5 para 0.3
   - Recompensa por número de células detectadas

3. **Estagnação**:
   - Mecanismos automáticos de reinjeção
   - Reinjeção ocasional de 20% de indivíduos aleatórios
   - Seleção por torneio para maior diversidade

### Métricas ajustadas:

- **Almod normalizado**: `(média_diferença_por_pixel) × sqrt(área)` - não penaliza segmentações com mais células
- **Threshold de seleção**: 0.3 (permite mais células válidas serem selecionadas)
- **Fitness combinada**: Almod (85%) + Qualidade Forma (15%) + Penalização células (10%)

## Arquitetura Modular

O código está organizado em módulos especializados:

- **config.py**: Todas as configurações centralizadas
- **preprocessing.py**: Funções de pré-processamento
- **segmentation.py**: Watershed e seleção ALC
- **postprocessing.py**: Pós-processamento aprendido
- **metrics.py**: Cálculo de métricas (Almod, qualidade de forma, ellipse fit)
- **genetic_algorithm.py**: Operadores do GA (crossover, mutação, seleção)
- **pipeline.py**: Pipeline completo de segmentação
- **ga_runner.py**: Execução do algoritmo genético
- **results.py**: Salvamento de resultados e visualizações
- **logger_utils.py**: Sistema de logging

## Dicas

- **Testes rápidos**: Reduza `NUM_GENERATIONS` e `POP_SIZE` em `src/config.py`
- **Ajuste de mutação**: Se convergir muito rápido, aumente `MUTATION_RATE`
- **Visualização**: Explore as imagens em `outputs/generation_results/` para acompanhar a evolução
- **Análise**: Use os arquivos JSON em `outputs/logs/` para análise estatística
- **Comparação**: Compare visualmente as gerações para ver a melhoria da segmentação

## Referências

- **Daguano, E. M. (2020)**: "Algoritmo Genético para Segmentação de Imagens utilizando Tamanho e Forma dos Objetos" - UNICAMP

## Licença

Este projeto é para fins acadêmicos e de pesquisa.
