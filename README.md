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

- Python 3.8 ou superior
- Dependências listadas em `requirements.txt`

## Instalação e Execução Rápida

### Método Automatizado (Recomendado)

O projeto inclui scripts de automação que detectam Python, criam o ambiente virtual, instalam dependências e executam o projeto automaticamente.

**Linux/Mac:**

```bash
git clone <url-do-repositorio>
cd algen-pp
./run.sh
```

**Windows:**

```cmd
git clone <url-do-repositorio>
cd algen-pp
run.bat
```

Os scripts fazem automaticamente:
1. Detectam Python (3.8+)
2. Criam ambiente virtual se não existir
3. Ativam o ambiente virtual
4. Instalam/atualizam pip
5. Instalam todas as dependências de `requirements.txt`
6. Verificam a estrutura do projeto
7. Executam o algoritmo

**Nota**: Na primeira execução, o script pode levar alguns minutos para instalar todas as dependências.

### Método Manual

Se preferir fazer manualmente ou os scripts não funcionarem:

#### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd algen-pp
```

#### 2. Crie e ative um ambiente virtual (recomendado)

**Linux/Mac:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Instale as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Isso instalará automaticamente:

- `opencv-python` - Processamento de imagens
- `numpy` - Computação numérica
- `scikit-image` - Operações avançadas de imagem
- `scipy` - Ferramentas científicas
- `matplotlib` - Visualização (opcional)
- `tqdm` - Barras de progresso
- `tifffile` - Leitura de arquivos TIFF

## Como Executar

### Passo 1: Prepare suas imagens

Coloque suas imagens `.tif` na pasta `images/` (na raiz do projeto):

```bash
# A pasta images/ deve conter arquivos .tif
images/
  ├── imagem1.tif
  ├── imagem2.tif
  └── ...
```

### Passo 2: (Opcional) Configure os parâmetros do algoritmo

Edite `src/config.py` se quiser ajustar os parâmetros do algoritmo genético:

```python
POP_SIZE = 20                    # Tamanho da população (padrão: 20)
NUM_GENERATIONS = 20            # Número de gerações (padrão: 20)
MUTATION_RATE = 0.50             # Taxa de mutação (padrão: 50%)
ELITISM = 2                      # Número de melhores preservados (padrão: 2)
```

**Para testes rápidos**, você pode reduzir:

- `POP_SIZE = 10`
- `NUM_GENERATIONS = 10`

### Passo 3: Execute o projeto

**Método Recomendado: Script de Automação**

**Linux/Mac:**
```bash
./run.sh
```

**Windows:**
```cmd
run.bat
```

**Método Alternativo: Execução Manual**

Se preferir executar manualmente (após ativar o ambiente virtual):

```bash
# A partir da raiz do projeto
python3 src/main.py

# Ou, se estiver dentro de src/
cd src
python3 main.py
```

### O que acontece durante a execução

1. O algoritmo carrega todas as imagens `.tif` da pasta `images/`
2. Inicializa uma população de indivíduos (cada um com parâmetros diferentes)
3. Executa as gerações do algoritmo genético:
   - Avalia cada indivíduo (aplica pipeline de segmentação e calcula fitness)
   - Seleciona os melhores
   - Aplica crossover e mutação
   - Cria nova geração
4. Ao final, salva os melhores resultados em `outputs/`

**Tempo estimado**: Depende do número de gerações e imagens, mas pode levar alguns minutos a horas.

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
├── images/                        # Imagens de entrada (.tif) - criar manualmente
├── outputs/                       # Resultados gerados (criado automaticamente, no .gitignore)
│   ├── generation_results/        # Imagens de cada geração
│   ├── algen_basic_results/       # Resultado final
│   └── logs/                      # Logs e histórico JSON
├── venv/                          # Ambiente virtual (não versionado, no .gitignore)
├── docs/                          # Documentação adicional
│   └── ARTIGO_DAGUANO.md          # Documentação detalhada do artigo base
├── requirements.txt               # Dependências do projeto
├── run.sh                         # Script de automação (Linux/Mac)
├── run.bat                        # Script de automação (Windows)
├── .gitignore                     # Arquivos ignorados pelo Git
└── README.md                      # Este arquivo
```

**Nota importante**:

- A pasta `venv/` e `outputs/` estão no `.gitignore` e não são versionadas
- As imagens `.tif` na pasta `images/` podem ser versionadas se necessário (descomente no `.gitignore` se não quiser versionar)

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

## Estrutura de Saída

Após a execução, você encontrará em `outputs/`:

- **`outputs/logs/`**:

  - `algen_evolution_YYYYMMDD_HHMMSS.log` - Log completo da execução
  - `algen_history_YYYYMMDD_HHMMSS.json` - Histórico estruturado (fitness, parâmetros por geração)

- **`outputs/generation_results/YYYYMMDD_HHMMSS/`**:

  - `generation_01/`, `generation_02/`, ... - Resultados visuais de cada geração
  - Permite visualizar a evolução do algoritmo ao longo das gerações

- **`outputs/algen_basic_results/final/`**:
  - `*_segmented.png` - Segmentações binárias finais
  - `*_comparison.png` - Comparações com contornos
  - `*_side_by_side.png` - Imagens lado a lado

## Dicas e Troubleshooting

### Testes rápidos

Reduza `NUM_GENERATIONS` e `POP_SIZE` em `src/config.py`:

```python
POP_SIZE = 10           # Reduzir para testes
NUM_GENERATIONS = 10    # Reduzir para testes
```

### Ajuste de performance

- **Convergência muito rápida**: Aumente `MUTATION_RATE` (ex: 0.70)
- **Estagnação**: O algoritmo já tem mecanismos anti-estagnação, mas você pode aumentar `DIVERSITY_REINJECTION_RATE`

### Problemas comuns

**"ERRO: Nenhuma imagem encontrada!"**

- Verifique se a pasta `images/` existe na raiz do projeto
- Verifique se há arquivos `.tif` na pasta `images/`
- Verifique os caminhos em `src/config.py` (padrão: `"../images"`)

**Erro de importação de módulos**

- Certifique-se de estar executando de dentro da pasta `src/` ou ajuste os imports
- Verifique se todas as dependências foram instaladas: `pip install -r requirements.txt`
- Use os scripts `run.sh` ou `run.bat` que fazem isso automaticamente

**Ambiente virtual não ativado**

- Sempre ative o ambiente virtual antes de executar: `source venv/bin/activate` (Linux/Mac) ou `venv\Scripts\activate` (Windows)
- Ou use os scripts de automação que fazem isso automaticamente

**Script não executa (Linux/Mac)**

- Dê permissão de execução: `chmod +x run.sh`
- Execute com: `./run.sh`

**Python não encontrado**

- Instale Python 3.8 ou superior
- Certifique-se de que Python está no PATH do sistema
- No Linux/Mac, tente `python3` em vez de `python`

## Análise de Resultados

- **Visualização**: Explore as imagens em `outputs/generation_results/` para acompanhar a evolução
- **Análise estatística**: Use os arquivos JSON em `outputs/logs/` para análise de convergência
- **Comparação**: Compare visualmente as gerações para ver a melhoria da segmentação

## Documentação do Artigo Base

Documentação detalhada sobre o artigo original e comparações:

- **[docs/ARTIGO_DAGUANO.md](docs/ARTIGO_DAGUANO.md)** - Documentação completa sobre:

  - Como o algoritmo é aplicado
  - Parâmetros e configurações usados
  - Resultados obtidos no artigo original
  - Comparação com melhorias implementadas no Algen-PP

- **[docs/COMPARACAO_ARTIGO.md](docs/COMPARACAO_ARTIGO.md)** - Comparação detalhada:

  - Correções implementadas (gradiente morfológico, etc.)
  - Melhorias além do artigo (detecção de bordas, etc.)
  - Variações necessárias e justificativas
  - Checklist de validação

## Referências

- **Daguano, E. M. (2020)**: "Algoritmo Genético para Segmentação de Imagens utilizando Tamanho e Forma dos Objetos" - UNICAMP
  - PDF disponível em `assets/Daguano_EduardoManarin_M.pdf`
  - Documentação detalhada em `docs/ARTIGO_DAGUANO.md`

## Licença

Este projeto é para fins acadêmicos e de pesquisa.
