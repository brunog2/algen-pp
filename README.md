# Algen-PP: Algoritmo Genético para Segmentação de Imagens

Algoritmo genético para segmentação automática de imagens biológicas (células) usando otimização de parâmetros de processamento de imagem.

## 📋 Descrição

Este projeto implementa um algoritmo genético melhorado baseado no trabalho de Daguano (2020) para segmentação de imagens de células. O algoritmo otimiza automaticamente parâmetros de um pipeline de segmentação que inclui:

- Pré-processamento (Gaussian blur, filtros morfológicos)
- Segmentação Watershed híbrida (distance transform + intensidade)
- Seleção por tamanho e forma (métricas ALC)
- Pós-processamento aprendido (refinamento iterativo)
- Avaliação combinada (Almod + qualidade de forma)

## 🚀 Características Principais

### Melhorias em relação ao algoritmo original:

1. **Watershed Híbrido**: Combina marcadores baseados em distance transform e intensidade local
2. **Fitness Combinada**: Almod (85%) + Qualidade de Forma (15%)
3. **Seleção por Torneio**: Maior diversidade genética
4. **Crossover BLX-alpha**: Melhor exploração do espaço de busca
5. **Refinamento Adaptativo**: Pós-processamento iterativo (0-2 iterações)
6. **Mecanismos Anti-Estagnação**: Reinjeção de diversidade e mutação aumentada

## 📦 Requisitos

```bash
pip install opencv-python numpy scikit-image scipy matplotlib tqdm
```

## 🎯 Uso

### Execução básica:

```bash
python3 algen_basic_test.py
```

### Parâmetros configuráveis:

No arquivo `algen_basic_test.py`:

```python
POP_SIZE = 20                    # Tamanho da população
NUM_GENERATIONS = 100            # Número de gerações
MUTATION_RATE = 0.50             # Taxa de mutação (50%)
ELITISM = 2                      # Número de melhores preservados
```

### Estrutura de saída:

```
outputs/
├── generation_results/          # Imagens de cada geração
│   └── YYYYMMDD_HHMMSS/
│       ├── generation_01/
│       ├── generation_02/
│       └── ...
├── algen_basic_results/         # Resultado final
│   └── final/
└── logs/                        # Logs e histórico
    ├── algen_evolution_*.log
    └── algen_history_*.json
```

## 📊 Parâmetros Otimizados

O algoritmo genético otimiza os seguintes parâmetros:

- **Pré-processamento**: `gaussian_sigma`, `median_ksize`, `erosion`, `dilation`
- **Watershed**: `intensity_weight` (peso para marcadores de intensidade)
- **Seleção ALC**: `size_min`, `size_max`, `weight_size`, `weight_shape`
- **Pós-processamento**: `closing_kernel`, `merge_threshold`, `min_area`, `refinement_iterations`

## 📁 Estrutura do Projeto

```
.
├── algen_basic_test.py          # Implementação principal (melhorada)
├── algen_pp.py                  # Implementação original completa
├── algen_2_pp.py                # Versão simplificada
├── images_tif/                  # Imagens de entrada (.tif)
├── outputs/                     # Resultados gerados
└── README.md                    # Este arquivo
```

## 🔧 Configuração

1. Coloque suas imagens `.tif` na pasta `images_tif/`
2. Ajuste os parâmetros do GA no arquivo `algen_basic_test.py` se necessário
3. Execute: `python3 algen_basic_test.py`

## 📈 Resultados

O algoritmo gera:

- **Logs detalhados**: Cada geração com estatísticas completas
- **Imagens por geração**: Comparação visual da evolução
- **Histórico JSON**: Dados estruturados para análise
- **Resultado final**: Melhor segmentação encontrada

## 🎓 Referências

- **Daguano, E. M. (2020)**: "Algoritmo Genético para Segmentação de Imagens utilizando Tamanho e Forma dos Objetos" - UNICAMP

## 📝 Licença

Este projeto é para fins acadêmicos e de pesquisa.

## 🔍 Diferenciais do Algoritmo

### Problemas resolvidos:

1. **Convergência prematura**: Mutação aumentada (50%) e reinjeção de diversidade
2. **Baixa cobertura**: Normalização da métrica Almod e threshold reduzido
3. **Estagnação**: Mecanismos automáticos de reinjeção após 5 gerações sem melhoria

### Métricas ajustadas:

- **Almod normalizado**: Não penaliza segmentações com mais células
- **Threshold de seleção**: Reduzido de 0.5 para 0.3 (mais células selecionadas)
- **Recompensa por células**: Penalização suave que incentiva detecção de células

## 💡 Dicas

- Para testes rápidos, reduza `NUM_GENERATIONS` e `POP_SIZE`
- Ajuste `MUTATION_RATE` se o algoritmo estiver convergindo muito rápido
- Visualize as imagens em `outputs/generation_results/` para acompanhar a evolução
- Use os logs JSON para análise estatística dos resultados
