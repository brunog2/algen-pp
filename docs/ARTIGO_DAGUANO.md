# Documentação: Artigo Daguano (2020)

## Referência

**Daguano, Eduardo Manarin (2020)**  
_"Algoritmo Genético para Segmentação de Imagens utilizando Tamanho e Forma dos Objetos"_  
Dissertação de Mestrado - Faculdade de Tecnologia, UNICAMP  
Orientador: Prof. Dr. Ulisses Martins Dias

---

## 1. Visão Geral

O trabalho de Daguano propõe uma técnica de segmentação de imagens de células utilizando:

- **Transformada Watershed Hierárquica** como técnica de segmentação
- **Árvore dos Lagos Críticos (ALC)** como estrutura representativa
- **Algoritmo Genético** para otimização de parâmetros
- **Métricas baseadas em tamanho e forma** (curvatura elíptica) dos objetos

### 1.1 Três Algoritmos Principais

O trabalho desenvolveu três algoritmos complementares:

1. **Algen**: Algoritmo genético que aprimora resultados ao longo das gerações
2. **Algal**: Executa segmentação repetidamente de forma semi-aleatória
3. **Almod**: Classifica e avalia os resultados ao final de cada processamento

---

## 2. Pipeline de Segmentação

### 2.1 Etapas do Processo

O pipeline proposto possui **4 etapas principais**:

#### Etapa 1: Pré-processamento

- **Gaussian Blur**: Suavização da imagem para redução de ruído
- **Median Blur**: Filtro mediano adicional
- **Erosão e Dilatação**: Operações morfológicas em diferentes níveis de intensidade
- **Resultado**: Gradiente morfológico obtido da diferença entre dilatação e erosão

#### Etapa 2: Segmentação Watershed

- Computação da **Transformada Watershed Hierárquica**
- Geração da **Árvore dos Lagos Críticos (ALC)**
- Estrutura hierárquica representando junções de bacias durante a inundação

#### Etapa 3: Seleção por Métricas ALC

- Utilização de métricas baseadas em **tamanho** e **forma** dos objetos
- Aplicação de scores de tamanho e curvatura elíptica
- Seleção de componentes conexas de interesse

#### Etapa 4: Seleção do Melhor Resultado

- Seleção dos **5 melhores resultados** para análise conjunta
- Avaliação final utilizando métrica Almod

---

## 3. Parâmetros do Algoritmo Genético

### 3.1 Configuração do Algen (Artigo Original)

**População:**

- **Tamanho da população**: 16 indivíduos (nos testes do artigo)
- **Seleção**: Exclusão da pior metade (50% elitismo negativo)
- **Elitismo**: Manutenção da melhor metade

**Gerações:**

- **Número de gerações**: 7 gerações (nos testes)
- **Critério de parada**: Número pré-definido de gerações

**Crossover:**

- **Tipo**: Média simples entre dois indivíduos
- **Pareamento**: Melhor com último melhor, segundo com penúltimo, etc.
- Exemplo: População de 16 → cruza 1º com 16º, 2º com 15º, etc.

**Mutação:**

- **Taxa de mutação**: 10% por indivíduo
- **Amplitude de mutação**: 5% a 15% do valor do gene
- **Exemplo**: Gene com valor 100 pode variar de 85 a 115

### 3.2 Parâmetros Otimizados

O algoritmo genético otimiza **6 hiper-parâmetros** principais:

#### Pré-processamento:

1. **Gaussian Blur (sigma)**: Intensidade da suavização
2. **Median Blur (ksize)**: Tamanho do kernel mediano
3. **Erosão (kernel)**: Tamanho do kernel de erosão
4. **Dilatação (kernel)**: Tamanho do kernel de dilatação

#### Seleção ALC:

5. **Tamanho mínimo (size_min)**: Tamanho mínimo esperado dos objetos
6. **Tamanho máximo (size_max)**: Tamanho máximo esperado dos objetos

**Nota**: O artigo menciona também pesos para tamanho e forma, mas não detalha todos os parâmetros específicos.

---

## 4. Métricas de Seleção

### 4.1 Score de Tamanho

O score de tamanho é calculado considerando um **intervalo estendido**:

- **Intervalo original**: `[size_min, size_max]` (fornecido pelo usuário)
- **Intervalo estendido**: `[2/3 × size_min, 4/3 × size_max]`

**Função de score:**

- Se objeto dentro do intervalo original → `score = 1.0` (máximo)
- Se objeto acima do máximo → `score = size_max / object_size`
- Se objeto abaixo do mínimo → `score = object_size / size_min`

### 4.2 Score de Forma (Curvatura Elíptica)

Cálculo baseado na **curvatura elíptica ideal**:

```
score_forma = area_objeto / (π × a × b)
```

Onde:

- `a` e `b` são os semi-eixos da elipse ideal que circunscreve o objeto
- `area_objeto` é a área real do objeto
- Valor varia de 0 (longe de elipse) a 1 (elipse perfeita)

### 4.3 Score Final (Seleção ALC)

O artigo menciona uma **média ponderada** entre:

- Score de tamanho
- Score de forma (curvatura elíptica)

**Threshold de seleção**: Componentes com score abaixo de um threshold são descartadas (o artigo original usa threshold de 0.5, mas menciona que foi ajustado).

---

## 5. Métrica Almod (Fitness Function)

### 5.1 Definição

A métrica **Almod** é usada como **fitness function** no algoritmo genético:

```
Almod(I, S) = Σ Σ |I[i,j] - 255 × S[i,j]|
```

Onde:

- `I` = Imagem original
- `S` = Imagem segmentada (binária)
- `255 × S[i,j]` = Valor do pixel na imagem segmentada (0 ou 255)

**Interpretação**: Quanto **menor** o valor de Almod, **melhor** a segmentação.

### 5.2 Validação da Métrica

O artigo valida a métrica Almod executando:

- **5.376 segmentações** (112 por imagem, 48 imagens)
- População de 16 indivíduos, 7 gerações

**Resultados da validação:**

- Em **56% dos casos**, Almod encontrou a solução com melhor F-Score
- Em **17% dos casos**, Almod falhou em encontrar a melhor solução
- Demonstrou correlação positiva entre Almod baixo e F-Score alto

---

## 6. Resultados do Artigo

### 6.1 Base de Dados

**Laboratório Murphy:**

- 48 imagens de células Hoechst 33342
- Tipos: 3T3 e U20S
- Formatos: PNG, XCF, PSD

**ATCC (American Type Culture Collection):**

- Conjunto adicional de imagens de células

### 6.2 Desempenho do Algen

**Configuração de teste:**

- População: 16 indivíduos
- Gerações: 7
- Métrica: F-Score

**Resultados (Laboratório Murphy):**

- **96% das instâncias** com F-Score acima de 60%
- **Média geral**: 73%
- **Pior resultado**: 44%
- **Melhor resultado**: 90%

**Resultados (Execução conjunta Algal + Algen):**

- **100% das instâncias** com F-Score acima de 75%
- **Média geral**: 86%
- **Pior resultado**: 75%
- **Melhor resultado**: 90%

### 6.3 Performance

**Hardware de teste:**

- 8GB RAM
- Processador i3-4170 @ 3.70GHz
- Windows 10

**Tempo de processamento:**

- **5.376 segmentações em ~17 minutos**
- **~0,19 segundos por imagem**

### 6.4 Estabilidade

O artigo avalia a **volatilidade dos resultados**:

- Alta estabilidade quando imagens têm similaridade
- Resultados estáveis mesmo com parâmetros iniciais imprecisos
- Mecanismo de tratamento automático de parâmetros de entrada (Altrat)

---

## 7. Mecanismo de Estabilidade (Altrat)

### 7.1 Funcionamento

**Algoritmo Altrat** (Algoritmo de Tratamento de Parâmetros de Entrada):

1. **Verificação inicial**: Avalia os 5 primeiros resultados
2. **Critério de alerta**: Se F-Score < 50% para todos os 5 resultados
3. **Ajuste automático**: Recalcula valores iniciais (tamanho dos objetos)
4. **Critério de parada**: 5 resultados sequenciais com F-Score > 50%

**Objetivo**: Garantir qualidade mesmo com parâmetros iniciais imprecisos.

---

## 8. Comparações com Outras Técnicas

### 8.1 Técnicas Comparadas

O artigo compara com **27 técnicas** da literatura, incluindo:

- Otsu (1979)
- Minimum (Xu e Uberbacher, 1997)
- MoG (Manduchi, 2000)
- Triangle (Sun et al., 2002)
- Renyi Entropy (Li et al., 2006)
- Mean Shift (Tao, Jin e Zhang, 2007)
- Dual Background (Singh et al., 2009)
- ISODATA (El-Zaart, 2010)
- MRI (Li et al., 2011)
- E outras...

### 8.2 Resultados Comparativos

**Laboratório Murphy:**

- Algen/Algal está entre as melhores técnicas
- Superior à maioria das técnicas em estabilidade
- Melhor desempenho em F-Score médio

**ATCC:**

- Resultados consistentes com Murphy
- Alta estabilidade mesmo com menor similaridade entre imagens

---

## 9. Diferenças: Artigo Original vs. Nosso Projeto (Algen-PP)

### 9.1 Melhorias Implementadas no Algen-PP

| Aspecto               | Artigo Original (Daguano)      | Algen-PP (Nosso Projeto)                                   |
| --------------------- | ------------------------------ | ---------------------------------------------------------- |
| **Watershed**         | Watershed hierárquica clássica | **Watershed híbrido** (distance transform + intensidade)   |
| **Fitness**           | Almod apenas                   | **Fitness combinada**: Almod (85%) + Qualidade Forma (15%) |
| **Seleção**           | Exclusão da pior metade        | **Seleção por torneio** (mais diversidade)                 |
| **Crossover**         | Média simples                  | **Crossover BLX-alpha** (melhor exploração)                |
| **Mutação**           | 10% taxa, 5-15% amplitude      | **50% taxa**, ±30% amplitude (maior diversidade)           |
| **Pós-processamento** | Não mencionado                 | **Pós-processamento adaptativo** (0-2 iterações)           |
| **Anti-estagnação**   | Não mencionado                 | **Reinjeção de diversidade** automática                    |
| **Threshold ALC**     | 0.5 (implícito)                | **0.3** (permite mais células válidas)                     |

### 9.2 Parâmetros Adicionais Otimizados (Algen-PP)

Nosso projeto otimiza **mais parâmetros**:

1. **intensity_weight**: Peso para marcadores baseados em intensidade (0.0 - 1.0)
2. **weight_size**: Peso do score de tamanho (0.0 - 1.0)
3. **weight_shape**: Peso do score de forma (0.0 - 1.0)
4. **closing_kernel**: Tamanho do kernel de fechamento (1 - 11)
5. **merge_threshold**: Threshold de fusão de regiões (0.0 - 0.3)
6. **min_area**: Área mínima para manter região (5 - 200)
7. **refinement_iterations**: Número de iterações de refinamento (0 - 2)

**Total**: 13 parâmetros vs. 6 parâmetros do artigo original

### 9.3 Almod Normalizado

**Artigo original:**

```
Almod = Σ Σ |I[i,j] - 255 × S[i,j]|
```

**Algen-PP (normalizado):**

```
Almod_normalizado = (média_diferença_por_pixel) × sqrt(área)
```

**Motivo**: Não penalizar segmentações com mais células detectadas.

---

## 10. Limitações e Trabalhos Futuros (Artigo Original)

### 10.1 Limitações Identificadas

1. **Sobreposição de objetos**: Dificuldades em detectar células sobrepostas
2. **Contornos precisos**: Foco em encontrar objetos, não em contornos pixel-perfeitos
3. **Escala reduzida**: Segmentação em escalas reduzidas para velocidade
4. **Contexto específico**: Aplicado exclusivamente a imagens de células

### 10.2 Sugestões de Melhorias Futuras

1. Utilização de **gradiente** e **textura**
2. Detecção de **sobreposição de objetos**
3. Detetores de **bordas** e de **sobreposição**
4. Suporte a **outras formas geométricas** além de elipses
5. Aplicação a **outros contextos** além de células

---

## 11. Referências Principais do Artigo

- **Ranefall & Wälhlby (2016)**: Intervalo de tamanho (Size Interval Precision - SIP)
- **Ranefall, Sadanandan & Wählby (2016)**: Curvatura elíptica (Per Object Ellipse Fit - POE)
- **Beasley, Bull & Martin (1993)**: Algoritmo genético
- **Carvalho (2004)**: Árvore dos Lagos Críticos
- **Coelho, Shariff & Murphy (2009)**: Base de dados de células Hoechst 33342

---

## 12. Aplicação Prática no Algen-PP

### 12.1 Como Usar Esta Documentação

Esta documentação serve como referência para:

- Entender as bases teóricas do nosso projeto
- Comparar melhorias implementadas
- Validar escolhas de parâmetros
- Documentar decisões de design

### 12.2 Pontos de Atenção

1. **Parâmetros padrão**: Nosso projeto usa configurações mais agressivas (50% mutação vs. 10%)
2. **Diversidade**: Mecanismos anti-estagnação não presentes no artigo original
3. **Fitness combinada**: Melhoria sobre Almod puro
4. **Watershed híbrido**: Combina múltiplas estratégias de marcadores

### 12.3 Validação

Os resultados do artigo validam que:

- Algoritmo genético é efetivo para otimização de segmentação
- Métrica Almod correlaciona com F-Score
- Abordagem é estável e robusta

---

## 13. Conclusões

O trabalho de Daguano (2020) estabelece uma base sólida para segmentação de imagens de células utilizando:

- Árvore dos Lagos Críticos como estrutura representativa
- Métricas baseadas em tamanho e forma (curvatura elíptica)
- Algoritmo genético para otimização automática
- Alta estabilidade e bons resultados (F-Score médio 73-86%)

O projeto **Algen-PP** expande este trabalho com:

- Melhorias no watershed (híbrido)
- Fitness combinada (Almod + qualidade de forma)
- Mecanismos anti-estagnação
- Maior número de parâmetros otimizados
- Pós-processamento adaptativo

---

**Última atualização**: Baseado no artigo completo extraído do PDF (65 páginas)
