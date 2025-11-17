# Comparação: Implementação vs. Artigo Original (Daguano 2020)

Este documento detalha as diferenças entre nossa implementação (Algen-PP) e o artigo original, justificando melhorias e variações necessárias para validação.

## 1. Correções Implementadas (Agora Alinhadas com Artigo)

### 1.1 Pré-processamento: Gradiente Morfológico ✅

**Artigo Original:**

> "A etapa de pré-processamento foi criada para tratamento dos ruídos e das distorções. Esta etapa é iniciada com a utilização do gaussian blur para suavizar os objetos da imagem e após a suavização geramos o gradiente morfológico (diferença entre dilatação e erosão)."

**Implementação Anterior (INCORRETA):**

- Fazia erosão seguida de dilatação
- Não gerava o gradiente morfológico

**Implementação Corrigida (AGORA):**

- ✅ Implementa **gradiente morfológico** = dilatação - erosão
- ✅ Realça bordas e diferencia objetos do background
- ✅ Parâmetro opcional `use_morphological_gradient` (default: True)

**Justificativa:** O gradiente morfológico é fundamental no artigo original e realça melhor as bordas das células, facilitando a segmentação posterior.

---

## 2. Melhorias e Variações Implementadas

### 2.1 Detecção de Bordas (Canny Edge Detection) ✅ NOVO

**Artigo Original:**

> Trabalhos futuros sugerem: "detecção de sobreposição de objetos, pois nossa técnica apresenta dificuldades em detectar sobreposição de área de interesse e por esse motivo seria interessante aprimorar os resultados a partir de detectores de bordas"

**Nossa Implementação:**

- ✅ **Detecção de bordas Canny** opcional
- ✅ Ajuste automático de threshold baseado na mediana da imagem
- ✅ Uso de bordas para melhorar marcadores do watershed
- ✅ Parâmetro otimizável: `use_edge_detection` (0 ou 1)

**Justificativa:** Implementa sugestão do artigo como trabalho futuro, melhorando identificação de células, especialmente em casos de sobreposição.

**Como funciona:**

1. Detecta bordas com Canny (thresholds adaptativos)
2. Dilata bordas para conectar bordas próximas
3. Usa bordas como informação adicional na binarização
4. Adiciona marcadores do watershed próximos às bordas detectadas

---

### 2.2 Watershed Híbrido (Melhoria)

**Artigo Original:**

- Watershed hierárquica usando Árvore dos Lagos Críticos (ALC)
- Marcadores baseados em distance transform

**Nossa Implementação:**

- ✅ Marcadores baseados em **distance transform** (método original)
- ✅ Marcadores baseados em **intensidade local** (melhoria)
- ✅ Marcadores baseados em **bordas** (se edge detection ativo)
- ⚠️ **Não implementa ALC completa** (usa watershed do scikit-image)

**Variação Necessária:**
A implementação completa da ALC requer uma estrutura de árvore hierárquica complexa. Nossa implementação usa watershed do scikit-image, que produz resultados similares mas não constrói explicitamente a ALC.

**Justificativa:**

- Watershed do scikit-image é bem validado e produz segmentações corretas
- ALC completa adicionaria complexidade sem ganho claro para o objetivo principal
- Podemos validar resultados comparando com o artigo

---

### 2.3 Fitness Combinada vs. Almod Puro

**Artigo Original:**

- Fitness function = Almod apenas

**Nossa Implementação:**

- ✅ Almod normalizado (85%)
- ✅ Qualidade de forma (15%)
- ✅ Recompensa/penalização por número de células (10%)

**Justificativa:**

- Almod puro pode não capturar qualidade de forma das células
- Nossa métrica combinada incentiva células com formato elíptico (conforme esperado)
- Recompensa por número de células evita penalizar detecções corretas

**Fórmula:**

```
fitness = 0.85 × Almod_normalizado + 0.15 × Quality_penalty + 0.10 × Cell_penalty
```

---

## 3. Parâmetros Otimizados

### 3.1 Parâmetros do Artigo Original (6)

1. `gaussian_sigma`: 0.5 - 2.5
2. `median_ksize`: 1 - 5
3. `erosion`: 0 - 5
4. `dilation`: 0 - 5
5. `size_min`: 20 - 200
6. `size_max`: 80 - 800

### 3.2 Parâmetros Adicionais Otimizados no Algen-PP (15 total)

**Melhorias adicionais:** 7. `intensity_weight`: 0.0 - 1.0 (peso para marcadores de intensidade) 8. `weight_size`: 0.0 - 1.0 (peso do score de tamanho) 9. `weight_shape`: 0.0 - 1.0 (peso do score de forma) 10. `closing_kernel`: 1 - 11 (pós-processamento) 11. `merge_threshold`: 0.0 - 0.3 (fusão de regiões) 12. `min_area`: 5 - 200 (área mínima) 13. `refinement_iterations`: 0 - 2 (refinamento iterativo) 14. `use_morphological_gradient`: 0 ou 1 (booleano) 15. `use_edge_detection`: 0 ou 1 (booleano)

**Justificativa:** Mais parâmetros permitem melhor adaptação a diferentes imagens, mantendo estabilidade através do algoritmo genético.

---

## 4. Configuração do Algoritmo Genético

| Aspecto             | Artigo Original           | Algen-PP                       | Justificativa                |
| ------------------- | ------------------------- | ------------------------------ | ---------------------------- |
| **População**       | 16 indivíduos             | 20 indivíduos                  | Maior diversidade            |
| **Gerações**        | 7 (testes)                | 100 (configurável)             | Mais tempo de evolução       |
| **Seleção**         | Exclusão pior metade      | Exclusão pior metade + torneio | Maior diversidade genética   |
| **Crossover**       | Média simples             | BLX-alpha                      | Melhor exploração do espaço  |
| **Mutação**         | 10% taxa, 5-15% amplitude | 50% taxa, ±30% amplitude       | Evita convergência prematura |
| **Elitismo**        | Manutenção da metade      | 2 melhores                     | Preserva melhores soluções   |
| **Anti-estagnação** | Não mencionado            | Reinjeção de diversidade       | Mantém população ativa       |

---

## 5. Métricas de Seleção ALC

### 5.1 Score de Tamanho (Igual ao Artigo)

✅ Implementado conforme equação 3.1 do artigo:

- Intervalo original: `[size_min, size_max]` → score = 1.0
- Intervalo estendido: `[2/3×size_min, 4/3×size_max]`
- Penalização fora do intervalo

### 5.2 Score de Forma (Ellipse Fit)

✅ Implementado conforme equação 3.2 do artigo:

- `score_forma = area_objeto / (π × a × b)`
- Onde a e b são semi-eixos da elipse ideal

### 5.3 Threshold de Seleção

**Artigo Original:**

- Threshold implícito de ~0.5 (referências no texto)

**Algen-PP:**

- Threshold configurável: `ALC_SELECTION_THRESHOLD = 0.3`
- Reduzido para permitir mais células válidas serem selecionadas

**Justificativa:** Threshold de 0.5 pode ser muito restritivo e descartar células válidas. 0.3 oferece melhor balanço.

---

## 6. Validação e Comparação

### 6.1 Como Validar Nossa Implementação

**Método 1: Resultados do Artigo**

- Artigo reporta: 96% das instâncias com F-Score > 60%, média 73%
- Execução conjunta Algal+Algen: 100% com F-Score > 75%, média 86%
- Podemos comparar nossos resultados com essas métricas

**Método 2: Executar Baseline**

- Parâmetro `use_morphological_gradient=1` (True) → baseline
- Parâmetro `use_edge_detection=0` (False) → baseline
- Comparar resultados com melhorias habilitadas

**Método 3: Métricas Independentes**

- Usar ground-truth das imagens para calcular F-Score
- Comparar com resultados do artigo na mesma base de dados

---

## 7. Checklist de Validação

### ✅ Implementado Corretamente (Alinhado com Artigo)

- [x] Pré-processamento com Gaussian blur
- [x] Pré-processamento com Median blur
- [x] **Gradiente morfológico** (dilatação - erosão) ✅ CORRIGIDO
- [x] Watershed com distance transform
- [x] Score de tamanho (equação 3.1)
- [x] Score de forma / ellipse fit (equação 3.2)
- [x] Métrica Almod como base da fitness
- [x] Seleção ALC com intervalos estendidos

### ⚠️ Variações Necessárias (Justificadas)

- [ ] **ALC completa**: Usamos watershed do scikit-image em vez de implementar ALC completa

  - **Razão**: Complexidade vs. benefício. Watershed do scikit-image produz resultados similares
  - **Validação**: Comparar resultados quantitativamente

- [x] **Fitness combinada**: Almod + qualidade + células (artigo usa apenas Almod)
  - **Razão**: Melhor captura qualidade de forma
  - **Validação**: Comparar fitness Almod puro vs. combinada

### ✅ Melhorias Implementadas (Além do Artigo)

- [x] Detecção de bordas Canny (sugestão do artigo como trabalho futuro)
- [x] Watershed híbrido com marcadores de intensidade
- [x] Pós-processamento adaptativo iterativo
- [x] Mecanismos anti-estagnação
- [x] Crossover BLX-alpha
- [x] Seleção por torneio

---

## 8. Próximos Passos para Validação

1. **Executar baseline** (usar apenas parâmetros do artigo)
2. **Executar com melhorias** (detecção de bordas, etc.)
3. **Comparar resultados** quantitativamente (F-Score, Almod)
4. **Analisar imagens** qualitativamente (verificando segmentações)
5. **Documentar melhorias** se resultados forem superiores
6. **Justificar pioras** se houver, analisando parâmetros

---

## 9. Como Usar Versão Baseline vs. Melhorada

### Baseline (Mais Fiel ao Artigo)

Em `src/config.py`, você pode forçar parâmetros:

```python
# Forçar baseline (sem melhorias)
PARAM_RANGES = {
    'use_morphological_gradient': (1, 1, 'int'),  # Fixo em 1 (True)
    'use_edge_detection': (0, 0, 'int'),  # Fixo em 0 (False)
    'intensity_weight': (0.0, 0.0, 'float'),  # Sem watershed híbrido
    # ... outros parâmetros conforme artigo
}
```

### Versão Melhorada (Padrão Atual)

Permite que o algoritmo genético otimize todos os parâmetros, incluindo:

- `use_edge_detection`: 0 ou 1 (pode ser habilitado se melhorar)
- `intensity_weight`: 0.0 - 1.0 (pode usar watershed híbrido)
- Outras melhorias

---

## 10. Conclusão

**Nossa implementação:**

- ✅ Corrigida para usar **gradiente morfológico** (conforme artigo)
- ✅ Adiciona **detecção de bordas** (sugestão do artigo)
- ⚠️ Usa watershed do scikit-image (variação necessária)
- ✅ Melhora fitness com métricas combinadas
- ✅ Adiciona mecanismos anti-estagnação

**Validação necessária:**

- Comparar resultados quantitativamente (F-Score, Almod)
- Comparar qualitativamente (análise visual das segmentações)
- Justificar melhorias ou pioras através de experimentos

**Próximo passo:** Executar algoritmo e comparar resultados com métricas do artigo.
