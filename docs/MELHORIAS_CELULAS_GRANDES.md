# Melhorias para Detecção de Células Grandes

## Problema Identificado

O algoritmo não estava detectando células grandes, especialmente no centro da imagem. Por exemplo, a célula central em `hoech069` não era identificada mesmo após 100 gerações.

## Causas Identificadas

1. **Intervalo de tamanho muito restritivo**: `size_max` máximo de 800 pixels não cobria células grandes
2. **Intervalo estendido insuficiente**: Células acima de `4/3 × size_max` eram completamente descartadas
3. **Penalização muito severa**: Score de tamanho para células grandes caía muito rápido
4. **Marcadores insuficientes**: Footprint de 3×3 não detectava bem células grandes no watershed
5. **Threshold fixo**: Não adaptava para células grandes válidas

## Correções Implementadas

### 1. Intervalo de Tamanho Aumentado

**Antes:**
```python
'size_max': (80, 800, 'int')
```

**Agora:**
```python
'size_max': (80, 1200, 'int')  # Aumentado para 1200 para detectar células grandes
```

### 2. Intervalo Estendido Flexível para Células Grandes

**Antes:**
- Intervalo estendido: `[2/3×size_min, 4/3×size_max]`
- Células acima de `4/3×size_max` eram descartadas

**Agora:**
- Intervalo estendido normal: `[2/3×size_min, 4/3×size_max]`
- **Limite grande**: Até `2×size_max` ou `max_ext×1.5` (o que for maior)
- Permite detectar células grandes até 2x o máximo configurado

### 3. Penalização Mais Suave para Células Grandes

**Antes:**
```python
elif area >= size_max:
    score_size = size_max / area  # Pode cair muito rápido (ex: 800/1600 = 0.5)
```

**Agora:**
```python
elif area > size_max:
    if area <= max_ext:
        # Dentro do intervalo estendido normal
        score_size = size_max / area
    else:
        # Fora do intervalo estendido mas ainda dentro do limite grande
        # Penalização mais suave: mantém score mínimo de 0.3 para células grandes válidas
        ratio = area / size_max
        if ratio <= 2.0:
            score_size = max(0.3, size_max / area)  # Mínimo de 0.3
        else:
            score_size = 0.1  # Muito grande, mas ainda considerar
```

**Exemplo:**
- Célula de 1600 pixels (2× o máximo de 800):
  - Antes: `score_size = 800/1600 = 0.5`
  - Agora: `score_size = max(0.3, 800/1600) = 0.5` (mantém mínimo)
  
- Célula de 2400 pixels (3× o máximo):
  - Antes: Descartada (fora do intervalo)
  - Agora: `score_size = 0.1` (ainda considerada)

### 4. Threshold Adaptativo para Células Grandes

**Antes:**
```python
if score >= config.ALC_SELECTION_THRESHOLD:  # Fixo em 0.3
    selected_mask[...] = 255
```

**Agora:**
```python
threshold = config.ALC_SELECTION_THRESHOLD  # 0.3 padrão
if area > size_max and area <= size_max * 2.0:
    # Células grandes: usar threshold um pouco mais baixo
    threshold = max(0.2, threshold - 0.05)  # 0.25 para células grandes

if score >= threshold:
    selected_mask[...] = 255
```

### 5. Melhorias nos Marcadores do Watershed

**Antes:**
```python
coords_dist = feature.peak_local_max(dist, footprint=np.ones((3, 3)), labels=bw)
```

**Agora:**
```python
# Footprint maior para detectar células grandes
coords_dist = feature.peak_local_max(dist, footprint=np.ones((5, 5)), labels=bw, min_distance=3)

# MELHORIA ADICIONAL: Se há poucos marcadores, usar threshold mais baixo
if np.sum(local_maxi_dist) < 5:
    # Tentar novamente com threshold mais baixo e footprint maior
    coords_dist_large = feature.peak_local_max(
        dist, 
        footprint=np.ones((7, 7)),  # Footprint ainda maior
        labels=bw, 
        min_distance=5,
        threshold_abs=np.max(dist) * 0.3  # Threshold mais baixo (30% do máximo)
    )
```

**Benefício:** Detecta melhor células grandes que podem ter menos contrastes locais.

### 6. Threshold Adaptativo para Intensidade

**Antes:**
```python
coords_intensity = feature.peak_local_max(
    img_norm, 
    footprint=np.ones((5, 5)), 
    threshold_abs=0.6,  # Fixo
    min_distance=5
)
```

**Agora:**
```python
# Threshold adaptativo baseado no percentil 70 da imagem
intensity_threshold = np.percentile(img_norm[img_norm > 0], 70)
intensity_threshold = max(0.5, min(0.8, intensity_threshold))  # Entre 0.5 e 0.8

coords_intensity = feature.peak_local_max(
    img_norm, 
    footprint=np.ones((7, 7)),  # Footprint maior para células grandes
    threshold_abs=intensity_threshold,  # Adaptativo
    min_distance=7  # Distância maior
)
```

### 7. Normalização de Pesos

**Antes:**
```python
score = weight_size * score_size + weight_shape * score_shape
# Problema: Se weights não somam 1.0, scores podem ser artificiais
```

**Agora:**
```python
total_weight = weight_size + weight_shape
if total_weight > 0:
    normalized_weight_size = weight_size / total_weight
    normalized_weight_shape = weight_shape / total_weight
else:
    normalized_weight_size = 0.5
    normalized_weight_shape = 0.5

score = normalized_weight_size * score_size + normalized_weight_shape * score_shape
```

## Resultados Esperados

Com essas melhorias, o algoritmo deve:

1. ✅ Detectar células grandes no centro da imagem
2. ✅ Manter boa detecção de células pequenas/médias
3. ✅ Melhorar evolução ao longo das gerações (detecção deve melhorar)
4. ✅ Reduzir falsos negativos para células grandes

## Validação

Execute o algoritmo novamente e verifique:

1. **Primeira geração**: Deve detectar algumas células grandes (mesmo que não perfeitas)
2. **Geração 100**: Deve melhorar significativamente a detecção da célula central
3. **Fitness**: Deve melhorar ao longo das gerações (menor = melhor)

## Notas Técnicas

- As mudanças são compatíveis com o artigo original (mantém fórmulas base)
- Melhorias são adaptativas (não quebram detecção de células pequenas)
- Parâmetros ainda são otimizáveis pelo algoritmo genético
- Limite de `2×size_max` evita falsos positivos extremos

## Próximos Passos (se necessário)

Se ainda houver problemas:

1. **Aumentar ainda mais `size_max`** para 1500-2000 se necessário
2. **Ajustar threshold mínimo** de 0.3 para células grandes (pode ir até 0.2)
3. **Melhorar detecção de marcadores** com técnicas de multi-scale
4. **Usar detecção de bordas** mais agressiva para células grandes

