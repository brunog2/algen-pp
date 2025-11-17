# Correções: Redução de Falsos Positivos e Melhoria de Detecção

## Problemas Identificados

1. **Fitness estagnado**: Todos os indivíduos têm fitness ~99,500.05 por várias gerações
2. **Alucinações**: Em imagens com poucas células (ex: uma célula no centro), detecta vários "riscos" (linhas verdes) mas não segmenta a célula correta
3. **Filtro de bordas muito agressivo**: Remove células válidas que tocam bordas (células cortadas pela metade)

## Correções Implementadas

### 1. Filtro de Bordas Inteligente ✅

**Problema**: O filtro removia TODAS as células que tocam bordas, incluindo células válidas cortadas.

**Solução**: Implementado filtro inteligente que distingue entre:
- **Linhas artificiais** (falsos positivos): muito finas (< 10 pixels) E alongadas (> 3x a dimensão menor)
- **Células válidas cortadas**: têm área e forma adequadas mesmo que toquem bordas

**Critérios de rejeição**:
- Rejeita se: `(min(altura, largura) < 10) AND (max(altura, largura) > 3×min(altura, largura))`
- Rejeita se: `(área < 50) AND (max(altura, largura) > 2×min(altura, largura))`
- Mantém células válidas que tocam bordas mas têm área e forma adequadas

**Localizações**:
- `src/segmentation.py` → `select_regions_by_size_shape()`
- `src/postprocessing.py` → `post_processing_learned()` (múltiplas etapas)

### 2. Marcadores Mais Seletivos ✅

**Problema**: Marcadores muito sensíveis detectavam ruído como células.

**Solução**: Aumentados thresholds para evitar falsos positivos:

**Distance Transform Markers**:
- Threshold mínimo: `max(3.0 pixels, 20% do máximo)` ao invés de sem threshold
- Fallback apenas se detectou < 3 marcadores E `max_dist > 5`

**Edge-Based Markers**:
- Modo agressivo: threshold de 25% do máximo (antes sem threshold)
- Modo conservador: threshold de 35% do máximo (antes 40% mas com menos filtros)

**Intensidade Markers**:
- Mantido threshold adaptativo (70º percentil, entre 0.5 e 0.8)
- Footprint maior (7x7) para detectar células grandes

### 3. Seleção de Regiões Mais Seletiva ✅

**Problema**: Falsos positivos passavam pela seleção mesmo com forma ruim.

**Solução**: Adicionado filtro adicional antes da seleção final:
- Rejeita células com `score_shape < 0.3` E `área fora do ideal` (fora de `size_min` ou > `size_max * 1.5`)
- Isso remove falsos positivos que têm forma muito ruim mas passaram pelo score combinado

**Localização**: `src/segmentation.py` → `select_regions_by_size_shape()`

## Resultados Esperados

Com essas correções:

1. ✅ **Falsos positivos reduzidos**: Linhas artificiais nas bordas são removidas, mas células válidas são mantidas
2. ✅ **Melhor detecção**: Marcadores mais seletivos detectam menos ruído
3. ✅ **Células cortadas preservadas**: Células válidas que tocam bordas são mantidas (ex: células pela metade)
4. ✅ **Menos "alucinações"**: Menos linhas verdes espúrias, mais células reais detectadas

## Validação

Execute novamente e verifique:

1. **Bordas da imagem**: 
   - ✅ Linhas artificiais finas e alongadas nas bordas devem ser removidas
   - ✅ Células válidas que tocam bordas devem ser mantidas

2. **Detecção de células**:
   - ✅ Menos "riscos" (linhas verdes) espúrios
   - ✅ Mais células reais detectadas corretamente

3. **Fitness**:
   - ✅ Fitness deve variar mais entre indivíduos
   - ✅ Deve haver evolução ao longo das gerações

## Ajustes Adicionais (se necessário)

Se ainda houver problemas:

1. **Aumentar thresholds de marcadores**: Aumentar percentuais (20% → 25%, 25% → 30%)
2. **Ajustar critérios de bordas**: Tornar filtro mais/menos restritivo (`< 10` → `< 8` ou `< 12`)
3. **Melhorar seleção de forma**: Aumentar threshold mínimo de `score_shape` (0.3 → 0.35)

## Notas Técnicas

- O filtro de bordas inteligente preserva células com área >= `size_min` ou forma adequada (`score_shape >= 0.3`)
- Marcadores baseados em bordas agora têm thresholds mais altos para evitar detecção de ruído
- Seleção final agora tem dupla verificação: score combinado E forma individual

