# Changelog - Algen-PP

## [Melhorias Recentes] - 2025-11-16

### ✅ Correções Implementadas

1. **Pré-processamento: Gradiente Morfológico** 
   - ✅ **CORRIGIDO**: Agora implementa **gradiente morfológico** (dilatação - erosão) conforme artigo original
   - Antes: Fazia apenas erosão seguida de dilatação
   - Agora: Calcula diferença entre dilatação e erosão, realçando bordas
   - Parâmetro opcional: `use_morphological_gradient` (default: True)

2. **Detecção de Bordas (Canny Edge Detection)**
   - ✅ **NOVO**: Implementada detecção de bordas Canny para melhor identificação de células
   - Thresholds adaptativos baseados na mediana da imagem
   - Usa bordas para melhorar marcadores do watershed
   - Parâmetro otimizável: `use_edge_detection` (0 ou 1)

### 📊 Mudanças no Pipeline

**Pré-processamento:**
- Agora usa gradiente morfológico por padrão (conforme artigo)
- Opção de usar versão alternativa (erosão + dilatação) para comparação

**Segmentação:**
- Watershed híbrido: distance transform + intensidade + bordas (opcional)
- Detecção de bordas melhora identificação de células, especialmente em sobreposições

**Parâmetros Otimizáveis:**
- Adicionados: `use_morphological_gradient` e `use_edge_detection`
- Total: 15 parâmetros (vs. 6 do artigo original)

### 📝 Documentação

- ✅ Criado `docs/ARTIGO_DAGUANO.md`: Documentação completa do artigo base
- ✅ Criado `docs/COMPARACAO_ARTIGO.md`: Comparação detalhada implementação vs. artigo
- ✅ Atualizado `README.md`: Instruções de execução atualizadas

### 🔧 Ajustes Técnicos

- Corrigida importação de `cv2` em `segmentation.py`
- Corrigida conversão de parâmetros booleanos (0/1 → True/False)
- Validação de linter: sem erros

### 🎯 Próximos Passos

1. Executar algoritmo com as correções
2. Validar resultados quantitativamente (F-Score, Almod)
3. Comparar com métricas do artigo original
4. Ajustar parâmetros se necessário
5. Documentar melhorias ou pioras encontradas

---

## Melhorias Anteriores

### Watershed Híbrido
- Combina marcadores baseados em distance transform e intensidade local

### Fitness Combinada
- Almod (85%) + Qualidade de Forma (15%) + Recompensa por células (10%)

### Algoritmo Genético
- Seleção por torneio
- Crossover BLX-alpha
- Mutação aumentada (50% taxa, ±30% amplitude)
- Mecanismos anti-estagnação

### Pós-processamento Adaptativo
- Refinamento iterativo (0-2 iterações)
- Fusão de regiões adjacentes
- Remoção de regiões pequenas

