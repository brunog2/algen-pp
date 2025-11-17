# Correções: Detecção e Eliminação de Dominância de Indivíduos

## Problema Identificado

Um indivíduo estava dominando todas as gerações (mesmo fitness aparecendo em >50% da população por 18+ gerações). Isso indica:
1. **Convergência prematura**: População convergindo para um mínimo local
2. **Elitismo excessivo**: Mesmo indivíduo sendo preservado demais
3. **Falta de diversidade**: Indivíduos repetidos não sendo removidos
4. **Mutação insuficiente**: Não conseguindo escapar da dominância

## Correções Implementadas

### 1. Detecção de Dominância ✅

**Implementado**: Sistema que detecta quando muitos indivíduos têm o mesmo fitness.

**Critérios**:
- **Dominância moderada**: >50% da população com mesmo fitness
- **Dominância crítica**: >70% da população com mesmo fitness

**Código**:
```python
fitness_counts = {}
for fit in fitnesses:
    fitness_counts[fit] = fitness_counts.get(fit, 0) + 1

most_common_fitness = max(fitness_counts.items(), key=lambda x: x[1])
dominant_fitness, dominant_count = most_common_fitness
dominance_ratio = dominant_count / len(fitnesses)

has_dominance = dominance_ratio > 0.5
has_critical_dominance = dominance_ratio > 0.7
```

### 2. Elitismo Adaptativo ✅

**Mudança**: O elitismo agora se adapta ao nível de dominância.

**Antes**:
- Sempre preservava 2 melhores indivíduos (fixo)

**Agora**:
- **Sem dominância**: Preserva 2 melhores (padrão)
- **Dominância moderada** (>50%): Preserva apenas 1 melhor
- **Dominância crítica** (>70%): **Elitismo DESABILITADO** (0 indivíduos preservados)

**Objetivo**: Impedir que o indivíduo dominante se multiplique indefinidamente.

### 3. Remoção de Indivíduos Repetidos ✅

**Implementado**: Sistema que detecta e remove cópias de indivíduos.

**Mecanismos**:
1. **Seleção de indivíduos únicos**: Durante elitismo, seleciona apenas indivíduos com fitness diferentes (arredondado para 2 casas decimais)
2. **Não adicionar survivors repetidos**: Se há dominância crítica, NÃO adiciona survivors que têm o fitness dominante
3. **Limpeza durante seleção**: Filtra indivíduos repetidos antes de adicionar à nova população

**Código**:
```python
# Arredondar fitness para detectar "iguais" (diferença < 0.01)
fit_rounded = round(fit, 2)
if fit_rounded not in seen_fitness:
    unique_survivors.append((ind.copy(), fit))
    seen_fitness.add(fit_rounded)
```

### 4. Mutação Ultra-Agressiva Durante Dominância ✅

**Mudança**: Durante dominância, a mutação fica muito mais agressiva.

**Durante Dominância Moderada**:
- Nível de estagnação artificial: `max(5, generations_without_improvement + 3)`
- Taxa de mutação: até 90%
- Genes mutados: até 95%
- Amplitude: até ±60%

**Durante Dominância Crítica** (>70%):
- Nível de estagnação artificial: `max(10, generations_without_improvement + 7)`
- Taxa de mutação: **95%** (quase sempre muta)
- Genes mutados: **98%** (quase todos os genes)
- Amplitude: **±75%** (variação muito grande)
- Reset completo: **85%** (reset completo de genes)

**Objetivo**: Forçar variação suficiente para escapar da dominância.

### 5. Reinjeção de Diversidade Adaptativa ✅

**Mudança**: A taxa de reinjeção de diversidade aumenta durante dominância.

**Taxas**:
- **Normal**: 30% (base)
- **Dominância moderada** (>50%): **60%** (dobra)
- **Dominância crítica** (>70%): **80%** (quase sempre cria aleatório)

**Objetivo**: Inundar população com indivíduos aleatórios durante dominância.

### 6. Logs e Avisos ✅

**Implementado**: Sistema de alertas quando dominância é detectada.

**Mensagens**:
- `[⚠️ DOMINÂNCIA DETECTADA]`: Quando >50% da população tem mesmo fitness
- `[⚠️ DOMINÂNCIA CRÍTICA DETECTADA]`: Quando >70% da população tem mesmo fitness
- `[AÇÃO] Removendo cópias e forçando mutação agressiva`: Indica ações tomadas

## Exemplo de Comportamento

### Cenário: 18 gerações sem melhoria, 90% da população com mesmo fitness

**Antes**:
- Elitismo: 2 indivíduos preservados (ambos com fitness dominante)
- Mutação: 50% taxa, ±30% amplitude
- Reinjeção: 20% taxa
- **Resultado**: Mesmo indivíduo continua dominando

**Agora**:
- **Detecção**: `[⚠️ DOMINÂNCIA CRÍTICA DETECTADA] Fitness 105,686.04 aparece em 18/20 indivíduos (90.0%)`
- **Elitismo**: 0 (desabilitado)
- **Survivors**: Não adiciona indivíduos com fitness dominante
- **Mutação**: 
  - Nível: 18 + 7 = 25 (muito alto)
  - Taxa: 95%
  - Genes: 98%
  - Amplitude: ±75%
  - Reset: 85%
- **Reinjeção**: 80% (quase sempre cria aleatório)
- **Resultado**: População completamente renovada, sem indivíduos dominantes

## Resultados Esperados

Com essas correções:

1. ✅ **Dominância detectada e eliminada**: Sistema detecta e age sobre dominância
2. ✅ **Elitismo adaptativo**: Reduzido/desabilitado durante dominância
3. ✅ **Indivíduos repetidos removidos**: Cópias são filtradas
4. ✅ **Mutação ultra-agressiva**: Escapa de mínimos locais durante dominância
5. ✅ **Mais diversidade**: Reinjeção aumentada durante dominância

## Validação

Execute novamente e observe:

1. **Mensagens de alerta**: Quando >50% ou >70% têm mesmo fitness
2. **Elitismo reduzido**: 1 ou 0 indivíduos preservados durante dominância
3. **Mutação mais agressiva**: Taxa e amplitude aumentam muito durante dominância
4. **Mais indivíduos aleatórios**: Reinjeção aumenta para 60-80% durante dominância
5. **Fitness variando**: Diferentes indivíduos devem ter fitness mais diversos

## Notas Técnicas

- A detecção de dominância usa fitness arredondado para 2 casas decimais (detecta "iguais")
- Durante dominância crítica, o elitismo é completamente desabilitado
- O nível de estagnação artificial é aumentado em +3 ou +7 durante dominância (além do nível real)
- A taxa de reinjeção de diversidade é aumentada para 60% ou 80% durante dominância
- Indivíduos repetidos são filtrados antes de serem adicionados à nova população

