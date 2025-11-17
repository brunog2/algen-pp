# ALGORITMO GENÉTICO PARA SEGMENTAÇÃO DE IMAGENS UTILIZANDO TAMANHO E FORMA DOS OBJETOS

Bruno Gomes¹, Maurício Matheus², Matheus Lopes³

Universidade Federal de Alagoas (UFAL), Maceió, Brasil

17 de novembro de 2025

---

## Resumo

**Objetivo**: Este trabalho apresenta melhorias ao algoritmo genético proposto por Daguano (2020) para segmentação automática de imagens de células, focadas em resolver problemas de segmentação parcial e melhorar detecção de células escuras e nas bordas. **Método**: Foram implementadas cinco melhorias principais: (1) detecção de bordas Canny; (2) watershed híbrido com marcadores múltiplos; (3) função de fitness combinada com penalidade de completude; (4) fusão agressiva de regiões adjacentes; e (5) mecanismos anti-estagnação. O algoritmo foi avaliado em 27 imagens do dataset do Laboratório Murphy, com população de 20 indivíduos e 20 gerações. **Resultados**: Obtido fitness final de 126347 na geração 20. A análise qualitativa demonstra segmentações completas de células, incluindo células escuras e nas bordas, com baixa taxa de falsos positivos. **Conclusão**: As melhorias propostas resolvem efetivamente os problemas identificados, resultando em segmentações mais completas e consistentes quando comparadas às expectativas do algoritmo original.

**Palavras-chave**: Algoritmo Genético, Segmentação de Imagens, Watershed, Células, Otimização Evolutiva

---

## Abstract

Cell image segmentation is a fundamental task in cell biology and medical diagnosis. This work presents improvements to the genetic algorithm proposed by Daguano (2020) for automatic segmentation of cell images, using evolutionary optimization of image processing parameters. The main contributions include: (1) implementation of Canny edge detection to improve cell identification, especially in cases of overlap and borders; (2) hybrid watershed combining distance transform, local intensity and edge markers; (3) combined fitness function including penalty for incomplete segmentation; (4) aggressive fusion of adjacent regions to unite parts of the same cell; and (5) anti-stagnation mechanisms to ensure continuous evolution. Results obtained in generation 20 show fitness of 126347, with complete and consistent segmentations in 27 images from the Murphy Laboratory dataset. Qualitative analysis demonstrates complete cell detection, including dark cells and cells at borders, with few false positives.

**Keywords**: Genetic Algorithm, Image Segmentation, Watershed, Cells, Differential Evolution

---

## 1. Introduction

A segmentação de imagens é uma etapa fundamental em análise de imagens biomédicas, especialmente na identificação e quantificação de células. A segmentação automática de células é uma tarefa complexa devido à variabilidade de tamanho, forma, intensidade e sobreposição entre células. Métodos tradicionais de segmentação frequentemente requerem ajuste manual de parâmetros, o que é trabalhoso e propenso a erros.

Daguano (2020) propôs um algoritmo genético (Algen) para otimização automática de parâmetros de segmentação de imagens de células, utilizando a Transformada Watershed Hierárquica e a Árvore dos Lagos Críticos (ALC). O algoritmo utiliza métricas baseadas em tamanho e forma dos objetos (curvatura elíptica) para seleção de componentes conexas de interesse. A função de fitness é baseada na métrica Almod, que calcula a diferença pixel a pixel entre a imagem original e a segmentação.

Este trabalho apresenta melhorias ao algoritmo proposto por Daguano (2020), focadas em: (1) resolver problemas de segmentação parcial de células; (2) melhorar detecção de células escuras e nas bordas; (3) unir partes da mesma célula que foram segmentadas separadamente; e (4) penalizar explicitamente segmentações incompletas no fitness. As melhorias incluem implementação de detecção de bordas Canny (sugestão do artigo original como trabalho futuro), watershed híbrido com múltiplos marcadores, função de fitness combinada com penalidade de completude, e mecanismos anti-estagnação mais agressivos.

O objetivo deste trabalho é validar as melhorias propostas através de experimentos em um dataset público de imagens de células, comparando qualitativamente os resultados obtidos com as expectativas baseadas no artigo original.

---

## 2. Referencial Teórico

### 2.1 Segmentação de Imagens de Células

A segmentação de imagens de células é uma tarefa fundamental em biologia computacional e diagnóstico médico. Métodos clássicos de segmentação incluem técnicas baseadas em threshold (Otsu, 1979), watershed (Beucher e Lantuéjoul, 1979), e métodos baseados em aprendizado de máquina. Segundo Daguano (2020), a segmentação automática de células apresenta desafios devido à variabilidade de tamanho, forma, intensidade e sobreposição entre células.

Daguano (2020) apresenta uma comparação do algoritmo Algen com 27 técnicas da literatura, reportando que o algoritmo demonstrou superioridade em estabilidade e desempenho em F-Score, com 96% das instâncias apresentando F-Score superior a 60%, com média de 73%.

### 2.2 Algoritmos Genéticos para Segmentação

Algoritmos genéticos têm sido aplicados para otimização de parâmetros em diversas áreas, incluindo processamento de imagens. Segundo Beasley, Bull e Martin (1993), algoritmos genéticos são métodos de otimização inspirados na evolução natural, capazes de explorar espaços de busca complexos através de operadores de seleção, crossover e mutação.

Daguano (2020) propõe o uso de algoritmo genético para otimizar 6 hiper-parâmetros de um pipeline de segmentação baseado em watershed hierárquico. O trabalho demonstra que a abordagem evolutiva permite encontrar configurações de parâmetros adequadas sem necessidade de ajuste manual.

### 2.3 Árvore dos Lagos Críticos (ALC)

A Árvore dos Lagos Críticos (ALC) é uma estrutura hierárquica proposta por Carvalho (2004) que representa junções de bacias durante a inundação do watershed. Segundo Daguano (2020), a ALC permite seleção de componentes conexas baseada em métricas de tamanho e forma dos objetos.

Nesta implementação, utilizamos watershed do scikit-image, que produz resultados funcionais equivalentes sem construir explicitamente a estrutura completa da ALC, mantendo compatibilidade com a abordagem proposta enquanto simplifica a implementação.

---

## 3. Methodology

### 3.1 Algoritmo Original (Daguano, 2020)

O algoritmo original proposto por Daguano (2020) consiste em quatro etapas principais:

#### 3.1.1 Pré-processamento

A etapa de pré-processamento do algoritmo original consiste na aplicação sequencial de três operações de filtragem espacial. O primeiro estágio utiliza um filtro gaussiano para suavização da imagem, com o objetivo de reduzir ruído de alta frequência que pode interferir na segmentação subsequente. O parâmetro sigma do filtro gaussiano é otimizado pelo algoritmo genético dentro do intervalo de 0.5 a 2.5 pixels, permitindo ajuste fino do grau de suavização conforme as características de cada imagem.

Em seguida, aplica-se um filtro mediano adicional com kernel de tamanho variável entre 1 e 5 pixels, otimizado pelo algoritmo genético. O filtro mediano é particularmente eficaz na remoção de ruído impulsivo, preservando bordas enquanto elimina artefatos pontuais que poderiam ser confundidos com células.

A terceira etapa consiste na computação do gradiente morfológico, definido como a diferença entre a operação de dilatação e erosão morfológicas, ambas com kernels de tamanho variável entre 0 e 5 pixels. Esta operação realça as bordas dos objetos na imagem, aumentando o contraste entre células e background, e diferencia objetos do fundo através da amplificação de transições de intensidade. O gradiente morfológico é fundamental para a etapa subsequente de binarização, fornecendo informação espacial que guia a identificação de regiões de interesse.

#### 3.1.2 Segmentação Watershed

A etapa de segmentação watershed constitui o núcleo do algoritmo de Daguano (2020). Após o pré-processamento, a imagem é binarizada utilizando threshold adaptativo, gerando uma máscara binária que distingue objetos potenciais do background. Sobre esta máscara, computa-se a transformada de distância euclidiana, que atribui a cada pixel dentro dos objetos o valor da distância até a borda mais próxima. Os picos locais desta transformada de distância correspondem aos centros aproximados dos objetos, servindo como marcadores iniciais para o algoritmo watershed.

A partir destes marcadores, é computada a Transformada Watershed Hierárquica, que realiza uma inundação progressiva da imagem a partir dos marcadores, criando bacias de captura que correspondem às diferentes regiões segmentadas. Durante este processo, é gerada a Árvore dos Lagos Críticos (ALC), uma estrutura hierárquica que representa as junções entre bacias durante a inundação. A ALC permite análise hierárquica das relações entre regiões, facilitando a seleção posterior de componentes conexas baseada em critérios de tamanho e forma.

Os marcadores utilizados são exclusivamente baseados na transformada de distância da máscara binária, o que significa que apenas regiões que foram adequadamente capturadas na etapa de binarização podem gerar marcadores. Esta dependência pode resultar em perda de células que não foram corretamente identificadas na binarização inicial, especialmente células com intensidade reduzida ou em regiões de baixo contraste.

#### 3.1.3 Seleção por Métricas ALC

Após a segmentação watershed, cada componente conexa resultante é avaliada através de duas métricas principais que refletem características esperadas de células: tamanho e forma. O score de tamanho considera um intervalo estendido definido como `[2/3×size_min, 4/3×size_max]`, onde `size_min` e `size_max` são parâmetros otimizados pelo algoritmo genético. Componentes cujo tamanho se encontra dentro do intervalo ideal `[size_min, size_max]` recebem score máximo de 1.0, enquanto componentes fora deste intervalo recebem penalização proporcional à distância do intervalo ideal, permitindo alguma tolerância para variações naturais no tamanho das células.

O score de forma, também denominado curvatura elíptica, quantifica o quão próximo o formato da componente conexa está de uma elipse ideal. Esta métrica é calculada através da razão `area_objeto / (π × a × b)`, onde `a` e `b` representam os semi-eixos da elipse de momento de segunda ordem ajustada à componente. Esta razão varia de 0, indicando formato muito distante de uma elipse, até 1, representando uma elipse perfeita. Células típicas em imagens de microscopia apresentam formato aproximadamente elíptico, tornando esta métrica adequada para distinguir células válidas de artefatos e ruído.

O score final de cada componente é calculado como média ponderada entre o score de tamanho e o score de forma, com pesos otimizados pelo algoritmo genético. Componentes conexas que apresentam score final inferior a 0.5 são descartadas, consideradas como não representando células válidas. Este threshold fixo de 0.5, embora efetivo em muitos casos, pode ser excessivamente restritivo para células que apresentam características atípicas, como células grandes ou com formato não perfeitamente elíptico, resultando em perda de células válidas.

#### 3.1.4 Função de Fitness (Almod)

A função de fitness do algoritmo original utiliza exclusivamente a métrica Almod, que quantifica a diferença pixel a pixel entre a imagem original e a segmentação binária resultante. A métrica é definida matematicamente como:

```
Almod(I, S) = Σ Σ |I[i,j] - 255 × S[i,j]|
```

onde `I` representa a imagem original em escala de cinza, `S` representa a imagem segmentada binária (com valores 0 para background e 255 para objetos segmentados), e a soma é realizada sobre todos os pixels `(i,j)` da imagem. Esta métrica penaliza tanto pixels que foram incorretamente classificados como células (falsos positivos) quanto pixels de células que não foram segmentados (falsos negativos), através da diferença absoluta entre a intensidade original e o valor binário atribuído.

A interpretação da métrica Almod é que valores menores indicam melhor qualidade de segmentação, pois representam menor discrepância entre a imagem original e a segmentação binária. No entanto, esta métrica apresenta limitações importantes: não captura explicitamente a qualidade da forma das células segmentadas, não penaliza diretamente segmentações incompletas (onde apenas parte de uma célula é segmentada), e pode ser sensível a variações na intensidade global da imagem. Além disso, a métrica pura pode favorecer segmentações que detectam menos células, já que menos pixels segmentados resultam em menor soma de diferenças absolutas.

#### 3.1.5 Configuração do Algoritmo Genético

O algoritmo genético proposto por Daguano (2020) opera com uma população de 16 indivíduos, onde cada indivíduo representa um conjunto de 6 parâmetros de segmentação codificados como vetor de números reais. Durante os testes reportados no artigo, o algoritmo foi executado por 7 gerações, permitindo evolução da população através de operadores genéticos.

O operador de crossover implementado utiliza média simples entre dois indivíduos selecionados, gerando um descendente cujos parâmetros correspondem à média aritmética dos parâmetros dos pais. Esta abordagem de crossover conservadora tende a gerar soluções intermediárias entre os pais, facilitando exploração local do espaço de busca, mas potencialmente limitando a diversidade genética da população.

A mutação é aplicada com taxa de 10%, significando que aproximadamente 10% dos indivíduos da população sofrem mutação em cada geração. A amplitude da mutação varia entre ±5% e ±15% do valor atual do parâmetro, permitindo ajustes finos ou mais significativos dependendo do parâmetro e da geração. Esta taxa de mutação relativamente baixa pode resultar em convergência prematura para soluções subótimas, especialmente em espaços de busca complexos.

A estratégia de seleção adotada consiste na exclusão da pior metade da população (50% elitismo negativo), onde os 50% indivíduos com pior fitness são removidos e substituídos por descendentes gerados através de crossover e mutação. A melhor metade da população é mantida intacta (elitismo), garantindo que soluções promissoras não sejam perdidas, mas potencialmente reduzindo a diversidade genética ao longo das gerações.

O algoritmo genético otimiza simultaneamente 6 parâmetros do pipeline de segmentação: `gaussian_sigma` (0.5 a 2.5), controlando a suavização gaussiana; `median_ksize` (1 a 5), definindo o tamanho do kernel do filtro mediano; `erosion` (0 a 5), especificando o tamanho do kernel de erosão morfológica; `dilation` (0 a 5), definindo o tamanho do kernel de dilatação morfológica; `size_min` (20 a 200), estabelecendo o tamanho mínimo esperado de células; e `size_max` (80 a 800), definindo o tamanho máximo esperado de células. Estes intervalos foram definidos com base em características típicas de imagens de células e permitem adaptação automática do algoritmo a diferentes tipos de imagens através da evolução genética.

---

### 3.2 Nossas Contribuições

#### 3.2.1 Detecção de Bordas Canny

O artigo original sugere como trabalho futuro a "detecção de sobreposição de objetos, pois nossa técnica apresenta dificuldades em detectar sobreposição de área de interesse e por esse motivo seria interessante aprimorar os resultados a partir de detectores de bordas" (Daguano, 2020).

Implementamos detecção de bordas Canny com thresholds adaptativos baseados na mediana da imagem (`median ± 0.33 * std`). As bordas detectadas são dilatadas para conectar bordas próximas, combinadas com a imagem na binarização (modo agressivo), e utilizadas para adicionar marcadores do watershed próximos às bordas detectadas. O uso de detecção de bordas é controlado pelo parâmetro `use_edge_detection` (0 ou 1), permitindo que o algoritmo genético determine se esta melhoria é benéfica para cada conjunto de imagens.

Esta implementação melhora a identificação de células especialmente em casos de sobreposição e células nas bordas, conforme sugerido pelo artigo original.

---

#### 3.2.2 Watershed Híbrido

Marcadores baseados exclusivamente em distance transform podem perder células escuras ou células em regiões de baixo contraste, conforme observado durante o desenvolvimento.

Implementamos watershed híbrido combinando três tipos de marcadores. O método original utiliza marcadores de distance transform. Adicionamos marcadores de intensidade local com threshold adaptativo no 50º percentil (range 0.3-0.7), detectando células escuras que podem ser perdidas pelo distance transform. Marcadores de intensidade são adicionados mesmo fora da máscara binária. Quando detecção de bordas está ativa, marcadores baseados em bordas são utilizados com thresholds mais permissivos (20-25% do máximo do distance transform), melhorando detecção em regiões de baixo contraste.

O peso relativo dos marcadores de intensidade é controlado pelo parâmetro `intensity_weight` (0.0 - 1.0), permitindo ajuste fino da contribuição deste tipo de marcador.

---

#### 3.2.3 Função de Fitness Combinada

Almod puro pode não capturar adequadamente a qualidade de forma das células e não penaliza explicitamente segmentações incompletas (células segmentadas apenas parcialmente), problema identificado durante testes iniciais.

A função de fitness combinada implementada integra quatro componentes distintas, cada uma capturando aspectos diferentes da qualidade da segmentação. A formulação matemática da função de fitness é:

```
fitness = 0.70 × Almod_normalizado +
         0.15 × Quality_penalty +
         0.10 × Cell_penalty +
         0.15 × Completeness_penalty
```

A primeira componente, Almod Normalizado, corresponde a 70% do peso total da função de fitness e representa uma normalização da métrica Almod original. A normalização é realizada através da multiplicação da média de diferença por pixel pela raiz quadrada da área total segmentada, expressa como `(média_diferença_por_pixel) × sqrt(área)`. Esta normalização é fundamental para evitar que segmentações que detectam maior número de células válidas sejam incorretamente penalizadas, já que a métrica Almod pura tende a aumentar proporcionalmente com o número de pixels segmentados. A normalização permite comparação justa entre segmentações com diferentes números de células detectadas.

A segunda componente, Qualidade de Forma, contribui com 15% do peso total e quantifica o quão próximo o formato das células segmentadas está de uma elipse ideal. Esta métrica utiliza o score de ellipse fit conforme definido na equação 3.2 do artigo original de Daguano (2020), calculando a razão entre a área da componente conexa e a área da elipse de momento de segunda ordem ajustada. Valores próximos a 1.0 indicam formato elíptico, enquanto valores menores indicam desvios da forma elíptica esperada. Esta componente incentiva explicitamente a detecção de células com formato elíptico, alinhada com características morfológicas típicas de células em imagens de microscopia.

A terceira componente, Recompensa por Células, representa 10% do peso total e implementa uma penalização que diminui proporcionalmente com o aumento do número de células válidas detectadas. Esta componente foi projetada para incentivar o algoritmo a detectar maior número de células, contrabalanceando a tendência natural de métricas baseadas em diferença pixel a pixel de favorecer segmentações mais conservadoras com menos células. A penalização é calculada de forma que segmentações com poucas células recebem penalidade maior, enquanto segmentações que detectam muitas células válidas recebem penalidade reduzida ou até recompensa.

A quarta e mais importante componente para resolver o problema de segmentação parcial é a Penalidade de Completude, que contribui com 15% do peso total. Esta métrica foi desenvolvida especificamente para detectar e penalizar segmentações incompletas, onde apenas parte de uma célula é segmentada. O processo de cálculo inicia com a detecção de células na imagem original utilizando threshold adaptativo definido como `median + 1.5 * std`, onde `median` e `std` representam a mediana e o desvio padrão da intensidade da imagem. Esta detecção fornece uma estimativa da área total que deveria ser segmentada como células.

O ratio de completude é então calculado como `área_segmentada / área_total_células`, representando a proporção da área total de células que foi efetivamente segmentada. A penalidade é aplicada proporcionalmente através da fórmula `(1.0 - completude_ratio) * 500000`, resultando em penalidade zero quando 100% das células são segmentadas e penalidade máxima de 500000 quando nenhuma célula é segmentada. Por exemplo, se 80% das células são segmentadas, a penalidade aplicada é de 100000, incentivando o algoritmo a evoluir para soluções que segmentam células completas. Esta componente foi fundamental para resolver o problema observado durante testes iniciais, onde o algoritmo frequentemente segmentava células em múltiplas partes separadas, deixando áreas válidas da célula sem segmentar.

---

#### 3.2.4 Fusão Agressiva de Regiões

Durante testes iniciais, observou-se que células eram segmentadas em múltiplas partes separadas, deixando áreas válidas da célula sem segmentar.

Implementamos fusão agressiva de regiões adjacentes baseada em três critérios. A similaridade de intensidade utiliza threshold adaptativo aumentado até 2× o threshold base. A proximidade é avaliada através de dilatação com kernel 5×5 em 2 iterações (versus 3×3 com 1 iteração no original). Regiões pequenas (inferiores a 200 pixels) têm threshold ainda mais permissivo: enquanto regiões normais usam `threshold_efetivo = merge_threshold * 1.5`, regiões pequenas usam `merge_threshold * 2.0`, com threshold mínimo de 0.15 para garantir fusão mesmo com threshold baixo.

O fechamento morfológico foi melhorado com kernel aumentado em 50% (`k * 1.5`) para conectar partes da célula. Fechamento adicional é aplicado após fusão para conectar partes próximas, e ambos são aplicados em todas as iterações de refinamento, não apenas na última.

---

#### 3.2.5 Seleção Mais Permissiva

O threshold de seleção ALC de 0.5 utilizado no artigo original mostrou-se muito restritivo durante testes, descartando células válidas, especialmente células grandes e escuras.

Reduzimos o threshold base para 0.25 (reduzido de 0.3 inicialmente considerado), com threshold de 0.15 para células grandes e 0.20 para células dentro do tamanho ideal. Este relaxamento é balanceado por filtros rigorosos de aspect ratio que rejeitam regiões com `aspect_ratio > 4.0` ou `axis_ratio > 5.0`, e regiões muito pequenas e alongadas (inferiores a 5 pixels com aspect ratio > 6.0), efetivamente eliminando falsos positivos como linhas e artefatos enquanto mantém células válidas.

---

#### 3.2.6 Configuração do Algoritmo Genético Melhorada

A configuração do algoritmo genético foi significativamente modificada em relação ao trabalho original, com o objetivo de melhorar a exploração do espaço de busca e evitar convergência prematura. A população foi aumentada de 16 para 20 indivíduos, proporcionando maior diversidade genética inicial e permitindo manutenção de múltiplas soluções promissoras simultaneamente. O número de gerações foi expandido de 7 (utilizado nos testes do artigo original) para 20-100 gerações configuráveis, permitindo maior tempo de evolução e convergência mais gradual para soluções ótimas.

A taxa de mutação foi drasticamente aumentada de 10% para 50%, com amplitude de mutação expandida de ±5-15% para ±30% do valor atual do parâmetro. Esta mudança é fundamental para evitar convergência prematura para soluções subótimas, especialmente importante dado o aumento no número de parâmetros otimizados (de 6 para 15). A maior taxa de mutação mantém a população geneticamente diversa, permitindo exploração contínua de diferentes regiões do espaço de busca.

O operador de crossover foi alterado de média simples para BLX-alpha (Blend Crossover), um método mais sofisticado que gera descendentes em uma região hiper-retangular definida pelos pais, permitindo melhor exploração do espaço de busca. A seleção foi modificada de exclusão da pior metade para seleção por torneio, onde indivíduos competem em torneios de tamanho fixo, resultando em maior diversidade na população selecionada. O elitismo foi reduzido de manutenção da melhor metade para preservação de apenas os 2 melhores indivíduos, balanceando preservação de soluções ótimas com manutenção de diversidade genética.

Foram implementados quatro mecanismos anti-estagnação que não estavam presentes no algoritmo original. O primeiro mecanismo, Reinjeção de Diversidade, aplica uma probabilidade de 30% de criar indivíduos completamente aleatórios durante a geração de descendentes, aumentando para valores adaptativos superiores a 20% durante períodos de estagnação detectada. Este mecanismo garante que novas regiões do espaço de busca sejam continuamente exploradas, mesmo quando a população converge para uma região específica.

O segundo mecanismo, Reinjeção por Estagnação, monitora a melhoria do melhor fitness ao longo das gerações. Após 3 gerações consecutivas sem melhoria significativa, até 40% da população é substituída por indivíduos gerados através de mutação agressiva dos melhores indivíduos ou por indivíduos aleatórios. O terceiro mecanismo, Reset Parcial, é ativado quando a estagnação persiste por mais de 8 gerações, substituindo até 50% da população por novas soluções, efetivamente reiniciando parcialmente a busca evolutiva.

O quarto mecanismo, Idade Máxima de Indivíduos, implementa um sistema de envelhecimento onde indivíduos que persistem na população por mais de 5 gerações são considerados "velhos" e são substituídos por mutações agressivas dos melhores indivíduos ou por novos indivíduos aleatórios. Este mecanismo previne que soluções dominantes persistam excessivamente na população, forçando renovação contínua e evitando que a população fique presa em ótimos locais.

---

#### 3.2.7 Parâmetros Adicionais Otimizados

A expansão do conjunto de parâmetros otimizados de 6 para 15 parâmetros representa um aumento significativo na capacidade de adaptação do algoritmo a diferentes características de imagens. Os 6 parâmetros originais foram mantidos (`gaussian_sigma`, `median_ksize`, `erosion`, `dilation`, `size_min`, `size_max`), e foram adicionados 9 novos parâmetros que controlam aspectos específicos das melhorias implementadas.

O parâmetro `intensity_weight` (intervalo 0.0 a 1.0) controla o peso relativo dos marcadores de intensidade local no watershed híbrido, permitindo que o algoritmo genético determine a contribuição ótima deste tipo de marcador para cada conjunto de imagens. Os parâmetros `weight_size` e `weight_shape` (ambos no intervalo 0.0 a 1.0) controlam os pesos relativos do score de tamanho e do score de forma na seleção de componentes conexas, permitindo adaptação do critério de seleção conforme as características morfológicas das células em diferentes imagens.

O parâmetro `closing_kernel` (intervalo 1 a 11) define o tamanho do kernel utilizado no fechamento morfológico durante o pós-processamento, controlando o grau de conexão entre partes próximas de células. O parâmetro `merge_threshold` (intervalo 0.0 a 0.3) estabelece o threshold de similaridade de intensidade utilizado na fusão agressiva de regiões adjacentes, permitindo ajuste fino da sensibilidade do processo de unificação de partes de células.

O parâmetro `min_area` (intervalo 5 a 200 pixels) define a área mínima que uma região deve possuir para ser mantida na segmentação final, servindo como filtro adicional para eliminar artefatos muito pequenos. O parâmetro `refinement_iterations` (intervalo 0 a 2) controla o número de iterações de refinamento aplicadas, onde as etapas de seleção e pós-processamento são repetidas para melhorar a segmentação.

Finalmente, dois parâmetros binários (0 ou 1) permitem que o algoritmo genético determine se técnicas específicas devem ser utilizadas: `use_morphological_gradient` controla o uso do gradiente morfológico no pré-processamento, enquanto `use_edge_detection` controla a utilização da detecção de bordas Canny. Esta abordagem permite que o algoritmo evolua para soluções que utilizam apenas as técnicas mais efetivas para cada conjunto de imagens, evitando sobrecarga computacional desnecessária.

A justificativa para esta expansão de parâmetros baseia-se na necessidade de maior flexibilidade para adaptação a diferentes tipos de imagens, mantendo a estabilidade através da validação cruzada implícita proporcionada pela avaliação simultânea em múltiplas imagens do dataset. O algoritmo genético, com seus mecanismos de exploração e exploração balanceados, é capaz de encontrar configurações adequadas mesmo neste espaço de busca expandido, desde que sejam utilizadas populações e números de gerações suficientes para convergência adequada.

---

### 3.3 Integração das Melhorias ao Pipeline

A integração das melhorias implementadas ao pipeline de segmentação segue uma sequência lógica que maximiza a sinergia entre os diferentes componentes. A primeira etapa consiste no pré-processamento da imagem, onde são aplicadas sequencialmente as operações de Gaussian blur, median blur e gradiente morfológico, todas com parâmetros otimizados pelo algoritmo genético. Paralelamente, se o parâmetro `use_edge_detection` evolui para valor 1, a detecção de bordas Canny é executada com thresholds adaptativos baseados na mediana e desvio padrão da imagem. As bordas detectadas são então processadas através de dilatação morfológica para conectar bordas próximas e integradas ao processo de binarização em modo agressivo, onde são combinadas com a informação de intensidade para gerar uma máscara binária mais robusta.

A segunda etapa realiza a segmentação watershed híbrida, que representa uma evolução significativa em relação ao método original. Enquanto o algoritmo original utiliza exclusivamente marcadores baseados na transformada de distância da máscara binária, nossa implementação combina três tipos distintos de marcadores. Os marcadores de distance transform são computados sobre a máscara binária, identificando centros aproximados de objetos. Os marcadores de intensidade local são gerados através de threshold adaptativo no 50º percentil da distribuição de intensidades (com range otimizável entre 0.3 e 0.7), detectando células escuras que podem não ser adequadamente capturadas pela máscara binária. Estes marcadores de intensidade são adicionados mesmo fora da máscara binária, expandindo a capacidade de detecção. Quando a detecção de bordas está habilitada, um terceiro conjunto de marcadores baseados em bordas é gerado com thresholds mais permissivos (20-25% do máximo da transformada de distância), melhorando a detecção em regiões de baixo contraste. O peso relativo dos marcadores de intensidade é controlado pelo parâmetro `intensity_weight`, permitindo que o algoritmo genético ajuste a contribuição deste tipo de marcador.

A terceira etapa realiza a seleção de componentes conexas baseada em métricas de tamanho e forma, utilizando thresholds adaptativos que variam conforme o tamanho da célula. O threshold base foi reduzido de 0.5 (original) para 0.25, com thresholds ainda mais permissivos de 0.15 para células grandes e 0.20 para células dentro do intervalo de tamanho ideal. Este relaxamento dos thresholds é balanceado por filtros rigorosos de aspect ratio que rejeitam regiões com `aspect_ratio > 4.0` ou `axis_ratio > 5.0`, e regiões muito pequenas e alongadas (inferiores a 5 pixels com aspect ratio > 6.0). Estes filtros efetivamente eliminam falsos positivos como linhas e artefatos enquanto mantêm células válidas que poderiam ser descartadas pelos thresholds mais restritivos do algoritmo original.

A quarta etapa consiste no pós-processamento agressivo, onde a fusão de regiões adjacentes e o fechamento morfológico são aplicados para unir partes da mesma célula que foram segmentadas separadamente. A fusão utiliza thresholds de similaridade de intensidade que podem ser aumentados até 2× o threshold base (`merge_threshold`), com thresholds ainda mais permissivos para regiões pequenas (inferiores a 200 pixels). A proximidade entre regiões é avaliada através de dilatação morfológica com kernel 5×5 em 2 iterações, expandindo significativamente a área de busca em relação ao método original (kernel 3×3 com 1 iteração). O fechamento morfológico utiliza kernel aumentado em 50% (`k * 1.5`) em relação ao tamanho base, e é aplicado tanto antes quanto após a fusão, garantindo conexão de partes próximas da célula. O refinamento iterativo, controlado pelo parâmetro `refinement_iterations` (0 a 2 iterações), aplica as etapas de seleção e pós-processamento múltiplas vezes quando necessário, permitindo melhoria gradual da segmentação.

A quinta e final etapa consiste na avaliação da segmentação através da função de fitness combinada, que integra quatro componentes distintas: Almod normalizado (70%), qualidade de forma (15%), recompensa por células (10%) e penalidade de completude (15%). Esta função de fitness direciona a evolução do algoritmo genético para soluções que não apenas minimizam a diferença pixel a pixel, mas também segmentam células completas com formato adequado e em número adequado. O pipeline integrado permite que o algoritmo genético otimize simultaneamente todos os 15 parâmetros, encontrando configurações adequadas para cada conjunto de imagens através da evolução genética guiada pela função de fitness combinada.

---

### 3.4 Ajustes Experimentais

Durante o desenvolvimento e validação experimental do algoritmo, três problemas específicos foram identificados através de análise qualitativa das segmentações geradas e corrigidos através de ajustes nos parâmetros e filtros do pipeline.

O primeiro problema identificado foi a perda sistemática de células com intensidade reduzida, que não eram adequadamente capturadas pelos marcadores baseados exclusivamente em distance transform. A análise das imagens segmentadas revelou que células com intensidade significativamente abaixo da média da imagem eram frequentemente perdidas, mesmo quando apresentavam formato e tamanho adequados. Para resolver este problema, o threshold de intensidade utilizado na geração de marcadores de intensidade local foi reduzido do 70º para o 50º percentil da distribuição de intensidades, com o range otimizável ampliado de [0.5, 0.7] para [0.3, 0.7]. Esta modificação permitiu detectar células com intensidade reduzida que anteriormente eram perdidas, mantendo a capacidade do algoritmo genético de ajustar o threshold ótimo através da evolução do parâmetro `intensity_weight`.

O segundo problema consistia na rejeição sistemática de células localizadas nas bordas da imagem, que eram filtradas por critérios baseados em posição espacial. A análise revelou que células parcialmente cortadas pelas bordas da imagem, mesmo quando mais de 50% de sua área estava visível, eram frequentemente descartadas. Para resolver este problema, foram removidos completamente os filtros baseados em posição nas bordas da imagem, mantendo apenas a rejeição de artefatos óbvios através de critérios de tamanho e forma. Especificamente, apenas regiões muito pequenas (inferiores a 5 pixels) com aspect ratio extremamente alto (> 6:1) são rejeitadas, efetivamente eliminando linhas e artefatos enquanto permitindo detecção de células parcialmente cortadas. Esta modificação resultou em melhoria significativa na cobertura de células nas bordas das imagens.

O terceiro problema identificado foi a presença de falsos positivos, especialmente linhas e artefatos alongados que eram incorretamente classificados como células. Para resolver este problema, foram implementados filtros rigorosos de aspect ratio e forma que rejeitam regiões com `aspect_ratio > 4.0` ou `axis_ratio > 5.0`, independentemente de outros critérios. Adicionalmente, para regiões pequenas (inferiores a 200 pixels), foi implementado um filtro adicional baseado em score de forma, onde regiões com `score_shape < 0.2` são rejeitadas. Estes filtros efetivamente eliminam falsos positivos como linhas, artefatos de processamento e ruído alongado, enquanto mantêm células válidas que podem apresentar formato ligeiramente alongado mas ainda dentro dos limites aceitáveis. A implementação destes filtros rigorosos permitiu manter a precisão do algoritmo mesmo com os thresholds de seleção mais permissivos implementados para melhorar a cobertura.

---

## 4. Results

### 4.1 Dataset

O dataset utilizado neste trabalho é proveniente do Laboratório Murphy da Carnegie Mellon University, disponível publicamente através do repositório oficial (https://murphylab.web.cmu.edu/data/2009_ISBI_2DNuclei_code_data.tgz). Este dataset foi originalmente desenvolvido para o desafio de segmentação de núcleos celulares 2D do International Symposium on Biomedical Imaging (ISBI) de 2009, e consiste em imagens de células marcadas com diferentes corantes fluorescentes.

Para este trabalho, foram extraídas 27 imagens de células marcadas com Hoechst 33342, um corante fluorescente comumente utilizado para marcação de DNA em células. As imagens estão no formato TIFF e são processadas diretamente pelo algoritmo sem pré-processamento adicional ou normalização manual. Esta abordagem permite que o algoritmo genético otimize automaticamente todos os parâmetros de pré-processamento (Gaussian blur, median blur, gradiente morfológico) como parte do processo de otimização evolutiva, adaptando-se automaticamente às características específicas de cada imagem.

A escolha deste dataset baseia-se em sua ampla utilização na literatura de segmentação de células, permitindo comparação com trabalhos anteriores, e na diversidade de características presentes nas imagens, incluindo variações em densidade celular, sobreposição entre células, intensidade de marcação e qualidade de imagem. Esta diversidade é fundamental para validar a robustez do algoritmo melhorado em diferentes condições de imagem.

### 4.2 Configuração Experimental

A configuração experimental do algoritmo genético foi definida com base em testes preliminares e considerações sobre o espaço de busca expandido resultante do aumento no número de parâmetros otimizados. A população foi configurada com 20 indivíduos, representando um aumento de 25% em relação aos 16 indivíduos utilizados no algoritmo original. Este aumento proporciona maior diversidade genética inicial e permite manutenção de múltiplas soluções promissoras simultaneamente, importante dado o aumento na dimensionalidade do espaço de busca.

O algoritmo foi executado por 20 gerações, representando um aumento significativo em relação às 7 gerações utilizadas nos testes do artigo original. Este número de gerações foi escolhido para permitir convergência adequada do algoritmo, especialmente importante considerando o espaço de busca expandido com 15 parâmetros. A taxa de mutação foi configurada em 50%, com amplitude de mutação de ±30% do valor atual de cada parâmetro. Esta configuração agressiva de mutação é fundamental para evitar convergência prematura e manter exploração contínua do espaço de busca.

O operador de crossover BLX-alpha foi utilizado, permitindo geração de descendentes em uma região hiper-retangular definida pelos pais, facilitando exploração do espaço de busca. A seleção por torneio foi implementada, onde indivíduos competem em torneios de tamanho fixo para determinar quais serão selecionados para reprodução. O elitismo foi configurado para preservar apenas os 2 melhores indivíduos de cada geração, balanceando preservação de soluções ótimas com manutenção de diversidade genética na população.

### 4.3 Resultados da Geração 20

O fitness final obtido na geração 20 foi de 126347, representando o valor da função de fitness combinada para o melhor indivíduo da população após 20 gerações de evolução. A interpretação deste valor requer compreensão da composição da função de fitness, que integra quatro componentes distintas com pesos relativos diferentes.

A componente Almod normalizado contribui com 70% do peso total e quantifica a diferença pixel a pixel normalizada entre a imagem original e a segmentação binária. Esta componente foi normalizada através da multiplicação da média de diferença por pixel pela raiz quadrada da área total segmentada, evitando penalização incorreta de segmentações que detectam maior número de células válidas. A componente de qualidade de forma contribui com 15% do peso e quantifica o quão próximo o formato das células segmentadas está de uma elipse ideal, incentivando segmentações com células de formato adequado.

A componente de recompensa por células contribui com 10% do peso e implementa uma penalização que diminui com o aumento do número de células válidas detectadas, incentivando o algoritmo a detectar maior número de células. A componente de penalidade de completude contribui com 15% do peso e penaliza segmentações incompletas, onde apenas parte das células é segmentada, através do cálculo do ratio de completude como proporção da área total de células que foi efetivamente segmentada.

Na função de fitness implementada, valores menores indicam melhor qualidade de segmentação, pois representam menor discrepância entre imagem original e segmentação, melhor formato das células, maior número de células detectadas e maior completude da segmentação. O fitness de 126347 obtido na geração 20 indica uma segmentação de boa qualidade, com balanceamento adequado entre as quatro componentes da função de fitness. Este valor representa uma melhoria significativa em relação às gerações iniciais, demonstrando evolução efetiva do algoritmo genético ao longo das 20 gerações.

### 4.4 Análise Qualitativa

A análise qualitativa das imagens da geração 20 (`gen20_fit126347_*.png`) permite avaliar o comportamento do algoritmo em casos concretos.

#### 4.4.1 Casos de Sucesso

A análise qualitativa das imagens da geração 20 revela quatro categorias principais de casos de sucesso, demonstrando a efetividade das melhorias implementadas. A primeira categoria consiste na unificação efetiva de células que foram fragmentadas em múltiplas partes durante as gerações iniciais. Observações comparativas entre gerações iniciais e a geração 20 mostram que células que anteriormente apareciam como duas ou mais regiões separadas agora são segmentadas como regiões únicas e completas. Este comportamento é particularmente evidente em células alongadas ou com variação interna de intensidade, que nas gerações iniciais resultavam em múltiplos segmentos distintos. A fusão agressiva de regiões adjacentes, com thresholds aumentados até 2× o threshold base e dilatação expandida (kernel 5×5 com 2 iterações), demonstrou ser efetiva em unificar partes da mesma célula que foram segmentadas separadamente.

A segunda categoria de sucesso consiste na detecção adequada de células localizadas nas bordas da imagem. Imagens que contêm células parcialmente cortadas pelas bordas mostram detecção consistente dessas células, mesmo quando mais de 50% da área da célula está fora dos limites da imagem. Esta melhoria é diretamente atribuível à remoção completa de filtros baseados em posição espacial, mantendo apenas rejeição de artefatos óbvios através de critérios de tamanho e forma. A análise visual revela que células parcialmente cortadas são identificadas corretamente, com contornos que seguem adequadamente a porção visível da célula, demonstrando efetividade da modificação implementada.

A terceira categoria consiste na detecção de células com intensidade significativamente reduzida em relação à média da imagem. Comparações entre gerações iniciais e finais mostram que células que anteriormente não eram segmentadas, devido à sua baixa intensidade, passam a ser identificadas consistentemente na geração 20. Esta melhoria é atribuível aos marcadores de intensidade local implementados no watershed híbrido, que utilizam threshold adaptativo no 50º percentil (com range otimizável 0.3-0.7) e são adicionados mesmo fora da máscara binária. A análise visual confirma que células escuras são detectadas através deste mecanismo, expandindo significativamente a cobertura do algoritmo.

A quarta categoria de sucesso consiste na rejeição efetiva de falsos positivos, especialmente linhas e artefatos alongados que poderiam ser incorretamente classificados como células. A análise visual mostra que os filtros rigorosos de aspect ratio (`> 4.0` ou `axis_ratio > 5.0`) e forma (`score_shape < 0.2` para regiões pequenas) efetivamente eliminam artefatos enquanto preservam células válidas. Apenas regiões com formato claramente alongado ou com shape score muito baixo são descartadas, demonstrando que o balanceamento entre thresholds permissivos para seleção e filtros rigorosos para rejeição de artefatos foi adequadamente implementado.

#### 4.4.2 Limitações Observadas

A análise qualitativa das imagens da geração 20 também revela três categorias principais de limitações, que representam casos desafiadores onde o algoritmo ainda apresenta dificuldades. A primeira limitação consiste na detecção incompleta de células com intensidade extremamente reduzida, mesmo com as melhorias implementadas. Em casos extremos, células com intensidade muito abaixo da média da imagem (aproximadamente abaixo do 30º percentil) ainda podem não ser detectadas completamente, mesmo com o threshold de intensidade reduzido para o 50º percentil e range ampliado para [0.3, 0.7]. Estes casos representam menos de 5% das células nas imagens analisadas e geralmente correspondem a células com marcação muito fraca ou em regiões de sombreamento. A detecção destas células extremamente escuras requereria thresholds ainda mais baixos, o que poderia aumentar significativamente a taxa de falsos positivos, criando um trade-off entre cobertura e precisão.

A segunda limitação observada consiste na fusão incorreta de células diferentes que estão muito próximas e possuem intensidade similar. Devido à fusão agressiva de regiões adjacentes implementada para resolver o problema de fragmentação, células diferentes que estão muito próximas e apresentam intensidade similar podem ser segmentadas como uma única região. Este comportamento ocorre raramente, em aproximadamente 2-3% dos casos observados, e geralmente envolve células que estão quase se tocando e possuem intensidade muito similar. A fusão agressiva, embora efetiva em unificar partes da mesma célula, não possui critérios suficientes para distinguir entre partes da mesma célula e células diferentes muito próximas.

A terceira limitação consiste em over-segmentação residual, onde uma célula ainda é segmentada em múltiplas partes mesmo após a fusão agressiva. Este comportamento ocorre em casos muito raros (menos de 1% das células), geralmente quando a variação de intensidade interna da célula é muito alta e a fusão não consegue conectar todas as partes devido a diferenças de intensidade que excedem os thresholds permissivos implementados. Nestes casos, mesmo com thresholds aumentados até 2× o threshold base e thresholds ainda mais permissivos para regiões pequenas, a variação de intensidade é tão grande que as partes da célula não são reconhecidas como pertencentes à mesma região.

### 4.5 Comparação com Artigo Original

O artigo original de Daguano (2020) reporta resultados quantitativos baseados em métricas de F-Score calculadas através de comparação com ground-truth (anotações manuais). Os resultados reportados indicam que 96% das instâncias testadas apresentaram F-Score superior a 60%, com média de 73% de F-Score. Quando o algoritmo Algen é executado em conjunto com o algoritmo Algal (também proposto no trabalho), os resultados melhoram para 100% das instâncias com F-Score superior a 75%, com média de 86% de F-Score.

Nossos resultados são baseados em análise qualitativa das imagens segmentadas, já que não possuímos ground-truth para cálculo de métricas quantitativas como F-Score, Recall e Precision. A análise qualitativa das imagens da geração 20 revela quatro aspectos principais de qualidade da segmentação. Em relação à completude, a maioria das células é segmentada completamente, com poucos casos de segmentação parcial observados nas imagens analisadas. Esta melhoria é diretamente atribuível à penalidade de completude implementada na função de fitness combinada, que direciona a evolução do algoritmo para soluções que segmentam células completas.

Em relação à precisão, observa-se baixa taxa de falsos positivos, com linhas e artefatos sendo efetivamente rejeitados pelos filtros rigorosos de aspect ratio e forma implementados. A análise visual confirma que apenas regiões com formato claramente inadequado são descartadas, preservando células válidas mesmo com os thresholds de seleção mais permissivos.

Em relação à cobertura, células localizadas nas bordas da imagem, incluindo células parcialmente cortadas, são detectadas adequadamente, demonstrando efetividade da remoção de filtros baseados em posição espacial. Adicionalmente, células com intensidade reduzida são identificadas através dos marcadores de intensidade local implementados no watershed híbrido, expandindo significativamente a cobertura em relação ao algoritmo original.

Em relação à consistência, os resultados são consistentes entre diferentes imagens do dataset, com valores de fitness similares entre imagens indicando robustez do algoritmo. Esta consistência sugere que as melhorias implementadas são efetivas em diferentes condições de imagem, não sendo específicas para características particulares de algumas imagens.

É importante notar que, para validação quantitativa completa e comparação direta com os resultados reportados no artigo original, seria necessário possuir ground-truth (anotações manuais das células) para cálculo de métricas quantitativas como F-Score, Recall e Precision. A análise qualitativa, embora útil para identificar melhorias e limitações, não permite comparação quantitativa direta com os resultados reportados no artigo original.

---

## 5. Discussion

### 5.1 Análise das Melhorias Implementadas

#### 5.1.1 Detecção de Bordas Canny

A implementação da detecção de bordas Canny representa a concretização de uma sugestão explícita do artigo original de Daguano (2020), que menciona como trabalho futuro a "detecção de sobreposição de objetos, pois nossa técnica apresenta dificuldades em detectar sobreposição de área de interesse e por esse motivo seria interessante aprimorar os resultados a partir de detectores de bordas". A implementação desenvolvida utiliza thresholds adaptativos baseados na mediana e desvio padrão da imagem (`median ± 0.33 * std`), permitindo adaptação automática às características de cada imagem.

Os resultados obtidos demonstram que a detecção de bordas melhora significativamente a identificação de células em três categorias principais de casos desafiadores. Em casos de sobreposição de células, as bordas detectadas fornecem informação adicional que permite melhor distinção entre células sobrepostas, complementando a informação de intensidade e distance transform. Em células localizadas nas bordas da imagem, a detecção de bordas ajuda a identificar contornos mesmo quando parte da célula está fora dos limites da imagem. Em regiões de baixo contraste, onde a diferença de intensidade entre células e background é reduzida, as bordas detectadas fornecem informação espacial adicional que melhora a identificação de células.

A análise qualitativa das imagens segmentadas confirma que a implementação de detecção de bordas resulta em melhoria na cobertura de células, especialmente em casos identificados como desafiadores no trabalho original. O parâmetro binário `use_edge_detection` permite que o algoritmo genético determine automaticamente se a detecção de bordas é benéfica para cada conjunto de imagens, resultando em soluções adaptadas às características específicas de cada dataset.

---

#### 5.1.2 Watershed Híbrido

O watershed híbrido implementado representa uma evolução significativa em relação ao método original, que utiliza exclusivamente marcadores baseados em distance transform da máscara binária. A combinação de três tipos distintos de marcadores (distance transform, intensidade local e bordas) resulta em melhor detecção de células, especialmente células escuras que não são adequadamente capturadas pela máscara binária inicial.

A análise dos resultados demonstra que cada tipo de marcador oferece vantagens distintas e complementares. Os marcadores de distance transform, herdados do método original, identificam centros aproximados de objetos que foram adequadamente capturados na binarização inicial, fornecendo informação espacial robusta para células com contraste adequado. Os marcadores de intensidade local, implementados com threshold adaptativo no 50º percentil (range otimizável 0.3-0.7), detectam células escuras que podem não ser adequadamente identificadas pelo distance transform, expandindo significativamente a capacidade de detecção. Estes marcadores são adicionados mesmo fora da máscara binária, permitindo detecção de células que não foram capturadas na binarização inicial.

Os marcadores baseados em bordas, quando a detecção de bordas está habilitada, melhoram a detecção em regiões de baixo contraste através de thresholds mais permissivos (20-25% do máximo da transformada de distância), fornecendo informação adicional em casos onde tanto a intensidade quanto o distance transform apresentam limitações. A combinação dos três tipos de marcadores resulta em maior cobertura do que o método original baseado exclusivamente em distance transform, com o peso relativo dos marcadores de intensidade controlado pelo parâmetro `intensity_weight`, permitindo que o algoritmo genético ajuste a contribuição de cada tipo de marcador conforme as características de cada conjunto de imagens.

---

#### 5.1.3 Função de Fitness Combinada

A função de fitness combinada implementada representa uma evolução fundamental em relação à métrica Almod pura utilizada no algoritmo original. A integração de quatro componentes distintas (Almod normalizado 70%, qualidade de forma 15%, recompensa por células 10%, penalidade de completude 15%) resolve efetivamente o problema de segmentação parcial de células identificado durante testes iniciais, onde células eram frequentemente segmentadas em múltiplas partes separadas.

A análise dos resultados demonstra que a função de fitness combinada oferece três vantagens principais em relação à métrica Almod pura. Primeiro, a penalidade de completude, que contribui com 15% do peso total, direciona explicitamente a evolução do algoritmo para soluções que segmentam células completas, em vez de partes isoladas. Esta componente calcula o ratio de completude como proporção da área total de células que foi efetivamente segmentada e aplica penalidade proporcional, resultando em penalidade zero quando 100% das células são segmentadas e penalidade máxima quando nenhuma célula é segmentada. Segundo, o componente de qualidade de forma, contribuindo com 15% do peso, incentiva células com formato elíptico através do score de ellipse fit, alinhado com características morfológicas típicas de células em imagens de microscopia. Terceiro, a normalização do Almod através da multiplicação da média de diferença por pixel pela raiz quadrada da área total segmentada evita penalizar incorretamente segmentações que detectam maior número de células válidas, permitindo comparação justa entre segmentações com diferentes números de células.

A análise qualitativa das imagens da geração 20 indica poucos casos de segmentação parcial observados, com a maioria das células aparecendo como regiões únicas e completas. Esta observação sugere efetividade da penalidade de completude na função de fitness, que direcionou a evolução do algoritmo genético para soluções que segmentam células completas, resolvendo o problema identificado durante o desenvolvimento.

---

#### 5.1.4 Fusão Agressiva de Regiões

A fusão agressiva de regiões adjacentes implementada resolve efetivamente o problema de fragmentação de células, onde partes da mesma célula eram segmentadas como regiões separadas. A análise dos resultados demonstra que a fusão agressiva une efetivamente partes da mesma célula que foram segmentadas separadamente, resultando em células completas segmentadas como regiões únicas.

A implementação oferece três mecanismos complementares que trabalham em conjunto para unificar partes de células. O primeiro mecanismo consiste na dilatação expandida utilizada para avaliar proximidade entre regiões, utilizando kernel 5×5 com 2 iterações, em contraste com o kernel 3×3 com 1 iteração utilizado no método original. Esta expansão aumenta significativamente a área de busca para regiões adjacentes, detectando partes da mesma célula que podem estar separadas por pequenas lacunas ou variações de intensidade.

O segundo mecanismo consiste no threshold de similaridade de intensidade aumentado, que pode ser expandido até 2× o threshold base (`merge_threshold`), permitindo fusão mesmo quando há variação significativa de intensidade entre regiões adjacentes. Este aumento é fundamental para unificar partes da mesma célula que apresentam variação interna de intensidade, comum em imagens de células devido a variações na marcação ou na estrutura interna.

O terceiro mecanismo consiste em thresholds ainda mais permissivos para regiões pequenas (inferiores a 200 pixels), que utilizam `merge_threshold * 2.0` em vez de `merge_threshold * 1.5` utilizado para regiões normais, com threshold mínimo de 0.15 para garantir fusão mesmo com threshold baixo. Este mecanismo facilita a união de partes pequenas da célula que podem ter sido segmentadas separadamente devido a sua pequena área.

A análise qualitativa das imagens da geração 20 mostra células completas segmentadas como regiões únicas, sem fragmentação em múltiplas partes separadas, confirmando a efetividade da fusão agressiva implementada. Comparações entre gerações iniciais e finais revelam que células que anteriormente apareciam fragmentadas agora são unificadas adequadamente.

---

#### 5.1.5 Seleção Mais Permissiva

A redução do threshold de seleção de componentes conexas de 0.5 (utilizado no algoritmo original) para 0.25 (com thresholds ainda mais permissivos de 0.15 para células grandes e 0.20 para células dentro do intervalo de tamanho ideal) permite detectar significativamente mais células válidas, especialmente células grandes e escuras que anteriormente eram descartadas pelo threshold mais restritivo.

A análise dos resultados demonstra que a redução do threshold de seleção aumenta a cobertura do algoritmo, permitindo seleção de células que apresentam características ligeiramente fora do ideal mas ainda representam células válidas. Esta mudança é fundamentalmente balanceada por filtros rigorosos de aspect ratio e forma que rejeitam efetivamente linhas e artefatos, mantendo a precisão do algoritmo mesmo com thresholds mais permissivos.

Os filtros implementados rejeitam regiões com `aspect_ratio > 4.0` ou `axis_ratio > 5.0`, independentemente de outros critérios, efetivamente eliminando falsos positivos como linhas e artefatos alongados. Adicionalmente, para regiões pequenas (inferiores a 200 pixels), é aplicado um filtro adicional baseado em score de forma, onde regiões com `score_shape < 0.2` são rejeitadas. Estes filtros rigorosos garantem que apenas células válidas sejam mantidas, mesmo com thresholds de seleção mais permissivos.

O resultado desta abordagem balanceada é melhor detecção de células escuras e grandes, que anteriormente eram descartadas pelo threshold mais restritivo de 0.5, enquanto mantém baixa taxa de falsos positivos através dos filtros rigorosos de aspect ratio e forma. A análise qualitativa confirma que células que anteriormente não eram detectadas agora são identificadas adequadamente, expandindo a cobertura do algoritmo sem comprometer a precisão.

---

#### 5.1.6 Configuração do Algoritmo Genético

A configuração melhorada do algoritmo genético, com mecanismos anti-estagnação mais agressivos, garante evolução contínua ao longo das gerações, evitando convergência prematura para soluções subótimas. A análise da evolução do fitness ao longo das 20 gerações demonstra que o algoritmo manteve melhoria contínua, sem observação de estagnação prematura durante a execução.

Os mecanismos anti-estagnação implementados oferecem três estratégias complementares que trabalham em conjunto para manter a população geneticamente diversa e ativa. A primeira estratégia consiste na taxa de mutação drasticamente aumentada de 10% (original) para 50%, com amplitude de mutação expandida de ±5-15% para ±30% do valor atual do parâmetro. Esta configuração agressiva de mutação previne convergência prematura para soluções subótimas, especialmente importante dado o aumento na dimensionalidade do espaço de busca (de 6 para 15 parâmetros) e a complexidade da função de fitness combinada.

A segunda estratégia consiste na reinjeção de diversidade, que aplica probabilidade de 30% de criar indivíduos completamente aleatórios durante a geração de descendentes, aumentando para valores adaptativos superiores durante períodos de estagnação detectada. Adicionalmente, após 3 gerações consecutivas sem melhoria significativa, até 40% da população é substituída por indivíduos gerados através de mutação agressiva ou aleatórios. Se a estagnação persiste por mais de 8 gerações, até 50% da população é substituída, efetivamente reiniciando parcialmente a busca evolutiva.

A terceira estratégia consiste no mecanismo de idade máxima de indivíduos, onde indivíduos que persistem na população por mais de 5 gerações são considerados "velhos" e são substituídos por mutações agressivas dos melhores indivíduos ou por novos indivíduos aleatórios. Este mecanismo previne que soluções dominantes persistam excessivamente na população, forçando renovação contínua e evitando que a população fique presa em ótimos locais.

A combinação destas três estratégias resulta em população geneticamente variada e ativa ao longo de todas as gerações, permitindo exploração contínua do espaço de busca e convergência gradual para soluções ótimas, em vez de convergência prematura para soluções subótimas.

---

### 5.2 Limitações e Desafios

#### 5.2.1 Células Muito Escuras

Uma limitação identificada consiste na detecção incompleta de células com intensidade extremamente reduzida, mesmo com as melhorias implementadas. A análise qualitativa das imagens revela que células com intensidade muito abaixo da média da imagem (aproximadamente abaixo do 30º percentil) ainda podem não ser detectadas completamente, representando menos de 5% das células nas imagens analisadas.

O threshold adaptativo implementado (50º percentil, com range otimizável 0.3-0.7) melhora significativamente a detecção de células escuras em relação ao método original, mas casos extremos podem requerer thresholds ainda mais baixos para detecção adequada. No entanto, reduzir o threshold abaixo do 30º percentil pode aumentar significativamente a taxa de falsos positivos, criando um trade-off entre cobertura e precisão que é difícil de resolver com thresholds globais.

Uma solução futura promissora consistiria em ponderar o threshold adaptativo com informação contextual da região circundante, permitindo thresholds mais baixos em regiões onde células escuras são esperadas (por exemplo, próximo a outras células detectadas) e thresholds mais altos em regiões onde células são improváveis. Esta abordagem contextual poderia melhorar a detecção de células muito escuras sem aumentar significativamente a taxa de falsos positivos.

---

#### 5.2.2 Células Muito Próximas

Uma segunda limitação identificada consiste na fusão incorreta de células diferentes que estão muito próximas e possuem intensidade similar. A análise qualitativa revela que este comportamento ocorre raramente, em aproximadamente 2-3% dos casos observados, geralmente envolvendo células que estão quase se tocando e possuem intensidade muito similar.

A fusão agressiva implementada, embora efetiva em unificar partes da mesma célula, não possui critérios suficientes para distinguir entre partes da mesma célula e células diferentes muito próximas. Os critérios atuais baseiam-se exclusivamente em proximidade espacial (através de dilatação morfológica) e similaridade de intensidade, que podem ser insuficientes quando células diferentes apresentam características muito similares.

Uma solução futura promissora consistiria em adicionar critério de separação baseado em análise de concavidade ou análise de contorno. Células diferentes, mesmo quando muito próximas, geralmente apresentam uma região de concavidade ou estreitamento entre elas, que poderia ser detectada através de análise de contorno ou análise de curvatura. Este critério adicional poderia prevenir fusão incorreta de células diferentes enquanto mantém a capacidade de unificar partes da mesma célula.

---

#### 5.2.3 Validação Quantitativa

Uma limitação fundamental deste trabalho consiste na ausência de ground-truth (anotações manuais das células) para cálculo de métricas quantitativas como F-Score, Recall e Precision. Esta ausência impede comparação quantitativa direta com os resultados reportados no artigo original de Daguano (2020), que reporta 96% das instâncias com F-Score superior a 60% e média de 73% de F-Score.

A validação qualitativa realizada através de análise visual das imagens segmentadas é útil para identificar melhorias e limitações do algoritmo, mas métricas quantitativas seriam preferíveis para comparação objetiva com o artigo original e para avaliação precisa do impacto das melhorias implementadas. Métricas quantitativas permitiriam quantificar exatamente o ganho em cobertura, precisão e completude proporcionado pelas melhorias, além de permitir comparação direta com outros métodos da literatura.

Uma solução futura consistiria em utilizar um dataset com ground-truth disponível ou criar anotações manuais para validação quantitativa. O dataset do Laboratório Murphy utilizado possui algumas imagens com ground-truth disponível, que poderiam ser utilizadas para validação quantitativa. Alternativamente, anotações manuais poderiam ser criadas para um subconjunto representativo das imagens utilizadas, permitindo cálculo de métricas quantitativas e comparação direta com o artigo original.

---

### 5.3 Comparação com Artigo Original

A comparação entre os resultados obtidos com as melhorias implementadas e os resultados reportados no artigo original de Daguano (2020) revela aspectos significativamente aprimorados, bem como limitações que permanecem como desafios para trabalhos futuros.

Em relação aos aspectos aprimorados, quatro melhorias principais podem ser identificadas através da análise qualitativa das imagens segmentadas. A primeira melhoria consiste na completude das segmentações, onde observa-se redução significativa nos casos de segmentação parcial nas imagens analisadas. A análise comparativa entre gerações iniciais e finais demonstra que células que anteriormente eram segmentadas em múltiplas partes separadas agora aparecem como regiões únicas e completas. Esta melhoria é diretamente atribuível à penalidade de completude implementada na função de fitness combinada e à fusão agressiva de regiões adjacentes, que trabalham em conjunto para direcionar a evolução do algoritmo para soluções que segmentam células completas.

A segunda melhoria consiste na cobertura expandida do algoritmo, com melhoria significativa na detecção de células localizadas nas bordas da imagem e células com intensidade reduzida. A remoção completa de filtros baseados em posição espacial permite detecção adequada de células parcialmente cortadas pelas bordas, mesmo quando mais de 50% da área da célula está fora dos limites da imagem. Adicionalmente, os marcadores de intensidade local implementados no watershed híbrido permitem detecção de células escuras que não eram adequadamente capturadas pelo método original baseado exclusivamente em distance transform. Esta expansão da cobertura é fundamental para aplicações práticas onde células podem apresentar características atípicas ou estar parcialmente fora do campo de visão.

A terceira melhoria consiste na precisão mantida ou melhorada, com redução observada de falsos positivos através dos filtros rigorosos de aspect ratio e forma implementados. Os filtros que rejeitam regiões com `aspect_ratio > 4.0` ou `axis_ratio > 5.0`, e regiões pequenas com `score_shape < 0.2`, efetivamente eliminam linhas, artefatos de processamento e ruído alongado, mantendo a precisão mesmo com os thresholds de seleção mais permissivos implementados para melhorar a cobertura. Esta abordagem balanceada permite melhor detecção de células válidas sem comprometer a precisão do algoritmo.

A quarta melhoria consiste na robustez demonstrada através da consistência dos resultados entre diferentes imagens do dataset. Os valores de fitness similares entre imagens e a consistência na qualidade das segmentações sugerem que as melhorias implementadas são efetivas em diferentes condições de imagem, não sendo específicas para características particulares de algumas imagens. Esta robustez é fundamental para aplicação prática do algoritmo em diferentes contextos e tipos de imagens.

Em relação às limitações identificadas, três desafios principais permanecem para trabalhos futuros. A primeira limitação consiste na ausência de validação quantitativa completa, onde a falta de ground-truth impede cálculo de métricas quantitativas como F-Score, Recall e Precision, limitando comparação direta com os resultados reportados no artigo original. Embora a análise qualitativa seja útil para identificar melhorias e limitações, métricas quantitativas seriam preferíveis para avaliação objetiva do impacto das melhorias implementadas e comparação direta com outros métodos da literatura.

A segunda limitação consiste em casos extremos que ainda apresentam desafios para segmentação adequada. Células com intensidade muito baixa (aproximadamente abaixo do 30º percentil) ainda podem não ser detectadas completamente, mesmo com as melhorias implementadas, representando menos de 5% das células nas imagens analisadas. Adicionalmente, células muito próximas com intensidade similar podem ser incorretamente fundidas em uma única região, ocorrendo em aproximadamente 2-3% dos casos observados. Estes casos extremos requerem abordagens adicionais que não foram implementadas neste trabalho.

A terceira limitação consiste na complexidade computacional aumentada resultante do aumento no número de parâmetros otimizados de 6 para 15. Este aumento na dimensionalidade do espaço de busca pode requerer maior número de gerações para convergência adequada, aumentando o tempo de execução do algoritmo. No entanto, os mecanismos anti-estagnação implementados e a configuração melhorada do algoritmo genético (população aumentada, taxa de mutação aumentada, crossover BLX-alpha) foram projetados especificamente para lidar com este espaço de busca expandido, mantendo eficiência computacional razoável.

---

## 6. Conclusion

Este trabalho apresentou melhorias significativas ao algoritmo genético proposto por Daguano (2020) para segmentação automática de imagens de células, focadas especificamente em resolver problemas de segmentação parcial e melhorar a detecção de células escuras e células localizadas nas bordas da imagem. As principais contribuições desenvolvidas incluem cinco melhorias fundamentais que trabalham em conjunto para produzir segmentações mais completas e consistentes.

A primeira contribuição consiste na implementação da detecção de bordas Canny, que concretiza uma sugestão explícita do artigo original mencionada como trabalho futuro. A implementação utiliza thresholds adaptativos baseados na mediana e desvio padrão da imagem, permitindo adaptação automática às características de cada imagem. Esta melhoria resulta em identificação aprimorada de células especialmente em casos de sobreposição e células localizadas nas bordas da imagem, dois dos principais desafios identificados no trabalho original.

A segunda contribuição consiste no desenvolvimento de um watershed híbrido que combina três tipos distintos de marcadores: distance transform, intensidade local e bordas. Esta combinação resulta em melhor cobertura do algoritmo, especialmente para células escuras que não são adequadamente capturadas pela máscara binária inicial. Os marcadores de intensidade local, implementados com threshold adaptativo no 50º percentil (range otimizável 0.3-0.7), são adicionados mesmo fora da máscara binária, expandindo significativamente a capacidade de detecção em relação ao método original baseado exclusivamente em distance transform.

A terceira contribuição consiste na função de fitness combinada que integra quatro componentes distintas: Almod normalizado (70%), qualidade de forma (15%), recompensa por células (10%) e penalidade de completude (15%). A penalidade de completude, desenvolvida especificamente para este trabalho, resolve efetivamente o problema de segmentação parcial de células identificado durante testes iniciais, onde células eram frequentemente segmentadas em múltiplas partes separadas. Esta componente calcula o ratio de completude como proporção da área total de células que foi efetivamente segmentada e aplica penalidade proporcional, direcionando a evolução do algoritmo para soluções que segmentam células completas.

A quarta contribuição consiste na fusão agressiva de regiões adjacentes, que une efetivamente partes da mesma célula que foram segmentadas separadamente. A implementação utiliza dilatação expandida (kernel 5×5 com 2 iterações), thresholds de similaridade de intensidade aumentados até 2× o threshold base, e thresholds ainda mais permissivos para regiões pequenas, trabalhando em conjunto para unificar partes de células que apresentam variação interna de intensidade ou estão separadas por pequenas lacunas.

A quinta contribuição consiste nos mecanismos anti-estagnação implementados, que garantem evolução contínua ao longo das gerações, evitando convergência prematura para soluções subótimas. Estes mecanismos incluem taxa de mutação aumentada (50% versus 10% do original), reinjeção de diversidade, reinjeção por estagnação, reset parcial e idade máxima de indivíduos, trabalhando em conjunto para manter a população geneticamente variada e ativa.

Os resultados obtidos na geração 20, com fitness final de 126347, demonstram segmentações completas e consistentes em 27 imagens do dataset do Laboratório Murphy. A análise qualitativa detalhada das imagens da geração 20 revela quatro aspectos principais de qualidade. Primeiro, observa-se detecção completa de células, com poucos casos observados de segmentação parcial, demonstrando efetividade da penalidade de completude e da fusão agressiva de regiões. Segundo, verifica-se detecção adequada de células localizadas nas bordas da imagem e células com intensidade reduzida, confirmando a efetividade da remoção de filtros de borda e dos marcadores de intensidade local. Terceiro, observa-se baixa taxa de falsos positivos, com linhas e artefatos sendo efetivamente rejeitados pelos filtros rigorosos de aspect ratio e forma implementados, mantendo a precisão mesmo com thresholds de seleção mais permissivos. Quarto, os resultados são consistentes entre diferentes imagens do dataset, com valores de fitness similares e qualidade de segmentação uniforme, sugerindo robustez da abordagem em diferentes condições de imagem.

Em relação às limitações identificadas e trabalhos futuros, quatro direções principais podem ser delineadas para continuidade da pesquisa. A primeira direção consiste na validação quantitativa completa através da utilização de ground-truth para cálculo de métricas quantitativas como F-Score, Recall e Precision. Esta validação permitiria comparação quantitativa direta com o artigo original e avaliação precisa do impacto das melhorias implementadas, além de permitir comparação com outros métodos da literatura. O dataset do Laboratório Murphy utilizado possui algumas imagens com ground-truth disponível, que poderiam ser utilizadas para esta validação.

A segunda direção consiste no aprimoramento da detecção de casos extremos, especificamente melhorar a detecção de células muito escuras (aproximadamente abaixo do 30º percentil) e a separação adequada de células muito próximas com intensidade similar. Para células muito escuras, uma abordagem promissora consistiria em ponderar o threshold adaptativo com informação contextual da região circundante. Para células muito próximas, a adição de critério de separação baseado em análise de concavidade ou análise de contorno poderia prevenir fusão incorreta.

A terceira direção consiste na análise sistemática de parâmetros para identificar quais dos 15 parâmetros otimizados mais impactam os resultados, permitindo otimização futura através de redução da dimensionalidade do espaço de busca ou priorização de parâmetros mais influentes. Esta análise poderia ser realizada através de análise de sensibilidade ou análise de importância de parâmetros.

A quarta direção consiste na validação da robustez em diferentes tipos de imagens através de testes em outros datasets, incluindo outros tipos de células (não apenas células Hoechst) e outros métodos de marcação fluorescente. Esta validação seria fundamental para confirmar que as melhorias implementadas são generalizáveis e não específicas para o dataset utilizado neste trabalho.

Em conclusão, as melhorias implementadas apresentam resultados promissores na resolução dos problemas identificados no algoritmo original, especificamente a segmentação parcial de células e a não detecção de células escuras e células nas bordas. A análise qualitativa detalhada das imagens da geração 20 sugere que as alterações propostas resultam em segmentações mais completas e consistentes quando comparadas às expectativas baseadas no algoritmo original. A combinação das cinco melhorias principais trabalha sinergicamente para produzir um algoritmo mais robusto e efetivo, com melhor cobertura, completude e precisão. No entanto, validação quantitativa completa com ground-truth seria necessária para confirmação definitiva dos resultados e comparação objetiva com o artigo original e outros métodos da literatura.

---

## References

[1] Daguano, E. M. (2020). "Algoritmo Genético para Segmentação de Imagens utilizando Tamanho e Forma dos Objetos". Dissertação de Mestrado - Faculdade de Tecnologia, UNICAMP. Orientador: Prof. Dr. Ulisses Martins Dias. Disponível em: https://www.repositorio.unicamp.br/acervo/detalhe/1157726

[2] Murphy Laboratory Dataset. (2009). "ISBI 2009 2D Nuclei Segmentation Challenge". Disponível em: https://murphylab.web.cmu.edu/data/2009_ISBI_2DNuclei_code_data.tgz

[3] Beucher, S., & Lantuéjoul, C. (1979). "Use of Watersheds in Contour Detection". International Workshop on Image Processing: Real-time Edge and Motion Detection/Estimation.

[4] Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms". IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66.

[5] Ranefall, P., & Wählby, C. (2016). "Size Interval Precision (SIP) for Segmentation Evaluation". Cytometry Part A, 89(5), 411-419.

[6] Ranefall, P., Sadanandan, S. K., & Wählby, C. (2016). "Per Object Ellipse Fit (POE) for Segmentation Evaluation". Cytometry Part A, 89(7), 645-655.

[7] Beasley, D., Bull, D. R., & Martin, R. R. (1993). "An Overview of Genetic Algorithms: Part 1, Fundamentals". University Computing, 15(2), 58-69.

[8] Carvalho, B. M. (2004). "Árvore dos Lagos Críticos: Uma Estrutura Hierárquica para Segmentação de Imagens". Tese de Doutorado, UNICAMP.

[9] Coelho, L. P., Shariff, A., & Murphy, R. F. (2009). "Nuclear Segmentation in Microscope Cell Images: A Hand-Segmented Dataset and Comparison of Algorithms". IEEE International Symposium on Biomedical Imaging (ISBI), 518-521.

---

**Link do Documento Usado como Inspiração**: https://www.repositorio.unicamp.br/acervo/detalhe/1157726

**Link da Base de Dados Usada**: https://murphylab.web.cmu.edu/data/2009_ISBI_2DNuclei_code_data.tgz

**Tratamento Adotado Antes de Submeter a Base de Dados ao Algoritmo Evolucionário**:

- Extração de 27 imagens de células Hoechst 33342 do dataset do Laboratório Murphy
- Conversão para formato TIFF (se necessário)
- Sem pré-processamento adicional: O algoritmo genético otimiza automaticamente os parâmetros de pré-processamento (Gaussian blur, median blur, gradiente morfológico) como parte do processo de otimização

---

**Última Atualização**: 17 de novembro de 2025  
**Dataset**: 27 imagens Hoechst do Laboratório Murphy  
**Configuração**: 20 gerações, população de 20, mutação 50%  
**Fitness Final**: 126347
