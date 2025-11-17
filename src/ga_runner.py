"""
Executor principal do algoritmo genético.
"""

import os
import sys
import random
import numpy as np
from datetime import datetime
import json

import config
import genetic_algorithm
import pipeline
import logger_utils
import results
import hashlib


def get_individual_hash(individual):
    """
    Cria um hash único para um indivíduo baseado em seus parâmetros.
    Isso permite rastrear se um indivíduo específico está se repetindo.
    
    Args:
        individual: Dicionário com parâmetros do indivíduo
    
    Returns:
        String hash única
    """
    # Criar representação ordenada dos parâmetros
    # Arredondar floats para 4 casas decimais para detectar "iguais"
    params_str = {}
    for k in sorted(individual.keys()):
        v = individual[k]
        if isinstance(v, float):
            params_str[k] = round(v, 4)
        else:
            params_str[k] = v
    
    # Criar JSON string e hash
    params_json = json.dumps(params_str, sort_keys=True)
    return hashlib.md5(params_json.encode()).hexdigest()


def run_genetic_algorithm(images, names, log_file=None, save_generation_images=False, timestamp=None):
    """
    Executa o algoritmo genético completo.
    Gera logs detalhados de cada geração e salva em arquivo.
    
    Args:
        images: Lista de imagens
        names: Lista de nomes das imagens
        log_file: Arquivo de log (opcional)
        save_generation_images: Se True, salva imagens do melhor de cada geração
        timestamp: Timestamp para organizar pastas de imagens
    
    Returns:
        tupla: (melhor indivíduo, melhor fitness, histórico)
    """
    # Configurar logger se arquivo fornecido
    original_stdout = sys.stdout
    if log_file:
        logger = logger_utils.TeeLogger(log_file)
        sys.stdout = logger
    
    # Diretório base para imagens de gerações
    if save_generation_images and timestamp:
        gen_images_dir = os.path.join(config.OUTPUT_DIR, "generation_results", timestamp)
        os.makedirs(gen_images_dir, exist_ok=True)
    else:
        gen_images_dir = None
    
    try:
        # Inicializar sementes
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        
        # Inicializar população
        population = [genetic_algorithm.create_random_individual(config.PARAM_RANGES) 
                     for _ in range(config.POP_SIZE)]
        for ind in population:
            genetic_algorithm.normalize_weights(ind)
            genetic_algorithm.fix_size_constraints(ind)
        
        best_global = None
        best_global_fitness = float('inf')
        generation_history = []
        generations_without_improvement = 0
        
        # CORREÇÃO: Rastrear idade dos indivíduos (quantas gerações sobreviveram)
        # Usar uma representação única do indivíduo (hash dos parâmetros) como chave
        individual_age_tracker = {}  # {individual_hash: age}
        MAX_INDIVIDUAL_AGE = 5  # Matar indivíduo após 5 gerações
        
        print("=" * 80)
        print("ALGORITMO GENÉTICO - EVOLUÇÃO COMPLETA")
        print("=" * 80)
        print(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"População: {config.POP_SIZE}")
        print(f"Gerações: {config.NUM_GENERATIONS}")
        print(f"Imagens: {len(images)}")
        print(f"Taxa de Mutação: {config.MUTATION_RATE} (50% - aumentada para evitar convergência prematura)")
        print(f"Amplitude de Mutação: ±30% (aumentada de ±15%)")
        print(f"Elitismo: {config.ELITISM}")
        print(f"Reinjeção de Diversidade: {config.DIVERSITY_REINJECTION_RATE*100}% (ocasional)")
        print(f"Reinjeção por Estagnação: Após {config.DIVERSITY_STAGNATION_THRESHOLD} gerações sem melhoria")
        print(f"Pesos Fitness: Almod {config.FITNESS_WEIGHT_ALMOD*100}% + Qualidade Forma {config.FITNESS_WEIGHT_QUALITY*100}% (ajustado para melhor cobertura)")
        print("=" * 80)
        print()
        
        for gen in range(config.NUM_GENERATIONS):
            print("=" * 80)
            print(f"GERAÇÃO {gen+1}/{config.NUM_GENERATIONS}")
            print("=" * 80)
            
            # Avaliar população
            print(f"\n[1/4] Avaliando {config.POP_SIZE} indivíduos...")
            fitnesses = []
            for i, ind in enumerate(population):
                fit = pipeline.evaluate_individual(ind, images, names)
                fitnesses.append(fit)
                print(f"  Indivíduo {i+1:2d}/{config.POP_SIZE}: fitness = {fit:,.2f}")
            
            # Estatísticas da geração
            print(f"\n[2/4] Estatísticas da Geração:")
            print(f"  Melhor fitness:  {min(fitnesses):,.2f}")
            print(f"  Pior fitness:    {max(fitnesses):,.2f}")
            print(f"  Média fitness:   {np.mean(fitnesses):,.2f}")
            print(f"  Desvio padrão:   {np.std(fitnesses):,.2f}")
            
            # Seleção
            survivors, (best_ind, best_fit) = genetic_algorithm.select_survivors(population, fitnesses)
            worst_fit = max(fitnesses)
            
            print(f"\n[3/4] Seleção:")
            print(f"  Melhor desta geração: {best_fit:,.2f}")
            print(f"  Survivors selecionados: {len(survivors)} (top 50%)")
            
            # Mostrar parâmetros do melhor da geração
            print(f"\n  Parâmetros do melhor desta geração:")
            for k, v in best_ind.items():
                if isinstance(v, float):
                    print(f"    {k:20s}: {v:.4f}")
                else:
                    print(f"    {k:20s}: {v}")
            
            # Atualizar melhor global
            improvement = False
            if best_fit < best_global_fitness:
                improvement = True
                previous_best = best_global_fitness
                best_global_fitness = best_fit
                best_global = best_ind.copy()
                improvement_amount = previous_best - best_global_fitness
                generations_without_improvement = 0
                print(f"\n  [NOVO MELHOR GLOBAL] {best_global_fitness:,.2f}")
                print(f"  Melhoria: {improvement_amount:,.2f} (de {previous_best:,.2f} para {best_global_fitness:,.2f})")
            else:
                generations_without_improvement += 1
                print(f"\n  Melhor global: {best_global_fitness:,.2f} (sem melhoria há {generations_without_improvement} gerações)")
            
            # Salvar histórico da geração
            gen_info = {
                'generation': gen + 1,
                'best_fitness': best_fit,
                'worst_fitness': worst_fit,
                'mean_fitness': float(np.mean(fitnesses)),
                'std_fitness': float(np.std(fitnesses)),
                'best_global_fitness': best_global_fitness,
                'improvement': improvement,
                'best_params': best_ind.copy()
            }
            generation_history.append(gen_info)
            
            # Salvar imagens do melhor indivíduo desta geração
            if save_generation_images and gen_images_dir:
                gen_dir = os.path.join(gen_images_dir, f"generation_{gen+1:02d}")
                os.makedirs(gen_dir, exist_ok=True)
                print(f"\n  Salvando imagens do melhor indivíduo em: {gen_dir}")
                results.save_individual_results(best_ind, images, names, gen_dir, 
                                               generation=gen+1, fitness=best_fit)
                print(f"  Imagens salvas para comparação")
            
            # Criar nova população
            print(f"\n[4/4] Reprodução e Nova Geração:")
            
            # CORREÇÃO CRÍTICA: Detectar dominância de indivíduos repetidos
            # Se muitos indivíduos têm o mesmo fitness, há dominância
            fitness_counts = {}
            for fit in fitnesses:
                fitness_counts[fit] = fitness_counts.get(fit, 0) + 1
            
            # Encontrar fitness mais comum e quantas vezes aparece
            most_common_fitness = max(fitness_counts.items(), key=lambda x: x[1])
            dominant_fitness, dominant_count = most_common_fitness
            
            # Calcular percentual de dominância
            dominance_ratio = dominant_count / len(fitnesses)
            
            # Se > 50% da população tem o mesmo fitness, há dominância crítica
            has_dominance = dominance_ratio > 0.5
            has_critical_dominance = dominance_ratio > 0.7  # > 70% = dominância crítica
            
            if has_critical_dominance:
                print(f"  [⚠️ DOMINÂNCIA CRÍTICA DETECTADA] Fitness {dominant_fitness:,.2f} aparece em {dominant_count}/{len(fitnesses)} indivíduos ({dominance_ratio*100:.1f}%)")
                print(f"  [AÇÃO] Removendo cópias e forçando mutação agressiva")
            elif has_dominance:
                print(f"  [⚠️ DOMINÂNCIA DETECTADA] Fitness {dominant_fitness:,.2f} aparece em {dominant_count}/{len(fitnesses)} indivíduos ({dominance_ratio*100:.1f}%)")
            
            # Criar nova população começando vazia
            new_population = []
            
            # CORREÇÃO: Calcular survivor_fitnesses ANTES de usar
            # Reprodução com seleção por torneio
            paired = list(zip(population, fitnesses))
            paired.sort(key=lambda x: x[1])
            survivor_fitnesses = [f for p, f in paired[:len(paired) // 2]]
            
            # Elitismo ADAPTATIVO: reduzir durante dominância
            elitism_count = config.ELITISM
            if has_critical_dominance:
                # Em dominância crítica, manter apenas 1 ou nenhum
                elitism_count = 0
                print(f"  Elitismo: 0 (desabilitado devido à dominância crítica)")
            elif has_dominance:
                # Em dominância moderada, reduzir para 1
                elitism_count = 1
                print(f"  Elitismo: {elitism_count} (reduzido devido à dominância)")
            else:
                print(f"  Elitismo: {elitism_count} melhores preservados")
            
            # Preservar elite (se não há dominância crítica)
            if not has_critical_dominance and len(survivors) > 0:
                # Selecionar indivíduos únicos do elite (evitar cópias)
                unique_survivors = []
                seen_fitness = set()
                for ind, fit in zip(survivors, survivor_fitnesses[:len(survivors)]):
                    # Arredondar fitness para detectar "iguais" (diferença < 0.01)
                    fit_rounded = round(fit, 2)
                    if fit_rounded not in seen_fitness or len(unique_survivors) < elitism_count:
                        unique_survivors.append((ind.copy(), fit))
                        seen_fitness.add(fit_rounded)
                        if len(unique_survivors) >= elitism_count:
                            break
                
                for ind, _ in unique_survivors[:elitism_count]:
                    new_population.append(ind)
            
            # Adicionar survivors únicos (evitar adicionar indivíduos repetidos)
            unique_survivors_list = []
            seen_fitness_survivors = set()
            for ind, fit in zip(survivors, survivor_fitnesses[:len(survivors)]):
                fit_rounded = round(fit, 2)
                if fit_rounded not in seen_fitness_survivors:
                    unique_survivors_list.append((ind.copy(), fit))
                    seen_fitness_survivors.add(fit_rounded)
            
            # CORREÇÃO CRÍTICA: Matar indivíduos que estão há mais de 5 gerações
            # Rastrear idade dos indivíduos e remover os velhos
            killed_count = 0
            aged_individuals = []  # Indivíduos que serão mortos
            
            # Atualizar idade dos survivors e identificar os que devem morrer
            for ind, fit in zip(survivors, survivor_fitnesses[:len(survivors)]):
                ind_hash = get_individual_hash(ind)
                
                # Se já está no tracker, incrementar idade
                if ind_hash in individual_age_tracker:
                    individual_age_tracker[ind_hash] += 1
                else:
                    # Novo indivíduo, começar em 1
                    individual_age_tracker[ind_hash] = 1
                
                # Se idade >= 5, marcar para morte
                if individual_age_tracker[ind_hash] >= MAX_INDIVIDUAL_AGE:
                    aged_individuals.append((ind, fit, ind_hash))
                    killed_count += 1
            
            if killed_count > 0:
                print(f"  [MORTE POR IDADE] {killed_count} indivíduo(s) morto(s) (idade >= {MAX_INDIVIDUAL_AGE} gerações)")
                # Remover indivíduos velhos da lista de survivors únicos
                aged_hashes = {h for _, _, h in aged_individuals}
                unique_survivors_list = [(ind, fit) for ind, fit in unique_survivors_list 
                                        if get_individual_hash(ind) not in aged_hashes]
                # Remover do tracker também
                for h in aged_hashes:
                    del individual_age_tracker[h]
            
            # Se há dominância crítica, NÃO adicionar survivors repetidos
            if not has_critical_dominance:
                # Adicionar survivors únicos (mas já temos elite, então limitar)
                remaining_slots = max(0, len(survivors) - elitism_count)
                for ind, _ in unique_survivors_list[elitism_count:elitism_count + remaining_slots]:
                    new_population.append(ind)
            
            # CORREÇÃO: Substituir indivíduos mortos por novos mutados agressivamente
            if killed_count > 0:
                for old_ind, old_fit, old_hash in aged_individuals:
                    # Criar novo indivíduo mutado agressivamente do velho
                    # Usar nível de mutação muito alto para escapar do mínimo local
                    new_ind = old_ind.copy()
                    # Aplicar mutação ultra-agressiva (nível 10+)
                    new_ind = genetic_algorithm.mutate(new_ind, config.PARAM_RANGES, 
                                                       stagnation_level=max(10, generations_without_improvement + 5))
                    genetic_algorithm.normalize_weights(new_ind)
                    genetic_algorithm.fix_size_constraints(new_ind)
                    
                    # Adicionar à nova população
                    new_population.append(new_ind)
                    print(f"    → Substituído por mutação agressiva (nível {max(10, generations_without_improvement + 5)})")
            
            new_children = 0
            diversity_reinjected = 0
            
            # Reinjeção de diversidade: após estagnação, substituir alguns indivíduos
            # CORREÇÃO: Mais agressivo - substituir mais indivíduos e com mais frequência
            if generations_without_improvement >= config.DIVERSITY_STAGNATION_THRESHOLD:
                # Substituir até 40% da população (ao invés de 20%) durante estagnação
                num_to_replace = max(2, min(config.POP_SIZE - len(new_population), 
                                            int(config.POP_SIZE * 0.4)))
                print(f"  [REINTRODUZINDO DIVERSIDADE] {num_to_replace} indivíduos aleatórios (estagnação há {generations_without_improvement} gerações)")
                for _ in range(num_to_replace):
                    new_ind = genetic_algorithm.create_random_individual(config.PARAM_RANGES)
                    genetic_algorithm.normalize_weights(new_ind)
                    genetic_algorithm.fix_size_constraints(new_ind)
                    new_population.append(new_ind)
                    diversity_reinjected += 1
                # NÃO resetar generations_without_improvement para 0 - manter pressão de diversidade
                # Apenas resetar se houver melhoria real
            
            # CORREÇÃO ADICIONAL: Se estagnação muito longa (> 8 gerações), resetar população parcialmente
            if generations_without_improvement > 8:
                num_to_replace = max(5, int(config.POP_SIZE * 0.5))
                print(f"  [RESET PARCIAL DA POPULAÇÃO] {num_to_replace} indivíduos aleatórios (estagnação crítica: {generations_without_improvement} gerações)")
                # Remover piores indivíduos e substituir por aleatórios
                if len(new_population) > num_to_replace:
                    # Manter apenas elite + alguns melhores
                    new_population = new_population[:len(new_population) - num_to_replace]
                    for _ in range(num_to_replace):
                        new_ind = genetic_algorithm.create_random_individual(config.PARAM_RANGES)
                        genetic_algorithm.normalize_weights(new_ind)
                        genetic_algorithm.fix_size_constraints(new_ind)
                        new_population.append(new_ind)
                        diversity_reinjected += 1
            
            while len(new_population) < config.POP_SIZE:
                # CORREÇÃO: Durante dominância, aumentar muito a reinjeção de diversidade
                diversity_rate = config.DIVERSITY_REINJECTION_RATE
                if has_critical_dominance:
                    diversity_rate = 0.80  # 80% de chance durante dominância crítica
                elif has_dominance:
                    diversity_rate = 0.60  # 60% de chance durante dominância moderada
                
                # Reinjeção ocasional de diversidade (taxa adaptativa)
                if random.random() < diversity_rate:
                    new_ind = genetic_algorithm.create_random_individual(config.PARAM_RANGES)
                    genetic_algorithm.normalize_weights(new_ind)
                    genetic_algorithm.fix_size_constraints(new_ind)
                    new_population.append(new_ind)
                    diversity_reinjected += 1
                    new_children += 1
                    continue
                
                # Seleção por torneio
                if len(survivors) > 1:
                    parent_a = genetic_algorithm.tournament_selection(survivors, survivor_fitnesses, tournament_size=3)
                    parent_b = genetic_algorithm.tournament_selection(survivors, survivor_fitnesses, tournament_size=3)
                else:
                    parent_a = survivors[0]
                    parent_b = survivors[0]
                
                # Crossover BLX-alpha
                child = genetic_algorithm.crossover(parent_a, parent_b, config.PARAM_RANGES, alpha=0.5)
                
                # Mutação adaptativa baseada em estagnação E dominância
                # Se há dominância, aumentar muito o nível de estagnação artificial
                mutation_stagnation_level = generations_without_improvement
                if has_critical_dominance:
                    # Dominância crítica = nível de mutação muito alto
                    mutation_stagnation_level = max(10, generations_without_improvement + 7)
                elif has_dominance:
                    # Dominância moderada = nível de mutação alto
                    mutation_stagnation_level = max(5, generations_without_improvement + 3)
                
                child = genetic_algorithm.mutate(child, config.PARAM_RANGES, 
                                                  stagnation_level=mutation_stagnation_level)
                
                # Normalizar
                genetic_algorithm.normalize_weights(child)
                genetic_algorithm.fix_size_constraints(child)
                
                # Verificar se o novo filho já existe (não adicionar repetidos)
                child_hash = get_individual_hash(child)
                if child_hash not in individual_age_tracker:
                    new_population.append(child)
                    # Registrar novo indivíduo com idade 0 (será incrementada na próxima geração)
                    individual_age_tracker[child_hash] = 0
                    new_children += 1
                else:
                    # Indivíduo já existe, criar outro mutado
                    child = genetic_algorithm.mutate(child, config.PARAM_RANGES,
                                                     stagnation_level=generations_without_improvement + 2)
                    genetic_algorithm.normalize_weights(child)
                    genetic_algorithm.fix_size_constraints(child)
                    child_hash = get_individual_hash(child)
                    new_population.append(child)
                    if child_hash not in individual_age_tracker:
                        individual_age_tracker[child_hash] = 0
                    new_children += 1
            
            # Atualizar idade dos indivíduos que passaram para a próxima geração
            # Incrementar idade de todos os indivíduos na nova população
            for ind in new_population:
                ind_hash = get_individual_hash(ind)
                if ind_hash in individual_age_tracker:
                    # Só incrementar se não foi resetado (novos indivíduos começam em 0)
                    pass  # Já incrementado durante processamento dos survivors
            
            if diversity_reinjected > 0:
                print(f"  Diversidade reintroduzida: {diversity_reinjected} indivíduos aleatórios")
            
            if killed_count > 0:
                print(f"  Indivíduos mortos e substituídos: {killed_count}")
            
            print(f"  Novos filhos criados: {new_children}")
            print(f"  Total população: {len(new_population)}")
            
            population = new_population
            print()
        
        # Resultado final
        print("=" * 80)
        print("RESULTADO FINAL")
        print("=" * 80)
        print(f"\nMelhor fitness encontrado: {best_global_fitness:,.2f}")
        print(f"\nMelhores parâmetros encontrados:")
        for k, v in best_global.items():
            if isinstance(v, float):
                print(f"  {k:20s}: {v:.4f}")
            else:
                print(f"  {k:20s}: {v}")
        
        print(f"\n\nEvolução do melhor fitness por geração:")
        for gen_info in generation_history:
            marker = "[MELHORIA]" if gen_info['improvement'] else "          "
            print(f"  Geração {gen_info['generation']:2d}: {gen_info['best_global_fitness']:,.2f} {marker}")
        
        print("\n" + "=" * 80)
        print("FIM DA EVOLUÇÃO")
        print("=" * 80)
        
    finally:
        # Restaurar stdout
        if log_file:
            sys.stdout = original_stdout
            logger.close()
    
    return best_global, best_global_fitness, generation_history

