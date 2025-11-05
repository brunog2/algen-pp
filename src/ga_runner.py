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
            new_population = survivors.copy()
            
            # Elitismo
            for _ in range(config.ELITISM):
                if len(survivors) > 0:
                    new_population.append(survivors[0].copy())
            print(f"  Elitismo: {config.ELITISM} melhores preservados")
            
            # Reprodução com seleção por torneio
            paired = list(zip(population, fitnesses))
            paired.sort(key=lambda x: x[1])
            survivor_fitnesses = [f for p, f in paired[:len(paired) // 2]]
            
            new_children = 0
            diversity_reinjected = 0
            
            # Reinjeção de diversidade: após estagnação, substituir alguns indivíduos
            if generations_without_improvement >= config.DIVERSITY_STAGNATION_THRESHOLD:
                num_to_replace = max(1, min(config.POP_SIZE - len(new_population), config.POP_SIZE // 5))
                print(f"  [REINTRODUZINDO DIVERSIDADE] {num_to_replace} indivíduos aleatórios (estagnação há {generations_without_improvement} gerações)")
                for _ in range(num_to_replace):
                    new_ind = genetic_algorithm.create_random_individual(config.PARAM_RANGES)
                    genetic_algorithm.normalize_weights(new_ind)
                    genetic_algorithm.fix_size_constraints(new_ind)
                    new_population.append(new_ind)
                    diversity_reinjected += 1
                generations_without_improvement = 0  # Reset após reinjeção
            
            while len(new_population) < config.POP_SIZE:
                # Reinjeção ocasional de diversidade
                if random.random() < config.DIVERSITY_REINJECTION_RATE:
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
                
                # Mutação
                child = genetic_algorithm.mutate(child, config.PARAM_RANGES)
                
                # Normalizar
                genetic_algorithm.normalize_weights(child)
                genetic_algorithm.fix_size_constraints(child)
                
                new_population.append(child)
                new_children += 1
            
            if diversity_reinjected > 0:
                print(f"  Diversidade reintroduzida: {diversity_reinjected} indivíduos aleatórios")
            
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

