"""
Algen-PP: Algoritmo Genético para Segmentação de Imagens
Implementação melhorada com diferenciações do algoritmo original.

Baseado em Daguano (2020) - "Algoritmo Genético para Segmentação de Imagens 
utilizando Tamanho e Forma dos Objetos"

MELHORIAS IMPLEMENTADAS:
1. Watershed híbrido: combina marcadores baseados em distance transform e intensidade local
2. Fitness combinada: Almod (85%) + métrica de qualidade de forma (15%)
3. Seleção por torneio: além da seleção por ranking, usa torneios para diversidade
4. Crossover BLX-alpha: crossover mais sofisticado que explora melhor o espaço de busca
5. Refinamento adaptativo: itera sobre segmentação para melhorar resultados
"""

import os
import json
from datetime import datetime

import config
import image_utils
import ga_runner
import results


def main():
    """Função principal."""
    print("=" * 60)
    print("Algen-PP: Algoritmo Genético para Segmentação")
    print("Implementação melhorada com logs detalhados")
    print("=" * 60)
    
    # Criar diretório de logs
    log_dir = os.path.join(config.OUTPUT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"algen_evolution_{timestamp}.log")
    
    print(f"\nArquivo de log: {log_file}")
    print("Todos os prints serão salvos neste arquivo.\n")
    
    # Carregar imagens
    print(f"Carregando imagens de: {config.IMAGES_DIR}")
    images, names = image_utils.load_images_from_folder(config.IMAGES_DIR, ext="tif")
    
    if len(images) == 0:
        print("ERRO: Nenhuma imagem encontrada!")
        exit(1)
    
    print(f"[OK] {len(images)} imagens carregadas")
    if len(images) > 0:
        print(f"  Dimensões: {images[0].shape}")
        print(f"  Tipo: {images[0].dtype}")
    
    # Executar algoritmo genético com logging e salvamento de imagens por geração
    best_params, best_fitness, generation_history = ga_runner.run_genetic_algorithm(
        images, names, 
        log_file=log_file, 
        save_generation_images=True,
        timestamp=timestamp
    )
    
    # Salvar histórico resumido em arquivo JSON
    history_file = os.path.join(log_dir, f"algen_history_{timestamp}.json")
    with open(history_file, 'w') as f:
        # Converter numpy types para tipos Python nativos
        history_serializable = []
        for gen_info in generation_history:
            gen_dict = {
                'generation': gen_info['generation'],
                'best_fitness': float(gen_info['best_fitness']),
                'worst_fitness': float(gen_info['worst_fitness']),
                'mean_fitness': float(gen_info['mean_fitness']),
                'std_fitness': float(gen_info['std_fitness']),
                'best_global_fitness': float(gen_info['best_global_fitness']),
                'improvement': gen_info['improvement'],
                'best_params': {k: (float(v) if isinstance(v, (float, int)) else v) 
                               for k, v in gen_info['best_params'].items()}
            }
            history_serializable.append(gen_dict)
        json.dump(history_serializable, f, indent=2)
    
    print(f"\n[OK] Histórico salvo em: {history_file}")
    
    # Aplicar melhor indivíduo e salvar resultados
    output_dir = os.path.join(config.OUTPUT_DIR, "algen_basic_results", "final")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Aplicando melhor indivíduo e salvando resultados ===")
    print(f"Diretório de saída: {output_dir}")
    
    results.save_individual_results(best_params, images, names, output_dir)
    
    print(f"\n[OK] Resultados salvos em: {output_dir}")
    print(f"  - Segmentações binárias: *_segmented.png")
    print(f"  - Comparações com contornos: *_comparison.png")
    print(f"  - Lado a lado: *_side_by_side.png")
    
    # Informar sobre imagens das gerações
    gen_images_dir = os.path.join(config.OUTPUT_DIR, "generation_results", timestamp)
    if os.path.exists(gen_images_dir):
        print(f"\nImagens das gerações salvas em: {gen_images_dir}")
        print(f"  Cada geração tem sua própria pasta: generation_01/, generation_02/, etc.")
        print(f"  Compare as imagens de cada geração para ver a evolução!")
    
    print("\n" + "=" * 60)
    print("Processo concluído!")
    print(f"Log completo salvo em: {log_file}")
    print(f"Histórico JSON salvo em: {history_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
