"""
Monitor em tempo real do progresso do GA.
Atualiza a cada 2 segundos mostrando o melhor indivíduo atual.
"""
import pickle
import os
import time
import sys

OUTPUT_DIR = "./outputs"
best_file = os.path.join(OUTPUT_DIR, "best_individual.pkl")

print("Monitorando progresso do Algen-PP... (Ctrl+C para sair)\n")

last_fit = None
gen_count = 0

try:
    while True:
        if os.path.exists(best_file):
            with open(best_file, "rb") as f:
                best = pickle.load(f)
            
            # Tentar estimar geração atual (verificar arquivo de histórico)
            history_file = os.path.join(OUTPUT_DIR, "algen_pp_history.pkl")
            if os.path.exists(history_file):
                with open(history_file, "rb") as f:
                    data = pickle.load(f)
                    history = data.get('history', [])
                    gen_count = len(history)
                    best_fit = data.get('fit', 'N/A')
            else:
                best_fit = 'Avaliando...'
            
            # Mostrar apenas se mudou
            if best_fit != last_fit:
                print(f"\n[{time.strftime('%H:%M:%S')}] Geração {gen_count}")
                print(f"  Melhor fitness: {best_fit}")
                if isinstance(best_fit, float):
                    print(f"  Parâmetros principais:")
                    print(f"    - gaussian_sigma: {best.get('gaussian_sigma', 'N/A'):.3f}")
                    print(f"    - size_min/max: {best.get('size_min', 'N/A')} / {best.get('size_max', 'N/A')}")
                    print(f"    - closing_kernel: {best.get('closing_kernel', 'N/A')}")
                    print(f"    - merge_threshold: {best.get('merge_threshold', 'N/A'):.3f}")
                last_fit = best_fit
        else:
            print(".", end="", flush=True)
        
        time.sleep(2)
        
except KeyboardInterrupt:
    print("\n\nMonitoramento interrompido.")
    sys.exit(0)

