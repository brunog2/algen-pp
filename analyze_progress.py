"""
Script para analisar o progresso do Algen-PP enquanto está rodando ou após conclusão.
"""
import pickle
import os

OUTPUT_DIR = "./outputs"

# Tentar carregar histórico
history_file = os.path.join(OUTPUT_DIR, "algen_pp_history.pkl")
best_file = os.path.join(OUTPUT_DIR, "best_individual.pkl")

print("=== Análise do Progresso do Algen-PP ===\n")

# Verificar melhor indivíduo atual
if os.path.exists(best_file):
    with open(best_file, "rb") as f:
        best = pickle.load(f)
    print("✓ Melhor indivíduo encontrado (salvo temporariamente)")
    print(f"  Parâmetros: {best}\n")
else:
    print("✗ Nenhum melhor indivíduo salvo ainda\n")

# Verificar histórico completo
if os.path.exists(history_file):
    with open(history_file, "rb") as f:
        data = pickle.load(f)
    
    best_global = data.get('best')
    best_fit = data.get('fit')
    history = data.get('history', [])
    
    print(f"✓ Histórico completo encontrado ({len(history)} gerações)\n")
    print(f"Melhor fitness global: {best_fit:.2f}\n")
    
    if len(history) > 0:
        print("Evolução do melhor fitness por geração:")
        print("Geração | Melhor Fitness | Melhorou?")
        print("-" * 50)
        prev_fit = float('inf')
        improvements = 0
        for gen, fit, _ in history:
            improved = "✓ SIM" if fit < prev_fit else "✗ NÃO"
            if fit < prev_fit:
                improvements += 1
            print(f"   {gen+1:2d}   | {fit:14.2f} | {improved}")
            prev_fit = min(prev_fit, fit)
        
        print(f"\nTotal de melhorias: {improvements}/{len(history)} gerações")
        
        # Análise de estagnação
        if len(history) >= 5:
            last_5 = [h[1] for h in history[-5:]]
            if len(set(last_5)) == 1:
                print("\n⚠ AVISO: Estagnação detectada (últimas 5 gerações com mesmo fitness)")
            else:
                print(f"\n✓ Sem estagnação detectada (variação: {max(last_5) - min(last_5):.2f})")
else:
    print("✗ Histórico completo ainda não disponível (GA ainda rodando ou não finalizado)")

print("\n" + "=" * 50)

