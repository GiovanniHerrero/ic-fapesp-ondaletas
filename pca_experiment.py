import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuração para reprodutibilidade
np.random.seed(42)

def generate_market_data(n_samples=200, n_assets=5):
    """
    Gera dados de 5 ativos correlacionados (simulando um setor de mercado).
    """
    # Matriz de correlação teórica (todos se movem juntos ~0.7)
    mean = [0] * n_assets
    cov = np.full((n_assets, n_assets), 0.7)
    np.fill_diagonal(cov, 1.0)
    
    # Gerar dados
    returns = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Adicionar ruído para diferenciar os ativos
    noise = np.random.normal(0, 0.2, returns.shape)
    return returns + noise

def run_pca_experiment():
    # 1. Gerar Dados
    data = generate_market_data()
    feature_names = [f'Ativo {i+1}' for i in range(data.shape[1])]
    
    # 2. Padronização (Equivale a usar Matriz de Correlação R)
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    
    # 3. Aplicar PCA
    pca = PCA()
    components = pca.fit_transform(data_std)
    
    eigenvalues = pca.explained_variance_
    explained_ratio = pca.explained_variance_ratio_ * 100
    
    # --- Configuração da Visualização Unificada (1 linha, 2 colunas) ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Gráfico 1: Scree Plot (Lado Esquerdo) ---
    ax_scree = axes[0]
    x_ticks = np.arange(1, len(eigenvalues) + 1)
    
    ax_scree.plot(x_ticks, eigenvalues, 'o-', color='black', label='Autovalores')
    ax_scree.axhline(y=1, color='red', linestyle='--', label=r'Critério de Kaiser ($\lambda=1$)')
    
    ax_scree.set_title('Scree Plot')
    ax_scree.set_xlabel('Número do Componente')
    ax_scree.set_ylabel('Autovalor (Variância)')
    ax_scree.set_xticks(x_ticks)
    ax_scree.grid(True, alpha=0.3)
    ax_scree.legend()
    
    # --- Gráfico 2: PC1 vs PC2 Scatter Plot (Lado Direito) ---
    ax_scatter = axes[1]
    ax_scatter.scatter(components[:, 0], components[:, 1], alpha=0.6, color='#1f77b4')
    
    ax_scatter.set_title('Projeção nos 2 Primeiros Componentes Principais')
    ax_scatter.set_xlabel(f'PC1 ({explained_ratio[0]:.1f}% da Variância)')
    ax_scatter.set_ylabel(f'PC2 ({explained_ratio[1]:.1f}% da Variância)')
    ax_scatter.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax_scatter.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    ax_scatter.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir Cargas (Loadings) para interpretação no texto
    print("Autovetores (Cargas):")
    print(pd.DataFrame(pca.components_.T, index=feature_names, columns=[f'PC{i+1}' for i in range(5)]))
    print("\nVariância Explicada (%):")
    print(explained_ratio)

if __name__ == "__main__":
    run_pca_experiment()
