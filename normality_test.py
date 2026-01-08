import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.spatial.distance import mahalanobis

np.random.seed(42)

def gerar_dados_correlacionados(n=200, p=3, rho=0.8):
    """
    Gera vetores normais multivariados usando decomposição de Cholesky
    para induzir correlação, conforme descrito na Metodologia.
    """
    # 1. Matriz de Covariância Teórica (Simulando mercado correlacionado)
    cov_matrix = np.full((p, p), rho)
    np.fill_diagonal(cov_matrix, 1.0)
    
    # 2. Decomposição de Cholesky
    L = np.linalg.cholesky(cov_matrix)
    
    # 3. Gerar ruído branco não correlacionado
    uncorrelated_data = np.random.normal(0, 1, size=(n, p))
    
    # 4. Induzir correlação: X = Z * L.T
    correlated_data = uncorrelated_data @ L.T
    return correlated_data

def plot_qq_univariado(data):
    """Gera a Figura 1 do Relatório: Gráficos Q-Q marginais"""
    n, p = data.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(p):
        stats.probplot(data[:, i], dist="norm", plot=axes[i])
        axes[i].set_title(f'Q-Q Plot: Ativo {i+1}')
        axes[i].set_xlabel('Quantis Teóricos')
        axes[i].set_ylabel('Quantis da Amostra')
        axes[i].grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    plt.show() 

def plot_chisquare(data):
    """Gera a Figura 2 do Relatório: Gráfico de Qui-Quadrado (Distância de Mahalanobis)"""
    n, p = data.shape
    
    # 1. Calcular vetor de médias e matriz de covariância amostral (S)
    mean_vec = np.mean(data, axis=0)
    cov_mat = np.cov(data, rowvar=False)
    inv_cov_mat = np.linalg.inv(cov_mat)
    
    # 2. Calcular distâncias de Mahalanobis ao quadrado (d^2)
    d2_values = []
    for i in range(n):
        d = mahalanobis(data[i], mean_vec, inv_cov_mat)
        d2_values.append(d**2)
    d2_values = np.sort(d2_values)
    
    # 3. Calcular quantis teóricos da Qui-Quadrado com p graus de liberdade
    quantiles = np.arange(1, n + 1) / (n + 0.5) # Ajuste de continuidade usual
    chi2_theor = stats.chi2.ppf(quantiles, df=p)
    
    # 4. Plotar
    plt.figure(figsize=(7, 6))
    plt.scatter(chi2_theor, d2_values, color='blue', alpha=0.6, label='Observações')
    
    # Linha de referência (y=x)
    max_val = max(d2_values.max(), chi2_theor.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Referência Teórica (Normal)')
    
    plt.title(f'Chi-Square Plot (p={p})')
    plt.xlabel(f'Quantis Teóricos $\chi^2_{{{p}}}$')
    plt.ylabel('Distâncias de Mahalanobis Quadradas ($d^2$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    print("--- Iniciando Experimento Capítulo 4 ---")
    dados = gerar_dados_correlacionados(n=200, p=3, rho=0.8)
    plot_qq_univariado(dados)
    plot_chisquare(dados)
    print("Concluído.")
