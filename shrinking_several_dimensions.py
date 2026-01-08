import numpy as np
import pywt
import matplotlib.pyplot as plt

np.random.seed(42)

def generate_correlated_random_walks(n_samples=1024, n_assets=3, rho=0.8):
    """
    Gera séries temporais multivariadas simulando ativos financeiros correlacionados.
    Modelo: y_t = f_t + epsilon_t
    """
    # Estrutura de covariância do ruído latente
    cov_matrix = np.full((n_assets, n_assets), rho)
    np.fill_diagonal(cov_matrix, 1.0)
    
    # Decomposição de Cholesky para induzir correlação
    L = np.linalg.cholesky(cov_matrix)
    
    # Geração do ruído branco multivariado
    white_noise = np.random.normal(0, 1, (n_assets, n_samples))
    correlated_innovations = np.dot(L, white_noise).T
    
    # Sinal Verdadeiro: Random Walk + Tendência Determinística Senoidal
    t = np.linspace(0, 1, n_samples)
    trend = 2 * np.sin(2 * np.pi * t)
    
    true_signal = np.cumsum(correlated_innovations, axis=0) * 0.1 + trend[:, None]
    
    # Adição de ruído de medição (alvo da filtragem)
    measurement_noise = np.random.normal(0, 0.5, true_signal.shape)
    observed_signal = true_signal + measurement_noise
    
    return t, true_signal, observed_signal

def estimate_sigma_mad(coeffs):
    """
    Estima o desvio padrão do ruído usando a MAD (Median Absolute Deviation)
    dos coeficientes do nível mais fino de decomposição.
    """
    # O nível mais fino é o último elemento da lista de coeficientes (cD1)
    # Concatenamos os detalhes de todos os ativos para uma estimativa global robusta
    finest_details = np.concatenate([c[-1] for c in coeffs])
    mad = np.median(np.abs(finest_details - np.median(finest_details)))
    return mad / 0.6745

def multivariate_wavelet_smoothing(data, wavelet='db4', level=4):
    """
    Aplica Multivariate Wavelet Shrinkage usando limiarização vetorial.
    """
    n_samples, n_assets = data.shape
    
    # Decomposição Wavelet 
    coeffs = [pywt.wavedec(data[:, i], wavelet, level=level) for i in range(n_assets)]
    
    # Estimação de sigma (nível mais fino)
    sigma_hat = estimate_sigma_mad(coeffs)
    
    # Limiar Universal Multivariado 
    threshold = sigma_hat * np.sqrt(3 * np.log(n_samples))
    
    # Processamento dos coeficientes de detalhe
    coeffs_thresh = [list(c) for c in coeffs]
    
    for j in range(1, len(coeffs[0])):
        # Extração vetorial dos coeficientes no nível j
        details = np.array([coeffs[i][j] for i in range(n_assets)]).T
        
        # Norma Euclidiana para cada instante k
        norms = np.linalg.norm(details, axis=1)
        
        # Soft Thresholding Vetorial
        with np.errstate(divide='ignore', invalid='ignore'):
            factors = np.maximum(0, 1 - threshold / norms)
        
        factors[norms == 0] = 0 # Correção para vetores nulos
        
        # Aplicação do fator de contração
        details_thresh = details * factors[:, None]
        
        # Atualização da estrutura de coeficientes
        for i in range(n_assets):
            coeffs_thresh[i][j] = details_thresh[:, i]
            
    # Reconstrução do sinal
    smoothed_data = np.zeros_like(data)
    for i in range(n_assets):
        smoothed_data[:, i] = pywt.waverec(coeffs_thresh[i], wavelet)[:n_samples]
        
    return smoothed_data

if __name__ == "__main__":
    # Geração de dados sintéticos
    t, true_signal, noisy_signal = generate_correlated_random_walks()
    
    # Filtragem
    smoothed_signal = multivariate_wavelet_smoothing(noisy_signal)
    
    # Visualização
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    assets_labels = ['Ativo A (Alta Cap)', 'Ativo B (Mid Cap)', 'Ativo C (Small Cap)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i in range(3):
        ax = axes[i]
        ax.plot(t, noisy_signal[:, i], color='silver', alpha=0.6, label='Observado (Ruidoso)', lw=1)
        ax.plot(t, true_signal[:, i], color='black', linestyle='--', alpha=0.7, label='Latente (Verdadeiro)', lw=1.5)
        ax.plot(t, smoothed_signal[:, i], color=colors[i], label='Wavelet Shrinkage (Estimado)', lw=2)
        
        ax.set_ylabel(assets_labels[i])
        ax.legend(loc='upper left', frameon=True, fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if i == 0:
            ax.set_title('Filtragem Multivariada via Wavelets (Soft Thresholding Vetorial)', fontsize=13, pad=15)

    plt.xlabel('Tempo')
    plt.tight_layout()
    
    plt.show() 
    print("Processo concluído.")
