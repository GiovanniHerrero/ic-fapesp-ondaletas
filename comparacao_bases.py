import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import beta, chi2
from sklearn.decomposition import PCA
import pywt

# ==============================================================================
# 1. PARÂMETROS GERAIS E SINAL VERDADEIRO
# ==============================================================================
n = 1024
p = 2048
sigma = 1.0
norm_rho = 10.0
mode = 'periodization'        
w = 0.995                     
seed_fixa = 10013  

# Geração do Sinal "Três Picos" 
t = np.linspace(0, 1, p)
beta1 = beta.pdf(t, 1500, 3000)
beta2 = beta.pdf(t, 1200, 900)
beta3 = beta.pdf(t, 600, 160)
f_raw = 0.7 * beta1 + 0.5 * beta2 + 0.5 * beta3
rho_true = f_raw / np.linalg.norm(f_raw) * norm_rho

# ==============================================================================
# 2. FUNÇÃO AUXILIAR: MÉTRICA R (OVERLAP)
# ==============================================================================
def calc_R(u, v):
    """
    Calcula o Overlap R(u, v) = |<u, v>| / (||u|| * ||v||).
    Eq. 4 de Johnstone & Lu (2009).
    Mede o cosseno do ângulo. 1.0 = Perfeito.
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    # Valor absoluto 
    return np.abs(np.dot(u, v)) / (norm_u * norm_v)

# ==============================================================================
# 3. ALGORITMO ASPCA 
# ==============================================================================
def processar_aspca(seed, wavelet_name, level=3, multiplicador_delta=1.0):
    np.random.seed(seed)
    
    # A. Geração dos dados 
    v_vec = np.random.normal(0, 1, size=(n, 1))
    Z = np.random.normal(0, 1, size=(n, p))
    X = v_vec @ rho_true.reshape(1, -1) + sigma * Z
    
    # B. Transformada Wavelet
    coeffs_first = pywt.wavedec(X[0], wavelet_name, level=level, mode=mode)
    slices = pywt.coeffs_to_array(coeffs_first)[1]
    p_coef = sum(c.size for c in coeffs_first)
    
    X_wavelet = np.zeros((n, p_coef))
    for i in range(n):
        c = pywt.wavedec(X[i], wavelet_name, level=level, mode=mode)
        arr, _ = pywt.coeffs_to_array(c)
        X_wavelet[i] = arr

    # C. Seleção de variáveis (Screening)
    variances = np.var(X_wavelet, axis=0, ddof=1)
    sigma_hat_sq = np.median(variances)
    sigma_hat = np.sqrt(sigma_hat_sq)
    
    sorted_indices = np.argsort(variances)[::-1]
    sorted_vars = variances[sorted_indices]
    
    # Baseline via PPF 
    ranks = np.arange(1, p_coef + 1)
    q_vals = (p_coef - ranks + 1) / (p_coef + 1)
    baseline = (sigma_hat_sq / (n - 1)) * chi2.ppf(q_vals, df=n - 1)
    
    eta_sq = np.maximum(sorted_vars - baseline, 0)
    total_excess = np.sum(eta_sq)
    cumulative_excess = np.cumsum(eta_sq)
    
    # Define k
    if total_excess == 0:
        k = 1
    else:
        k = np.searchsorted(cumulative_excess, w * total_excess) + 1
    
    selected_indices = sorted_indices[:k]

    # D. PCA Reduzido
    pca = PCA(n_components=1)
    pca.fit(X_wavelet[:, selected_indices])
    rho_reduced = pca.components_[0]
    
    rho_hat_w = np.zeros(p_coef)
    rho_hat_w[selected_indices] = rho_reduced
    
    # Ajuste de sinal (Sign Flipping)
    coeffs_rho_lvl = pywt.wavedec(rho_true, wavelet_name, level=level, mode=mode)
    rho_true_w_lvl, _ = pywt.coeffs_to_array(coeffs_rho_lvl)
    if np.dot(rho_hat_w, rho_true_w_lvl) < 0:
        rho_hat_w = -rho_hat_w

    # E. Thresholding (Eq. 17 de Johnstone & Lu)
    sum_diff = np.sum(np.maximum(variances - sigma_hat_sq, 0))
    lower_bound_norm = sigma_hat_sq * np.sqrt(p_coef / n)
    norm_rho_sq_est = max(sum_diff, lower_bound_norm)
    
    numerador = sigma_hat * np.sqrt(norm_rho_sq_est + sigma_hat_sq)
    denominador = np.sqrt(n) * norm_rho_sq_est
    tau_hat = numerador / denominador
    
    delta = multiplicador_delta * tau_hat * np.sqrt(2 * np.log(k))
    
    rho_hat_w_th = rho_hat_w.copy()
    rho_hat_w_th[np.abs(rho_hat_w_th) < delta] = 0
    k_final = np.count_nonzero(rho_hat_w_th)
    
    # F. Reconstrução
    def reconstruct(vec_w):
        c = pywt.array_to_coeffs(vec_w, slices, output_format='wavedec')
        sig = pywt.waverec(c, wavelet_name, mode=mode)
        return sig[:p]
    
    rho_rec_th = reconstruct(rho_hat_w_th)
    
    # Normalização final 
    if np.linalg.norm(rho_rec_th) > 1e-6:
        rho_rec_th = rho_rec_th * (norm_rho / np.linalg.norm(rho_rec_th))
    
    return rho_rec_th, k_final, delta

# ==============================================================================
# 4. LOOP DE COMPARAÇÃO E PLOTAGEM
# ==============================================================================
wavelets_to_test = ['sym8', 'db4', 'coif3', 'haar']
results_data = []

plt.figure(figsize=(14, 10))
plt.suptitle(f"Comparação de Bases Ondaleta (Seed {seed_fixa}, Nível 3)\nMétrica de Consistência R (Overlap)", fontsize=16)

for i, w_name in enumerate(wavelets_to_test):
    print(f"Processando {w_name}...")
    
    # Executa o algoritmo
    rho_final, k_fin, delta = processar_aspca(seed=seed_fixa, wavelet_name=w_name, level=3)
    
    # Calcula R (Overlap)
    r_score = calc_R(rho_final, rho_true)
    
    # Armazena resultados
    results_data.append({
        'Base': w_name,
        'Overlap (R)': r_score,
        'Esparsidade (k_final)': k_fin,
        'Delta': delta
    })
    
    # Plot
    plt.subplot(2, 2, i+1)
    plt.plot(rho_true, 'k--', alpha=0.3, label='Verdadeiro')
    plt.plot(rho_final, 'b', linewidth=1.2, label=f'Recuperado ({w_name})')
    
    # Título 
    plt.title(f"Base: {w_name}\nR = {r_score:.4f} | k_final = {k_fin}")
    plt.ylim(-0.5, 1.8)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()

# ==============================================================================
# 5. TABELA RESUMO
# ==============================================================================
df_results = pd.DataFrame(results_data)
df_results = df_results.sort_values(by='Overlap (R)', ascending=False)

print("\n" + "="*60)
print("TABELA COMPARATIVA DE DESEMPENHO (MÉTRICA R)")
print("="*60)
print(df_results.to_string(index=False, float_format="%.5f"))
print("="*60)
