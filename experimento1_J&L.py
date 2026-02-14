import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta, chi2
from sklearn.decomposition import PCA
import pywt
import time

# --------------------------------------------------------------------
# 1. PARÂMETROS GERAIS 
# --------------------------------------------------------------------
n = 1024
p = 2048
sigma = 1.0
norm_rho = 10.0
wavelet_name = 'sym8'
mode = 'periodization'        
w = 0.995                     


# --------------------------------------------------------------------
# 2. PREPARAÇÃO DO SINAL VERDADEIRO (RHO)
# --------------------------------------------------------------------
t = np.linspace(0, 1, p)
beta1 = beta.pdf(t, 1500, 3000)
beta2 = beta.pdf(t, 1200, 900)
beta3 = beta.pdf(t, 600, 160)
f_raw = 0.7 * beta1 + 0.5 * beta2 + 0.5 * beta3
rho_true = f_raw / np.linalg.norm(f_raw) * norm_rho

# --------------------------------------------------------------------
# 3. FUNÇÃO PRINCIPAL DO ALGORITMO 
# --------------------------------------------------------------------
def processar_aspca(seed, level, multiplicador_delta):
    """
    Executa o pipeline completo do ASPCA:
    Wavelet -> Seleção -> PCA Reduzido -> Thresholding -> Reconstrução
    Retorna: k_adaptive, k_final, ASE (opcional)
    """
    np.random.seed(seed)
    
    # A. Geração dos dados
    v = np.random.normal(0, 1, size=(n, 1))
    Z = np.random.normal(0, 1, size=(n, p))
    X = v @ rho_true.reshape(1, -1) + sigma * Z
    
    # B. Transformada Wavelet
    coeffs_first = pywt.wavedec(X[0], wavelet_name, level=level, mode=mode)
    slices = pywt.coeffs_to_array(coeffs_first)[1]
    p_coef = sum(c.size for c in coeffs_first)
    
    X_wavelet = np.zeros((n, p_coef))
    for i in range(n):
        c = pywt.wavedec(X[i], wavelet_name, level=level, mode=mode)
        arr, _ = pywt.coeffs_to_array(c)
        X_wavelet[i] = arr

    # C. Seleção de variáveis
    variances = np.var(X_wavelet, axis=0, ddof=1)
    sigma_hat_sq = np.median(variances)
    sigma_hat = np.sqrt(sigma_hat_sq)
    
    sorted_indices = np.argsort(variances)[::-1]
    sorted_vars = variances[sorted_indices]
    
    # Baseline ordenado 
    ranks = np.arange(1, p_coef + 1)
    q_vals = (p_coef - ranks + 1) / (p_coef + 1)

    baseline = (sigma_hat_sq / (n - 1)) * chi2.ppf(q_vals, df=n - 1)

    eta_sq = np.maximum(sorted_vars - baseline, 0)
    total_excess = np.sum(eta_sq)
    cumulative_excess = np.cumsum(eta_sq)
    
    k = np.searchsorted(cumulative_excess, w * total_excess) + 1
    
    selected_indices = sorted_indices[:k]

    # D. PCA Reduzido
    pca = PCA(n_components=1)
    pca.fit(X_wavelet[:, selected_indices])
    rho_reduced = pca.components_[0]
    
    rho_hat_w = np.zeros(p_coef)
    rho_hat_w[selected_indices] = rho_reduced
    
    # Ajuste de sinal
    coeffs_rho_lvl = pywt.wavedec(rho_true, wavelet_name, level=level, mode=mode)
    rho_true_w_lvl, _ = pywt.coeffs_to_array(coeffs_rho_lvl)
    if np.dot(rho_hat_w, rho_true_w_lvl) < 0:
        rho_hat_w = -rho_hat_w
        rho_reduced = -rho_reduced

    # E. Thresholding (Eq. 16 e 17)
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
    
    if np.dot(rho_hat_w_th, rho_true_w_lvl) < 0:
        rho_hat_w_th = -rho_hat_w_th

    # F. Reconstrução para o tempo (opcional, apenas para calcular ASE)
    def reconstruct(vec_w):
        c = pywt.array_to_coeffs(vec_w, slices, output_format='wavedec')
        sig = pywt.waverec(c, wavelet_name, mode=mode)
        return sig[:p]
    
    rho_rec_th = reconstruct(rho_hat_w_th)
    if np.linalg.norm(rho_rec_th) > 1e-6:
        rho_rec_th = rho_rec_th * (norm_rho / np.linalg.norm(rho_rec_th))
    
    ase = np.mean((rho_rec_th - rho_true) ** 2)
    
    return k, k_final, ase

# ==============================================================================
# 4. MONTE CARLO (2000 iterações) com nível fixo 3
# ==============================================================================
N_SIM = 2000
print(f"Iniciando simulação com {N_SIM} iterações (level=3, multiplicador_delta=1.0)...")
start = time.time()

k_adaptive_list = []
k_final_list = []
ase_list = []

for i in range(N_SIM):
    if i % 20 == 0:
        print(f"Processando {i}/{N_SIM}...")
    k_a, k_f, err = processar_aspca(seed=1000 + i, level=3, multiplicador_delta=1.0)
    k_adaptive_list.append(k_a)
    k_final_list.append(k_f)
    ase_list.append(err)

elapsed = time.time() - start
print(f"Concluído em {elapsed:.2f} segundos.")

# ==============================================================================
# 5. VISUALIZAÇÃO DOS RESULTADOS
# ==============================================================================
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histograma de k_adaptive
sns.histplot(k_adaptive_list, kde=True, ax=axes[0], color='skyblue', bins=30)
axes[0].set_title(r'Distribuição de $k_{adaptive}$ (antes do threshold) - Level 3')
axes[0].set_xlabel('Número de variáveis selecionadas')
axes[0].axvline(np.mean(k_adaptive_list), color='r', linestyle='--',
                label=f'Média: {np.mean(k_adaptive_list):.1f}')
axes[0].legend()

# Histograma de k_final
sns.histplot(k_final_list, kde=True, ax=axes[1], color='green', bins=30)
axes[1].set_title(r'Distribuição de $k_{final}$ (após threshold) - Level 3')
axes[1].set_xlabel('Coeficientes não nulos finais')
axes[1].axvline(np.mean(k_final_list), color='r', linestyle='--',
                label=f'Média: {np.mean(k_final_list):.1f}')
axes[1].legend()

plt.tight_layout()
plt.show()

# ==============================================================================
# 6. RESUMO ESTATÍSTICO
# ==============================================================================
print("\n" + "=" * 60)
print("RESUMO ESTATÍSTICO (2000 SIMULAÇÕES) - LEVEL 3")
print("=" * 60)
print(f"{'Métrica':<20} {'Média':>12} {'Mediana':>12} {'Desvio':>12} {'Mínimo':>12} {'Máximo':>12}")
print("-" * 80)
print(f"{'k_adaptive':<20} {np.mean(k_adaptive_list):>12.2f} {np.median(k_adaptive_list):>12.2f} "
      f"{np.std(k_adaptive_list):>12.2f} {np.min(k_adaptive_list):>12} {np.max(k_adaptive_list):>12}")
print(f"{'k_final':<20} {np.mean(k_final_list):>12.2f} {np.median(k_final_list):>12.2f} "
      f"{np.std(k_final_list):>12.2f} {np.min(k_final_list):>12} {np.max(k_final_list):>12}")
print(f"{'ASE':<20} {np.mean(ase_list):>12.2e} {np.median(ase_list):>12.2e} "
      f"{np.std(ase_list):>12.2e} {np.min(ase_list):>12.2e} {np.max(ase_list):>12.2e}")
print("=" * 60)
