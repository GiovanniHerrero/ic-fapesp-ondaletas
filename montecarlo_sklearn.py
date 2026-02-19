import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import beta
from sklearn.decomposition import SparsePCA
import pywt

# --------------------------------------------------------------------
# 1. PARÂMETROS GERAIS 
# --------------------------------------------------------------------
n = 1024
p = 2048
sigma = 1.0
norm_rho = 10.0
wavelet_name = 'sym8'
mode = 'periodization'        
level = 3

# Parâmetro de penalidade do SparsePCA (L1 penalty)
ALPHA_SPCA = 6

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
# 3. FUNÇÃO AUXILIAR: MÉTRICA R (OVERLAP)
# --------------------------------------------------------------------
def calc_R(u, v):
    """
    Calcula o Overlap R(u, v) = |<u, v>| / (||u|| * ||v||).
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return np.abs(np.dot(u, v)) / (norm_u * norm_v)

# --------------------------------------------------------------------
# 4. FUNÇÃO PRINCIPAL: SCIKIT-LEARN SPARSE PCA
# --------------------------------------------------------------------
def processar_sklearn_spca(seed):
    np.random.seed(seed)
    
    # A. Geração dos dados
    v_vec = np.random.normal(0, 1, size=(n, 1))
    Z = np.random.normal(0, 1, size=(n, p))
    X = v_vec @ rho_true.reshape(1, -1) + sigma * Z
    
    # B. Transformada Wavelet
    # Decompõe o primeiro para pegar dimensões
    coeffs_first = pywt.wavedec(X[0], wavelet_name, level=level, mode=mode)
    slices = pywt.coeffs_to_array(coeffs_first)[1]
    p_coef = sum(c.size for c in coeffs_first)
    
    X_wavelet = np.zeros((n, p_coef))
    for i in range(n):
        c = pywt.wavedec(X[i], wavelet_name, level=level, mode=mode)
        arr, _ = pywt.coeffs_to_array(c)
        X_wavelet[i] = arr

    # C. Centralização dos dados 
    X_centered = X_wavelet - X_wavelet.mean(axis=0)
    
    # D. Sparse PCA (Scikit-Learn)
    #  n_components=1 para  primeiro componente.
    spca = SparsePCA(n_components=1, alpha=ALPHA_SPCA, random_state=seed,
                     max_iter=100, tol=1e-2, method='lars')  #lars para aumentar velocidade
    
    spca.fit(X_centered)   
    rho_hat_w = spca.components_[0]
    
    # E. Ajuste de sinal
    coeffs_rho_lvl = pywt.wavedec(rho_true, wavelet_name, level=level, mode=mode)
    rho_true_w_lvl, _ = pywt.coeffs_to_array(coeffs_rho_lvl)
    
    if np.dot(rho_hat_w, rho_true_w_lvl) < 0:
        rho_hat_w = -rho_hat_w

    # Contagem de não-nulos
    k_final = np.count_nonzero(rho_hat_w)
    
    # F. Reconstrução
    def reconstruct(vec_w):
        c = pywt.array_to_coeffs(vec_w, slices, output_format='wavedec')
        sig = pywt.waverec(c, wavelet_name, mode=mode)
        return sig[:p]
    
    rho_rec = reconstruct(rho_hat_w)
    
    if np.linalg.norm(rho_rec) > 1e-6:
        rho_rec = rho_rec * (norm_rho / np.linalg.norm(rho_rec)) #só por garantia
    
    r_val = calc_R(rho_rec, rho_true)
    
    return k_final, r_val

# ==============================================================================
# 5. MONTE CARLO 
# ==============================================================================
N_SIM = 2000  
print(f"Iniciando simulação Scikit-Learn SparsePCA com {N_SIM} iterações...")
print(f"Alpha (penalidade): {ALPHA_SPCA}")

k_final_list = []
r_list = []

for i in range(N_SIM):
    if i % 10 == 0:
        print(f"Processando {i}/{N_SIM}")
        
    k_f, r_score = processar_sklearn_spca(seed=1000 + i)
    k_final_list.append(k_f)
    r_list.append(r_score)
# ==============================================================================
# 6. VISUALIZAÇÃO DOS RESULTADOS
# ==============================================================================
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5)) 

# Histograma de k_final
sns.histplot(k_final_list, kde=True, ax=axes[0], color='green', bins=30)
axes[0].set_title(r'Esparsidade Final ($k_{final}$) - Scikit-Learn')
axes[0].set_xlabel('Coeficientes Não-Nulos')
axes[0].axvline(np.mean(k_final_list), color='r', linestyle='--', label=f'Média: {np.mean(k_final_list):.1f}')
axes[0].legend()

# Histograma de R (Overlap)
sns.histplot(r_list, kde=True, ax=axes[1], color='orange', bins=50)
axes[1].set_title(r'Consistência ($R$) - Scikit-Learn')
axes[1].set_xlabel('Overlap (0 a 1)')
axes[1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
axes[1].axvline(np.mean(r_list), color='r', linestyle='--', label=f'Média: {np.mean(r_list):.4f}')
axes[1].legend(loc='upper left')

plt.tight_layout()
plt.show()

# ==============================================================================
# 7. RESUMO ESTATÍSTICO
# ==============================================================================
print("\n" + "=" * 80)
print("RESUMO ESTATÍSTICO - SCIKIT-LEARN SPARSE PCA")
print("=" * 80)
print(f"{'Métrica':<20} {'Média':>12} {'Mediana':>12} {'Desvio':>12} {'Mínimo':>12} {'Máximo':>12}")
print("-" * 80)
print(f"{'k_final':<20} {np.mean(k_final_list):>12.2f} {np.median(k_final_list):>12.2f} "
      f"{np.std(k_final_list):>12.2f} {np.min(k_final_list):>12} {np.max(k_final_list):>12}")
print(f"{'Overlap (R)':<20} {np.mean(r_list):>12.4f} {np.median(r_list):>12.4f} "
      f"{np.std(r_list):>12.4f} {np.min(r_list):>12.4f} {np.max(r_list):>12.4f}")
print("=" * 80)
