import numpy as np
import matplotlib.pyplot as plt

LLLR_NMSE_pool.npy = np.load('LLLR_NMSE_pool.npy')
CE_NMSE_pool.npy = np.load('CE_NMSE_pool.npy')

LLLR_mean = np.mean(LLLR_NMSE_pool, axis=0)
LLLR_sem = np.std(LLLR_NMSE_pool, axis=0) / np.sqrt(LLLR_NMSE_pool.shape[0])
CE_mean = np.mean(CE_NMSE_pool, axis=0)
CE_sem = np.std(CE_NMSE_pool, axis=0) / np.sqrt(CE_NMSE_pool.shape[0])

color1 = np.array([20, 80, 148, 128])/255
color2 = np.array([128, 0, 0, 128])/255

fig = plt.figure(figsize=(20,12.4))
fig.patch.set_facecolor('white')
plt.rcParams['font.size'] = 25
plt.plot(X, CE_mean, '-', color=color1[:-1], linewidth=3)
plt.plot(X, LLLR_mean, '-', color=color2[:-1], linewidth=3)
plt.fill_between(X, CE_mean - CE_sem, CE_mean + CE_sem, color=color1)
plt.fill_between(X, LLLR_mean - LLLR_sem, LLLR_mean + LLLR_sem, alpha=0.5, color=color2)
plt.xlabel('Iteration')
plt.ylabel('NMSE')
plt.legend(('Cross-Entropy', 'LLLR(proposed)'), loc='upper right')
plt.yscale('log')
plt.show()