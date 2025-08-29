#%% VSM muestra 8A de Pablo Tancredi - Agosto 2025
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score
from mlognormfit import fit3
from mvshtools import mvshtools as mt
import re
from uncertainties import ufloat
#%% Funciones
def lineal(x,m,n):
    return m*x+n

def coercive_field(H, M):
    """
    Devuelve el valor medio del campo coercitivo (Hc) como ufloat,
    imprime ambos valores de Hc encontrados.

    Parámetros:
    - H: np.array, campo magnético (en A/m o kA/m)
    - M: np.array, magnetización (en emu/g)

    Retorna:
    - hc_ufloat: ufloat con el valor medio y la diferencia absoluta como incertidumbre
    """
    H = np.asarray(H)
    M = np.asarray(M)
    hc_values = []

    for i in range(len(M)-1):
        if M[i]*M[i+1] < 0:  # Cambio de signo indica cruce por cero
            # Interpolación lineal entre (H[i], M[i]) y (H[i+1], M[i+1])
            h1, h2 = H[i], H[i+1]
            m1, m2 = M[i], M[i+1]
            hc = h1 - m1 * (h2 - h1) / (m2 - m1)
            hc_values.append(hc)

    if len(hc_values) != 2:
        print("Advertencia: Se encontraron {} valores de Hc, se esperaban 2.".format(len(hc_values)))
        return None

    print(f"Hc encontrados: {hc_values[0]:.3f}, {hc_values[1]:.3f}")

    # Valor medio considerando el signo
    hc_mean =abs((hc_values[0] - hc_values[1]) / 2)
    # Incertidumbre: diferencia entre los valores absolutos
    hc_unc = abs(abs(hc_values[0]) - abs(hc_values[1]))
    return ufloat(hc_mean, hc_unc)
#%% Levanto Archivos

data_8A_new = np.loadtxt('8A_250827.txt', skiprows=12)
data_8A_old = np.loadtxt('8A_250609.txt', skiprows=12)

masa_8A_FF = 0.0496*10/1000 #g

H_8A_old = data_8A_old[:, 0]  # Gauss
m_8A_old = data_8A_old[:, 1]  # emu

H_8A_new = data_8A_new[:, 0]  # Gauss
m_8A_new = data_8A_new[:, 1]  # emu

fig1, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
ax.plot(H_8A_old, m_8A_old, '-', label='8A FF old')
ax.plot(H_8A_new, m_8A_new, '-', label='8A FF new')

for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu)')
plt.xlabel('H (G)')
plt.savefig('8A_old_new_ra.png', dpi=300)
plt.show()


#%% Fitting
# Para almacenar resultados para graficar luego

# Crear sesión de ajuste
fit = fit3.session(H_8A_old, m_8A_old, fname='8A FF old', divbymass=True, mass=masa_8A_FF)
fit.fix('sig0')
fit.fix('mu0')
fit.free('dc')
fit.fit()
fit.update()
fit.free('sig0')
fit.free('mu0')
fit.set_yE_as('sep')  # pesos por separación
fit.fit()
fit.update()
fit.save()
fit.print_pars()
pars = fit.derived_parameters()
for key, val in pars.items():
    unit = fit3.session.units.get(key, '')
    print(f"{key:15s} = {val} {unit}")
H_fit = fit.X
m_fit = fit.Yfit  # resultado del ajuste
m_fit_sin_diamag = m_fit - H_fit*fit.params['C'].value - fit.params['dc'].value
ms = pars['m_s']
mu_mu = pars['<mu>_mu']
#hc = pars['H_c']

ms_str = f"$M_s$ = {ms:.2uf} emu/g"
mu_mu_str = f"$\\langle\\mu\\rangle_\\mu$ = {mu_mu:.2uP} $\\mu_B$" if mu_mu is not None else ""
#hc_str = f"$H_c$ = {hc:.2uf} G"
ajuste_text_1 = ms_str + "\n" + mu_mu_str  # + hc_str


#%%
fit2 = fit3.session(H_8A_new, m_8A_new, fname='8A FF new', divbymass=True, mass=masa_8A_FF)
fit2.fix('sig0')
fit2.fix('mu0')
fit2.free('dc')
fit2.fit()
fit2.update()
fit2.free('sig0')
fit2.free('mu0')
fit2.set_yE_as('sep')  # pesos por separación
fit2.fit()
fit2.update()
fit2.save()
fit2.print_pars()
pars_2 = fit2.derived_parameters()
for key, val in pars.items():
    unit = fit3.session.units.get(key, '')
    print(f"{key:15s} = {val} {unit}")
H_fit_2 = fit.X
m_fit_2 = fit.Yfit  # resultado del ajuste
m_fit_sin_diamag_2 = m_fit_2 - H_fit_2*fit2.params['C'].value - fit2.params['dc'].value
ms_2 = pars_2['m_s']
mu_mu_2 = pars_2['<mu>_mu']
#hc = pars['H_c']

ms_str_2 = f"$M_s$ = {ms_2:.2uf} emu/g"
mu_mu_str_2 = f"$\\langle\\mu\\rangle_\\mu$ = {mu_mu:.2uP} $\\mu_B$" if mu_mu is not None else ""
#hc_str = f"$H_c$ = {hc:.2uf} G"
ajuste_text_2 = ms_str_2 + "\n" + mu_mu_str_2  # + hc_str

#%% Crear figura
plt.figure(figsize=(6,4), constrained_layout=True)

plt.plot(H_8A_old, m_8A_old/masa_8A_FF, '.-', label=f'8A FF old', alpha=0.7)
plt.plot(H_fit, m_fit, '-', label=f'8A FF old fit', linewidth=2)

plt.plot(H_8A_new, m_8A_new/masa_8A_FF, '.-', label=f'8A FF new', alpha=0.7)
plt.plot(H_fit_2, m_fit_2, '-', label=f'8A FF new fit2', linewidth=2)

plt.xlabel('H (G)')
plt.ylabel('m (emu/g)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.gca().text(
    0.75, 0.6, ajuste_text_1, transform=plt.gca().transAxes,
    fontsize=10, va='center', ha='center',
    bbox=dict(boxstyle='round', facecolor='tab:orange', alpha=0.7))

plt.gca().text(
    0.75, 0.3, ajuste_text_2, transform=plt.gca().transAxes,
    fontsize=10, va='center', ha='center',
    bbox=dict(boxstyle='round', facecolor='tab:red', alpha=0.7))


plt.savefig(f'8A_FF_comparativa_old_new.png', dpi=300)
plt.show()


# %%
