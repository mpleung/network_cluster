import numpy as np, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

model = 'RGG'
alternatives = np.linspace(0,0.75,101)[1:]
results = pd.read_csv('results/results_power_' + model + '.csv')

sns.set_theme(style='dark', font='candara')
plt.rcParams['figure.figsize'] = (5, 2.5)
figure, axis = plt.subplots(1, 2)
plt.setp(axis, xticks=[0, 0.25, 0.5, 0.75], yticks=[0.5,1])

hac250 = results['hac250'].values
X_Y_Spline = make_interp_spline(alternatives, hac250)
X_ = np.linspace(alternatives.min(), alternatives.max(), 50)
Y_ = X_Y_Spline(X_)
axis[0].plot(X_, Y_, label='HAC')
rand250 = results['rand250'].values
X_Y_Spline = make_interp_spline(alternatives, rand250)
X_ = np.linspace(alternatives.min(), alternatives.max(), 50)
Y_ = X_Y_Spline(X_)
axis[0].plot(X_, Y_, label='rand')
axis[0].set_title('n=250')
axis[0].legend(loc='lower right')

hac500 = results['hac500'].values
X_Y_Spline = make_interp_spline(alternatives, hac500)
X_ = np.linspace(alternatives.min(), alternatives.max(), 50)
Y_ = X_Y_Spline(X_)
axis[1].plot(X_, Y_, label='HAC')
rand500 = results['rand500'].values
X_Y_Spline = make_interp_spline(alternatives, rand500)
X_ = np.linspace(alternatives.min(), alternatives.max(), 50)
Y_ = X_Y_Spline(X_)
axis[1].plot(X_, Y_, label='rand')
axis[1].set_title('n=500')
axis[1].legend(loc='lower right')

figure.tight_layout()
plt.savefig('results/power_' + model + '.png',bbox_inches='tight',dpi=500)

