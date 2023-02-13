import numpy as np, pandas as pd

RGG = pd.read_csv('results/results_test_SE_RGG.csv',index_col=0).values
RCM = pd.read_csv('results/results_test_SE_RCM.csv',index_col=0).values
config1 = pd.read_csv('results/results_test_SE_config_1.csv',index_col=0).values
config2 = pd.read_csv('results/results_test_SE_config_2.csv',index_col=0).values
config4 = pd.read_csv('results/results_test_SE_config_4.csv',index_col=0).values

table = pd.DataFrame( np.hstack([ RGG, RCM, config1, config2, config4 ]) )
table.index = ['LIM Rand', 'LIM HAC', 'TSI Rand', 'TSI HAC', '$\max_S \phi(S)$', '\# Clusters', '1st Clus.', '2nd Clus.', 'Last Clus.', '$n$']
table.columns = pd.MultiIndex.from_product([['RGG', 'RCM', 'Configuration'], [1,2,4]])
print('\n\\begin{table}[ht]')
print('\centering')
print('\caption{Size Under Design 2}')
print('\\begin{threeparttable}')
print(table.to_latex(float_format = lambda x: '%.3f' % x, header=True, escape=False, multicolumn_format='c'))
print('\\begin{tablenotes}[para,flushleft]')
print("  \\footnotesize Averages over 5k simulations. The first four rows report the sizes of level-5\% tests, with LIM = linear-in-means, BG = binary game, Rand = randomization test, HAC = $t$-test with HAC estimator. The last three rows report cluster sizes in descending order of size.")
print('\end{tablenotes}')
print('\end{threeparttable}')
print('\end{table}')

