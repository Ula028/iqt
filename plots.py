import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size

n_pairs = [11353, 22710, 56781, 113567, 283924, 567850]

dt_rmse_lin_reg = [0.0000800431746232638, 0.0000766568075352493, 0.0000743721326899161, 0.000073641829165017, 0.0000731426976556691, 0.0000729702084813825]
std_lin_reg = [0.00000263285542087394, 0.00000257898531327114, 0.00000256473147839579, 0.00000254215378952099, 0.00000253924270610379, 0.00000254806643266342]
time_lin_reg = [26.32748728, 29.13614464, 28.62143869, 27.65516533, 27.13675988, 26.12976934]

dt_rmse_reg_tree = [0.000237444929786386, 0.000225269376274043, 0.000218274971202301, 0.000216688675652142, 0.000207484909086454, 0.000199988931289043]
std_reg_tree = [0.00000303377008115736, 0.00000373835027420745, 0.00000333939643297023, 0.00000366568649559447, 0.00000310828265427883, 0.0000030895572628493]
time_reg_tree = [21.78957263, 21.46395584, 21.3854788, 22.40279128, 21.82013485, 21.96900319]

fig = plt.figure()
plt.errorbar(n_pairs, dt_rmse_lin_reg, std_lin_reg, label="Linear regression")
plt.errorbar(n_pairs, dt_rmse_reg_tree, std_reg_tree, label="Regression tree")
plt.xlabel('Number of patch pairs in training set')
plt.ylabel('DT-RMSE')
plt.title('Reconstruction errors as a function of training set size')
plt.legend()
plt.show()