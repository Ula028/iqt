import matplotlib.pyplot as plt

n_pairs = [11353, 22710, 56781, 113567, 283924, 567850]

dt_rmse_inter0 = 0.000162674884381243
dt_rmse_inter1 = 0.0000929099932639929

dt_rmse_lin_reg = [0.0000800431746232638, 0.0000766568075352493, 0.0000743721326899161, 0.000073641829165017, 0.0000731426976556691, 0.0000729702084813825]
std_lin_reg = [0.00000263285542087394, 0.00000257898531327114, 0.00000256473147839579, 0.00000254215378952099, 0.00000253924270610379, 0.00000254806643266342]
time_lin_reg = [26.32748728, 29.13614464, 28.62143869, 27.65516533, 27.13675988, 26.12976934]

dt_rmse_reg_tree = [0.000237444929786386, 0.000225269376274043, 0.000218274971202301, 0.000216688675652142, 0.000207484909086454, 0.000199988931289043]
std_reg_tree = [0.00000303377008115736, 0.00000373835027420745, 0.00000333939643297023, 0.00000366568649559447, 0.00000310828265427883, 0.0000030895572628493]
time_reg_tree = [21.78957263, 21.46395584, 21.3854788, 22.40279128, 21.82013485, 21.96900319]

dt_rmse_ran_forest = [0.000148482257451443, 0.000141387980873381, 0.000133319742457978, 0.000128229763272654, 0.00012295618372367, 0.000115612076038944]
std_ran_forest = [0.00000263436916929117, 0.00000252274817451567, 0.00000248444456226188, 0.00000245375132644346, 0.00000223383961287359, 0.00000220481378080495]
time_ran_forest = [682.8789777, 691.8780693, 689.2010544, 691.8780693, 687.3269928]

fig = plt.figure()
plt.xscale('log')
plt.hlines(dt_rmse_inter0, 11353, 567850, label="Nearest-neighbour interpolation", colors='y')
plt.hlines(dt_rmse_inter1, 11353, 567850, label="Trilinear interpolation", colors='r')
plt.errorbar(n_pairs, dt_rmse_lin_reg, std_lin_reg, label="Linear regression")
plt.errorbar(n_pairs, dt_rmse_reg_tree, std_reg_tree, label="Regression tree")
plt.errorbar(n_pairs, dt_rmse_ran_forest, std_ran_forest, label="Random forest")
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('Number of patch pairs in training set')
plt.ylabel('DT-RMSE')
# plt.title('Reconstruction errors as a function of training set size')
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
plt.tight_layout()
plt.savefig('plots/plot_dataset.png')
plt.show()