import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

input_radius = [1, 2, 3, 4]

dt_rmse_inter0 = 0.000162674884381243
dt_rmse_inter1 = 0.0000929099932639929

dt_rmse_lin_reg = [0.0000824551913481772, 0.0000743721326899161, 0.0000750944648334972, 0.000081069240511781]
std_lin_reg = [0.00000291948120312767, 0.00000256473147839579, 0.00000239968482404982, 0.00000238685976029714]
time_lin_reg = [24.03235674, 28.62143869, 25.46609095, 23.70786746]

dt_rmse_reg_tree = [0.000198408142173172, 0.000218274971202301, 0.000240633405591824, 0.000257243510415847]
std_reg_tree = [0.0000030791844188107, 0.00000333939643297023, 0.00000360604164471556, 0.00000447848650302777]
time_reg_tree = [25.21175256, 21.3854788, 19.50181619, 17.61524476]

dt_rmse_ran_forest = [0.000133266024718439, 0.000133319742457978, 0.000137603071910721, 0.000140572776784643]
std_ran_forest = [0.00000243890277870168, 0.00000248444456226188, 0.00000275376326200792, 0.00000334588292473952]
time_ran_forest = [708.4109573, 689.2010544]

fig = plt.figure()
ax = fig.gca()
plt.hlines(dt_rmse_inter0, 1, 4, label="Nearest-neighbour interpolation", colors='y')
plt.hlines(dt_rmse_inter1, 1, 4, label="Trilinear interpolation", colors='r')
plt.errorbar(input_radius, dt_rmse_lin_reg, std_lin_reg, label="Linear regression")
plt.errorbar(input_radius, dt_rmse_reg_tree, std_reg_tree, label="Regression tree")
plt.errorbar(input_radius, dt_rmse_ran_forest, std_ran_forest, label="Random forest")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('Radius of the input patch')
plt.ylabel('DT-RMSE')
# plt.title('Reconstruction errors as a function of radius of the input patch')
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
plt.tight_layout()
plt.savefig('plots/plot_input_radius.png')
plt.show()