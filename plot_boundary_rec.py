import numpy as np
import matplotlib.pyplot as plt

N = 3
yes = [0.0000804148691205217, 0.000224278619023327, 0.000138905364574381]
no = [0.0000743721326899161, 0.000218274971202301, 0.000133319742457978]

ind = np.arange(N)
width = 0.35
plt.bar(ind, no, width, label="No boundary reconstruction")
plt.bar(ind + width, yes, width, label="Boundary reconstruction")

plt.xlabel('Model')
plt.ylabel('DT-RMSE')

plt.xticks(ind + width / 2, ('Linear regression', 'Regression tree', 'Regression forest'))
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(loc='best')
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
plt.tight_layout()
plt.savefig('plots/plot_boundary.png')
plt.show()