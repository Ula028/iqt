import numpy as np
import matplotlib.pyplot as plt

N = 3
yes = [216.1734247/60, 203.488666/60, 1671.640468/60]
no = [27.50112752/60, 21.8051561/60, 687.6736902/60]

ind = np.arange(N)
width = 0.35
plt.bar(ind, no, width, label="No boundary reconstruction")
plt.bar(ind + width, yes, width, label="Boundary reconstruction")

plt.xlabel('Model')
plt.ylabel('Execution time (minutes)')

plt.xticks(ind + width / 2, ('Linear regression', 'Regression tree', 'Regression forest'))
plt.legend(loc='best')
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
plt.tight_layout()
plt.savefig('plots/plot_time.png')
plt.show()