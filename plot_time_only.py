import matplotlib.pyplot as plt

x = ['Linear regression', 'Regression tree', 'Regression forest']
energy = [27.50112752/60, 21.8051561/60, 687.6736902/60]

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color='green')
plt.xlabel("Model")
plt.ylabel("Reconstruction time (minutes)")
plt.title("Average image reconstruction time per subject")

plt.xticks(x_pos, x)

plt.show()
plt.savefig('plots/plot_time_only.png')
