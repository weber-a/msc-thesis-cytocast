import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib as mpl

mpl.rcParams["axes.linewidth"] = 2

N = 10
X = np.concatenate(
    (np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N)))
)[:, np.newaxis]

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(X)
log_dens = kde.score_samples(X_plot)

kernel_values = []
for data_point in X[:, 0]:
    X_0 = np.ones(X_plot.shape)*data_point
    kernel = 1/(np.sqrt(2*np.pi)) * np.exp(-0.5*(X_0 - X_plot)**2)
    kernel_values.append(kernel)


X_counter = Counter(X[:, 0])

plt.figure(figsize=(20, 10))
ax = plt.gca()
plt.hist(X[:, 0], density=True, color="0.2")
for i in range(len(kernel_values)):
    kernel = kernel_values[i]
    plt.fill(X_plot[:, 0], kernel, alpha=0.3)
plt.vlines(x=X[:,0], ymin=0, ymax=0.8, color="maroon", linestyles="dashed")
plt.ylim([0, 0.5])
plt.xlabel("x", size=40)
plt.ylabel("Density", size=40)
plt.xticks(fontsize="40")
plt.yticks(fontsize="40")
ax.tick_params(size=10, width=2)
plt.tight_layout()
# plt.savefig("kde_demo.pdf")