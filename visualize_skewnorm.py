import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

mean = 6
std = 2
skew = 0
num_samples = 5000
x = np.linspace(-10, 40, 500)
pdf = skewnorm.pdf(x, a=skew, loc=mean, scale=std)
samples = skewnorm.rvs(a=skew, loc=mean, scale=std, size=num_samples)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, label="Skew-Normal PDF", linewidth=2)
plt.hist(samples, bins=100, density=True, alpha=0.4, label="Sampled Data")
plt.title(f"Skewed Normal Distribution\nmean={mean}, std={std}, skew={skew}")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("./figures/visualized_skewnorm.png")
