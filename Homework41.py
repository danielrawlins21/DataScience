import numpy as np
from scipy.stats import norm

# Given data
data = np.array([-1, 0, 4, 5, 6])

# Initial parameter setting
pi1, pi2, mu1, mu2, sigma1_sq, sigma2_sq = 0.5, 0.5, 6, 7, 1, 4

weights = []

for x in data:
    # Calculate the posterior probabilities
    prob_y1 = pi1 * norm.pdf(x, loc=mu1, scale=np.sqrt(sigma1_sq))
    prob_y2 = pi2 * norm.pdf(x, loc=mu2, scale=np.sqrt(sigma2_sq))

    # Assign weight based on the comparison of posterior probabilities
    weight = 2 if prob_y2 > prob_y1 else 1
    weights.append(weight)

print("Assigned weights:", weights)
