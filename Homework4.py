import numpy as np
from scipy.stats import norm

# Given data
data = np.array([-1, 0, 4, 5, 6])

# Initial parameter setting
pi1, pi2, mu1, mu2, sigma1_sq, sigma2_sq = 0.5, 0.5, 6, 7, 1, 4

# Calculate the log-likelihood
likelihoods = (
    pi1 * norm.pdf(data, loc=mu1, scale=np.sqrt(sigma1_sq)) +
    pi2 * norm.pdf(data, loc=mu2, scale=np.sqrt(sigma2_sq))
)

log_likelihood = np.sum(np.log(likelihoods))

# Round to the nearest tenth
log_likelihood_rounded = round(log_likelihood, 1)

print("Log-Likelihood of the data:", log_likelihood)
